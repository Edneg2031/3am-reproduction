from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

FeatureLayerSpec = str | int


@dataclass(frozen=True)
class ExternalBackboneConfig:
    sam2_checkpoint: Path | None = None
    sam2_config: str | Path | None = None
    sam2_repo: Path | None = None
    must3r_checkpoint: Path | None = None
    must3r_repo: Path | None = None
    must3r_device: str | None = None
    must3r_image_size: int = 512
    must3r_amp: str | bool = "bf16"
    must3r_max_bs: int = 1
    must3r_decode_batch_size: int = 1
    must3r_feature_layers: tuple[FeatureLayerSpec, ...] = ("encoder", 4, 7, 11)
    must3r_expected_channels: tuple[int, ...] | None = None


class ExternalDependencyError(RuntimeError):
    pass


def _prepend_repo_to_sys_path(repo: Path | None) -> None:
    if repo is None:
        return
    repo = repo.expanduser().resolve()
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _parse_must3r_feature_layers(value: str | Sequence[FeatureLayerSpec] | None) -> tuple[FeatureLayerSpec, ...]:
    if value is None:
        return ("encoder", 4, 7, 11)
    if isinstance(value, str):
        parts: Sequence[Any] = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = value
    layers: list[FeatureLayerSpec] = []
    for part in parts:
        if isinstance(part, str) and part.lower() == "encoder":
            layers.append("encoder")
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("MUSt3R feature layers must contain at least one layer")
    return tuple(layers)


def _dtype_for_must3r_amp(amp: str | bool) -> torch.dtype:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return torch.float32


def _normalize_must3r_amp(value: str | bool) -> str | bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"false", "none", "off", "0"}:
        return False
    if lowered in {"bf16", "fp16"}:
        return lowered
    raise ValueError("MUSt3R amp must be false, bf16, or fp16")


def _sam2_config_name(config: str | Path) -> str:
    text = str(config)
    path = Path(text)
    if path.is_absolute():
        parts = path.parts
        if "configs" in parts:
            return "/".join(parts[parts.index("configs") :])
    return text


class Sam2TrainingAdapter(nn.Module):
    """Thin adapter boundary for official SAM2 code.

    Full SAM2 training internals are imported lazily because installation requires
    the official repository and CUDA dependencies.
    """

    def __init__(self, config: ExternalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model: nn.Module | None = None
        self._last_feature_payload: Any | None = None
        self._last_feature_selector: tuple[str, str | int] | None = None
        self._last_sam_feature_shape: tuple[int, ...] | None = None

    def load(self) -> None:
        _prepend_repo_to_sys_path(self.config.sam2_repo)
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError(
                "Install facebookresearch/sam2 before loading SAM2, or set external.sam2_repo to the cloned repo root."
            ) from error
        if self.config.sam2_config is None or self.config.sam2_checkpoint is None:
            raise ValueError("sam2_config and sam2_checkpoint are required")
        config_name = _sam2_config_name(self.config.sam2_config)
        try:
            self.model = build_sam2_video_predictor(config_name, str(self.config.sam2_checkpoint))
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError(
                "Failed to load SAM2 config "
                f"{config_name!r}. Run `pip install -e .` in the SAM2 repo or set external.sam2_repo so the "
                "repo root is on PYTHONPATH; official SAM2 expects Hydra config names like "
                "`configs/sam2.1/sam2.1_hiera_l.yaml`."
            ) from error

    def freeze_image_encoder(self) -> None:
        if self.model is None:
            return
        for name, parameter in self.model.named_parameters():
            if "image_encoder" in name or "vision_encoder" in name:
                parameter.requires_grad_(False)

    def freeze_vision_encoder(self) -> None:
        self.freeze_image_encoder()

    def encode_sam_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        self._validate_image_tensor(images)
        if hasattr(self.model, "encode_sam_features"):
            features = self.model.encode_sam_features(images)  # type: ignore[attr-defined]
            sam_feature, selector = self._select_sam_feature(features)
            self._remember_feature_payload(features, selector, sam_feature)
            return sam_feature
        if hasattr(self.model, "forward_image"):
            features = self.model.forward_image(images)  # type: ignore[attr-defined]
            sam_feature, selector = self._select_sam_feature(features)
            self._remember_feature_payload(features, selector, sam_feature)
            return sam_feature
        if hasattr(self.model, "image_encoder"):
            features = self.model.image_encoder(images)  # type: ignore[attr-defined]
            sam_feature, selector = self._select_sam_feature(features)
            self._remember_feature_payload(features, selector, sam_feature)
            return sam_feature
        raise ExternalDependencyError(
            "SAM2 adapter could not obtain trainable image features. Install official SAM2 training internals "
            "or subclass Sam2TrainingAdapter.encode_sam_features for the selected SAM2 release."
        )

    def _validate_image_tensor(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ExternalDependencyError(f"SAM2 images must have shape BCHW/TCHW, got {tuple(images.shape)}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if height % 32 != 0 or width % 32 != 0:
            raise ExternalDependencyError(
                "SAM2 Hiera input size must be divisible by 32 so its 8x8 patch-window positional embedding "
                f"tiles exactly; got image size {(height, width)}. Set model.sam_image_size to 1024 "
                "or another multiple of 32."
            )

    def forward_train_sequence(self, batch: Any, merged_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        backbone_out = self._inject_merged_features(merged_features)
        for method_name in (
            "forward_train_sequence_with_backbone",
            "forward_train_sequence_from_backbone",
            "training_forward_with_backbone",
            "training_forward_from_backbone",
            "forward_train_with_backbone",
            "forward_train_from_backbone",
            "forward_train_sequence",
            "training_forward",
            "forward_train",
        ):
            method = getattr(self.model, method_name, None)
            if callable(method):
                outputs = self._call_training_method(
                    method,
                    batch=batch,
                    merged_features=merged_features,
                    backbone_out=backbone_out,
                )
                if outputs is None:
                    continue
                return self._validate_training_outputs(outputs, method_name)
        raise ExternalDependencyError(
            "SAM2 training forward API is not available for merged 3AM features. Provide a SAM2 training wrapper "
            "with one of forward_train_sequence_with_backbone(...), forward_train_sequence_from_backbone(...), "
            "or forward_train_sequence(batch, merged_features), returning mask_logits, iou_scores, and "
            "occlusion_logits."
        )

    def _remember_feature_payload(
        self,
        payload: Any,
        selector: tuple[str, str | int] | None,
        sam_feature: torch.Tensor,
    ) -> None:
        self._last_feature_payload = payload
        self._last_feature_selector = selector
        self._last_sam_feature_shape = tuple(sam_feature.shape)

    def _select_sam_feature(self, features: Any) -> tuple[torch.Tensor, tuple[str, str | int] | None]:
        if isinstance(features, torch.Tensor):
            return features, None
        if isinstance(features, dict):
            for key in ("vision_features", "image_embed"):
                value = features.get(key)
                if isinstance(value, torch.Tensor):
                    return value, ("dict", key)
            value = features.get("backbone_fpn")
            if isinstance(value, (list, tuple)) and value and isinstance(value[-1], torch.Tensor):
                return value[-1], ("dict_list", "backbone_fpn")
            if isinstance(value, torch.Tensor):
                return value, ("dict", "backbone_fpn")
        if isinstance(features, (list, tuple)) and features and isinstance(features[-1], torch.Tensor):
            return features[-1], ("sequence", len(features) - 1)
        raise ExternalDependencyError("SAM2 image encoder returned unsupported feature payload")

    def _inject_merged_features(self, merged_features: torch.Tensor) -> Any | None:
        if self._last_sam_feature_shape is not None and tuple(merged_features.shape) != self._last_sam_feature_shape:
            raise ExternalDependencyError(
                "Merged 3AM feature shape does not match the SAM2 feature selected during encode: "
                f"got {tuple(merged_features.shape)}, expected {self._last_sam_feature_shape}."
            )
        payload = self._last_feature_payload
        selector = self._last_feature_selector
        if payload is None or selector is None:
            return None
        kind, key = selector
        if kind == "dict":
            updated = dict(payload)
            updated[key] = merged_features
            return updated
        if kind == "dict_list":
            updated = dict(payload)
            values = list(updated[key])
            values[-1] = merged_features
            updated[key] = values
            return updated
        if kind == "sequence":
            values = list(payload)
            values[int(key)] = merged_features
            return values
        return None

    def _call_training_method(
        self,
        method: Any,
        *,
        batch: Any,
        merged_features: torch.Tensor,
        backbone_out: Any | None,
    ) -> Any | None:
        kwargs = {
            "batch": batch,
            "input": batch,
            "merged_features": merged_features,
            "image_embeddings": merged_features,
            "image_embed": merged_features,
            "backbone_out": backbone_out,
        }
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(batch=batch, merged_features=merged_features, backbone_out=backbone_out)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return method(**kwargs)
        supported = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters and not (name == "backbone_out" and value is None)
        }
        missing_required = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in supported
        ]
        if missing_required:
            return None
        return method(**supported)

    def _validate_training_outputs(self, outputs: Any, method_name: str) -> dict[str, torch.Tensor]:
        if not isinstance(outputs, dict):
            raise ExternalDependencyError(f"SAM2 {method_name} must return a dict of training tensors")
        required = ("mask_logits", "iou_scores", "occlusion_logits")
        missing = [key for key in required if key not in outputs]
        if missing:
            raise ExternalDependencyError(
                f"SAM2 {method_name} output is missing required training tensors: {missing}"
            )
        for key in required:
            if not isinstance(outputs[key], torch.Tensor):
                raise ExternalDependencyError(f"SAM2 {method_name} output {key!r} must be a torch.Tensor")
        return outputs


class Must3rFeatureAdapter(nn.Module):
    """Adapter boundary for official MUSt3R feature extraction."""

    def __init__(self, config: ExternalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model: Any | None = None

    def load(self) -> None:
        _prepend_repo_to_sys_path(self.config.must3r_repo)
        try:
            from must3r.model import load_model  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError(
                "Could not import must3r.model.load_model. Install naver/must3r and its MASt3R dependencies, "
                "or set external.must3r_repo to the cloned repo root."
            ) from error
        if self.config.must3r_checkpoint is None or not self.config.must3r_checkpoint.exists():
            raise ExternalDependencyError(f"MUSt3R checkpoint does not exist: {self.config.must3r_checkpoint}")
        self.model = load_model(
            str(self.config.must3r_checkpoint),
            device=str(self._device()),
            img_size=int(self.config.must3r_image_size),
            verbose=False,
        )
        self._freeze_loaded_model()

    def extract_features(
        self,
        images: torch.Tensor,
        *,
        image_paths: Sequence[Path] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.model is None:
            raise ExternalDependencyError("MUSt3R model is not loaded")
        if self._is_official_model(self.model):
            if image_paths is None:
                raise ExternalDependencyError("Online MUSt3R extraction requires image paths from the training manifest")
            return self.extract_features_from_paths(image_paths)
        for method_name in ("extract_features", "forward_features", "encode"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                features = method(images)
                if isinstance(features, torch.Tensor):
                    return (features,)
                if isinstance(features, (list, tuple)) and all(isinstance(feature, torch.Tensor) for feature in features):
                    return tuple(features)
                raise ExternalDependencyError(f"MUSt3R {method_name} returned unsupported feature payload")
        raise ExternalDependencyError(
            "MUSt3R feature extraction API is not available. Precompute cache files or subclass "
            "Must3rFeatureAdapter.extract_features for the selected MUSt3R release."
        )

    def extract_features_from_paths(self, image_paths: Sequence[Path]) -> tuple[torch.Tensor, ...]:
        if self.model is None:
            self.load()
        if not self._is_official_model(self.model):
            raise ExternalDependencyError("MUSt3R official encoder/decoder pair is not loaded")
        encoder, decoder = self.model
        patch_size = self._patch_size(encoder)
        if not image_paths:
            raise ExternalDependencyError("Online MUSt3R extraction received no image paths")
        true_shape_chunks: list[torch.Tensor] = []
        pos_chunks: list[torch.Tensor] = []
        selected_chunks: list[list[torch.Tensor]] | None = None
        selected_specs: tuple[FeatureLayerSpec, ...] | None = None
        selected_indices: tuple[int, ...] | None = None
        device = self._device()
        with torch.no_grad():
            dtype = _dtype_for_must3r_amp(self.config.must3r_amp)
            with torch.autocast(device.type, dtype=dtype, enabled=self.config.must3r_amp is not False and device.type == "cuda"):
                for start in range(0, len(image_paths), max(1, int(self.config.must3r_decode_batch_size))):
                    end = min(len(image_paths), start + max(1, int(self.config.must3r_decode_batch_size)))
                    images, true_shape = self._load_official_images(image_paths[start:end], patch_size)
                    true_shape_tensor = torch.stack(true_shape, dim=0)
                    true_shape_chunks.append(true_shape_tensor)
                    image_tensor = torch.stack(images, dim=0).to(device)
                    chunk_true_shape = true_shape_tensor.to(device)
                    encoder_tokens, pos = self._encode_frames(encoder, image_tensor, chunk_true_shape)
                    pos_chunks.append(pos.detach().cpu())
                    raw_levels = self._decode_feature_levels(decoder, encoder_tokens, pos, chunk_true_shape)
                    if selected_indices is None:
                        selected_specs, selected_indices = self._normalize_feature_layers(len(raw_levels))
                        selected_chunks = [[] for _ in selected_indices]
                    if selected_chunks is None or selected_specs is None or selected_indices is None:
                        raise ExternalDependencyError("MUSt3R decoder returned no selectable feature levels")
                    if any(index >= len(raw_levels) for index in selected_indices):
                        raise ExternalDependencyError(
                            f"MUSt3R decoder returned {len(raw_levels)} levels, cannot select "
                            f"{list(self.config.must3r_feature_layers)}"
                        )
                    for output_index, raw_index in enumerate(selected_indices):
                        selected_chunks[output_index].append(raw_levels[raw_index])
                    del image_tensor, chunk_true_shape, encoder_tokens, pos, raw_levels
        if selected_chunks is None or selected_specs is None:
            raise ExternalDependencyError("MUSt3R online extraction produced no feature chunks")
        true_shape_tensor = torch.cat(true_shape_chunks, dim=0)
        pos_tensor = torch.cat(pos_chunks, dim=0)
        selected_token_levels = tuple(torch.cat(chunks, dim=0) for chunks in selected_chunks)
        features = tuple(
            self._tokens_to_chw(level, pos_tensor, true_shape_tensor, patch_size, spec)
            for level, spec in zip(selected_token_levels, selected_specs, strict=True)
        )
        self._validate_expected_channels(features)
        return tuple(feature.detach().cpu() for feature in features)

    def _device(self) -> torch.device:
        if self.config.must3r_device is not None:
            return torch.device(self.config.must3r_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _freeze_loaded_model(self) -> None:
        if not self._is_official_model(self.model):
            return
        for module in self.model:
            if hasattr(module, "eval"):
                module.eval()
            if hasattr(module, "parameters"):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _is_official_model(self, model: Any) -> bool:
        return isinstance(model, (list, tuple)) and len(model) == 2

    def _patch_size(self, encoder: Any) -> int:
        patch_size = getattr(encoder, "patch_size", 16)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        return int(patch_size)

    def _load_official_images(
        self,
        image_paths: Sequence[Path],
        patch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        try:
            from must3r.demo.inference import load_images  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError("Could not import must3r.demo.inference.load_images") from error
        views = load_images(
            [str(path) for path in image_paths],
            size=int(self.config.must3r_image_size),
            patch_size=patch_size,
            verbose=False,
        )
        images = [view["img"].to("cpu") for view in views]
        true_shape = [torch.as_tensor(view["true_shape"], dtype=torch.int64) for view in views]
        return images, true_shape

    def _encode_frames(self, encoder: Any, images: torch.Tensor, true_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        max_bs = max(1, int(self.config.must3r_max_bs))
        for start in range(0, images.shape[0], max_bs):
            end = min(images.shape[0], start + max_bs)
            encoded, pos = encoder(images[start:end], true_shape[start:end])
            tokens.append(encoded)
            positions.append(pos)
        return torch.cat(tokens, dim=0), torch.cat(positions, dim=0)

    def _decode_feature_levels(
        self,
        decoder: Any,
        encoder_tokens: torch.Tensor,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not hasattr(decoder, "forward_list"):
            raise ExternalDependencyError("MUSt3R decoder has no forward_list(..., return_feats=True) hook")
        level_chunks: list[list[torch.Tensor]] | None = None
        decode_bs = max(1, int(self.config.must3r_decode_batch_size))
        for start in range(0, encoder_tokens.shape[0], decode_bs):
            end = min(encoder_tokens.shape[0], start + decode_bs)
            output = decoder.forward_list(
                [encoder_tokens[start:end].unsqueeze(0)],
                [pos[start:end].unsqueeze(0)],
                [true_shape[start:end].unsqueeze(0)],
                return_feats=True,
            )
            if not isinstance(output, tuple) or len(output) != 3:
                raise ExternalDependencyError("MUSt3R decoder.forward_list did not return (memory, pointmaps, feats)")
            _, _, grouped_feats = output
            if not grouped_feats or not grouped_feats[0]:
                raise ExternalDependencyError("MUSt3R decoder returned no feature levels")
            chunk_levels = [feature[0].detach().cpu() for feature in grouped_feats[0]]
            if level_chunks is None:
                level_chunks = [[] for _ in chunk_levels]
            if len(chunk_levels) != len(level_chunks):
                raise ExternalDependencyError(
                    f"MUSt3R decoder returned {len(chunk_levels)} levels, expected {len(level_chunks)}"
                )
            for level_index, chunk in enumerate(chunk_levels):
                level_chunks[level_index].append(chunk)
            del output, grouped_feats, chunk_levels
        if level_chunks is None:
            raise ExternalDependencyError("MUSt3R decoder returned no feature levels")
        return [torch.cat(chunks, dim=0) for chunks in level_chunks]

    def _normalize_feature_layers(self, num_levels: int) -> tuple[tuple[FeatureLayerSpec, ...], tuple[int, ...]]:
        specs: list[FeatureLayerSpec] = []
        indices: list[int] = []
        for layer in self.config.must3r_feature_layers:
            if layer == "encoder":
                index = 0
            elif isinstance(layer, int) and layer >= 0:
                index = layer + 1
            elif isinstance(layer, int):
                index = num_levels + layer
            else:
                raise ExternalDependencyError(f"Unsupported MUSt3R feature layer spec: {layer!r}")
            if index < 0 or index >= num_levels:
                raise ExternalDependencyError(
                    f"Requested MUSt3R feature layer {layer}, but decoder returned {num_levels} levels"
                )
            specs.append(layer)
            indices.append(index)
        return tuple(specs), tuple(indices)

    def _tokens_to_chw(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
        patch_size: int,
        spec: FeatureLayerSpec,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ExternalDependencyError(f"Expected MUSt3R token features with shape TNC, got {tuple(tokens.shape)}")
        expected_channels = self._expected_channels(spec)
        if tokens.shape[2] != expected_channels:
            raise ExternalDependencyError(
                "MUSt3R feature channel mismatch for "
                f"{self._feature_layer_label(spec)}: got {tokens.shape[2]}, expected {expected_channels}. "
                "The 3AM paper uses encoder=1024 channels and decoder layers=768 channels."
            )
        if pos.shape[:2] != tokens.shape[:2]:
            raise ExternalDependencyError(
                f"MUSt3R token/position mismatch: tokens {tuple(tokens.shape)}, pos {tuple(pos.shape)}"
            )
        if not torch.equal(true_shape, true_shape[:1].expand_as(true_shape)):
            raise ExternalDependencyError(
                "All frames in one training sample must resize to the same true_shape for MUSt3R token-grid export"
            )
        height = int(true_shape[0, 0].item())
        width = int(true_shape[0, 1].item())
        grid_h = height // patch_size
        grid_w = width // patch_size
        if grid_h * grid_w != tokens.shape[1]:
            raise ExternalDependencyError(
                f"Could not reshape {tokens.shape[1]} MUSt3R tokens into token grid {(grid_h, grid_w)} "
                f"for true_shape {(height, width)} and patch_size {patch_size}."
            )
        self._validate_pos_grid(pos, grid_h, grid_w)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w).contiguous()

    def _validate_pos_grid(self, pos: torch.Tensor, grid_h: int, grid_w: int) -> None:
        for frame_index, frame_pos in enumerate(pos.detach().cpu()):
            if frame_pos.ndim != 2 or frame_pos.shape[1] != 2:
                raise ExternalDependencyError(f"Expected MUSt3R positions with shape N2, got {tuple(frame_pos.shape)}")
            first = torch.unique(frame_pos[:, 0]).numel()
            second = torch.unique(frame_pos[:, 1]).numel()
            if sorted((int(first), int(second))) != sorted((grid_h, grid_w)):
                raise ExternalDependencyError(
                    f"MUSt3R position grid mismatch for frame {frame_index}: "
                    f"pos unique counts {(int(first), int(second))}, expected {(grid_h, grid_w)}."
                )

    def _validate_expected_channels(self, features: tuple[torch.Tensor, ...]) -> None:
        expected = self.config.must3r_expected_channels
        if expected is None:
            return
        actual = tuple(int(feature.shape[1]) for feature in features)
        if actual != expected:
            raise ExternalDependencyError(
                "Online MUSt3R feature channel mismatch: "
                f"extracted {list(actual)}, but model.must3r_channels is {list(expected)}. "
                f"Set model.must3r_channels to {list(actual)} or choose matching feature layers."
            )

    def _expected_channels(self, spec: FeatureLayerSpec) -> int:
        return 1024 if spec == "encoder" else 768

    def _feature_layer_label(self, spec: FeatureLayerSpec) -> str:
        return "encoder" if spec == "encoder" else f"decoder_{spec}"


Sam2Adapter = Sam2TrainingAdapter
Must3rAdapter = Must3rFeatureAdapter
