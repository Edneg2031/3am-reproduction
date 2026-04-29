from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ExternalBackboneConfig:
    sam2_checkpoint: Path | None = None
    sam2_config: str | Path | None = None
    sam2_repo: Path | None = None
    must3r_checkpoint: Path | None = None
    must3r_repo: Path | None = None


class ExternalDependencyError(RuntimeError):
    pass


def _prepend_repo_to_sys_path(repo: Path | None) -> None:
    if repo is None:
        return
    repo = repo.expanduser().resolve()
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


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
        self.model: nn.Module | None = None

    def load(self) -> None:
        _prepend_repo_to_sys_path(self.config.must3r_repo)
        try:
            import mast3r  # noqa: F401  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError("Install naver/must3r and its MASt3R dependencies first") from error
        raise NotImplementedError(
            "MUSt3R public APIs vary by release; wire checkpoint loading here after installing naver/must3r."
        )

    def extract_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.model is None:
            raise ExternalDependencyError("MUSt3R model is not loaded")
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


Sam2Adapter = Sam2TrainingAdapter
Must3rAdapter = Must3rFeatureAdapter
