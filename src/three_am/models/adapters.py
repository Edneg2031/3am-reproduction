from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .feature_merger import Must3rFeatureBundle

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
    must3r_memory_window: int | None = None
    must3r_full_scene_memory: bool = False
    must3r_feature_layers: tuple[FeatureLayerSpec, ...] = ("encoder", 4, 7, 11)
    must3r_expected_channels: tuple[int, ...] | None = None
    strict_paper: bool = False


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

    @property
    def last_feature_selector(self) -> tuple[str, str | int] | None:
        return self._last_feature_selector

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
        features = self._encode_sam_backbone_payload(images)
        sam_feature, selector = self._select_sam_feature(features)
        self._remember_feature_payload(features, selector, sam_feature)
        return sam_feature

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

    def _encode_sam_backbone_payload(self, images: torch.Tensor) -> Any:
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        if hasattr(self.model, "encode_sam_features"):
            return self.model.encode_sam_features(images)  # type: ignore[attr-defined]
        if hasattr(self.model, "forward_image"):
            return self.model.forward_image(images)  # type: ignore[attr-defined]
        if hasattr(self.model, "image_encoder"):
            return self.model.image_encoder(images)  # type: ignore[attr-defined]
        raise ExternalDependencyError(
            "SAM2 adapter could not obtain image features. Install official SAM2 training internals "
            "or subclass Sam2TrainingAdapter for the selected SAM2 release."
        )

    def forward_train_sequence(self, batch: Any, merged_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        backbone_out = self._inject_merged_features(merged_features)
        if self.config.strict_paper:
            self._validate_strict_backbone_injection(backbone_out, merged_features)
            outputs = self._forward_with_official_sam2_modules(batch, merged_features, backbone_out, require_tracking=True)
            if outputs is None:
                raise ExternalDependencyError(
                    "Strict paper training requires official SAM2 tracking modules: _prepare_backbone_features, "
                    "_track_step, and _encode_memory_in_output. The per-frame SAM head fallback is disabled."
                )
            return outputs
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
        outputs = self._forward_with_official_sam2_modules(batch, merged_features, backbone_out)
        if outputs is not None:
            return outputs
        raise ExternalDependencyError(
            "SAM2 training forward API is not available for merged 3AM features. Provide a SAM2 training wrapper "
            "with one of forward_train_sequence_with_backbone(...), forward_train_sequence_from_backbone(...), "
            "or forward_train_sequence(batch, merged_features), returning mask_logits, iou_scores, and "
            "occlusion_logits."
        )

    def predict_masks_from_points(
        self,
        images: torch.Tensor,
        points_by_frame: Sequence[torch.Tensor],
        labels_by_frame: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Generate detached SAM2 masks from image-space point prompts."""
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        self._validate_image_tensor(images)
        if len(points_by_frame) != int(images.shape[0]) or len(labels_by_frame) != int(images.shape[0]):
            raise ValueError("points_by_frame and labels_by_frame must have one entry per image frame")
        with torch.no_grad():
            backbone_out = self._encode_sam_backbone_payload(images)
            feature, _ = self._select_sam_feature(backbone_out)
            if hasattr(self.model, "_prepare_backbone_features") and hasattr(self.model, "_track_step"):
                return self._predict_point_masks_with_tracking(images, backbone_out, points_by_frame, labels_by_frame)
            if hasattr(self.model, "_forward_sam_heads"):
                return self._predict_point_masks_per_frame(images, feature, backbone_out, points_by_frame, labels_by_frame)
        raise ExternalDependencyError(
            "SAM2 point-prompt mask generation requires official _track_step or _forward_sam_heads internals."
        )

    def track_masks_from_points(
        self,
        images: torch.Tensor,
        *,
        reference_index: int,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate detached video masks by tracking one point prompt through the sequence."""
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        self._validate_image_tensor(images)
        if not (hasattr(self.model, "_prepare_backbone_features") and hasattr(self.model, "_track_step")):
            raise ExternalDependencyError("SAM2 point-prompt tracking requires official _prepare_backbone_features and _track_step")
        reference_index = min(max(int(reference_index), 0), int(images.shape[0]) - 1)
        with torch.no_grad():
            backbone_out = self._encode_sam_backbone_payload(images)
            return self._track_point_masks_with_backbone(images, backbone_out, reference_index, points, labels)

    def _remember_feature_payload(
        self,
        payload: Any,
        selector: tuple[str, str | int] | None,
        sam_feature: torch.Tensor,
    ) -> None:
        self._last_feature_payload = payload
        self._last_feature_selector = selector
        self._last_sam_feature_shape = tuple(sam_feature.shape)

    def _call_with_grad(self, method: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Call SAM2 internals without inference/no-grad decorators when possible."""
        callable_method = method
        receiver = getattr(method, "__self__", None)
        while hasattr(callable_method, "__wrapped__"):
            callable_method = callable_method.__wrapped__
        with torch.enable_grad():
            if receiver is not None and not inspect.ismethod(callable_method):
                return callable_method(receiver, *args, **kwargs)
            return callable_method(*args, **kwargs)

    def _select_sam_feature(self, features: Any) -> tuple[torch.Tensor, tuple[str, str | int] | None]:
        if isinstance(features, torch.Tensor):
            return features, None
        if isinstance(features, dict):
            value = features.get("backbone_fpn")
            if isinstance(value, (list, tuple)) and value and isinstance(value[-1], torch.Tensor):
                return value[-1], ("dict_list", "backbone_fpn")
            if isinstance(value, torch.Tensor):
                return value, ("dict", "backbone_fpn")
            for key in ("vision_features", "image_embed"):
                value = features.get(key)
                if isinstance(value, torch.Tensor):
                    return value, ("dict", key)
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
            self._replace_matching_feature_aliases(updated, merged_features)
            return updated
        if kind == "dict_list":
            updated = dict(payload)
            values = list(updated[key])
            values[-1] = merged_features
            updated[key] = values
            self._replace_matching_feature_aliases(updated, merged_features)
            return updated
        if kind == "sequence":
            values = list(payload)
            values[int(key)] = merged_features
            return values
        return None

    def _replace_matching_feature_aliases(self, payload: dict[str, Any], merged_features: torch.Tensor) -> None:
        expected_shape = tuple(merged_features.shape)
        for key in ("vision_features", "image_embed"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape:
                payload[key] = merged_features
        value = payload.get("backbone_fpn")
        if isinstance(value, (list, tuple)) and value:
            values = list(value)
            for index in range(len(values) - 1, -1, -1):
                if isinstance(values[index], torch.Tensor) and tuple(values[index].shape) == expected_shape:
                    values[index] = merged_features
                    break
            payload["backbone_fpn"] = values
        elif isinstance(value, torch.Tensor) and tuple(value.shape) == expected_shape:
            payload["backbone_fpn"] = merged_features

    def _validate_strict_backbone_injection(self, backbone_out: Any | None, merged_features: torch.Tensor) -> None:
        if not isinstance(backbone_out, dict):
            raise ExternalDependencyError(
                "Strict paper training requires merged 3AM features to be injected into SAM2 backbone_fpn."
            )
        feature_maps = backbone_out.get("backbone_fpn")
        tensors: list[torch.Tensor] = []
        if isinstance(feature_maps, torch.Tensor):
            tensors = [feature_maps]
        elif isinstance(feature_maps, (list, tuple)):
            tensors = [feature for feature in feature_maps if isinstance(feature, torch.Tensor)]
        if not tensors:
            raise ExternalDependencyError(
                "Strict paper training requires SAM2 backbone_fpn tensors so merged 3AM features feed tracking."
            )
        if not any(feature is merged_features for feature in tensors):
            raise ExternalDependencyError(
                "Strict paper training did not inject merged 3AM features into backbone_fpn. "
                f"Selected SAM2 feature selector was {self._last_feature_selector!r}."
            )

    def _forward_with_official_sam2_modules(
        self,
        batch: Any,
        merged_features: torch.Tensor,
        backbone_out: Any | None,
        *,
        require_tracking: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        if backbone_out is None:
            if require_tracking:
                return None
            return self._forward_sam_heads_per_frame(batch, merged_features, backbone_out=None)
        if (
            hasattr(self.model, "_prepare_backbone_features")
            and hasattr(self.model, "_track_step")
            and hasattr(self.model, "_encode_memory_in_output")
        ):
            return self._forward_track_steps(batch, backbone_out)
        if require_tracking:
            return None
        return self._forward_sam_heads_per_frame(batch, merged_features, backbone_out=backbone_out)

    def _forward_track_steps(self, batch: Any, backbone_out: Any) -> dict[str, torch.Tensor] | None:
        if self.model is None or not hasattr(self.model, "_prepare_backbone_features"):
            return None
        try:
            _, vision_feats, vision_pos_embeds, feat_sizes = self._call_with_grad(
                self.model._prepare_backbone_features,
                backbone_out,
            )
        except Exception as error:
            raise ExternalDependencyError("SAM2 could not prepare merged backbone features for tracking") from error
        num_frames = int(getattr(batch, "target_masks").shape[0])
        reference_index = int(getattr(batch.prompt, "frame_index", 0))
        output_dict: dict[str, dict[int, dict[str, Any]]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        ordered = [reference_index]
        ordered.extend(index for index in range(reference_index + 1, num_frames))
        ordered.extend(index for index in range(reference_index - 1, -1, -1))
        frame_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for frame_index in ordered:
            current_feats = [feature[:, frame_index : frame_index + 1, :] for feature in vision_feats]
            current_pos = [pos[:, frame_index : frame_index + 1, :] for pos in vision_pos_embeds]
            point_inputs, mask_inputs = self._prompt_inputs_for_frame(batch, frame_index, current_feats[-1].device)
            is_cond_frame = frame_index == reference_index
            try:
                current_out, sam_outputs, _, _ = self._call_with_grad(
                    self.model._track_step,
                    frame_idx=frame_index,
                    is_init_cond_frame=is_cond_frame,
                    current_vision_feats=current_feats,
                    current_vision_pos_embeds=current_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=frame_index < reference_index,
                    prev_sam_mask_logits=None,
                )
            except Exception as error:
                raise ExternalDependencyError("SAM2 _track_step failed on merged 3AM features") from error
            low_res_masks, high_res_masks, iou_scores, object_score_logits = self._unpack_sam_outputs(sam_outputs)
            current_out["pred_masks"] = low_res_masks
            current_out["pred_masks_high_res"] = high_res_masks
            current_out["obj_ptr"] = sam_outputs[5]
            current_out["object_score_logits"] = object_score_logits
            if hasattr(self.model, "_encode_memory_in_output"):
                self._call_with_grad(
                    self.model._encode_memory_in_output,
                    current_feats,
                    feat_sizes,
                    point_inputs,
                    True,
                    high_res_masks,
                    object_score_logits,
                    current_out,
                )
            output_group = "cond_frame_outputs" if is_cond_frame else "non_cond_frame_outputs"
            output_dict[output_group][frame_index] = current_out
            frame_outputs[frame_index] = (high_res_masks[:, 0], iou_scores, object_score_logits)
        return self._format_sam_training_outputs(batch, frame_outputs)

    def _forward_sam_heads_per_frame(
        self,
        batch: Any,
        merged_features: torch.Tensor,
        *,
        backbone_out: Any | None,
    ) -> dict[str, torch.Tensor] | None:
        if self.model is None or not hasattr(self.model, "_forward_sam_heads"):
            return None
        num_frames = int(getattr(batch, "target_masks").shape[0])
        high_res_features_all = self._high_res_features_from_backbone(backbone_out)
        frame_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for frame_index in range(num_frames):
            feature = merged_features[frame_index : frame_index + 1]
            point_inputs, mask_inputs = self._prompt_inputs_for_frame(batch, frame_index, feature.device)
            high_res_features = None
            if high_res_features_all is not None:
                high_res_features = [level[frame_index : frame_index + 1] for level in high_res_features_all]
            try:
                sam_outputs = self._call_with_grad(
                    self.model._forward_sam_heads,
                    backbone_features=feature,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=False,
                )
            except Exception as error:
                raise ExternalDependencyError("SAM2 _forward_sam_heads failed on merged 3AM features") from error
            _, high_res_masks, iou_scores, object_score_logits = self._unpack_sam_outputs(sam_outputs)
            frame_outputs[frame_index] = (high_res_masks[:, 0], iou_scores, object_score_logits)
        return self._format_sam_training_outputs(batch, frame_outputs)

    def _predict_point_masks_with_tracking(
        self,
        images: torch.Tensor,
        backbone_out: Any,
        points_by_frame: Sequence[torch.Tensor],
        labels_by_frame: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        try:
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)  # type: ignore[union-attr]
        except Exception as error:
            raise ExternalDependencyError("SAM2 could not prepare backbone features for point-prompt masks") from error
        masks: list[torch.Tensor] = []
        output_dict: dict[str, dict[int, dict[str, Any]]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        num_frames = int(images.shape[0])
        for frame_index in range(num_frames):
            current_feats = [feature[:, frame_index : frame_index + 1, :] for feature in vision_feats]
            current_pos = [pos[:, frame_index : frame_index + 1, :] for pos in vision_pos_embeds]
            point_inputs = self._point_prompt_inputs(points_by_frame[frame_index], labels_by_frame[frame_index], current_feats[-1].device)
            try:
                current_out, sam_outputs, _, _ = self.model._track_step(  # type: ignore[union-attr]
                    frame_idx=frame_index,
                    is_init_cond_frame=True,
                    current_vision_feats=current_feats,
                    current_vision_pos_embeds=current_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=False,
                    prev_sam_mask_logits=None,
                )
            except Exception as error:
                raise ExternalDependencyError("SAM2 _track_step failed while generating point-prompt masks") from error
            _, high_res_masks, _, object_score_logits = self._unpack_sam_outputs(sam_outputs)
            current_out["pred_masks_high_res"] = high_res_masks
            current_out["object_score_logits"] = object_score_logits
            masks.append(high_res_masks[0, 0])
        return self._resize_predicted_masks(torch.stack(masks, dim=0), tuple(images.shape[-2:]))

    def _track_point_masks_with_backbone(
        self,
        images: torch.Tensor,
        backbone_out: Any,
        reference_index: int,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        try:
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)  # type: ignore[union-attr]
        except Exception as error:
            raise ExternalDependencyError("SAM2 could not prepare backbone features for point-prompt tracking") from error
        num_frames = int(images.shape[0])
        output_dict: dict[str, dict[int, dict[str, Any]]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        ordered = [reference_index]
        ordered.extend(index for index in range(reference_index + 1, num_frames))
        ordered.extend(index for index in range(reference_index - 1, -1, -1))
        frame_masks: dict[int, torch.Tensor] = {}
        for frame_index in ordered:
            current_feats = [feature[:, frame_index : frame_index + 1, :] for feature in vision_feats]
            current_pos = [pos[:, frame_index : frame_index + 1, :] for pos in vision_pos_embeds]
            point_inputs = (
                self._point_prompt_inputs(points, labels, current_feats[-1].device)
                if frame_index == reference_index
                else None
            )
            try:
                current_out, sam_outputs, _, _ = self.model._track_step(  # type: ignore[union-attr]
                    frame_idx=frame_index,
                    is_init_cond_frame=frame_index == reference_index,
                    current_vision_feats=current_feats,
                    current_vision_pos_embeds=current_pos,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=frame_index < reference_index,
                    prev_sam_mask_logits=None,
                )
            except Exception as error:
                raise ExternalDependencyError("SAM2 _track_step failed while tracking point-prompt masks") from error
            low_res_masks, high_res_masks, _, object_score_logits = self._unpack_sam_outputs(sam_outputs)
            current_out["pred_masks"] = low_res_masks
            current_out["pred_masks_high_res"] = high_res_masks
            current_out["obj_ptr"] = sam_outputs[5]
            current_out["object_score_logits"] = object_score_logits
            if hasattr(self.model, "_encode_memory_in_output"):
                self.model._encode_memory_in_output(  # type: ignore[union-attr]
                    current_feats,
                    feat_sizes,
                    point_inputs,
                    True,
                    high_res_masks,
                    object_score_logits,
                    current_out,
                )
            group = "cond_frame_outputs" if frame_index == reference_index else "non_cond_frame_outputs"
            output_dict[group][frame_index] = current_out
            frame_masks[frame_index] = high_res_masks[0, 0]
        return self._resize_predicted_masks(
            torch.stack([frame_masks[index] for index in range(num_frames)], dim=0),
            tuple(images.shape[-2:]),
        )

    def _predict_point_masks_per_frame(
        self,
        images: torch.Tensor,
        feature: torch.Tensor,
        backbone_out: Any | None,
        points_by_frame: Sequence[torch.Tensor],
        labels_by_frame: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        high_res_features_all = self._high_res_features_from_backbone(backbone_out)
        masks: list[torch.Tensor] = []
        for frame_index in range(int(images.shape[0])):
            frame_feature = feature[frame_index : frame_index + 1]
            high_res_features = None
            if high_res_features_all is not None:
                high_res_features = [level[frame_index : frame_index + 1] for level in high_res_features_all]
            point_inputs = self._point_prompt_inputs(points_by_frame[frame_index], labels_by_frame[frame_index], frame_feature.device)
            try:
                sam_outputs = self.model._forward_sam_heads(  # type: ignore[union-attr]
                    backbone_features=frame_feature,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    high_res_features=high_res_features,
                    multimask_output=False,
                )
            except Exception as error:
                raise ExternalDependencyError("SAM2 _forward_sam_heads failed while generating point-prompt masks") from error
            _, high_res_masks, _, _ = self._unpack_sam_outputs(sam_outputs)
            masks.append(high_res_masks[0, 0])
        return self._resize_predicted_masks(torch.stack(masks, dim=0), tuple(images.shape[-2:]))

    def _point_prompt_inputs(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "point_coords": points.to(device=device, dtype=torch.float32)[None],
            "point_labels": labels.to(device=device, dtype=torch.int32)[None],
        }

    def _resize_predicted_masks(self, masks: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if tuple(masks.shape[-2:]) == target_size:
            return masks.float()
        return F.interpolate(
            masks[:, None].float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]

    def _prompt_inputs_for_frame(
        self,
        batch: Any,
        frame_index: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        prompt = getattr(batch, "prompt", None)
        if prompt is None or int(prompt.frame_index) != frame_index:
            return None, None
        if prompt.type == "mask" and prompt.mask is not None:
            return None, prompt.mask.to(device=device, dtype=torch.float32)[None, None]
        if prompt.type == "point" and prompt.points is not None and prompt.point_labels is not None:
            return {
                "point_coords": prompt.points.to(device=device, dtype=torch.float32)[None],
                "point_labels": prompt.point_labels.to(device=device, dtype=torch.int32)[None],
            }, None
        if prompt.type == "box" and prompt.box is not None:
            box = prompt.box.to(device=device, dtype=torch.float32)
            coords = torch.stack([box[:2], box[2:]], dim=0)[None]
            labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
            return {"point_coords": coords, "point_labels": labels}, None
        return None, None

    def _high_res_features_from_backbone(self, backbone_out: Any | None) -> list[torch.Tensor] | None:
        if not isinstance(backbone_out, dict):
            return None
        feature_maps = backbone_out.get("backbone_fpn")
        if not isinstance(feature_maps, (list, tuple)) or len(feature_maps) < 2:
            return None
        num_levels = int(getattr(self.model, "num_feature_levels", min(3, len(feature_maps))))
        selected = list(feature_maps[-num_levels:])
        return [feature for feature in selected[:-1] if isinstance(feature, torch.Tensor)] or None

    def _unpack_sam_outputs(self, sam_outputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(sam_outputs, (list, tuple)) or len(sam_outputs) < 7:
            raise ExternalDependencyError("SAM2 SAM heads returned an unsupported output payload")
        low_res_masks = sam_outputs[3]
        high_res_masks = sam_outputs[4]
        iou_scores = sam_outputs[2]
        object_score_logits = sam_outputs[6]
        if iou_scores.ndim == 2 and iou_scores.shape[1] > 1:
            iou_scores = iou_scores.max(dim=1).values
        else:
            iou_scores = iou_scores.flatten()
        if object_score_logits.ndim == 1:
            object_score_logits = object_score_logits[:, None]
        if object_score_logits.shape[1] != 1:
            object_score_logits = object_score_logits[:, :1]
        return low_res_masks, high_res_masks, iou_scores, object_score_logits

    def _format_sam_training_outputs(
        self,
        batch: Any,
        frame_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        target_masks = getattr(batch, "target_masks")
        masks: list[torch.Tensor] = []
        ious: list[torch.Tensor] = []
        object_scores: list[torch.Tensor] = []
        for frame_index in range(int(target_masks.shape[0])):
            if frame_index not in frame_outputs:
                raise ExternalDependencyError(f"SAM2 did not produce output for frame index {frame_index}")
            mask_logits, iou_score, object_score = frame_outputs[frame_index]
            if mask_logits.shape[-2:] != target_masks.shape[-2:]:
                mask_logits = F.interpolate(
                    mask_logits[:, None].float(),
                    size=tuple(target_masks.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            masks.append(mask_logits[0])
            ious.append(iou_score.reshape(-1)[0])
            object_scores.append(object_score.reshape(-1)[0])
        object_scores_tensor = torch.stack(object_scores)
        return {
            "mask_logits": torch.stack(masks, dim=0),
            "iou_scores": torch.stack(ious, dim=0),
            "occlusion_logits": torch.stack([-object_scores_tensor, object_scores_tensor], dim=1),
        }

    def _frame_output_grad_summary(
        self,
        frame_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> str:
        parts: list[str] = []
        for frame_index in sorted(frame_outputs):
            mask_logits, iou_score, object_score = frame_outputs[frame_index]
            parts.append(
                "frame="
                f"{frame_index}:mask_grad={mask_logits.requires_grad},"
                f"iou_grad={iou_score.requires_grad},object_grad={object_score.requires_grad}"
            )
        return "; ".join(parts)

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
            return self._call_with_grad(method, batch=batch, merged_features=merged_features, backbone_out=backbone_out)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return self._call_with_grad(method, **kwargs)
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
        return self._call_with_grad(method, **supported)

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
                "or set external.must3r_repo to the cloned repo root. "
                f"Underlying import error: {type(error).__name__}: {error}"
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
    ) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
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

    def extract_features_from_paths(self, image_paths: Sequence[Path]) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
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
                chunk_size = self._decode_chunk_size(len(image_paths))
                point_chunks: list[torch.Tensor] = []
                for start in range(0, len(image_paths), chunk_size):
                    end = min(len(image_paths), start + chunk_size)
                    images, true_shape = self._load_official_images(image_paths[start:end], patch_size)
                    true_shape_tensor = torch.stack(true_shape, dim=0)
                    true_shape_chunks.append(true_shape_tensor)
                    image_tensor = torch.stack(images, dim=0).to(device)
                    chunk_true_shape = true_shape_tensor.to(device)
                    encoder_tokens, pos = self._encode_frames(encoder, image_tensor, chunk_true_shape)
                    pos_chunks.append(pos.detach().cpu())
                    raw_levels, point_map = self._decode_feature_levels(
                        decoder,
                        encoder_tokens,
                        pos,
                        chunk_true_shape,
                        return_point_map=True,
                    )
                    if point_map is not None:
                        point_chunks.append(point_map)
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
        features = tuple(feature.detach().cpu() for feature in features)
        if self.config.strict_paper:
            point_map = torch.cat(point_chunks, dim=0) if point_chunks else None
            if point_map is None:
                raise ExternalDependencyError("Strict paper online MUSt3R extraction requires decoder point maps for PE3D")
            pe2d = self._positions_to_chw(pos_tensor, true_shape_tensor, patch_size)
            ray_map = self._ray_map_from_pe2d(pe2d)
            metadata = {
                "feature_specs": [self._feature_layer_label(spec) for spec in selected_specs],
                "feature_channels": [int(feature.shape[1]) for feature in features],
                "decoder_memory": self._decoder_memory_enabled(len(image_paths)),
                "memory_window": self.config.must3r_memory_window,
                "full_scene_memory": self.config.must3r_full_scene_memory,
            }
            return Must3rFeatureBundle(levels=features, pe2d=pe2d, point_map=point_map, ray_map=ray_map, metadata=metadata)
        return features

    def _decode_chunk_size(self, num_frames: int) -> int:
        if self.config.must3r_full_scene_memory:
            return max(1, num_frames)
        if self.config.must3r_memory_window is not None:
            return max(1, int(self.config.must3r_memory_window))
        return max(1, int(self.config.must3r_decode_batch_size))

    def _decoder_memory_enabled(self, num_frames: int) -> bool:
        return self.config.must3r_full_scene_memory or self._decode_chunk_size(num_frames) > 1

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
        *,
        return_point_map: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor | None]:
        if not hasattr(decoder, "forward_list"):
            raise ExternalDependencyError("MUSt3R decoder has no forward_list(..., return_feats=True) hook")
        level_chunks: list[list[torch.Tensor]] | None = None
        decode_bs = (
            max(1, int(encoder_tokens.shape[0]))
            if self.config.must3r_full_scene_memory or self.config.must3r_memory_window is not None
            else max(1, int(self.config.must3r_decode_batch_size))
        )
        point_chunks: list[torch.Tensor] = []
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
            _, pointmaps, grouped_feats = output
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
            point_map = self._pointmaps_to_chw(pointmaps, pos[start:end], true_shape[start:end])
            if point_map is not None:
                point_chunks.append(point_map)
            del output, grouped_feats, chunk_levels
        if level_chunks is None:
            raise ExternalDependencyError("MUSt3R decoder returned no feature levels")
        levels = [torch.cat(chunks, dim=0) for chunks in level_chunks]
        if return_point_map:
            return levels, torch.cat(point_chunks, dim=0) if point_chunks else None
        return levels

    def _pointmaps_to_chw(
        self,
        pointmaps: Any,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
    ) -> torch.Tensor | None:
        tensors: list[torch.Tensor] = []
        self._collect_tensors(pointmaps, tensors)
        for tensor in reversed(tensors):
            candidate = tensor.detach().cpu()
            point_map = self._candidate_pointmap_to_chw(
                candidate,
                pos.detach().cpu(),
                true_shape.detach().cpu(),
            )
            if point_map is not None:
                return point_map
        return None

    def _candidate_pointmap_to_chw(
        self,
        candidate: torch.Tensor,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
    ) -> torch.Tensor | None:
        patch_size = self._patch_size_from_pos(pos)
        num_frames = int(pos.shape[0])
        if candidate.ndim == 5 and candidate.shape[0] == 1:
            candidate = candidate[0]
        elif candidate.ndim == 5 and candidate.shape[0] * candidate.shape[1] == num_frames:
            candidate = candidate.reshape(num_frames, *candidate.shape[2:])
        if candidate.ndim == 3 and candidate.shape[0] == num_frames and candidate.shape[-1] >= 3:
            return self._tokens_to_chw(candidate[..., :3], pos, true_shape, patch_size, "point_map")
        if candidate.ndim != 4 or candidate.shape[0] != num_frames:
            return None
        grid_h, grid_w = self._grid_shape(true_shape, patch_size)
        height = int(true_shape[0, 0].item())
        width = int(true_shape[0, 1].item())
        if candidate.shape[1] >= 3 and self._looks_like_pointmap_spatial(candidate.shape[-2:], height, width, grid_h, grid_w):
            return self._resize_pointmap_to_grid(candidate[:, :3].contiguous(), grid_h, grid_w)
        if candidate.shape[-1] >= 3 and self._looks_like_pointmap_spatial(candidate.shape[1:3], height, width, grid_h, grid_w):
            return self._resize_pointmap_to_grid(candidate[..., :3].permute(0, 3, 1, 2).contiguous(), grid_h, grid_w)
        return None

    def _grid_shape(self, true_shape: torch.Tensor, patch_size: int) -> tuple[int, int]:
        if not torch.equal(true_shape, true_shape[:1].expand_as(true_shape)):
            raise ExternalDependencyError("All frames in one training sample must resize to the same true_shape")
        height = int(true_shape[0, 0].item())
        width = int(true_shape[0, 1].item())
        return height // patch_size, width // patch_size

    def _looks_like_pointmap_spatial(
        self,
        spatial: torch.Size | tuple[int, int],
        height: int,
        width: int,
        grid_h: int,
        grid_w: int,
    ) -> bool:
        if len(spatial) != 2:
            return False
        shape = (int(spatial[0]), int(spatial[1]))
        return shape in {(height, width), (grid_h, grid_w)}

    def _resize_pointmap_to_grid(self, point_map: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if tuple(point_map.shape[-2:]) == (grid_h, grid_w):
            return point_map.float().contiguous()
        return F.interpolate(point_map.float(), size=(grid_h, grid_w), mode="area").contiguous()

    def _collect_tensors(self, value: Any, out: list[torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                self._collect_tensors(item, out)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._collect_tensors(item, out)

    def _patch_size_from_pos(self, pos: torch.Tensor) -> int:
        return 16

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
        if spec == "point_map":
            return 3
        return 1024 if spec == "encoder" else 768

    def _feature_layer_label(self, spec: FeatureLayerSpec) -> str:
        return "encoder" if spec == "encoder" else f"decoder_{spec}"

    def _positions_to_chw(self, pos: torch.Tensor, true_shape: torch.Tensor, patch_size: int) -> torch.Tensor:
        grid_h, grid_w = self._grid_shape(true_shape, patch_size)
        if grid_h * grid_w != pos.shape[1]:
            raise ExternalDependencyError("Could not reshape MUSt3R PE2D positions into token grid")
        return pos.detach().cpu().transpose(1, 2).reshape(pos.shape[0], 2, grid_h, grid_w).contiguous().float()

    def _ray_map_from_pe2d(self, pe2d: torch.Tensor) -> torch.Tensor:
        pe = pe2d.float()
        if pe.numel() > 0:
            denom = pe.flatten(2).abs().amax(dim=2).clamp_min(1.0)[..., None, None]
            pe = pe / denom
        ones = torch.ones(pe.shape[0], 1, *pe.shape[-2:], dtype=pe.dtype)
        return F.normalize(torch.cat([pe, ones], dim=1), dim=1)


Sam2Adapter = Sam2TrainingAdapter
Must3rAdapter = Must3rFeatureAdapter
