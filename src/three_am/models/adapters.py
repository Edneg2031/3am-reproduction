from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ExternalBackboneConfig:
    sam2_checkpoint: Path | None = None
    sam2_config: Path | None = None
    must3r_checkpoint: Path | None = None


class ExternalDependencyError(RuntimeError):
    pass


class Sam2TrainingAdapter(nn.Module):
    """Thin adapter boundary for official SAM2 code.

    Full SAM2 training internals are imported lazily because installation requires
    the official repository and CUDA dependencies.
    """

    def __init__(self, config: ExternalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model: nn.Module | None = None

    def load(self) -> None:
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise ExternalDependencyError("Install facebookresearch/sam2 before loading SAM2") from error
        if self.config.sam2_config is None or self.config.sam2_checkpoint is None:
            raise ValueError("sam2_config and sam2_checkpoint are required")
        self.model = build_sam2_video_predictor(str(self.config.sam2_config), str(self.config.sam2_checkpoint))

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
            return self.model.encode_sam_features(images)  # type: ignore[no-any-return]
        if hasattr(self.model, "image_encoder"):
            features = self.model.image_encoder(images)  # type: ignore[attr-defined]
            if isinstance(features, torch.Tensor):
                return features
            if isinstance(features, dict):
                for key in ("vision_features", "backbone_fpn", "image_embed"):
                    value = features.get(key)
                    if isinstance(value, torch.Tensor):
                        return value
                    if isinstance(value, list) and value and isinstance(value[-1], torch.Tensor):
                        return value[-1]
            if isinstance(features, (list, tuple)) and features and isinstance(features[-1], torch.Tensor):
                return features[-1]
        raise ExternalDependencyError(
            "SAM2 adapter could not obtain trainable image features. Install official SAM2 training internals "
            "or subclass Sam2TrainingAdapter.encode_sam_features for the selected SAM2 release."
        )

    def forward_train_sequence(self, batch: Any, merged_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ExternalDependencyError("SAM2 model is not loaded")
        for method_name in ("forward_train_sequence", "training_forward", "forward_train"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                outputs = method(batch=batch, merged_features=merged_features)
                if not isinstance(outputs, dict):
                    raise ExternalDependencyError(f"SAM2 {method_name} must return a dict of training tensors")
                return outputs
        raise ExternalDependencyError(
            "SAM2 training forward API is not available. Provide an adapter that returns "
            "mask_logits, iou_scores, and occlusion_logits from forward_train_sequence()."
        )


class Must3rFeatureAdapter(nn.Module):
    """Adapter boundary for official MUSt3R feature extraction."""

    def __init__(self, config: ExternalBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model: nn.Module | None = None

    def load(self) -> None:
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
