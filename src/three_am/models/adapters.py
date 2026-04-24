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


class Sam2Adapter(nn.Module):
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

    def freeze_vision_encoder(self) -> None:
        if self.model is None:
            return
        for name, parameter in self.model.named_parameters():
            if "image_encoder" in name or "vision_encoder" in name:
                parameter.requires_grad_(False)


class Must3rAdapter(nn.Module):
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

    def extract_features(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        if self.model is None:
            raise ExternalDependencyError("MUSt3R model is not loaded")
        raise NotImplementedError("Connect this to MUSt3R intermediate feature extraction API.")
