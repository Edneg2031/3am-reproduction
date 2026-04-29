from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .feature_merger import FeatureMerger, Must3rFeatureBundle


@dataclass(frozen=True)
class ThreeAMConfig:
    sam_channels: int = 256
    must3r_channels: tuple[int, ...] = (256, 512, 768)
    hidden_channels: int = 256
    attention_heads: int = 8
    geometry_channels: int | None = None
    strict_paper: bool = False


class ThreeAMCore(nn.Module):
    """Reimplementation core for the 3AM feature-fusion contract."""

    def __init__(self, config: ThreeAMConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_merger = FeatureMerger(
            sam_channels=config.sam_channels,
            must3r_channels=config.must3r_channels,
            hidden_channels=config.hidden_channels,
            num_heads=config.attention_heads,
            geometry_channels=config.geometry_channels,
            strict_paper=config.strict_paper,
        )

    def forward_features(
        self,
        sam_feature: torch.Tensor,
        must3r_features: Sequence[torch.Tensor] | Must3rFeatureBundle,
    ) -> torch.Tensor:
        return self.feature_merger(sam_feature, must3r_features)
