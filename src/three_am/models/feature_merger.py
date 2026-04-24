from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FeatureMerger(nn.Module):
    """Fuse SAM2 appearance features with multi-level MUSt3R geometry features.

    The paper describes a lightweight merger with cross-attention and convolutional
    refinement. This implementation keeps that contract explicit while avoiding any
    dependency on unreleased 3AM code.
    """

    def __init__(
        self,
        sam_channels: int,
        must3r_channels: Sequence[int],
        hidden_channels: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if not must3r_channels:
            raise ValueError("must3r_channels must contain at least one level")
        self.sam_projection = nn.Conv2d(sam_channels, hidden_channels, kernel_size=1)
        self.must3r_projections = nn.ModuleList(
            nn.Conv2d(channels, hidden_channels, kernel_size=1) for channels in must3r_channels
        )
        self.cross_attention = nn.MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, sam_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(nn.Conv2d(sam_channels * 2, sam_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, sam_feature: torch.Tensor, must3r_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(must3r_features) != len(self.must3r_projections):
            raise ValueError(
                f"Expected {len(self.must3r_projections)} MUSt3R levels, got {len(must3r_features)}"
            )
        batch_size, _, height, width = sam_feature.shape
        sam_hidden = self.sam_projection(sam_feature)
        geometry_hidden = torch.zeros_like(sam_hidden)
        for feature, projection in zip(must3r_features, self.must3r_projections, strict=True):
            projected = projection(feature)
            if projected.shape[-2:] != (height, width):
                projected = F.interpolate(projected, size=(height, width), mode="bilinear", align_corners=False)
            geometry_hidden = geometry_hidden + projected
        geometry_hidden = geometry_hidden / len(must3r_features)

        query = sam_hidden.flatten(2).transpose(1, 2)
        key_value = geometry_hidden.flatten(2).transpose(1, 2)
        attended, _ = self.cross_attention(query=query, key=key_value, value=key_value, need_weights=False)
        attended_map = attended.transpose(1, 2).reshape(batch_size, -1, height, width)
        update = self.refine(torch.cat([sam_hidden, attended_map], dim=1))
        gate = self.gate(torch.cat([sam_feature, update], dim=1))
        return sam_feature + gate * update
