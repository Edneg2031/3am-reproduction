from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Must3rFeatureBundle:
    levels: tuple[torch.Tensor, ...]
    pe2d: torch.Tensor | None = None
    point_map: torch.Tensor | None = None
    ray_map: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "Must3rFeatureBundle":
        def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if dtype is not None and tensor.is_floating_point():
                return tensor.to(device=device, dtype=dtype)
            return tensor.to(device=device)

        return Must3rFeatureBundle(
            levels=tuple(move(level) for level in self.levels),  # type: ignore[arg-type]
            pe2d=move(self.pe2d),
            point_map=move(self.point_map),
            ray_map=move(self.ray_map),
            metadata=self.metadata,
        )


class _AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.self_norm = nn.LayerNorm(channels)
        self.cross_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, coarse: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        self_update, _ = self.self_attn(coarse, coarse, coarse, need_weights=False)
        coarse = self.self_norm(coarse + self_update)
        cross_update, _ = self.cross_attn(coarse, key_value, key_value, need_weights=False)
        coarse = self.cross_norm(coarse + cross_update)
        return self.ffn_norm(coarse + self.ffn(coarse))


class FeatureMerger(nn.Module):
    """Fuse SAM2 appearance features with multi-level MUSt3R geometry features.

    In strict-paper mode this follows the paper-level contract: initialize a coarse
    feature from the MUSt3R encoder level, update it sequentially with decoder
    levels through attention blocks, then fuse the result with the SAM2 frame
    embedding. The non-strict path keeps tuple inputs working for tests and
    lightweight development runs by deriving simple positional maps when geometry
    fields are unavailable.
    """

    def __init__(
        self,
        sam_channels: int,
        must3r_channels: Sequence[int],
        hidden_channels: int = 256,
        num_heads: int = 8,
        geometry_channels: int | None = None,
        strict_paper: bool = False,
    ) -> None:
        super().__init__()
        if not must3r_channels:
            raise ValueError("must3r_channels must contain at least one level")
        self.strict_paper = strict_paper
        self.must3r_channels = tuple(int(channels) for channels in must3r_channels)
        self.geometry_channels = int(geometry_channels or hidden_channels)
        self.encoder_projection = nn.Conv2d(self.must3r_channels[0], self.geometry_channels, kernel_size=1)
        self.level_projections = nn.ModuleList(
            nn.Conv2d(channels, self.geometry_channels, kernel_size=1) for channels in self.must3r_channels[1:]
        )
        self.pe3d_projection = nn.Conv2d(6, self.geometry_channels, kernel_size=1)
        self.pe2d_projection = nn.Conv2d(2, self.geometry_channels, kernel_size=1)
        self.initial_self_attention = nn.MultiheadAttention(self.geometry_channels, num_heads, batch_first=True)
        self.initial_norm = nn.LayerNorm(self.geometry_channels)
        self.blocks = nn.ModuleList(_AttentionBlock(self.geometry_channels, num_heads) for _ in self.must3r_channels[1:])
        self.geometry_refine = nn.Sequential(
            nn.Conv2d(self.geometry_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_refine = nn.Sequential(
            nn.Conv2d(sam_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, sam_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(nn.Conv2d(sam_channels * 2, sam_channels, kernel_size=1), nn.Sigmoid())

    def forward(
        self,
        sam_feature: torch.Tensor,
        must3r_features: Sequence[torch.Tensor] | Must3rFeatureBundle,
    ) -> torch.Tensor:
        if sam_feature.ndim != 4:
            raise ValueError(f"SAM2 feature must have shape TCHW, got {tuple(sam_feature.shape)}")
        bundle = self._as_bundle(must3r_features)
        levels = bundle.levels
        if len(levels) != len(self.must3r_channels):
            raise ValueError(
                f"Expected {len(self.must3r_channels)} MUSt3R levels, got {len(levels)}"
            )
        batch_size, _, sam_height, sam_width = sam_feature.shape
        for feature, channels in zip(levels, self.must3r_channels, strict=True):
            if feature.ndim != 4:
                raise ValueError(f"MUSt3R feature must have shape TCHW, got {tuple(feature.shape)}")
            if feature.shape[0] != batch_size:
                raise ValueError(
                    f"MUSt3R feature batch/time dimension {feature.shape[0]} does not match SAM2 {batch_size}"
                )
            if feature.shape[1] != channels:
                raise ValueError(f"MUSt3R feature channel mismatch: got {feature.shape[1]}, expected {channels}")
        levels = tuple(level.to(device=sam_feature.device, dtype=sam_feature.dtype) for level in levels)
        token_size = tuple(levels[0].shape[-2:])
        if any(tuple(level.shape[-2:]) != token_size for level in levels):
            levels = tuple(
                F.interpolate(level, size=token_size, mode="bilinear", align_corners=False)
                if tuple(level.shape[-2:]) != token_size
                else level
                for level in levels
            )

        pe3d = self._pe3d(bundle, levels[0], sam_feature.dtype)
        pe2d = self._pe2d(bundle, levels[0], sam_feature.dtype)
        encoder = self.encoder_projection(levels[0]) + pe3d
        coarse = encoder.flatten(2).transpose(1, 2)
        pe2d_tokens = pe2d.flatten(2).transpose(1, 2)
        initial_update, _ = self.initial_self_attention(coarse + pe2d_tokens, coarse + pe2d_tokens, coarse, need_weights=False)
        coarse = self.initial_norm(coarse + initial_update)

        for level, projection, block in zip(levels[1:], self.level_projections, self.blocks, strict=True):
            key_value = projection(level) + pe3d
            coarse = block(coarse + pe3d.flatten(2).transpose(1, 2), key_value.flatten(2).transpose(1, 2))

        geometry = coarse.transpose(1, 2).reshape(batch_size, self.geometry_channels, *token_size)
        geometry = self.geometry_refine(geometry)
        if geometry.shape[-2:] != (sam_height, sam_width):
            geometry = F.interpolate(geometry, size=(sam_height, sam_width), mode="bilinear", align_corners=False)
        update = self.output_refine(torch.cat([sam_feature, geometry], dim=1))
        gate = self.gate(torch.cat([sam_feature, update], dim=1))
        return sam_feature + gate * update

    def _as_bundle(self, features: Sequence[torch.Tensor] | Must3rFeatureBundle) -> Must3rFeatureBundle:
        if isinstance(features, Must3rFeatureBundle):
            return features
        return Must3rFeatureBundle(levels=tuple(features))

    def _pe3d(self, bundle: Must3rFeatureBundle, reference: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if bundle.point_map is None or bundle.ray_map is None:
            if self.strict_paper:
                raise ValueError("Strict 3AM FeatureMerger requires MUSt3R point_map and ray_map for PE3D")
            point_map = torch.zeros(reference.shape[0], 3, *reference.shape[-2:], device=reference.device, dtype=dtype)
            ray_map = self._default_ray_map(reference)
        else:
            point_map = self._resize_geometry(bundle.point_map, reference, channels=3)
            ray_map = self._resize_geometry(bundle.ray_map, reference, channels=3)
        return self.pe3d_projection(torch.cat([point_map, ray_map], dim=1))

    def _pe2d(self, bundle: Must3rFeatureBundle, reference: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if bundle.pe2d is None:
            if self.strict_paper:
                raise ValueError("Strict 3AM FeatureMerger requires MUSt3R PE2D")
            return self.pe2d_projection(self._default_pe2d(reference))
        return self.pe2d_projection(self._resize_geometry(bundle.pe2d, reference, channels=2))

    def _resize_geometry(self, geometry: torch.Tensor, reference: torch.Tensor, *, channels: int) -> torch.Tensor:
        if geometry.ndim != 4:
            raise ValueError(f"MUSt3R geometry tensor must have shape TCHW, got {tuple(geometry.shape)}")
        if geometry.shape[0] != reference.shape[0]:
            raise ValueError(
                f"MUSt3R geometry time dimension {geometry.shape[0]} does not match features {reference.shape[0]}"
            )
        if geometry.shape[1] < channels:
            raise ValueError(f"MUSt3R geometry tensor must have at least {channels} channels, got {geometry.shape[1]}")
        geometry = geometry[:, :channels].to(device=reference.device, dtype=reference.dtype)
        if geometry.shape[-2:] != reference.shape[-2:]:
            geometry = F.interpolate(geometry, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return geometry

    def _default_pe2d(self, reference: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype),
            torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype),
            indexing="ij",
        )
        pe = torch.stack([x, y], dim=0).expand(reference.shape[0], -1, -1, -1)
        return pe.contiguous()

    def _default_ray_map(self, reference: torch.Tensor) -> torch.Tensor:
        pe2d = self._default_pe2d(reference)
        ones = torch.ones(reference.shape[0], 1, *reference.shape[-2:], device=reference.device, dtype=reference.dtype)
        rays = torch.cat([pe2d, ones], dim=1)
        return F.normalize(rays, dim=1)
