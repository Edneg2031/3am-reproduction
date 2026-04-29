from __future__ import annotations

import torch
import pytest

from three_am.models.feature_merger import FeatureMerger, Must3rFeatureBundle


def test_feature_merger_preserves_sam_shape() -> None:
    merger = FeatureMerger(sam_channels=32, must3r_channels=(16, 24), hidden_channels=32, num_heads=4)
    sam = torch.randn(2, 32, 8, 8)
    must3r = [torch.randn(2, 16, 4, 4), torch.randn(2, 24, 8, 8)]
    merged = merger(sam, must3r)
    assert merged.shape == sam.shape
    assert torch.isfinite(merged).all()


def test_feature_merger_accepts_paper_must3r_shapes_and_cache_dtype() -> None:
    merger = FeatureMerger(sam_channels=256, must3r_channels=(1024, 768, 768, 768), hidden_channels=256, num_heads=8)
    sam = torch.randn(2, 256, 16, 16)
    must3r = [
        torch.randn(2, 1024, 32, 24, dtype=torch.float16),
        torch.randn(2, 768, 32, 24, dtype=torch.float16),
        torch.randn(2, 768, 32, 24, dtype=torch.float16),
        torch.randn(2, 768, 32, 24, dtype=torch.float16),
    ]

    merged = merger(sam, must3r)

    assert merged.shape == sam.shape
    assert merged.dtype == sam.dtype
    assert torch.isfinite(merged).all()


def test_strict_feature_merger_requires_geometry_bundle() -> None:
    merger = FeatureMerger(
        sam_channels=32,
        must3r_channels=(16, 24),
        hidden_channels=32,
        num_heads=4,
        strict_paper=True,
    )
    sam = torch.randn(2, 32, 8, 8)
    must3r = [torch.randn(2, 16, 4, 4), torch.randn(2, 24, 4, 4)]

    with pytest.raises(ValueError, match="point_map and ray_map"):
        merger(sam, must3r)


def test_strict_feature_merger_accepts_paper_geometry_bundle() -> None:
    merger = FeatureMerger(
        sam_channels=32,
        must3r_channels=(16, 24),
        hidden_channels=32,
        num_heads=4,
        strict_paper=True,
    )
    sam = torch.randn(2, 32, 8, 8)
    bundle = Must3rFeatureBundle(
        levels=(torch.randn(2, 16, 4, 4), torch.randn(2, 24, 4, 4)),
        pe2d=torch.randn(2, 2, 4, 4),
        point_map=torch.randn(2, 3, 4, 4),
        ray_map=torch.randn(2, 3, 4, 4),
    )

    merged = merger(sam, bundle)
    loss = merged.square().mean()
    loss.backward()

    assert merged.shape == sam.shape
    assert all(parameter.grad is not None for parameter in merger.parameters() if parameter.requires_grad)
