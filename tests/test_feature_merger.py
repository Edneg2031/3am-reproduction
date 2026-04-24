from __future__ import annotations

import torch

from three_am.models.feature_merger import FeatureMerger


def test_feature_merger_preserves_sam_shape() -> None:
    merger = FeatureMerger(sam_channels=32, must3r_channels=(16, 24), hidden_channels=32, num_heads=4)
    sam = torch.randn(2, 32, 8, 8)
    must3r = [torch.randn(2, 16, 4, 4), torch.randn(2, 24, 8, 8)]
    merged = merger(sam, must3r)
    assert merged.shape == sam.shape
    assert torch.isfinite(merged).all()
