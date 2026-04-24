from __future__ import annotations

import numpy as np

from three_am.data.sampling import SamplingConfig, continuous_indices, fov_aware_indices


def test_continuous_indices_are_ordered() -> None:
    assert continuous_indices(10, 4, start=2) == [2, 3, 4, 5]


def test_fov_aware_indices_use_overlap_threshold() -> None:
    overlaps = np.eye(4)
    overlaps[0, 1] = overlaps[1, 0] = 0.5
    indices = fov_aware_indices(overlaps, SamplingConfig(fov_threshold=0.25, sequence_length=2))
    assert len(indices) <= 2
