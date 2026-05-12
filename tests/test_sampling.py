from __future__ import annotations

import random

import numpy as np

from three_am.data.sampling import SamplingConfig, choose_indices, continuous_indices, fov_aware_indices


def test_continuous_indices_are_ordered() -> None:
    assert continuous_indices(10, 4, start=2) == [2, 3, 4, 5]


def test_fov_aware_indices_use_overlap_threshold() -> None:
    overlaps = np.eye(4)
    overlaps[0, 1] = overlaps[1, 0] = 0.5
    indices = fov_aware_indices(overlaps, SamplingConfig(fov_threshold=0.25, sequence_length=2))
    assert len(indices) <= 2


def test_fov_aware_indices_keep_reference_first_when_requested() -> None:
    overlaps = np.zeros((4, 4), dtype=np.float32)
    overlaps[2, 0] = 0.7
    overlaps[2, 3] = 0.6

    indices = fov_aware_indices(
        overlaps,
        SamplingConfig(fov_threshold=0.25, sequence_length=3),
        reference_index=2,
    )

    assert indices == [2, 0, 3]


def test_fov_aware_indices_keep_no_object_candidates() -> None:
    overlaps = np.zeros((3, 3), dtype=np.float32)

    indices = fov_aware_indices(
        overlaps,
        SamplingConfig(fov_threshold=0.25, sequence_length=2),
        reference_index=0,
        candidate_has_object=[True, True, False],
    )

    assert indices == [0, 2]


def test_choose_indices_preserves_legacy_sorted_order_without_reference() -> None:
    overlaps = np.zeros((4, 4), dtype=np.float32)
    overlaps[3, 1] = 0.9

    indices = choose_indices(
        4,
        SamplingConfig(fov_sampling_probability=1.0, fov_threshold=0.25, sequence_length=2),
        overlaps,
        rng=random.Random(0),
    )

    assert indices == [1, 3]
