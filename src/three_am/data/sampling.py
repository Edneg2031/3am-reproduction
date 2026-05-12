from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    fov_sampling_probability: float = 0.8
    fov_threshold: float = 0.25
    sequence_length: int = 8


def _random_float(rng: Any | None = None) -> float:
    rng = random if rng is None else rng
    return float(rng.random())


def _random_int(rng: Any | None, low: int, high: int) -> int:
    rng = random if rng is None else rng
    return int(rng.randrange(low, high + 1))


def _shuffle(values: list[Any], rng: Any | None = None) -> None:
    rng = random if rng is None else rng
    rng.shuffle(values)


def continuous_indices(
    num_frames: int,
    sequence_length: int,
    start: int | None = None,
    *,
    rng: Any | None = None,
) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    max_start = max(0, num_frames - sequence_length)
    start_index = _random_int(rng, 0, max_start) if start is None else min(start, max_start)
    return list(range(start_index, min(num_frames, start_index + sequence_length)))


def fov_aware_indices(
    overlap_matrix: np.ndarray,
    config: SamplingConfig,
    *,
    reference_index: int | None = None,
    candidate_has_object: Sequence[bool] | None = None,
    rng: Any | None = None,
) -> list[int]:
    if overlap_matrix.ndim != 2 or overlap_matrix.shape[0] != overlap_matrix.shape[1]:
        raise ValueError("overlap_matrix must be square")
    num_frames = overlap_matrix.shape[0]
    if num_frames <= 0:
        raise ValueError("overlap_matrix must contain at least one frame")
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if reference_index is None:
        reference_index = _random_int(rng, 0, num_frames - 1)
    if not 0 <= reference_index < num_frames:
        raise ValueError("reference_index is out of range")
    if candidate_has_object is not None and len(candidate_has_object) != num_frames:
        raise ValueError("candidate_has_object must have one entry per frame")

    has_object = [True] * num_frames if candidate_has_object is None else [bool(value) for value in candidate_has_object]
    selected = [int(reference_index)]
    eligible: list[tuple[float, int]] = []
    for candidate_index in range(num_frames):
        if candidate_index == reference_index:
            continue
        if not has_object[candidate_index]:
            eligible.append((0.0, candidate_index))
            continue
        overlap = float(overlap_matrix[reference_index, candidate_index])
        if overlap >= config.fov_threshold:
            eligible.append((overlap, candidate_index))

    _shuffle(eligible, rng)
    eligible.sort(key=lambda item: item[0], reverse=True)
    selected.extend(index for _, index in eligible[: max(0, config.sequence_length - 1)])
    return selected


def choose_indices(
    num_frames: int,
    config: SamplingConfig,
    overlap_matrix: np.ndarray | None = None,
    *,
    reference_index: int | None = None,
    candidate_has_object: Sequence[bool] | None = None,
    rng: Any | None = None,
) -> list[int]:
    use_fov = overlap_matrix is not None and _random_float(rng) < config.fov_sampling_probability
    if use_fov:
        indices = fov_aware_indices(
            overlap_matrix,
            config,
            reference_index=reference_index,
            candidate_has_object=candidate_has_object,
            rng=rng,
        )
        if len(indices) >= 2:
            return indices if reference_index is not None else sorted(indices)
    return continuous_indices(num_frames, config.sequence_length, rng=rng)
