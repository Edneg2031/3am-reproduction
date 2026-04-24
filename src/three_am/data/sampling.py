from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    fov_sampling_probability: float = 0.8
    fov_threshold: float = 0.25
    sequence_length: int = 8


def continuous_indices(num_frames: int, sequence_length: int, start: int | None = None) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    max_start = max(0, num_frames - sequence_length)
    start_index = random.randint(0, max_start) if start is None else min(start, max_start)
    return list(range(start_index, min(num_frames, start_index + sequence_length)))


def fov_aware_indices(overlap_matrix: np.ndarray, config: SamplingConfig) -> list[int]:
    if overlap_matrix.ndim != 2 or overlap_matrix.shape[0] != overlap_matrix.shape[1]:
        raise ValueError("overlap_matrix must be square")
    num_frames = overlap_matrix.shape[0]
    seed = random.randrange(num_frames)
    selected = [seed]
    candidates = set(range(num_frames)) - {seed}
    while candidates and len(selected) < config.sequence_length:
        candidate = max(candidates, key=lambda index: float(overlap_matrix[np.ix_(selected, [index])].max()))
        if float(overlap_matrix[np.ix_(selected, [candidate])].max()) < config.fov_threshold:
            break
        selected.append(candidate)
        candidates.remove(candidate)
    return sorted(selected)


def choose_indices(num_frames: int, config: SamplingConfig, overlap_matrix: np.ndarray | None = None) -> list[int]:
    use_fov = overlap_matrix is not None and random.random() < config.fov_sampling_probability
    if use_fov:
        indices = fov_aware_indices(overlap_matrix, config)
        if len(indices) >= 2:
            return indices
    return continuous_indices(num_frames, config.sequence_length)
