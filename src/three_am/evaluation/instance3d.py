from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fragment3D:
    instance_id: int
    points: np.ndarray
    frame_id: str


def backproject_mask(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    valid = z > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_points = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z, np.ones_like(z)], axis=1)
    world_points = (pose @ camera_points.T).T[:, :3]
    return world_points.astype(np.float32)


def point_overlap(a: np.ndarray, b: np.ndarray, voxel_size: float = 0.02) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    voxels_a = {tuple(np.floor(point / voxel_size).astype(int)) for point in a}
    voxels_b = {tuple(np.floor(point / voxel_size).astype(int)) for point in b}
    union = len(voxels_a | voxels_b)
    return 0.0 if union == 0 else len(voxels_a & voxels_b) / union
