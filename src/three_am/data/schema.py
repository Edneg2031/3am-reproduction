from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetName = Literal["scannetpp", "ase", "mose", "replica"]


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    image_path: Path
    mask_path: Path | None = None
    depth_path: Path | None = None
    pose_path: Path | None = None
    intrinsics_path: Path | None = None
    must3r_feature_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SceneRecord:
    dataset: DatasetName
    scene_id: str
    split: str
    frames: tuple[FrameRecord, ...]
    instances_path: Path | None = None

    @property
    def has_geometry(self) -> bool:
        return any(frame.depth_path and frame.pose_path for frame in self.frames)

    def validate(self) -> None:
        if not self.frames:
            raise ValueError(f"Scene {self.scene_id} has no frames")
        for frame in self.frames:
            if not frame.image_path.exists():
                raise FileNotFoundError(frame.image_path)
            for optional_path in (
                frame.mask_path,
                frame.depth_path,
                frame.pose_path,
                frame.intrinsics_path,
                *frame.must3r_feature_paths,
            ):
                if optional_path is not None and not optional_path.exists():
                    raise FileNotFoundError(optional_path)
