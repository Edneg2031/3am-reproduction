from __future__ import annotations

import json
from pathlib import Path

from .schema import FrameRecord, SceneRecord


def read_manifest(path: str | Path) -> list[SceneRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenes: list[SceneRecord] = []
    for scene in payload.get("scenes", []):
        frames = tuple(
            FrameRecord(
                frame_id=str(frame["frame_id"]),
                image_path=Path(frame["image_path"]),
                mask_path=Path(frame["mask_path"]) if frame.get("mask_path") else None,
                depth_path=Path(frame["depth_path"]) if frame.get("depth_path") else None,
                pose_path=Path(frame["pose_path"]) if frame.get("pose_path") else None,
                intrinsics_path=Path(frame["intrinsics_path"]) if frame.get("intrinsics_path") else None,
            )
            for frame in scene.get("frames", [])
        )
        scenes.append(
            SceneRecord(
                dataset=scene["dataset"],
                scene_id=str(scene["scene_id"]),
                split=str(scene.get("split", "train")),
                frames=frames,
                instances_path=Path(scene["instances_path"]) if scene.get("instances_path") else None,
            )
        )
    return scenes


def write_manifest(path: str | Path, scenes: list[SceneRecord]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "three_am_manifest_v1",
        "scenes": [
            {
                "dataset": scene.dataset,
                "scene_id": scene.scene_id,
                "split": scene.split,
                "instances_path": str(scene.instances_path) if scene.instances_path else None,
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "image_path": str(frame.image_path),
                        "mask_path": str(frame.mask_path) if frame.mask_path else None,
                        "depth_path": str(frame.depth_path) if frame.depth_path else None,
                        "pose_path": str(frame.pose_path) if frame.pose_path else None,
                        "intrinsics_path": str(frame.intrinsics_path) if frame.intrinsics_path else None,
                    }
                    for frame in scene.frames
                ],
            }
            for scene in scenes
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
