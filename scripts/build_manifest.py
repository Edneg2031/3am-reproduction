#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_scene(dataset: str, scene_dir: Path, split: str) -> SceneRecord | None:
    image_root = scene_dir / "frames"
    if not image_root.exists():
        image_root = scene_dir / "images"
    if not image_root.exists():
        return None
    frames: list[FrameRecord] = []
    for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        frame_id = image_path.stem
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        depth_path = scene_dir / "depth" / f"{frame_id}.png"
        pose_path = scene_dir / "poses" / f"{frame_id}.txt"
        intrinsics_path = scene_dir / "intrinsics" / f"{frame_id}.txt"
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                mask_path=mask_path if mask_path.exists() else None,
                depth_path=depth_path if depth_path.exists() else None,
                pose_path=pose_path if pose_path.exists() else None,
                intrinsics_path=intrinsics_path if intrinsics_path.exists() else None,
            )
        )
    if not frames:
        return None
    instances_path = scene_dir / "instances.json"
    return SceneRecord(
        dataset=dataset, scene_id=scene_dir.name, split=split, frames=tuple(frames), instances_path=instances_path if instances_path.exists() else None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified 3AM manifest from normalized dataset folders")
    parser.add_argument("--dataset", required=True, choices=["scannetpp", "ase", "mose", "replica"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    scenes = [scene for scene_dir in sorted(root.iterdir()) if scene_dir.is_dir() for scene in [discover_scene(args.dataset, scene_dir, args.split)] if scene]
    write_manifest(args.output, scenes)
    print(f"Wrote {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
