#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ManifestFormat = Literal["auto", "normalized", "nerfstudio_3dgs", "shapenet_tracking"]
NERFSTUDIO_TRANSFORM_NAMES = ("transforms.json", "transforms_train.json", "transforms_val.json", "transforms_test.json")


@dataclass(frozen=True)
class CameraRecord:
    pose: tuple[tuple[float, ...], ...]
    intrinsics: tuple[tuple[float, ...], ...]


def _image_root(scene_dir: Path) -> Path | None:
    for candidate in (scene_dir / "frames", scene_dir / "images", scene_dir / "rgb"):
        if candidate.exists():
            return candidate
    return None


def _normalized_frame(scene_dir: Path, image_path: Path) -> FrameRecord:
    frame_id = image_path.stem
    mask_path = scene_dir / "masks" / f"{frame_id}.png"
    depth_path = scene_dir / "depth" / f"{frame_id}.png"
    pose_path = scene_dir / "poses" / f"{frame_id}.txt"
    intrinsics_path = scene_dir / "intrinsics" / f"{frame_id}.txt"
    return FrameRecord(
        frame_id=frame_id,
        image_path=image_path,
        mask_path=mask_path if mask_path.exists() else None,
        depth_path=depth_path if depth_path.exists() else None,
        pose_path=pose_path if pose_path.exists() else None,
        intrinsics_path=intrinsics_path if intrinsics_path.exists() else None,
    )


def _load_label_map(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{path}: mask must be a 2D label map, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def _foreground_ratio(array: np.ndarray) -> float:
    return float((array > 0).mean()) if array.size else 0.0


def _positive_ids(array: np.ndarray) -> np.ndarray:
    values = np.unique(array)
    return values[values > 0]


def _validate_scannetpp_masks(
    scene_dir: Path,
    frames: tuple[FrameRecord, ...],
    *,
    require_instances: bool,
    max_singleton_foreground_ratio: float,
) -> None:
    instances_path = scene_dir / "instances.json"
    if require_instances and not instances_path.exists():
        raise ValueError(
            f"{scene_dir}: ScanNet++ scenes must include instances.json generated from projected 3D instance labels. "
            "Run scripts/preprocess_scannetpp_instance_masks.py on ScanNet++ obj_ids/<scene_id>/*.pth outputs before "
            "building the manifest, or pass --allow-missing-instances for a legacy/debug manifest."
        )
    missing_masks = [frame.frame_id for frame in frames if frame.mask_path is None or not frame.mask_path.exists()]
    if missing_masks:
        preview = ", ".join(missing_masks[:8])
        suffix = "" if len(missing_masks) <= 8 else f", ... ({len(missing_masks)} total)"
        raise ValueError(f"{scene_dir}: ScanNet++ frames are missing per-frame instance label masks: {preview}{suffix}")
    for frame in frames:
        if frame.mask_path is None:
            continue
        label_map = _load_label_map(frame.mask_path)
        positive_ids = _positive_ids(label_map)
        foreground_ratio = _foreground_ratio(label_map)
        if len(positive_ids) <= 1 and foreground_ratio >= max_singleton_foreground_ratio:
            raise ValueError(
                f"{frame.mask_path}: one positive id covers {foreground_ratio:.3f} of the image. This looks like a "
                "full-frame/valid-region mask, not a ScanNet++ projected instance-id label map."
            )


def _nerfstudio_transform_paths(scene_dir: Path) -> list[Path]:
    search_roots = [scene_dir / "nerfstudio", scene_dir]
    paths: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for name in NERFSTUDIO_TRANSFORM_NAMES:
            path = root / name
            if path.exists() and path not in paths:
                paths.append(path)
        for path in sorted(root.glob("transforms*.json")):
            if path not in paths:
                paths.append(path)
    return paths


def _matrix_from_payload(value: object, *, path: Path, field: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path}: {field} must be a 4x4 matrix")
    matrix: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{path}: {field} must be a 4x4 matrix")
        matrix.append(tuple(float(item) for item in row))
    return tuple(matrix)


def _camera_value(frame: dict[str, object], payload: dict[str, object], name: str) -> float | None:
    value = frame.get(name, payload.get(name))
    return None if value is None else float(value)


def _focal_from_angle(frame: dict[str, object], payload: dict[str, object], *, axis: str) -> float | None:
    size_name = "w" if axis == "x" else "h"
    angle_name = f"camera_angle_{axis}"
    size = _camera_value(frame, payload, size_name)
    angle = _camera_value(frame, payload, angle_name)
    if size is None or angle is None:
        return None
    return 0.5 * size / math.tan(0.5 * angle)


def _intrinsics_from_payload(frame: dict[str, object], payload: dict[str, object], *, path: Path) -> tuple[tuple[float, ...], ...]:
    fl_x = _camera_value(frame, payload, "fl_x") or _focal_from_angle(frame, payload, axis="x")
    fl_y = _camera_value(frame, payload, "fl_y") or _focal_from_angle(frame, payload, axis="y") or fl_x
    width = _camera_value(frame, payload, "w")
    height = _camera_value(frame, payload, "h")
    cx = _camera_value(frame, payload, "cx")
    cy = _camera_value(frame, payload, "cy")
    if cx is None and width is not None:
        cx = width * 0.5
    if cy is None and height is not None:
        cy = height * 0.5
    if fl_x is None or fl_y is None or cx is None or cy is None:
        raise ValueError(f"{path}: missing fl_x/fl_y/cx/cy camera parameters")
    return ((fl_x, 0.0, cx), (0.0, fl_y, cy), (0.0, 0.0, 1.0))


def _opencv_camera_to_world(pose: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    axis_flip = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return tuple(tuple(sum(pose[row][index] * axis_flip[index][column] for index in range(4)) for column in range(4)) for row in range(4))


def _camera_from_nerfstudio_frame(frame: dict[str, object], payload: dict[str, object], *, path: Path) -> CameraRecord:
    if "transform_matrix" not in frame:
        raise ValueError(f"{path}: frame is missing transform_matrix")
    pose = _opencv_camera_to_world(_matrix_from_payload(frame["transform_matrix"], path=path, field="transform_matrix"))
    intrinsics = _intrinsics_from_payload(frame, payload, path=path)
    return CameraRecord(pose=pose, intrinsics=intrinsics)


def _load_nerfstudio_cameras(scene_dir: Path) -> dict[str, CameraRecord]:
    cameras: dict[str, CameraRecord] = {}
    for transform_path in _nerfstudio_transform_paths(scene_dir):
        with transform_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frames = payload.get("frames", [])
        if not isinstance(frames, list):
            warnings.warn(f"{transform_path}: frames must be a list; skipping")
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                warnings.warn(f"{transform_path}: found non-object frame; skipping")
                continue
            file_path = frame.get("file_path")
            if not file_path:
                warnings.warn(f"{transform_path}: frame is missing file_path; skipping")
                continue
            frame_id = Path(str(file_path)).stem
            if frame_id in cameras:
                continue
            try:
                cameras[frame_id] = _camera_from_nerfstudio_frame(frame, payload, path=transform_path)
            except ValueError as exc:
                warnings.warn(str(exc))
    return cameras


def _write_matrix(path: Path, matrix: tuple[tuple[float, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(f"{value:.10g}" for value in row))
            handle.write("\n")


def _fill_nerfstudio_cameras(scene_dir: Path, frames: tuple[FrameRecord, ...], sidecar_root: Path) -> tuple[FrameRecord, ...]:
    cameras = _load_nerfstudio_cameras(scene_dir)
    if not cameras:
        warnings.warn(f"{scene_dir}: no usable nerfstudio transforms*.json camera records found")
        return frames
    poses_dir = sidecar_root / scene_dir.name / "poses"
    intrinsics_dir = sidecar_root / scene_dir.name / "intrinsics"
    updated_frames: list[FrameRecord] = []
    for frame in frames:
        if frame.pose_path and frame.intrinsics_path:
            updated_frames.append(frame)
            continue
        camera = cameras.get(frame.frame_id)
        if camera is None:
            warnings.warn(f"{scene_dir}: no nerfstudio camera found for frame {frame.frame_id}")
            updated_frames.append(frame)
            continue
        pose_path = frame.pose_path
        intrinsics_path = frame.intrinsics_path
        if pose_path is None:
            pose_path = poses_dir / f"{frame.frame_id}.txt"
            _write_matrix(pose_path, camera.pose)
        if intrinsics_path is None:
            intrinsics_path = intrinsics_dir / f"{frame.frame_id}.txt"
            _write_matrix(intrinsics_path, camera.intrinsics)
        updated_frames.append(replace(frame, pose_path=pose_path, intrinsics_path=intrinsics_path))
    return tuple(updated_frames)


def _require_complete_cameras(scene_dir: Path, frames: tuple[FrameRecord, ...]) -> None:
    missing = [frame.frame_id for frame in frames if frame.pose_path is None or frame.intrinsics_path is None]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(f"{scene_dir}: missing pose/intrinsics for frames: {preview}{suffix}")


def discover_scene(
    dataset: str,
    scene_dir: Path,
    split: str,
    *,
    manifest_format: ManifestFormat = "auto",
    camera_sidecar_root: Path | None = None,
    require_cameras: bool = False,
    allow_missing_instances: bool = False,
    max_singleton_foreground_ratio: float = 0.98,
) -> SceneRecord | None:
    image_root = _image_root(scene_dir)
    if image_root is None:
        return None
    frames = tuple(
        _normalized_frame(scene_dir, image_path)
        for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    )
    if not frames:
        return None
    use_nerfstudio = (
        manifest_format == "nerfstudio_3dgs"
        or (
            manifest_format == "auto"
            and dataset != "shapenet"
            and bool(_nerfstudio_transform_paths(scene_dir))
        )
    )
    if use_nerfstudio:
        if camera_sidecar_root is None:
            raise ValueError("camera_sidecar_root is required for nerfstudio_3dgs manifest generation")
        frames = _fill_nerfstudio_cameras(scene_dir, frames, camera_sidecar_root)
    if require_cameras:
        _require_complete_cameras(scene_dir, frames)
    if dataset == "shapenet" and manifest_format == "shapenet_tracking":
        missing_masks = [frame.frame_id for frame in frames if frame.mask_path is None or not frame.mask_path.exists()]
        if missing_masks:
            preview = ", ".join(missing_masks[:8])
            suffix = "" if len(missing_masks) <= 8 else f", ... ({len(missing_masks)} total)"
            raise ValueError(f"{scene_dir}: ShapeNet tracking scenes are missing per-frame masks: {preview}{suffix}")
    if dataset == "scannetpp":
        _validate_scannetpp_masks(
            scene_dir,
            frames,
            require_instances=not allow_missing_instances,
            max_singleton_foreground_ratio=max_singleton_foreground_ratio,
        )
    instances_path = scene_dir / "instances.json"
    return SceneRecord(
        dataset=dataset, scene_id=scene_dir.name, split=split, frames=frames, instances_path=instances_path if instances_path.exists() else None
    )


def discover_scenes(
    dataset: str,
    root: Path,
    split: str,
    *,
    manifest_format: ManifestFormat,
    camera_sidecar_root: Path,
    require_cameras: bool = False,
    allow_missing_instances: bool = False,
    max_singleton_foreground_ratio: float = 0.98,
) -> list[SceneRecord]:
    return [
        scene
        for scene_dir in sorted(root.iterdir())
        if scene_dir.is_dir()
        for scene in [
            discover_scene(
                dataset,
                scene_dir,
                split,
                manifest_format=manifest_format,
                camera_sidecar_root=camera_sidecar_root,
                require_cameras=require_cameras,
                allow_missing_instances=allow_missing_instances,
                max_singleton_foreground_ratio=max_singleton_foreground_ratio,
            )
        ]
        if scene
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified 3AM manifest from normalized dataset folders")
    parser.add_argument("--dataset", required=True, choices=["scannetpp", "ase", "mose", "replica", "shapenet"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "normalized", "nerfstudio_3dgs", "shapenet_tracking"],
        help="scene folder format to discover",
    )
    parser.add_argument("--require-cameras", action="store_true", help="fail if any frame is missing pose or intrinsics")
    parser.add_argument(
        "--allow-missing-instances",
        action="store_true",
        help="allow ScanNet++ scenes without instances.json; intended only for legacy/debug manifests",
    )
    parser.add_argument(
        "--max-singleton-foreground-ratio",
        type=float,
        default=0.98,
        help="reject ScanNet++ masks with one positive id covering at least this fraction of the image",
    )
    args = parser.parse_args()
    root = Path(args.root)
    output_path = Path(args.output)
    camera_sidecar_root = output_path.parent / f"{output_path.stem}_cameras"
    scenes = discover_scenes(
        args.dataset,
        root,
        args.split,
        manifest_format=args.format,
        camera_sidecar_root=camera_sidecar_root,
        require_cameras=args.require_cameras,
        allow_missing_instances=args.allow_missing_instances,
        max_singleton_foreground_ratio=args.max_singleton_foreground_ratio,
    )
    if not scenes:
        raise RuntimeError(
            f"No {args.dataset} scenes were found under {root}. "
            "Expected scene directories containing images/ or frames/ plus masks/."
        )
    write_manifest(args.output, scenes)
    frame_count = sum(len(scene.frames) for scene in scenes)
    print(f"Wrote {len(scenes)} scenes ({frame_count} frames) to {args.output}")


if __name__ == "__main__":
    main()
