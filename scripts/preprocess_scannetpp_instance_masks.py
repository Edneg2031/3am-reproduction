#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
INSTANCE_ARRAY_KEYS = (
    "obj_ids",
    "obj_id",
    "object_ids",
    "object_id",
    "instance_ids",
    "instance_id",
    "instances",
    "labels",
    "mask",
)


def _scene_ids(scene_list: Path | None, obj_id_root: Path) -> list[str]:
    if scene_list is not None:
        return [line.strip() for line in scene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path.name for path in obj_id_root.iterdir() if path.is_dir())


def _payload_to_array(payload: Any, *, path: Path) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().numpy()
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, dict):
        for key in INSTANCE_ARRAY_KEYS:
            if key in payload:
                return _payload_to_array(payload[key], path=path)
        for value in payload.values():
            try:
                array = _payload_to_array(value, path=path)
            except (TypeError, ValueError):
                continue
            if array.ndim in {2, 3}:
                return array
        raise ValueError(f"{path} dictionary does not contain a 2D object-id array")
    if isinstance(payload, (list, tuple)) and len(payload) == 1:
        return _payload_to_array(payload[0], path=path)
    try:
        return np.asarray(payload)
    except Exception as exc:  # pragma: no cover - defensive conversion guard
        raise TypeError(f"{path} could not be converted to a numpy array") from exc


def _as_2d_label_array(array: np.ndarray, *, path: Path) -> np.ndarray:
    if array.ndim == 3 and 1 in {array.shape[0], array.shape[-1]}:
        array = array[0] if array.shape[0] == 1 else array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{path} must contain a 2D instance-id map, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def _load_instance_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".pth", ".pt"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        array = _payload_to_array(payload, path=path)
    elif path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".npz":
        payload = np.load(path)
        key = "obj_ids" if "obj_ids" in payload else sorted(payload.files)[0]
        array = payload[key]
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    return _as_2d_label_array(array, path=path)


def _save_label_png(array: np.ndarray, path: Path) -> None:
    labels = array.astype(np.int64, copy=True)
    labels[labels < 0] = 0
    if labels.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError(f"{path}: instance IDs exceed uint16 PNG range")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels.astype(np.uint16)).save(path)


def _foreground_ratio(array: np.ndarray) -> float:
    return float((array > 0).mean()) if array.size else 0.0


def _positive_ids(array: np.ndarray) -> np.ndarray:
    values = np.unique(array)
    return values[values > 0]


def _looks_like_full_frame_singleton_mask(array: np.ndarray, *, max_foreground_ratio: float) -> bool:
    positive_ids = _positive_ids(array)
    return len(positive_ids) <= 1 and _foreground_ratio(array) >= max_foreground_ratio


def _frame_stem(instance_path: Path) -> str:
    name = instance_path.name
    for suffix in (".pth", ".pt", ".npy", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Path(name).stem


def _source_image_path(data_root: Path, scene_id: str, image_subdir: str, instance_path: Path) -> Path | None:
    image_name = instance_path.name
    for suffix in (".pth", ".pt", ".npy", ".npz"):
        if image_name.endswith(suffix):
            image_name = image_name[: -len(suffix)]
            break
    candidate = data_root / scene_id / image_subdir / image_name
    if candidate.exists():
        return candidate
    stem = Path(image_name).stem
    image_dir = data_root / scene_id / image_subdir
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _link_or_copy_image(source: Path, destination: Path, *, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def _instance_files(scene_obj_dir: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.pth", "*.pt", "*.npy", "*.npz", "*.png"):
        files.extend(scene_obj_dir.glob(suffix))
    return sorted(files)


def preprocess_scene(
    scene_id: str,
    *,
    obj_id_root: Path,
    output_root: Path,
    data_root: Path | None,
    image_subdir: str,
    copy_images: bool,
    min_visible_pixels: int,
    max_foreground_ratio: float,
) -> dict[str, Any]:
    scene_obj_dir = obj_id_root / scene_id
    if not scene_obj_dir.exists():
        raise FileNotFoundError(f"Missing ScanNet++ 2D object-id directory: {scene_obj_dir}")
    output_scene = output_root / scene_id
    mask_dir = output_scene / "masks"
    image_dir = output_scene / "images"
    instances: dict[str, dict[str, Any]] = {}
    written = 0
    skipped_empty = 0
    linked = 0
    for instance_path in _instance_files(scene_obj_dir):
        array = _load_instance_array(instance_path)
        labels = array.copy()
        labels[labels < 0] = 0
        visible_pixels = int((labels > 0).sum())
        if visible_pixels < min_visible_pixels:
            skipped_empty += 1
            continue
        ratio = _foreground_ratio(labels)
        if _looks_like_full_frame_singleton_mask(labels, max_foreground_ratio=max_foreground_ratio):
            raise ValueError(
                f"{instance_path} has one positive id covering {ratio:.3f} of the image. This looks like an "
                "anonymization/valid-region/full-frame mask, not an instance-id map from ScanNet++ 2D semantics."
            )
        frame_id = _frame_stem(instance_path)
        _save_label_png(labels, mask_dir / f"{frame_id}.png")
        for obj_id in np.unique(labels):
            obj_id = int(obj_id)
            if obj_id <= 0:
                continue
            instances.setdefault(str(obj_id), {"id": obj_id, "frames": 0, "pixels": 0})
            instances[str(obj_id)]["frames"] += 1
            instances[str(obj_id)]["pixels"] += int((labels == obj_id).sum())
        if data_root is not None:
            source = _source_image_path(data_root, scene_id, image_subdir, instance_path)
            if source is None:
                raise FileNotFoundError(
                    f"Could not find source image for {instance_path.name} under {data_root / scene_id / image_subdir}"
                )
            destination = image_dir / f"{frame_id}{source.suffix.lower()}"
            _link_or_copy_image(source, destination, copy_images=copy_images)
            linked += 1
        written += 1
    instances_path = output_scene / "instances.json"
    instances_path.parent.mkdir(parents=True, exist_ok=True)
    instances_path.write_text(
        json.dumps(
            {
                "schema": "three_am_scannetpp_instances_v1",
                "scene_id": scene_id,
                "source": str(scene_obj_dir),
                "instances": sorted(instances.values(), key=lambda item: int(item["id"])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "scene_id": scene_id,
        "masks": written,
        "images": linked,
        "skipped_empty": skipped_empty,
        "instances": len(instances),
    }


def preprocess_scannetpp(
    *,
    obj_id_root: Path,
    output_root: Path,
    data_root: Path | None = None,
    scene_list: Path | None = None,
    image_subdir: str = "dslr/resized_undistorted_images",
    copy_images: bool = False,
    min_visible_pixels: int = 1,
    max_foreground_ratio: float = 0.98,
) -> list[dict[str, Any]]:
    obj_id_root = obj_id_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve() if data_root is not None else None
    summaries = []
    for scene_id in _scene_ids(scene_list, obj_id_root):
        summaries.append(
            preprocess_scene(
                scene_id,
                obj_id_root=obj_id_root,
                output_root=output_root,
                data_root=data_root,
                image_subdir=image_subdir,
                copy_images=copy_images,
                min_visible_pixels=min_visible_pixels,
                max_foreground_ratio=max_foreground_ratio,
            )
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ScanNet++ official 2D object-id maps into normalized per-frame instance label maps for 3AM. "
            "Run the ScanNet++ toolbox rasterize + semantic.prep.semantics_2d first with save_objid_gt_2d=true."
        )
    )
    parser.add_argument("--obj-id-root", required=True, help="Directory containing obj_ids/<scene_id>/*.pth from ScanNet++ semantics_2d")
    parser.add_argument("--output-root", required=True, help="Normalized output root, e.g. data/processed/scannetpp")
    parser.add_argument("--data-root", default=None, help="Official ScanNet++ data root containing data/<scene_id> or <scene_id> folders")
    parser.add_argument("--scene-list", default=None, help="Optional text file with scene ids to process")
    parser.add_argument("--image-subdir", default="dslr/resized_undistorted_images", help="Image subdirectory inside each scene")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    parser.add_argument("--min-visible-pixels", type=int, default=1)
    parser.add_argument("--max-foreground-ratio", type=float, default=0.98)
    args = parser.parse_args()
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and (data_root / "data").exists():
        data_root = data_root / "data"
    summaries = preprocess_scannetpp(
        obj_id_root=Path(args.obj_id_root),
        output_root=Path(args.output_root),
        data_root=data_root,
        scene_list=Path(args.scene_list) if args.scene_list else None,
        image_subdir=args.image_subdir,
        copy_images=args.copy_images,
        min_visible_pixels=args.min_visible_pixels,
        max_foreground_ratio=args.max_foreground_ratio,
    )
    print(json.dumps({"scenes": summaries, "total_scenes": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
