#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from three_am.data.io import read_manifest, write_manifest
from three_am.data.schema import SceneRecord
from three_am.training.dataset import resolve_project_path
from three_am.utils.config import load_yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_manifest import discover_scenes  # noqa: E402
from precompute_must3r_features import (  # noqa: E402
    PrecomputeOptions,
    parse_amp,
    parse_feature_layers,
    run_precompute,
)
from preprocess_scannetpp_instance_masks import preprocess_scannetpp  # noqa: E402


@dataclass(frozen=True)
class PrepareOptions:
    obj_id_root: Path
    data_root: Path | None
    output_root: Path
    manifest_output: Path
    split: str
    scene_list: Path | None
    image_subdir: str
    copy_images: bool
    min_visible_pixels: int
    max_foreground_ratio: float
    require_cameras: bool
    precompute_must3r: bool
    config: Path
    feature_output_dir: Path
    device: str
    weights: Path | None
    must3r_repo: Path | None
    image_size: int
    amp: str | bool
    max_bs: int
    decode_batch_size: int
    memory_window: int | None
    full_scene_memory: bool
    cache_dtype: str
    feature_layers: tuple[str | int, ...]
    limit_scenes: int | None
    dry_run_precompute: bool


def _load_label_map(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D instance-id label map, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def _positive_ids(mask: np.ndarray) -> np.ndarray:
    values = np.unique(mask)
    return values[values > 0]


def summarize_scannetpp_manifest(manifest: str | Path, *, max_foreground_ratio: float = 0.98) -> dict[str, Any]:
    scenes = read_manifest(manifest)
    frames = 0
    empty_frames = 0
    singleton_full_frames: list[dict[str, Any]] = []
    positive_instance_ids: set[int] = set()
    per_scene_instances: dict[str, int] = {}
    for scene in scenes:
        if scene.dataset != "scannetpp":
            continue
        if scene.instances_path is None or not scene.instances_path.exists():
            raise ValueError(f"ScanNet++ scene {scene.scene_id} is missing instances.json in the manifest")
        scene_ids: set[int] = set()
        for frame in scene.frames:
            frames += 1
            if frame.mask_path is None or not frame.mask_path.exists():
                raise ValueError(f"ScanNet++ frame {scene.scene_id}/{frame.frame_id} is missing mask_path")
            mask = _load_label_map(frame.mask_path)
            ids = {int(value) for value in _positive_ids(mask)}
            if not ids:
                empty_frames += 1
                continue
            positive_instance_ids.update(ids)
            scene_ids.update(ids)
            foreground_ratio = float((mask > 0).mean()) if mask.size else 0.0
            if len(ids) <= 1 and foreground_ratio >= max_foreground_ratio:
                singleton_full_frames.append(
                    {
                        "scene_id": scene.scene_id,
                        "frame_id": frame.frame_id,
                        "mask_path": str(frame.mask_path),
                        "foreground_ratio": foreground_ratio,
                    }
                )
        per_scene_instances[scene.scene_id] = len(scene_ids)
    if singleton_full_frames:
        preview = singleton_full_frames[:5]
        raise ValueError(
            "ScanNet++ manifest contains full-frame singleton masks, which look like valid-region masks: "
            f"{preview}"
        )
    return {
        "manifest": str(manifest),
        "scenes": len([scene for scene in scenes if scene.dataset == "scannetpp"]),
        "frames": frames,
        "empty_frames": empty_frames,
        "instances": len(positive_instance_ids),
        "per_scene_instances": per_scene_instances,
    }


def _build_scannetpp_manifest(options: PrepareOptions) -> list[SceneRecord]:
    scenes = discover_scenes(
        "scannetpp",
        options.output_root,
        options.split,
        manifest_format="normalized",
        camera_sidecar_root=options.manifest_output.parent / f"{options.manifest_output.stem}_cameras",
        require_cameras=options.require_cameras,
        allow_missing_instances=False,
        max_singleton_foreground_ratio=options.max_foreground_ratio,
    )
    if not scenes:
        raise ValueError(f"No ScanNet++ scenes were discovered under {options.output_root}")
    write_manifest(options.manifest_output, scenes)
    return scenes


def _resolve_optional_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_project_path(config, value)


def _precompute_options(options: PrepareOptions, config: dict[str, Any]) -> PrecomputeOptions:
    feature_output_dir = resolve_project_path(config, options.feature_output_dir)
    if feature_output_dir is None:
        raise ValueError("feature output dir resolved to None")
    return PrecomputeOptions(
        config=config,
        manifest=options.manifest_output,
        output_dir=feature_output_dir,
        device=options.device,
        weights=_resolve_optional_path(config, options.weights)
        or _resolve_optional_path(config, config.get("external", {}).get("must3r_checkpoint")),
        must3r_repo=_resolve_optional_path(config, options.must3r_repo)
        or _resolve_optional_path(config, config.get("external", {}).get("must3r_repo")),
        image_size=options.image_size,
        amp=options.amp,
        max_bs=options.max_bs,
        decode_batch_size=options.decode_batch_size,
        memory_window=options.memory_window,
        full_scene_memory=options.full_scene_memory,
        cache_dtype=options.cache_dtype,
        feature_layers=options.feature_layers,
        write_manifest=options.manifest_output,
        limit_scenes=options.limit_scenes,
        dry_run=options.dry_run_precompute,
    )


def prepare_scannetpp_training_data(options: PrepareOptions) -> dict[str, Any]:
    summaries = preprocess_scannetpp(
        obj_id_root=options.obj_id_root,
        output_root=options.output_root,
        data_root=options.data_root,
        scene_list=options.scene_list,
        image_subdir=options.image_subdir,
        copy_images=options.copy_images,
        min_visible_pixels=options.min_visible_pixels,
        max_foreground_ratio=options.max_foreground_ratio,
    )
    scenes = _build_scannetpp_manifest(options)
    audit = summarize_scannetpp_manifest(options.manifest_output, max_foreground_ratio=options.max_foreground_ratio)
    payload: dict[str, Any] = {
        "preprocess": {"scenes": summaries, "total_scenes": len(summaries)},
        "manifest": {
            "path": str(options.manifest_output),
            "scenes": len(scenes),
            "frames": sum(len(scene.frames) for scene in scenes),
        },
        "audit": audit,
    }
    if options.precompute_must3r:
        config = load_yaml(options.config)
        payload["precompute"] = run_precompute(_precompute_options(options, config))
    print(json.dumps(payload, indent=2))
    return payload


def options_from_args(args: argparse.Namespace) -> PrepareOptions:
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None and (data_root / "data").exists():
        data_root = data_root / "data"
    memory_window = args.memory_window
    if memory_window is not None and memory_window < 2:
        raise ValueError("--memory-window must be >= 2 when set")
    if args.max_bs < 1:
        raise ValueError("--max-bs must be >= 1")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be >= 1")
    return PrepareOptions(
        obj_id_root=Path(args.obj_id_root),
        data_root=data_root,
        output_root=Path(args.output_root),
        manifest_output=Path(args.manifest_output),
        split=args.split,
        scene_list=Path(args.scene_list) if args.scene_list else None,
        image_subdir=args.image_subdir,
        copy_images=bool(args.copy_images),
        min_visible_pixels=int(args.min_visible_pixels),
        max_foreground_ratio=float(args.max_foreground_ratio),
        require_cameras=bool(args.require_cameras),
        precompute_must3r=bool(args.precompute_must3r),
        config=Path(args.config),
        feature_output_dir=Path(args.feature_output_dir),
        device=args.device,
        weights=Path(args.weights) if args.weights else None,
        must3r_repo=Path(args.must3r_repo) if args.must3r_repo else None,
        image_size=int(args.image_size),
        amp=parse_amp(args.amp),
        max_bs=int(args.max_bs),
        decode_batch_size=int(args.decode_batch_size),
        memory_window=memory_window,
        full_scene_memory=bool(args.full_scene_memory),
        cache_dtype=args.cache_dtype,
        feature_layers=parse_feature_layers(args.feature_layers),
        limit_scenes=args.limit_scenes,
        dry_run_precompute=bool(args.dry_run_precompute),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ScanNet++ instance-id label maps, a 3AM manifest, and optional MUSt3R feature cache "
            "for ScanNet++-first 3AM training."
        )
    )
    parser.add_argument("--obj-id-root", required=True, help="Directory containing obj_ids/<scene_id>/*.pth")
    parser.add_argument("--data-root", default="data/raw/scannetpp", help="Official ScanNet++ data root")
    parser.add_argument("--output-root", default="data/processed/scannetpp")
    parser.add_argument("--manifest-output", default="data/processed/scannetpp_manifest.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-list", default=None, help="Optional text file with 1-3 smoke scene ids or a larger split")
    parser.add_argument("--image-subdir", default="dslr/resized_undistorted_images")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    parser.add_argument("--min-visible-pixels", type=int, default=1)
    parser.add_argument("--max-foreground-ratio", type=float, default=0.98)
    parser.add_argument("--require-cameras", action="store_true", help="Require pose/intrinsics in the normalized scene")
    parser.add_argument("--precompute-must3r", action="store_true", help="Run MUSt3R feature precompute after manifest creation")
    parser.add_argument("--config", default="configs/wlh_test.yaml")
    parser.add_argument("--feature-output-dir", default="outputs/must3r_features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--must3r-repo", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--amp", default="bf16", choices=["false", "bf16", "fp16"])
    parser.add_argument("--max-bs", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--memory-window", type=int, default=8)
    parser.add_argument("--full-scene-memory", action="store_true")
    parser.add_argument("--cache-dtype", default="bf16", choices=["fp16", "bf16", "float32"])
    parser.add_argument("--feature-layers", default="encoder,4,7,11")
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--dry-run-precompute", action="store_true")
    prepare_scannetpp_training_data(options_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
