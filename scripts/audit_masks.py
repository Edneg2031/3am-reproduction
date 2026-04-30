#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from three_am.data.io import read_manifest
from three_am.training.dataset import load_mask_array


def _foreground_ratio(mask_path: Path | None) -> float:
    if mask_path is None:
        return 0.0
    mask = load_mask_array(mask_path)
    if mask.size == 0:
        return 0.0
    return float((mask > 0).mean())


def _image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def audit_masks(
    manifest: str | Path,
    *,
    dataset: str | None = None,
    sample_limit: int | None = None,
    full_ratio: float = 0.98,
    empty_ratio: float = 0.0,
) -> dict[str, Any]:
    scenes = read_manifest(manifest)
    ratios: list[float] = []
    full_examples: list[dict[str, Any]] = []
    empty_examples: list[dict[str, Any]] = []
    missing = 0
    seen = 0
    for scene in scenes:
        if dataset is not None and scene.dataset != dataset:
            continue
        for frame in scene.frames:
            if sample_limit is not None and seen >= sample_limit:
                break
            seen += 1
            if frame.mask_path is None or not frame.mask_path.exists():
                missing += 1
                ratio = 0.0
            else:
                ratio = _foreground_ratio(frame.mask_path)
            ratios.append(ratio)
            example = {
                "dataset": scene.dataset,
                "scene_id": scene.scene_id,
                "frame_id": frame.frame_id,
                "mask_path": str(frame.mask_path) if frame.mask_path else None,
                "image_path": str(frame.image_path),
                "image_size": _image_size(frame.image_path),
                "foreground_ratio": ratio,
            }
            if ratio >= full_ratio and len(full_examples) < 10:
                full_examples.append(example)
            if ratio <= empty_ratio and len(empty_examples) < 10:
                empty_examples.append(example)
        if sample_limit is not None and seen >= sample_limit:
            break
    ratios_np = np.asarray(ratios, dtype=np.float64)
    payload: dict[str, Any] = {
        "manifest": str(manifest),
        "dataset_filter": dataset,
        "frames_checked": seen,
        "missing_masks": missing,
        "full_ratio_threshold": full_ratio,
        "empty_ratio_threshold": empty_ratio,
        "full_mask_frames": int((ratios_np >= full_ratio).sum()) if ratios_np.size else 0,
        "empty_mask_frames": int((ratios_np <= empty_ratio).sum()) if ratios_np.size else 0,
        "full_mask_fraction": float((ratios_np >= full_ratio).mean()) if ratios_np.size else 0.0,
        "empty_mask_fraction": float((ratios_np <= empty_ratio).mean()) if ratios_np.size else 0.0,
        "foreground_ratio_mean": mean(ratios) if ratios else 0.0,
        "foreground_ratio_min": float(ratios_np.min()) if ratios_np.size else 0.0,
        "foreground_ratio_p50": float(np.quantile(ratios_np, 0.5)) if ratios_np.size else 0.0,
        "foreground_ratio_p95": float(np.quantile(ratios_np, 0.95)) if ratios_np.size else 0.0,
        "foreground_ratio_max": float(ratios_np.max()) if ratios_np.size else 0.0,
        "full_examples": full_examples,
        "empty_examples": empty_examples,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit foreground ratios in 3AM training masks")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--full-ratio", type=float, default=0.98)
    parser.add_argument("--empty-ratio", type=float, default=0.0)
    args = parser.parse_args()
    payload = audit_masks(
        args.manifest,
        dataset=args.dataset,
        sample_limit=args.sample_limit,
        full_ratio=args.full_ratio,
        empty_ratio=args.empty_ratio,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
