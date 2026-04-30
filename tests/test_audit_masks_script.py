from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "audit_masks.py"
    spec = importlib.util.spec_from_file_location("audit_masks_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_masks_reports_full_mask_frames(tmp_path: Path) -> None:
    audit_masks = _load_audit_module()
    scene_dir = tmp_path / "data" / "processed" / "scannetpp" / "scene_a"
    frames: list[FrameRecord] = []
    for index, full in enumerate((True, False)):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
        mask = np.full((4, 4), 255, dtype=np.uint8) if full else np.zeros((4, 4), dtype=np.uint8)
        Image.fromarray(mask).save(mask_path)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [SceneRecord("scannetpp", "scene_a", "train", tuple(frames))])

    payload = audit_masks.audit_masks(manifest, dataset="scannetpp", full_ratio=0.98)

    assert payload["frames_checked"] == 2
    assert payload["full_mask_frames"] == 1
    assert payload["empty_mask_frames"] == 1
    assert payload["foreground_ratio_max"] == 1.0
    assert payload["full_examples"][0]["frame_id"] == "000"
