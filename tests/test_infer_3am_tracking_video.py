from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_infer_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "infer_3am_tracking_video.py"
    spec = importlib.util.spec_from_file_location("infer_3am_tracking_video_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_infer_script_letterboxes_images_and_reference_mask(tmp_path: Path) -> None:
    infer = _load_infer_module()
    frame = tmp_path / "frame.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.full((24, 40, 3), 180, dtype=np.uint8)).save(frame)
    mask_array = np.zeros((24, 40), dtype=np.uint8)
    mask_array[6:18, 12:28] = 255
    Image.fromarray(mask_array).save(mask)

    images, source_shape = infer._load_images([frame], 1024)
    reference_mask = infer._load_reference_mask(mask, source_shape, 1024)

    assert images.shape == (1, 3, 1024, 1024)
    assert reference_mask.shape == (1024, 1024)
    assert images[0, :, :100, :].sum().item() == 0.0
    assert reference_mask.sum().item() > 0
