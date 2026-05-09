from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
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


def test_infer_script_resolves_reference_frame_names_before_numeric_indices(tmp_path: Path) -> None:
    infer = _load_infer_module()
    frames: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index + 1:06d}.png"
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(path)
        frames.append(path)

    assert infer._resolve_reference_index(frames, reference_index=None, reference_frame="000001") == 0
    assert infer._resolve_reference_index(frames, reference_index=None, reference_frame="000003.png") == 2
    assert infer._resolve_reference_index(frames, reference_index=None, reference_frame="1") == 1


def test_infer_script_saves_masks_in_source_resolution(tmp_path: Path) -> None:
    infer = _load_infer_module()
    transform = infer._letterbox_transform(source_width=40, source_height=24, target_size=64)
    logits = np.full((1, 64, 64), -12.0, dtype=np.float32)
    logits[:, transform.pad_top + 8 : transform.pad_top + 22, 20:44] = 12.0

    paths = infer._save_masks(
        torch.from_numpy(logits),
        tmp_path / "masks",
        threshold=0.5,
        transform=transform,
        source_shape=(24, 40),
    )

    with Image.open(paths[0]) as image:
        mask = np.asarray(image)
        assert image.size == (40, 24)
    assert mask.shape == (24, 40)
    assert mask.sum() > 0


def test_infer_script_full_scene_chunk_ranges_cover_video() -> None:
    infer = _load_infer_module()
    forward = infer._chunk_ranges_forward(reference_index=4, num_frames=10, chunk_size=3)
    backward = infer._chunk_ranges_backward(reference_index=4, chunk_size=3)

    assert forward == [(4, 7), (6, 9), (8, 10)]
    assert backward == [(2, 5), (0, 3)]
    covered = set()
    for start, end in forward + backward:
        covered.update(range(start, end))
    assert covered == set(range(10))
