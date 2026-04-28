from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.training.dataset import ThreeAMTrainingDataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _scene(tmp_path: Path, dataset: str, masks: list[np.ndarray]) -> SceneRecord:
    scene_dir = tmp_path / "data" / "processed" / dataset / "scene_a"
    frames: list[FrameRecord] = []
    for index, mask in enumerate(masks):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, mask)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    return SceneRecord(dataset=dataset, scene_id="scene_a", split="train", frames=tuple(frames))


def _config(tmp_path: Path, dataset: str, manifest: Path, *, fov_probability: float = 0.0) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "datasets": {
            dataset: {
                "manifest": str(manifest.relative_to(tmp_path)),
                "fov_sampling_probability": fov_probability,
            }
        },
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {"cache_root": "outputs/must3r_features"},
        "model": {"must3r_channels": [2]},
    }


def _dataset(tmp_path: Path, dataset: str, masks: list[np.ndarray], *, fov_probability: float = 0.0) -> ThreeAMTrainingDataset:
    manifest = tmp_path / "data" / "processed" / f"{dataset}_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, dataset, masks)])
    return ThreeAMTrainingDataset.from_config(
        _config(tmp_path, dataset, manifest, fov_probability=fov_probability),
        rng=random.Random(0),
    )


def test_training_dataset_builds_binary_mask_prompt(tmp_path: Path) -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    dataset = _dataset(tmp_path, "scannetpp", [mask, mask])

    batch = dataset.sample()

    assert batch.images.shape == (2, 3, 4, 4)
    assert batch.target_masks.shape == (2, 4, 4)
    assert batch.prompt.type == "mask"
    assert batch.prompt.mask is not None
    assert batch.has_object.tolist() == [True, True]


def test_training_dataset_keeps_integer_instance_identity(tmp_path: Path) -> None:
    first = np.zeros((4, 4), dtype=np.uint8)
    first[1:3, 1:3] = 5
    second = np.zeros((4, 4), dtype=np.uint8)
    dataset = _dataset(tmp_path, "ase", [first, second])

    batch = dataset.sample()

    assert batch.target_masks[0].sum().item() == 4
    assert batch.target_masks[1].sum().item() == 0
    assert batch.has_object.tolist() == [True, False]


def test_training_dataset_handles_no_object_frames(tmp_path: Path) -> None:
    empty = np.zeros((4, 4), dtype=np.uint8)
    dataset = _dataset(tmp_path, "scannetpp", [empty, empty])

    batch = dataset.sample()

    assert batch.target_masks.sum().item() == 0
    assert batch.prompt.type == "mask"
    assert batch.prompt.mask is not None
    assert batch.prompt.mask.sum().item() == 0


def test_fov_cache_hit_loads_non_contiguous_features(tmp_path: Path) -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    dataset = _dataset(tmp_path, "scannetpp", [mask, mask, mask, mask], fov_probability=1.0)
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    overlap = np.zeros((4, 4), dtype=np.float32)
    overlap[3, 1] = overlap[1, 3] = 0.9
    np.save(scene_cache / "overlap.npy", overlap)
    for frame_id in ("000", "001", "002", "003"):
        torch.save(torch.full((2, 2, 2), float(int(frame_id))), scene_cache / f"{frame_id}_level0.pt")

    random.seed(0)
    batch = dataset.sample()

    assert batch.frame_ids == ("001", "003")
    assert batch.must3r_features is not None
    assert batch.must3r_features[0].shape == (2, 2, 2, 2)


def test_missing_fov_cache_falls_back_to_continuous_sampling(tmp_path: Path) -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    dataset = _dataset(tmp_path, "scannetpp", [mask, mask, mask, mask], fov_probability=1.0)

    random.seed(0)
    batch = dataset.sample()

    assert batch.frame_ids == ("001", "002")
    assert batch.must3r_features is None


def test_mose_uses_continuous_sampling_even_with_overlap_cache(tmp_path: Path) -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    dataset = _dataset(tmp_path, "mose", [mask, mask, mask, mask], fov_probability=0.0)
    scene_cache = tmp_path / "outputs" / "must3r_features" / "mose" / "scene_a"
    scene_cache.mkdir(parents=True)
    overlap = np.eye(4, dtype=np.float32)
    overlap[0, 3] = overlap[3, 0] = 1.0
    np.save(scene_cache / "overlap.npy", overlap)

    random.seed(0)
    batch = dataset.sample()

    assert batch.frame_ids == ("001", "002")
