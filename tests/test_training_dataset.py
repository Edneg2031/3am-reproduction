from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from three_am.data.io import read_manifest, write_manifest
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.training.dataset import FeatureCacheCompatibilityError, MaskCompatibilityError, ThreeAMTrainingDataset, configured_manifest_path


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
    instances_path = None
    if dataset == "scannetpp":
        instances_path = scene_dir / "instances.json"
        instances_path.write_text(
            '{"schema":"three_am_scannetpp_instances_v1","instances":[{"id":1},{"id":5},{"id":7},{"id":255}]}',
            encoding="utf-8",
        )
    return SceneRecord(dataset=dataset, scene_id="scene_a", split="train", frames=tuple(frames), instances_path=instances_path)


def _dense_instance_mask() -> np.ndarray:
    return np.array([[5, 5, 0, 0], [5, 5, 7, 0], [0, 7, 7, 0], [0, 0, 0, 0]], dtype=np.uint8)


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


def _strict_config(tmp_path: Path, dataset: str, manifest: Path, *, fov_probability: float = 1.0) -> dict[str, object]:
    config = _config(tmp_path, dataset, manifest, fov_probability=fov_probability)
    config["training"] = {"strict_paper": True}
    config["features"] = {"cache_root": "outputs/must3r_features", "require_decoder_memory": False}
    return config


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
    assert [path.name for path in batch.image_paths] == ["000.png", "001.png"]
    assert batch.has_object.tolist() == [True, True]


def test_training_dataset_resizes_sam_inputs_and_masks(tmp_path: Path) -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", [mask, mask])])
    config = _config(tmp_path, "scannetpp", manifest)
    config["model"] = {"must3r_channels": [2], "sam_image_size": 32}

    dataset = ThreeAMTrainingDataset.from_config(config, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.images.shape == (2, 3, 32, 32)
    assert batch.target_masks.shape == (2, 32, 32)


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


def test_configured_manifest_path_accepts_plural_train_mapping(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "datasets": {"scannetpp": {"manifests": {"train": "custom/train_manifest.json"}}},
    }

    assert configured_manifest_path(config, "scannetpp") == tmp_path / "custom" / "train_manifest.json"


def test_training_dataset_uses_absolute_manifest_and_frame_paths(tmp_path: Path) -> None:
    outside_root = tmp_path / "outside_project"
    scene = _scene(outside_root, "scannetpp", [np.ones((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)])
    manifest = outside_root / "absolute_manifest.json"
    write_manifest(manifest, [scene])
    config = {
        "project_root": str(tmp_path / "project"),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "datasets": {"scannetpp": {"manifests": {"train": str(manifest)}, "fov_sampling_probability": 0.0}},
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {"cache_root": str(outside_root / "features")},
        "model": {"must3r_channels": [2]},
    }

    dataset = ThreeAMTrainingDataset.from_config(config, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.scene_id == "scene_a"
    assert batch.images.shape == (2, 3, 4, 4)


def test_manifest_without_feature_paths_reads_as_empty_tuple(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"three_am_manifest_v1","scenes":[{"dataset":"scannetpp","scene_id":"scene_a","split":"train",'
        '"frames":[{"frame_id":"000","image_path":"/abs/image.png","mask_path":"/abs/mask.png"}]}]}',
        encoding="utf-8",
    )

    scenes = read_manifest(manifest)

    assert scenes[0].frames[0].must3r_feature_paths == ()


def test_training_dataset_uses_manifest_feature_paths(tmp_path: Path) -> None:
    scene_dir = tmp_path / "absolute_scene"
    feature_dir = tmp_path / "absolute_features"
    frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        feature_path = feature_dir / f"{frame_id}_must3r_level0.pt"
        _write_image(image_path)
        _write_mask(mask_path, np.ones((4, 4), dtype=np.uint8))
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((2, 2, 2), float(index)), feature_path)
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                mask_path=mask_path,
                must3r_feature_paths=(feature_path,),
            )
        )
    manifest = tmp_path / "manifest_with_features.json"
    write_manifest(manifest, [SceneRecord("scannetpp", "scene_a", "train", tuple(frames))])
    config = {
        "project_root": str(tmp_path / "project"),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "datasets": {"scannetpp": {"manifest": str(manifest), "fov_sampling_probability": 0.0}},
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {"cache_root": str(tmp_path / "unused_cache")},
        "model": {"must3r_channels": [2]},
    }

    dataset = ThreeAMTrainingDataset.from_config(config, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.must3r_features is not None
    assert batch.must3r_features[0].shape == (2, 2, 2, 2)


def test_training_dataset_reports_feature_channel_mismatch(tmp_path: Path) -> None:
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    for frame_id in ("000", "001"):
        torch.save(torch.ones(3, 2, 2), scene_cache / f"{frame_id}_level0.pt")
    dataset = _dataset(tmp_path, "scannetpp", [np.ones((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)])

    with pytest.raises(FeatureCacheCompatibilityError, match="model.must3r_channels"):
        dataset.sample()


def test_strict_training_dataset_keeps_reference_as_prompt_frame_zero(tmp_path: Path) -> None:
    masks = [_dense_instance_mask() for _ in range(4)]
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", masks)])
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    np.save(scene_cache / "overlap.npy", np.ones((4, 4), dtype=np.float32))

    dataset = ThreeAMTrainingDataset.from_config(
        _strict_config(tmp_path, "scannetpp", manifest),
        load_feature_cache=False,
        rng=random.Random(0),
    )
    batch = dataset.sample()

    assert batch.sampling_mode == "fov"
    assert batch.prompt.frame_index == 0
    assert batch.reference_frame_id == batch.frame_ids[0]


def test_strict_fov_fallback_does_not_add_visible_low_overlap_frames(tmp_path: Path) -> None:
    masks = [_dense_instance_mask() for _ in range(4)]
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", masks)])
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    np.save(scene_cache / "overlap.npy", np.zeros((4, 4), dtype=np.float32))

    dataset = ThreeAMTrainingDataset.from_config(
        _strict_config(tmp_path, "scannetpp", manifest),
        load_feature_cache=False,
        rng=random.Random(0),
    )
    batch = dataset.sample()

    assert batch.sampling_mode == "fov"
    assert len(batch.frame_ids) == 1
    assert batch.has_object.tolist() == [True]


def test_strict_fov_fallback_can_add_empty_frames(tmp_path: Path) -> None:
    masks = [_dense_instance_mask()] + [np.zeros((4, 4), dtype=np.uint8) for _ in range(3)]
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", masks)])
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    np.save(scene_cache / "overlap.npy", np.zeros((4, 4), dtype=np.float32))

    dataset = ThreeAMTrainingDataset.from_config(
        _strict_config(tmp_path, "scannetpp", manifest),
        load_feature_cache=False,
        rng=random.Random(1),
    )
    batch = dataset.sample()

    assert len(batch.frame_ids) == 2
    assert batch.has_object.tolist() == [True, False]


def test_strict_training_dataset_applies_mask_ignore_values(tmp_path: Path) -> None:
    mask = np.full((4, 4), 255, dtype=np.uint8)
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", [mask, mask])])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)
    config["datasets"]["scannetpp"]["mask_ignore_values"] = [255]  # type: ignore[index]

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.target_masks.sum().item() == 0
    assert batch.has_object.tolist() == [False, False]


def test_strict_scannetpp_rejects_full_frame_singleton_masks(tmp_path: Path) -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", [mask, mask])])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))

    with pytest.raises(MaskCompatibilityError, match="valid-region/full-frame mask"):
        dataset.sample()


def test_strict_scannetpp_requires_instances_path(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data" / "processed" / "scannetpp" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, _dense_instance_mask())
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [SceneRecord("scannetpp", "scene_a", "train", tuple(frames))])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))

    with pytest.raises(MaskCompatibilityError, match="instances_path"):
        dataset.sample()


def test_strict_training_dataset_rejects_cache_without_decoder_memory(tmp_path: Path) -> None:
    masks = [_dense_instance_mask() for _ in range(2)]
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [_scene(tmp_path, "scannetpp", masks)])
    scene_cache = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    scene_cache.mkdir(parents=True)
    (scene_cache / "metadata.json").write_text(
        '{"decoder_memory": false, "feature_specs": ["decoder_0"], "feature_channels": [2]}',
        encoding="utf-8",
    )
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)
    config["features"]["require_decoder_memory"] = True  # type: ignore[index]

    dataset = ThreeAMTrainingDataset.from_config(config, rng=random.Random(0))

    with pytest.raises(FeatureCacheCompatibilityError, match="decoder_memory=true"):
        dataset.sample()


def test_missing_training_data_error_lists_checked_manifests(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "datasets": {},
        "model": {"must3r_channels": [2]},
    }

    with pytest.raises(ValueError, match="Checked manifests"):
        ThreeAMTrainingDataset.from_config(config)
