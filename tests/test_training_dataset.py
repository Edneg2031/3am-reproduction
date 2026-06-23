from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from three_am.data.io import read_manifest, write_manifest
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.training.dataset import (
    FeatureCacheCompatibilityError,
    MaskCompatibilityError,
    NoEligibleInstanceError,
    Prompt,
    TrainingBatch,
    ThreeAMTrainingDataset,
    _load_tensor,
    configured_manifest_path,
    configured_training_datasets,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(path)


def _write_image_sized(path: Path, *, height: int, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((height, width, 3), 127, dtype=np.uint8)).save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _content_box(image: torch.Tensor) -> tuple[int, int]:
    content = image.sum(dim=0) > 0
    rows = torch.nonzero(content.any(dim=1), as_tuple=False).flatten()
    cols = torch.nonzero(content.any(dim=0), as_tuple=False).flatten()
    return int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)


def _mask_box_area(mask: torch.Tensor) -> int:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return 0
    height = int(foreground[:, 0].max().item() - foreground[:, 0].min().item() + 1)
    width = int(foreground[:, 1].max().item() - foreground[:, 1].min().item() + 1)
    return height * width


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


def test_training_batch_to_applies_float_dtype_without_moving_geometry_bundle() -> None:
    bundle = Must3rFeatureBundle(
        levels=(torch.ones(2, 3, 4, 4, dtype=torch.float32),),
        pe2d=torch.zeros(2, 2, 4, 4, dtype=torch.float32),
        point_map=torch.zeros(2, 3, 4, 4, dtype=torch.float32),
        ray_map=torch.ones(2, 3, 4, 4, dtype=torch.float32),
        metadata={"decoder_memory": True},
    )
    batch = TrainingBatch(
        images=torch.rand(2, 3, 4, 4, dtype=torch.float32),
        target_masks=torch.rand(2, 4, 4, dtype=torch.float32),
        prompt=Prompt(type="mask", frame_index=0, mask=torch.ones(4, 4, dtype=torch.float32)),
        must3r_features=bundle,
        dataset="shapenet",
        scene_id="scene_a",
        frame_ids=("000000", "000001"),
        image_paths=(),
        has_object=torch.tensor([True, False]),
        object_visibility=torch.tensor([True, False]),
        must3r_geometry=bundle,
    )

    moved = batch.to("cpu", float_dtype=torch.bfloat16)

    assert moved.images.dtype == torch.bfloat16
    assert isinstance(moved.must3r_features, Must3rFeatureBundle)
    assert moved.must3r_features.levels[0].dtype == torch.bfloat16
    assert moved.must3r_features.pe2d is not None and moved.must3r_features.pe2d.dtype == torch.bfloat16
    assert moved.must3r_features.point_map is not None and moved.must3r_features.point_map.dtype == torch.bfloat16
    assert moved.must3r_features.ray_map is not None and moved.must3r_features.ray_map.dtype == torch.bfloat16
    assert moved.target_masks.dtype == torch.float32
    assert moved.prompt.mask is not None and moved.prompt.mask.dtype == torch.float32
    assert moved.must3r_geometry is bundle
    assert moved.must3r_geometry.levels[0].dtype == torch.float32


def test_load_tensor_preserves_cached_bfloat16_dtype(tmp_path: Path) -> None:
    tensor_path = tmp_path / "feature.pt"
    original = torch.randn(2, 4, 4, dtype=torch.bfloat16)
    torch.save(original, tensor_path)

    loaded = _load_tensor(tensor_path)

    assert loaded.dtype == torch.bfloat16
    assert loaded.shape == (2, 4, 4)


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

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.images.shape == (2, 3, 32, 32)
    assert batch.target_masks.shape == (2, 32, 32)


def test_shapenet_training_dataset_uses_dynamic_length_resize_and_incomplete_prompt_masks(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(6):
        frame_id = f"{index:06d}"
        image_path = scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image_sized(image_path, height=48, width=64)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:30, 20:44] = 255
        _write_mask(mask_path, mask)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(frames))])
    config = _config(tmp_path, "shapenet", manifest)
    config["model"]["sam_image_size"] = 1024  # type: ignore[index]
    config["datasets"]["shapenet"].update(  # type: ignore[index]
        {
            "sequence_length_min": 3,
            "sequence_length_max": 5,
            "dynamic_resize": {"enabled": True, "min_size": 32, "max_size": 64, "multiple": 32},
            "prompt_mask_augment": {
                "enabled": True,
                "probability": 1.0,
                "crop_probability": 1.0,
                "erase_probability": 1.0,
                "min_area_ratio": 0.1,
                "max_area_ratio": 0.5,
            },
        }
    )

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert 3 <= len(batch.frame_ids) <= 5
    assert batch.images.shape[-2:] == (1024, 1024)
    assert batch.target_masks.shape[-2:] == (1024, 1024)
    assert batch.prompt.type == "mask"
    assert batch.prompt.mask is not None
    assert batch.prompt.mask.sum().item() > 0
    assert batch.prompt.mask.sum().item() < batch.target_masks[batch.prompt.frame_index].sum().item()
    content_height, content_width = _content_box(batch.images[0])
    assert content_height in {32, 64}
    assert content_width in {32, 64}
    padding = ~(batch.images[0].sum(dim=0) > 0)
    assert torch.count_nonzero(batch.target_masks[batch.prompt.frame_index][padding]).item() == 0


def test_shapenet_sequence_length_bounds_override_global_memory_frames(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(12):
        frame_id = f"{index:06d}"
        image_path = scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image_sized(image_path, height=48, width=64)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[8:28, 18:42] = 255
        _write_mask(mask_path, mask)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(frames))])
    config = _config(tmp_path, "shapenet", manifest)
    config["sampling"]["sequence_length"] = 2  # type: ignore[index]
    config["training"] = {"memory_frames": 4}
    config["model"]["sam_image_size"] = 1024  # type: ignore[index]
    config["datasets"]["shapenet"].update(  # type: ignore[index]
        {
            "sequence_length_min": 8,
            "sequence_length_max": 8,
            "dynamic_resize": {"enabled": True, "min_size": 32, "max_size": 32, "multiple": 32},
        }
    )

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert len(batch.frame_ids) == 8


def test_strict_shapenet_training_dataset_uses_dynamic_sampling_when_feature_paths_exist(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    feature_dir = tmp_path / "features"
    frames: list[FrameRecord] = []
    for index in range(6):
        frame_id = f"{index:06d}"
        image_path = scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        feature_path = feature_dir / f"{frame_id}_level0.pt"
        _write_image(image_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:6, 1:6] = 255
        _write_mask(mask_path, mask)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(2, 2, 2), feature_path)
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=image_path,
                mask_path=mask_path,
                must3r_feature_paths=(feature_path,),
            )
        )
    manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(frames))])
    cache_metadata = tmp_path / "outputs" / "must3r_features" / "shapenet" / "scene_a" / "metadata.json"
    cache_metadata.parent.mkdir(parents=True)
    cache_metadata.write_text(
        '{"decoder_memory": true, "feature_specs": ["encoder"], "feature_channels": [2]}',
        encoding="utf-8",
    )
    config = _strict_config(tmp_path, "shapenet", manifest, fov_probability=0.0)
    config["features"]["feature_layers"] = "encoder"  # type: ignore[index]
    config["features"]["require_decoder_memory"] = False  # type: ignore[index]
    config["model"]["sam_image_size"] = 1024  # type: ignore[index]
    config["datasets"]["shapenet"].update(  # type: ignore[index]
        {
            "sequence_length_min": 3,
            "sequence_length_max": 5,
            "dynamic_resize": {"enabled": True, "min_size": 32, "max_size": 64, "multiple": 32},
            "prompt_mask_augment": {"enabled": True, "probability": 1.0, "crop_probability": 1.0},
        }
    )

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=True, rng=random.Random(0))
    batch = dataset.sample()

    assert 3 <= len(batch.frame_ids) <= 5
    assert batch.images.shape[-2:] == (1024, 1024)
    assert batch.prompt.type == "mask"
    assert batch.prompt.mask is not None
    assert 0 < batch.prompt.mask.sum().item() <= batch.target_masks[batch.prompt.frame_index].sum().item()
    assert batch.must3r_features is not None


def test_shapenet_letterbox_keeps_dynamic_content_scale(tmp_path: Path) -> None:
    scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(3):
        frame_id = f"{index:06d}"
        image_path = scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image_sized(image_path, height=48, width=64)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[12:36, 20:44] = 255
        _write_mask(mask_path, mask)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(frames))])

    config_small = _config(tmp_path, "shapenet", manifest)
    config_small["model"]["sam_image_size"] = 1024  # type: ignore[index]
    config_small["datasets"]["shapenet"].update(  # type: ignore[index]
        {
            "sequence_length_min": 2,
            "sequence_length_max": 2,
            "dynamic_resize": {"enabled": True, "min_size": 32, "max_size": 32, "multiple": 32},
        }
    )
    config_large = _config(tmp_path, "shapenet", manifest)
    config_large["model"]["sam_image_size"] = 1024  # type: ignore[index]
    config_large["datasets"]["shapenet"].update(  # type: ignore[index]
        {
            "sequence_length_min": 2,
            "sequence_length_max": 2,
            "dynamic_resize": {"enabled": True, "min_size": 64, "max_size": 64, "multiple": 32},
        }
    )

    batch_small = ThreeAMTrainingDataset.from_config(config_small, load_feature_cache=False, rng=random.Random(0)).sample()
    batch_large = ThreeAMTrainingDataset.from_config(config_large, load_feature_cache=False, rng=random.Random(0)).sample()

    assert _mask_box_area(batch_large.target_masks[0]) > _mask_box_area(batch_small.target_masks[0])


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

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.scene_id == "scene_a"
    assert batch.images.shape == (2, 3, 4, 4)


def test_configured_training_datasets_prefers_explicit_training_list(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "training": {"datasets": ["shapenet"]},
    }

    assert configured_training_datasets(config) == ("shapenet",)


def test_training_dataset_filters_out_unlisted_manifests(tmp_path: Path) -> None:
    shapenet_scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    shapenet_frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:06d}"
        image_path = shapenet_scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = shapenet_scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, np.pad(np.ones((2, 2), dtype=np.uint8) * 255, 1))
        shapenet_frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))

    shapenet_manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(shapenet_manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(shapenet_frames))])

    scannet_scene_dir = tmp_path / "data" / "processed" / "scannetpp" / "scene_bad"
    scannet_frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        image_path = scannet_scene_dir / "images" / f"{frame_id}.png"
        mask_path = scannet_scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, np.ones((4, 4), dtype=np.uint8))
        scannet_frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))

    scannet_manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(scannet_manifest, [SceneRecord("scannetpp", "scene_bad", "train", tuple(scannet_frames))])

    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "training": {"datasets": ["shapenet"], "strict_paper": True},
        "datasets": {
            "shapenet": {
                "manifest": str(shapenet_manifest.relative_to(tmp_path)),
                "sequence_length_min": 2,
                "sequence_length_max": 2,
                "dynamic_resize": {"enabled": False},
            },
            "scannetpp": {
                "manifest": str(scannet_manifest.relative_to(tmp_path)),
                "require_instance_label_maps": True,
            },
        },
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {"cache_root": "outputs/must3r_features", "online": True},
        "model": {"must3r_channels": [2], "geometry_channels": 4},
    }

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.dataset == "shapenet"
    assert batch.scene_id == "scene_a"
    assert batch.frame_ids == ("000000", "000001")


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


def test_strict_continuous_sampling_filters_noise_and_structural_categories(tmp_path: Path) -> None:
    mask = np.array(
        [
            [1, 2, 2, 0],
            [3, 3, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    scene = _scene(tmp_path, "scannetpp", [mask, mask, mask])
    assert scene.instances_path is not None
    scene.instances_path.write_text(
        '{"schema":"three_am_scannetpp_instances_v1","instances":['
        '{"id":1,"category":"noise"},{"id":2,"category":"wall"},{"id":3,"category":"cup"}]}',
        encoding="utf-8",
    )
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [scene])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)
    config["datasets"]["scannetpp"]["instance_sampling"] = {  # type: ignore[index]
        "enabled": True,
        "min_reference_pixels": 2,
        "min_reference_area_ratio": 0.0,
        "max_reference_area_ratio": 0.5,
        "min_visible_frames": 2,
        "min_visible_pixels_per_frame": 2,
        "excluded_categories": ["wall", "floor"],
    }

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.sampling_mode == "continuous"
    assert batch.prompt.frame_index == 0
    assert batch.instance_id == 3
    assert batch.target_masks.sum().item() == 6


def test_strict_continuous_sampling_filters_large_instances_without_categories(tmp_path: Path) -> None:
    mask = np.array(
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [3, 3, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    scene = _scene(tmp_path, "scannetpp", [mask, mask])
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [scene])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)
    config["datasets"]["scannetpp"]["instance_sampling"] = {  # type: ignore[index]
        "enabled": True,
        "min_reference_pixels": 2,
        "max_reference_area_ratio": 0.25,
        "min_visible_frames": 2,
        "min_visible_pixels_per_frame": 2,
        "excluded_categories": ["wall"],
    }

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))
    batch = dataset.sample()

    assert batch.instance_id == 3


def test_strict_continuous_sampling_reports_when_no_instance_passes_thresholds(tmp_path: Path) -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 5
    scene = _scene(tmp_path, "scannetpp", [mask, mask])
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [scene])
    config = _strict_config(tmp_path, "scannetpp", manifest, fov_probability=0.0)
    config["training"]["sample_resample_attempts"] = 2  # type: ignore[index]
    config["datasets"]["scannetpp"]["instance_sampling"] = {  # type: ignore[index]
        "enabled": True,
        "min_reference_pixels": 4,
        "min_visible_frames": 2,
    }

    dataset = ThreeAMTrainingDataset.from_config(config, load_feature_cache=False, rng=random.Random(0))

    with pytest.raises(NoEligibleInstanceError, match="after 2 attempts"):
        dataset.sample()


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
