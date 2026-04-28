from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image

from three_am.data.io import read_manifest
from three_am.data.sampling import SamplingConfig, choose_indices
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.utils.config import ProjectPaths

PromptType = Literal["mask", "point", "box"]


class FeatureCacheError(RuntimeError):
    pass


class RandomLike(Protocol):
    def choice(self, sequence: Any) -> Any: ...

    def randrange(self, *args: int) -> int: ...


@dataclass(frozen=True)
class Prompt:
    type: PromptType
    frame_index: int
    mask: torch.Tensor | None = None
    points: torch.Tensor | None = None
    point_labels: torch.Tensor | None = None
    box: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "Prompt":
        return replace(
            self,
            mask=self.mask.to(device) if self.mask is not None else None,
            points=self.points.to(device) if self.points is not None else None,
            point_labels=self.point_labels.to(device) if self.point_labels is not None else None,
            box=self.box.to(device) if self.box is not None else None,
        )


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    target_masks: torch.Tensor
    prompt: Prompt
    must3r_features: tuple[torch.Tensor, ...] | None
    dataset: str
    scene_id: str
    frame_ids: tuple[str, ...]
    has_object: torch.Tensor

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return replace(
            self,
            images=self.images.to(device),
            target_masks=self.target_masks.to(device),
            prompt=self.prompt.to(device),
            must3r_features=tuple(feature.to(device) for feature in self.must3r_features)
            if self.must3r_features is not None
            else None,
            has_object=self.has_object.to(device),
        )


@dataclass(frozen=True)
class ManifestStatus:
    dataset: str
    manifest: Path
    exists: bool
    scenes: int = 0
    frames: int = 0
    feature_cache_scenes: int = 0


def resolve_project_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ProjectPaths.from_config(config).root / path


def default_manifest_path(config: dict[str, Any], dataset: str) -> Path:
    paths = ProjectPaths.from_config(config)
    return paths.data_processed / f"{dataset}_manifest.json"


def configured_manifest_path(config: dict[str, Any], dataset: str, split: str = "train") -> Path:
    dataset_config = config.get("datasets", {}).get(dataset, {})
    manifest = dataset_config.get("manifest")
    manifests = dataset_config.get("manifests")
    if manifest is None and isinstance(manifests, dict):
        manifest = manifests.get(split)
    if manifest is None and isinstance(manifests, (str, Path)):
        manifest = manifests
    return resolve_project_path(config, manifest) or default_manifest_path(config, dataset)


def configured_feature_cache_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        resolved = resolve_project_path(config, override)
        if resolved is None:
            raise ValueError("feature cache override resolved to None")
        return resolved
    features = config.get("features", {})
    configured = features.get("cache_root", "outputs/must3r_features")
    resolved = resolve_project_path(config, configured)
    if resolved is None:
        raise ValueError("features.cache_root resolved to None")
    return resolved


def load_training_scenes(config: dict[str, Any], datasets: tuple[str, ...] = ("scannetpp", "ase", "mose")) -> list[SceneRecord]:
    scenes: list[SceneRecord] = []
    for dataset in datasets:
        manifest = configured_manifest_path(config, dataset)
        if not manifest.exists():
            continue
        scenes.extend(scene for scene in read_manifest(manifest) if scene.split == "train")
    return scenes


def missing_training_data_message(config: dict[str, Any]) -> str:
    statuses = manifest_statuses(config)
    checked = "\n".join(
        f"  - {status.dataset}: {status.manifest} ({'exists' if status.exists else 'missing'}, train scenes={status.scenes})"
        for status in statuses
    )
    commands = "\n".join(
        [
            "  PYTHONPATH=src python scripts/build_manifest.py --dataset scannetpp --root data/processed/scannetpp --split train --output data/processed/scannetpp_manifest.json",
            "  PYTHONPATH=src python scripts/build_manifest.py --dataset ase --root data/processed/ase --split train --output data/processed/ase_manifest.json",
            "  PYTHONPATH=src python scripts/build_manifest.py --dataset mose --root data/processed/mose --split train --output data/processed/mose_manifest.json",
        ]
    )
    return (
        "No training scenes found.\n"
        "Checked manifests:\n"
        f"{checked}\n"
        "Point datasets.<name>.manifest or datasets.<name>.manifests.train at an existing manifest. "
        "The frame paths inside that manifest may be absolute paths and do not need to live under data/processed.\n"
        "If you still need to generate manifests from normalized folders, typical commands are:\n"
        f"{commands}"
    )


def manifest_statuses(
    config: dict[str, Any],
    feature_cache_root: str | Path | None = None,
    datasets: tuple[str, ...] = ("scannetpp", "ase", "mose"),
) -> list[ManifestStatus]:
    cache_root = configured_feature_cache_root(config, feature_cache_root)
    statuses: list[ManifestStatus] = []
    for dataset in datasets:
        manifest = configured_manifest_path(config, dataset)
        if manifest.exists():
            scenes = read_manifest(manifest)
            cache_dir = cache_root / dataset
            statuses.append(
                ManifestStatus(
                    dataset=dataset,
                    manifest=manifest,
                    exists=True,
                    scenes=len(scenes),
                    frames=sum(len(scene.frames) for scene in scenes),
                    feature_cache_scenes=sum(1 for scene in scenes if (cache_dir / scene.scene_id).exists()),
                )
            )
        else:
            statuses.append(ManifestStatus(dataset=dataset, manifest=manifest, exists=False))
    return statuses


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask_array(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.int64)
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != shape:
        with Image.open(path) as image:
            resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
            array = np.asarray(resized)
            if array.ndim == 3:
                array = array[..., 0]
    return array.astype(np.int64, copy=False)


def _mask_is_binary(mask: np.ndarray) -> bool:
    values = np.unique(mask)
    return len(values) <= 2 and set(int(value) for value in values).issubset({0, 1, 255})


def _candidate_instance_ids(mask: np.ndarray) -> list[int]:
    if _mask_is_binary(mask):
        return [1] if (mask > 0).any() else []
    return [int(value) for value in np.unique(mask) if int(value) != 0]


def _select_reference(mask_arrays: list[np.ndarray], rng: RandomLike) -> tuple[int, int | None, bool]:
    candidates: list[tuple[int, int]] = []
    binary_reference: int | None = None
    for index, mask in enumerate(mask_arrays):
        ids = _candidate_instance_ids(mask)
        if not ids:
            continue
        if _mask_is_binary(mask):
            if binary_reference is None:
                binary_reference = index
            continue
        candidates.extend((index, instance_id) for instance_id in ids)
    if candidates:
        frame_index, instance_id = rng.choice(candidates)
        return frame_index, instance_id, False
    if binary_reference is not None:
        return binary_reference, 1, True
    return 0, None, True


def _target_mask(mask: np.ndarray, instance_id: int | None, binary_mode: bool) -> torch.Tensor:
    if instance_id is None:
        target = np.zeros(mask.shape, dtype=np.float32)
    elif binary_mode:
        target = (mask > 0).astype(np.float32)
    else:
        target = (mask == instance_id).astype(np.float32)
    return torch.from_numpy(target)


def _bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return torch.zeros(4, dtype=torch.float32)
    y_min = foreground[:, 0].min().float()
    x_min = foreground[:, 1].min().float()
    y_max = foreground[:, 0].max().float()
    x_max = foreground[:, 1].max().float()
    return torch.stack([x_min, y_min, x_max, y_max])


def _point_prompt_from_mask(mask: torch.Tensor, rng: RandomLike) -> tuple[torch.Tensor, torch.Tensor]:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    background = torch.nonzero(mask <= 0.5, as_tuple=False)
    points: list[torch.Tensor] = []
    labels: list[int] = []
    if foreground.numel() > 0:
        yx = foreground[rng.randrange(foreground.shape[0])]
        points.append(torch.tensor([float(yx[1]), float(yx[0])], dtype=torch.float32))
        labels.append(1)
    if background.numel() > 0:
        yx = background[rng.randrange(background.shape[0])]
        points.append(torch.tensor([float(yx[1]), float(yx[0])], dtype=torch.float32))
        labels.append(0)
    if not points:
        points.append(torch.zeros(2, dtype=torch.float32))
        labels.append(0)
    return torch.stack(points), torch.tensor(labels, dtype=torch.int64)


def build_prompt(dataset: str, reference_mask: torch.Tensor, frame_index: int, rng: RandomLike) -> Prompt:
    if dataset in {"scannetpp", "ase"}:
        return Prompt(type="mask", frame_index=frame_index, mask=reference_mask)
    prompt_type = rng.choice(("mask", "point", "box"))
    if prompt_type == "mask":
        return Prompt(type="mask", frame_index=frame_index, mask=reference_mask)
    if prompt_type == "point":
        points, labels = _point_prompt_from_mask(reference_mask, rng)
        return Prompt(type="point", frame_index=frame_index, points=points, point_labels=labels)
    return Prompt(type="box", frame_index=frame_index, box=_bbox_from_mask(reference_mask))


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise FeatureCacheError(f"{path} did not contain a tensor")
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise FeatureCacheError(f"{path} must contain a CHW tensor, got shape {tuple(tensor.shape)}")
    return tensor.float()


class Must3rFeatureCache:
    def __init__(self, root: Path, num_levels: int) -> None:
        self.root = root
        self.num_levels = num_levels

    def scene_dir(self, scene: SceneRecord) -> Path:
        return self.root / scene.dataset / scene.scene_id

    def overlap_matrix(self, scene: SceneRecord) -> np.ndarray | None:
        path = self.scene_dir(scene) / "overlap.npy"
        if not path.exists():
            return None
        return np.load(path)

    def load(self, scene: SceneRecord, frames: list[FrameRecord]) -> tuple[torch.Tensor, ...]:
        scene_dir = self.scene_dir(scene)
        levels: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        missing: list[Path] = []
        for frame in frames:
            for level in range(self.num_levels):
                path = scene_dir / f"{frame.frame_id}_level{level}.pt"
                if not path.exists():
                    missing.append(path)
                    continue
                levels[level].append(_load_tensor(path))
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} total)"
            raise FeatureCacheError(f"Missing MUSt3R feature cache files: {preview}{suffix}")
        return tuple(torch.stack(level_tensors, dim=0) for level_tensors in levels)


class ThreeAMTrainingDataset:
    def __init__(
        self,
        scenes: list[SceneRecord],
        config: dict[str, Any],
        *,
        feature_cache_root: Path,
        rng: RandomLike | None = None,
    ) -> None:
        if not scenes:
            raise ValueError(missing_training_data_message(config))
        self.scenes = scenes
        self.config = config
        self.rng = rng or random
        model_config = config.get("model", {})
        must3r_channels = tuple(model_config.get("must3r_channels", (256, 512, 768)))
        self.feature_cache = Must3rFeatureCache(feature_cache_root, num_levels=len(must3r_channels))

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        feature_cache_root: str | Path | None = None,
        rng: RandomLike | None = None,
    ) -> "ThreeAMTrainingDataset":
        return cls(
            load_training_scenes(config),
            config,
            feature_cache_root=configured_feature_cache_root(config, feature_cache_root),
            rng=rng,
        )

    def sample(self) -> TrainingBatch:
        scene = self.rng.choice(self.scenes)
        frames = list(scene.frames)
        sampling_config = self._sampling_config(scene)
        overlap = self.feature_cache.overlap_matrix(scene) if sampling_config.fov_sampling_probability > 0 else None
        selected_indices = choose_indices(len(frames), sampling_config, overlap)
        selected_frames = [frames[index] for index in selected_indices]
        images = [load_image_tensor(frame.image_path) for frame in selected_frames]
        image_shape = tuple(images[0].shape[-2:])
        if any(tuple(image.shape[-2:]) != image_shape for image in images):
            raise ValueError(f"Scene {scene.scene_id} produced variable image sizes in one training sample")
        mask_arrays = [load_mask_array(frame.mask_path, image_shape) for frame in selected_frames]
        reference_index, instance_id, binary_mode = _select_reference(mask_arrays, self.rng)
        target_masks = torch.stack([_target_mask(mask, instance_id, binary_mode) for mask in mask_arrays], dim=0)
        prompt = build_prompt(scene.dataset, target_masks[reference_index], reference_index, self.rng)
        has_object = target_masks.flatten(1).any(dim=1)
        try:
            must3r_features = self.feature_cache.load(scene, selected_frames)
        except FeatureCacheError:
            must3r_features = None
        return TrainingBatch(
            images=torch.stack(images, dim=0),
            target_masks=target_masks,
            prompt=prompt,
            must3r_features=must3r_features,
            dataset=scene.dataset,
            scene_id=scene.scene_id,
            frame_ids=tuple(frame.frame_id for frame in selected_frames),
            has_object=has_object,
        )

    def _sampling_config(self, scene: SceneRecord) -> SamplingConfig:
        global_sampling = self.config.get("sampling", {})
        dataset_config = self.config.get("datasets", {}).get(scene.dataset, {})
        return SamplingConfig(
            fov_sampling_probability=float(dataset_config.get("fov_sampling_probability", 0.0)),
            fov_threshold=float(global_sampling.get("fov_threshold", 0.25)),
            sequence_length=int(global_sampling.get("sequence_length", self.config.get("training", {}).get("memory_frames", 8))),
        )
