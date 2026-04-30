from __future__ import annotations

import random
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image

from three_am.data.io import read_manifest
from three_am.data.sampling import SamplingConfig, choose_indices
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.utils.config import ProjectPaths

PromptType = Literal["mask", "point", "box"]
SamplingMode = Literal["continuous", "fov"]


class FeatureCacheError(RuntimeError):
    pass


class FeatureCacheMissingError(FeatureCacheError):
    pass


class FeatureCacheCompatibilityError(FeatureCacheError):
    pass


class MaskCompatibilityError(RuntimeError):
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
    must3r_features: tuple[torch.Tensor, ...] | Must3rFeatureBundle | None
    dataset: str
    scene_id: str
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    has_object: torch.Tensor
    sampling_mode: SamplingMode = "continuous"
    reference_frame_id: str = ""
    instance_id: int | None = None
    object_visibility: torch.Tensor | None = None
    must3r_geometry: Must3rFeatureBundle | None = None
    target_source: str = "dataset_mask"

    def to(self, device: torch.device | str) -> "TrainingBatch":
        must3r_features: tuple[torch.Tensor, ...] | Must3rFeatureBundle | None
        if isinstance(self.must3r_features, Must3rFeatureBundle):
            must3r_features = self.must3r_features.to(device)
        elif self.must3r_features is not None:
            must3r_features = tuple(feature.to(device) for feature in self.must3r_features)
        else:
            must3r_features = None
        must3r_geometry = self.must3r_geometry.to(device) if self.must3r_geometry is not None else None
        return replace(
            self,
            images=self.images.to(device),
            target_masks=self.target_masks.to(device),
            prompt=self.prompt.to(device),
            must3r_features=must3r_features,
            has_object=self.has_object.to(device),
            object_visibility=self.object_visibility.to(device) if self.object_visibility is not None else None,
            must3r_geometry=must3r_geometry,
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


def _configured_sam_image_size(config: dict[str, Any]) -> int | None:
    value = config.get("model", {}).get("sam_image_size", config.get("training", {}).get("sam_image_size"))
    if value is None:
        return None
    size = int(value)
    return size if size > 0 else None


def load_image_tensor(path: Path, image_size: int | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask_array(
    path: Path | None,
    shape: tuple[int, int] | None = None,
    *,
    ignore_values: tuple[int, ...] = (),
) -> np.ndarray:
    if path is None:
        if shape is None:
            raise ValueError("shape is required when mask path is None")
        return np.zeros(shape, dtype=np.int64)
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if shape is not None and array.shape != shape:
        with Image.open(path) as image:
            resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
            array = np.asarray(resized)
            if array.ndim == 3:
                array = array[..., 0]
    array = array.astype(np.int64, copy=True)
    for value in ignore_values:
        array[array == int(value)] = 0
    return array


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


def _mask_ignore_values(config: dict[str, Any], dataset: str) -> tuple[int, ...]:
    values = config.get("datasets", {}).get(dataset, {}).get("mask_ignore_values", ())
    if values is None:
        return ()
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(value) for value in values)


def _strict_paper_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("strict_paper", False))


def _mask_max_foreground_ratio(config: dict[str, Any], dataset: str) -> float | None:
    dataset_config = config.get("datasets", {}).get(dataset, {})
    value = dataset_config.get("mask_max_foreground_ratio", config.get("training", {}).get("mask_max_foreground_ratio"))
    if value is None:
        return None
    ratio = float(value)
    return ratio if ratio > 0 else None


def _validate_mask_foreground_ratio(mask: torch.Tensor, *, dataset: str, scene_id: str, frame_id: str, max_ratio: float | None) -> None:
    if max_ratio is None:
        return
    ratio = float(mask.float().mean().item())
    if ratio > max_ratio:
        raise MaskCompatibilityError(
            f"Target mask for {dataset}/{scene_id}/{frame_id} covers {ratio:.3f} of the image, "
            f"above mask_max_foreground_ratio={max_ratio:.3f}. Check mask_path, RGB label images, "
            "or datasets.<name>.mask_ignore_values."
        )


def _scannetpp_requires_instance_label_maps(config: dict[str, Any]) -> bool:
    dataset_config = config.get("datasets", {}).get("scannetpp", {})
    training_config = config.get("training", {})
    default = bool(training_config.get("strict_paper", False))
    pseudo = training_config.get("sam2_point_pseudo_masks", {})
    pseudo_mode = str(pseudo.get("mode", "off")).lower() if isinstance(pseudo, dict) else "off"
    if pseudo_mode not in {"off", "false", "none", "0"}:
        default = False
    return bool(dataset_config.get("require_instance_label_maps", default))


def _looks_like_full_frame_singleton_mask(mask: np.ndarray, *, max_ratio: float = 0.98) -> bool:
    if mask.size == 0:
        return False
    positive_ids = [int(value) for value in np.unique(mask) if int(value) > 0]
    if len(positive_ids) > 1:
        return False
    return float((mask > 0).mean()) >= max_ratio


def _validate_scannetpp_source_masks(
    config: dict[str, Any],
    scene: SceneRecord,
    frames: list[FrameRecord],
    mask_arrays: list[np.ndarray],
) -> None:
    if scene.dataset != "scannetpp" or not _scannetpp_requires_instance_label_maps(config):
        return
    if scene.instances_path is None:
        raise MaskCompatibilityError(
            f"ScanNet++ scene {scene.scene_id} has no instances_path in the manifest. "
            "Use scripts/preprocess_scannetpp_instance_masks.py to convert ScanNet++ obj_ids into per-frame "
            "instance-id label maps before training."
        )
    for frame, mask in zip(frames, mask_arrays, strict=True):
        if frame.mask_path is None:
            raise MaskCompatibilityError(f"ScanNet++ frame {scene.scene_id}/{frame.frame_id} has no instance mask path")
        if _looks_like_full_frame_singleton_mask(mask):
            ratio = float((mask > 0).mean())
            raise MaskCompatibilityError(
                f"ScanNet++ mask {frame.mask_path} has one positive id covering {ratio:.3f} of the image. "
                "This looks like a valid-region/full-frame mask, not a projected instance-id label map."
            )


def _load_depth_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image).astype(np.float32)
    if np.nanmax(depth) > 100:
        depth = depth / 1000.0
    return depth


def _load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _backproject_masked_points(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.empty((0, 4), dtype=np.float64)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points_camera = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (pose @ points_camera.T).T


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
        raise FeatureCacheCompatibilityError(f"{path} did not contain a tensor")
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise FeatureCacheCompatibilityError(f"{path} must contain a CHW tensor, got shape {tuple(tensor.shape)}")
    return tensor.float()


class Must3rFeatureCache:
    def __init__(
        self,
        root: Path,
        expected_channels: tuple[int, ...],
        *,
        strict_paper: bool = False,
        require_decoder_memory: bool = False,
        expected_feature_specs: tuple[Any, ...] = ("encoder", 4, 7, 11),
    ) -> None:
        self.root = root
        self.expected_channels = expected_channels
        self.num_levels = len(expected_channels)
        self.strict_paper = strict_paper
        self.require_decoder_memory = require_decoder_memory
        self.expected_feature_specs = tuple(_feature_spec_label(spec) for spec in expected_feature_specs)

    def scene_dir(self, scene: SceneRecord) -> Path:
        return self.root / scene.dataset / scene.scene_id

    def overlap_matrix(self, scene: SceneRecord) -> np.ndarray | None:
        path = self.scene_dir(scene) / "overlap.npy"
        if not path.exists():
            return None
        return np.load(path)

    def load(self, scene: SceneRecord, frames: list[FrameRecord]) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
        metadata = self._metadata(scene)
        self._validate_metadata(scene, metadata)
        if all(frame.must3r_feature_paths for frame in frames):
            levels = self._load_from_manifest_paths(frames)
            return self._bundle_if_strict(scene, frames, levels, metadata)
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
            raise FeatureCacheMissingError(f"Missing MUSt3R feature cache files: {preview}{suffix}")
        stacked = self._stack_and_validate(levels)
        return self._bundle_if_strict(scene, frames, stacked, metadata)

    def expected_paths(self, dataset: str, scene_id: str, frame_ids: tuple[str, ...]) -> list[Path]:
        scene_dir = self.root / dataset / scene_id
        return [scene_dir / f"{frame_id}_level{level}.pt" for frame_id in frame_ids for level in range(self.num_levels)]

    def _load_from_manifest_paths(self, frames: list[FrameRecord]) -> tuple[torch.Tensor, ...]:
        levels: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        missing: list[Path] = []
        for frame in frames:
            if len(frame.must3r_feature_paths) != self.num_levels:
                raise FeatureCacheCompatibilityError(
                    f"Frame {frame.frame_id} has {len(frame.must3r_feature_paths)} MUSt3R feature paths, "
                    f"expected {self.num_levels}"
                )
            for level, path in enumerate(frame.must3r_feature_paths):
                if not path.exists():
                    missing.append(path)
                    continue
                levels[level].append(_load_tensor(path))
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} total)"
            raise FeatureCacheMissingError(f"Missing manifest MUSt3R feature files: {preview}{suffix}")
        return self._stack_and_validate(levels)

    def _stack_and_validate(self, levels: list[list[torch.Tensor]]) -> tuple[torch.Tensor, ...]:
        stacked = tuple(torch.stack(level_tensors, dim=0) for level_tensors in levels)
        actual_channels = tuple(int(level.shape[1]) for level in stacked)
        if actual_channels != self.expected_channels:
            raise FeatureCacheCompatibilityError(
                "MUSt3R feature channel mismatch: "
                f"cache has {list(actual_channels)}, but model.must3r_channels is {list(self.expected_channels)}. "
                f"Set model.must3r_channels to {list(actual_channels)} or regenerate features with matching channels."
            )
        return stacked

    def _metadata(self, scene: SceneRecord) -> dict[str, Any] | None:
        path = self.scene_dir(scene) / "metadata.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None

    def _validate_metadata(self, scene: SceneRecord, metadata: dict[str, Any] | None) -> None:
        if not self.strict_paper:
            return
        if metadata is None:
            raise FeatureCacheCompatibilityError(
                f"Strict paper training requires MUSt3R cache metadata.json for {scene.dataset}/{scene.scene_id}"
            )
        if self.require_decoder_memory and metadata.get("decoder_memory") is not True:
            raise FeatureCacheCompatibilityError(
                f"Strict paper training requires decoder_memory=true MUSt3R cache for {scene.dataset}/{scene.scene_id}; "
                "regenerate with scripts/precompute_must3r_features.py --memory-window K or --full-scene-memory."
            )
        actual_specs = metadata.get("feature_specs", metadata.get("feature_layers"))
        if actual_specs is None:
            raise FeatureCacheCompatibilityError("MUSt3R metadata is missing feature_specs/feature_layers")
        actual_labels = tuple(_feature_spec_label(spec) for spec in actual_specs)
        if actual_labels != self.expected_feature_specs:
            raise FeatureCacheCompatibilityError(
                f"MUSt3R feature layer mismatch: cache has {list(actual_labels)}, "
                f"expected {list(self.expected_feature_specs)}"
            )
        channels = tuple(int(value) for value in metadata.get("feature_channels", ()))
        if channels and channels != self.expected_channels:
            raise FeatureCacheCompatibilityError(
                f"MUSt3R metadata channel mismatch: cache has {list(channels)}, expected {list(self.expected_channels)}"
            )

    def _bundle_if_strict(
        self,
        scene: SceneRecord,
        frames: list[FrameRecord],
        levels: tuple[torch.Tensor, ...],
        metadata: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
        if not self.strict_paper:
            return levels
        geometry = self._load_geometry(scene, frames)
        return Must3rFeatureBundle(levels=levels, metadata=metadata, **geometry)

    def _load_geometry(self, scene: SceneRecord, frames: list[FrameRecord]) -> dict[str, torch.Tensor]:
        scene_dir = self.scene_dir(scene)
        tensors: dict[str, list[torch.Tensor]] = {"pe2d": [], "point_map": [], "ray_map": []}
        missing: list[Path] = []
        for frame in frames:
            for key in tensors:
                path = scene_dir / f"{frame.frame_id}_{key}.pt"
                if not path.exists():
                    missing.append(path)
                    continue
                tensors[key].append(_load_tensor(path))
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} total)"
            raise FeatureCacheCompatibilityError(
                f"Strict paper training requires MUSt3R PE2D/point/ray geometry cache files: {preview}{suffix}"
            )
        return {key: torch.stack(value, dim=0) for key, value in tensors.items()}


def _feature_spec_label(spec: Any) -> str:
    if isinstance(spec, str):
        lowered = spec.lower()
        if lowered == "encoder":
            return "encoder"
        if lowered.startswith("decoder_"):
            return lowered
        try:
            return f"decoder_{int(lowered)}"
        except ValueError:
            return lowered
    return f"decoder_{int(spec)}"


class ThreeAMTrainingDataset:
    def __init__(
        self,
        scenes: list[SceneRecord],
        config: dict[str, Any],
        *,
        feature_cache_root: Path,
        load_feature_cache: bool = True,
        rng: RandomLike | None = None,
    ) -> None:
        if not scenes:
            raise ValueError(missing_training_data_message(config))
        self.scenes = scenes
        self.config = config
        self.load_feature_cache = load_feature_cache
        self.rng = rng or random
        model_config = config.get("model", {})
        must3r_channels = tuple(model_config.get("must3r_channels", (256, 512, 768)))
        self.strict_paper = _strict_paper_enabled(config)
        self.feature_cache = Must3rFeatureCache(
            feature_cache_root,
            expected_channels=tuple(int(c) for c in must3r_channels),
            strict_paper=self.strict_paper,
            require_decoder_memory=bool(config.get("features", {}).get("require_decoder_memory", False)),
            expected_feature_specs=tuple(config.get("features", {}).get("feature_layers", "encoder,4,7,11").split(","))
            if isinstance(config.get("features", {}).get("feature_layers", "encoder,4,7,11"), str)
            else tuple(config.get("features", {}).get("feature_layers", ("encoder", 4, 7, 11))),
        )
        self.sam_image_size = _configured_sam_image_size(config)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        feature_cache_root: str | Path | None = None,
        load_feature_cache: bool = True,
        rng: RandomLike | None = None,
    ) -> "ThreeAMTrainingDataset":
        return cls(
            load_training_scenes(config),
            config,
            feature_cache_root=configured_feature_cache_root(config, feature_cache_root),
            load_feature_cache=load_feature_cache,
            rng=rng,
        )

    def sample(self) -> TrainingBatch:
        if self.strict_paper:
            return self._sample_strict_paper()
        return self._sample_compatible()

    def _sample_compatible(self) -> TrainingBatch:
        scene = self.rng.choice(self.scenes)
        frames = list(scene.frames)
        sampling_config = self._sampling_config(scene)
        overlap = self.feature_cache.overlap_matrix(scene) if sampling_config.fov_sampling_probability > 0 else None
        selected_indices = choose_indices(len(frames), sampling_config, overlap)
        selected_frames = [frames[index] for index in selected_indices]
        images = [load_image_tensor(frame.image_path, self.sam_image_size) for frame in selected_frames]
        image_shape = tuple(images[0].shape[-2:])
        if any(tuple(image.shape[-2:]) != image_shape for image in images):
            raise ValueError(f"Scene {scene.scene_id} produced variable image sizes in one training sample")
        ignore_values = _mask_ignore_values(self.config, scene.dataset)
        mask_arrays = [load_mask_array(frame.mask_path, image_shape, ignore_values=ignore_values) for frame in selected_frames]
        _validate_scannetpp_source_masks(self.config, scene, selected_frames, mask_arrays)
        reference_index, instance_id, binary_mode = _select_reference(mask_arrays, self.rng)
        target_masks = torch.stack([_target_mask(mask, instance_id, binary_mode) for mask in mask_arrays], dim=0)
        max_ratio = _mask_max_foreground_ratio(self.config, scene.dataset)
        for frame, target in zip(selected_frames, target_masks, strict=True):
            _validate_mask_foreground_ratio(
                target,
                dataset=scene.dataset,
                scene_id=scene.scene_id,
                frame_id=frame.frame_id,
                max_ratio=max_ratio,
            )
        prompt = build_prompt(scene.dataset, target_masks[reference_index], reference_index, self.rng)
        has_object = target_masks.flatten(1).any(dim=1)
        must3r_features = None
        if self.load_feature_cache:
            try:
                must3r_features = self.feature_cache.load(scene, selected_frames)
            except FeatureCacheMissingError:
                must3r_features = None
        reference_frame_id = selected_frames[reference_index].frame_id if selected_frames else ""
        return TrainingBatch(
            images=torch.stack(images, dim=0),
            target_masks=target_masks,
            prompt=prompt,
            must3r_features=must3r_features,
            dataset=scene.dataset,
            scene_id=scene.scene_id,
            frame_ids=tuple(frame.frame_id for frame in selected_frames),
            image_paths=tuple(frame.image_path for frame in selected_frames),
            has_object=has_object,
            sampling_mode="fov" if overlap is not None and len(selected_indices) > 1 else "continuous",
            reference_frame_id=reference_frame_id,
            instance_id=instance_id,
            object_visibility=has_object,
            must3r_geometry=must3r_features if isinstance(must3r_features, Must3rFeatureBundle) else None,
        )

    def _sample_strict_paper(self) -> TrainingBatch:
        scene = self.rng.choice(self.scenes)
        frames = list(scene.frames)
        sampling_config = self._sampling_config(scene)
        ignore_values = _mask_ignore_values(self.config, scene.dataset)
        raw_masks = [load_mask_array(frame.mask_path, ignore_values=ignore_values) for frame in frames]
        _validate_scannetpp_source_masks(self.config, scene, frames, raw_masks)
        use_fov = (
            scene.dataset in {"scannetpp", "ase"}
            and sampling_config.fov_sampling_probability > 0
            and self._random_float() < sampling_config.fov_sampling_probability
        )
        if use_fov:
            selected_indices, instance_id, binary_mode = self._select_strict_fov_indices(scene, raw_masks, sampling_config)
            sampling_mode: SamplingMode = "fov"
        else:
            selected_indices = self._select_strict_continuous_indices(len(frames), sampling_config.sequence_length)
            reference_mask = raw_masks[selected_indices[0]]
            instance_id, binary_mode = self._select_instance_from_mask(reference_mask)
            sampling_mode = "continuous"
        selected_frames = [frames[index] for index in selected_indices]
        images = [load_image_tensor(frame.image_path, self.sam_image_size) for frame in selected_frames]
        image_shape = tuple(images[0].shape[-2:])
        mask_arrays = [load_mask_array(frame.mask_path, image_shape, ignore_values=ignore_values) for frame in selected_frames]
        target_masks = torch.stack([_target_mask(mask, instance_id, binary_mode) for mask in mask_arrays], dim=0)
        max_ratio = _mask_max_foreground_ratio(self.config, scene.dataset)
        for frame, target in zip(selected_frames, target_masks, strict=True):
            _validate_mask_foreground_ratio(
                target,
                dataset=scene.dataset,
                scene_id=scene.scene_id,
                frame_id=frame.frame_id,
                max_ratio=max_ratio,
            )
        prompt = build_prompt(scene.dataset, target_masks[0], 0, self.rng)
        has_object = target_masks.flatten(1).any(dim=1)
        must3r_features = None
        if self.load_feature_cache:
            must3r_features = self.feature_cache.load(scene, selected_frames)
        reference_frame_id = selected_frames[0].frame_id if selected_frames else ""
        return TrainingBatch(
            images=torch.stack(images, dim=0),
            target_masks=target_masks,
            prompt=prompt,
            must3r_features=must3r_features,
            dataset=scene.dataset,
            scene_id=scene.scene_id,
            frame_ids=tuple(frame.frame_id for frame in selected_frames),
            image_paths=tuple(frame.image_path for frame in selected_frames),
            has_object=has_object,
            sampling_mode=sampling_mode,
            reference_frame_id=reference_frame_id,
            instance_id=instance_id,
            object_visibility=has_object,
            must3r_geometry=must3r_features if isinstance(must3r_features, Must3rFeatureBundle) else None,
        )

    def _select_strict_continuous_indices(self, num_frames: int, sequence_length: int) -> list[int]:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        max_start = max(0, num_frames - sequence_length)
        start = self.rng.randrange(max_start + 1) if hasattr(self.rng, "randrange") else random.randrange(max_start + 1)
        return list(range(start, min(num_frames, start + sequence_length)))

    def _select_instance_from_mask(self, mask: np.ndarray) -> tuple[int | None, bool]:
        ids = _candidate_instance_ids(mask)
        if not ids:
            return None, True
        if _mask_is_binary(mask):
            return 1, True
        return int(self.rng.choice(ids)), False

    def _select_strict_fov_indices(
        self,
        scene: SceneRecord,
        raw_masks: list[np.ndarray],
        sampling_config: SamplingConfig,
    ) -> tuple[list[int], int | None, bool]:
        frames = list(scene.frames)
        references: list[int] = [index for index, mask in enumerate(raw_masks) if _candidate_instance_ids(mask)]
        if not references:
            indices = self._select_strict_continuous_indices(len(frames), sampling_config.sequence_length)
            return indices, None, True
        reference_index = int(self.rng.choice(references))
        instance_id, binary_mode = self._select_instance_from_mask(raw_masks[reference_index])
        selected = [reference_index]
        eligible: list[tuple[float, int]] = []
        for candidate_index, candidate_mask in enumerate(raw_masks):
            if candidate_index == reference_index:
                continue
            target = _target_mask(candidate_mask, instance_id, binary_mode).numpy() > 0.5
            if not target.any():
                eligible.append((0.0, candidate_index))
                continue
            overlap = self._target_fov_overlap(scene, reference_index, candidate_index, target)
            if overlap >= sampling_config.fov_threshold:
                eligible.append((overlap, candidate_index))
        self._shuffle(eligible)
        eligible.sort(key=lambda item: item[0], reverse=True)
        for _, candidate_index in eligible:
            if len(selected) >= sampling_config.sequence_length:
                break
            selected.append(candidate_index)
        if len(selected) < min(sampling_config.sequence_length, len(frames)):
            fallback = self._select_strict_continuous_indices(len(frames), sampling_config.sequence_length)
            for index in fallback:
                if index not in selected and len(selected) < sampling_config.sequence_length:
                    target = _target_mask(raw_masks[index], instance_id, binary_mode).numpy() > 0.5
                    if target.any():
                        continue
                    selected.append(index)
        return selected, instance_id, binary_mode

    def _target_fov_overlap(
        self,
        scene: SceneRecord,
        reference_index: int,
        candidate_index: int,
        candidate_target_mask: np.ndarray,
    ) -> float:
        frames = list(scene.frames)
        ref = frames[reference_index]
        candidate = frames[candidate_index]
        if not (ref.depth_path and ref.pose_path and ref.intrinsics_path and candidate.depth_path and candidate.pose_path and candidate.intrinsics_path):
            overlap = self.feature_cache.overlap_matrix(scene)
            if overlap is None:
                return 0.0
            return float(overlap[reference_index, candidate_index])
        depth = _load_depth_array(candidate.depth_path)
        if candidate_target_mask.shape != depth.shape:
            resized = Image.fromarray(candidate_target_mask.astype(np.uint8)).resize(
                (depth.shape[1], depth.shape[0]), resample=Image.Resampling.NEAREST
            )
            candidate_target_mask = np.asarray(resized) > 0
        points_world = _backproject_masked_points(
            depth,
            candidate_target_mask,
            _load_matrix(candidate.intrinsics_path),
            _load_matrix(candidate.pose_path),
        )
        if points_world.size == 0:
            return 0.0
        ref_depth = _load_depth_array(ref.depth_path)
        ref_pose_inv = np.linalg.inv(_load_matrix(ref.pose_path))
        ref_k = _load_matrix(ref.intrinsics_path)
        points_ref = (ref_pose_inv @ points_world.T).T[:, :3]
        z = points_ref[:, 2]
        valid_z = z > 1e-6
        projected = (ref_k @ points_ref.T).T
        u = projected[:, 0] / np.maximum(projected[:, 2], 1e-6)
        v = projected[:, 1] / np.maximum(projected[:, 2], 1e-6)
        ref_h, ref_w = ref_depth.shape
        inside = valid_z & (u >= 0) & (u < ref_w) & (v >= 0) & (v < ref_h)
        return float(inside.mean())

    def _random_float(self) -> float:
        method = getattr(self.rng, "random", None)
        return float(method()) if callable(method) else random.random()

    def _shuffle(self, values: list[Any]) -> None:
        method = getattr(self.rng, "shuffle", None)
        if callable(method):
            method(values)
        else:
            random.shuffle(values)

    def _sampling_config(self, scene: SceneRecord) -> SamplingConfig:
        global_sampling = self.config.get("sampling", {})
        dataset_config = self.config.get("datasets", {}).get(scene.dataset, {})
        return SamplingConfig(
            fov_sampling_probability=float(dataset_config.get("fov_sampling_probability", 0.0)),
            fov_threshold=float(global_sampling.get("fov_threshold", 0.25)),
            sequence_length=int(global_sampling.get("sequence_length", self.config.get("training", {}).get("memory_frames", 8))),
        )
