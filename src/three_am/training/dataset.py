from __future__ import annotations

import random
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

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
class LetterboxTransform:
    source_width: int
    source_height: int
    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    crop_left: int = 0
    crop_top: int = 0
    pad_left: int = 0
    pad_top: int = 0

    @property
    def scale_x(self) -> float:
        return float(self.content_width) / float(self.source_width)

    @property
    def scale_y(self) -> float:
        return float(self.content_height) / float(self.source_height)

    @property
    def crop_width(self) -> int:
        return min(self.content_width, self.canvas_width)

    @property
    def crop_height(self) -> int:
        return min(self.content_height, self.canvas_height)

    @property
    def pad_width(self) -> int:
        return max(0, self.canvas_width - self.crop_width)

    @property
    def pad_height(self) -> int:
        return max(0, self.canvas_height - self.crop_height)

    def with_crop(self, *, crop_left: int, crop_top: int) -> "LetterboxTransform":
        return replace(self, crop_left=max(0, int(crop_left)), crop_top=max(0, int(crop_top)))

    def resize_content_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"letterbox image must be CHW, got {tuple(image.shape)}")
        return F.interpolate(
            image[None].float(),
            size=(self.content_height, self.content_width),
            mode="bilinear",
            align_corners=False,
        )[0]

    def resize_content_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"letterbox mask must be HW, got {tuple(mask.shape)}")
        return F.interpolate(mask[None, None].float(), size=(self.content_height, self.content_width), mode="nearest")[0, 0]

    def _place_content(self, content: torch.Tensor) -> torch.Tensor:
        if content.ndim == 2:
            content = content[None, ...]
            squeeze = True
        elif content.ndim == 3:
            squeeze = False
        else:
            raise ValueError(f"letterbox content must be CHW or HW, got {tuple(content.shape)}")
        crop_left = min(self.crop_left, max(0, self.content_width - self.crop_width))
        crop_top = min(self.crop_top, max(0, self.content_height - self.crop_height))
        cropped = content[:, crop_top : crop_top + self.crop_height, crop_left : crop_left + self.crop_width]
        canvas = torch.full(
            (content.shape[0], self.canvas_height, self.canvas_width),
            0.0,
            dtype=cropped.dtype,
            device=cropped.device,
        )
        pad_left = self.pad_left
        pad_top = self.pad_top
        canvas[:, pad_top : pad_top + cropped.shape[-2], pad_left : pad_left + cropped.shape[-1]] = cropped
        return canvas[0] if squeeze else canvas

    def resize_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.scale_x - self.crop_left + self.pad_left,
            y * self.scale_y - self.crop_top + self.pad_top,
        )

    def resize_mask(self, mask: torch.Tensor, *, fill_value: float = 0.0) -> torch.Tensor:
        resized = self.resize_content_mask(mask)
        canvas = torch.full((self.canvas_height, self.canvas_width), float(fill_value), dtype=resized.dtype, device=resized.device)
        canvas[...] = self._place_content(resized)
        return canvas

    def resize_image(self, image: torch.Tensor, *, fill_value: float = 0.0) -> torch.Tensor:
        resized = self.resize_content_image(image)
        canvas = torch.full(
            (image.shape[0], self.canvas_height, self.canvas_width),
            float(fill_value),
            dtype=resized.dtype,
            device=resized.device,
        )
        canvas[...] = self._place_content(resized)
        return canvas


def _letterbox_transform(
    *,
    source_width: int,
    source_height: int,
    target_size: int,
) -> LetterboxTransform:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    scale = min(target_size / float(source_width), target_size / float(source_height))
    content_width = max(1, min(target_size, int(round(source_width * scale))))
    content_height = max(1, min(target_size, int(round(source_height * scale))))
    pad_left = max(0, (target_size - content_width) // 2)
    pad_top = max(0, (target_size - content_height) // 2)
    return LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        content_width=content_width,
        content_height=content_height,
        canvas_width=target_size,
        canvas_height=target_size,
        crop_left=0,
        crop_top=0,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def _resolve_content_size(
    requested_size: int | tuple[int, int] | None,
    *,
    source_shape: tuple[int, int],
    multiple: int = 1,
    preserve_aspect_if_scalar: bool = False,
) -> tuple[int, int]:
    source_height, source_width = int(source_shape[0]), int(source_shape[1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_shape must be positive")
    if requested_size is None:
        width, height = source_width, source_height
    elif isinstance(requested_size, int):
        if preserve_aspect_if_scalar:
            short_side = max(1, min(source_width, source_height))
            scale = float(requested_size) / float(short_side)
            width = max(1, int(round(source_width * scale)))
            height = max(1, int(round(source_height * scale)))
        else:
            width = int(requested_size)
            height = int(requested_size)
    else:
        width = int(requested_size[0])
        height = int(requested_size[1])
    snap = max(1, int(multiple))
    width = max(1, int(round(width / snap)) * snap)
    height = max(1, int(round(height / snap)) * snap)
    return width, height


def build_letterbox_transform(
    *,
    source_shape: tuple[int, int],
    requested_size: int | tuple[int, int] | None,
    canvas_size: int,
    multiple: int = 1,
    preserve_aspect_if_scalar: bool = False,
) -> LetterboxTransform:
    content_width, content_height = _resolve_content_size(
        requested_size,
        source_shape=source_shape,
        multiple=multiple,
        preserve_aspect_if_scalar=preserve_aspect_if_scalar,
    )
    crop_left = max(0, (content_width - canvas_size) // 2)
    crop_top = max(0, (content_height - canvas_size) // 2)
    pad_left = max(0, (canvas_size - content_width) // 2)
    pad_top = max(0, (canvas_size - content_height) // 2)
    return LetterboxTransform(
        source_width=int(source_shape[1]),
        source_height=int(source_shape[0]),
        content_width=content_width,
        content_height=content_height,
        canvas_width=canvas_size,
        canvas_height=canvas_size,
        crop_left=crop_left,
        crop_top=crop_top,
        pad_left=pad_left,
        pad_top=pad_top,
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

    def to(self, device: torch.device | str, *, float_dtype: torch.dtype | None = None) -> "TrainingBatch":
        def move_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if float_dtype is not None and tensor.is_floating_point():
                return tensor.to(device=device, dtype=float_dtype)
            return tensor.to(device=device)

        must3r_features: tuple[torch.Tensor, ...] | Must3rFeatureBundle | None
        if isinstance(self.must3r_features, Must3rFeatureBundle):
            must3r_features = self.must3r_features.to(device, dtype=float_dtype)
        elif self.must3r_features is not None:
            must3r_features = tuple(move_tensor(feature) for feature in self.must3r_features)
        else:
            must3r_features = None
        return replace(
            self,
            images=move_tensor(self.images),
            target_masks=self.target_masks.to(device),
            prompt=self.prompt.to(device),
            must3r_features=must3r_features,
            has_object=self.has_object.to(device),
            object_visibility=self.object_visibility.to(device) if self.object_visibility is not None else None,
            must3r_geometry=self.must3r_geometry,
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


DEFAULT_TRAINING_DATASETS = ("scannetpp", "ase", "mose", "shapenet")


def configured_training_datasets(config: dict[str, Any]) -> tuple[str, ...]:
    training = config.get("training", {})
    value = training.get("datasets")
    if value is None:
        return DEFAULT_TRAINING_DATASETS
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError("training.datasets must contain at least one dataset name when set")
    return tuple(items)


def load_training_scenes(config: dict[str, Any], datasets: tuple[str, ...] = DEFAULT_TRAINING_DATASETS) -> list[SceneRecord]:
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
            "  PYTHONPATH=src python scripts/build_manifest.py --dataset shapenet --root data/processed/shapenet_tracking --split train --format shapenet_tracking --output data/processed/shapenet_manifest.json",
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
    datasets: tuple[str, ...] | None = None,
) -> list[ManifestStatus]:
    cache_root = configured_feature_cache_root(config, feature_cache_root)
    statuses: list[ManifestStatus] = []
    selected_datasets = datasets or configured_training_datasets(config)
    for dataset in selected_datasets:
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


def _random_float(rng: RandomLike) -> float:
    method = getattr(rng, "random", None)
    return float(method()) if callable(method) else random.random()


def _random_int(rng: RandomLike, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"invalid random integer range [{low}, {high}]")
    return int(rng.randrange(low, high + 1)) if hasattr(rng, "randrange") else random.randrange(low, high + 1)


def _random_uniform(rng: RandomLike, low: float, high: float) -> float:
    return low + (high - low) * _random_float(rng)


def _positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set")
    return parsed


def load_image_tensor(path: Path, image_size: int | tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None:
            size = (image_size, image_size) if isinstance(image_size, int) else image_size
            image = image.resize((int(size[0]), int(size[1])), resample=Image.Resampling.BILINEAR)
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
    binary_references: list[int] = []
    for index, mask in enumerate(mask_arrays):
        ids = _candidate_instance_ids(mask)
        if not ids:
            continue
        if _mask_is_binary(mask):
            binary_references.append(index)
            continue
        candidates.extend((index, instance_id) for instance_id in ids)
    if candidates:
        frame_index, instance_id = rng.choice(candidates)
        return frame_index, instance_id, False
    if binary_references:
        return int(rng.choice(binary_references)), 1, True
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


def _crop_window_for_mask(
    mask: torch.Tensor,
    *,
    crop_width: int,
    crop_height: int,
    rng: RandomLike,
    attempts: int = 24,
) -> tuple[int, int]:
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop dimensions must be positive")
    mask = mask > 0.5
    height, width = int(mask.shape[0]), int(mask.shape[1])
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    foreground = torch.nonzero(mask, as_tuple=False)
    if foreground.numel() == 0:
        return max(0, (width - crop_width) // 2), max(0, (height - crop_height) // 2)
    y_min = int(foreground[:, 0].min().item())
    y_max = int(foreground[:, 0].max().item())
    x_min = int(foreground[:, 1].min().item())
    x_max = int(foreground[:, 1].max().item())
    left_min = max(0, x_max - crop_width + 1)
    left_max = min(x_min, width - crop_width)
    top_min = max(0, y_max - crop_height + 1)
    top_max = min(y_min, height - crop_height)
    for _ in range(max(1, attempts)):
        left = _random_int(rng, left_min, left_max) if left_max >= left_min else max(0, min(x_min, width - crop_width))
        top = _random_int(rng, top_min, top_max) if top_max >= top_min else max(0, min(y_min, height - crop_height))
        if bool(mask[top : top + crop_height, left : left + crop_width].any()):
            return left, top
    left = max(0, min(int(round((x_min + x_max + 1) / 2.0 - crop_width / 2.0)), width - crop_width))
    top = max(0, min(int(round((y_min + y_max + 1) / 2.0 - crop_height / 2.0)), height - crop_height))
    return left, top


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


def _prompt_mask_augmentation_config(config: dict[str, Any] | None, dataset: str) -> dict[str, Any]:
    if config is None:
        return {}
    training_value = config.get("training", {}).get("prompt_mask_augment", {})
    dataset_value = config.get("datasets", {}).get(dataset, {}).get("prompt_mask_augment", {})
    merged: dict[str, Any] = {}
    if isinstance(training_value, dict):
        merged.update(training_value)
    if isinstance(dataset_value, dict):
        merged.update(dataset_value)
    if "enabled" not in merged and dataset == "shapenet":
        merged["enabled"] = True
    return merged


def _random_prompt_crop(mask: torch.Tensor, rng: RandomLike, *, min_area_ratio: float, max_area_ratio: float) -> torch.Tensor:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return mask
    y_min = int(foreground[:, 0].min().item())
    y_max = int(foreground[:, 0].max().item())
    x_min = int(foreground[:, 1].min().item())
    x_max = int(foreground[:, 1].max().item())
    box_h = max(1, y_max - y_min + 1)
    box_w = max(1, x_max - x_min + 1)
    area_ratio = _random_uniform(rng, min_area_ratio, max_area_ratio)
    side_ratio = max(0.05, min(1.0, area_ratio ** 0.5))
    crop_h = max(1, min(mask.shape[0], int(round(box_h * _random_uniform(rng, side_ratio, 1.0)))))
    crop_w = max(1, min(mask.shape[1], int(round(box_w * _random_uniform(rng, side_ratio, 1.0)))))
    anchor = foreground[_random_int(rng, 0, foreground.shape[0] - 1)]
    anchor_y = int(anchor[0].item())
    anchor_x = int(anchor[1].item())
    top_min = max(0, anchor_y - crop_h + 1)
    top_max = min(anchor_y, mask.shape[0] - crop_h)
    left_min = max(0, anchor_x - crop_w + 1)
    left_max = min(anchor_x, mask.shape[1] - crop_w)
    top = _random_int(rng, top_min, top_max) if top_max >= top_min else max(0, min(anchor_y, mask.shape[0] - crop_h))
    left = _random_int(rng, left_min, left_max) if left_max >= left_min else max(0, min(anchor_x, mask.shape[1] - crop_w))
    keep = torch.zeros_like(mask, dtype=torch.bool)
    keep[top : top + crop_h, left : left + crop_w] = True
    return (mask > 0.5) & keep


def _erase_prompt_region(mask: torch.Tensor, rng: RandomLike) -> torch.Tensor:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.shape[0] <= 1:
        return mask
    y_min = int(foreground[:, 0].min().item())
    y_max = int(foreground[:, 0].max().item())
    x_min = int(foreground[:, 1].min().item())
    x_max = int(foreground[:, 1].max().item())
    box_h = max(1, y_max - y_min + 1)
    box_w = max(1, x_max - x_min + 1)
    erase_h = max(1, int(round(box_h * _random_uniform(rng, 0.2, 0.55))))
    erase_w = max(1, int(round(box_w * _random_uniform(rng, 0.2, 0.55))))
    anchor = foreground[_random_int(rng, 0, foreground.shape[0] - 1)]
    top = max(0, min(int(anchor[0].item()) - erase_h // 2, mask.shape[0] - erase_h))
    left = max(0, min(int(anchor[1].item()) - erase_w // 2, mask.shape[1] - erase_w))
    candidate = (mask > 0.5).clone()
    candidate[top : top + erase_h, left : left + erase_w] = False
    return candidate if bool(candidate.any()) else mask


def _single_pixel_prompt_mask(mask: torch.Tensor, rng: RandomLike) -> torch.Tensor:
    foreground = torch.nonzero(mask > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return mask
    yx = foreground[_random_int(rng, 0, foreground.shape[0] - 1)]
    candidate = torch.zeros_like(mask, dtype=torch.float32)
    candidate[int(yx[0].item()), int(yx[1].item())] = 1.0
    return candidate


def _augment_prompt_mask(mask: torch.Tensor, rng: RandomLike, config: dict[str, Any]) -> torch.Tensor:
    if not bool(config.get("enabled", False)):
        return mask
    original = (mask > 0.5).float()
    original_count = int(original.sum().item())
    if original_count == 0:
        return original
    if _random_float(rng) > float(config.get("probability", 1.0)):
        return original
    min_area_ratio = max(0.001, min(1.0, float(config.get("min_area_ratio", 0.08))))
    max_area_ratio = max(min_area_ratio, min(1.0, float(config.get("max_area_ratio", 0.65))))
    crop_probability = float(config.get("crop_probability", 0.8))
    erase_probability = float(config.get("erase_probability", 0.35))
    attempts = max(1, int(config.get("attempts", 8)))
    for _ in range(attempts):
        candidate = original
        if _random_float(rng) < crop_probability:
            candidate = _random_prompt_crop(candidate, rng, min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio).float()
        if _random_float(rng) < erase_probability:
            candidate = _erase_prompt_region(candidate, rng).float()
        candidate_count = int(candidate.sum().item())
        if 0 < candidate_count < original_count:
            return candidate
        if original_count == 1 and candidate_count == 1:
            return candidate
    if original_count > 1:
        return _single_pixel_prompt_mask(original, rng)
    return original


def build_prompt(
    dataset: str,
    reference_mask: torch.Tensor,
    frame_index: int,
    rng: RandomLike,
    config: dict[str, Any] | None = None,
) -> Prompt:
    if dataset in {"scannetpp", "ase", "shapenet"}:
        prompt_mask = _augment_prompt_mask(reference_mask, rng, _prompt_mask_augmentation_config(config, dataset))
        return Prompt(type="mask", frame_index=frame_index, mask=prompt_mask)
    prompt_type = rng.choice(("mask", "point", "box"))
    if prompt_type == "mask":
        prompt_mask = _augment_prompt_mask(reference_mask, rng, _prompt_mask_augmentation_config(config, dataset))
        return Prompt(type="mask", frame_index=frame_index, mask=prompt_mask)
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
    if not tensor.is_floating_point():
        raise FeatureCacheCompatibilityError(f"{path} must contain a floating-point tensor, got {tensor.dtype}")
    return tensor.contiguous()


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
        stacked = self.load_levels(scene, frames)
        return self._bundle_if_strict(scene, frames, stacked, metadata)

    def load_levels(self, scene: SceneRecord, frames: list[FrameRecord]) -> tuple[torch.Tensor, ...]:
        if all(frame.must3r_feature_paths for frame in frames):
            return self._load_from_manifest_paths(frames)
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
        return self._stack_and_validate(levels)

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
        if scene.dataset == "shapenet":
            geometry = self._synthetic_geometry(levels[0])
            return Must3rFeatureBundle(levels=levels, metadata=metadata, **geometry)
        geometry = self._load_geometry(scene, frames)
        return Must3rFeatureBundle(levels=levels, metadata=metadata, **geometry)

    def _synthetic_geometry(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = reference.shape
        device = reference.device
        dtype = reference.dtype
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        pe2d = torch.stack([x, y], dim=0).expand(batch_size, -1, -1, -1).contiguous()
        point_map = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)
        point_map[:, 2] = 1.0
        ray_map = torch.cat([pe2d, torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)], dim=1)
        ray_map = F.normalize(ray_map, dim=1)
        return {"pe2d": pe2d, "point_map": point_map, "ray_map": ray_map}

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
            load_training_scenes(config, configured_training_datasets(config)),
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
        if scene.dataset != "shapenet":
            return self._sample_compatible_scene(scene)
        shapenet_scenes = [candidate for candidate in self.scenes if candidate.dataset == "shapenet"]
        attempts = max(1, int(self.config.get("training", {}).get("sample_resample_attempts", 24)))
        last_error: Exception | None = None
        for _ in range(attempts):
            scene = self.rng.choice(shapenet_scenes)
            batch = self._sample_compatible_scene(scene)
            if batch.prompt.mask is not None and bool((batch.prompt.mask > 0.5).any()) and bool(batch.has_object.any()):
                return batch
            last_error = RuntimeError(
                f"ShapeNet sample {scene.scene_id} produced an empty prompt or empty target masks; resampling"
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to sample a valid batch")

    def _sample_compatible_scene(self, scene: SceneRecord, *, allow_missing_feature_cache: bool = True) -> TrainingBatch:
        frames = list(scene.frames)
        sampling_config = self._sampling_config(scene)
        overlap = self.feature_cache.overlap_matrix(scene) if sampling_config.fov_sampling_probability > 0 else None
        if scene.dataset == "shapenet":
            selected_indices = self._select_continuous_indices(len(frames), sampling_config.sequence_length)
        else:
            selected_indices = choose_indices(len(frames), sampling_config, overlap)
        selected_frames = [frames[index] for index in selected_indices]
        image_size = self._image_size(scene, selected_frames)
        ignore_values = _mask_ignore_values(self.config, scene.dataset)
        raw_images = [load_image_tensor(frame.image_path, None) for frame in selected_frames]
        if not raw_images:
            raise RuntimeError(f"Scene {scene.scene_id} produced no images")
        source_shape = tuple(raw_images[0].shape[-2:])
        if any(tuple(image.shape[-2:]) != source_shape for image in raw_images):
            raise ValueError(f"Scene {scene.scene_id} produced variable source image sizes in one training sample")
        canvas_size = int(self.sam_image_size or max(_resolve_content_size(image_size, source_shape=source_shape)))
        letterbox = build_letterbox_transform(
            source_shape=source_shape,
            requested_size=image_size,
            canvas_size=canvas_size,
        )
        content_images = [letterbox.resize_content_image(image) for image in raw_images]
        content_masks = [
            letterbox.resize_content_mask(
                torch.from_numpy(load_mask_array(frame.mask_path, source_shape, ignore_values=ignore_values).astype(np.float32))
            )
            for frame in selected_frames
        ]
        mask_arrays = [mask.numpy().astype(np.int64) for mask in content_masks]
        _validate_scannetpp_source_masks(self.config, scene, selected_frames, mask_arrays)
        reference_index, instance_id, binary_mode = _select_reference(mask_arrays, self.rng)
        target_masks_content = torch.stack([_target_mask(mask, instance_id, binary_mode) for mask in mask_arrays], dim=0)
        if letterbox.content_width > letterbox.canvas_width or letterbox.content_height > letterbox.canvas_height:
            crop_left, crop_top = _crop_window_for_mask(
                target_masks_content[reference_index],
                crop_width=letterbox.crop_width,
                crop_height=letterbox.crop_height,
                rng=self.rng,
                attempts=max(1, int(self.config.get("training", {}).get("sample_resample_attempts", 24))),
            )
            letterbox = letterbox.with_crop(crop_left=crop_left, crop_top=crop_top)
        images = [letterbox._place_content(image) for image in content_images]
        target_masks = torch.stack([letterbox._place_content(mask) for mask in target_masks_content], dim=0)
        max_ratio = _mask_max_foreground_ratio(self.config, scene.dataset)
        for frame, target in zip(selected_frames, target_masks, strict=True):
            _validate_mask_foreground_ratio(
                target,
                dataset=scene.dataset,
                scene_id=scene.scene_id,
                frame_id=frame.frame_id,
                max_ratio=max_ratio,
            )
        prompt = build_prompt(scene.dataset, target_masks[reference_index], reference_index, self.rng, self.config)
        if prompt.mask is not None:
            prompt = replace(prompt, mask=prompt.mask.to(dtype=torch.float32))
        has_object = target_masks.flatten(1).any(dim=1)
        must3r_features = None
        if self.load_feature_cache:
            try:
                must3r_features = self.feature_cache.load(scene, selected_frames)
            except FeatureCacheMissingError:
                if not allow_missing_feature_cache:
                    raise
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
        if scene.dataset != "shapenet":
            return self._sample_strict_paper_scene(scene)
        shapenet_scenes = [candidate for candidate in self.scenes if candidate.dataset == "shapenet"]
        attempts = max(1, int(self.config.get("training", {}).get("sample_resample_attempts", 24)))
        last_error: Exception | None = None
        for _ in range(attempts):
            scene = self.rng.choice(shapenet_scenes)
            batch = self._sample_compatible_scene(scene, allow_missing_feature_cache=False)
            if batch.prompt.mask is not None and bool((batch.prompt.mask > 0.5).any()) and bool(batch.has_object.any()):
                return batch
            last_error = RuntimeError(
                f"ShapeNet sample {scene.scene_id} produced an empty prompt or empty target masks; resampling"
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to sample a valid strict ShapeNet batch")

    def _sample_strict_paper_scene(self, scene: SceneRecord) -> TrainingBatch:
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
        image_size = self._image_size(scene, selected_frames)
        raw_images = [load_image_tensor(frame.image_path, None) for frame in selected_frames]
        if not raw_images:
            raise RuntimeError("Strict paper scene sampling produced no frames")
        source_shape = tuple(raw_images[0].shape[-2:])
        if any(tuple(image.shape[-2:]) != source_shape for image in raw_images):
            raise ValueError(f"Scene {scene.scene_id} produced variable source image sizes in one training sample")
        canvas_size = int(self.sam_image_size or max(_resolve_content_size(image_size, source_shape=source_shape)))
        letterbox = build_letterbox_transform(
            source_shape=source_shape,
            requested_size=image_size,
            canvas_size=canvas_size,
        )
        content_images = [letterbox.resize_content_image(image) for image in raw_images]
        content_masks = [
            letterbox.resize_content_mask(
                torch.from_numpy(load_mask_array(frame.mask_path, source_shape, ignore_values=ignore_values).astype(np.float32))
            )
            for frame in selected_frames
        ]
        mask_arrays = [mask.numpy().astype(np.int64) for mask in content_masks]
        target_masks_content = torch.stack([_target_mask(mask, instance_id, binary_mode) for mask in mask_arrays], dim=0)
        if letterbox.content_width > letterbox.canvas_width or letterbox.content_height > letterbox.canvas_height:
            crop_left, crop_top = _crop_window_for_mask(
                target_masks_content[0],
                crop_width=letterbox.crop_width,
                crop_height=letterbox.crop_height,
                rng=self.rng,
                attempts=max(1, int(self.config.get("training", {}).get("sample_resample_attempts", 24))),
            )
            letterbox = letterbox.with_crop(crop_left=crop_left, crop_top=crop_top)
        images = [letterbox._place_content(image) for image in content_images]
        target_masks = torch.stack([letterbox._place_content(mask) for mask in target_masks_content], dim=0)
        max_ratio = _mask_max_foreground_ratio(self.config, scene.dataset)
        for frame, target in zip(selected_frames, target_masks, strict=True):
            _validate_mask_foreground_ratio(
                target,
                dataset=scene.dataset,
                scene_id=scene.scene_id,
                frame_id=frame.frame_id,
                max_ratio=max_ratio,
            )
        prompt = build_prompt(scene.dataset, target_masks[0], 0, self.rng, self.config)
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

    def _select_continuous_indices(self, num_frames: int, sequence_length: int) -> list[int]:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        max_start = max(0, num_frames - sequence_length)
        start = _random_int(self.rng, 0, max_start)
        return list(range(start, min(num_frames, start + sequence_length)))

    def _image_size(self, scene: SceneRecord, frames: list[FrameRecord]) -> int | tuple[int, int] | None:
        dataset_config = self.config.get("datasets", {}).get(scene.dataset, {})
        size_config = dataset_config.get("dynamic_resize", self.config.get("training", {}).get("dynamic_resize", {}))
        if not isinstance(size_config, dict) or not bool(size_config.get("enabled", scene.dataset == "shapenet")):
            return self.sam_image_size
        min_size = _positive_int(
            size_config.get("min_size", size_config.get("short_side_min", dataset_config.get("resize_min"))),
            name="dynamic_resize.min_size",
        )
        max_size = _positive_int(
            size_config.get("max_size", size_config.get("short_side_max", dataset_config.get("resize_max"))),
            name="dynamic_resize.max_size",
        )
        if min_size is None and max_size is None:
            return self.sam_image_size
        if min_size is None:
            min_size = max_size
        if max_size is None:
            max_size = min_size
        assert min_size is not None and max_size is not None
        if max_size < min_size:
            raise ValueError("dynamic_resize.max_size must be >= min_size")
        multiple = max(1, int(size_config.get("multiple", 32)))
        sampled = _random_int(self.rng, min_size, max_size)
        sampled = max(multiple, int(round(sampled / multiple)) * multiple)
        if bool(size_config.get("keep_aspect", False)):
            if not frames:
                return sampled
            with Image.open(frames[0].image_path) as image:
                width, height = image.size
            short_side = max(1, min(width, height))
            scale = sampled / short_side
            new_width = max(multiple, int(round(width * scale / multiple)) * multiple)
            new_height = max(multiple, int(round(height * scale / multiple)) * multiple)
            return new_width, new_height
        return sampled

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
        sequence_length = self._sequence_length(scene)
        return SamplingConfig(
            fov_sampling_probability=float(dataset_config.get("fov_sampling_probability", 0.0)),
            fov_threshold=float(global_sampling.get("fov_threshold", 0.25)),
            sequence_length=sequence_length,
        )

    def _sequence_length(self, scene: SceneRecord) -> int:
        global_sampling = self.config.get("sampling", {})
        dataset_config = self.config.get("datasets", {}).get(scene.dataset, {})
        base = int(global_sampling.get("sequence_length", self.config.get("training", {}).get("memory_frames", 8)))
        min_value = dataset_config.get("sequence_length_min", global_sampling.get("sequence_length_min"))
        max_value = dataset_config.get("sequence_length_max", global_sampling.get("sequence_length_max"))
        if min_value is None and max_value is None:
            return base
        min_length = _positive_int(min_value if min_value is not None else base, name="sequence_length_min")
        max_length = _positive_int(max_value if max_value is not None else base, name="sequence_length_max")
        assert min_length is not None and max_length is not None
        if max_length < min_length:
            raise ValueError("sequence_length_max must be >= sequence_length_min")
        return _random_int(self.rng, min_length, max_length)
