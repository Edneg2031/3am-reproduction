#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

from three_am.models.adapters import (
    ExternalBackboneConfig,
    ExternalDependencyError,
    Must3rFeatureAdapter,
    Sam2TrainingAdapter,
    _normalize_must3r_amp,
    _parse_must3r_feature_layers,
)
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.models.three_am import ThreeAMConfig, ThreeAMCore
from three_am.training.dataset import (
    TrainingBatch,
    ThreeAMTrainingDataset,
    configured_feature_cache_root,
    manifest_statuses,
    resolve_project_path,
)
from three_am.training.losses import Sam2LossWeights, sam2_training_loss
from three_am.training.optim import build_adamw, mark_trainable_3am_modules
from three_am.utils.config import ProjectPaths, load_yaml


class ThreeAMTrainingWrapper(nn.Module):
    def __init__(self, core: ThreeAMCore, sam2_adapter: Sam2TrainingAdapter, *, strict_paper: bool = False) -> None:
        super().__init__()
        self.core = core
        self.sam2_adapter = sam2_adapter
        self.strict_paper = strict_paper

    def forward(
        self,
        batch: TrainingBatch,
        must3r_features: tuple[torch.Tensor, ...] | Must3rFeatureBundle,
    ) -> dict[str, torch.Tensor]:
        if self.strict_paper and not isinstance(must3r_features, Must3rFeatureBundle):
            raise ValueError("Strict paper training requires MUSt3R features as Must3rFeatureBundle with PE2D/point/ray maps")
        sam_features = self.sam2_adapter.encode_sam_features(batch.images)
        if sam_features.ndim != 4:
            raise ValueError(f"SAM2 features must have shape TCHW, got {tuple(sam_features.shape)}")
        merged_features = self.core.forward_features(sam_features, must3r_features)
        return self.sam2_adapter.forward_train_sequence(batch, merged_features)


def build_core(config: dict[str, Any]) -> ThreeAMCore:
    model_config = config["model"]
    training_config = config.get("training", {})
    return ThreeAMCore(
        ThreeAMConfig(
            sam_channels=int(model_config["sam_channels"]),
            must3r_channels=tuple(int(channels) for channels in model_config["must3r_channels"]),
            hidden_channels=int(model_config["hidden_channels"]),
            attention_heads=int(model_config["attention_heads"]),
            geometry_channels=int(model_config["geometry_channels"]) if model_config.get("geometry_channels") else None,
            strict_paper=bool(training_config.get("strict_paper", False)),
        )
    )


def external_config(config: dict[str, Any], *, must3r_device: str | None = None) -> ExternalBackboneConfig:
    external = config.get("external", {})
    features = config.get("features", {})
    model = config.get("model", {})
    sam2_checkpoint = resolve_project_path(config, external.get("sam2_checkpoint"))
    sam2_config = external.get("sam2_config")
    if sam2_config is not None:
        sam2_config = str(sam2_config)
    sam2_repo = resolve_project_path(config, external.get("sam2_repo"))
    must3r_checkpoint = resolve_project_path(config, external.get("must3r_checkpoint"))
    must3r_repo = resolve_project_path(config, external.get("must3r_repo"))
    must3r_channels = model.get("must3r_channels")
    return ExternalBackboneConfig(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_repo=sam2_repo,
        must3r_checkpoint=must3r_checkpoint,
        must3r_repo=must3r_repo,
        must3r_device=must3r_device,
        must3r_image_size=int(features.get("image_size", 512)),
        must3r_amp=_normalize_must3r_amp(features.get("amp", "bf16")),
        must3r_max_bs=int(features.get("max_bs", 1)),
        must3r_decode_batch_size=int(features.get("decode_batch_size", 1)),
        must3r_memory_window=int(features["memory_window"]) if features.get("memory_window") else None,
        must3r_full_scene_memory=bool(features.get("full_scene_memory", False)),
        must3r_feature_layers=_parse_must3r_feature_layers(features.get("feature_layers", "encoder,4,7,11")),
        must3r_expected_channels=tuple(int(channels) for channels in must3r_channels)
        if must3r_channels is not None
        else None,
        strict_paper=bool(config.get("training", {}).get("strict_paper", False)),
    )


def smoke_train(config_path: str, iterations: int) -> None:
    config = load_yaml(config_path)
    model = build_core(config)
    mark_trainable_3am_modules(model)
    optimizer = build_adamw(model, config.get("training", {}).get("learning_rates"))
    for step in range(iterations):
        sam = torch.randn(1, model.config.sam_channels, 16, 16)
        must3r = tuple(torch.randn(1, channels, 16, 16) for channels in model.config.must3r_channels)
        merged = model.forward_features(sam, must3r)
        loss = merged.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f}")
    output = resolve_project_path(config, config["training"]["checkpoint_out"])
    if output is None:
        raise ValueError("training.checkpoint_out resolved to None")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)
    print(f"Wrote smoke checkpoint to {output}")


def _availability(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sam2_config_exists(config: ExternalBackboneConfig) -> bool:
    if config.sam2_config is None:
        return False
    config_text = str(config.sam2_config)
    candidates = [Path(config_text).expanduser()]
    if config.sam2_repo is not None:
        repo = config.sam2_repo.expanduser()
        candidates.extend(
            [
                repo / config_text,
                repo / "sam2" / config_text,
                repo / "sam2_configs" / config_text,
            ]
        )
    return any(candidate.exists() for candidate in candidates)


def _online_must3r_enabled(config: dict[str, Any], override: bool | None = None) -> bool:
    if override is not None:
        return override
    return bool(config.get("features", {}).get("online", False))


def dry_run(
    config: dict[str, Any],
    feature_cache: str | Path | None = None,
    *,
    online_must3r: bool | None = None,
) -> dict[str, Any]:
    cache_root = configured_feature_cache_root(config, feature_cache)
    statuses = manifest_statuses(config, cache_root)
    external = external_config(config)
    features = config.get("features", {})
    model = config.get("model", {})
    online_enabled = _online_must3r_enabled(config, online_must3r)
    strict_paper = bool(config.get("training", {}).get("strict_paper", False))
    paths = ProjectPaths.from_config(config)
    training = config.get("training", {})
    payload = {
        "manifests": [
            {
                "dataset": status.dataset,
                "manifest": str(status.manifest),
                "exists": status.exists,
                "scenes": status.scenes,
                "frames": status.frames,
                "feature_cache_scenes": status.feature_cache_scenes,
            }
            for status in statuses
        ],
        "feature_cache_root": str(cache_root),
        "model": {
            "sam_image_size": model.get("sam_image_size", config.get("training", {}).get("sam_image_size")),
            "sam_channels": model.get("sam_channels"),
            "must3r_channels": model.get("must3r_channels"),
            "geometry_channels": model.get("geometry_channels"),
        },
        "features": {
            "online_must3r": online_enabled,
            "cache_enabled": not online_enabled,
            "require_decoder_memory": bool(features.get("require_decoder_memory", False)),
            "memory_window": features.get("memory_window"),
            "full_scene_memory": bool(features.get("full_scene_memory", False)),
            "image_size": int(features.get("image_size", 512)),
            "amp": features.get("amp", "bf16"),
            "max_bs": int(features.get("max_bs", 1)),
            "decode_batch_size": int(features.get("decode_batch_size", 1)),
            "feature_layers": list(_parse_must3r_feature_layers(features.get("feature_layers", "encoder,4,7,11"))),
        },
        "visualization": {
            "visualize_every": int(config.get("training", {}).get("visualize_every", 0)),
            "visualization_dir": str(
                _resolve_visualization_dir(config, config.get("training", {}).get("visualization_dir"))
            ),
            "visualization_max_frames": int(config.get("training", {}).get("visualization_max_frames", 4)),
            "visualization_max_side": int(config.get("training", {}).get("visualization_max_side", 384)),
        },
        "checkpoints": {
            "checkpoint_out": str(resolve_project_path(config, training.get("checkpoint_out"))),
            "latest": str(paths.checkpoints / "latest.pt"),
            "model_out": str(resolve_project_path(config, training.get("model_out", "outputs/checkpoints/3am_model.pt"))),
            "model_latest": str(paths.checkpoints / "model_latest.pt"),
            "checkpoint_every": int(training.get("checkpoint_every", 5000)),
            "save_model_every": int(training.get("save_model_every", training.get("checkpoint_every", 5000))),
            "auto_resume": bool(training.get("auto_resume", False)),
            "strict_paper": bool(training.get("strict_paper", False)),
            "validate_every": int(training.get("validate_every", 0)),
        },
        "external": {
            "sam2_importable": _availability("sam2"),
            "mast3r_importable": _availability("mast3r"),
            "sam2_repo": str(external.sam2_repo) if external.sam2_repo else None,
            "sam2_repo_exists": bool(external.sam2_repo and external.sam2_repo.exists()),
            "sam2_checkpoint": str(external.sam2_checkpoint) if external.sam2_checkpoint else None,
            "sam2_checkpoint_exists": bool(external.sam2_checkpoint and external.sam2_checkpoint.exists()),
            "sam2_config": str(external.sam2_config) if external.sam2_config else None,
            "sam2_config_exists": _sam2_config_exists(external),
            "must3r_repo": str(external.must3r_repo) if external.must3r_repo else None,
            "must3r_repo_exists": bool(external.must3r_repo and external.must3r_repo.exists()),
            "must3r_checkpoint": str(external.must3r_checkpoint) if external.must3r_checkpoint else None,
            "must3r_checkpoint_exists": bool(external.must3r_checkpoint and external.must3r_checkpoint.exists()),
        },
    }
    print(json.dumps(payload, indent=2))
    return payload


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover - older torch


def _grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:  # pragma: no cover - older torch
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - older torch


def _load_torch(path: Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=map_location)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _load_model_state(wrapper: ThreeAMTrainingWrapper, payload: dict[str, Any]) -> None:
    if "model" not in payload:
        raise ValueError("Checkpoint is missing model state")
    state = payload["model"]
    state_type = payload.get("model_state_type", "full")
    if state_type == "full":
        wrapper.load_state_dict(state)
        return
    if state_type != "trainable":
        raise ValueError(f"Unsupported model_state_type: {state_type}")
    incompatible = wrapper.load_state_dict(state, strict=False)
    missing_trainable = [
        name
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad and name in incompatible.missing_keys
    ]
    if missing_trainable:
        preview = ", ".join(missing_trainable[:5])
        suffix = "" if len(missing_trainable) <= 5 else f", ... ({len(missing_trainable)} total)"
        raise ValueError(f"Checkpoint is missing trainable model keys: {preview}{suffix}")
    if incompatible.unexpected_keys:
        preview = ", ".join(incompatible.unexpected_keys[:5])
        suffix = "" if len(incompatible.unexpected_keys) <= 5 else f", ... ({len(incompatible.unexpected_keys)} total)"
        raise ValueError(f"Checkpoint has unexpected model keys: {preview}{suffix}")


def save_checkpoint(
    path: Path,
    *,
    wrapper: ThreeAMTrainingWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "three_am_training_checkpoint",
            "model_state_type": "trainable",
            "strict_paper": bool(config.get("training", {}).get("strict_paper", False)),
            "model": _trainable_state_dict(wrapper),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config,
            "rng": _rng_state(),
        },
        path,
    )


def save_model_weights(
    path: Path,
    *,
    wrapper: ThreeAMTrainingWrapper,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _trainable_state_dict(wrapper)
    torch.save(
        {
            "type": "three_am_model_weights",
            "model_state_type": "trainable",
            "strict_paper": bool(config.get("training", {}).get("strict_paper", False)),
            "model": state,
            "trainable_keys": sorted(state),
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    wrapper: ThreeAMTrainingWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    expected_strict_paper: bool | None = None,
) -> int:
    payload = _load_torch(path, map_location=device)
    if "optimizer" not in payload:
        raise ValueError(f"{path} is a model-weights file, not a training checkpoint; resume from latest.pt or step_*.pt")
    if expected_strict_paper is True and payload.get("strict_paper") is not True:
        raise ValueError(
            f"{path} is not a strict-paper training checkpoint. Start a new run or resume from a strict-paper checkpoint."
        )
    if expected_strict_paper is False and payload.get("strict_paper") is True:
        raise ValueError(f"{path} is a strict-paper checkpoint; resume with training.strict_paper=true or --strict-paper.")
    _load_model_state(wrapper, payload)
    optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng_state(payload.get("rng", {}))
    return int(payload["step"])


def _load_missing_must3r_features(
    batch: TrainingBatch,
    adapter: Must3rFeatureAdapter | None,
    config: dict[str, Any],
    device: torch.device,
    *,
    online_must3r: bool,
) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
    if batch.must3r_features is not None:
        return batch.must3r_features
    if not online_must3r:
        cache_root = configured_feature_cache_root(config)
        num_levels = len(tuple(config.get("model", {}).get("must3r_channels", (256, 512, 768))))
        expected = [
            cache_root / batch.dataset / batch.scene_id / f"{frame_id}_level{level}.pt"
            for frame_id in batch.frame_ids
            for level in range(num_levels)
        ]
        preview = ", ".join(str(path) for path in expected[: min(3, len(expected))])
        suffix = "" if len(expected) <= 3 else f", ... ({len(expected)} expected files)"
        raise FeatureCacheRuntimeError(
            "MUSt3R feature cache is missing for "
            f"{batch.dataset}/{batch.scene_id} frames {list(batch.frame_ids)}. "
            f"Expected cache files like: {preview}{suffix}. "
            "Generate cache with scripts/precompute_must3r_features.py, add absolute must3r_feature_paths "
            "to the manifest, or run training with --online-must3r / features.online=true."
        )
    adapter = adapter or Must3rFeatureAdapter(external_config(config, must3r_device=str(device)))
    try:
        if getattr(adapter, "model", object()) is None:
            load = getattr(adapter, "load", None)
            if callable(load):
                load()
        features = _extract_online_must3r_features(adapter, batch)
    except (ExternalDependencyError, NotImplementedError) as error:
        cache_root = configured_feature_cache_root(config)
        num_levels = len(tuple(config.get("model", {}).get("must3r_channels", (256, 512, 768))))
        expected = [
            cache_root / batch.dataset / batch.scene_id / f"{frame_id}_level{level}.pt"
            for frame_id in batch.frame_ids
            for level in range(num_levels)
        ]
        preview = ", ".join(str(path) for path in expected[: min(3, len(expected))])
        suffix = "" if len(expected) <= 3 else f", ... ({len(expected)} expected files)"
        raise FeatureCacheRuntimeError(
            "MUSt3R feature cache is missing for "
            f"{batch.dataset}/{batch.scene_id} frames {list(batch.frame_ids)} and online MUSt3R extraction is unavailable. "
            f"Expected cache files like: {preview}{suffix}. "
            "Either generate these cache files under features.cache_root/--feature-cache, or put absolute per-frame "
            "feature paths in the manifest as must3r_feature_paths."
        ) from error
    if isinstance(features, Must3rFeatureBundle):
        return features.to(device)
    return tuple(feature.to(device) for feature in features)


def _extract_online_must3r_features(
    adapter: Must3rFeatureAdapter,
    batch: TrainingBatch,
) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
    try:
        return adapter.extract_features(batch.images, image_paths=batch.image_paths)
    except TypeError as error:
        if "image_paths" not in str(error):
            raise
        return adapter.extract_features(batch.images)  # type: ignore[call-arg]


def _tracking_metrics(outputs: dict[str, torch.Tensor], batch: TrainingBatch) -> dict[str, float]:
    logits = _mask_logits_for_visualization(outputs["mask_logits"].detach(), tuple(batch.target_masks.shape[-2:]))
    predictions = logits.sigmoid() >= 0.5
    targets = batch.target_masks.detach() > 0.5
    intersection = (predictions & targets).flatten(1).sum(dim=1).float()
    union = (predictions | targets).flatten(1).sum(dim=1).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
    visible = batch.has_object.detach().cpu().bool()
    visible_count = max(int(visible.sum().item()), 1)
    tracking_recall = float(((iou.cpu() > 0.0) & visible).float().sum().item() / visible_count)
    successful = (iou.cpu() > 0.0) & visible
    accuracy = float(iou.cpu()[successful].mean().item()) if bool(successful.any()) else 0.0
    return {
        "val_iou": float(iou.mean().item()),
        "val_tracking_recall": tracking_recall,
        "val_accuracy": accuracy,
    }


def run_validation_step(
    *,
    wrapper: ThreeAMTrainingWrapper,
    dataset: ThreeAMTrainingDataset,
    must3r_adapter: Must3rFeatureAdapter | None,
    config: dict[str, Any],
    device: torch.device,
    online_must3r: bool,
) -> dict[str, float]:
    was_training = wrapper.training
    wrapper.eval()
    with torch.no_grad():
        batch = dataset.sample().to(device)
        must3r_features = _load_missing_must3r_features(
            batch,
            must3r_adapter,
            config,
            device,
            online_must3r=online_must3r,
        )
        outputs = wrapper(batch, must3r_features)
        metrics = _tracking_metrics(outputs, batch)
    if was_training:
        wrapper.train()
    return metrics


class FeatureCacheRuntimeError(RuntimeError):
    pass


def _resolve_visualization_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    configured = override or config.get("training", {}).get("visualization_dir", "outputs/visualizations/train")
    resolved = resolve_project_path(config, configured)
    if resolved is None:
        raise ValueError("visualization directory resolved to None")
    return resolved


def _mask_logits_for_visualization(mask_logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0]
    if mask_logits.ndim != 3:
        raise ValueError(f"mask_logits for visualization must have shape THW, got {tuple(mask_logits.shape)}")
    if tuple(mask_logits.shape[-2:]) != target_size:
        mask_logits = F.interpolate(
            mask_logits[:, None].float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return mask_logits


def _tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _overlay_mask(image: np.ndarray, mask: torch.Tensor, color: tuple[int, int, int]) -> np.ndarray:
    alpha = mask.detach().float().cpu().clamp(0, 1).numpy()[..., None] * 0.55
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = image.astype(np.float32) * (1.0 - alpha) + color_array * alpha
    return blended.round().clip(0, 255).astype(np.uint8)


def _match_error_overlay(
    image: np.ndarray,
    target_mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    target = target_mask.detach().float().cpu().numpy() > 0.5
    pred = prediction.detach().float().cpu().numpy() >= threshold
    if target.shape != pred.shape:
        raise ValueError(f"target and prediction masks must have the same shape, got {target.shape} and {pred.shape}")
    overlay = image.astype(np.float32).copy()
    alpha = 0.65
    categories = (
        (target & pred, np.asarray((255, 220, 0), dtype=np.float32)),
        (target & ~pred, np.asarray((0, 220, 80), dtype=np.float32)),
        (~target & pred, np.asarray((255, 60, 40), dtype=np.float32)),
    )
    for category_mask, color in categories:
        if category_mask.any():
            overlay[category_mask] = overlay[category_mask] * (1.0 - alpha) + color * alpha
    return overlay.round().clip(0, 255).astype(np.uint8)


def _binary_iou_by_frame(predictions: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must have the same shape, got {predictions.shape} and {targets.shape}")
    pred = predictions.detach().float().cpu() >= threshold
    target = targets.detach().float().cpu() > 0.5
    intersection = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _training_visualization_header_lines(
    *,
    step: int,
    dataset: str,
    scene_id: str,
    prompt_type: str,
    reference_frame: str,
    frame_ids: tuple[str, ...],
    mean_iou: float | None,
    visible_iou: float | None,
    tracking_recall: float | None,
    sampling_mode: str = "unknown",
) -> list[str]:
    joined_frame_ids = ", ".join(frame_ids)
    return [
        f"3AM tracking batch | step={step} dataset={dataset} scene={scene_id} sampling={sampling_mode}",
        (
            f"prompt={prompt_type} ref={reference_frame} frames=[{joined_frame_ids}] "
            f"mean_iou={_format_metric(mean_iou)} visible_iou={_format_metric(visible_iou)} "
            f"tracking_recall_like={_format_metric(tracking_recall)}"
        ),
        "colors: GT=green prediction=red overlap=yellow gt-only=green pred-only=red",
    ]


def _visualization_frame_order(num_frames: int, reference_index: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    reference_index = min(max(reference_index, 0), num_frames - 1)
    limit = min(num_frames, max(1, max_frames))
    order = [reference_index]
    order.extend(index for index in range(reference_index + 1, num_frames))
    order.extend(index for index in range(reference_index - 1, -1, -1))
    return order[:limit]


def _resize_for_visualization(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)


def save_training_visualization(
    *,
    batch: TrainingBatch,
    outputs: dict[str, torch.Tensor],
    step: int,
    output_dir: Path,
    max_frames: int = 4,
    max_side: int = 384,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch.images.detach()
    target_masks = batch.target_masks.detach()
    logits = _mask_logits_for_visualization(outputs["mask_logits"].detach(), tuple(target_masks.shape[-2:]))
    probabilities = logits.sigmoid()
    num_frames = int(images.shape[0])
    reference_index = min(max(int(batch.prompt.frame_index), 0), max(num_frames - 1, 0))
    frame_order = _visualization_frame_order(num_frames, reference_index, max_frames)
    ious = _binary_iou_by_frame(probabilities, target_masks)
    has_object = batch.has_object.detach().cpu().bool()
    mean_iou = float(ious.mean().item()) if ious.numel() else None
    visible_iou = float(ious[has_object].mean().item()) if bool(has_object.any()) else None
    tracking_recall = (
        float(((ious > 0.0) & has_object).float().sum().item() / has_object.float().sum().item())
        if bool(has_object.any())
        else None
    )
    rows: list[Image.Image] = []
    for frame_index in frame_order:
        base = _tensor_image_to_uint8(images[frame_index])
        gt_overlay = _overlay_mask(base, target_masks[frame_index], (0, 220, 80))
        pred_overlay = _overlay_mask(base, probabilities[frame_index], (255, 60, 40))
        match_overlay = _match_error_overlay(base, target_masks[frame_index], probabilities[frame_index])
        panels = [
            _resize_for_visualization(Image.fromarray(base), max_side),
            _resize_for_visualization(Image.fromarray(gt_overlay), max_side),
            _resize_for_visualization(Image.fromarray(pred_overlay), max_side),
            _resize_for_visualization(Image.fromarray(match_overlay), max_side),
        ]
        row_width = sum(panel.width for panel in panels)
        row_height = max(panel.height for panel in panels)
        label_height = 38
        row = Image.new("RGB", (row_width, row_height + label_height), (24, 24, 24))
        draw = ImageDraw.Draw(row)
        frame_id = batch.frame_ids[frame_index]
        is_reference = frame_index == reference_index
        visibility = "visible GT" if bool(has_object[frame_index]) else "absent GT"
        frame_label = f"image | frame {frame_id}"
        if is_reference:
            frame_label = f"image | REF/PROMPT {batch.prompt.type} {frame_id}"
        labels = [
            frame_label,
            "gt mask",
            "prediction",
            "match/error",
        ]
        detail = f"IoU={ious[frame_index].item():.3f} | {visibility}"
        x = 0
        for panel, label in zip(panels, labels, strict=True):
            row.paste(panel, (x, label_height))
            fill = (255, 236, 120) if is_reference and "REF/PROMPT" in label else (235, 235, 235)
            draw.text((x + 4, 4), label, fill=fill)
            draw.text((x + 4, 20), detail if label.startswith("image") else "", fill=(205, 205, 205))
            x += panel.width
        rows.append(row)
    canvas_width = max(720, *(row.width for row in rows))
    header_height = 58
    canvas_height = header_height + sum(row.height for row in rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    reference_frame = batch.frame_ids[reference_index] if 0 <= reference_index < len(batch.frame_ids) else str(reference_index)
    header_lines = _training_visualization_header_lines(
        step=step,
        dataset=batch.dataset,
        scene_id=batch.scene_id,
        prompt_type=batch.prompt.type,
        reference_frame=reference_frame,
        frame_ids=batch.frame_ids,
        mean_iou=mean_iou,
        visible_iou=visible_iou,
        tracking_recall=tracking_recall,
        sampling_mode=batch.sampling_mode,
    )
    for line_index, line in enumerate(header_lines):
        draw.text((8, 6 + line_index * 16), line[:180], fill=(235, 235, 235))
    y = 0
    for row in rows:
        canvas.paste(row, (0, header_height + y))
        y += row.height
    safe_scene = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in batch.scene_id)
    path = output_dir / f"step_{step:07d}_{batch.dataset}_{safe_scene}.png"
    canvas.save(path)
    return path


def run_training(
    config_path: str,
    *,
    iterations: int | None = None,
    resume: str | Path | None = None,
    device_name: str | None = None,
    feature_cache: str | Path | None = None,
    log_every: int | None = None,
    checkpoint_every: int | None = None,
    visualize_every: int | None = None,
    visualization_dir: str | Path | None = None,
    save_model_every: int | None = None,
    model_out: str | Path | None = None,
    auto_resume: bool = False,
    dry_run_only: bool = False,
    online_must3r: bool | None = None,
    strict_paper: bool | None = None,
    sam2_adapter: Sam2TrainingAdapter | None = None,
    must3r_adapter: Must3rFeatureAdapter | None = None,
) -> int:
    config = load_yaml(config_path)
    if strict_paper is not None:
        config.setdefault("training", {})["strict_paper"] = bool(strict_paper)
    online_enabled = _online_must3r_enabled(config, online_must3r)
    strict_paper_enabled = bool(config.get("training", {}).get("strict_paper", False))
    if dry_run_only:
        dry_run(config, feature_cache, online_must3r=online_must3r)
        return 0

    training_config = config.get("training", {})
    total_iterations = int(iterations if iterations is not None else training_config.get("iterations", 1_000_000))
    log_every = int(log_every if log_every is not None else training_config.get("log_every", 50))
    checkpoint_every = int(
        checkpoint_every if checkpoint_every is not None else training_config.get("checkpoint_every", 5000)
    )
    save_model_every = int(
        save_model_every
        if save_model_every is not None
        else training_config.get("save_model_every", checkpoint_every)
    )
    visualize_every = int(
        visualize_every if visualize_every is not None else training_config.get("visualize_every", 0)
    )
    validate_every = int(training_config.get("validate_every", 0))
    resolved_visualization_dir = _resolve_visualization_dir(config, visualization_dir)
    visualization_max_frames = int(training_config.get("visualization_max_frames", 4))
    visualization_max_side = int(training_config.get("visualization_max_side", 384))
    device = _device(device_name)
    dataset = ThreeAMTrainingDataset.from_config(
        config,
        feature_cache_root=feature_cache,
        load_feature_cache=not online_enabled,
    )
    validation_dataset = (
        ThreeAMTrainingDataset.from_config(
            config,
            feature_cache_root=feature_cache,
            load_feature_cache=not online_enabled,
        )
        if validate_every > 0
        else None
    )

    core = build_core(config)
    sam2_adapter = sam2_adapter or Sam2TrainingAdapter(external_config(config))
    must3r_adapter = must3r_adapter or Must3rFeatureAdapter(external_config(config, must3r_device=str(device)))
    if getattr(sam2_adapter, "model", object()) is None:
        sam2_adapter.load()
    wrapper = ThreeAMTrainingWrapper(core, sam2_adapter, strict_paper=strict_paper_enabled).to(device)
    mark_trainable_3am_modules(wrapper)
    freeze_image_encoder = getattr(wrapper.sam2_adapter, "freeze_image_encoder", None)
    if callable(freeze_image_encoder):
        freeze_image_encoder()
    optimizer = build_adamw(
        wrapper,
        training_config.get("learning_rates"),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = _grad_scaler(device, amp_enabled)
    paths = ProjectPaths.from_config(config)
    checkpoint_out = resolve_project_path(config, training_config["checkpoint_out"])
    if checkpoint_out is None:
        raise ValueError("training.checkpoint_out resolved to None")
    model_out_path = resolve_project_path(
        config,
        model_out if model_out is not None else training_config.get("model_out", "outputs/checkpoints/3am_model.pt"),
    )
    if model_out_path is None:
        raise ValueError("training.model_out resolved to None")
    latest_path = paths.checkpoints / "latest.pt"
    model_latest_path = paths.checkpoints / "model_latest.pt"
    start_step = 0
    if resume is None and (auto_resume or bool(training_config.get("auto_resume", False))) and latest_path.exists():
        resume = latest_path
    if resume is not None:
        resume_path = resolve_project_path(config, resume)
        if resume_path is None:
            raise ValueError("resume path resolved to None")
        start_step = load_checkpoint(
            resume_path,
            wrapper=wrapper,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_strict_paper=strict_paper_enabled,
        )
        print(f"Resumed training from {resume_path} at step {start_step}")

    weights = Sam2LossWeights()
    wrapper.train()
    last_step = start_step
    started = time.time()
    for step_index in range(start_step, total_iterations):
        step = step_index + 1
        batch = dataset.sample().to(device)
        must3r_features = _load_missing_must3r_features(
            batch,
            must3r_adapter,
            config,
            device,
            online_must3r=online_enabled,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            outputs = wrapper(batch, must3r_features)
            loss = sam2_training_loss(outputs, batch.target_masks, batch.has_object, weights)
        scaler.scale(loss.total).backward()
        scaler.step(optimizer)
        scaler.update()
        last_step = step
        if visualize_every > 0 and (step == 1 or step % visualize_every == 0):
            visualization_path = save_training_visualization(
                batch=batch,
                outputs=outputs,
                step=step,
                output_dir=resolved_visualization_dir,
                max_frames=visualization_max_frames,
                max_side=visualization_max_side,
            )
            print(f"visualization={visualization_path}")
        if log_every > 0 and (step == 1 or step % log_every == 0):
            items = loss.detached_items()
            elapsed = max(time.time() - started, 1e-6)
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"dataset={batch.dataset}",
                        f"scene={batch.scene_id}",
                        f"fps={step / elapsed:.3f}",
                        *(f"{key}={value:.6f}" for key, value in items.items()),
                    ]
                )
            )
        if validation_dataset is not None and validate_every > 0 and step % validate_every == 0:
            metrics = run_validation_step(
                wrapper=wrapper,
                dataset=validation_dataset,
                must3r_adapter=must3r_adapter,
                config=config,
                device=device,
                online_must3r=online_enabled,
            )
            print(" ".join([f"validation_step={step}", *(f"{key}={value:.6f}" for key, value in metrics.items())]))
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                paths.checkpoints / f"step_{step}.pt",
                wrapper=wrapper,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                config=config,
            )
            save_checkpoint(
                latest_path,
                wrapper=wrapper,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                config=config,
            )
        if save_model_every > 0 and step % save_model_every == 0:
            save_model_weights(
                paths.checkpoints / f"model_step_{step}.pt",
                wrapper=wrapper,
                step=step,
                config=config,
            )
            save_model_weights(
                model_latest_path,
                wrapper=wrapper,
                step=step,
                config=config,
            )

    save_checkpoint(
        checkpoint_out,
        wrapper=wrapper,
        optimizer=optimizer,
        scaler=scaler,
        step=last_step,
        config=config,
    )
    save_checkpoint(
        latest_path,
        wrapper=wrapper,
        optimizer=optimizer,
        scaler=scaler,
        step=last_step,
        config=config,
    )
    save_model_weights(
        model_out_path,
        wrapper=wrapper,
        step=last_step,
        config=config,
    )
    save_model_weights(
        model_latest_path,
        wrapper=wrapper,
        step=last_step,
        config=config,
    )
    print(f"Wrote checkpoint to {checkpoint_out}")
    print(f"Wrote model weights to {model_out_path}")
    return last_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unofficial 3AM reimplementation")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    parser.add_argument("--smoke", action="store_true", help="run synthetic FeatureMerger training")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--save-model-every", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--visualization-dir", default=None)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-paper", action="store_true", dest="strict_paper", default=None)
    strict_group.add_argument("--no-strict-paper", action="store_false", dest="strict_paper")
    parser.add_argument(
        "--online-must3r",
        action="store_true",
        default=None,
        help="compute MUSt3R features inside the training loop instead of loading feature cache files",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_train(args.config, args.iterations or 10)
        return
    run_training(
        args.config,
        iterations=args.iterations,
        resume=args.resume,
        device_name=args.device,
        feature_cache=args.feature_cache,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        save_model_every=args.save_model_every,
        model_out=args.model_out,
        auto_resume=args.auto_resume,
        visualize_every=args.visualize_every,
        visualization_dir=args.visualization_dir,
        dry_run_only=args.dry_run,
        online_must3r=args.online_must3r,
        strict_paper=args.strict_paper,
    )


if __name__ == "__main__":
    main()
