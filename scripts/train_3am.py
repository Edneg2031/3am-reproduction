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
    def __init__(self, core: ThreeAMCore, sam2_adapter: Sam2TrainingAdapter) -> None:
        super().__init__()
        self.core = core
        self.sam2_adapter = sam2_adapter

    def forward(self, batch: TrainingBatch, must3r_features: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        sam_features = self.sam2_adapter.encode_sam_features(batch.images)
        if sam_features.ndim != 4:
            raise ValueError(f"SAM2 features must have shape TCHW, got {tuple(sam_features.shape)}")
        merged_features = self.core.forward_features(sam_features, must3r_features)
        return self.sam2_adapter.forward_train_sequence(batch, merged_features)


def build_core(config: dict[str, Any]) -> ThreeAMCore:
    model_config = config["model"]
    return ThreeAMCore(
        ThreeAMConfig(
            sam_channels=int(model_config["sam_channels"]),
            must3r_channels=tuple(int(channels) for channels in model_config["must3r_channels"]),
            hidden_channels=int(model_config["hidden_channels"]),
            attention_heads=int(model_config["attention_heads"]),
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
        must3r_feature_layers=_parse_must3r_feature_layers(features.get("feature_layers", "encoder,4,7,11")),
        must3r_expected_channels=tuple(int(channels) for channels in must3r_channels)
        if must3r_channels is not None
        else None,
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
        },
        "features": {
            "online_must3r": online_enabled,
            "cache_enabled": not online_enabled,
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
) -> int:
    payload = _load_torch(path, map_location=device)
    if "optimizer" not in payload:
        raise ValueError(f"{path} is a model-weights file, not a training checkpoint; resume from latest.pt or step_*.pt")
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
) -> tuple[torch.Tensor, ...]:
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
    return tuple(feature.to(device) for feature in features)


def _extract_online_must3r_features(
    adapter: Must3rFeatureAdapter,
    batch: TrainingBatch,
) -> tuple[torch.Tensor, ...]:
    try:
        return adapter.extract_features(batch.images, image_paths=batch.image_paths)
    except TypeError as error:
        if "image_paths" not in str(error):
            raise
        return adapter.extract_features(batch.images)  # type: ignore[call-arg]


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
    frame_count = min(int(images.shape[0]), max(1, max_frames))
    rows: list[Image.Image] = []
    for frame_index in range(frame_count):
        base = _tensor_image_to_uint8(images[frame_index])
        gt_overlay = _overlay_mask(base, target_masks[frame_index], (0, 220, 80))
        pred_overlay = _overlay_mask(base, probabilities[frame_index], (255, 60, 40))
        panels = [
            _resize_for_visualization(Image.fromarray(base), max_side),
            _resize_for_visualization(Image.fromarray(gt_overlay), max_side),
            _resize_for_visualization(Image.fromarray(pred_overlay), max_side),
        ]
        row_width = sum(panel.width for panel in panels)
        row_height = max(panel.height for panel in panels)
        row = Image.new("RGB", (row_width, row_height + 22), (24, 24, 24))
        draw = ImageDraw.Draw(row)
        labels = [
            f"frame {batch.frame_ids[frame_index]}",
            "gt mask",
            "prediction",
        ]
        x = 0
        for panel, label in zip(panels, labels, strict=True):
            row.paste(panel, (x, 22))
            suffix = " prompt" if frame_index == batch.prompt.frame_index and label.startswith("frame") else ""
            draw.text((x + 4, 4), label + suffix, fill=(235, 235, 235))
            x += panel.width
        rows.append(row)
    canvas_width = max(row.width for row in rows)
    canvas_height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
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
    sam2_adapter: Sam2TrainingAdapter | None = None,
    must3r_adapter: Must3rFeatureAdapter | None = None,
) -> int:
    config = load_yaml(config_path)
    online_enabled = _online_must3r_enabled(config, online_must3r)
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
    resolved_visualization_dir = _resolve_visualization_dir(config, visualization_dir)
    visualization_max_frames = int(training_config.get("visualization_max_frames", 4))
    visualization_max_side = int(training_config.get("visualization_max_side", 384))
    device = _device(device_name)
    dataset = ThreeAMTrainingDataset.from_config(
        config,
        feature_cache_root=feature_cache,
        load_feature_cache=not online_enabled,
    )

    core = build_core(config)
    sam2_adapter = sam2_adapter or Sam2TrainingAdapter(external_config(config))
    must3r_adapter = must3r_adapter or Must3rFeatureAdapter(external_config(config, must3r_device=str(device)))
    if getattr(sam2_adapter, "model", object()) is None:
        sam2_adapter.load()
    wrapper = ThreeAMTrainingWrapper(core, sam2_adapter).to(device)
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
        start_step = load_checkpoint(resume_path, wrapper=wrapper, optimizer=optimizer, scaler=scaler, device=device)
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
    )


if __name__ == "__main__":
    main()
