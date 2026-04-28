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
from torch import nn

from three_am.models.adapters import (
    ExternalBackboneConfig,
    ExternalDependencyError,
    Must3rFeatureAdapter,
    Sam2TrainingAdapter,
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


def external_config(config: dict[str, Any]) -> ExternalBackboneConfig:
    external = config.get("external", {})
    sam2_checkpoint = resolve_project_path(config, external.get("sam2_checkpoint"))
    sam2_config = external.get("sam2_config")
    if sam2_config is not None:
        sam2_config = str(sam2_config)
    sam2_repo = resolve_project_path(config, external.get("sam2_repo"))
    must3r_checkpoint = resolve_project_path(config, external.get("must3r_checkpoint"))
    must3r_repo = resolve_project_path(config, external.get("must3r_repo"))
    return ExternalBackboneConfig(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        sam2_repo=sam2_repo,
        must3r_checkpoint=must3r_checkpoint,
        must3r_repo=must3r_repo,
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


def dry_run(config: dict[str, Any], feature_cache: str | Path | None = None) -> dict[str, Any]:
    cache_root = configured_feature_cache_root(config, feature_cache)
    statuses = manifest_statuses(config, cache_root)
    external = external_config(config)
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
            "model": wrapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config,
            "rng": _rng_state(),
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
    wrapper.load_state_dict(payload["model"])
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
) -> tuple[torch.Tensor, ...]:
    if batch.must3r_features is not None:
        return batch.must3r_features
    adapter = adapter or Must3rFeatureAdapter(external_config(config))
    try:
        if getattr(adapter, "model", object()) is None:
            load = getattr(adapter, "load", None)
            if callable(load):
                load()
        features = adapter.extract_features(batch.images)
    except (ExternalDependencyError, NotImplementedError) as error:
        raise FeatureCacheRuntimeError(
            "MUSt3R feature cache is missing for "
            f"{batch.dataset}/{batch.scene_id} frames {list(batch.frame_ids)} and online MUSt3R extraction is unavailable. "
            "Run scripts/precompute_must3r_features.py after wiring MUSt3R, or provide --feature-cache."
        ) from error
    return tuple(feature.to(device) for feature in features)


class FeatureCacheRuntimeError(RuntimeError):
    pass


def run_training(
    config_path: str,
    *,
    iterations: int | None = None,
    resume: str | Path | None = None,
    device_name: str | None = None,
    feature_cache: str | Path | None = None,
    log_every: int | None = None,
    checkpoint_every: int | None = None,
    dry_run_only: bool = False,
    sam2_adapter: Sam2TrainingAdapter | None = None,
    must3r_adapter: Must3rFeatureAdapter | None = None,
) -> int:
    config = load_yaml(config_path)
    if dry_run_only:
        dry_run(config, feature_cache)
        return 0

    training_config = config.get("training", {})
    total_iterations = int(iterations if iterations is not None else training_config.get("iterations", 1_000_000))
    log_every = int(log_every if log_every is not None else training_config.get("log_every", 50))
    checkpoint_every = int(
        checkpoint_every if checkpoint_every is not None else training_config.get("checkpoint_every", 5000)
    )
    device = _device(device_name)
    dataset = ThreeAMTrainingDataset.from_config(config, feature_cache_root=feature_cache)

    core = build_core(config)
    sam2_adapter = sam2_adapter or Sam2TrainingAdapter(external_config(config))
    must3r_adapter = must3r_adapter or Must3rFeatureAdapter(external_config(config))
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
    start_step = 0
    if resume is not None:
        resume_path = resolve_project_path(config, resume)
        if resume_path is None:
            raise ValueError("resume path resolved to None")
        start_step = load_checkpoint(resume_path, wrapper=wrapper, optimizer=optimizer, scaler=scaler, device=device)
        print(f"Resumed training from {resume_path} at step {start_step}")

    paths = ProjectPaths.from_config(config)
    checkpoint_out = resolve_project_path(config, training_config["checkpoint_out"])
    if checkpoint_out is None:
        raise ValueError("training.checkpoint_out resolved to None")
    latest_path = paths.checkpoints / "latest.pt"
    weights = Sam2LossWeights()
    wrapper.train()
    last_step = start_step
    started = time.time()
    for step_index in range(start_step, total_iterations):
        step = step_index + 1
        batch = dataset.sample().to(device)
        must3r_features = _load_missing_must3r_features(batch, must3r_adapter, config, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            outputs = wrapper(batch, must3r_features)
            loss = sam2_training_loss(outputs, batch.target_masks, batch.has_object, weights)
        scaler.scale(loss.total).backward()
        scaler.step(optimizer)
        scaler.update()
        last_step = step
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
    print(f"Wrote checkpoint to {checkpoint_out}")
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
        dry_run_only=args.dry_run,
    )


if __name__ == "__main__":
    main()
