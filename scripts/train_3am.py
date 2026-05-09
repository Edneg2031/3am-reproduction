#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.nn import functional as F

from three_am.data.schema import SceneRecord
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
    LetterboxTransform,
    Prompt,
    TrainingBatch,
    ThreeAMTrainingDataset,
    configured_feature_cache_root,
    configured_training_datasets,
    load_image_tensor,
    load_mask_array,
    load_training_scenes,
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


class TrainingVisualizationArtifact(NamedTuple):
    summary_path: Path
    video_path: Path | None = None


class ValidationStepResult(NamedTuple):
    metrics: dict[str, float]
    visualization: TrainingVisualizationArtifact | None = None


class _CudaMemoryDebugger:
    _PHASE_TO_FIELD = {
        "after_batch_to_device": "cuda_mem_batch_mb",
        "after_must3r_ready": "cuda_mem_must3r_mb",
        "after_forward": "cuda_mem_forward_mb",
        "after_backward": "cuda_mem_backward_mb",
    }

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = True
        self._records: dict[str, float] = {}

    def start_step(self) -> None:
        self._records = {}
        torch.cuda.reset_peak_memory_stats(device=self.device)

    def snapshot(self, phase: str) -> None:
        field = self._PHASE_TO_FIELD.get(phase)
        if field is None:
            raise KeyError(f"Unknown CUDA memory debug phase: {phase}")
        allocated_mb = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0)
        self._records[field] = round(float(allocated_mb), 3)

    def fields(self) -> dict[str, float]:
        return dict(self._records)


class _NoopCudaMemoryDebugger:
    enabled = False

    def start_step(self) -> None:
        return None

    def snapshot(self, phase: str) -> None:
        return None

    def fields(self) -> dict[str, float]:
        return {}


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


def _find_spec(name: str, repo: Path | None = None) -> importlib.machinery.ModuleSpec | None:
    added_repo = False
    repo_text: str | None = None
    if repo is not None:
        resolved = repo.expanduser().resolve()
        if resolved.exists():
            repo_text = str(resolved)
            if repo_text not in sys.path:
                sys.path.insert(0, repo_text)
                added_repo = True
    try:
        importlib.invalidate_caches()
        return importlib.util.find_spec(name)
    except Exception:
        return None
    finally:
        if added_repo and repo_text is not None:
            try:
                sys.path.remove(repo_text)
            except ValueError:
                pass


def _availability(name: str, repo: Path | None = None) -> bool:
    return _find_spec(name, repo) is not None


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


def _configured_sequence_lengths(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    sampling = config.get("sampling", {})
    training = config.get("training", {})
    datasets = config.get("datasets", {})
    base = int(sampling.get("sequence_length", training.get("memory_frames", 8)))
    summary: dict[str, dict[str, int]] = {}
    for dataset_name in configured_training_datasets(config):
        dataset_config = datasets.get(dataset_name, {})
        min_value = dataset_config.get("sequence_length_min", sampling.get("sequence_length_min", base))
        max_value = dataset_config.get("sequence_length_max", sampling.get("sequence_length_max", base))
        summary[dataset_name] = {
            "min": int(min_value if min_value is not None else base),
            "max": int(max_value if max_value is not None else base),
        }
    return summary


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
        "training": {
            "datasets": list(configured_training_datasets(config)),
            "memory_frames": int(training.get("memory_frames", 8)),
            "sequence_lengths": _configured_sequence_lengths(config),
            "strict_paper": strict_paper,
            "validation_fraction": float(training.get("validation_fraction", 0.1 if bool(training.get("validate_every", 0)) else 0.0)),
            "validation_seed": int(training.get("validation_seed", 0)),
            "validation_scenes": [str(scene) for scene in training.get("validation_scenes", [])],
            "amp": bool(training.get("amp", True)),
            "cuda_memory_debug": bool(training.get("cuda_memory_debug", False)),
        },
        "visualization": {
            "visualize_every": int(config.get("training", {}).get("visualize_every", 0)),
            "visualization_dir": str(
                _resolve_visualization_dir(config, config.get("training", {}).get("visualization_dir"))
            ),
            "visualization_full_video": bool(config.get("training", {}).get("visualization_full_video", False)),
            "visualization_chunk_size": int(config.get("training", {}).get("visualization_chunk_size", 32)),
            "visualization_max_frames": int(config.get("training", {}).get("visualization_max_frames", 4)),
            "visualization_max_side": int(config.get("training", {}).get("visualization_max_side", 384)),
            "visualization_fps": int(config.get("training", {}).get("visualization_fps", 6)),
            "visualization_full_scene_source_fps": config.get("training", {}).get("visualization_full_scene_source_fps"),
            "visualization_full_scene_sample_fps": config.get("training", {}).get("visualization_full_scene_sample_fps"),
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
            "sam2_point_pseudo_masks": training.get("sam2_point_pseudo_masks", {}),
        },
        "external": {
            "sam2_repo": str(external.sam2_repo) if external.sam2_repo else None,
            "sam2_repo_exists": bool(external.sam2_repo and external.sam2_repo.exists()),
            "sam2_importable": _availability("sam2", external.sam2_repo),
            "sam2_checkpoint": str(external.sam2_checkpoint) if external.sam2_checkpoint else None,
            "sam2_checkpoint_exists": bool(external.sam2_checkpoint and external.sam2_checkpoint.exists()),
            "sam2_config": str(external.sam2_config) if external.sam2_config else None,
            "sam2_config_exists": _sam2_config_exists(external),
            "must3r_repo": str(external.must3r_repo) if external.must3r_repo else None,
            "must3r_repo_exists": bool(external.must3r_repo and external.must3r_repo.exists()),
            "must3r_importable": _availability("must3r", external.must3r_repo),
            "mast3r_importable": _availability("mast3r", external.must3r_repo),
            "dust3r_importable": _availability("dust3r", external.must3r_repo),
            "must3r_checkpoint": str(external.must3r_checkpoint) if external.must3r_checkpoint else None,
            "must3r_checkpoint_exists": bool(external.must3r_checkpoint and external.must3r_checkpoint.exists()),
        },
    }
    print(json.dumps(payload, indent=2))
    return payload


def _scene_identity(scene: SceneRecord) -> tuple[str, str]:
    return str(scene.dataset), scene.scene_id


def _split_training_scenes(
    scenes: Sequence[SceneRecord],
    *,
    validation_fraction: float,
    validation_seed: int,
    validation_scene_ids: Sequence[str] = (),
) -> tuple[list[SceneRecord], list[SceneRecord]]:
    ordered_scenes = sorted(scenes, key=_scene_identity)
    if not ordered_scenes:
        return [], []

    requested_validation_ids = [str(value).strip() for value in validation_scene_ids if str(value).strip()]
    if requested_validation_ids:
        scene_lookup = {_scene_identity(scene): scene for scene in ordered_scenes}
        scene_ids: dict[str, list[tuple[str, str]]] = {}
        for scene in ordered_scenes:
            scene_ids.setdefault(scene.scene_id, []).append(_scene_identity(scene))
        validation_keys: set[tuple[str, str]] = set()
        missing: list[str] = []
        ambiguous: list[str] = []
        for raw_value in requested_validation_ids:
            dataset_name: str | None = None
            scene_id = raw_value
            for separator in ("/", ":"):
                if separator in raw_value:
                    dataset_name, scene_id = raw_value.split(separator, 1)
                    dataset_name = dataset_name.strip() or None
                    scene_id = scene_id.strip()
                    break
            if dataset_name is not None:
                key = (dataset_name, scene_id)
                if key not in scene_lookup:
                    missing.append(raw_value)
                else:
                    validation_keys.add(key)
                continue
            matches = scene_ids.get(scene_id, [])
            if not matches:
                missing.append(raw_value)
            elif len(matches) > 1:
                ambiguous.append(raw_value)
            else:
                validation_keys.add(matches[0])
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise ValueError(f"training.validation_scenes did not match any scenes: {preview}{suffix}")
        if ambiguous:
            preview = ", ".join(ambiguous[:8])
            suffix = "" if len(ambiguous) <= 8 else f", ... ({len(ambiguous)} total)"
            raise ValueError(f"training.validation_scenes is ambiguous; qualify these scene IDs with dataset/: {preview}{suffix}")
        train_scenes = [scene for scene in ordered_scenes if _scene_identity(scene) not in validation_keys]
        validation_scenes = [scene for scene in ordered_scenes if _scene_identity(scene) in validation_keys]
        if not validation_scenes:
            raise ValueError("training.validation_scenes resolved to no validation scenes")
        if not train_scenes:
            if len(ordered_scenes) == 1:
                return list(ordered_scenes), list(ordered_scenes)
            raise ValueError("training.validation_scenes would leave no training scenes")
        return train_scenes, validation_scenes

    fraction = float(validation_fraction)
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError("training.validation_fraction must be in [0, 1)")
    if fraction <= 0.0:
        return list(ordered_scenes), []
    if len(ordered_scenes) == 1:
        return list(ordered_scenes), list(ordered_scenes)

    rng = random.Random(int(validation_seed))
    shuffled = list(ordered_scenes)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_keys = {_scene_identity(scene) for scene in shuffled[:validation_count]}
    train_scenes = [scene for scene in ordered_scenes if _scene_identity(scene) not in validation_keys]
    validation_scenes = [scene for scene in ordered_scenes if _scene_identity(scene) in validation_keys]
    return train_scenes, validation_scenes


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _training_float_dtype(device: torch.device, amp_enabled: bool) -> torch.dtype:
    if amp_enabled and device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
    return torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enabled)  # pragma: no cover - older torch


def _grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:  # pragma: no cover - older torch
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - older torch


def _build_cuda_memory_debugger(device: torch.device, enabled: bool) -> _CudaMemoryDebugger | _NoopCudaMemoryDebugger:
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return _NoopCudaMemoryDebugger()
    return _CudaMemoryDebugger(device)


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


def _trainable_parameter_summary(module: nn.Module, *, max_names: int = 8) -> str:
    names = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    preview = ", ".join(names[:max_names])
    if len(names) > max_names:
        preview += f", ... ({len(names)} total)"
    return preview or "none"


def _output_grad_summary(outputs: dict[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}:requires_grad={value.requires_grad},grad_fn={type(value.grad_fn).__name__ if value.grad_fn else 'None'}")
    return "; ".join(parts) or "no tensor outputs"


def _ensure_loss_is_differentiable(
    loss: torch.Tensor,
    *,
    wrapper: ThreeAMTrainingWrapper,
    outputs: dict[str, torch.Tensor],
    batch: TrainingBatch | None = None,
) -> None:
    if loss.requires_grad and loss.grad_fn is not None:
        return
    selector = getattr(wrapper.sam2_adapter, "last_feature_selector", None)
    batch_context = ""
    if batch is not None:
        batch_context = (
            f"; batch_frames={list(batch.frame_ids)}; has_object={batch.has_object.detach().cpu().tolist()}; "
            f"prompt_frame_index={int(batch.prompt.frame_index)}; sampling_mode={batch.sampling_mode}; "
            f"target_source={batch.target_source}"
        )
    raise RuntimeError(
        "3AM training loss is detached before backward. This usually means the selected SAM2 tracking forward path "
        "ran under no_grad/inference_mode or did not use the merged 3AM backbone feature. "
        f"sam2_feature_selector={selector!r}; trainable_parameters={_trainable_parameter_summary(wrapper)}; "
        f"outputs=({_output_grad_summary(outputs)}){batch_context}"
    )


def _loss_is_differentiable(loss: torch.Tensor) -> bool:
    return bool(loss.requires_grad and loss.grad_fn is not None)


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
    float_dtype: torch.dtype | None = None,
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
            f"Online extraction failed with {type(error).__name__}: {error}. "
            f"Expected cache files like: {preview}{suffix}. "
            "Either generate these cache files under features.cache_root/--feature-cache, or put absolute per-frame "
            "feature paths in the manifest as must3r_feature_paths. If you intended to use --online-must3r, run "
            "`--dry-run --online-must3r` and make sure must3r_importable, mast3r_importable, and "
            "dust3r_importable are all true."
        ) from error
    if isinstance(features, Must3rFeatureBundle):
        return features.to(device, dtype=float_dtype)
    converted: list[torch.Tensor] = []
    for feature in features:
        if float_dtype is not None and feature.is_floating_point():
            converted.append(feature.to(device=device, dtype=float_dtype))
        else:
            converted.append(feature.to(device=device))
    return tuple(converted)


def _mask_foreground_ratios(target_masks: torch.Tensor) -> torch.Tensor:
    return target_masks.detach().float().flatten(1).mean(dim=1).cpu()


def _batch_masks_exceed_ratio(batch: TrainingBatch, max_ratio: float) -> bool:
    if batch.target_masks.numel() == 0:
        return False
    return bool((_mask_foreground_ratios(batch.target_masks) >= float(max_ratio)).any().item())


def _scan_dataset_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("training", {}).get("sam2_point_pseudo_masks", {}))


def _sam2_point_pseudo_masks_enabled(config: dict[str, Any], batch: TrainingBatch) -> bool:
    pseudo = _scan_dataset_training_config(config)
    datasets = tuple(str(value) for value in pseudo.get("datasets", ("scannetpp",)))
    if batch.dataset not in datasets:
        return False
    mode = str(pseudo.get("mode", "auto")).lower()
    if mode in {"off", "false", "none", "0"}:
        return False
    if mode in {"on", "true", "always", "force", "1"}:
        return True
    threshold = float(pseudo.get("auto_foreground_ratio_threshold", 0.98))
    return _batch_masks_exceed_ratio(batch, threshold)


def _random_prompt_point_from_image(batch: TrainingBatch) -> tuple[int, torch.Tensor, torch.Tensor]:
    num_frames = int(batch.images.shape[0])
    reference_index = random.randrange(max(1, num_frames))
    height, width = int(batch.images.shape[-2]), int(batch.images.shape[-1])
    point = torch.tensor(
        [[float(random.randrange(max(1, width))), float(random.randrange(max(1, height)))]],
        dtype=torch.float32,
        device=batch.images.device,
    )
    labels = torch.ones(1, dtype=torch.int64, device=batch.images.device)
    return reference_index, point, labels


def _binarize_pseudo_masks(masks: torch.Tensor, threshold: float) -> torch.Tensor:
    return (masks.detach().float().sigmoid() >= float(threshold)).to(dtype=torch.float32)


def _valid_pseudo_mask_areas(
    target_masks: torch.Tensor,
    *,
    reference_index: int,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    ratios = _mask_foreground_ratios(target_masks)
    if ratios.numel() == 0:
        return False
    reference_index = min(max(int(reference_index), 0), ratios.numel() - 1)
    reference_valid = float(min_ratio) <= float(ratios[reference_index].item()) <= float(max_ratio)
    no_full_frame = bool((ratios <= float(max_ratio)).all().item())
    return reference_valid and no_full_frame


def _apply_sam2_point_pseudo_masks(
    batch: TrainingBatch,
    *,
    sam2_adapter: Sam2TrainingAdapter,
    config: dict[str, Any],
) -> TrainingBatch:
    if not _sam2_point_pseudo_masks_enabled(config, batch):
        return batch
    pseudo = _scan_dataset_training_config(config)
    attempts = max(1, int(pseudo.get("max_attempts", 8)))
    threshold = float(pseudo.get("mask_threshold", 0.5))
    min_ratio = float(pseudo.get("min_foreground_ratio", 0.001))
    max_ratio = float(pseudo.get("max_foreground_ratio", 0.9))
    source_reason = (
        "sam2_point_pseudo_mask_forced"
        if str(pseudo.get("mode", "auto")).lower() in {"on", "true", "always", "force", "1"}
        else "sam2_point_pseudo_mask_auto_full_dataset_mask"
    )
    best: tuple[TrainingBatch, torch.Tensor] | None = None
    best_distance = float("inf")
    last_error: Exception | None = None
    for _ in range(attempts):
        reference_index, point, labels = _random_prompt_point_from_image(batch)
        try:
            pseudo_logits = _sam2_point_prompt_masks_for_batch(
                sam2_adapter,
                batch,
                reference_index=reference_index,
                point=point,
                labels=labels,
            )
        except Exception as error:
            last_error = error
            continue
        target_masks = _binarize_pseudo_masks(pseudo_logits, threshold)
        has_object = target_masks.flatten(1).any(dim=1).to(device=batch.has_object.device)
        prompt = Prompt(
            type="point",
            frame_index=reference_index,
            points=point.detach().clone(),
            point_labels=labels.detach().clone(),
        )
        updated = replace(
            batch,
            target_masks=target_masks.to(device=batch.target_masks.device),
            has_object=has_object,
            object_visibility=has_object,
            prompt=prompt,
            reference_frame_id=batch.frame_ids[reference_index],
            instance_id=None,
            target_source=source_reason,
        )
        if _valid_pseudo_mask_areas(
            updated.target_masks,
            reference_index=reference_index,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        ):
            return updated
        ratios = _mask_foreground_ratios(updated.target_masks)
        distance = float(torch.minimum((ratios - min_ratio).abs(), (ratios - max_ratio).abs()).mean().item())
        if best is None or distance < best_distance:
            best = (updated, ratios)
            best_distance = distance
    if best is not None and bool(pseudo.get("allow_out_of_range_fallback", False)):
        updated, ratios = best
        print(
            " ".join(
                [
                    "sam2_point_pseudo_mask_warning=area_out_of_range",
                    f"dataset={batch.dataset}",
                    f"scene={batch.scene_id}",
                    f"frames={list(batch.frame_ids)}",
                    f"ratios={[round(float(value), 6) for value in ratios.tolist()]}",
                    f"accepted_range=[{min_ratio},{max_ratio}]",
                ]
            )
        )
        return updated
    if last_error is not None:
        raise ExternalDependencyError("SAM2 point-prompt pseudo-mask generation failed") from last_error
    raise ExternalDependencyError("SAM2 point-prompt pseudo-mask generation produced no usable masks")


def _sam2_point_prompt_masks_for_batch(
    sam2_adapter: Sam2TrainingAdapter,
    batch: TrainingBatch,
    *,
    reference_index: int,
    point: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    track_method = getattr(sam2_adapter, "track_masks_from_points", None)
    if callable(track_method):
        try:
            return track_method(
                batch.images,
                reference_index=reference_index,
                points=point,
                labels=labels,
            )
        except ExternalDependencyError:
            pass
    predict_method = getattr(sam2_adapter, "predict_masks_from_points", None)
    if not callable(predict_method):
        raise ExternalDependencyError("SAM2 adapter does not expose point-prompt mask generation")
    points_by_frame = [point.detach().clone() for _ in range(int(batch.images.shape[0]))]
    labels_by_frame = [labels.detach().clone() for _ in range(int(batch.images.shape[0]))]
    return predict_method(batch.images, points_by_frame, labels_by_frame)


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
    step: int | None = None,
    visualization_dir: Path | None = None,
    visualization_max_frames: int = 4,
    visualization_max_side: int = 384,
    visualization_fps: int = 6,
) -> ValidationStepResult:
    was_training = wrapper.training
    wrapper.eval()
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    float_dtype = _training_float_dtype(device, amp_enabled)
    visualization: TrainingVisualizationArtifact | None = None
    with torch.no_grad():
        batch = dataset.sample().to(device, float_dtype=float_dtype)
        batch = _apply_sam2_point_pseudo_masks(batch, sam2_adapter=wrapper.sam2_adapter, config=config)
        must3r_features = _load_missing_must3r_features(
            batch,
            must3r_adapter,
            config,
            device,
            online_must3r=online_must3r,
            float_dtype=float_dtype,
        )
        with _autocast(device, amp_enabled):
            outputs = wrapper(batch, must3r_features)
        metrics = _tracking_metrics(outputs, batch)
        if visualization_dir is not None and step is not None:
            visualization = save_training_visualization(
                batch=batch,
                outputs=outputs,
                step=step,
                output_dir=visualization_dir,
                max_frames=visualization_max_frames,
                max_side=visualization_max_side,
                fps=visualization_fps,
                name_suffix="_validation",
            )
    if was_training:
        wrapper.train()
    return ValidationStepResult(metrics=metrics, visualization=visualization)


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


def _mask_bool(mask: torch.Tensor, *, threshold: float = 0.5) -> np.ndarray:
    return mask.detach().float().cpu().numpy() >= threshold


def _mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_edges(mask: np.ndarray) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) > 0
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) > 0
    return dilated ^ eroded


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int] | None, color: tuple[int, int, int]) -> None:
    if bbox is None:
        return
    draw.rectangle(bbox, outline=color, width=2)


def _draw_mask_contour(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    draw_bbox: bool = True,
) -> Image.Image:
    result = image.convert("RGB").copy()
    array = np.asarray(result).copy()
    edges = _mask_edges(mask)
    array[edges] = np.asarray(color, dtype=np.uint8)
    result = Image.fromarray(array)
    if draw_bbox:
        _draw_bbox(ImageDraw.Draw(result), _mask_bbox(mask), color)
    return result


def _draw_prompt_points(image: Image.Image, prompt: Prompt, frame_index: int) -> Image.Image:
    if prompt.type != "point" or int(prompt.frame_index) != frame_index or prompt.points is None:
        return image
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    labels = prompt.point_labels
    for point_index, point in enumerate(prompt.points.detach().float().cpu()):
        x, y = float(point[0]), float(point[1])
        label = int(labels[point_index].item()) if labels is not None and point_index < labels.numel() else 1
        color = (80, 180, 255) if label > 0 else (255, 180, 60)
        radius = 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 0, 0), width=3)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        if label > 0:
            draw.line((x - radius, y, x + radius, y), fill=color, width=2)
            draw.line((x, y - radius, x, y + radius), fill=color, width=2)
    return result


def _binary_mask_panel(
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    background: tuple[int, int, int] = (12, 12, 12),
) -> Image.Image:
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[:, :] = np.asarray(background, dtype=np.uint8)
    panel[mask.astype(bool)] = np.asarray(color, dtype=np.uint8)
    image = Image.fromarray(panel)
    image = _draw_mask_contour(image, mask, (255, 255, 255), draw_bbox=False)
    _draw_bbox(ImageDraw.Draw(image), _mask_bbox(mask), color)
    return image


def _comparison_panel(base: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    image = Image.fromarray(base)
    image = _draw_mask_contour(image, target, (0, 255, 80))
    image = _draw_mask_contour(image, prediction, (255, 70, 60))
    return image


def _error_panel(target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    if target.shape != prediction.shape:
        raise ValueError(f"target and prediction masks must have the same shape, got {target.shape} and {prediction.shape}")
    panel = np.zeros((*target.shape, 3), dtype=np.uint8)
    panel[:, :] = np.asarray((14, 14, 14), dtype=np.uint8)
    panel[target & prediction] = np.asarray((255, 220, 0), dtype=np.uint8)
    panel[target & ~prediction] = np.asarray((0, 220, 80), dtype=np.uint8)
    panel[~target & prediction] = np.asarray((255, 60, 40), dtype=np.uint8)
    return Image.fromarray(panel)


def _heatmap_panel(probability: torch.Tensor) -> Image.Image:
    prob = probability.detach().float().cpu().clamp(0, 1).numpy()
    red = (prob * 255.0).round()
    green = ((1.0 - np.abs(prob - 0.5) * 2.0) * 180.0).clip(0, 180).round()
    blue = ((1.0 - prob) * 255.0).round()
    heatmap = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(heatmap)


def _match_error_overlay(
    image: np.ndarray,
    target_mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    target = _mask_bool(target_mask, threshold=0.5)
    pred = _mask_bool(prediction, threshold=threshold)
    error = np.asarray(_error_panel(target, pred))
    background = image.astype(np.float32) * 0.2
    foreground = error.astype(np.float32) * 0.8
    mask = (target | pred)[..., None]
    return np.where(mask, foreground + background, background).round().clip(0, 255).astype(np.uint8)


def _binary_iou_by_frame(predictions: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must have the same shape, got {predictions.shape} and {targets.shape}")
    pred = predictions.detach().float().cpu() >= threshold
    target = targets.detach().float().cpu() > 0.5
    intersection = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


def _format_sequence(values: Sequence[Any], *, max_items: int = 8) -> str:
    items = [str(value) for value in values[:max_items]]
    if len(values) > max_items:
        items.append("...")
    return "[" + ",".join(items) + "]"


def _format_bool_flags(values: Sequence[bool], *, max_items: int = 16) -> str:
    items = ["1" if value else "0" for value in values[:max_items]]
    if len(values) > max_items:
        items.append("...")
    return "".join(items)


def _visualization_diagnostics(
    probabilities: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    has_object: torch.Tensor,
    reference_index: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    ious = _binary_iou_by_frame(probabilities, target_masks, threshold=threshold)
    predictions = probabilities.detach().float().cpu() >= threshold
    targets = target_masks.detach().float().cpu() > 0.5
    target_areas = targets.flatten(1).sum(dim=1).to(dtype=torch.int64)
    prediction_areas = predictions.flatten(1).sum(dim=1).to(dtype=torch.int64)
    visible = has_object.detach().cpu().bool()
    if visible.numel() != target_masks.shape[0]:
        raise ValueError(f"has_object must have one value per frame, got {tuple(has_object.shape)}")
    non_ref = torch.ones_like(visible, dtype=torch.bool)
    if non_ref.numel():
        reference_index = min(max(reference_index, 0), non_ref.numel() - 1)
        non_ref[reference_index] = False
    visible_non_ref = visible & non_ref
    empty_empty = (target_areas == 0) & (prediction_areas == 0)
    warnings: list[str] = []
    if int(visible.sum().item()) == 0:
        warnings.append("[WARN] all target masks are empty; batch_iou_empty_is_one can be 1.000 without tracking signal")
    elif not bool(visible[reference_index]):
        warnings.append("[WARN] reference/prompt mask is empty")
    if not bool(visible_non_ref.any()):
        warnings.append("[WARN] no non-reference visible target frames in visualization")
    return {
        "ious": ious,
        "batch_iou_empty_is_one": float(ious.mean().item()) if ious.numel() else None,
        "visible_iou": float(ious[visible].mean().item()) if bool(visible.any()) else None,
        "non_ref_visible_iou": float(ious[visible_non_ref].mean().item()) if bool(visible_non_ref.any()) else None,
        "ref_iou": float(ious[reference_index].item()) if ious.numel() else None,
        "tracking_recall": float(((ious > 0.0) & visible).float().sum().item() / visible.float().sum().item())
        if bool(visible.any())
        else None,
        "visible_frames": int(visible.sum().item()),
        "empty_empty_frames": int(empty_empty.sum().item()),
        "target_areas": [int(value) for value in target_areas.tolist()],
        "prediction_areas": [int(value) for value in prediction_areas.tolist()],
        "has_object_flags": [bool(value) for value in visible.tolist()],
        "warnings": warnings,
    }


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
    batch_iou_empty_is_one: float | None,
    visible_iou: float | None,
    non_ref_visible_iou: float | None,
    ref_iou: float | None,
    tracking_recall: float | None,
    visible_frames: int,
    empty_empty_frames: int,
    target_areas: Sequence[int] = (),
    has_object_flags: Sequence[bool] = (),
    instance_id: int | None = None,
    warnings: Sequence[str] = (),
    sampling_mode: str = "unknown",
    target_source: str = "dataset_mask",
) -> list[str]:
    joined_frame_ids = ", ".join(frame_ids)
    lines = [
        (
            f"3AM tracking batch | step={step} dataset={dataset} scene={scene_id} "
            f"sampling={sampling_mode} target_source={target_source}"
        ),
        (
            f"prompt={prompt_type} ref={reference_frame} frames=[{joined_frame_ids}] "
            f"batch_iou_empty_is_one={_format_metric(batch_iou_empty_is_one)} "
            f"visible_iou={_format_metric(visible_iou)} non_ref_visible_iou={_format_metric(non_ref_visible_iou)} "
            f"ref_iou={_format_metric(ref_iou)} tracking_recall_like={_format_metric(tracking_recall)}"
        ),
        (
            f"visible_frames={visible_frames}/{len(frame_ids)} empty_empty_frames={empty_empty_frames} "
            f"instance_id={instance_id if instance_id is not None else 'n/a'} "
            f"has_object={_format_bool_flags(has_object_flags)} target_areas={_format_sequence(target_areas)}"
        ),
        "panels: image contours | GT binary | prediction binary | contour compare | error map | confidence heatmap",
    ]
    lines.extend(warnings)
    return lines


def _visualization_frame_order(num_frames: int, reference_index: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    reference_index = min(max(reference_index, 0), num_frames - 1)
    limit = min(num_frames, max(1, max_frames))
    order = [reference_index]
    order.extend(index for index in range(reference_index + 1, num_frames))
    order.extend(index for index in range(reference_index - 1, -1, -1))
    return order[:limit]


def _visualization_video_frame_order(num_frames: int, reference_index: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    reference_index = min(max(reference_index, 0), num_frames - 1)
    limit = min(num_frames, max(1, max_frames))
    if limit >= num_frames:
        return list(range(num_frames))
    start = reference_index - limit // 2
    start = max(0, min(start, num_frames - limit))
    return list(range(start, start + limit))


def _resize_for_visualization(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)


def _bbox_text(mask: np.ndarray) -> str:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "bbox=none"
    return f"bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"


def _visualization_stem(*, batch: TrainingBatch, step: int) -> str:
    safe_scene = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in batch.scene_id)
    return f"step_{step:07d}_{batch.dataset}_{safe_scene}"


def _build_training_visualization_row(
    *,
    batch: TrainingBatch,
    images: torch.Tensor,
    probabilities: torch.Tensor,
    target_masks: torch.Tensor,
    ious: torch.Tensor,
    has_object: torch.Tensor,
    frame_index: int,
    reference_index: int,
    max_side: int,
) -> Image.Image:
    base = _tensor_image_to_uint8(images[frame_index])
    target_bool = _mask_bool(target_masks[frame_index], threshold=0.5)
    pred_bool = _mask_bool(probabilities[frame_index], threshold=0.5)
    image_panel = _draw_mask_contour(Image.fromarray(base), target_bool, (0, 255, 80))
    image_panel = _draw_mask_contour(image_panel, pred_bool, (255, 60, 40))
    image_panel = _draw_prompt_points(image_panel, batch.prompt, frame_index)
    gt_panel = _binary_mask_panel(target_bool, color=(0, 220, 80))
    pred_panel = _binary_mask_panel(pred_bool, color=(255, 60, 40))
    compare_panel = _comparison_panel(base, target_bool, pred_bool)
    error_panel = _error_panel(target_bool, pred_bool)
    heatmap_panel = _heatmap_panel(probabilities[frame_index])
    panels = [
        _resize_for_visualization(image_panel, max_side),
        _resize_for_visualization(gt_panel, max_side),
        _resize_for_visualization(pred_panel, max_side),
        _resize_for_visualization(compare_panel, max_side),
        _resize_for_visualization(error_panel, max_side),
        _resize_for_visualization(heatmap_panel, max_side),
    ]
    row_width = sum(panel.width for panel in panels)
    row_height = max(panel.height for panel in panels)
    label_height = 54
    row = Image.new("RGB", (row_width, row_height + label_height), (24, 24, 24))
    draw = ImageDraw.Draw(row)
    frame_id = batch.frame_ids[frame_index]
    is_reference = frame_index == reference_index
    visibility = "visible GT" if bool(has_object[frame_index]) else "absent GT"
    frame_label = f"image | frame {frame_id}"
    if is_reference:
        frame_label = f"image | REF/PROMPT {batch.prompt.type} {frame_id}"
    gt_area = _mask_area(target_bool)
    pred_area = _mask_area(pred_bool)
    overlap_area = _mask_area(target_bool & pred_bool)
    fp_area = _mask_area(~target_bool & pred_bool)
    fn_area = _mask_area(target_bool & ~pred_bool)
    labels = [
        frame_label,
        "GT binary",
        "SAM2/3AM prediction",
        "GT green + Pred red",
        "error map",
        "prediction confidence",
    ]
    sublabels = [
        f"IoU={ious[frame_index].item():.3f} | {visibility}",
        f"GT area={gt_area} | {_bbox_text(target_bool)}",
        f"Pred area={pred_area} | {_bbox_text(pred_bool)}",
        f"overlap={overlap_area} fp={fp_area} fn={fn_area}",
        f"yellow=ok green=missed red=extra",
        f"red=high blue=low",
    ]
    x = 0
    for panel, label, sublabel in zip(panels, labels, sublabels, strict=True):
        row.paste(panel, (x, label_height))
        fill = (255, 236, 120) if is_reference and "REF/PROMPT" in label else (235, 235, 235)
        draw.text((x + 4, 4), label, fill=fill)
        draw.text((x + 4, 20), sublabel[:56], fill=(205, 205, 205))
        if label.startswith("image"):
            prompt_note = " prompt point=blue" if batch.prompt.type == "point" and is_reference else ""
            draw.text((x + 4, 36), f"contours: GT=green Pred=red{prompt_note}"[:64], fill=(170, 210, 170))
        x += panel.width
    return row


def _render_training_visualization_canvas(header_lines: Sequence[str], rows: Sequence[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("training visualization requires at least one row")
    canvas_width = max(720, *(row.width for row in rows))
    header_height = max(58, 10 + len(header_lines) * 16)
    canvas_height = header_height + sum(row.height for row in rows)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(header_lines):
        fill = (255, 105, 80) if line.startswith("[WARN]") else (235, 235, 235)
        draw.text((8, 6 + line_index * 16), line[:180], fill=fill)
    y = 0
    for row in rows:
        canvas.paste(row, (0, header_height + y))
        y += row.height
    return canvas


def _write_training_visualization_video(frames: Sequence[Image.Image], output_path: Path, *, fps: int) -> Path | None:
    if not frames:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[WARN] ffmpeg is unavailable; skipping training visualization video export", flush=True)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="three_am_viz_") as temp_dir:
        temp_root = Path(temp_dir)
        for frame_index, frame in enumerate(frames):
            frame.save(temp_root / f"{frame_index:06d}.png")
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(max(1, fps)),
            "-i",
            str(temp_root / "%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(f"[WARN] ffmpeg failed to write training visualization video {output_path}: {error}", flush=True)
            return None
    return output_path


def _write_training_visualization_video_from_rows(
    header_lines: Sequence[str],
    row_paths: Sequence[Path],
    output_path: Path,
    *,
    fps: int,
) -> Path | None:
    if not row_paths:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[WARN] ffmpeg is unavailable; skipping training visualization video export", flush=True)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="three_am_viz_") as temp_dir:
        temp_root = Path(temp_dir)
        for frame_index, row_path in enumerate(row_paths):
            with Image.open(row_path) as row_image:
                canvas = _render_training_visualization_canvas(header_lines, [row_image.copy()])
            canvas.save(temp_root / f"{frame_index:06d}.png")
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(max(1, fps)),
            "-i",
            str(temp_root / "%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(f"[WARN] ffmpeg failed to write training visualization video {output_path}: {error}", flush=True)
            return None
    return output_path


def _clone_prompt_to_cpu(prompt: Prompt, *, frame_index: int) -> Prompt:
    return Prompt(
        type=prompt.type,
        frame_index=int(frame_index),
        mask=prompt.mask.detach().cpu().clone() if prompt.mask is not None else None,
        points=prompt.points.detach().cpu().clone() if prompt.points is not None else None,
        point_labels=prompt.point_labels.detach().cpu().clone() if prompt.point_labels is not None else None,
        box=prompt.box.detach().cpu().clone() if prompt.box is not None else None,
    )


def _target_mask_for_visualization(mask: np.ndarray, *, instance_id: int | None, binary_mode: bool) -> torch.Tensor:
    if binary_mode:
        return torch.from_numpy((mask > 0).astype(np.float32))
    if instance_id is None:
        raise ValueError("Full-scene visualization requires instance_id for non-binary masks")
    return torch.from_numpy((mask == int(instance_id)).astype(np.float32))


def _scene_for_visualization(dataset: ThreeAMTrainingDataset, batch: TrainingBatch):
    for scene in dataset.scenes:
        if scene.dataset == batch.dataset and scene.scene_id == batch.scene_id:
            return scene
    raise ValueError(f"Could not find scene record for {batch.dataset}/{batch.scene_id}")


def _visualization_scene_components(
    dataset: ThreeAMTrainingDataset,
    batch: TrainingBatch,
) -> tuple[Any, LetterboxTransform, list[Any], tuple[int, ...], bool, int | None, int]:
    scene = _scene_for_visualization(dataset, batch)
    letterbox = batch.letterbox_transform
    if letterbox is None:
        raise ValueError("Training batch is missing letterbox_transform for full-scene visualization")
    if not isinstance(letterbox, LetterboxTransform):
        raise TypeError(f"Unexpected letterbox transform payload: {type(letterbox)!r}")
    frames = list(scene.frames)
    if not frames:
        raise ValueError(f"Scene {scene.scene_id} has no frames for full-scene visualization")
    reference_frame_id = batch.reference_frame_id or (
        batch.frame_ids[min(max(int(batch.prompt.frame_index), 0), max(len(batch.frame_ids) - 1, 0))]
        if batch.frame_ids
        else ""
    )
    reference_index = next((index for index, frame in enumerate(frames) if frame.frame_id == reference_frame_id), -1)
    if reference_index < 0:
        raise ValueError(
            f"Reference frame {reference_frame_id!r} is not present in scene {scene.dataset}/{scene.scene_id}"
        )
    ignore_values = tuple(int(value) for value in dataset.config.get("datasets", {}).get(scene.dataset, {}).get("mask_ignore_values", ()))
    binary_mode = bool(batch.binary_mode) if batch.binary_mode is not None else scene.dataset == "shapenet"
    instance_id = 1 if binary_mode else batch.instance_id
    return scene, letterbox, frames, ignore_values, binary_mode, instance_id, reference_index


def _build_visualization_scene_batch(
    *,
    scene: Any,
    frames: Sequence[Any],
    selected_indices: Sequence[int],
    letterbox: LetterboxTransform,
    ignore_values: Sequence[int],
    instance_id: int | None,
    binary_mode: bool,
    prompt: Prompt,
    target_source: str,
) -> TrainingBatch:
    selected_frames = [frames[index] for index in selected_indices]
    if not selected_frames:
        raise ValueError("Visualization chunk requires at least one frame")
    raw_images = [load_image_tensor(frame.image_path, None) for frame in selected_frames]
    source_shape = tuple(raw_images[0].shape[-2:])
    if (letterbox.source_height, letterbox.source_width) != source_shape:
        raise ValueError(
            "Full-scene visualization source resolution does not match the sampled batch transform: "
            f"scene has {source_shape}, transform expects {(letterbox.source_height, letterbox.source_width)}"
        )
    images = torch.stack([letterbox.resize_image(image) for image in raw_images], dim=0)
    raw_masks = [load_mask_array(frame.mask_path, source_shape, ignore_values=tuple(ignore_values)) for frame in selected_frames]
    target_masks = torch.stack(
        [letterbox.resize_mask(_target_mask_for_visualization(mask, instance_id=instance_id, binary_mode=binary_mode)) for mask in raw_masks],
        dim=0,
    )
    has_object = target_masks.flatten(1).any(dim=1)
    return TrainingBatch(
        images=images,
        target_masks=target_masks,
        prompt=prompt,
        must3r_features=None,
        dataset=scene.dataset,
        scene_id=scene.scene_id,
        frame_ids=tuple(frame.frame_id for frame in selected_frames),
        image_paths=tuple(frame.image_path for frame in selected_frames),
        has_object=has_object,
        sampling_mode="full_scene",
        reference_frame_id=selected_frames[min(max(int(prompt.frame_index), 0), len(selected_frames) - 1)].frame_id,
        instance_id=None if binary_mode else instance_id,
        binary_mode=binary_mode,
        object_visibility=has_object,
        must3r_geometry=None,
        target_source=target_source,
        letterbox_transform=letterbox,
    )


def save_training_visualization(
    *,
    batch: TrainingBatch,
    outputs: dict[str, torch.Tensor],
    step: int,
    output_dir: Path,
    max_frames: int = 4,
    max_side: int = 384,
    fps: int = 6,
    name_suffix: str = "",
    video_frame_indices: Sequence[int] | None = None,
) -> TrainingVisualizationArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch.images.detach()
    target_masks = batch.target_masks.detach()
    logits = _mask_logits_for_visualization(outputs["mask_logits"].detach(), tuple(target_masks.shape[-2:]))
    probabilities = logits.sigmoid()
    num_frames = int(images.shape[0])
    reference_index = min(max(int(batch.prompt.frame_index), 0), max(num_frames - 1, 0))
    summary_order = _visualization_frame_order(num_frames, reference_index, max_frames)
    video_order = (
        [int(index) for index in video_frame_indices if 0 <= int(index) < num_frames]
        if video_frame_indices is not None
        else _visualization_video_frame_order(num_frames, reference_index, max_frames)
    )
    if not video_order:
        video_order = _visualization_video_frame_order(num_frames, reference_index, max_frames)
    diagnostics = _visualization_diagnostics(
        probabilities,
        target_masks,
        has_object=batch.has_object,
        reference_index=reference_index,
    )
    ious = diagnostics["ious"]
    has_object = batch.has_object.detach().cpu().bool()
    selected_indices = sorted(set(summary_order) | set(video_order))
    rows = {
        frame_index: _build_training_visualization_row(
            batch=batch,
            images=images,
            probabilities=probabilities,
            target_masks=target_masks,
            ious=ious,
            has_object=has_object,
            frame_index=frame_index,
            reference_index=reference_index,
            max_side=max_side,
        )
        for frame_index in selected_indices
    }
    reference_frame = batch.frame_ids[reference_index] if 0 <= reference_index < len(batch.frame_ids) else str(reference_index)
    header_lines = _training_visualization_header_lines(
        step=step,
        dataset=batch.dataset,
        scene_id=batch.scene_id,
        prompt_type=batch.prompt.type,
        reference_frame=reference_frame,
        frame_ids=batch.frame_ids,
        batch_iou_empty_is_one=diagnostics["batch_iou_empty_is_one"],
        visible_iou=diagnostics["visible_iou"],
        non_ref_visible_iou=diagnostics["non_ref_visible_iou"],
        ref_iou=diagnostics["ref_iou"],
        tracking_recall=diagnostics["tracking_recall"],
        visible_frames=diagnostics["visible_frames"],
        empty_empty_frames=diagnostics["empty_empty_frames"],
        target_areas=diagnostics["target_areas"],
        has_object_flags=diagnostics["has_object_flags"],
        instance_id=batch.instance_id,
        warnings=diagnostics["warnings"],
        sampling_mode=batch.sampling_mode,
        target_source=batch.target_source,
    )
    stem = f"{_visualization_stem(batch=batch, step=step)}{name_suffix}"
    summary_path = output_dir / f"{stem}.png"
    summary_canvas = _render_training_visualization_canvas(header_lines, [rows[index] for index in summary_order])
    summary_canvas.save(summary_path)
    video_frames = [_render_training_visualization_canvas(header_lines, [rows[index]]) for index in video_order]
    video_path = _write_training_visualization_video(
        video_frames,
        output_dir / f"{stem}.mp4",
        fps=max(1, fps),
    )
    return TrainingVisualizationArtifact(summary_path=summary_path, video_path=video_path)


def _diagnostics_from_frame_statistics(
    *,
    ious: Sequence[float],
    target_areas: Sequence[int],
    prediction_areas: Sequence[int],
    has_object_flags: Sequence[bool],
    reference_index: int,
) -> dict[str, Any]:
    ious_tensor = torch.tensor(list(ious), dtype=torch.float32)
    target_areas_tensor = torch.tensor(list(target_areas), dtype=torch.int64)
    prediction_areas_tensor = torch.tensor(list(prediction_areas), dtype=torch.int64)
    visible = torch.tensor(list(has_object_flags), dtype=torch.bool)
    if visible.numel() != ious_tensor.numel():
        raise ValueError("Frame statistics must have matching lengths")
    non_ref = torch.ones_like(visible, dtype=torch.bool)
    if non_ref.numel():
        reference_index = min(max(int(reference_index), 0), non_ref.numel() - 1)
        non_ref[reference_index] = False
    visible_non_ref = visible & non_ref
    empty_empty = (target_areas_tensor == 0) & (prediction_areas_tensor == 0)
    warnings: list[str] = []
    if int(visible.sum().item()) == 0:
        warnings.append("[WARN] all target masks are empty; batch_iou_empty_is_one can be 1.000 without tracking signal")
    elif not bool(visible[reference_index]):
        warnings.append("[WARN] reference/prompt mask is empty")
    if not bool(visible_non_ref.any()):
        warnings.append("[WARN] no non-reference visible target frames in visualization")
    return {
        "ious": ious_tensor,
        "batch_iou_empty_is_one": float(ious_tensor.mean().item()) if ious_tensor.numel() else None,
        "visible_iou": float(ious_tensor[visible].mean().item()) if bool(visible.any()) else None,
        "non_ref_visible_iou": float(ious_tensor[visible_non_ref].mean().item()) if bool(visible_non_ref.any()) else None,
        "ref_iou": float(ious_tensor[reference_index].item()) if ious_tensor.numel() else None,
        "tracking_recall": float(((ious_tensor > 0.0) & visible).float().sum().item() / visible.float().sum().item())
        if bool(visible.any())
        else None,
        "visible_frames": int(visible.sum().item()),
        "empty_empty_frames": int(empty_empty.sum().item()),
        "target_areas": [int(value) for value in target_areas_tensor.tolist()],
        "prediction_areas": [int(value) for value in prediction_areas_tensor.tolist()],
        "has_object_flags": [bool(value) for value in visible.tolist()],
        "warnings": warnings,
    }


def _chunk_visualization_ranges_forward(reference_index: int, num_frames: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = int(reference_index)
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        ranges.append((start, end))
        if end >= num_frames:
            break
        start = end - 1
    return ranges


def _chunk_visualization_ranges_backward(reference_index: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    end = int(reference_index) + 1
    while end > 0:
        start = max(0, end - chunk_size)
        ranges.append((start, end))
        if start == 0:
            break
        end = start + 1
    return ranges


def _sample_full_scene_visualization_indices(
    *,
    num_frames: int,
    reference_index: int,
    source_fps: float | None,
    sample_fps: float | None,
) -> list[int]:
    if num_frames <= 0:
        return []
    reference_index = min(max(int(reference_index), 0), num_frames - 1)
    if sample_fps is None or sample_fps <= 0 or source_fps is None or source_fps <= 0 or sample_fps >= source_fps:
        indices = set(range(num_frames))
    else:
        stride = max(1, int(round(float(source_fps) / float(sample_fps))))
        indices = set(range(0, num_frames, stride))
        indices.add(num_frames - 1)
    indices.add(0)
    indices.add(reference_index)
    indices.add(num_frames - 1)
    return sorted(index for index in indices if 0 <= index < num_frames)


def _global_row_path(temp_dir: Path, global_index: int) -> Path:
    return temp_dir / f"row_{global_index:06d}.png"


def save_full_scene_training_visualization(
    *,
    dataset: ThreeAMTrainingDataset,
    batch: TrainingBatch,
    wrapper: ThreeAMTrainingWrapper,
    must3r_adapter: Must3rFeatureAdapter | None,
    config: dict[str, Any],
    device: torch.device,
    online_must3r: bool,
    float_dtype: torch.dtype,
    amp_enabled: bool,
    step: int,
    output_dir: Path,
    max_frames: int = 4,
    max_side: int = 384,
    fps: int = 6,
    chunk_size: int = 32,
    source_fps: float | None = None,
    sample_fps: float | None = None,
) -> TrainingVisualizationArtifact:
    scene, letterbox, frames, ignore_values, binary_mode, instance_id, reference_index = _visualization_scene_components(
        dataset,
        batch,
    )
    chunk_size = max(2, int(chunk_size))
    original_total_frames = len(frames)
    sampled_indices = _sample_full_scene_visualization_indices(
        num_frames=original_total_frames,
        reference_index=reference_index,
        source_fps=source_fps,
        sample_fps=sample_fps,
    )
    if not sampled_indices:
        raise ValueError("Full-scene visualization has no sampled frames")
    reference_index = sampled_indices.index(reference_index)
    frames = [frames[index] for index in sampled_indices]
    total_frames = len(frames)
    summary_order = _visualization_frame_order(total_frames, reference_index, max_frames)
    row_metrics_size = total_frames
    ious_by_index: list[float | None] = [None] * row_metrics_size
    target_areas_by_index: list[int | None] = [None] * row_metrics_size
    prediction_areas_by_index: list[int | None] = [None] * row_metrics_size
    has_object_by_index: list[bool | None] = [None] * row_metrics_size
    original_prompt = _clone_prompt_to_cpu(batch.prompt, frame_index=reference_index)
    was_training = wrapper.training
    wrapper.eval()
    with tempfile.TemporaryDirectory(prefix="three_am_full_scene_rows_") as temp_dir:
        row_dir = Path(temp_dir)

        def run_chunk(global_indices: list[int], prompt_mask: torch.Tensor, *, local_reference_index: int) -> torch.Tensor:
            chunk_prompt = Prompt(type="mask", frame_index=local_reference_index, mask=prompt_mask.detach().cpu().clone())
            chunk_batch_cpu = _build_visualization_scene_batch(
                scene=scene,
                frames=frames,
                selected_indices=global_indices,
                letterbox=letterbox,
                ignore_values=ignore_values,
                instance_id=instance_id,
                binary_mode=binary_mode,
                prompt=chunk_prompt,
                target_source=batch.target_source,
            )
            if dataset.load_feature_cache:
                selected_frames = [frames[index] for index in global_indices]
                chunk_features = dataset.feature_cache.load(scene, selected_frames)
                chunk_batch_cpu = replace(
                    chunk_batch_cpu,
                    must3r_features=chunk_features,
                    must3r_geometry=chunk_features if isinstance(chunk_features, Must3rFeatureBundle) else None,
                )
            if global_indices[local_reference_index] == reference_index:
                render_prompt = _clone_prompt_to_cpu(original_prompt, frame_index=local_reference_index)
                render_reference_index = local_reference_index
            else:
                render_prompt = Prompt(type=chunk_prompt.type, frame_index=-1, mask=chunk_prompt.mask)
                render_reference_index = -1
            render_batch = replace(chunk_batch_cpu, prompt=render_prompt)
            chunk_batch = chunk_batch_cpu.to(device, float_dtype=float_dtype)
            must3r_features = _load_missing_must3r_features(
                chunk_batch,
                must3r_adapter,
                config,
                device,
                online_must3r=online_must3r,
                float_dtype=float_dtype,
            )
            with torch.no_grad():
                with _autocast(device, amp_enabled):
                    outputs = wrapper(chunk_batch, must3r_features)
            probabilities = _mask_logits_for_visualization(
                outputs["mask_logits"].detach(),
                tuple(chunk_batch.target_masks.shape[-2:]),
            ).sigmoid().cpu()
            diagnostics = _visualization_diagnostics(
                probabilities,
                chunk_batch.target_masks.detach().cpu(),
                has_object=chunk_batch.has_object.detach().cpu(),
                reference_index=local_reference_index if render_reference_index >= 0 else 0,
            )
            local_ious = diagnostics["ious"].cpu()
            local_targets = chunk_batch.target_masks.detach().cpu()
            local_visible = chunk_batch.has_object.detach().cpu().bool()
            thresholded = probabilities >= 0.5
            target_areas = local_targets.flatten(1).sum(dim=1).to(dtype=torch.int64)
            prediction_areas = thresholded.flatten(1).sum(dim=1).to(dtype=torch.int64)
            for local_index, global_index in enumerate(global_indices):
                row = _build_training_visualization_row(
                    batch=render_batch,
                    images=chunk_batch.images.detach().cpu(),
                    probabilities=probabilities,
                    target_masks=local_targets,
                    ious=local_ious,
                    has_object=local_visible,
                    frame_index=local_index,
                    reference_index=render_reference_index,
                    max_side=max_side,
                )
                row.save(_global_row_path(row_dir, global_index))
                ious_by_index[global_index] = float(local_ious[local_index].item())
                target_areas_by_index[global_index] = int(target_areas[local_index].item())
                prediction_areas_by_index[global_index] = int(prediction_areas[local_index].item())
                has_object_by_index[global_index] = bool(local_visible[local_index].item())
            return thresholded.float()

        prompt_mask = original_prompt.mask
        if prompt_mask is None:
            raise ValueError("Full-scene ShapeNet visualization currently requires a mask prompt")
        for start, end in _chunk_visualization_ranges_forward(reference_index, total_frames, chunk_size):
            indices = list(range(start, end))
            predicted = run_chunk(indices, prompt_mask, local_reference_index=0)
            prompt_mask = predicted[-1]
        prompt_mask = original_prompt.mask
        for chunk_index, (start, end) in enumerate(_chunk_visualization_ranges_backward(reference_index, chunk_size)):
            indices = list(range(start, end))
            predicted = run_chunk(indices, prompt_mask, local_reference_index=len(indices) - 1)
            prompt_mask = predicted[0]
            if chunk_index == 0:
                continue

        if was_training:
            wrapper.train()

        if any(value is None for value in ious_by_index + target_areas_by_index + prediction_areas_by_index + has_object_by_index):
            raise RuntimeError("Chunked full-scene visualization did not cover every frame")
        diagnostics = _diagnostics_from_frame_statistics(
            ious=[float(value) for value in ious_by_index if value is not None],
            target_areas=[int(value) for value in target_areas_by_index if value is not None],
            prediction_areas=[int(value) for value in prediction_areas_by_index if value is not None],
            has_object_flags=[bool(value) for value in has_object_by_index if value is not None],
            reference_index=reference_index,
        )
        warnings = list(diagnostics["warnings"])
        if total_frames < original_total_frames:
            warnings.append(
                "[INFO] full-scene visualization sampled "
                f"{total_frames}/{original_total_frames} frames "
                f"(source_fps={source_fps}, sample_fps={sample_fps})"
            )
        frame_ids = tuple(frame.frame_id for frame in frames)
        reference_frame = frame_ids[reference_index] if 0 <= reference_index < len(frame_ids) else str(reference_index)
        header_lines = _training_visualization_header_lines(
            step=step,
            dataset=scene.dataset,
            scene_id=scene.scene_id,
            prompt_type=original_prompt.type,
            reference_frame=reference_frame,
            frame_ids=frame_ids,
            batch_iou_empty_is_one=diagnostics["batch_iou_empty_is_one"],
            visible_iou=diagnostics["visible_iou"],
            non_ref_visible_iou=diagnostics["non_ref_visible_iou"],
            ref_iou=diagnostics["ref_iou"],
            tracking_recall=diagnostics["tracking_recall"],
            visible_frames=diagnostics["visible_frames"],
            empty_empty_frames=diagnostics["empty_empty_frames"],
            target_areas=diagnostics["target_areas"],
            has_object_flags=diagnostics["has_object_flags"],
            instance_id=batch.instance_id,
            warnings=warnings,
            sampling_mode="full_scene_chunked_sampled" if total_frames < original_total_frames else "full_scene_chunked",
            target_source=batch.target_source,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_visualization_stem(batch=batch, step=step)}_full_scene"
        summary_path = output_dir / f"{stem}.png"
        summary_rows: list[Image.Image] = []
        for index in summary_order:
            with Image.open(_global_row_path(row_dir, index)) as row_image:
                summary_rows.append(row_image.copy())
        summary_canvas = _render_training_visualization_canvas(header_lines, summary_rows)
        summary_canvas.save(summary_path)
        video_path = _write_training_visualization_video_from_rows(
            header_lines,
            [_global_row_path(row_dir, index) for index in range(total_frames)],
            output_dir / f"{stem}.mp4",
            fps=max(1, fps),
        )
    if was_training:
        wrapper.train()
    return TrainingVisualizationArtifact(summary_path=summary_path, video_path=video_path)


def run_training(
    config_path: str,
    *,
    iterations: int | None = None,
    resume: str | Path | None = None,
    device_name: str | None = None,
    feature_cache: str | Path | None = None,
    log_every: int | None = None,
    checkpoint_every: int | None = None,
    validate_every: int | None = None,
    validation_visualize_every: int | None = None,
    visualize_every: int | None = None,
    visualization_dir: str | Path | None = None,
    visualization_fps: int | None = None,
    visualization_full_scene_source_fps: float | None = None,
    visualization_full_scene_sample_fps: float | None = None,
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
    visualization_full_video = bool(training_config.get("visualization_full_video", False))
    visualization_chunk_size = int(training_config.get("visualization_chunk_size", 32))
    visualization_full_scene_source_fps = (
        visualization_full_scene_source_fps
        if visualization_full_scene_source_fps is not None
        else training_config.get("visualization_full_scene_source_fps")
    )
    visualization_full_scene_sample_fps = (
        visualization_full_scene_sample_fps
        if visualization_full_scene_sample_fps is not None
        else training_config.get("visualization_full_scene_sample_fps")
    )
    visualization_full_scene_source_fps = (
        None if visualization_full_scene_source_fps is None else float(visualization_full_scene_source_fps)
    )
    visualization_full_scene_sample_fps = (
        None if visualization_full_scene_sample_fps is None else float(visualization_full_scene_sample_fps)
    )
    validate_every = int(validate_every if validate_every is not None else training_config.get("validate_every", 0))
    validation_visualize_every = int(
        validation_visualize_every
        if validation_visualize_every is not None
        else training_config.get("validation_visualize_every", 0)
    )
    resolved_visualization_dir = _resolve_visualization_dir(config, visualization_dir)
    visualization_max_frames = int(training_config.get("visualization_max_frames", 4))
    visualization_max_side = int(training_config.get("visualization_max_side", 384))
    resolved_visualization_fps = int(
        visualization_fps if visualization_fps is not None else training_config.get("visualization_fps", 6)
    )
    device = _device(device_name)
    all_scenes = load_training_scenes(config, configured_training_datasets(config))
    validation_fraction = float(training_config.get("validation_fraction", 0.1 if validate_every > 0 else 0.0))
    validation_seed = int(training_config.get("validation_seed", 0))
    validation_scene_ids = tuple(str(value) for value in training_config.get("validation_scenes", ()))
    train_scenes, validation_scenes = _split_training_scenes(
        all_scenes,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        validation_scene_ids=validation_scene_ids,
    )
    print(
        " ".join(
            [
                "training_split",
                f"total_scenes={len(all_scenes)}",
                f"train_scenes={len(train_scenes)}",
                f"validation_scenes={len(validation_scenes)}",
                f"validation_fraction={validation_fraction:.6f}",
                f"validation_seed={validation_seed}",
                f"validate_every={validate_every}",
                f"validation_visualize_every={validation_visualize_every}",
            ]
        )
    )
    feature_cache_root = configured_feature_cache_root(config, feature_cache)
    dataset = ThreeAMTrainingDataset(
        train_scenes,
        config,
        feature_cache_root=feature_cache_root,
        load_feature_cache=not online_enabled,
    )
    validation_dataset = (
        ThreeAMTrainingDataset(
            validation_scenes,
            config,
            feature_cache_root=feature_cache_root,
            load_feature_cache=not online_enabled,
            rng=random.Random(validation_seed),
        )
        if validate_every > 0 and validation_scenes
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
    training_float_dtype = _training_float_dtype(device, amp_enabled)
    scaler = _grad_scaler(device, amp_enabled and training_float_dtype == torch.float16)
    cuda_memory_debugger = _build_cuda_memory_debugger(device, bool(training_config.get("cuda_memory_debug", False)))
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
    max_detached_loss_resample_attempts = int(training_config.get("max_detached_loss_resample_attempts", 10))
    for step_index in range(start_step, total_iterations):
        step = step_index + 1
        cuda_memory_debugger.start_step()
        outputs: dict[str, torch.Tensor] | None = None
        loss = None
        batch: TrainingBatch | None = None
        must3r_features: tuple[torch.Tensor, ...] | Must3rFeatureBundle | None = None
        for attempt in range(max(1, max_detached_loss_resample_attempts)):
            batch = dataset.sample().to(device, float_dtype=training_float_dtype)
            cuda_memory_debugger.snapshot("after_batch_to_device")
            batch = _apply_sam2_point_pseudo_masks(batch, sam2_adapter=wrapper.sam2_adapter, config=config)
            must3r_features = _load_missing_must3r_features(
                batch,
                must3r_adapter,
                config,
                device,
                online_must3r=online_enabled,
                float_dtype=training_float_dtype,
            )
            cuda_memory_debugger.snapshot("after_must3r_ready")
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_enabled):
                outputs = wrapper(batch, must3r_features)
                loss = sam2_training_loss(outputs, batch.target_masks, batch.has_object, weights)
            cuda_memory_debugger.snapshot("after_forward")
            if _loss_is_differentiable(loss.total):
                break
            if attempt + 1 < max_detached_loss_resample_attempts:
                print(
                    " ".join(
                        [
                            f"step={step}",
                            f"detached_loss_resample={attempt + 1}",
                            f"dataset={batch.dataset}",
                            f"scene={batch.scene_id}",
                            f"frames={list(batch.frame_ids)}",
                            f"has_object={batch.has_object.detach().cpu().tolist()}",
                            f"prompt_index={int(batch.prompt.frame_index)}",
                            f"sampling={batch.sampling_mode}",
                            f"target_source={batch.target_source}",
                        ]
                    )
                )
                continue
        if outputs is None or loss is None or batch is None:
            raise RuntimeError("Training loop failed to produce a batch")
        _ensure_loss_is_differentiable(loss.total, wrapper=wrapper, outputs=outputs, batch=batch)
        scaler.scale(loss.total).backward()
        scaler.step(optimizer)
        scaler.update()
        cuda_memory_debugger.snapshot("after_backward")
        last_step = step
        if visualize_every > 0 and (step == 1 or step % visualize_every == 0):
            visualization = save_training_visualization(
                batch=batch,
                outputs=outputs,
                step=step,
                output_dir=resolved_visualization_dir,
                max_frames=visualization_max_frames,
                max_side=visualization_max_side,
                fps=resolved_visualization_fps,
            )
            parts = [f"visualization={visualization.video_path or visualization.summary_path}"]
            parts.append(f"visualization_summary={visualization.summary_path}")
            if visualization.video_path is not None:
                parts.append(f"visualization_video={visualization.video_path}")
            if visualization_full_video and batch.dataset == "shapenet":
                full_scene_visualization = save_full_scene_training_visualization(
                    dataset=dataset,
                    batch=batch,
                    wrapper=wrapper,
                    must3r_adapter=must3r_adapter,
                    config=config,
                    device=device,
                    online_must3r=online_enabled,
                    float_dtype=training_float_dtype,
                    amp_enabled=amp_enabled,
                    step=step,
                    output_dir=resolved_visualization_dir,
                    max_frames=visualization_max_frames,
                    max_side=visualization_max_side,
                    fps=resolved_visualization_fps,
                    chunk_size=visualization_chunk_size,
                    source_fps=visualization_full_scene_source_fps,
                    sample_fps=visualization_full_scene_sample_fps,
                )
                parts.append(f"visualization_full_scene_summary={full_scene_visualization.summary_path}")
                if full_scene_visualization.video_path is not None:
                    parts.append(f"visualization_full_scene_video={full_scene_visualization.video_path}")
            print(" ".join(parts))
        should_log = cuda_memory_debugger.enabled or (log_every > 0 and (step == 1 or step % log_every == 0))
        if should_log:
            items = loss.detached_items()
            elapsed = max(time.time() - started, 1e-6)
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"dataset={batch.dataset}",
                        f"scene={batch.scene_id}",
                        f"target_source={batch.target_source}",
                        f"fps={step / elapsed:.3f}",
                        *(f"{key}={value:.3f}" for key, value in cuda_memory_debugger.fields().items()),
                        *(f"{key}={value:.6f}" for key, value in items.items()),
                    ]
                )
            )
        clear_cached_backbone_payload = getattr(wrapper.sam2_adapter, "clear_cached_backbone_payload", None)
        if callable(clear_cached_backbone_payload):
            clear_cached_backbone_payload()
        del outputs, loss, batch, must3r_features
        if validation_dataset is not None and validate_every > 0 and step % validate_every == 0:
            visualize_validation = validation_visualize_every > 0 and step % validation_visualize_every == 0
            validation_result = run_validation_step(
                wrapper=wrapper,
                dataset=validation_dataset,
                must3r_adapter=must3r_adapter,
                config=config,
                device=device,
                online_must3r=online_enabled,
                step=step if visualize_validation else None,
                visualization_dir=resolved_visualization_dir if visualize_validation else None,
                visualization_max_frames=visualization_max_frames,
                visualization_max_side=visualization_max_side,
                visualization_fps=resolved_visualization_fps,
            )
            validation_parts = [
                f"validation_step={step}",
                *(f"{key}={value:.6f}" for key, value in validation_result.metrics.items()),
            ]
            if validation_result.visualization is not None:
                validation_parts.append(f"validation_visualization_summary={validation_result.visualization.summary_path}")
                if validation_result.visualization.video_path is not None:
                    validation_parts.append(f"validation_visualization_video={validation_result.visualization.video_path}")
            print(" ".join(validation_parts))
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
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--validation-visualize-every", type=int, default=None)
    parser.add_argument("--save-model-every", type=int, default=None)
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--visualization-dir", default=None)
    parser.add_argument("--visualization-fps", type=int, default=None)
    parser.add_argument("--visualization-full-scene-source-fps", type=float, default=None)
    parser.add_argument("--visualization-full-scene-sample-fps", type=float, default=None)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-paper", action="store_true", dest="strict_paper", default=None)
    strict_group.add_argument("--no-strict-paper", action="store_false", dest="strict_paper")
    online_group = parser.add_mutually_exclusive_group()
    online_group.add_argument(
        "--online-must3r",
        action="store_true",
        default=None,
        help="compute MUSt3R features inside the training loop instead of loading feature cache files",
    )
    online_group.add_argument(
        "--offline-must3r",
        action="store_false",
        dest="online_must3r",
        default=None,
        help="load MUSt3R feature cache files even when features.online=true in the config",
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
        validate_every=args.validate_every,
        validation_visualize_every=args.validation_visualize_every,
        save_model_every=args.save_model_every,
        model_out=args.model_out,
        auto_resume=args.auto_resume,
        visualize_every=args.visualize_every,
        visualization_dir=args.visualization_dir,
        visualization_fps=args.visualization_fps,
        visualization_full_scene_source_fps=args.visualization_full_scene_source_fps,
        visualization_full_scene_sample_fps=args.visualization_full_scene_sample_fps,
        dry_run_only=args.dry_run,
        online_must3r=args.online_must3r,
        strict_paper=args.strict_paper,
    )


if __name__ == "__main__":
    main()
