#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import importlib.util
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from three_am.models.adapters import Must3rFeatureAdapter, Sam2TrainingAdapter
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.training.dataset import Prompt, TrainingBatch, _letterbox_transform, load_image_tensor, load_mask_array
from three_am.utils.config import load_yaml


def _load_train_3am_module():
    script_path = Path(__file__).resolve().with_name("train_3am.py")
    spec = importlib.util.spec_from_file_location("train_3am_infer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TRAIN_3AM = _load_train_3am_module()
ThreeAMTrainingWrapper = _TRAIN_3AM.TrainingWrapper if hasattr(_TRAIN_3AM, "TrainingWrapper") else _TRAIN_3AM.ThreeAMTrainingWrapper
_device = _TRAIN_3AM._device
_extract_online_must3r_features = _TRAIN_3AM._extract_online_must3r_features
_load_model_state = _TRAIN_3AM._load_model_state
_load_torch = _TRAIN_3AM._load_torch
build_core = _TRAIN_3AM.build_core
external_config = _TRAIN_3AM.external_config
mark_trainable_3am_modules = _TRAIN_3AM.mark_trainable_3am_modules

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _frame_paths(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        frames = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)
        if not frames:
            raise ValueError(f"{path} does not contain image frames")
        return frames
    return _extract_video_frames(path)


def _extract_video_frames(path: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required when --video points to a video file. Pass a frame directory instead.")
    temp_dir = Path(tempfile.mkdtemp(prefix="three_am_infer_frames_"))
    pattern = temp_dir / "%06d.png"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), str(pattern)]
    subprocess.run(command, check=True)
    frames = sorted(temp_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from {path}")
    return frames


def _load_images(paths: Sequence[Path], image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    raw_images = [load_image_tensor(path, None) for path in paths]
    if not raw_images:
        raise ValueError("No frames to load")
    source_shape = tuple(raw_images[0].shape[-2:])
    if any(tuple(image.shape[-2:]) != source_shape for image in raw_images):
        raise ValueError("All inference frames must share one source resolution for SAM2 letterbox inference")
    transform = _letterbox_transform(
        source_width=source_shape[1],
        source_height=source_shape[0],
        target_size=image_size,
    )
    images = torch.stack([transform.resize_image(image) for image in raw_images], dim=0)
    return images, source_shape


def _load_reference_mask(path: Path, source_shape: tuple[int, int], image_size: int) -> torch.Tensor:
    transform = _letterbox_transform(
        source_width=source_shape[1],
        source_height=source_shape[0],
        target_size=image_size,
    )
    mask = torch.from_numpy((load_mask_array(path, source_shape) > 0).astype(np.float32))
    mask = transform.resize_mask(mask)
    if not bool(mask.any()):
        raise ValueError(f"Reference mask is empty after letterbox resizing: {path}")
    return mask


def _load_features(
    adapter: Must3rFeatureAdapter,
    batch: TrainingBatch,
    device: torch.device,
) -> tuple[torch.Tensor, ...] | Must3rFeatureBundle:
    if getattr(adapter, "model", object()) is None:
        adapter.load()
    features = _extract_online_must3r_features(adapter, batch)
    if isinstance(features, Must3rFeatureBundle):
        return features.to(device)
    return tuple(feature.to(device) for feature in features)


def _load_wrapper(config: dict[str, Any], checkpoint: Path, device: torch.device) -> ThreeAMTrainingWrapper:
    core = build_core(config)
    sam2_adapter = Sam2TrainingAdapter(external_config(config))
    sam2_adapter.load()
    wrapper = ThreeAMTrainingWrapper(
        core,
        sam2_adapter,
        strict_paper=bool(config.get("training", {}).get("strict_paper", False)),
    ).to(device)
    mark_trainable_3am_modules(wrapper)
    payload = _load_torch(checkpoint, map_location=device)
    _load_model_state(wrapper, payload)
    wrapper.eval()
    return wrapper


def _save_masks(logits: torch.Tensor, output_dir: Path, *, threshold: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = logits.detach().float().sigmoid().cpu()
    paths: list[Path] = []
    for index, probability in enumerate(probabilities):
        mask = (probability >= threshold).numpy().astype(np.uint8) * 255
        path = output_dir / f"{index:06d}.png"
        Image.fromarray(mask).save(path)
        paths.append(path)
    return paths


def _write_video(mask_dir: Path, output_path: Path, *, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write --output-video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(mask_dir / "%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    device = _device(args.device)
    image_size = int(args.image_size or config.get("model", {}).get("sam_image_size", 1024))
    video_path = Path(args.video).expanduser()
    frame_paths = _frame_paths(video_path)
    if args.start_frame is not None or args.num_frames is not None:
        start = max(0, int(args.start_frame or 0))
        end = len(frame_paths) if args.num_frames is None else min(len(frame_paths), start + int(args.num_frames))
        frame_paths = frame_paths[start:end]
    if not frame_paths:
        raise ValueError("No frames selected for inference")
    reference_index = int(args.reference_index)
    if reference_index < 0 or reference_index >= len(frame_paths):
        raise ValueError(f"--reference-index must be in [0, {len(frame_paths) - 1}]")

    images, source_shape = _load_images(frame_paths, image_size)
    images = images.to(device)
    reference_mask = _load_reference_mask(Path(args.reference_mask).expanduser(), source_shape, image_size).to(device)
    output_root = Path(args.output_dir).expanduser()
    mask_dir = output_root / "masks"
    prompt = Prompt(type="mask", frame_index=reference_index, mask=reference_mask)
    batch = TrainingBatch(
        images=images,
        target_masks=torch.zeros(images.shape[0], images.shape[-2], images.shape[-1], device=device),
        prompt=prompt,
        must3r_features=None,
        dataset="inference",
        scene_id=video_path.stem,
        frame_ids=tuple(path.stem for path in frame_paths),
        image_paths=tuple(path.resolve() for path in frame_paths),
        has_object=torch.zeros(images.shape[0], dtype=torch.bool, device=device),
        reference_frame_id=frame_paths[reference_index].stem,
    )
    wrapper = _load_wrapper(config, Path(args.checkpoint).expanduser(), device)
    must3r_adapter = Must3rFeatureAdapter(external_config(config, must3r_device=str(device)))
    with torch.no_grad():
        sam_features = wrapper.sam2_adapter.encode_sam_features(images)
        must3r_features = _load_features(must3r_adapter, batch, device)
        merged = wrapper.core.forward_features(sam_features, must3r_features)
        backbone_out = wrapper.sam2_adapter._inject_merged_features(merged)
        logits = wrapper.sam2_adapter.track_masks_from_mask(
            images,
            reference_index=reference_index,
            mask=reference_mask,
            backbone_out=backbone_out,
        )
    mask_paths = _save_masks(logits, mask_dir, threshold=float(args.threshold))
    if args.output_video:
        _write_video(mask_dir, Path(args.output_video).expanduser(), fps=int(args.fps))
    summary = {
        "frames": len(frame_paths),
        "reference_index": reference_index,
        "reference_frame": str(frame_paths[reference_index]),
        "reference_mask": str(Path(args.reference_mask).expanduser()),
        "output_dir": str(output_root),
        "mask_dir": str(mask_dir),
        "mask_paths": [str(path) for path in mask_paths],
    }
    if args.summary:
        summary_path = Path(args.summary).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Track one object through a video using 3AM and a reference mask prompt")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    parser.add_argument("--checkpoint", required=True, help="3AM model weights or training checkpoint")
    parser.add_argument("--video", required=True, help="video file or directory of image frames")
    parser.add_argument("--reference-index", type=int, required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=24)
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()
