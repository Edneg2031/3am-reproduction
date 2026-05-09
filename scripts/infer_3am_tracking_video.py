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
from torch.nn import functional as F

from three_am.models.adapters import Must3rFeatureAdapter, Sam2TrainingAdapter
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.training.dataset import (
    LetterboxTransform,
    Prompt,
    TrainingBatch,
    _letterbox_transform,
    load_image_tensor,
    load_mask_array,
)
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


def _source_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def _load_images_with_transform(paths: Sequence[Path], transform: LetterboxTransform) -> torch.Tensor:
    raw_images = [load_image_tensor(path, None) for path in paths]
    if not raw_images:
        raise ValueError("No frames to load")
    expected = (transform.source_height, transform.source_width)
    for path, image in zip(paths, raw_images, strict=True):
        if tuple(image.shape[-2:]) != expected:
            raise ValueError(
                "All inference frames must share one source resolution for SAM2 letterbox inference; "
                f"{path} has {tuple(image.shape[-2:])}, expected {expected}"
            )
    return torch.stack([transform.resize_image(image) for image in raw_images], dim=0)


def _load_images(paths: Sequence[Path], image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not paths:
        raise ValueError("No frames to load")
    source_shape = _source_shape(paths[0])
    transform = _letterbox_transform(
        source_width=source_shape[1],
        source_height=source_shape[0],
        target_size=image_size,
    )
    return _load_images_with_transform(paths, transform), source_shape


def _load_reference_mask_with_transform(path: Path, source_shape: tuple[int, int], transform: LetterboxTransform) -> torch.Tensor:
    mask = torch.from_numpy((load_mask_array(path, source_shape) > 0).astype(np.float32))
    mask = transform.resize_mask(mask)
    if not bool(mask.any()):
        raise ValueError(f"Reference mask is empty after letterbox resizing: {path}")
    return mask


def _load_reference_mask(path: Path, source_shape: tuple[int, int], image_size: int) -> torch.Tensor:
    transform = _letterbox_transform(
        source_width=source_shape[1],
        source_height=source_shape[0],
        target_size=image_size,
    )
    return _load_reference_mask_with_transform(path, source_shape, transform)


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


def _resolve_reference_index(
    frame_paths: Sequence[Path],
    *,
    reference_index: int | None,
    reference_frame: str | Path | None,
) -> int:
    if not frame_paths:
        raise ValueError("No frames selected for inference")
    if reference_index is not None:
        index = int(reference_index)
    elif reference_frame is not None:
        text = str(reference_frame)
        reference_path = Path(text).expanduser()
        resolved_reference = reference_path.resolve() if reference_path.exists() else None
        matches: list[int] = []
        for candidate_index, frame_path in enumerate(frame_paths):
            if resolved_reference is not None and frame_path.resolve() == resolved_reference:
                matches.append(candidate_index)
            elif frame_path.name == reference_path.name or frame_path.stem == reference_path.stem:
                matches.append(candidate_index)
        if matches:
            if len(matches) > 1:
                raise ValueError(f"--reference-frame {reference_frame!r} matched multiple selected frames: {matches}")
            index = matches[0]
        elif text.strip().isdigit():
            index = int(text)
        else:
            raise ValueError(
                f"Could not match --reference-frame {reference_frame!r} to selected frame paths. "
                "Pass --reference-index for video-file inputs or use a frame directory with stable names."
            )
    else:
        raise ValueError("Pass either --reference-index or --reference-frame")
    if index < 0 or index >= len(frame_paths):
        raise ValueError(f"reference frame index must be in [0, {len(frame_paths) - 1}], got {index}")
    return index


def _chunk_ranges_forward(reference_index: int, num_frames: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = int(reference_index)
    while start < num_frames:
        end = min(num_frames, start + chunk_size)
        ranges.append((start, end))
        if end >= num_frames:
            break
        start = end - 1
    return ranges


def _chunk_ranges_backward(reference_index: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    end = int(reference_index) + 1
    while end > 0:
        start = max(0, end - chunk_size)
        ranges.append((start, end))
        if start == 0:
            break
        end = start + 1
    return ranges


def _unletterbox_probability(
    probability: torch.Tensor,
    transform: LetterboxTransform,
    source_shape: tuple[int, int],
) -> torch.Tensor:
    if probability.ndim != 2:
        raise ValueError(f"probability mask must have shape HW, got {tuple(probability.shape)}")
    expected_canvas = (transform.canvas_height, transform.canvas_width)
    if tuple(probability.shape[-2:]) != expected_canvas:
        probability = F.interpolate(
            probability[None, None].float(),
            size=expected_canvas,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    crop_top = min(transform.crop_top, max(0, transform.content_height - transform.crop_height))
    crop_left = min(transform.crop_left, max(0, transform.content_width - transform.crop_width))
    y0 = int(transform.pad_top)
    x0 = int(transform.pad_left)
    y1 = y0 + int(transform.crop_height)
    x1 = x0 + int(transform.crop_width)
    content = probability[y0:y1, x0:x1]
    if crop_top or crop_left or tuple(content.shape[-2:]) != (transform.content_height, transform.content_width):
        restored_content = torch.zeros(
            (transform.content_height, transform.content_width),
            dtype=content.dtype,
            device=content.device,
        )
        restored_content[
            crop_top : crop_top + content.shape[-2],
            crop_left : crop_left + content.shape[-1],
        ] = content
        content = restored_content
    if tuple(content.shape[-2:]) != tuple(source_shape):
        content = F.interpolate(
            content[None, None].float(),
            size=tuple(source_shape),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return content


def _probability_to_mask_array(
    probability: torch.Tensor,
    *,
    threshold: float,
    transform: LetterboxTransform | None = None,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if (transform is None) != (source_shape is None):
        raise ValueError("transform and source_shape must be provided together")
    probability = probability.detach().float().cpu()
    if transform is not None and source_shape is not None:
        probability = _unletterbox_probability(probability, transform, source_shape)
    return (probability >= float(threshold)).numpy().astype(np.uint8) * 255


def _save_probability_mask(
    probability: torch.Tensor,
    path: Path,
    *,
    threshold: float,
    transform: LetterboxTransform | None = None,
    source_shape: tuple[int, int] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = _probability_to_mask_array(
        probability,
        threshold=threshold,
        transform=transform,
        source_shape=source_shape,
    )
    Image.fromarray(mask).save(path)
    return path


def _save_masks(
    logits: torch.Tensor,
    output_dir: Path,
    *,
    threshold: float,
    transform: LetterboxTransform | None = None,
    source_shape: tuple[int, int] | None = None,
    indices: Sequence[int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = logits.detach().float().sigmoid().cpu()
    if indices is None:
        indices = list(range(int(probabilities.shape[0])))
    if len(indices) != int(probabilities.shape[0]):
        raise ValueError("indices must have the same length as logits")
    paths: list[Path] = []
    for local_index, global_index in enumerate(indices):
        path = output_dir / f"{int(global_index):06d}.png"
        paths.append(
            _save_probability_mask(
                probabilities[local_index],
                path,
                threshold=threshold,
                transform=transform,
                source_shape=source_shape,
            )
        )
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
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _build_inference_batch(
    *,
    images: torch.Tensor,
    frame_paths: Sequence[Path],
    prompt_mask: torch.Tensor,
    reference_index: int,
    video_path: Path,
    transform: LetterboxTransform,
    sampling_mode: str,
) -> TrainingBatch:
    return TrainingBatch(
        images=images,
        target_masks=torch.zeros(images.shape[0], images.shape[-2], images.shape[-1], device=images.device),
        prompt=Prompt(type="mask", frame_index=reference_index, mask=prompt_mask),
        must3r_features=None,
        dataset="inference",
        scene_id=video_path.stem,
        frame_ids=tuple(path.stem for path in frame_paths),
        image_paths=tuple(path.resolve() for path in frame_paths),
        has_object=torch.zeros(images.shape[0], dtype=torch.bool, device=images.device),
        sampling_mode=sampling_mode,  # type: ignore[arg-type]
        reference_frame_id=frame_paths[reference_index].stem,
        target_source="inference_reference_mask",
        letterbox_transform=transform,
    )


def _predict_logits(
    *,
    wrapper: ThreeAMTrainingWrapper,
    must3r_adapter: Must3rFeatureAdapter,
    batch: TrainingBatch,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    with torch.no_grad():
        sam_features = wrapper.sam2_adapter.encode_sam_features(batch.images)
        must3r_features = _load_features(must3r_adapter, batch, device)
        merged = wrapper.core.forward_features(sam_features, must3r_features)
        backbone_out = wrapper.sam2_adapter._inject_merged_features(merged)
        return wrapper.sam2_adapter.track_masks_from_mask(
            batch.images,
            reference_index=int(batch.prompt.frame_index),
            mask=batch.prompt.mask,
            backbone_out=backbone_out,
        )


def _run_sequence_inference(
    *,
    frame_paths: Sequence[Path],
    video_path: Path,
    reference_index: int,
    reference_mask: torch.Tensor,
    transform: LetterboxTransform,
    wrapper: ThreeAMTrainingWrapper,
    must3r_adapter: Must3rFeatureAdapter,
    device: torch.device,
) -> torch.Tensor:
    images = _load_images_with_transform(frame_paths, transform)
    batch = _build_inference_batch(
        images=images,
        frame_paths=frame_paths,
        prompt_mask=reference_mask,
        reference_index=reference_index,
        video_path=video_path,
        transform=transform,
        sampling_mode="full_scene",
    )
    return _predict_logits(wrapper=wrapper, must3r_adapter=must3r_adapter, batch=batch, device=device)


def _run_chunked_inference(
    *,
    frame_paths: Sequence[Path],
    video_path: Path,
    reference_index: int,
    reference_mask: torch.Tensor,
    transform: LetterboxTransform,
    wrapper: ThreeAMTrainingWrapper,
    must3r_adapter: Must3rFeatureAdapter,
    device: torch.device,
    chunk_size: int,
    mask_dir: Path,
    threshold: float,
    output_transform: LetterboxTransform | None,
    output_source_shape: tuple[int, int] | None,
) -> dict[int, Path]:
    if chunk_size < 2:
        raise ValueError("--chunk-size must be 0 or at least 2")
    saved_paths: dict[int, Path] = {}

    def run_chunk(global_indices: list[int], prompt_mask: torch.Tensor, local_reference_index: int) -> torch.Tensor:
        selected_paths = [frame_paths[index] for index in global_indices]
        images = _load_images_with_transform(selected_paths, transform)
        batch = _build_inference_batch(
            images=images,
            frame_paths=selected_paths,
            prompt_mask=prompt_mask.detach().cpu().float(),
            reference_index=local_reference_index,
            video_path=video_path,
            transform=transform,
            sampling_mode="full_scene",
        )
        logits = _predict_logits(wrapper=wrapper, must3r_adapter=must3r_adapter, batch=batch, device=device)
        probabilities = logits.detach().float().sigmoid().cpu()
        for local_index, global_index in enumerate(global_indices):
            path = mask_dir / f"{global_index:06d}.png"
            saved_paths[global_index] = _save_probability_mask(
                probabilities[local_index],
                path,
                threshold=threshold,
                transform=output_transform,
                source_shape=output_source_shape,
            )
        return (probabilities >= float(threshold)).float()

    prompt_mask = reference_mask.detach().cpu().float()
    for start, end in _chunk_ranges_forward(reference_index, len(frame_paths), chunk_size):
        indices = list(range(start, end))
        predicted = run_chunk(indices, prompt_mask, local_reference_index=0)
        prompt_mask = predicted[-1]

    prompt_mask = reference_mask.detach().cpu().float()
    for start, end in _chunk_ranges_backward(reference_index, chunk_size):
        indices = list(range(start, end))
        predicted = run_chunk(indices, prompt_mask, local_reference_index=len(indices) - 1)
        prompt_mask = predicted[0]

    missing = [index for index in range(len(frame_paths)) if index not in saved_paths]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise RuntimeError(f"Chunked inference did not write masks for frame indices: {preview}{suffix}")
    return saved_paths


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
    reference_index = _resolve_reference_index(
        frame_paths,
        reference_index=getattr(args, "reference_index", None),
        reference_frame=getattr(args, "reference_frame", None),
    )

    source_shape = _source_shape(frame_paths[0])
    transform = _letterbox_transform(
        source_width=source_shape[1],
        source_height=source_shape[0],
        target_size=image_size,
    )
    reference_mask = _load_reference_mask_with_transform(
        Path(args.reference_mask).expanduser(),
        source_shape,
        transform,
    )
    output_root = Path(args.output_dir).expanduser()
    mask_dir = output_root / "masks"
    save_letterbox_masks = bool(getattr(args, "save_letterbox_masks", False))
    output_transform = None if save_letterbox_masks else transform
    output_source_shape = None if save_letterbox_masks else source_shape
    wrapper = _load_wrapper(config, Path(args.checkpoint).expanduser(), device)
    must3r_adapter = Must3rFeatureAdapter(external_config(config, must3r_device=str(device)))

    configured_chunk_size = config.get("inference", {}).get(
        "chunk_size",
        config.get("training", {}).get("visualization_chunk_size", 32),
    )
    raw_chunk_size = getattr(args, "chunk_size", None)
    chunk_size = int(configured_chunk_size if raw_chunk_size is None else raw_chunk_size)
    if chunk_size < 0:
        raise ValueError("--chunk-size must be 0 or a positive integer")
    chunked = chunk_size > 0 and len(frame_paths) > chunk_size
    if chunked:
        saved_by_index = _run_chunked_inference(
            frame_paths=frame_paths,
            video_path=video_path,
            reference_index=reference_index,
            reference_mask=reference_mask,
            transform=transform,
            wrapper=wrapper,
            must3r_adapter=must3r_adapter,
            device=device,
            chunk_size=chunk_size,
            mask_dir=mask_dir,
            threshold=float(args.threshold),
            output_transform=output_transform,
            output_source_shape=output_source_shape,
        )
        mask_paths = [saved_by_index[index] for index in range(len(frame_paths))]
    else:
        logits = _run_sequence_inference(
            frame_paths=frame_paths,
            video_path=video_path,
            reference_index=reference_index,
            reference_mask=reference_mask,
            transform=transform,
            wrapper=wrapper,
            must3r_adapter=must3r_adapter,
            device=device,
        )
        mask_paths = _save_masks(
            logits,
            mask_dir,
            threshold=float(args.threshold),
            transform=output_transform,
            source_shape=output_source_shape,
        )

    output_video_path: Path | None = None
    if not bool(getattr(args, "no_output_video", False)):
        output_video_path = (
            Path(args.output_video).expanduser()
            if getattr(args, "output_video", None)
            else output_root / "masks.mp4"
        )
        _write_video(mask_dir, output_video_path, fps=int(args.fps))
    summary = {
        "frames": len(frame_paths),
        "frame_ids": [path.stem for path in frame_paths],
        "reference_index": reference_index,
        "reference_frame": str(frame_paths[reference_index]),
        "reference_mask": str(Path(args.reference_mask).expanduser()),
        "source_shape": list(source_shape),
        "sam_image_size": image_size,
        "mask_size": "letterbox" if save_letterbox_masks else "source",
        "chunk_size": chunk_size,
        "chunked": chunked,
        "threshold": float(args.threshold),
        "output_dir": str(output_root),
        "mask_dir": str(mask_dir),
        "output_video": str(output_video_path) if output_video_path else None,
        "mask_paths": [str(path) for path in mask_paths],
    }
    if getattr(args, "summary", None):
        summary_path = Path(args.summary).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def infer_video_masks(
    *,
    checkpoint: str | Path,
    video: str | Path,
    reference_mask: str | Path,
    output_dir: str | Path,
    reference_index: int | None = None,
    reference_frame: str | Path | None = None,
    config: str | Path = "configs/full_reproduction.yaml",
    output_video: str | Path | None = None,
    write_output_video: bool = True,
    summary: str | Path | None = None,
    device: str | None = None,
    image_size: int | None = None,
    start_frame: int | None = None,
    num_frames: int | None = None,
    threshold: float = 0.5,
    fps: int = 24,
    chunk_size: int | None = None,
    save_letterbox_masks: bool = False,
) -> dict[str, Any]:
    """Convenience Python API for video mask propagation from one reference mask."""
    return run_inference(
        argparse.Namespace(
            config=str(config),
            checkpoint=str(checkpoint),
            video=str(video),
            reference_index=reference_index,
            reference_frame=str(reference_frame) if reference_frame is not None else None,
            reference_mask=str(reference_mask),
            output_dir=str(output_dir),
            output_video=str(output_video) if output_video is not None else None,
            no_output_video=not write_output_video,
            summary=str(summary) if summary is not None else None,
            device=device,
            image_size=image_size,
            start_frame=start_frame,
            num_frames=num_frames,
            threshold=threshold,
            fps=fps,
            chunk_size=chunk_size,
            save_letterbox_masks=save_letterbox_masks,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track one object through a video using 3AM and a reference mask prompt")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    parser.add_argument("--checkpoint", required=True, help="3AM model weights or training checkpoint")
    parser.add_argument("--video", required=True, help="video file or directory of image frames")
    parser.add_argument("--reference-index", type=int, default=None, help="zero-based index within the selected frames")
    parser.add_argument("--reference-frame", default=None, help="frame path/name/stem, or zero-based index as text")
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-video", default=None, help="mask video path; defaults to <output-dir>/masks.mp4")
    parser.add_argument("--no-output-video", action="store_true", help="only write per-frame masks")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="full-scene propagation chunk size; 0 disables chunking, default uses config inference.chunk_size or training.visualization_chunk_size",
    )
    parser.add_argument(
        "--save-letterbox-masks",
        action="store_true",
        help="save masks in SAM letterbox canvas resolution instead of the source video resolution",
    )
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()
