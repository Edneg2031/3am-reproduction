#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image

from three_am.data.io import read_manifest, write_manifest
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.training.dataset import resolve_project_path
from three_am.utils.config import load_yaml


FeatureLayerSpec = str | int


class FeatureExtractor(Protocol):
    def extract_scene(self, scene: SceneRecord) -> "SceneFeatures": ...


@dataclass(frozen=True)
class SceneFeatures:
    levels: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PrecomputeOptions:
    config: dict[str, Any]
    manifest: Path
    output_dir: Path
    device: str
    weights: Path | None
    must3r_repo: Path | None
    image_size: int
    amp: str | bool
    max_bs: int
    decode_batch_size: int
    cache_dtype: str
    feature_layers: tuple[FeatureLayerSpec, ...]
    write_manifest: Path | None
    limit_scenes: int | None
    dry_run: bool


class Must3rPrecomputeError(RuntimeError):
    pass


def _prepend_repo_to_sys_path(repo: Path | None) -> None:
    if repo is None:
        return
    repo = repo.expanduser().resolve()
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def parse_feature_layers(value: str) -> tuple[FeatureLayerSpec, ...]:
    layers: list[FeatureLayerSpec] = []
    for part in (part.strip() for part in value.split(",") if part.strip()):
        if part.lower() == "encoder":
            layers.append("encoder")
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("--feature-layers must contain at least one layer, such as encoder,4,7,11")
    return tuple(layers)


def parse_amp(value: str) -> str | bool:
    if value.lower() in {"false", "none", "off", "0"}:
        return False
    if value.lower() in {"bf16", "fp16"}:
        return value.lower()
    raise ValueError("--amp must be false, bf16, or fp16")


def _dtype_for_amp(amp: str | bool) -> torch.dtype:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return torch.float32


def _dtype_for_cache(cache_dtype: str) -> torch.dtype:
    if cache_dtype == "fp16":
        return torch.float16
    if cache_dtype == "bf16":
        return torch.bfloat16
    if cache_dtype == "float32":
        return torch.float32
    raise ValueError("--cache-dtype must be fp16, bf16, or float32")


def _external_path(config: dict[str, Any], name: str, override: str | Path | None = None) -> Path | None:
    if override is not None:
        return resolve_project_path(config, override)
    return resolve_project_path(config, config.get("external", {}).get(name))


def _resolve_manifest(config: dict[str, Any], manifest: str | Path | None) -> Path:
    if manifest is not None:
        path = resolve_project_path(config, manifest)
        if path is None:
            raise ValueError("manifest resolved to None")
        return path
    raise ValueError("--manifest is required")


def _scene_limit(scenes: list[SceneRecord], limit: int | None) -> list[SceneRecord]:
    return scenes if limit is None else scenes[:limit]


class OfficialMust3rExtractor:
    def __init__(
        self,
        *,
        weights: Path,
        must3r_repo: Path | None,
        device: str,
        image_size: int,
        amp: str | bool,
        max_bs: int,
        decode_batch_size: int,
        cache_dtype: str,
        feature_layers: tuple[FeatureLayerSpec, ...],
    ) -> None:
        if max_bs < 1:
            raise ValueError("max_bs must be >= 1")
        if decode_batch_size < 1:
            raise ValueError("decode_batch_size must be >= 1")
        self.weights = weights
        self.must3r_repo = must3r_repo
        self.device = torch.device(device)
        self.image_size = image_size
        self.amp = amp
        self.max_bs = max_bs
        self.decode_batch_size = decode_batch_size
        self.cache_dtype = cache_dtype
        self.feature_layers = feature_layers
        self.model: tuple[Any, Any] | None = None

    def load(self) -> None:
        _prepend_repo_to_sys_path(self.must3r_repo)
        try:
            from must3r.model import load_model  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise Must3rPrecomputeError(
                "Could not import must3r.model.load_model. Install naver/must3r or set --must3r-repo/external.must3r_repo."
            ) from error
        if self.weights is None or not self.weights.exists():
            raise Must3rPrecomputeError(f"MUSt3R checkpoint does not exist: {self.weights}")
        self.model = load_model(str(self.weights), device=str(self.device), img_size=self.image_size, verbose=False)

    def extract_scene(self, scene: SceneRecord) -> SceneFeatures:
        if self.model is None:
            self.load()
        if self.model is None:
            raise Must3rPrecomputeError("MUSt3R model failed to load")
        encoder, decoder = self.model
        patch_size = int(encoder.patch_size)
        if not scene.frames:
            raise Must3rPrecomputeError(f"Scene {scene.scene_id} has no frames")
        true_shape_chunks: list[torch.Tensor] = []
        pos_chunks: list[torch.Tensor] = []
        selected_chunks: list[list[torch.Tensor]] | None = None
        selected_specs: tuple[FeatureLayerSpec, ...] | None = None
        selected_indices: tuple[int, ...] | None = None
        with torch.no_grad():
            dtype = _dtype_for_amp(self.amp)
            with torch.autocast(self.device.type, dtype=dtype, enabled=bool(self.amp) and self.device.type == "cuda"):
                for start in range(0, len(scene.frames), self.decode_batch_size):
                    end = min(len(scene.frames), start + self.decode_batch_size)
                    images, true_shape = self._load_images(scene.frames[start:end], patch_size)
                    if not images:
                        raise Must3rPrecomputeError(
                            f"Scene {scene.scene_id} frame chunk {start}:{end} produced no loaded images"
                        )
                    true_shape_tensor = torch.stack(true_shape, dim=0)
                    true_shape_chunks.append(true_shape_tensor)
                    image_tensor = torch.stack(images, dim=0).to(self.device)
                    chunk_true_shape = true_shape_tensor.to(self.device)
                    encoder_tokens, pos = self._encode_frames(encoder, image_tensor, chunk_true_shape)
                    pos_chunks.append(pos.detach().cpu())
                    raw_levels = self._decode_feature_levels(decoder, encoder_tokens, pos, chunk_true_shape)
                    if selected_indices is None:
                        selected_specs, selected_indices = self._normalize_feature_layers(len(raw_levels))
                        selected_chunks = [[] for _ in selected_indices]
                    if selected_chunks is None or selected_specs is None or selected_indices is None:
                        raise Must3rPrecomputeError("MUSt3R decoder returned no selectable feature levels")
                    if any(index >= len(raw_levels) for index in selected_indices):
                        raise Must3rPrecomputeError(
                            f"MUSt3R decoder returned {len(raw_levels)} levels, cannot select {list(self.feature_layers)}"
                        )
                    for output_index, raw_index in enumerate(selected_indices):
                        selected_chunks[output_index].append(raw_levels[raw_index])
                    del image_tensor, chunk_true_shape, encoder_tokens, pos, raw_levels
        if selected_chunks is None:
            raise Must3rPrecomputeError(f"Scene {scene.scene_id} produced no MUSt3R feature chunks")
        if selected_specs is None:
            raise Must3rPrecomputeError(f"Scene {scene.scene_id} produced no selected MUSt3R feature specs")
        true_shape_tensor = torch.cat(true_shape_chunks, dim=0)
        pos_tensor = torch.cat(pos_chunks, dim=0)
        selected_token_levels = tuple(torch.cat(chunks, dim=0) for chunks in selected_chunks)
        selected_levels = tuple(
            self._tokens_to_chw(level, pos_tensor, true_shape_tensor, patch_size, spec).to(
                dtype=_dtype_for_cache(self.cache_dtype)
            )
            for level, spec in zip(selected_token_levels, selected_specs, strict=True)
        )
        return SceneFeatures(
            levels=selected_levels,
            metadata={
                "extractor": "official_must3r_token_grid",
                "feature_layers": [self._feature_layer_label(spec) for spec in self.feature_layers],
                "feature_specs": [self._feature_layer_label(spec) for spec in selected_specs],
                "feature_channels": [int(level.shape[1]) for level in selected_levels],
                "feature_grid": [list(level.shape[-2:]) for level in selected_levels],
                "image_size": self.image_size,
                "amp": self.amp,
                "encoder_batch_size": self.max_bs,
                "decode_batch_size": self.decode_batch_size,
                "cache_dtype": self.cache_dtype,
                "patch_size": patch_size,
                "decoder_memory": False,
                "checkpoint": str(self.weights),
                "frame_ids": [frame.frame_id for frame in scene.frames],
            },
        )

    def _load_images(self, frames: tuple[FrameRecord, ...], patch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        try:
            from must3r.demo.inference import load_images  # type: ignore
        except Exception as error:  # pragma: no cover - depends on external repo
            raise Must3rPrecomputeError("Could not import must3r.demo.inference.load_images") from error
        views = load_images([str(frame.image_path) for frame in frames], size=self.image_size, patch_size=patch_size, verbose=False)
        images = [view["img"].to("cpu") for view in views]
        true_shape = [torch.as_tensor(view["true_shape"], dtype=torch.int64) for view in views]
        return images, true_shape

    def _encode_frames(self, encoder: Any, images: torch.Tensor, true_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for start in range(0, images.shape[0], max(1, self.max_bs)):
            end = min(images.shape[0], start + max(1, self.max_bs))
            encoded, pos = encoder(images[start:end], true_shape[start:end])
            tokens.append(encoded)
            positions.append(pos)
        return torch.cat(tokens, dim=0), torch.cat(positions, dim=0)

    def _decode_feature_levels(
        self,
        decoder: Any,
        encoder_tokens: torch.Tensor,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not hasattr(decoder, "forward_list"):
            raise Must3rPrecomputeError("MUSt3R decoder has no forward_list(..., return_feats=True) hook")
        level_chunks: list[list[torch.Tensor]] | None = None
        for start in range(0, encoder_tokens.shape[0], self.decode_batch_size):
            end = min(encoder_tokens.shape[0], start + self.decode_batch_size)
            batched_tokens = encoder_tokens[start:end].unsqueeze(0)
            batched_pos = pos[start:end].unsqueeze(0)
            batched_true_shape = true_shape[start:end].unsqueeze(0)
            output = decoder.forward_list([batched_tokens], [batched_pos], [batched_true_shape], return_feats=True)
            if not isinstance(output, tuple) or len(output) != 3:
                raise Must3rPrecomputeError("MUSt3R decoder.forward_list did not return (memory, pointmaps, feats)")
            _, _, grouped_feats = output
            if not grouped_feats or not grouped_feats[0]:
                raise Must3rPrecomputeError("MUSt3R decoder returned no feature levels")
            chunk_levels = [feature[0].detach().cpu() for feature in grouped_feats[0]]
            if level_chunks is None:
                level_chunks = [[] for _ in chunk_levels]
            if len(chunk_levels) != len(level_chunks):
                raise Must3rPrecomputeError(
                    f"MUSt3R decoder returned {len(chunk_levels)} levels, expected {len(level_chunks)}"
                )
            for level_index, chunk in enumerate(chunk_levels):
                level_chunks[level_index].append(chunk)
            del output, grouped_feats, chunk_levels
        if level_chunks is None:
            raise Must3rPrecomputeError("MUSt3R decoder returned no feature levels")
        return [torch.cat(chunks, dim=0) for chunks in level_chunks]

    def _normalize_feature_layers(
        self, num_levels: int
    ) -> tuple[tuple[FeatureLayerSpec, ...], tuple[int, ...]]:
        specs: list[FeatureLayerSpec] = []
        indices: list[int] = []
        for layer in self.feature_layers:
            if layer == "encoder":
                index = 0
            elif isinstance(layer, int) and layer >= 0:
                index = layer + 1
            elif isinstance(layer, int):
                index = num_levels + layer
            else:
                raise Must3rPrecomputeError(f"Unsupported MUSt3R feature layer spec: {layer!r}")
            if index < 0 or index >= num_levels:
                raise Must3rPrecomputeError(
                    f"Requested MUSt3R feature layer {layer}, but decoder returned {num_levels} levels"
                )
            specs.append(layer)
            indices.append(index)
        return tuple(specs), tuple(indices)

    def _tokens_to_chw(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        true_shape: torch.Tensor,
        patch_size: int,
        spec: FeatureLayerSpec,
    ) -> torch.Tensor:
        # tokens: T, N, C
        if tokens.ndim != 3:
            raise Must3rPrecomputeError(f"Expected token features with shape TNC, got {tuple(tokens.shape)}")
        expected_channels = self._expected_channels(spec)
        if tokens.shape[2] != expected_channels:
            raise Must3rPrecomputeError(
                "MUSt3R feature channel mismatch for "
                f"{self._feature_layer_label(spec)}: got {tokens.shape[2]}, expected {expected_channels}. "
                "The 3AM paper uses encoder=1024 channels and decoder layers=768 channels."
            )
        if pos.shape[:2] != tokens.shape[:2]:
            raise Must3rPrecomputeError(
                f"MUSt3R token/position mismatch: tokens {tuple(tokens.shape)}, pos {tuple(pos.shape)}"
            )
        height = int(true_shape[0, 0].item())
        width = int(true_shape[0, 1].item())
        if not torch.equal(true_shape, true_shape[:1].expand_as(true_shape)):
            raise Must3rPrecomputeError(
                "All frames in a scene must resize to the same true_shape for token-grid feature export"
            )
        grid_h = height // patch_size
        grid_w = width // patch_size
        if grid_h * grid_w != tokens.shape[1]:
            raise Must3rPrecomputeError(
                f"Could not reshape {tokens.shape[1]} MUSt3R tokens into paper token grid "
                f"{(grid_h, grid_w)} for true_shape {(height, width)} and patch_size {patch_size}. "
                "Refusing to save a dense full-resolution feature map."
            )
        self._validate_pos_grid(pos, grid_h, grid_w)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w).contiguous()

    def _validate_pos_grid(self, pos: torch.Tensor, grid_h: int, grid_w: int) -> None:
        for frame_index, frame_pos in enumerate(pos.detach().cpu()):
            if frame_pos.ndim != 2 or frame_pos.shape[1] != 2:
                raise Must3rPrecomputeError(f"Expected MUSt3R positions with shape N2, got {tuple(frame_pos.shape)}")
            first = torch.unique(frame_pos[:, 0]).numel()
            second = torch.unique(frame_pos[:, 1]).numel()
            if sorted((int(first), int(second))) != sorted((grid_h, grid_w)):
                raise Must3rPrecomputeError(
                    "MUSt3R position grid mismatch for frame "
                    f"{frame_index}: pos unique counts {(int(first), int(second))}, expected {(grid_h, grid_w)}. "
                    "Refusing to save non-token-grid features."
                )

    def _expected_channels(self, spec: FeatureLayerSpec) -> int:
        return 1024 if spec == "encoder" else 768

    def _feature_layer_label(self, spec: FeatureLayerSpec) -> str:
        return "encoder" if spec == "encoder" else f"decoder_{spec}"


def _load_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image).astype(np.float32)
    if np.nanmax(depth) > 100:
        depth = depth / 1000.0
    return depth


def _load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=bool)
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != shape:
        with Image.open(path) as image:
            mask = np.asarray(image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST))
            if mask.ndim == 3:
                mask = mask[..., 0]
    return mask > 0


def compute_fov_overlap(scene: SceneRecord) -> np.ndarray | None:
    if not all(frame.depth_path and frame.pose_path and frame.intrinsics_path for frame in scene.frames):
        return None
    frames = list(scene.frames)
    depths = [_load_depth(frame.depth_path) for frame in frames if frame.depth_path]
    poses = [_load_matrix(frame.pose_path) for frame in frames if frame.pose_path]
    intrinsics = [_load_matrix(frame.intrinsics_path) for frame in frames if frame.intrinsics_path]
    masks = [_load_mask(frame.mask_path, depth.shape) for frame, depth in zip(frames, depths, strict=True)]
    num_frames = len(frames)
    overlap = np.eye(num_frames, dtype=np.float32)
    for ref_index in range(num_frames):
        ref_pose_inv = np.linalg.inv(poses[ref_index])
        ref_k = intrinsics[ref_index]
        ref_h, ref_w = depths[ref_index].shape
        for candidate_index in range(num_frames):
            if ref_index == candidate_index:
                continue
            candidate_points = _backproject_masked_points(
                depths[candidate_index],
                masks[candidate_index],
                intrinsics[candidate_index],
                poses[candidate_index],
            )
            if candidate_points.size == 0:
                overlap[ref_index, candidate_index] = 0.0
                continue
            points_ref = (ref_pose_inv @ candidate_points.T).T[:, :3]
            z = points_ref[:, 2]
            valid_z = z > 1e-6
            projected = (ref_k @ points_ref.T).T
            u = projected[:, 0] / np.maximum(projected[:, 2], 1e-6)
            v = projected[:, 1] / np.maximum(projected[:, 2], 1e-6)
            inside = valid_z & (u >= 0) & (u < ref_w) & (v >= 0) & (v < ref_h)
            overlap[ref_index, candidate_index] = float(inside.mean())
    return overlap


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


def save_scene_features(
    scene: SceneRecord,
    features: SceneFeatures,
    *,
    output_dir: Path,
    manifest: Path,
) -> SceneRecord:
    scene_dir = output_dir / scene.dataset / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    level_paths_by_frame: list[list[Path]] = [[] for _ in scene.frames]
    for level_index, level_tensor in enumerate(features.levels):
        if level_tensor.shape[0] != len(scene.frames):
            raise Must3rPrecomputeError(
                f"Feature level {level_index} has {level_tensor.shape[0]} frames, expected {len(scene.frames)}"
            )
        for frame_index, frame in enumerate(scene.frames):
            path = scene_dir / f"{frame.frame_id}_level{level_index}.pt"
            torch.save(level_tensor[frame_index].detach().cpu(), path)
            level_paths_by_frame[frame_index].append(path.resolve())
    metadata = {
        **features.metadata,
        "feature_channels": [int(level.shape[1]) for level in features.levels],
        "feature_grid": [list(level.shape[-2:]) for level in features.levels],
        "source_manifest": str(manifest),
        "dataset": scene.dataset,
        "scene_id": scene.scene_id,
        "frame_ids": [frame.frame_id for frame in scene.frames],
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    overlap = compute_fov_overlap(scene)
    if overlap is not None:
        np.save(scene_dir / "overlap.npy", overlap)
    updated_frames = tuple(
        FrameRecord(
            frame_id=frame.frame_id,
            image_path=frame.image_path,
            mask_path=frame.mask_path,
            depth_path=frame.depth_path,
            pose_path=frame.pose_path,
            intrinsics_path=frame.intrinsics_path,
            must3r_feature_paths=tuple(level_paths_by_frame[index]),
        )
        for index, frame in enumerate(scene.frames)
    )
    return SceneRecord(
        dataset=scene.dataset,
        scene_id=scene.scene_id,
        split=scene.split,
        frames=updated_frames,
        instances_path=scene.instances_path,
    )


def run_precompute(options: PrecomputeOptions, extractor: FeatureExtractor | None = None) -> dict[str, Any]:
    scenes = read_manifest(options.manifest)
    selected = _scene_limit(scenes, options.limit_scenes)
    summary: dict[str, Any] = {
        "manifest": str(options.manifest),
        "output_dir": str(options.output_dir),
        "write_manifest": str(options.write_manifest) if options.write_manifest else None,
        "dry_run": options.dry_run,
        "scenes": len(selected),
        "frames": sum(len(scene.frames) for scene in selected),
        "weights": str(options.weights) if options.weights else None,
        "must3r_repo": str(options.must3r_repo) if options.must3r_repo else None,
        "amp": options.amp,
        "max_bs": options.max_bs,
        "decode_batch_size": options.decode_batch_size,
        "cache_dtype": options.cache_dtype,
        "feature_layers": list(options.feature_layers),
    }
    if options.dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    if extractor is None:
        if options.weights is None:
            raise Must3rPrecomputeError("--weights or external.must3r_checkpoint is required")
        extractor = OfficialMust3rExtractor(
            weights=options.weights,
            must3r_repo=options.must3r_repo,
            device=options.device,
            image_size=options.image_size,
            amp=options.amp,
            max_bs=options.max_bs,
            decode_batch_size=options.decode_batch_size,
            cache_dtype=options.cache_dtype,
            feature_layers=options.feature_layers,
        )
    options.output_dir.mkdir(parents=True, exist_ok=True)
    updated_by_key: dict[tuple[str, str], SceneRecord] = {}
    for scene in selected:
        features = extractor.extract_scene(scene)
        updated = save_scene_features(scene, features, output_dir=options.output_dir, manifest=options.manifest)
        updated_by_key[(scene.dataset, scene.scene_id)] = updated
        channels = [int(level.shape[1]) for level in features.levels]
        print(f"scene={scene.dataset}/{scene.scene_id} frames={len(scene.frames)} channels={channels}")
    if options.write_manifest is not None:
        updated_scenes = [updated_by_key.get((scene.dataset, scene.scene_id), scene) for scene in scenes]
        write_manifest(options.write_manifest, updated_scenes)
        print(f"Wrote feature manifest to {options.write_manifest}")
    print(json.dumps({**summary, "processed_scenes": len(updated_by_key)}, indent=2))
    return {**summary, "processed_scenes": len(updated_by_key)}


def options_from_args(args: argparse.Namespace) -> PrecomputeOptions:
    if args.max_bs < 1:
        raise ValueError("--max-bs must be >= 1")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be >= 1")
    config = load_yaml(args.config) if args.config else {}
    manifest = _resolve_manifest(config, args.manifest)
    output_dir = resolve_project_path(config, args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir resolved to None")
    weights = _external_path(config, "must3r_checkpoint", args.weights)
    must3r_repo = _external_path(config, "must3r_repo", args.must3r_repo)
    write_manifest = resolve_project_path(config, args.write_manifest) if args.write_manifest else None
    return PrecomputeOptions(
        config=config,
        manifest=manifest,
        output_dir=output_dir,
        device=args.device,
        weights=weights,
        must3r_repo=must3r_repo,
        image_size=args.image_size,
        amp=parse_amp(args.amp),
        max_bs=args.max_bs,
        decode_batch_size=args.decode_batch_size,
        cache_dtype=args.cache_dtype,
        feature_layers=parse_feature_layers(args.feature_layers),
        write_manifest=write_manifest,
        limit_scenes=args.limit_scenes,
        dry_run=args.dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MUSt3R features and FoV caches for 3AM training")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--must3r-repo", default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--amp", default="bf16", choices=["false", "bf16", "fp16"])
    parser.add_argument("--max-bs", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument("--cache-dtype", default="fp16", choices=["fp16", "bf16", "float32"])
    parser.add_argument("--feature-layers", default="encoder,4,7,11")
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    run_precompute(options_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
