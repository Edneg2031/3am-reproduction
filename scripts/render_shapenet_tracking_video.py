#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ShapeNetObject:
    synset_id: str
    model_id: str
    obj_path: Path

    @property
    def output_name(self) -> str:
        stem = f"{self.synset_id}_{self.model_id}" if self.synset_id else self.model_id
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "object"


@dataclass(frozen=True)
class AbsenceWindow:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class IndexedShapeNetObject:
    selected_index: int
    obj: ShapeNetObject


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    worker_count: int
    gpu_id: str
    summary_path: Path
    command: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class CameraPose:
    location: tuple[float, float, float]
    target: tuple[float, float, float]
    roll_rad: float = 0.0


def _candidate_obj_paths(model_dir: Path) -> tuple[Path, ...]:
    return (
        model_dir / "models" / "model_normalized.obj",
        model_dir / "models" / "model.obj",
        model_dir / "model_normalized.obj",
        model_dir / "model.obj",
    )


def resolve_shapenet_obj(root: Path, synset_id: str, model_id: str) -> ShapeNetObject:
    model_dir = root / synset_id / model_id
    for obj_path in _candidate_obj_paths(model_dir):
        if obj_path.exists():
            return ShapeNetObject(synset_id=synset_id, model_id=model_id, obj_path=obj_path.resolve())
    tried = "\n".join(f"  - {path}" for path in _candidate_obj_paths(model_dir))
    raise FileNotFoundError(f"Could not find OBJ for {synset_id}/{model_id}. Tried:\n{tried}")


def infer_object_from_obj_path(path: Path, root: Path | None = None) -> ShapeNetObject:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    synset_id = ""
    model_id = resolved.parent.name
    if resolved.parent.name == "models":
        model_id = resolved.parent.parent.name
        synset_id = resolved.parent.parent.parent.name
    elif root is not None:
        try:
            relative = resolved.relative_to(root.resolve())
            parts = relative.parts
            if len(parts) >= 3:
                synset_id = parts[0]
                model_id = parts[1]
        except ValueError:
            pass
    return ShapeNetObject(synset_id=synset_id, model_id=model_id, obj_path=resolved)


def discover_shapenet_objects(root: Path, *, synset_ids: set[str] | None = None) -> list[ShapeNetObject]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    objects: list[ShapeNetObject] = []
    for synset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if synset_ids is not None and synset_dir.name not in synset_ids:
            continue
        for model_dir in sorted(path for path in synset_dir.iterdir() if path.is_dir()):
            for obj_path in _candidate_obj_paths(model_dir):
                if obj_path.exists():
                    objects.append(ShapeNetObject(synset_id=synset_dir.name, model_id=model_dir.name, obj_path=obj_path.resolve()))
                    break
    if not objects:
        filter_text = f" for synsets {sorted(synset_ids)}" if synset_ids else ""
        raise FileNotFoundError(f"No ShapeNet OBJ files found under {root}{filter_text}")
    return objects


def read_objects_file(path: Path, root: Path | None = None) -> list[ShapeNetObject]:
    objects: list[ShapeNetObject] = []
    for line_number, raw_line in enumerate(path.expanduser().read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) == 1 and parts[0].endswith(".obj"):
            objects.append(infer_object_from_obj_path(Path(parts[0]), root))
            continue
        if len(parts) == 1 and "/" in parts[0]:
            synset_id, model_id = parts[0].split("/", 1)
        elif len(parts) >= 2:
            synset_id, model_id = parts[0], parts[1]
        else:
            raise ValueError(f"{path}:{line_number}: expected 'synset_id/model_id', 'synset_id model_id', or an OBJ path")
        if root is None:
            raise ValueError(f"{path}:{line_number}: synset/model entries require --shapenet-root")
        objects.append(resolve_shapenet_obj(root, synset_id, model_id))
    if not objects:
        raise ValueError(f"{path} did not contain any objects")
    return objects


def choose_objects(args: argparse.Namespace) -> list[ShapeNetObject]:
    root = Path(args.shapenet_root).expanduser().resolve() if args.shapenet_root else None
    if args.objects_file:
        objects = read_objects_file(Path(args.objects_file), root)
    elif args.obj_path:
        objects = [infer_object_from_obj_path(Path(args.obj_path), root)]
    else:
        if root is None:
            raise ValueError("--shapenet-root is required unless --objects-file contains OBJ paths or --obj-path is used")
        synset_ids = set(args.synset_id or []) or None
        objects = discover_shapenet_objects(root, synset_ids=synset_ids)

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(objects)
    else:
        objects = sorted(objects, key=lambda item: (item.synset_id, item.model_id, str(item.obj_path)))
    if args.start_index:
        objects = objects[args.start_index :]
    if args.num_videos is not None:
        objects = objects[: args.num_videos]
    if not objects:
        raise ValueError("No objects selected for rendering")
    return objects


def choose_indexed_objects(args: argparse.Namespace) -> list[IndexedShapeNetObject]:
    return [IndexedShapeNetObject(index, obj) for index, obj in enumerate(choose_objects(args))]


def shard_indexed_objects(
    objects: Sequence[IndexedShapeNetObject],
    *,
    worker_id: int,
    worker_count: int,
) -> list[IndexedShapeNetObject]:
    if worker_count < 1:
        raise ValueError("--worker-count must be positive")
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError("--worker-id must be in [0, --worker-count)")
    return [item for position, item in enumerate(objects) if position % worker_count == worker_id]


def _detect_cuda_gpu_ids() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if gpu_ids:
                return gpu_ids
        except (OSError, subprocess.CalledProcessError):
            pass

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible != "-1":
        return [part.strip() for part in visible.split(",") if part.strip()]
    return []


def parse_parallel_gpus(value: str | None, *, detected_gpu_ids: Sequence[str] | None = None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    value = value.strip()
    if value.lower() == "auto":
        gpu_ids = list(detected_gpu_ids) if detected_gpu_ids is not None else _detect_cuda_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("--parallel-gpus auto could not detect any CUDA GPUs")
        return gpu_ids

    gpu_ids = [part.strip() for part in value.split(",")]
    if any(not gpu_id for gpu_id in gpu_ids):
        raise ValueError("--parallel-gpus must be a comma-separated list like 0,1,2,3")
    for gpu_id in gpu_ids:
        if gpu_id.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", gpu_id):
            raise ValueError(f"Invalid GPU id in --parallel-gpus: {gpu_id!r}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--parallel-gpus contains duplicate GPU ids; use --workers-per-gpu for multiple workers per GPU")
    return gpu_ids


def absence_window(total_frames: int, *, start: int | None, length: int | None) -> AbsenceWindow:
    if total_frames < 8:
        raise ValueError("--frames must be at least 8")
    selected_start = int(round(total_frames * 0.38)) if start is None else int(start)
    selected_length = max(3, int(round(total_frames * 0.24))) if length is None else int(length)
    if selected_start < 1:
        raise ValueError("--absent-start must leave at least one visible frame before absence")
    if selected_length < 1:
        raise ValueError("--absent-frames must be positive")
    selected_end = selected_start + selected_length
    if selected_end >= total_frames - 1:
        raise ValueError("--absent-start + --absent-frames must leave frames for reappearance")
    return AbsenceWindow(selected_start, selected_end)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def object_x(frame_index: int, total_frames: int, window: AbsenceWindow, *, offscreen_x: float) -> float:
    if frame_index < window.start:
        ratio = smoothstep(frame_index / max(1, window.start - 1))
        return -0.75 + 1.2 * ratio
    if frame_index < window.end:
        return offscreen_x
    ratio = smoothstep((frame_index - window.end) / max(1, total_frames - window.end - 1))
    return offscreen_x + (-0.55 - offscreen_x) * ratio


def has_absent_then_reappearing_visibility(visible: Sequence[bool], *, min_absent_frames: int) -> bool:
    saw_visible = False
    absent_count = 0
    for is_visible in visible:
        if is_visible:
            if saw_visible and absent_count >= min_absent_frames:
                return True
            saw_visible = True
            absent_count = 0
        elif saw_visible:
            absent_count += 1
    return False


def visibility_runs(visible: Sequence[bool]) -> list[dict[str, int | bool]]:
    if not visible:
        return []
    runs: list[dict[str, int | bool]] = []
    current = bool(visible[0])
    start = 0
    for index, value in enumerate(visible[1:], start=1):
        value = bool(value)
        if value == current:
            continue
        runs.append({"visible": current, "start": start, "end": index})
        current = value
        start = index
    runs.append({"visible": current, "start": start, "end": len(visible)})
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-render ShapeNetCore.v2 objects into tracking videos and matching per-frame masks. "
            "By default the object stays fixed while the camera moves, looks away until the object leaves "
            "the view for a span, then looks back so the object reappears."
        )
    )
    parser.add_argument("--shapenet-root", help="ShapeNetCore.v2 root, e.g. /data/ShapeNetCore.v2")
    parser.add_argument("--output-root", required=True, help="Directory where one subfolder per video will be written")
    parser.add_argument("--num-videos", type=int, default=5, help="Number of selected objects to render")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many selected objects before rendering")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle discovered objects before selecting --num-videos")
    parser.add_argument("--synset-id", action="append", help="Restrict discovery to a ShapeNet synset id; can be repeated")
    parser.add_argument("--objects-file", help="Optional list of objects: synset/model, synset model, or absolute OBJ path")
    parser.add_argument("--obj-path", help="Render a single OBJ path instead of discovering the dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallel-gpus", help="Launch one or more Blender workers on these GPUs, e.g. 0,1,2,3 or auto")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of Blender workers to launch per GPU when --parallel-gpus is set")
    parser.add_argument("--blender-bin", default="blender", help="Blender executable used by the parallel Python launcher")
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-summary-path", help=argparse.SUPPRESS)

    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--engine",
        default="workbench",
        choices=["auto", "workbench", "eevee", "cycles"],
        help="Use workbench for fast synthetic rendering; use eevee/cycles only when you need higher-fidelity RGB",
    )
    parser.add_argument(
        "--cycles-device",
        default="auto",
        choices=["auto", "optix", "cuda", "cpu"],
        help="Cycles compute device; auto prefers OptiX/CUDA GPU rendering",
    )
    parser.add_argument("--axis-convention", default="y-up", choices=["y-up", "z-up"])
    parser.add_argument("--object-size", type=float, default=1.35)
    parser.add_argument("--motion", default="camera", choices=["camera", "object"], help="camera keeps the object fixed and moves/pans the camera; object keeps the old moving-object behavior")
    parser.add_argument("--mask-mode", default="project", choices=["project", "render"], help="project rasterizes mesh triangles directly; render uses a slower Blender mask pass")
    parser.add_argument(
        "--camera-path",
        default="stochastic-two-body",
        choices=["stochastic-two-body", "human-linear", "orbit"],
        help="stochastic-two-body simulates smooth random camera/look-at dynamics; human-linear and orbit are legacy paths",
    )
    parser.add_argument("--camera-radius", type=float, default=4.2)
    parser.add_argument("--camera-height", type=float, default=1.12)
    parser.add_argument("--camera-orbit-deg", type=float, default=28.0)
    parser.add_argument("--camera-force-std", type=float, default=0.42, help="Gaussian force std for --camera-path stochastic-two-body")
    parser.add_argument("--lookat-force-std", type=float, default=0.28, help="Gaussian look-at force std for --camera-path stochastic-two-body")
    parser.add_argument("--camera-drag", type=float, default=0.88, help="Velocity damping for --camera-path stochastic-two-body")
    parser.add_argument("--camera-max-speed", type=float, default=0.9, help="Per-frame camera speed cap for --camera-path stochastic-two-body")
    parser.add_argument("--lookat-max-speed", type=float, default=0.55, help="Per-frame look-at speed cap for --camera-path stochastic-two-body")
    parser.add_argument("--camera-radius-min", type=float, help="Minimum camera radius for --camera-path stochastic-two-body; default is 0.9 * --camera-radius")
    parser.add_argument("--camera-radius-max", type=float, help="Maximum camera radius for --camera-path stochastic-two-body; default is 1.12 * --camera-radius")
    parser.add_argument("--camera-elevation-min-deg", type=float, default=8.0, help="Minimum camera elevation angle for --camera-path stochastic-two-body")
    parser.add_argument("--camera-elevation-max-deg", type=float, default=28.0, help="Maximum camera elevation angle for --camera-path stochastic-two-body")
    parser.add_argument("--camera-resample-attempts", type=int, default=24, help="Trajectory attempts when stochastic camera motion fails visibility checks")
    parser.add_argument("--camera-linear-x", type=float, default=1.3, help="Total x travel for --camera-path human-linear")
    parser.add_argument("--camera-linear-y", type=float, default=0.18, help="Total y travel for --camera-path human-linear")
    parser.add_argument("--camera-linear-z", type=float, default=0.12, help="Total z travel for --camera-path human-linear")
    parser.add_argument("--camera-jitter-std", type=float, default=0.025, help="World-space smoothed Gaussian camera-position jitter")
    parser.add_argument("--camera-aim-jitter-std", type=float, default=0.018, help="World-space smoothed Gaussian look-target jitter")
    parser.add_argument("--camera-roll-jitter-deg", type=float, default=0.35, help="Smoothed Gaussian camera roll jitter in degrees")
    parser.add_argument("--camera-jitter-smooth-frames", type=float, default=8.0, help="Gaussian smoothing sigma for camera jitter, in frames")
    parser.add_argument("--camera-jitter-ramp-frames", type=int, default=8, help="Frames used to fade hand-held jitter in and out")
    parser.add_argument("--lookaway-x", type=float, default=4.0, help="World-space look target x-offset used to make the fixed object leave the camera view")
    parser.add_argument("--lookaway-transition-frames", type=int, default=10)
    parser.add_argument("--offscreen-x", type=float, default=4.2)

    parser.add_argument("--absent-start", type=int, help="First frame where the object is forced outside the view")
    parser.add_argument("--absent-frames", type=int, help="Number of frames to keep the object outside the view")
    parser.add_argument("--min-absent-frames", type=int, default=3)
    parser.add_argument("--min-visible-mask-pixels", type=int, default=64)
    parser.add_argument("--no-video", action="store_true", help="Write PNG frames and masks only, no mp4 files")
    parser.add_argument("--ffmpeg-codec", default="libx264", choices=["libx264", "h264_nvenc"], help="Optional video codec override for mp4 output")
    return parser.parse_args(argv)


def _require_blender() -> tuple[object, object]:
    try:
        import bpy  # type: ignore
        from mathutils import Vector  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires Blender
        raise RuntimeError(
            "Run this script with Blender, for example:\n"
            "blender -b --python scripts/render_shapenet_tracking_video.py -- "
            "--shapenet-root /path/to/ShapeNetCore.v2 --output-root outputs/shapenet_tracking --num-videos 10"
        ) from exc
    return bpy, Vector


def _clear_scene(bpy: object) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _refresh_cycles_devices(preferences: object) -> None:
    if hasattr(preferences, "refresh_devices"):
        preferences.refresh_devices()
    elif hasattr(preferences, "get_devices"):
        preferences.get_devices()


def _set_cycles_device(bpy: object, *, cycles_device: str) -> None:
    scene = bpy.context.scene
    if cycles_device == "cpu":
        scene.cycles.device = "CPU"
        return

    scene.cycles.device = "GPU"
    device_types = ("OPTIX", "CUDA") if cycles_device == "auto" else (cycles_device.upper(),)
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        print("[WARN] Cycles preferences are unavailable; falling back to CPU rendering", flush=True)
        scene.cycles.device = "CPU"
        return

    for device_type in device_types:
        try:
            preferences.compute_device_type = device_type
            _refresh_cycles_devices(preferences)
        except Exception:
            continue
        enabled = 0
        for device in getattr(preferences, "devices", []):
            is_cpu = getattr(device, "type", "") == "CPU"
            device.use = not is_cpu
            enabled += 0 if is_cpu else 1
        if enabled:
            print(f"[INFO] Cycles GPU device type: {device_type}", flush=True)
            return

    print(f"[WARN] No usable Cycles GPU device found for {cycles_device}; falling back to CPU rendering", flush=True)
    scene.cycles.device = "CPU"


def _set_render_engine(bpy: object, *, engine: str, samples: int, cycles_device: str) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        _set_cycles_device(bpy, cycles_device=cycles_device)
    elif engine == "workbench" and "BLENDER_WORKBENCH" in engines:
        scene.render.engine = "BLENDER_WORKBENCH"
    elif engine == "auto" and "BLENDER_WORKBENCH" in engines:
        scene.render.engine = "BLENDER_WORKBENCH"
    elif "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif engine != "cycles" and "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"
    if getattr(scene.render, "engine", "") == "BLENDER_WORKBENCH":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, samples)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = max(1, samples)


def _material(bpy: object, name: str, color: tuple[float, float, float, float], *, emission: bool = False) -> object:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    if emission:
        nodes = material.node_tree.nodes
        nodes.clear()
        emission_node = nodes.new(type="ShaderNodeEmission")
        emission_node.inputs["Color"].default_value = color
        emission_node.inputs["Strength"].default_value = 1.0
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        material.node_tree.links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
        return material
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
        if "Roughness" in shader.inputs:
            shader.inputs["Roughness"].default_value = 0.64
    return material


def _look_at(obj: object, target: object) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _import_obj(bpy: object, obj_path: Path) -> list[object]:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    meshes = [obj for obj in bpy.data.objects if obj not in before and getattr(obj, "type", None) == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh was imported from {obj_path}")
    return meshes


def _world_bbox(meshes: Sequence[object], Vector: object) -> tuple[object, object]:
    points = []
    for mesh in meshes:
        points.extend(mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box)
    mins = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maxs = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return mins, maxs


def _normalize_meshes(bpy: object, Vector: object, meshes: Sequence[object], *, object_size: float, axis_convention: str) -> object:
    mins, maxs = _world_bbox(meshes, Vector)
    center = (mins + maxs) * 0.5
    extent = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z)
    if extent <= 0:
        raise RuntimeError("Imported object has an empty bounding box")
    scale = object_size / extent
    for mesh in meshes:
        world = mesh.matrix_world.copy()
        mesh.matrix_world.identity()
        for vertex in mesh.data.vertices:
            vertex.co = (world @ vertex.co - center) * scale
        mesh.data.update()
    root = bpy.data.objects.new("tracked_object", None)
    bpy.context.scene.collection.objects.link(root)
    for mesh in meshes:
        mesh.parent = root
    if axis_convention == "y-up":
        root.rotation_euler[0] = math.radians(90.0)
    return root


def _setup_scene(bpy: object, Vector: object, args: argparse.Namespace, obj_path: Path) -> tuple[object, list[object], object, object]:
    _clear_scene(bpy)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = args.frames - 1
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.035, 0.038, 0.04)
    _set_render_engine(bpy, engine=args.engine, samples=args.samples, cycles_device=args.cycles_device)

    meshes = _import_obj(bpy, obj_path)
    root = _normalize_meshes(bpy, Vector, meshes, object_size=args.object_size, axis_convention=args.axis_convention)

    floor_mat = _material(bpy, "floor_rgb", (0.56, 0.56, 0.53, 1.0))
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.45, -args.object_size * 0.52))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(floor_mat)

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -3.4, 4.0))
    key = bpy.context.object
    key.data.energy = 480.0
    key.data.size = 4.0
    bpy.ops.object.light_add(type="POINT", location=(2.4, -2.0, 2.2))
    fill = bpy.context.object
    fill.data.energy = 70.0

    bpy.ops.object.camera_add(location=(0.0, -4.2, 1.12))
    camera = bpy.context.object
    camera.data.angle = math.radians(args.fov_deg)
    _look_at(camera, Vector((0.0, 0.0, 0.1)))
    scene.camera = camera
    return root, meshes, floor, camera


def _render_still(bpy: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _switch_to_fast_mask_render(bpy: object) -> tuple[str, str | None, str | None]:
    scene = bpy.context.scene
    old_engine = scene.render.engine
    old_light = getattr(scene.display.shading, "light", None)
    old_color_type = getattr(scene.display.shading, "color_type", None)
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if "BLENDER_WORKBENCH" in engines:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "MATERIAL"
    return old_engine, old_light, old_color_type


def _restore_render_state(bpy: object, state: tuple[str, str | None, str | None]) -> None:
    engine, light, color_type = state
    scene = bpy.context.scene
    scene.render.engine = engine
    if light is not None:
        scene.display.shading.light = light
    if color_type is not None:
        scene.display.shading.color_type = color_type


def _mask_pixel_count(bpy: object, path: Path) -> int:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = image.pixels
        count = 0
        for index in range(0, len(pixels), 4):
            if pixels[index] > 0.5:
                count += 1
        return count
    finally:
        bpy.data.images.remove(image)


def _camera_intrinsics(camera: object, *, width: int, height: int) -> tuple[float, float, float, float]:
    fx = 0.5 * width / math.tan(0.5 * camera.data.angle_x)
    fy = 0.5 * height / math.tan(0.5 * camera.data.angle_y)
    return fx, fy, width * 0.5, height * 0.5


def _rasterize_triangle(mask: np.ndarray, points: np.ndarray) -> None:
    height, width = mask.shape
    min_x = max(0, int(math.floor(float(points[:, 0].min()))))
    max_x = min(width - 1, int(math.ceil(float(points[:, 0].max()))))
    min_y = max(0, int(math.floor(float(points[:, 1].min()))))
    max_y = min(height - 1, int(math.ceil(float(points[:, 1].max()))))
    if max_x < min_x or max_y < min_y:
        return
    p0, p1, p2 = points.astype(np.float64, copy=False)
    area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if abs(area) < 1e-8:
        return
    ys, xs = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    px = xs.astype(np.float64) + 0.5
    py = ys.astype(np.float64) + 0.5
    e0 = (p1[0] - p0[0]) * (py - p0[1]) - (p1[1] - p0[1]) * (px - p0[0])
    e1 = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
    e2 = (p0[0] - p2[0]) * (py - p2[1]) - (p0[1] - p2[1]) * (px - p2[0])
    if area < 0:
        inside = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
    else:
        inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
    mask[min_y : max_y + 1, min_x : max_x + 1][inside] = 255


def _projected_mesh_mask(
    meshes: Sequence[object],
    camera: object,
    Vector: object,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    fx, fy, cx, cy = _camera_intrinsics(camera, width=width, height=height)
    camera_inv = camera.matrix_world.inverted()
    mask = np.zeros((height, width), dtype=np.uint8)
    for mesh in meshes:
        world = mesh.matrix_world
        vertices_camera = [camera_inv @ (world @ vertex.co) for vertex in mesh.data.vertices]
        for polygon in mesh.data.polygons:
            indices = list(polygon.vertices)
            if len(indices) < 3:
                continue
            for offset in range(1, len(indices) - 1):
                tri = [vertices_camera[indices[0]], vertices_camera[indices[offset]], vertices_camera[indices[offset + 1]]]
                projected: list[tuple[float, float]] = []
                skip = False
                for point in tri:
                    depth = -float(point.z)
                    if depth <= 1e-5:
                        skip = True
                        break
                    projected.append((fx * float(point.x) / depth + cx, cy - fy * float(point.y) / depth))
                if skip:
                    continue
                _rasterize_triangle(mask, np.asarray(projected, dtype=np.float64))
    return mask


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_gray_png(path: Path, mask: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError(f"mask must be HW, got {mask.shape}")
    height, width = mask.shape
    rows = b"".join(b"\x00" + mask[row].astype(np.uint8, copy=False).tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(rows, level=1))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_projected_mask(
    path: Path,
    *,
    meshes: Sequence[object],
    camera: object,
    Vector: object,
    width: int,
    height: int,
) -> int:
    mask = _projected_mesh_mask(meshes, camera, Vector, width=width, height=height)
    _write_gray_png(path, mask)
    return int((mask > 0).sum())


def _render_mask(bpy: object, path: Path, *, meshes: Sequence[object], floor: object, white: object, black: object) -> int:
    mesh_materials = {mesh.name: [slot.material for slot in mesh.material_slots] for mesh in meshes}
    floor_materials = [slot.material for slot in floor.material_slots]
    old_world = bpy.context.scene.world.color[:]
    old_render_state = (bpy.context.scene.render.engine, None, None)
    try:
        old_render_state = _switch_to_fast_mask_render(bpy)
        for mesh in meshes:
            mesh.data.materials.clear()
            mesh.data.materials.append(white)
        floor.data.materials.clear()
        floor.data.materials.append(black)
        bpy.context.scene.world.color = (0.0, 0.0, 0.0)
        _render_still(bpy, path)
        return _mask_pixel_count(bpy, path)
    finally:
        _restore_render_state(bpy, old_render_state)
        bpy.context.scene.world.color = old_world
        for mesh in meshes:
            mesh.data.materials.clear()
            for material in mesh_materials[mesh.name]:
                mesh.data.materials.append(material)
        floor.data.materials.clear()
        for material in floor_materials:
            floor.data.materials.append(material)


def _write_video_from_frames(frame_pattern: Path, output_path: Path, *, fps: int, codec: str) -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write mp4 videos. Install ffmpeg or pass --no-video to keep PNG sequences only.")
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
    ]
    if codec != "libx264":
        command.extend(["-c:v", codec])
    command.append(str(output_path))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_path)


def _lookaway_offset(frame_index: int, window: AbsenceWindow, *, offset: float, transition_frames: int) -> float:
    transition = max(1, int(transition_frames))
    ramp_in_start = max(0, window.start - transition)
    ramp_out_end = window.end + transition
    if frame_index < ramp_in_start:
        return 0.0
    if frame_index < window.start:
        return offset * smoothstep((frame_index - ramp_in_start) / max(1, window.start - ramp_in_start))
    if frame_index < window.end:
        return offset
    if frame_index < ramp_out_end:
        return offset * (1.0 - smoothstep((frame_index - window.end) / max(1, ramp_out_end - window.end)))
    return 0.0


def _linear_lookaway_offset(frame_index: int, window: AbsenceWindow, *, offset: float, transition_frames: int) -> float:
    transition = max(1, int(transition_frames))
    ramp_in_start = max(0, window.start - transition)
    ramp_out_end = window.end + transition
    if frame_index < ramp_in_start:
        return 0.0
    if frame_index < window.start:
        return offset * ((frame_index - ramp_in_start) / max(1, window.start - ramp_in_start))
    if frame_index < window.end:
        return offset
    if frame_index < ramp_out_end:
        return offset * (1.0 - ((frame_index - window.end) / max(1, ramp_out_end - window.end)))
    return 0.0


def _lerp_tuple(start: tuple[float, float, float], end: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(float(a + (b - a) * amount) for a, b in zip(start, end))


def _add_scaled(vector: tuple[float, float, float], direction: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(float(value + axis * scale) for value, axis in zip(vector, direction))


def _add_tuple(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(left + right) for left, right in zip(a, b))


def _scale_tuple(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(float(value * scale) for value in vector)


def _sub_tuple(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(left - right) for left, right in zip(a, b))


def _tuple_length(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _clip_tuple_length(vector: tuple[float, float, float], max_length: float) -> tuple[float, float, float]:
    if max_length < 0.0:
        raise ValueError("max length must be non-negative")
    length = _tuple_length(vector)
    if length <= max_length or length <= 1e-8:
        return vector
    return _scale_tuple(vector, max_length / length)


def _cross_tuple(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize_tuple(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-8:
        return (0.0, 0.0, 0.0)
    return tuple(float(value / length) for value in vector)


def _camera_basis(location: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    forward = _normalize_tuple(_sub_tuple(target, location))
    world_up = (0.0, 0.0, 1.0)
    right = _normalize_tuple(_cross_tuple(forward, world_up))
    if right == (0.0, 0.0, 0.0):
        right = (1.0, 0.0, 0.0)
    up = _normalize_tuple(_cross_tuple(right, forward))
    return right, up, forward


def _gaussian_kernel_1d(sigma_frames: float) -> np.ndarray:
    if sigma_frames <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma_frames)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma_frames)) ** 2)
    return kernel / kernel.sum()


def _smooth_gaussian_noise(noise: np.ndarray, *, sigma_frames: float) -> np.ndarray:
    kernel = _gaussian_kernel_1d(sigma_frames)
    radius = len(kernel) // 2
    if radius == 0:
        smoothed = noise.astype(np.float64, copy=True)
    elif noise.ndim == 1:
        padded = np.pad(noise, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    else:
        columns = []
        for channel in range(noise.shape[1]):
            padded = np.pad(noise[:, channel], (radius, radius), mode="edge")
            columns.append(np.convolve(padded, kernel, mode="valid"))
        smoothed = np.stack(columns, axis=1)

    axis = 0 if smoothed.ndim > 1 else None
    std = np.std(smoothed, axis=axis, keepdims=smoothed.ndim > 1)
    return np.divide(smoothed, std, out=np.zeros_like(smoothed, dtype=np.float64), where=std > 1e-8)


def _jitter_envelope(frame_index: int, total_frames: int, ramp_frames: int) -> float:
    if ramp_frames <= 0 or total_frames <= 1:
        return 1.0
    edge = min(float(ramp_frames), max(1.0, (total_frames - 1) * 0.5))
    amount = min(1.0, frame_index / edge, (total_frames - 1 - frame_index) / edge)
    return smoothstep(amount)


def _camera_seed(args: argparse.Namespace, obj: ShapeNetObject) -> int:
    payload = f"{args.seed}:{obj.synset_id}:{obj.model_id}:{obj.obj_path}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def _orbit_camera_pose(frame_index: int, total_frames: int, window: AbsenceWindow, args: argparse.Namespace) -> CameraPose:
    progress = frame_index / max(1, total_frames - 1)
    orbit = math.radians(args.camera_orbit_deg) * math.sin(2.0 * math.pi * progress)
    radius = float(args.camera_radius)
    location = (
        radius * math.sin(orbit),
        -radius * math.cos(orbit),
        float(args.camera_height) + 0.12 * math.sin(2.0 * math.pi * progress + 0.7),
    )
    lookaway = _lookaway_offset(
        frame_index,
        window,
        offset=float(args.lookaway_x),
        transition_frames=int(args.lookaway_transition_frames),
    )
    target = (lookaway, 0.0, 0.1 + 0.05 * math.sin(2.0 * math.pi * progress))
    return CameraPose(location=location, target=target)


def _human_linear_camera_poses(total_frames: int, window: AbsenceWindow, args: argparse.Namespace, *, seed: int) -> list[CameraPose]:
    rng = np.random.default_rng(seed)
    position_noise = _smooth_gaussian_noise(rng.normal(size=(total_frames, 3)), sigma_frames=float(args.camera_jitter_smooth_frames))
    aim_noise = _smooth_gaussian_noise(rng.normal(size=(total_frames, 2)), sigma_frames=float(args.camera_jitter_smooth_frames))
    roll_noise = _smooth_gaussian_noise(rng.normal(size=total_frames), sigma_frames=float(args.camera_jitter_smooth_frames))

    radius = float(args.camera_radius)
    height = float(args.camera_height)
    camera_start = (-0.5 * float(args.camera_linear_x), -radius - 0.5 * float(args.camera_linear_y), height - 0.5 * float(args.camera_linear_z))
    camera_end = (0.5 * float(args.camera_linear_x), -radius + 0.5 * float(args.camera_linear_y), height + 0.5 * float(args.camera_linear_z))
    base_target_z = 0.1
    poses: list[CameraPose] = []
    for frame_index in range(total_frames):
        progress = frame_index / max(1, total_frames - 1)
        location = _lerp_tuple(camera_start, camera_end, progress)
        lookaway = _linear_lookaway_offset(
            frame_index,
            window,
            offset=float(args.lookaway_x),
            transition_frames=int(args.lookaway_transition_frames),
        )
        target = (lookaway, 0.0, base_target_z)
        right, up, forward = _camera_basis(location, target)
        envelope = _jitter_envelope(frame_index, total_frames, int(args.camera_jitter_ramp_frames))

        location = _add_scaled(location, right, envelope * float(args.camera_jitter_std) * float(position_noise[frame_index, 0]))
        location = _add_scaled(location, up, envelope * float(args.camera_jitter_std) * float(position_noise[frame_index, 1]))
        location = _add_scaled(location, forward, envelope * float(args.camera_jitter_std) * 0.35 * float(position_noise[frame_index, 2]))
        target = _add_scaled(target, right, envelope * float(args.camera_aim_jitter_std) * float(aim_noise[frame_index, 0]))
        target = _add_scaled(target, up, envelope * float(args.camera_aim_jitter_std) * float(aim_noise[frame_index, 1]))
        roll = envelope * math.radians(float(args.camera_roll_jitter_deg)) * float(roll_noise[frame_index])
        poses.append(CameraPose(location=location, target=target, roll_rad=roll))
    return poses


def _camera_focus_center() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.1)


def _camera_radius_bounds(args: argparse.Namespace) -> tuple[float, float]:
    radius = float(args.camera_radius)
    radius_min = float(args.camera_radius_min) if args.camera_radius_min is not None else radius * 0.9
    radius_max = float(args.camera_radius_max) if args.camera_radius_max is not None else radius * 1.12
    if radius_min <= 0.0:
        raise ValueError("--camera-radius-min must be positive")
    if radius_max < radius_min:
        raise ValueError("--camera-radius-max must be greater than or equal to --camera-radius-min")
    return radius_min, radius_max


def _camera_elevation_bounds(args: argparse.Namespace) -> tuple[float, float]:
    elevation_min = math.radians(float(args.camera_elevation_min_deg))
    elevation_max = math.radians(float(args.camera_elevation_max_deg))
    if elevation_max < elevation_min:
        raise ValueError("--camera-elevation-max-deg must be greater than or equal to --camera-elevation-min-deg")
    return elevation_min, elevation_max


def _project_camera_position(
    position: tuple[float, float, float],
    *,
    center: tuple[float, float, float],
    radius_min: float,
    radius_max: float,
    elevation_min: float,
    elevation_max: float,
) -> tuple[float, float, float]:
    vector = _sub_tuple(position, center)
    radius = max(radius_min, min(radius_max, _tuple_length(vector)))
    horizontal = math.hypot(vector[0], vector[1])
    azimuth = math.atan2(vector[0], vector[1]) if horizontal > 1e-8 else math.pi
    elevation = math.atan2(vector[2], horizontal)
    elevation = max(elevation_min, min(elevation_max, elevation))
    projected_horizontal = radius * math.cos(elevation)
    return (
        float(center[0] + projected_horizontal * math.sin(azimuth)),
        float(center[1] + projected_horizontal * math.cos(azimuth)),
        float(center[2] + radius * math.sin(elevation)),
    )


def _bounded_projected_step(
    previous: tuple[float, float, float],
    candidate: tuple[float, float, float],
    *,
    center: tuple[float, float, float],
    radius_min: float,
    radius_max: float,
    elevation_min: float,
    elevation_max: float,
    max_step: float,
) -> tuple[float, float, float]:
    projected = _project_camera_position(
        candidate,
        center=center,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
    )
    step = _sub_tuple(projected, previous)
    if _tuple_length(step) <= max_step:
        return projected

    amount = max_step / max(1e-8, _tuple_length(step))
    for _ in range(10):
        limited = _add_tuple(previous, _scale_tuple(step, amount))
        limited = _project_camera_position(
            limited,
            center=center,
            radius_min=radius_min,
            radius_max=radius_max,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
        )
        if _tuple_length(_sub_tuple(limited, previous)) <= max_step + 1e-8:
            return limited
        amount *= 0.75
    return previous


def _sample_front_shell_position(
    rng: np.random.Generator,
    *,
    center: tuple[float, float, float],
    radius_min: float,
    radius_max: float,
    elevation_min: float,
    elevation_max: float,
) -> tuple[float, float, float]:
    radius = float(rng.uniform(radius_min, radius_max))
    elevation = float(rng.uniform(elevation_min, elevation_max))
    azimuth = math.pi + float(rng.uniform(-math.radians(55.0), math.radians(55.0)))
    horizontal = radius * math.cos(elevation)
    return (
        float(center[0] + horizontal * math.sin(azimuth)),
        float(center[1] + horizontal * math.cos(azimuth)),
        float(center[2] + radius * math.sin(elevation)),
    )


def _clamp_lookat_position(position: tuple[float, float, float], attractor: tuple[float, float, float]) -> tuple[float, float, float]:
    x_min = min(0.0, attractor[0]) - 0.65
    x_max = max(0.0, attractor[0]) + 0.65
    z = position[2]
    if abs(attractor[0]) >= 1e-8:
        z = attractor[2] + 0.25 * (z - attractor[2])
    return (
        float(max(x_min, min(x_max, position[0]))),
        float(max(-0.35, min(0.35, position[1]))),
        float(max(0.02, min(0.38, z))),
    )


def _stochastic_two_body_camera_poses(total_frames: int, window: AbsenceWindow, args: argparse.Namespace, *, seed: int) -> list[CameraPose]:
    rng = np.random.default_rng(seed)
    center = _camera_focus_center()
    radius_min, radius_max = _camera_radius_bounds(args)
    elevation_min, elevation_max = _camera_elevation_bounds(args)
    camera_pos = _sample_front_shell_position(
        rng,
        center=center,
        radius_min=radius_min,
        radius_max=radius_max,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
    )
    camera_vel = _clip_tuple_length(
        tuple(float(value) for value in rng.normal(scale=0.08, size=3)),
        float(args.camera_max_speed) * 0.25,
    )
    lookat_pos = center
    lookat_vel = _clip_tuple_length(
        tuple(float(value) for value in rng.normal(scale=0.02, size=3)),
        float(args.lookat_max_speed) * 0.15,
    )
    roll = 0.0
    roll_limit = math.radians(float(args.camera_roll_jitter_deg))
    roll_step_std = roll_limit * 0.35
    drag = float(args.camera_drag)
    camera_max_speed = max(0.0, float(args.camera_max_speed))
    lookat_max_speed = max(0.0, float(args.lookat_max_speed))
    poses: list[CameraPose] = []

    for frame_index in range(total_frames):
        camera_pos = _project_camera_position(
            camera_pos,
            center=center,
            radius_min=radius_min,
            radius_max=radius_max,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
        )
        poses.append(CameraPose(location=camera_pos, target=lookat_pos, roll_rad=roll))

        lookaway = _lookaway_offset(
            frame_index,
            window,
            offset=float(args.lookaway_x),
            transition_frames=int(args.lookaway_transition_frames),
        )
        lookat_attractor = (lookaway, 0.0, center[2])

        camera_force = tuple(float(value) for value in rng.normal(scale=float(args.camera_force_std), size=3))
        camera_force = (camera_force[0], camera_force[1], camera_force[2] * 0.45)
        camera_vel = _add_tuple(_scale_tuple(camera_vel, drag), camera_force)
        camera_vel = _clip_tuple_length(camera_vel, camera_max_speed)
        camera_candidate = _add_tuple(camera_pos, camera_vel)
        next_camera_pos = _bounded_projected_step(
            camera_pos,
            camera_candidate,
            center=center,
            radius_min=radius_min,
            radius_max=radius_max,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            max_step=camera_max_speed,
        )
        camera_vel = _sub_tuple(next_camera_pos, camera_pos)
        camera_pos = next_camera_pos

        lookat_spring = _scale_tuple(_sub_tuple(lookat_attractor, lookat_pos), 0.35)
        lookat_force = tuple(float(value) for value in rng.normal(scale=float(args.lookat_force_std), size=3))
        lookat_force = (lookat_force[0], lookat_force[1] * 0.35, lookat_force[2] * 0.2)
        lookat_vel = _add_tuple(_scale_tuple(lookat_vel, drag), _add_tuple(lookat_spring, lookat_force))
        lookat_vel = _clip_tuple_length(lookat_vel, lookat_max_speed)
        lookat_pos = _clamp_lookat_position(_add_tuple(lookat_pos, lookat_vel), lookat_attractor)
        lookat_vel = _sub_tuple(lookat_pos, poses[-1].target)

        if roll_limit > 0.0:
            roll = max(-roll_limit, min(roll_limit, drag * roll + float(rng.normal(scale=roll_step_std))))
        else:
            roll = 0.0

    return poses


def build_camera_poses(total_frames: int, window: AbsenceWindow, args: argparse.Namespace, *, seed: int) -> list[CameraPose]:
    if args.camera_path == "orbit":
        return [_orbit_camera_pose(frame_index, total_frames, window, args) for frame_index in range(total_frames)]
    if args.camera_path == "stochastic-two-body":
        return _stochastic_two_body_camera_poses(total_frames, window, args, seed=seed)
    return _human_linear_camera_poses(total_frames, window, args, seed=seed)


def _set_camera_pose(camera: object, Vector: object, pose: CameraPose) -> None:
    camera.location = pose.location
    _look_at(camera, Vector(pose.target))
    if abs(pose.roll_rad) > 1e-10:
        camera.rotation_euler.rotate_axis("Z", pose.roll_rad)


def _verify_frame_sequence(directory: Path, *, expected_frames: int, label: str) -> None:
    frame_count = len(list(directory.glob("*.png")))
    if frame_count != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} {label} frames in {directory}, found {frame_count}")


def _object_output_dir(output_root: Path, item: IndexedShapeNetObject) -> Path:
    return output_root / f"{item.selected_index:04d}_{item.obj.output_name}"


def render_one_object(bpy: object, Vector: object, obj: ShapeNetObject, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
    rgb_dir = output_dir / "rgb"
    mask_dir = output_dir / "masks"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    root, meshes, floor, camera = _setup_scene(bpy, Vector, args, obj.obj_path)
    white = _material(bpy, "target_mask_white", (1.0, 1.0, 1.0, 1.0), emission=True)
    black = _material(bpy, "mask_black", (0.0, 0.0, 0.0, 1.0), emission=True)
    window = absence_window(args.frames, start=args.absent_start, length=args.absent_frames)
    base_camera_seed = _camera_seed(args, obj)
    attempts = (
        max(1, int(args.camera_resample_attempts))
        if args.motion == "camera" and args.camera_path == "stochastic-two-body"
        else 1
    )
    selected_camera_seed = base_camera_seed
    selected_attempt = 0
    mask_pixels: list[int] = []
    visible: list[bool] = []
    last_runs: list[dict[str, int | bool]] = []

    for attempt_index in range(attempts):
        selected_attempt = attempt_index
        selected_camera_seed = (base_camera_seed + attempt_index * 0x9E3779B1) & 0xFFFFFFFF
        camera_poses = (
            build_camera_poses(args.frames, window, args, seed=selected_camera_seed)
            if args.motion == "camera"
            else []
        )
        mask_pixels = []
        visible = []
        for frame_index in range(args.frames):
            bpy.context.scene.frame_set(frame_index)
            if args.motion == "camera":
                root.location.x = 0.0
                root.rotation_euler[2] = 0.0
                _set_camera_pose(camera, Vector, camera_poses[frame_index])
            else:
                root.location.x = object_x(frame_index, args.frames, window, offscreen_x=args.offscreen_x)
                root.rotation_euler[2] = 0.28 * math.sin(frame_index * 0.12)
            frame_id = f"{frame_index:06d}"
            _render_still(bpy, rgb_dir / f"{frame_id}.png")
            if args.mask_mode == "project":
                pixels = _write_projected_mask(
                    mask_dir / f"{frame_id}.png",
                    meshes=meshes,
                    camera=camera,
                    Vector=Vector,
                    width=args.width,
                    height=args.height,
                )
            else:
                pixels = _render_mask(
                    bpy,
                    mask_dir / f"{frame_id}.png",
                    meshes=meshes,
                    floor=floor,
                    white=white,
                    black=black,
                )
            mask_pixels.append(pixels)
            visible.append(pixels >= args.min_visible_mask_pixels)

        _verify_frame_sequence(rgb_dir, expected_frames=args.frames, label="RGB")
        _verify_frame_sequence(mask_dir, expected_frames=args.frames, label="mask")
        if has_absent_then_reappearing_visibility(visible, min_absent_frames=args.min_absent_frames):
            break
        last_runs = visibility_runs(visible)
        if attempt_index + 1 < attempts:
            print(
                f"[WARN] Camera trajectory attempt {attempt_index + 1}/{attempts} failed visibility for "
                f"{obj.synset_id}/{obj.model_id}: {last_runs}. Resampling.",
                flush=True,
            )
    else:
        raise RuntimeError(
            f"{obj.synset_id}/{obj.model_id} did not produce visible -> absent -> visible masks after {attempts} "
            f"camera trajectory attempts. Last visibility runs: {last_runs}. For camera motion, try increasing "
            "--lookaway-x or lowering --min-visible-mask-pixels. For object motion, try increasing --offscreen-x."
        )

    video_path = None
    mask_video_path = None
    if not args.no_video:
        video_path = _write_video_from_frames(
            rgb_dir / "%06d.png",
            output_dir / "video.mp4",
            fps=args.fps,
            codec=args.ffmpeg_codec,
        )
        mask_video_path = _write_video_from_frames(
            mask_dir / "%06d.png",
            output_dir / "mask.mp4",
            fps=args.fps,
            codec=args.ffmpeg_codec,
        )

    if args.camera_path == "stochastic-two-body":
        stochastic_radius_min, stochastic_radius_max = _camera_radius_bounds(args)
    else:
        stochastic_radius_min, stochastic_radius_max = args.camera_radius_min, args.camera_radius_max

    metadata = {
        "schema": "shapenet_tracking_render_v1",
        "synset_id": obj.synset_id,
        "model_id": obj.model_id,
        "obj_path": str(obj.obj_path),
        "frames": args.frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "motion": args.motion,
        "mask_mode": args.mask_mode,
        "camera_path": args.camera_path,
        "camera_seed": selected_camera_seed if args.motion == "camera" else None,
        "camera_resample_attempt": selected_attempt if args.motion == "camera" else None,
        "camera_stochastic": {
            "force_std": args.camera_force_std,
            "lookat_force_std": args.lookat_force_std,
            "drag": args.camera_drag,
            "camera_max_speed": args.camera_max_speed,
            "lookat_max_speed": args.lookat_max_speed,
            "radius_min": stochastic_radius_min,
            "radius_max": stochastic_radius_max,
            "elevation_min_deg": args.camera_elevation_min_deg,
            "elevation_max_deg": args.camera_elevation_max_deg,
            "resample_attempts": args.camera_resample_attempts,
        },
        "camera_linear": {
            "x": args.camera_linear_x,
            "y": args.camera_linear_y,
            "z": args.camera_linear_z,
        },
        "camera_jitter": {
            "position_std": args.camera_jitter_std,
            "aim_std": args.camera_aim_jitter_std,
            "roll_deg": args.camera_roll_jitter_deg,
            "smooth_frames": args.camera_jitter_smooth_frames,
            "ramp_frames": args.camera_jitter_ramp_frames,
        },
        "rgb_dir": str(rgb_dir),
        "mask_dir": str(mask_dir),
        "video_path": video_path,
        "mask_video_path": mask_video_path,
        "absence_window": {"start": window.start, "end": window.end, "length": window.length},
        "visibility_runs": visibility_runs(visible),
        "mask_pixels": mask_pixels,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def render_batch(args: argparse.Namespace) -> list[dict[str, object]]:
    bpy, Vector = _require_blender()
    indexed_objects = choose_indexed_objects(args)
    if args.worker_id is not None or args.worker_count is not None:
        if args.worker_id is None or args.worker_count is None:
            raise ValueError("--worker-id and --worker-count must be provided together")
        indexed_objects = shard_indexed_objects(indexed_objects, worker_id=args.worker_id, worker_count=args.worker_count)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    worker_label = f" worker={args.worker_id}/{args.worker_count}" if args.worker_id is not None else ""
    for local_index, item in enumerate(indexed_objects):
        output_dir = _object_output_dir(output_root, item)
        obj = item.obj
        print(f"[{local_index + 1}/{len(indexed_objects)}]{worker_label} Rendering {obj.synset_id}/{obj.model_id} -> {output_dir}", flush=True)
        metadata = render_one_object(bpy, Vector, obj, output_dir, args)
        results.append({"index": item.selected_index, "output_dir": str(output_dir), **metadata})

    summary_path = Path(args.worker_summary_path).expanduser().resolve() if args.worker_summary_path else output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"videos": results}, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "videos": results}, indent=2), flush=True)
    return results


def _running_inside_blender() -> bool:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    return True


def _strip_cli_options(argv: Sequence[str], options_with_values: set[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if any(token.startswith(f"{option}=") for option in options_with_values):
            continue
        stripped.append(token)
    return stripped


def _worker_gpu_assignments(gpu_ids: Sequence[str], *, workers_per_gpu: int, selected_count: int) -> list[str]:
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    assignments = [gpu_id for _ in range(workers_per_gpu) for gpu_id in gpu_ids]
    if not assignments:
        raise ValueError("--parallel-gpus did not provide any GPU ids")
    return assignments[: max(1, min(len(assignments), selected_count))]


def build_worker_specs(args: argparse.Namespace, argv: Sequence[str], *, selected_count: int) -> list[WorkerSpec]:
    gpu_ids = parse_parallel_gpus(args.parallel_gpus)
    assignments = _worker_gpu_assignments(gpu_ids, workers_per_gpu=args.workers_per_gpu, selected_count=selected_count)
    output_root = Path(args.output_root).expanduser().resolve()
    summary_root = output_root / ".worker_summaries" / f"run_{os.getpid()}"
    script_path = Path(__file__).resolve()
    worker_base_argv = _strip_cli_options(
        argv,
        {
            "--parallel-gpus",
            "--workers-per-gpu",
            "--blender-bin",
            "--worker-id",
            "--worker-count",
            "--worker-summary-path",
        },
    )

    specs: list[WorkerSpec] = []
    worker_count = len(assignments)
    for worker_id, gpu_id in enumerate(assignments):
        summary_path = summary_root / f"worker_{worker_id:03d}.json"
        worker_argv = [
            *worker_base_argv,
            "--worker-id",
            str(worker_id),
            "--worker-count",
            str(worker_count),
            "--worker-summary-path",
            str(summary_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        command = [args.blender_bin, "-b", "--python", str(script_path), "--", *worker_argv]
        specs.append(
            WorkerSpec(
                worker_id=worker_id,
                worker_count=worker_count,
                gpu_id=str(gpu_id),
                summary_path=summary_path,
                command=command,
                env=env,
            )
        )
    return specs


def merge_worker_summaries(summary_paths: Sequence[Path], output_root: Path) -> list[dict[str, object]]:
    videos: list[dict[str, object]] = []
    for summary_path in summary_paths:
        if not summary_path.exists():
            raise RuntimeError(f"Worker summary was not written: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        worker_videos = payload.get("videos", [])
        if not isinstance(worker_videos, list):
            raise RuntimeError(f"Worker summary has invalid 'videos' payload: {summary_path}")
        videos.extend(worker_videos)

    videos.sort(key=lambda item: int(item["index"]))
    indices = [int(item["index"]) for item in videos]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate video indices found while merging worker summaries: {indices}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps({"videos": videos}, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "videos": videos}, indent=2), flush=True)
    return videos


def run_parallel_launcher(args: argparse.Namespace, argv: Sequence[str]) -> list[dict[str, object]]:
    selected_count = len(choose_indexed_objects(args))
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = build_worker_specs(args, argv, selected_count=selected_count)
    for spec in specs:
        spec.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Launching {len(specs)} Blender workers for {selected_count} videos "
        f"on GPUs {', '.join(spec.gpu_id for spec in specs)}",
        flush=True,
    )
    processes: list[tuple[WorkerSpec, subprocess.Popen[bytes]]] = []
    for spec in specs:
        print(f"[INFO] worker {spec.worker_id}/{spec.worker_count} CUDA_VISIBLE_DEVICES={spec.gpu_id}", flush=True)
        processes.append((spec, subprocess.Popen(spec.command, env=spec.env)))

    failures: list[tuple[WorkerSpec, int]] = []
    for spec, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((spec, return_code))
    if failures:
        failure_text = ", ".join(f"worker {spec.worker_id} on GPU {spec.gpu_id} exited {return_code}" for spec, return_code in failures)
        raise RuntimeError(f"Parallel rendering failed; not writing final summary: {failure_text}")

    return merge_worker_summaries([spec.summary_path for spec in specs], output_root)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    inside_blender = _running_inside_blender()
    if args.parallel_gpus and args.worker_id is None:
        if inside_blender:
            raise RuntimeError("--parallel-gpus must be launched with regular Python, not an already-running Blender process")
        run_parallel_launcher(args, raw_argv)
    else:
        render_batch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None))
