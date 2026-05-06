#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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

    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", default="auto", choices=["auto", "eevee", "cycles"], help="Use cycles on headless Linux if EEVEE has no GPU/display context")
    parser.add_argument("--axis-convention", default="y-up", choices=["y-up", "z-up"])
    parser.add_argument("--object-size", type=float, default=1.35)
    parser.add_argument("--motion", default="camera", choices=["camera", "object"], help="camera keeps the object fixed and moves/pans the camera; object keeps the old moving-object behavior")
    parser.add_argument("--camera-radius", type=float, default=4.2)
    parser.add_argument("--camera-height", type=float, default=1.12)
    parser.add_argument("--camera-orbit-deg", type=float, default=28.0)
    parser.add_argument("--lookaway-x", type=float, default=4.0, help="World-space look target x-offset used to make the fixed object leave the camera view")
    parser.add_argument("--lookaway-transition-frames", type=int, default=10)
    parser.add_argument("--offscreen-x", type=float, default=4.2)

    parser.add_argument("--absent-start", type=int, help="First frame where the object is forced outside the view")
    parser.add_argument("--absent-frames", type=int, help="Number of frames to keep the object outside the view")
    parser.add_argument("--min-absent-frames", type=int, default=3)
    parser.add_argument("--min-visible-mask-pixels", type=int, default=64)
    parser.add_argument("--no-video", action="store_true", help="Write PNG frames and masks only, no mp4 files")
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


def _set_render_engine(bpy: object, *, engine: str, samples: int) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
    elif "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif engine != "cycles" and "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, samples)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = max(1, samples)


def _material(bpy: object, name: str, color: tuple[float, float, float, float], *, emission: bool = False) -> object:
    material = bpy.data.materials.new(name)
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
    _set_render_engine(bpy, engine=args.engine, samples=args.samples)

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


def _render_mask(bpy: object, path: Path, *, meshes: Sequence[object], floor: object, white: object, black: object) -> int:
    mesh_materials = {mesh.name: [slot.material for slot in mesh.material_slots] for mesh in meshes}
    floor_materials = [slot.material for slot in floor.material_slots]
    old_world = bpy.context.scene.world.color[:]
    try:
        for mesh in meshes:
            mesh.data.materials.clear()
            mesh.data.materials.append(white)
        floor.data.materials.clear()
        floor.data.materials.append(black)
        bpy.context.scene.world.color = (0.0, 0.0, 0.0)
        _render_still(bpy, path)
        return _mask_pixel_count(bpy, path)
    finally:
        bpy.context.scene.world.color = old_world
        for mesh in meshes:
            mesh.data.materials.clear()
            for material in mesh_materials[mesh.name]:
                mesh.data.materials.append(material)
        floor.data.materials.clear()
        for material in floor_materials:
            floor.data.materials.append(material)


def _write_video_from_frames(frame_pattern: Path, output_path: Path, *, fps: int) -> str | None:
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
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_path),
    ]
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


def _set_camera_for_frame(camera: object, Vector: object, frame_index: int, total_frames: int, window: AbsenceWindow, args: argparse.Namespace) -> None:
    progress = frame_index / max(1, total_frames - 1)
    orbit = math.radians(args.camera_orbit_deg) * math.sin(2.0 * math.pi * progress)
    radius = float(args.camera_radius)
    camera.location = (
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
    target = Vector((lookaway, 0.0, 0.1 + 0.05 * math.sin(2.0 * math.pi * progress)))
    _look_at(camera, target)


def _verify_frame_sequence(directory: Path, *, expected_frames: int, label: str) -> None:
    frame_count = len(list(directory.glob("*.png")))
    if frame_count != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} {label} frames in {directory}, found {frame_count}")


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

    mask_pixels: list[int] = []
    visible: list[bool] = []
    for frame_index in range(args.frames):
        bpy.context.scene.frame_set(frame_index)
        if args.motion == "camera":
            root.location.x = 0.0
            root.rotation_euler[2] = 0.0
            _set_camera_for_frame(camera, Vector, frame_index, args.frames, window, args)
        else:
            root.location.x = object_x(frame_index, args.frames, window, offscreen_x=args.offscreen_x)
            root.rotation_euler[2] = 0.28 * math.sin(frame_index * 0.12)
        frame_id = f"{frame_index:06d}"
        _render_still(bpy, rgb_dir / f"{frame_id}.png")
        pixels = _render_mask(bpy, mask_dir / f"{frame_id}.png", meshes=meshes, floor=floor, white=white, black=black)
        mask_pixels.append(pixels)
        visible.append(pixels >= args.min_visible_mask_pixels)

    _verify_frame_sequence(rgb_dir, expected_frames=args.frames, label="RGB")
    _verify_frame_sequence(mask_dir, expected_frames=args.frames, label="mask")
    if not has_absent_then_reappearing_visibility(visible, min_absent_frames=args.min_absent_frames):
        runs = visibility_runs(visible)
        raise RuntimeError(
            f"{obj.synset_id}/{obj.model_id} did not produce visible -> absent -> visible masks. "
            f"Visibility runs: {runs}. For camera motion, try increasing --lookaway-x or lowering "
            "--min-visible-mask-pixels. For object motion, try increasing --offscreen-x."
        )

    video_path = None
    mask_video_path = None
    if not args.no_video:
        video_path = _write_video_from_frames(rgb_dir / "%06d.png", output_dir / "video.mp4", fps=args.fps)
        mask_video_path = _write_video_from_frames(mask_dir / "%06d.png", output_dir / "mask.mp4", fps=args.fps)

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
    objects = choose_objects(args)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for index, obj in enumerate(objects):
        output_dir = output_root / f"{index:04d}_{obj.output_name}"
        print(f"[{index + 1}/{len(objects)}] Rendering {obj.synset_id}/{obj.model_id} -> {output_dir}", flush=True)
        metadata = render_one_object(bpy, Vector, obj, output_dir, args)
        results.append({"output_dir": str(output_dir), **metadata})

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps({"videos": results}, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "videos": results}, indent=2), flush=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    render_batch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None))
