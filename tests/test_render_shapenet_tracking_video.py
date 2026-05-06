from __future__ import annotations

import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


def _load_render_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "render_shapenet_tracking_video.py"
    spec = importlib.util.spec_from_file_location("render_shapenet_tracking_video", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_parallel_gpus_accepts_lists_and_auto() -> None:
    module = _load_render_module()

    assert module.parse_parallel_gpus("0,1,2,3") == ["0", "1", "2", "3"]
    assert module.parse_parallel_gpus(" auto ", detected_gpu_ids=["2", "3"]) == ["2", "3"]


def test_parse_parallel_gpus_rejects_invalid_lists() -> None:
    module = _load_render_module()

    with pytest.raises(ValueError, match="comma-separated"):
        module.parse_parallel_gpus("0,,1")
    with pytest.raises(ValueError, match="Invalid GPU id"):
        module.parse_parallel_gpus("-1")
    with pytest.raises(ValueError, match="duplicate"):
        module.parse_parallel_gpus("0,0")
    with pytest.raises(RuntimeError, match="could not detect"):
        module.parse_parallel_gpus("auto", detected_gpu_ids=[])


def _camera_radius_and_elevation(location, center=(0.0, 0.0, 0.1)) -> tuple[float, float]:
    vector = np.asarray(location, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radius = float(np.linalg.norm(vector))
    elevation = float(math.degrees(math.atan2(vector[2], np.linalg.norm(vector[:2]))))
    return radius, elevation


def test_default_camera_path_is_stochastic_two_body(tmp_path: Path) -> None:
    module = _load_render_module()

    args = module.parse_args(["--output-root", str(tmp_path)])

    assert args.camera_path == "stochastic-two-body"


def test_stochastic_two_body_camera_path_is_reproducible_and_bounded(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(["--output-root", str(tmp_path), "--frames", "96"])
    window = module.absence_window(args.frames, start=args.absent_start, length=args.absent_frames)

    poses = module.build_camera_poses(args.frames, window, args, seed=1234)
    poses_again = module.build_camera_poses(args.frames, window, args, seed=1234)
    poses_other = module.build_camera_poses(args.frames, window, args, seed=5678)

    assert poses == poses_again
    assert poses != poses_other
    radius_min, radius_max = module._camera_radius_bounds(args)
    for pose in poses:
        radius, elevation = _camera_radius_and_elevation(pose.location)
        assert radius_min - 1e-8 <= radius <= radius_max + 1e-8
        assert args.camera_elevation_min_deg - 1e-8 <= elevation <= args.camera_elevation_max_deg + 1e-8

    locations = np.asarray([pose.location for pose in poses])
    frame_steps = np.linalg.norm(np.diff(locations, axis=0), axis=1)
    assert float(frame_steps.max()) <= args.camera_max_speed + 1e-6


def test_stochastic_two_body_lookat_moves_toward_absence_target(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(["--output-root", str(tmp_path), "--frames", "96"])
    window = module.absence_window(args.frames, start=args.absent_start, length=args.absent_frames)

    poses = module.build_camera_poses(args.frames, window, args, seed=1234)
    target_x = np.asarray([pose.target[0] for pose in poses])

    assert float(np.abs(target_x[: max(1, window.start - args.lookaway_transition_frames)]).max()) < 1.0
    assert float(target_x[window.start : window.end].max()) > args.lookaway_x * 0.55
    assert abs(float(target_x[-1])) < 1.0


def test_camera_seed_changes_by_object_identity(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(["--output-root", str(tmp_path)])
    obj_a = module.ShapeNetObject("syn", "a", Path("/tmp/a.obj"))
    obj_b = module.ShapeNetObject("syn", "b", Path("/tmp/b.obj"))

    assert module._camera_seed(args, obj_a) != module._camera_seed(args, obj_b)


def test_human_linear_camera_path_is_reproducible_and_linear_without_jitter(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--frames",
            "96",
            "--camera-path",
            "human-linear",
            "--camera-jitter-std",
            "0",
            "--camera-aim-jitter-std",
            "0",
            "--camera-roll-jitter-deg",
            "0",
        ]
    )
    window = module.absence_window(args.frames, start=args.absent_start, length=args.absent_frames)

    poses = module.build_camera_poses(args.frames, window, args, seed=1234)
    poses_again = module.build_camera_poses(args.frames, window, args, seed=1234)

    assert poses == poses_again
    np.testing.assert_allclose(poses[0].location, (-0.65, -4.29, 1.06), atol=1e-8)
    np.testing.assert_allclose(poses[-1].location, (0.65, -4.11, 1.18), atol=1e-8)
    np.testing.assert_allclose(poses[0].target, (0.0, 0.0, 0.1), atol=1e-8)
    np.testing.assert_allclose(poses[-1].target, (0.0, 0.0, 0.1), atol=1e-8)
    assert all(pose.roll_rad == 0.0 for pose in poses)


def test_human_linear_camera_path_adds_seeded_smooth_jitter(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(["--output-root", str(tmp_path), "--frames", "96", "--camera-path", "human-linear"])
    window = module.absence_window(args.frames, start=args.absent_start, length=args.absent_frames)

    poses_a = module.build_camera_poses(args.frames, window, args, seed=1234)
    poses_b = module.build_camera_poses(args.frames, window, args, seed=1234)
    poses_c = module.build_camera_poses(args.frames, window, args, seed=5678)

    assert poses_a == poses_b
    assert poses_a != poses_c
    locations = np.asarray([pose.location for pose in poses_a])
    frame_deltas = np.linalg.norm(np.diff(locations, axis=0), axis=1)
    assert float(frame_deltas.max()) < 0.2
    assert any(abs(pose.roll_rad) > 0.0 for pose in poses_a[1:-1])


def test_legacy_orbit_camera_path_still_generates_poses(tmp_path: Path) -> None:
    module = _load_render_module()
    args = module.parse_args(["--output-root", str(tmp_path), "--frames", "96", "--camera-path", "orbit"])
    window = module.absence_window(args.frames, start=args.absent_start, length=args.absent_frames)

    poses = module.build_camera_poses(args.frames, window, args, seed=1234)

    assert len(poses) == args.frames
    assert poses[0].location != poses[len(poses) // 2].location


def test_shard_indexed_objects_is_stable_without_duplicates() -> None:
    module = _load_render_module()
    items = [
        module.IndexedShapeNetObject(index, module.ShapeNetObject("syn", f"model_{index}", Path(f"/tmp/{index}.obj")))
        for index in range(10)
    ]

    shards = [module.shard_indexed_objects(items, worker_id=worker_id, worker_count=3) for worker_id in range(3)]
    flattened = [item.selected_index for shard in shards for item in shard]

    assert sorted(flattened) == list(range(10))
    assert [item.selected_index for item in shards[0]] == [0, 3, 6, 9]
    assert [item.selected_index for item in shards[1]] == [1, 4, 7]
    assert [item.selected_index for item in shards[2]] == [2, 5, 8]


def test_build_worker_specs_strips_parent_options_and_sets_cuda_visible_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_render_module()
    monkeypatch.setattr(module, "parse_parallel_gpus", lambda value: ["0", "1"])
    monkeypatch.setattr(module.os, "getpid", lambda: 12345)
    args = Namespace(
        parallel_gpus="0,1",
        workers_per_gpu=2,
        output_root=str(tmp_path / "out"),
        blender_bin="/opt/blender/blender",
    )
    argv = [
        "--shapenet-root",
        "/data/ShapeNetCore.v2",
        "--output-root",
        str(tmp_path / "out"),
        "--num-videos",
        "5",
        "--parallel-gpus",
        "0,1",
        "--workers-per-gpu=2",
        "--blender-bin",
        "/opt/blender/blender",
        "--engine",
        "workbench",
    ]

    specs = module.build_worker_specs(args, argv, selected_count=3)

    assert len(specs) == 3
    assert [spec.gpu_id for spec in specs] == ["0", "1", "0"]
    assert [spec.env["CUDA_VISIBLE_DEVICES"] for spec in specs] == ["0", "1", "0"]
    for worker_id, spec in enumerate(specs):
        command = spec.command
        assert command[:4] == ["/opt/blender/blender", "-b", "--python", str(Path(module.__file__).resolve())]
        assert "--parallel-gpus" not in command
        assert not any(token.startswith("--workers-per-gpu") for token in command)
        assert "--blender-bin" not in command
        assert command[-6:] == [
            "--worker-id",
            str(worker_id),
            "--worker-count",
            "3",
            "--worker-summary-path",
            str(tmp_path / "out" / ".worker_summaries" / "run_12345" / f"worker_{worker_id:03d}.json"),
        ]


def test_merge_worker_summaries_sorts_and_writes_final_summary(tmp_path: Path) -> None:
    module = _load_render_module()
    summary_a = tmp_path / "worker_a.json"
    summary_b = tmp_path / "worker_b.json"
    summary_a.write_text(json.dumps({"videos": [{"index": 2, "output_dir": "c"}, {"index": 0, "output_dir": "a"}]}), encoding="utf-8")
    summary_b.write_text(json.dumps({"videos": [{"index": 1, "output_dir": "b"}]}), encoding="utf-8")

    videos = module.merge_worker_summaries([summary_a, summary_b], tmp_path)

    assert [item["index"] for item in videos] == [0, 1, 2]
    final_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [item["output_dir"] for item in final_summary["videos"]] == ["a", "b", "c"]


def test_merge_worker_summaries_rejects_missing_or_duplicate_indices(tmp_path: Path) -> None:
    module = _load_render_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"videos": [{"index": 0}, {"index": 0}]}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not written"):
        module.merge_worker_summaries([tmp_path / "missing.json"], tmp_path)
    with pytest.raises(RuntimeError, match="Duplicate"):
        module.merge_worker_summaries([duplicate], tmp_path)
