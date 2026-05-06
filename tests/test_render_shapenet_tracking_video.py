from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

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
