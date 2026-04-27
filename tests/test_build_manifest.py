from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run_build_manifest(root: Path, output: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_manifest.py"),
            "--dataset",
            "scannetpp",
            "--root",
            str(root),
            "--split",
            "train",
            "--output",
            str(output),
            *args,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check:
        result.check_returncode()
    return result


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_text(path: Path, text: str = "1 0 0\n0 1 0\n0 0 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _transform(file_path: str, tx: float) -> dict[str, object]:
    return {
        "file_path": file_path,
        "transform_matrix": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_nerfstudio_3dgs_manifest_writes_sidecar_cameras(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    scene = root / "scene_a"
    for frame_id in ("000", "001", "002"):
        _write_image(scene / "images" / f"{frame_id}.png")
        _write_image(scene / "masks" / f"{frame_id}.png")
    transforms = {
        "fl_x": 100.0,
        "fl_y": 110.0,
        "cx": 40.0,
        "cy": 50.0,
        "frames": [
            _transform("images/000.png", 0.0),
            _transform("./images/001.png", 1.0),
            _transform("002.png", 2.0),
        ],
    }
    _write_text(scene / "nerfstudio" / "transforms.json", json.dumps(transforms))

    output = tmp_path / "manifest.json"
    _run_build_manifest(root, output, "--format", "nerfstudio_3dgs")

    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest_scene = manifest["scenes"][0]
    sidecar_root = output.parent / "manifest_cameras" / "scene_a"
    assert manifest_scene["instances_path"] is None
    assert len(manifest_scene["frames"]) == 3
    for frame in manifest_scene["frames"]:
        assert frame["mask_path"] is not None
        assert frame["depth_path"] is None
        assert frame["pose_path"] is not None
        assert frame["intrinsics_path"] is not None
        assert sidecar_root in Path(frame["pose_path"]).parents
        assert sidecar_root in Path(frame["intrinsics_path"]).parents

    intrinsics = np.loadtxt(manifest_scene["frames"][0]["intrinsics_path"])
    pose = np.loadtxt(manifest_scene["frames"][0]["pose_path"])
    assert intrinsics.shape == (3, 3)
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(intrinsics, np.array([[100.0, 0.0, 40.0], [0.0, 110.0, 50.0], [0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(pose, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_auto_manifest_preserves_normalized_camera_paths(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    scene = root / "scene_b"
    _write_image(scene / "images" / "000.png")
    _write_image(scene / "masks" / "000.png")
    _write_image(scene / "depth" / "000.png")
    pose_path = scene / "poses" / "000.txt"
    intrinsics_path = scene / "intrinsics" / "000.txt"
    _write_text(pose_path, "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    _write_text(intrinsics_path)

    output = tmp_path / "normalized_manifest.json"
    _run_build_manifest(root, output, "--format", "auto")

    manifest = json.loads(output.read_text(encoding="utf-8"))
    frame = manifest["scenes"][0]["frames"][0]
    assert frame["depth_path"] == str(scene / "depth" / "000.png")
    assert frame["pose_path"] == str(pose_path)
    assert frame["intrinsics_path"] == str(intrinsics_path)
    assert not (output.parent / "normalized_manifest_cameras").exists()


def test_require_cameras_fails_when_pose_or_intrinsics_missing(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    scene = root / "scene_c"
    _write_image(scene / "images" / "000.png")

    output = tmp_path / "missing_manifest.json"
    result = _run_build_manifest(root, output, "--format", "nerfstudio_3dgs", "--require-cameras", check=False)

    assert result.returncode != 0
    assert "missing pose/intrinsics" in result.stderr
