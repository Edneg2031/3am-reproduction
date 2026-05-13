from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from three_am.data.io import read_manifest


def _load_prepare_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "prepare_scannetpp_training_data.py"
    spec = importlib.util.spec_from_file_location("prepare_scannetpp_training_data", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(path)


def test_prepare_scannetpp_builds_smoke_manifest_from_scene_list(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    obj_id_root = tmp_path / "obj_ids"
    for scene_id in ("scene_a", "scene_b"):
        scene_obj_dir = obj_id_root / scene_id
        scene_obj_dir.mkdir(parents=True)
        torch.save({"obj_ids": torch.tensor([[0, 5, 5], [0, 7, 0]], dtype=torch.int64)}, scene_obj_dir / "000.pth")
        _write_image(tmp_path / "raw" / scene_id / "dslr" / "resized_undistorted_images" / "000.JPG")
    scene_list = tmp_path / "smoke_scenes.txt"
    scene_list.write_text("scene_b\n", encoding="utf-8")

    options = prepare.PrepareOptions(
        obj_id_root=obj_id_root,
        data_root=tmp_path / "raw",
        output_root=tmp_path / "processed" / "scannetpp",
        manifest_output=tmp_path / "processed" / "scannetpp_manifest.json",
        split="train",
        scene_list=scene_list,
        image_subdir="dslr/resized_undistorted_images",
        copy_images=True,
        min_visible_pixels=1,
        max_foreground_ratio=0.98,
        require_cameras=False,
        precompute_must3r=False,
        config=tmp_path / "config.yaml",
        feature_output_dir=tmp_path / "features",
        device="cpu",
        weights=None,
        must3r_repo=None,
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=8,
        full_scene_memory=False,
        cache_dtype="bf16",
        feature_layers=("encoder", 4, 7, 11),
        limit_scenes=None,
        dry_run_precompute=False,
    )

    payload = prepare.prepare_scannetpp_training_data(options)

    assert payload["manifest"]["scenes"] == 1
    assert payload["manifest"]["frames"] == 1
    assert payload["audit"]["instances"] == 2
    scenes = read_manifest(options.manifest_output)
    assert [scene.scene_id for scene in scenes] == ["scene_b"]
    assert scenes[0].instances_path is not None
    assert scenes[0].instances_path.exists()
    assert scenes[0].frames[0].image_path.name == "000.jpg"
    with Image.open(scenes[0].frames[0].mask_path) as image:  # type: ignore[arg-type]
        labels = np.asarray(image)
    assert labels.dtype == np.uint16
    assert set(np.unique(labels).tolist()) == {0, 5, 7}


def test_prepare_scannetpp_rejects_full_frame_singleton_masks(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    scene_obj_dir = tmp_path / "obj_ids" / "scene_a"
    scene_obj_dir.mkdir(parents=True)
    torch.save(torch.ones(4, 4, dtype=torch.int64), scene_obj_dir / "000.pth")
    _write_image(tmp_path / "raw" / "scene_a" / "dslr" / "resized_undistorted_images" / "000.JPG")

    options = prepare.PrepareOptions(
        obj_id_root=tmp_path / "obj_ids",
        data_root=tmp_path / "raw",
        output_root=tmp_path / "processed" / "scannetpp",
        manifest_output=tmp_path / "processed" / "scannetpp_manifest.json",
        split="train",
        scene_list=None,
        image_subdir="dslr/resized_undistorted_images",
        copy_images=True,
        min_visible_pixels=1,
        max_foreground_ratio=0.98,
        require_cameras=False,
        precompute_must3r=False,
        config=tmp_path / "config.yaml",
        feature_output_dir=tmp_path / "features",
        device="cpu",
        weights=None,
        must3r_repo=None,
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=8,
        full_scene_memory=False,
        cache_dtype="bf16",
        feature_layers=("encoder", 4, 7, 11),
        limit_scenes=None,
        dry_run_precompute=False,
    )

    with pytest.raises(ValueError, match="one positive id covering 1.000"):
        prepare.prepare_scannetpp_training_data(options)


def test_prepare_scannetpp_can_dry_run_must3r_precompute(tmp_path: Path) -> None:
    prepare = _load_prepare_module()
    scene_obj_dir = tmp_path / "obj_ids" / "scene_a"
    scene_obj_dir.mkdir(parents=True)
    torch.save({"obj_ids": torch.tensor([[0, 5], [7, 0]], dtype=torch.int64)}, scene_obj_dir / "000.pth")
    _write_image(tmp_path / "raw" / "scene_a" / "dslr" / "resized_undistorted_images" / "000.JPG")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"project_root: {tmp_path}",
                "paths:",
                "  data_processed: processed",
                "  checkpoints: checkpoints",
                "  outputs: outputs",
                "external:",
                "  must3r_checkpoint: checkpoints/must3r.pt",
                "  must3r_repo: external/must3r",
            ]
        ),
        encoding="utf-8",
    )

    options = prepare.PrepareOptions(
        obj_id_root=tmp_path / "obj_ids",
        data_root=tmp_path / "raw",
        output_root=tmp_path / "processed" / "scannetpp",
        manifest_output=tmp_path / "processed" / "scannetpp_manifest.json",
        split="train",
        scene_list=None,
        image_subdir="dslr/resized_undistorted_images",
        copy_images=True,
        min_visible_pixels=1,
        max_foreground_ratio=0.98,
        require_cameras=False,
        precompute_must3r=True,
        config=config,
        feature_output_dir=Path("outputs/must3r_features"),
        device="cpu",
        weights=None,
        must3r_repo=None,
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=8,
        full_scene_memory=False,
        cache_dtype="bf16",
        feature_layers=("encoder", 4, 7, 11),
        limit_scenes=1,
        dry_run_precompute=True,
    )

    payload = prepare.prepare_scannetpp_training_data(options)

    assert payload["precompute"]["dry_run"] is True
    assert payload["precompute"]["scenes"] == 1
    assert payload["precompute"]["frames"] == 1
    assert payload["precompute"]["output_dir"] == str(tmp_path / "outputs" / "must3r_features")
