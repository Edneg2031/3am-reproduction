from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_preprocess_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "preprocess_scannetpp_instance_masks.py"
    spec = importlib.util.spec_from_file_location("preprocess_scannetpp_instance_masks", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(path)


def test_preprocess_scannetpp_writes_instance_label_maps_and_images(tmp_path: Path) -> None:
    preprocess = _load_preprocess_module()
    obj_id_root = tmp_path / "obj_ids"
    scene_obj_dir = obj_id_root / "scene_a"
    scene_obj_dir.mkdir(parents=True)
    torch.save({"obj_ids": torch.tensor([[0, 5, 5], [0, 7, 0]], dtype=torch.int64)}, scene_obj_dir / "000.pth")
    np.save(scene_obj_dir / "001.npy", np.array([[0, 0, 7], [5, 0, 0]], dtype=np.int64))
    data_root = tmp_path / "data"
    _write_image(data_root / "scene_a" / "dslr" / "resized_undistorted_images" / "000.JPG")
    _write_image(data_root / "scene_a" / "dslr" / "resized_undistorted_images" / "001.png")

    summaries = preprocess.preprocess_scannetpp(
        obj_id_root=obj_id_root,
        output_root=tmp_path / "processed",
        data_root=data_root,
        copy_images=True,
    )

    assert summaries == [{"scene_id": "scene_a", "masks": 2, "images": 2, "skipped_empty": 0, "instances": 2}]
    with Image.open(tmp_path / "processed" / "scene_a" / "masks" / "000.png") as image:
        labels = np.asarray(image)
    np.testing.assert_array_equal(labels, np.array([[0, 5, 5], [0, 7, 0]], dtype=np.uint16))
    instances = json.loads((tmp_path / "processed" / "scene_a" / "instances.json").read_text(encoding="utf-8"))
    assert instances["schema"] == "three_am_scannetpp_instances_v1"
    assert [item["id"] for item in instances["instances"]] == [5, 7]
    assert (tmp_path / "processed" / "scene_a" / "images" / "000.jpg").exists()
    assert (tmp_path / "processed" / "scene_a" / "images" / "001.png").exists()


def test_preprocess_scannetpp_rejects_full_frame_singleton_masks(tmp_path: Path) -> None:
    preprocess = _load_preprocess_module()
    obj_id_root = tmp_path / "obj_ids"
    scene_obj_dir = obj_id_root / "scene_a"
    scene_obj_dir.mkdir(parents=True)
    torch.save(torch.ones(4, 4, dtype=torch.int64), scene_obj_dir / "000.pth")

    with pytest.raises(ValueError, match="one positive id covering 1.000"):
        preprocess.preprocess_scannetpp(obj_id_root=obj_id_root, output_root=tmp_path / "processed")


def test_preprocess_scannetpp_keeps_dense_multi_instance_label_maps(tmp_path: Path) -> None:
    preprocess = _load_preprocess_module()
    obj_id_root = tmp_path / "obj_ids"
    scene_obj_dir = obj_id_root / "scene_a"
    scene_obj_dir.mkdir(parents=True)
    torch.save(torch.tensor([[5, 5], [7, 7]], dtype=torch.int64), scene_obj_dir / "000.pth")

    summaries = preprocess.preprocess_scannetpp(obj_id_root=obj_id_root, output_root=tmp_path / "processed")

    assert summaries[0]["masks"] == 1
    with Image.open(tmp_path / "processed" / "scene_a" / "masks" / "000.png") as image:
        labels = np.asarray(image)
    assert set(np.unique(labels).tolist()) == {5, 7}
