from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord


class FakeSam2Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.memory_attention = nn.Conv2d(4, 4, kernel_size=1)
        self.mask_decoder = nn.Conv2d(4, 1, kernel_size=1)

    def freeze_image_encoder(self) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)

    def encode_sam_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.memory_attention(self.image_encoder(images))

    def forward_train_sequence(self, batch, merged_features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mask_decoder(merged_features)[:, 0]
        pooled = logits.mean(dim=(1, 2))
        return {
            "mask_logits": logits,
            "iou_scores": torch.sigmoid(pooled),
            "occlusion_logits": torch.stack([-pooled, pooled], dim=1),
        }


def _load_train_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "train_3am.py"
    spec = importlib.util.spec_from_file_location("train_3am_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(path)


def _write_tiny_training_fixture(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "data" / "processed" / "scannetpp" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [SceneRecord("scannetpp", "scene_a", "train", tuple(frames))])
    feature_dir = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    feature_dir.mkdir(parents=True)
    for frame in frames:
        torch.save(torch.randn(2, 4, 4), feature_dir / f"{frame.frame_id}_level0.pt")
    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "external": {
            "sam2_checkpoint": "outputs/checkpoints/sam2.pt",
            "sam2_config": "external/sam2.yaml",
            "must3r_checkpoint": "outputs/checkpoints/must3r.pt",
        },
        "datasets": {
            "scannetpp": {
                "manifest": "data/processed/scannetpp_manifest.json",
                "fov_sampling_probability": 0.0,
            }
        },
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {"cache_root": "outputs/must3r_features"},
        "model": {"sam_channels": 4, "must3r_channels": [2], "hidden_channels": 4, "attention_heads": 2},
        "training": {
            "iterations": 2,
            "batch_size": 1,
            "learning_rates": {"memory_attention": 1e-3, "mask_decoder": 1e-3, "feature_merger": 1e-3},
            "memory_frames": 2,
            "log_every": 1,
            "checkpoint_every": 1,
            "amp": False,
            "checkpoint_out": "outputs/checkpoints/final.pt",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_training_script_runs_with_fake_adapter_and_resumes(tmp_path: Path) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path)

    first_step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
    )
    assert first_step == 1
    latest = tmp_path / "outputs" / "checkpoints" / "latest.pt"
    assert latest.exists()

    second_step = train_3am.run_training(
        str(config_path),
        iterations=2,
        resume=latest,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
    )
    assert second_step == 2
    payload = torch.load(tmp_path / "outputs" / "checkpoints" / "final.pt", weights_only=False)
    assert payload["step"] == 2


def test_training_script_dry_run_reports_inputs(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path)

    assert train_3am.run_training(str(config_path), dry_run_only=True) == 0
    captured = capsys.readouterr()

    assert "scannetpp_manifest.json" in captured.out
    assert "feature_cache_root" in captured.out
