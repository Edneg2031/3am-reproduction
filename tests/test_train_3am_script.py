from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from three_am.data.io import write_manifest
from three_am.data.schema import FrameRecord, SceneRecord
from three_am.models.feature_merger import Must3rFeatureBundle
from three_am.training.dataset import Prompt


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

    def track_masks_from_points(
        self,
        images: torch.Tensor,
        *,
        reference_index: int,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        masks = torch.full((images.shape[0], images.shape[-2], images.shape[-1]), -12.0, device=images.device)
        x = int(points[0, 0].detach().cpu().item())
        y = int(points[0, 1].detach().cpu().item())
        x0, x1 = max(0, x - 1), min(images.shape[-1], x + 2)
        y0, y1 = max(0, y - 1), min(images.shape[-2], y + 2)
        masks[:, y0:y1, x0:x1] = 12.0
        return masks


class FakeMust3rAdapter(nn.Module):
    model = object()

    def __init__(self) -> None:
        super().__init__()
        self.seen_paths = None

    def extract_features(self, images: torch.Tensor, *, image_paths=None) -> tuple[torch.Tensor, ...]:
        self.seen_paths = image_paths
        assert image_paths is not None
        return (torch.ones(images.shape[0], 2, images.shape[-2], images.shape[-1], device=images.device),)


class FakeMust3rBundleAdapter(FakeMust3rAdapter):
    def extract_features(self, images: torch.Tensor, *, image_paths=None) -> Must3rFeatureBundle:
        self.seen_paths = image_paths
        assert image_paths is not None
        levels = (torch.ones(images.shape[0], 2, images.shape[-2], images.shape[-1], device=images.device),)
        return Must3rFeatureBundle(
            levels=levels,
            pe2d=torch.zeros(images.shape[0], 2, images.shape[-2], images.shape[-1], device=images.device),
            point_map=torch.zeros(images.shape[0], 3, images.shape[-2], images.shape[-1], device=images.device),
            ray_map=torch.ones(images.shape[0], 3, images.shape[-2], images.shape[-1], device=images.device),
            metadata={"decoder_memory": True},
        )


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


def _write_mask(path: Path, *, full: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full((4, 4), 255, dtype=np.uint8) if full else np.zeros((4, 4), dtype=np.uint8)
    if not full:
        mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(path)


def _write_tiny_training_fixture(tmp_path: Path, *, write_feature_cache: bool = True, full_masks: bool = False) -> Path:
    scene_dir = tmp_path / "data" / "processed" / "scannetpp" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        image_path = scene_dir / "images" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, full=full_masks)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    instances_path = scene_dir / "instances.json"
    instances_path.write_text(
        '{"schema":"three_am_scannetpp_instances_v1","instances":[{"id":1},{"id":255}]}',
        encoding="utf-8",
    )
    manifest = tmp_path / "data" / "processed" / "scannetpp_manifest.json"
    write_manifest(manifest, [SceneRecord("scannetpp", "scene_a", "train", tuple(frames), instances_path=instances_path)])
    feature_dir = tmp_path / "outputs" / "must3r_features" / "scannetpp" / "scene_a"
    if write_feature_cache:
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


def _write_strict_training_fixture(tmp_path: Path) -> Path:
    config_path = _write_tiny_training_fixture(tmp_path, write_feature_cache=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["strict_paper"] = True
    config["training"]["validate_every"] = 1
    config["features"]["online"] = True
    config["features"]["require_decoder_memory"] = False
    config["model"]["geometry_channels"] = 4
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_shapenet_training_fixture(tmp_path: Path, *, num_frames: int = 5) -> Path:
    scene_dir = tmp_path / "data" / "processed" / "shapenet_tracking" / "scene_a"
    frames: list[FrameRecord] = []
    for index in range(num_frames):
        frame_id = f"{index:06d}"
        image_path = scene_dir / "rgb" / f"{frame_id}.png"
        mask_path = scene_dir / "masks" / f"{frame_id}.png"
        _write_image(image_path)
        _write_mask(mask_path, full=False)
        frames.append(FrameRecord(frame_id=frame_id, image_path=image_path, mask_path=mask_path))
    manifest = tmp_path / "data" / "processed" / "shapenet_manifest.json"
    write_manifest(manifest, [SceneRecord("shapenet", "scene_a", "train", tuple(frames))])
    feature_dir = tmp_path / "outputs" / "must3r_features" / "shapenet" / "scene_a"
    feature_dir.mkdir(parents=True)
    for frame in frames:
        torch.save(torch.randn(2, 4, 4), feature_dir / f"{frame.frame_id}_level0.pt")
    (feature_dir / "metadata.json").write_text(
        '{"decoder_memory": false, "feature_specs": ["encoder"], "feature_channels": [2]}',
        encoding="utf-8",
    )
    config = {
        "project_root": str(tmp_path),
        "paths": {"data_processed": "data/processed", "checkpoints": "outputs/checkpoints", "outputs": "outputs"},
        "external": {
            "sam2_checkpoint": "outputs/checkpoints/sam2.pt",
            "sam2_config": "external/sam2.yaml",
            "must3r_checkpoint": "outputs/checkpoints/must3r.pt",
        },
        "training": {
            "datasets": ["shapenet"],
            "strict_paper": True,
            "iterations": 1,
            "batch_size": 1,
            "learning_rates": {"memory_attention": 1e-3, "mask_decoder": 1e-3, "feature_merger": 1e-3},
            "memory_frames": 8,
            "log_every": 1,
            "checkpoint_every": 0,
            "save_model_every": 0,
            "amp": False,
            "checkpoint_out": "outputs/checkpoints/final.pt",
            "visualize_every": 1,
            "visualization_full_video": True,
            "visualization_chunk_size": 2,
            "visualization_max_frames": 4,
            "visualization_max_side": 64,
            "visualization_fps": 6,
        },
        "datasets": {
            "shapenet": {
                "manifest": "data/processed/shapenet_manifest.json",
                "fov_sampling_probability": 0.0,
                "sequence_length_min": 2,
                "sequence_length_max": 2,
                "dynamic_resize": {"enabled": False},
                "prompt_mask_augment": {"enabled": False},
            }
        },
        "sampling": {"sequence_length": 2, "fov_threshold": 0.25},
        "features": {
            "cache_root": "outputs/must3r_features",
            "online": False,
            "require_decoder_memory": False,
            "feature_layers": "encoder",
        },
        "model": {
            "sam_image_size": 32,
            "sam_channels": 4,
            "must3r_channels": [2],
            "hidden_channels": 4,
            "geometry_channels": 4,
            "attention_heads": 2,
        },
    }
    config_path = tmp_path / "shapenet_config.yaml"
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
    model_latest = tmp_path / "outputs" / "checkpoints" / "model_latest.pt"
    assert latest.exists()
    assert model_latest.exists()
    checkpoint_payload = torch.load(latest, weights_only=False)
    model_payload = torch.load(model_latest, weights_only=False)
    assert checkpoint_payload["type"] == "three_am_training_checkpoint"
    assert checkpoint_payload["model_state_type"] == "trainable"
    assert "optimizer" in checkpoint_payload
    assert model_payload["type"] == "three_am_model_weights"
    assert "optimizer" not in model_payload
    assert any(key.startswith("core.feature_merger") for key in model_payload["model"])

    second_step = train_3am.run_training(
        str(config_path),
        iterations=2,
        resume=latest,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
    )
    assert second_step == 2

    third_step = train_3am.run_training(
        str(config_path),
        iterations=3,
        auto_resume=True,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
    )
    assert third_step == 3
    payload = torch.load(tmp_path / "outputs" / "checkpoints" / "final.pt", weights_only=False)
    assert payload["step"] == 3
    exported = torch.load(tmp_path / "outputs" / "checkpoints" / "3am_model.pt", weights_only=False)
    assert exported["step"] == 3


def test_training_script_can_use_online_must3r_when_cache_is_missing(tmp_path: Path) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, write_feature_cache=False)
    must3r = FakeMust3rAdapter()

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        online_must3r=True,
        sam2_adapter=FakeSam2Adapter(),
        must3r_adapter=must3r,
    )

    assert step == 1
    assert must3r.seen_paths is not None
    assert [path.name for path in must3r.seen_paths] == ["000.png", "001.png"]


def test_training_script_uses_config_online_must3r_by_default(tmp_path: Path) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, write_feature_cache=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["features"]["online"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    must3r = FakeMust3rAdapter()

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
        must3r_adapter=must3r,
    )

    assert step == 1
    assert must3r.seen_paths is not None


def test_training_script_can_force_offline_must3r_from_online_config(tmp_path: Path) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, write_feature_cache=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["features"]["online"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(train_3am.FeatureCacheRuntimeError) as excinfo:
        train_3am.run_training(
            str(config_path),
            iterations=1,
            device_name="cpu",
            online_must3r=False,
            sam2_adapter=FakeSam2Adapter(),
            must3r_adapter=FakeMust3rAdapter(),
        )

    assert "MUSt3R feature cache is missing" in str(excinfo.value)


def test_training_replaces_full_scannetpp_masks_with_sam2_point_pseudo_masks(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, full_masks=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["sam2_point_pseudo_masks"] = {
        "mode": "auto",
        "datasets": ["scannetpp"],
        "auto_foreground_ratio_threshold": 0.98,
        "max_attempts": 1,
        "min_foreground_ratio": 0.001,
        "max_foreground_ratio": 0.9,
        "allow_out_of_range_fallback": False,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        log_every=1,
        checkpoint_every=0,
        save_model_every=0,
        sam2_adapter=FakeSam2Adapter(),
    )
    captured = capsys.readouterr()

    assert "target_source=sam2_point_pseudo_mask_auto_full_dataset_mask" in captured.out


def test_training_keeps_normal_scannetpp_dataset_masks_by_default(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, full_masks=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["sam2_point_pseudo_masks"] = {
        "mode": "auto",
        "datasets": ["scannetpp"],
        "auto_foreground_ratio_threshold": 0.98,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        log_every=1,
        checkpoint_every=0,
        save_model_every=0,
        sam2_adapter=FakeSam2Adapter(),
    )
    captured = capsys.readouterr()

    assert "target_source=dataset_mask" in captured.out


def test_online_must3r_dependency_failure_reports_underlying_error(tmp_path: Path) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path, write_feature_cache=False)

    class FailingMust3rAdapter(nn.Module):
        model = None

        def load(self) -> None:
            raise train_3am.ExternalDependencyError("No module named 'mast3r'")

    with pytest.raises(train_3am.FeatureCacheRuntimeError) as excinfo:
        train_3am.run_training(
            str(config_path),
            iterations=1,
            device_name="cpu",
            online_must3r=True,
            sam2_adapter=FakeSam2Adapter(),
            must3r_adapter=FailingMust3rAdapter(),
        )

    message = str(excinfo.value)
    assert "online MUSt3R extraction is unavailable" in message
    assert "No module named 'mast3r'" in message
    assert "mast3r_importable" in message


def test_strict_training_script_runs_with_bundle_and_validation(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_strict_training_fixture(tmp_path)
    must3r = FakeMust3rBundleAdapter()

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        online_must3r=True,
        sam2_adapter=FakeSam2Adapter(),
        must3r_adapter=must3r,
        strict_paper=True,
    )
    captured = capsys.readouterr()

    assert step == 1
    assert "training_split" in captured.out
    assert "validation_step=1" in captured.out


def test_training_script_validate_every_override_enables_validation(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_strict_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["validate_every"] = 0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        validate_every=1,
        online_must3r=True,
        sam2_adapter=FakeSam2Adapter(),
        must3r_adapter=FakeMust3rBundleAdapter(),
        strict_paper=True,
    )
    captured = capsys.readouterr()

    assert step == 1
    assert "validate_every=1" in captured.out
    assert "validation_step=1" in captured.out


def test_training_script_writes_validation_visualization_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    train_3am = _load_train_module()
    config_path = _write_strict_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["validation_visualize_every"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    visualization_dir = tmp_path / "outputs" / "visualizations" / "train"
    monkeypatch.setattr(train_3am.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def _fake_run(command, check):
        Path(command[-1]).write_bytes(b"fake validation mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_3am.subprocess, "run", _fake_run)

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        online_must3r=True,
        sam2_adapter=FakeSam2Adapter(),
        must3r_adapter=FakeMust3rBundleAdapter(),
        strict_paper=True,
    )
    captured = capsys.readouterr()

    validation_png = visualization_dir / "step_0000001_scannetpp_scene_a_validation.png"
    validation_mp4 = visualization_dir / "step_0000001_scannetpp_scene_a_validation.mp4"
    assert step == 1
    assert validation_png.exists()
    assert validation_mp4.read_bytes() == b"fake validation mp4"
    assert "validation_visualization_summary=" in captured.out
    assert "validation_visualization_video=" in captured.out


def test_training_script_splits_scenes_into_train_and_validation() -> None:
    train_3am = _load_train_module()
    scenes = tuple(
        SceneRecord(dataset="shapenet", scene_id=f"scene_{index}", split="train", frames=tuple())
        for index in range(4)
    )

    train_scenes, validation_scenes = train_3am._split_training_scenes(
        scenes,
        validation_fraction=0.25,
        validation_seed=0,
    )

    assert len(train_scenes) == 3
    assert len(validation_scenes) == 1
    assert {scene.scene_id for scene in train_scenes}.isdisjoint({scene.scene_id for scene in validation_scenes})
    assert sorted(scene.scene_id for scene in train_scenes + validation_scenes) == [f"scene_{index}" for index in range(4)]


def test_training_script_writes_visualization_png_and_mp4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path)
    visualization_dir = tmp_path / "outputs" / "visualizations" / "train"
    monkeypatch.setattr(train_3am.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def _fake_run(command, check):
        Path(command[-1]).write_bytes(b"fake mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_3am.subprocess, "run", _fake_run)

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        visualize_every=1,
        visualization_dir=visualization_dir,
        visualization_fps=6,
        sam2_adapter=FakeSam2Adapter(),
    )

    png_files = sorted(visualization_dir.glob("step_*.png"))
    mp4_files = sorted(visualization_dir.glob("step_*.mp4"))
    assert step == 1
    assert len(png_files) == 1
    assert len(mp4_files) == 1
    assert mp4_files[0].read_bytes() == b"fake mp4"
    with Image.open(png_files[0]) as image:
        assert image.size[0] >= 720
        assert image.size[1] > 58


def test_training_script_writes_full_scene_visualization_video_for_shapenet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    train_3am = _load_train_module()
    config_path = _write_shapenet_training_fixture(tmp_path, num_frames=5)
    visualization_dir = tmp_path / "outputs" / "visualizations" / "train"
    monkeypatch.setattr(train_3am.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    video_frame_counts: dict[str, int] = {}

    def _fake_run(command, check):
        pattern = Path(command[command.index("-i") + 1])
        video_frame_counts[Path(command[-1]).name] = len(list(pattern.parent.glob("*.png")))
        Path(command[-1]).write_bytes(b"fake mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_3am.subprocess, "run", _fake_run)

    step = train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        sam2_adapter=FakeSam2Adapter(),
    )
    captured = capsys.readouterr()

    batch_mp4 = visualization_dir / "step_0000001_shapenet_scene_a.mp4"
    full_scene_mp4 = visualization_dir / "step_0000001_shapenet_scene_a_full_scene.mp4"
    full_scene_png = visualization_dir / "step_0000001_shapenet_scene_a_full_scene.png"
    assert step == 1
    assert batch_mp4.exists()
    assert full_scene_mp4.exists()
    assert full_scene_png.exists()
    assert video_frame_counts[batch_mp4.name] == 2
    assert video_frame_counts[full_scene_mp4.name] == 5
    assert "visualization_full_scene_video=" in captured.out


def test_training_visualization_header_lines_include_tracking_context() -> None:
    train_3am = _load_train_module()

    lines = train_3am._training_visualization_header_lines(
        step=12,
        dataset="scannetpp",
        scene_id="scene_a",
        prompt_type="mask",
        reference_frame="001",
        frame_ids=("000", "001", "002"),
        batch_iou_empty_is_one=0.5,
        visible_iou=0.25,
        non_ref_visible_iou=0.125,
        ref_iou=1.0,
        tracking_recall=1.0,
        visible_frames=2,
        empty_empty_frames=1,
        target_areas=(0, 4, 8),
        has_object_flags=(False, True, True),
        instance_id=7,
        target_source="sam2_point_pseudo_mask_auto_full_dataset_mask",
    )
    text = "\n".join(lines)

    assert "3AM tracking batch" in text
    assert "target_source=sam2_point_pseudo_mask_auto_full_dataset_mask" in text
    assert "prompt=mask" in text
    assert "ref=001" in text
    assert "frames=[000, 001, 002]" in text
    assert "batch_iou_empty_is_one=0.500" in text
    assert "visible_iou=0.250" in text
    assert "non_ref_visible_iou=0.125" in text
    assert "ref_iou=1.000" in text
    assert "tracking_recall_like=1.000" in text
    assert "visible_frames=2/3" in text
    assert "empty_empty_frames=1" in text
    assert "instance_id=7" in text
    assert "has_object=011" in text
    assert "target_areas=[0,4,8]" in text


def test_training_visualization_diagnostics_separates_empty_empty_from_visible_signal() -> None:
    train_3am = _load_train_module()
    probabilities = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    targets = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )

    diagnostics = train_3am._visualization_diagnostics(
        probabilities,
        targets,
        has_object=torch.tensor([False, True, True]),
        reference_index=1,
    )

    assert diagnostics["batch_iou_empty_is_one"] == pytest.approx(2 / 3)
    assert diagnostics["visible_iou"] == pytest.approx(0.5)
    assert diagnostics["non_ref_visible_iou"] == pytest.approx(0.0)
    assert diagnostics["ref_iou"] == pytest.approx(1.0)
    assert diagnostics["empty_empty_frames"] == 1
    assert diagnostics["visible_frames"] == 2
    assert diagnostics["target_areas"] == [0, 1, 1]


def test_training_visualization_frame_order_starts_from_reference() -> None:
    train_3am = _load_train_module()

    assert train_3am._visualization_frame_order(5, 2, 4) == [2, 3, 4, 1]
    assert train_3am._visualization_frame_order(3, 99, 3) == [2, 1, 0]


def test_training_visualization_video_frame_order_is_contiguous() -> None:
    train_3am = _load_train_module()

    assert train_3am._visualization_video_frame_order(5, 2, 4) == [0, 1, 2, 3]
    assert train_3am._visualization_video_frame_order(8, 6, 4) == [4, 5, 6, 7]
    assert train_3am._visualization_video_frame_order(3, 99, 8) == [0, 1, 2]


def test_training_loss_differentiability_check_reports_batch_context() -> None:
    train_3am = _load_train_module()
    batch = train_3am.TrainingBatch(
        images=torch.zeros(1, 3, 4, 4),
        target_masks=torch.zeros(1, 4, 4),
        prompt=Prompt(type="mask", frame_index=0, mask=torch.zeros(4, 4)),
        must3r_features=None,
        dataset="scannetpp",
        scene_id="scene_a",
        frame_ids=("000",),
        image_paths=(),
        has_object=torch.tensor([False]),
        sampling_mode="fov",
    )

    with pytest.raises(RuntimeError) as excinfo:
        train_3am._ensure_loss_is_differentiable(
            torch.tensor(1.0),
            wrapper=type("Wrapper", (), {"sam2_adapter": object(), "named_parameters": lambda self: iter(())})(),
            outputs={"mask_logits": torch.zeros(1, 4, 4)},
            batch=batch,
        )

    message = str(excinfo.value)
    assert "batch_frames=['000']" in message
    assert "has_object=[False]" in message
    assert "sampling_mode=fov" in message


def test_match_error_overlay_color_semantics() -> None:
    train_3am = _load_train_module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    overlay = train_3am._match_error_overlay(image, target, prediction)
    overlap = overlay[0, 0]
    gt_only = overlay[0, 1]
    pred_only = overlay[1, 0]
    background = overlay[1, 1]

    assert overlap[0] > 100 and overlap[1] > 100 and overlap[2] < 50
    assert gt_only[1] > gt_only[0] and gt_only[1] > gt_only[2]
    assert pred_only[0] > pred_only[1] and pred_only[0] > pred_only[2]
    assert np.all(background < 10)


def test_training_visualization_mask_helpers_report_area_and_bbox() -> None:
    train_3am = _load_train_module()
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:5] = True

    assert train_3am._mask_area(mask) == 6
    assert train_3am._bbox_text(mask) == "bbox=(2,1)-(4,2)"
    assert train_3am._bbox_text(np.zeros((4, 5), dtype=bool)) == "bbox=none"


def test_training_visualization_binary_panel_uses_mask_color() -> None:
    train_3am = _load_train_module()
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    panel = np.asarray(train_3am._binary_mask_panel(mask, color=(255, 60, 40)))

    assert panel[1, 1, 0] >= panel[1, 1, 1]


def test_training_script_dry_run_reports_inputs(tmp_path: Path, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path)

    assert train_3am.run_training(str(config_path), dry_run_only=True) == 0
    captured = capsys.readouterr()

    assert "scannetpp_manifest.json" in captured.out
    assert "feature_cache_root" in captured.out
    assert "must3r_importable" in captured.out
    assert "mast3r_importable" in captured.out
    assert "dust3r_importable" in captured.out
    assert "sam2_point_pseudo_masks" in captured.out
    assert "\"online_must3r\": false" in captured.out
    assert "\"sequence_lengths\"" in captured.out
    assert "\"cuda_memory_debug\": false" in captured.out


def test_training_script_logs_cuda_memory_debug_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    train_3am = _load_train_module()
    config_path = _write_tiny_training_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["cuda_memory_debug"] = True
    config["training"]["log_every"] = 0
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    class FakeCudaMemoryDebugger:
        enabled = True

        def __init__(self) -> None:
            self._fields: dict[str, float] = {}

        def start_step(self) -> None:
            self._fields = {}

        def snapshot(self, phase: str) -> None:
            mapping = {
                "after_batch_to_device": ("cuda_mem_batch_mb", 111.0),
                "after_must3r_ready": ("cuda_mem_must3r_mb", 222.0),
                "after_forward": ("cuda_mem_forward_mb", 333.0),
                "after_backward": ("cuda_mem_backward_mb", 444.0),
            }
            field, value = mapping[phase]
            self._fields[field] = value

        def fields(self) -> dict[str, float]:
            return dict(self._fields)

    monkeypatch.setattr(train_3am, "_build_cuda_memory_debugger", lambda device, enabled: FakeCudaMemoryDebugger())

    train_3am.run_training(
        str(config_path),
        iterations=1,
        device_name="cpu",
        checkpoint_every=0,
        save_model_every=0,
        sam2_adapter=FakeSam2Adapter(),
    )
    captured = capsys.readouterr()

    assert "cuda_mem_batch_mb=111.000" in captured.out
    assert "cuda_mem_must3r_mb=222.000" in captured.out
    assert "cuda_mem_forward_mb=333.000" in captured.out
    assert "cuda_mem_backward_mb=444.000" in captured.out
