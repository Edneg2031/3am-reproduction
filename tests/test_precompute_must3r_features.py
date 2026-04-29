from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from three_am.data.io import read_manifest, write_manifest
from three_am.data.schema import FrameRecord, SceneRecord


def _load_precompute_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "precompute_must3r_features.py"
    spec = importlib.util.spec_from_file_location("precompute_must3r_features_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path) -> Path:
    frames = tuple(
        FrameRecord(
            frame_id=f"{index:03d}",
            image_path=tmp_path / "images" / f"{index:03d}.png",
            mask_path=tmp_path / "masks" / f"{index:03d}.png",
        )
        for index in range(2)
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, [SceneRecord("scannetpp", "scene_a", "train", frames)])
    return path


class FakeExtractor:
    def __init__(self, module) -> None:
        self.module = module

    def extract_scene(self, scene: SceneRecord):
        return self.module.SceneFeatures(
            levels=(
                torch.ones(len(scene.frames), 2, 2, 3),
                torch.ones(len(scene.frames), 3, 1, 2),
                torch.ones(len(scene.frames), 4, 1, 1),
            ),
            metadata={"extractor": "fake", "feature_layers": [0, 1, 2]},
        )


def test_precompute_writes_features_metadata_and_manifest(tmp_path: Path) -> None:
    module = _load_precompute_module()
    manifest = _manifest(tmp_path)
    output_dir = tmp_path / "cache"
    feature_manifest = tmp_path / "manifest_with_features.json"
    options = module.PrecomputeOptions(
        config={},
        manifest=manifest,
        output_dir=output_dir,
        device="cpu",
        weights=None,
        must3r_repo=None,
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=(0, 1, 2),
        write_manifest=feature_manifest,
        limit_scenes=None,
        dry_run=False,
    )

    summary = module.run_precompute(options, extractor=FakeExtractor(module))

    assert summary["processed_scenes"] == 1
    scene_dir = output_dir / "scannetpp" / "scene_a"
    assert (scene_dir / "000_level0.pt").exists()
    assert (scene_dir / "001_level2.pt").exists()
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_channels"] == [2, 3, 4]
    updated = read_manifest(feature_manifest)
    feature_paths = updated[0].frames[0].must3r_feature_paths
    assert len(feature_paths) == 3
    assert all(path.is_absolute() for path in feature_paths)


def test_compacts_frame_views_before_saving(tmp_path: Path) -> None:
    module = _load_precompute_module()
    backing = torch.zeros(8, 2, 3, 4, dtype=torch.float16)
    frame_view = backing[3]

    compact = module._compact_tensor_for_save(frame_view)

    assert tuple(compact.shape) == (2, 3, 4)
    assert compact.storage_offset() == 0
    assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()


def test_precompute_dry_run_does_not_create_output(tmp_path: Path, capsys) -> None:
    module = _load_precompute_module()
    manifest = _manifest(tmp_path)
    output_dir = tmp_path / "cache"
    options = module.PrecomputeOptions(
        config={},
        manifest=manifest,
        output_dir=output_dir,
        device="cpu",
        weights=None,
        must3r_repo=None,
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=(0, 1, 2),
        write_manifest=None,
        limit_scenes=None,
        dry_run=True,
    )

    module.run_precompute(options, extractor=FakeExtractor(module))
    captured = capsys.readouterr()

    assert '"frames": 2' in captured.out
    assert not output_dir.exists()


def test_official_extractor_passes_true_shape_as_list_to_decoder(tmp_path: Path) -> None:
    module = _load_precompute_module()
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=2,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=(0, 1),
    )

    class FakeDecoder:
        def forward_list(self, x, pos, true_shape, return_feats):
            assert isinstance(x, list)
            assert isinstance(pos, list)
            copied = true_shape.copy()
            assert isinstance(copied, list)
            assert copied[0].shape == (1, 2, 2)
            assert return_feats is True
            feats = [[torch.ones(1, 2, 6, 2), torch.ones(1, 2, 6, 3)]]
            return object(), object(), feats

    encoder_tokens = torch.ones(2, 6, 4)
    pos = torch.zeros(2, 6, 2)
    true_shape = torch.tensor([[2, 3], [2, 3]])

    levels = extractor._decode_feature_levels(FakeDecoder(), encoder_tokens, pos, true_shape)

    assert [tuple(level.shape) for level in levels] == [(2, 6, 2), (2, 6, 3)]


def test_official_extractor_chunks_decoder_by_decode_batch_size(tmp_path: Path) -> None:
    module = _load_precompute_module()
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=2,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=(0, 1),
    )
    calls: list[int] = []

    class FakeDecoder:
        def forward_list(self, x, pos, true_shape, return_feats):
            frames = x[0].shape[1]
            calls.append(frames)
            assert frames <= 2
            assert pos[0].shape[1] == frames
            assert true_shape[0].shape == (1, frames, 2)
            assert return_feats is True
            feats = [[torch.ones(1, frames, 6, 2), torch.ones(1, frames, 6, 3)]]
            return object(), object(), feats

    encoder_tokens = torch.ones(5, 6, 4)
    pos = torch.zeros(5, 6, 2)
    true_shape = torch.tensor([[2, 3]] * 5)

    levels = extractor._decode_feature_levels(FakeDecoder(), encoder_tokens, pos, true_shape)

    assert calls == [2, 2, 1]
    assert [tuple(level.shape) for level in levels] == [(5, 6, 2), (5, 6, 3)]


def test_feature_layer_parser_maps_paper_specs_to_return_feats_indices(tmp_path: Path) -> None:
    module = _load_precompute_module()
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=module.parse_feature_layers("encoder,4,7,11"),
    )

    specs, indices = extractor._normalize_feature_layers(13)

    assert specs == ("encoder", 4, 7, 11)
    assert indices == (0, 5, 8, 12)


def test_official_extractor_exports_paper_token_grid_features(tmp_path: Path) -> None:
    module = _load_precompute_module()
    frames = tuple(
        FrameRecord(frame_id=f"{index:03d}", image_path=tmp_path / f"{index:03d}.png") for index in range(2)
    )
    scene = SceneRecord("scannetpp", "scene_a", "train", frames)
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=module.parse_feature_layers("encoder,4,7,11"),
    )
    grid_h, grid_w = 32, 24
    num_tokens = grid_h * grid_w

    class FakeEncoder:
        patch_size = 16

        def __call__(self, images, true_shape):
            frames_in_chunk = images.shape[0]
            tokens = torch.ones(frames_in_chunk, num_tokens, 1024)
            y, x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
            pos = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1).to(dtype=torch.float32)
            return tokens, pos.unsqueeze(0).expand(frames_in_chunk, -1, -1).clone()

    class FakeDecoder:
        def forward_list(self, x, pos, true_shape, return_feats):
            frames_in_chunk = x[0].shape[1]
            feats = [x[0]]
            feats.extend(torch.ones(1, frames_in_chunk, num_tokens, 768) * layer for layer in range(12))
            return object(), object(), [feats]

    def fake_load_images(chunk_frames, patch_size):
        return [torch.zeros(3, 512, 384) for _ in chunk_frames], [
            torch.tensor([512, 384], dtype=torch.int64) for _ in chunk_frames
        ]

    extractor.model = (FakeEncoder(), FakeDecoder())
    extractor._load_images = fake_load_images  # type: ignore[method-assign]

    features = extractor.extract_scene(scene)
    output_dir = tmp_path / "cache"
    module.save_scene_features(scene, features, output_dir=output_dir, manifest=tmp_path / "manifest.json")

    scene_dir = output_dir / "scannetpp" / "scene_a"
    saved = [torch.load(scene_dir / f"000_level{level}.pt", map_location="cpu", weights_only=True) for level in range(4)]
    assert [tuple(tensor.shape) for tensor in saved] == [
        (1024, 32, 24),
        (768, 32, 24),
        (768, 32, 24),
        (768, 32, 24),
    ]
    assert all(tensor.dtype == torch.float16 for tensor in saved)
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_specs"] == ["encoder", "decoder_4", "decoder_7", "decoder_11"]
    assert metadata["feature_channels"] == [1024, 768, 768, 768]
    assert metadata["feature_grid"] == [[32, 24], [32, 24], [32, 24], [32, 24]]


def test_official_extractor_rejects_non_paper_channels(tmp_path: Path) -> None:
    module = _load_precompute_module()
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=("encoder",),
    )
    y, x = torch.meshgrid(torch.arange(32), torch.arange(24), indexing="ij")
    pos = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1).unsqueeze(0).to(dtype=torch.float32)
    true_shape = torch.tensor([[512, 384]], dtype=torch.int64)

    import pytest

    with pytest.raises(module.Must3rPrecomputeError, match="encoder=1024"):
        extractor._tokens_to_chw(torch.ones(1, 32 * 24, 768), pos, true_shape, 16, "encoder")


def test_official_extractor_rejects_dense_grid_features(tmp_path: Path) -> None:
    module = _load_precompute_module()
    extractor = module.OfficialMust3rExtractor(
        weights=tmp_path / "unused.pth",
        must3r_repo=None,
        device="cpu",
        image_size=512,
        amp=False,
        max_bs=1,
        decode_batch_size=1,
        memory_window=None,
        full_scene_memory=False,
        cache_dtype="fp16",
        feature_layers=(4,),
    )
    dense_tokens = 512 * 512
    pos = torch.zeros(1, dense_tokens, 2)
    true_shape = torch.tensor([[512, 384]], dtype=torch.int64)

    import pytest

    with pytest.raises(module.Must3rPrecomputeError, match="Refusing to save a dense full-resolution"):
        extractor._tokens_to_chw(torch.ones(1, dense_tokens, 768), pos, true_shape, 16, 4)


def test_compute_fov_overlap_skips_when_geometry_missing(tmp_path: Path) -> None:
    module = _load_precompute_module()
    manifest = _manifest(tmp_path)
    scene = read_manifest(manifest)[0]

    assert module.compute_fov_overlap(scene) is None


def test_compute_fov_overlap_writes_identity_for_same_camera(tmp_path: Path) -> None:
    module = _load_precompute_module()
    frames: list[FrameRecord] = []
    for index in range(2):
        frame_id = f"{index:03d}"
        depth_path = tmp_path / "depth" / f"{frame_id}.png"
        pose_path = tmp_path / "poses" / f"{frame_id}.txt"
        intrinsics_path = tmp_path / "intrinsics" / f"{frame_id}.txt"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(depth_path)
        np.savetxt(pose_path, np.eye(4))
        np.savetxt(intrinsics_path, np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]))
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=tmp_path / "images" / f"{frame_id}.png",
                depth_path=depth_path,
                pose_path=pose_path,
                intrinsics_path=intrinsics_path,
            )
        )
    scene = SceneRecord("scannetpp", "scene_a", "train", tuple(frames))

    overlap = module.compute_fov_overlap(scene)

    assert overlap is not None
    np.testing.assert_allclose(overlap, np.ones((2, 2), dtype=np.float32))
