from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from three_am.models.adapters import (
    ExternalBackboneConfig,
    ExternalDependencyError,
    Must3rFeatureAdapter,
    Sam2TrainingAdapter,
    _normalize_must3r_amp,
    _parse_must3r_feature_layers,
    _sam2_config_name,
)
from three_am.training.dataset import Prompt, TrainingBatch


def test_sam2_config_name_keeps_hydra_config_name() -> None:
    assert _sam2_config_name("configs/sam2.1/sam2.1_hiera_l.yaml") == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_sam2_config_name_converts_absolute_config_path_to_hydra_name() -> None:
    path = Path("/tmp/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    assert _sam2_config_name(path) == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_must3r_feature_layer_and_amp_parsers() -> None:
    assert _parse_must3r_feature_layers("encoder,4,7,11") == ("encoder", 4, 7, 11)
    assert _parse_must3r_feature_layers(["encoder", 4, "7"]) == ("encoder", 4, 7)
    assert _normalize_must3r_amp("false") is False
    assert _normalize_must3r_amp("bf16") == "bf16"


class FakeSam2BackboneAware(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_backbone = None

    def forward_image(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        return {
            "backbone_fpn": [
                torch.zeros(images.shape[0], 32, 16, 16),
                torch.ones(images.shape[0], 256, 8, 8),
            ],
            "vision_pos_enc": [
                torch.zeros(images.shape[0], 32, 16, 16),
                torch.zeros(images.shape[0], 256, 8, 8),
            ],
        }

    def forward_train_sequence_from_backbone(self, batch, backbone_out):
        self.seen_backbone = backbone_out
        logits = backbone_out["backbone_fpn"][-1].mean(dim=1)
        pooled = logits.mean(dim=(1, 2))
        return {
            "mask_logits": logits,
            "iou_scores": torch.sigmoid(pooled),
            "occlusion_logits": torch.stack([-pooled, pooled], dim=1),
        }


def test_sam2_adapter_injects_merged_features_into_backbone_fpn() -> None:
    adapter = Sam2TrainingAdapter(ExternalBackboneConfig())
    model = FakeSam2BackboneAware()
    adapter.model = model
    images = torch.randn(2, 3, 64, 64)

    sam_features = adapter.encode_sam_features(images)
    merged = torch.full_like(sam_features, 7.0)
    outputs = adapter.forward_train_sequence(batch=object(), merged_features=merged)

    assert model.seen_backbone is not None
    assert torch.equal(model.seen_backbone["backbone_fpn"][-1], merged)
    assert torch.equal(model.seen_backbone["backbone_fpn"][0], torch.zeros(2, 32, 16, 16))
    assert set(outputs) == {"mask_logits", "iou_scores", "occlusion_logits"}


def test_sam2_adapter_rejects_merged_feature_shape_mismatch() -> None:
    adapter = Sam2TrainingAdapter(ExternalBackboneConfig())
    adapter.model = FakeSam2BackboneAware()
    sam_features = adapter.encode_sam_features(torch.randn(2, 3, 64, 64))
    wrong = torch.randn(sam_features.shape[0], sam_features.shape[1], sam_features.shape[2] + 1, sam_features.shape[3])

    with pytest.raises(ExternalDependencyError, match="Merged 3AM feature shape"):
        adapter.forward_train_sequence(batch=object(), merged_features=wrong)


def test_sam2_adapter_rejects_non_window_aligned_image_size() -> None:
    adapter = Sam2TrainingAdapter(ExternalBackboneConfig())
    adapter.model = FakeSam2BackboneAware()

    with pytest.raises(ExternalDependencyError, match="divisible by 32"):
        adapter.encode_sam_features(torch.randn(1, 3, 844, 1024))


class FakeOfficialSam2Modules(nn.Module):
    num_feature_levels = 2
    hidden_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[tuple[int, bool]] = []

    def forward_image(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        frames = images.shape[0]
        return {
            "backbone_fpn": [
                torch.zeros(frames, 4, 16, 16, device=images.device),
                torch.ones(frames, 4, 8, 8, device=images.device),
            ],
            "vision_pos_enc": [
                torch.zeros(frames, 4, 16, 16, device=images.device),
                torch.zeros(frames, 4, 8, 8, device=images.device),
            ],
        }

    def _prepare_backbone_features(self, backbone_out):
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        pos_maps = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(feature.shape[-2], feature.shape[-1]) for feature in pos_maps]
        vision_feats = [feature.flatten(2).permute(2, 0, 1) for feature in feature_maps]
        vision_pos = [pos.flatten(2).permute(2, 0, 1) for pos in pos_maps]
        return backbone_out, vision_feats, vision_pos, feat_sizes

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        self.steps.append((frame_idx, is_init_cond_frame))
        value = current_vision_feats[-1].mean().reshape(1, 1, 1, 1)
        high_res = value.expand(1, 1, 64, 64).clone()
        low_res = value.expand(1, 1, 16, 16).clone()
        ious = torch.full((1, 1), 0.75, device=high_res.device)
        obj_ptr = torch.zeros(1, self.hidden_dim, device=high_res.device)
        object_score_logits = torch.ones(1, 1, device=high_res.device)
        sam_outputs = (low_res, high_res, ious, low_res, high_res, obj_ptr, object_score_logits)
        return {"point_inputs": point_inputs, "mask_inputs": mask_inputs}, sam_outputs, None, current_vision_feats[-1]

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        current_out["maskmem_features"] = high_res_masks
        current_out["maskmem_pos_enc"] = [torch.zeros_like(high_res_masks)]


def test_sam2_adapter_falls_back_to_official_track_step_modules() -> None:
    adapter = Sam2TrainingAdapter(ExternalBackboneConfig())
    model = FakeOfficialSam2Modules()
    adapter.model = model
    images = torch.randn(3, 3, 64, 64)
    target_masks = torch.zeros(3, 64, 64)
    batch = TrainingBatch(
        images=images,
        target_masks=target_masks,
        prompt=Prompt(type="mask", frame_index=1, mask=target_masks[1]),
        must3r_features=None,
        dataset="scannetpp",
        scene_id="scene_a",
        frame_ids=("000", "001", "002"),
        image_paths=(),
        has_object=torch.tensor([False, False, False]),
    )

    sam_features = adapter.encode_sam_features(images)
    outputs = adapter.forward_train_sequence(batch, sam_features)

    assert outputs["mask_logits"].shape == (3, 64, 64)
    assert outputs["iou_scores"].shape == (3,)
    assert outputs["occlusion_logits"].shape == (3, 2)
    assert model.steps[0] == (1, True)


class FakePerFrameSam2(nn.Module):
    def forward_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones(images.shape[0], 4, 8, 8)

    def _forward_sam_heads(self, **kwargs):
        low_res = torch.zeros(1, 1, 16, 16)
        high_res = torch.zeros(1, 1, 64, 64)
        ious = torch.zeros(1, 1)
        obj_ptr = torch.zeros(1, 4)
        object_score_logits = torch.zeros(1, 1)
        return low_res, high_res, ious, low_res, high_res, obj_ptr, object_score_logits


def test_strict_sam2_adapter_rejects_per_frame_head_fallback() -> None:
    adapter = Sam2TrainingAdapter(ExternalBackboneConfig(strict_paper=True))
    adapter.model = FakePerFrameSam2()
    images = torch.randn(1, 3, 64, 64)
    target_masks = torch.zeros(1, 64, 64)
    batch = TrainingBatch(
        images=images,
        target_masks=target_masks,
        prompt=Prompt(type="mask", frame_index=0, mask=target_masks[0]),
        must3r_features=None,
        dataset="scannetpp",
        scene_id="scene_a",
        frame_ids=("000",),
        image_paths=(),
        has_object=torch.tensor([False]),
    )
    sam_features = adapter.encode_sam_features(images)

    with pytest.raises(ExternalDependencyError, match="Strict paper training requires official SAM2 tracking"):
        adapter.forward_train_sequence(batch, sam_features)


def test_must3r_adapter_maps_paper_layers_to_decoder_indices() -> None:
    adapter = Must3rFeatureAdapter(
        ExternalBackboneConfig(must3r_feature_layers=_parse_must3r_feature_layers("encoder,4,7,11"))
    )

    specs, indices = adapter._normalize_feature_layers(13)

    assert specs == ("encoder", 4, 7, 11)
    assert indices == (0, 5, 8, 12)


def test_must3r_adapter_converts_official_dense_pointmaps_to_token_grid() -> None:
    adapter = Must3rFeatureAdapter(ExternalBackboneConfig())
    true_shape = torch.tensor([[32, 48], [32, 48]], dtype=torch.int64)
    y, x = torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    pos = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1).unsqueeze(0).repeat(2, 1, 1).float()
    dense = torch.ones(1, 2, 32, 48, 4)

    point_map = adapter._pointmaps_to_chw([dense], pos, true_shape)

    assert point_map is not None
    assert tuple(point_map.shape) == (2, 3, 2, 3)
    assert torch.allclose(point_map, torch.ones_like(point_map))
