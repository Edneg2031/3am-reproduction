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


def test_must3r_adapter_maps_paper_layers_to_decoder_indices() -> None:
    adapter = Must3rFeatureAdapter(
        ExternalBackboneConfig(must3r_feature_layers=_parse_must3r_feature_layers("encoder,4,7,11"))
    )

    specs, indices = adapter._normalize_feature_layers(13)

    assert specs == ("encoder", 4, 7, 11)
    assert indices == (0, 5, 8, 12)
