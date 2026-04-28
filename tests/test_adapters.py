from __future__ import annotations

from pathlib import Path

from three_am.models.adapters import _sam2_config_name


def test_sam2_config_name_keeps_hydra_config_name() -> None:
    assert _sam2_config_name("configs/sam2.1/sam2.1_hiera_l.yaml") == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_sam2_config_name_converts_absolute_config_path_to_hydra_name() -> None:
    path = Path("/tmp/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    assert _sam2_config_name(path) == "configs/sam2.1/sam2.1_hiera_l.yaml"
