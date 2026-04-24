from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    checkpoints: Path
    outputs: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ProjectPaths":
        root = Path(config.get("project_root", ".")).expanduser().resolve()
        paths = config.get("paths", {})
        return cls(
            root=root,
            data_raw=(root / paths.get("data_raw", "data/raw")).resolve(),
            data_processed=(root / paths.get("data_processed", "data/processed")).resolve(),
            checkpoints=(root / paths.get("checkpoints", "outputs/checkpoints")).resolve(),
            outputs=(root / paths.get("outputs", "outputs")).resolve(),
        )
