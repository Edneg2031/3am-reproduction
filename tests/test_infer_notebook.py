from __future__ import annotations

from pathlib import Path

import nbformat


def test_infer_notebook_is_valid_nbformat() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "infer.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)

    assert notebook["nbformat"] == 4
    assert any(
        "POINTS" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert any(
        "masks.mp4" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert any(
        "write_overlay_video" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert any(
        "SAMPLE_FPS" in "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
