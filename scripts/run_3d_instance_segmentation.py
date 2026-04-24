#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run class-agnostic 3D instance segmentation with propagated 3AM masks")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.parse_args()
    raise NotImplementedError(
        "This stage needs a trained 3AM checkpoint and SAM2 automatic mask generator outputs. The math helpers live in three_am.evaluation.instance3d."
    )


if __name__ == "__main__":
    main()
