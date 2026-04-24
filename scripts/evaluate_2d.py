#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from three_am.evaluation.track2d import evaluate_saved_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved 2D tracking masks with 3AM metrics")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction_paths = sorted(Path(args.pred_dir).glob("*.png"))
    target_paths = sorted(Path(args.gt_dir).glob("*.png"))
    metrics = evaluate_saved_masks(prediction_paths, target_paths, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
