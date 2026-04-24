#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from three_am.data.io import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute MUSt3R features and FoV caches for geometry datasets")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scenes = read_manifest(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"scenes": len(scenes), "frames": sum(len(scene.frames) for scene in scenes), "dry_run": args.dry_run}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    raise NotImplementedError(
        "Install and wire naver/must3r feature extraction, then save per-frame feature tensors and FoV overlap matrices here."
    )


if __name__ == "__main__":
    main()
