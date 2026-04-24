#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from three_am.utils.config import load_yaml


def download(url: str, output: Path) -> None:
    if not url:
        print(f"SKIP {output}: no URL configured")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"EXISTS {output}")
        return
    print(f"DOWNLOAD {url} -> {output}")
    urllib.request.urlretrieve(url, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM2/MUSt3R checkpoints for 3AM reproduction")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    external = config["external"]
    download(external["sam2_checkpoint_url"], Path(external["sam2_checkpoint"]))
    must3r_url = os.environ.get("MUST3R_CHECKPOINT_URL", external.get("must3r_checkpoint_url", ""))
    download(must3r_url, Path(external["must3r_checkpoint"]))


if __name__ == "__main__":
    main()
