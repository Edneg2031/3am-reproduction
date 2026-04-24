#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from three_am.utils.config import load_yaml


def run_if_present(name: str, command_env: str, root: Path) -> None:
    command = os.environ.get(command_env)
    root.mkdir(parents=True, exist_ok=True)
    if not command:
        marker = root / "DOWNLOAD_INSTRUCTIONS.txt"
        marker.write_text(
            f"Set {command_env} to the official/authenticated download command for {name}.\n"
            f"The command will run with this dataset root already created: {root}\n",
            encoding="utf-8",
        )
        print(f"SKIP {name}: set {command_env}; wrote {marker}")
        return
    print(f"RUN {name}: {command}")
    subprocess.run(command, shell=True, check=True, cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset download dispatcher for licensed 3AM datasets")
    parser.add_argument("--config", default="configs/full_reproduction.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    datasets = config["datasets"]
    run_if_present("ScanNet++", "SCANNETPP_DOWNLOAD_CMD", Path(datasets["scannetpp"]["root"]))
    run_if_present("ASE", "ASE_DOWNLOAD_CMD", Path(datasets["ase"]["root"]))
    run_if_present("MOSE", "MOSE_DOWNLOAD_CMD", Path(datasets["mose"]["root"]))
    run_if_present("Replica", "REPLICA_DOWNLOAD_CMD", Path(datasets["replica"]["root"]))


if __name__ == "__main__":
    main()
