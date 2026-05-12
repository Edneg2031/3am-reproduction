# 3AM Unofficial Full Reproduction

This repository contains a faithful, non-official reproduction scaffold for **3AM: 3egment Anything with Geometric Consistency in Videos**. The official project page currently lists code as coming soon, so this project defines explicit adapter boundaries around the official SAM2 and MUSt3R repositories.

## What is implemented

- Unified project structure for data, checkpoints, scripts, notebooks, and source code.
- `FeatureMerger`, the core 3AM fusion module that combines SAM2 appearance features with multi-level MUSt3R geometry-aware features.
- Dataset manifest schema for `{frames, masks, depth, poses, intrinsics, instances, split}`.
- FoV-aware and continuous sampling utilities matching the paper-level training policy.
- Metrics for 2D tracking: IoU, Tracking Recall, and Accuracy.
- Script entry points for external install, weight download, licensed dataset dispatch, manifest building, MUSt3R precompute, smoke training, 2D evaluation, and 3D instance segmentation.
- Notebook tutorial at `notebooks/3am_reproduction_tutorial.ipynb`.

## Full reproduction settings

The default config is `configs/full_reproduction.yaml`:

- Training datasets: ScanNet++ 855 scenes, ASE 2612 scenes, MOSE 1453 videos.
- Evaluation datasets: ScanNet++ and Replica.
- Sampling: ScanNet++/ASE use 0.8 FoV-aware + 0.2 continuous; MOSE uses continuous only.
- FoV threshold: `0.25`.
- Optimizer: AdamW.
- Iterations: `1_000_000`.
- Batch size: `1`.
- Memory frames: `8`.
- Learning rates: Memory Attention `5e-6`, Memory Encoder `5e-6`, Mask Decoder `5e-6`, Feature Merger `1e-5`.

## Quick smoke test

```bash
python -m pip install -e .[dev]
python scripts/train_3am.py --smoke --iterations 10
pytest -q
```

## External dependencies

```bash
bash scripts/install_external.sh
python scripts/download_weights.py --config configs/full_reproduction.yaml
```

`MUST3R_CHECKPOINT_URL` must be supplied if the MUSt3R checkpoint URL is not embedded in the selected upstream release.

## Licensed datasets

Dataset download commands are intentionally read from environment variables so credentials are never committed:

```bash
export SCANNETPP_DOWNLOAD_CMD='... official ScanNet++ command ...'
export ASE_DOWNLOAD_CMD='... official Project Aria ASE command ...'
export MOSE_DOWNLOAD_CMD='... official MOSE command ...'
export REPLICA_DOWNLOAD_CMD='... official Replica command ...'
python scripts/download_datasets.py
```

If a command is missing, the script writes a `DOWNLOAD_INSTRUCTIONS.txt` marker into the corresponding dataset root.

## Normalized dataset layout

Each scene/video should be normalized to one of these folder forms before manifest creation:

```text
scene_id/
  frames/ or images/
    000001.jpg
  masks/
    000001.png
  depth/
    000001.png
  poses/
    000001.txt
  intrinsics/
    000001.txt
  instances.json
```

MOSE does not need `depth`, `poses`, or `intrinsics`.

For ScanNet++, `masks/*.png` must be per-frame **instance-id label maps**, not anonymization masks, valid-pixel masks, or all-foreground binary masks. The 3AM paper trains/evaluates with 2D masks obtained by projecting 3D instance labels into each RGB frame, and uses those masks as mask prompts/supervision for ScanNet++ and ASE. ScanNet++ does not ship SAM2 prompt masks directly.

To prepare ScanNet++:

1. Use the official ScanNet++ toolbox to rasterize semantics and run `semantic.prep.semantics_2d` with object-id GT saving enabled, producing files like `obj_ids/<scene_id>/*.pth`.
2. Convert those object-id maps into the normalized layout:

```bash
PYTHONPATH=src python scripts/preprocess_scannetpp_instance_masks.py \
  --obj-id-root /path/to/scannetpp/obj_ids \
  --data-root data/raw/scannetpp \
  --output-root data/processed/scannetpp
```

The script writes 16-bit PNG label maps under `data/processed/scannetpp/<scene_id>/masks`, symlinks RGB images by default, and creates `instances.json`. It fails fast when an input mask has one positive id covering almost the whole frame, which usually means a valid-region/full-frame mask was supplied instead of projected instance labels.

Build manifests with:

```bash
python scripts/build_manifest.py --dataset scannetpp --root data/processed/scannetpp --split train --output data/processed/scannetpp_manifest.json
python scripts/build_manifest.py --dataset ase --root data/processed/ase --split train --output data/processed/ase_manifest.json
python scripts/build_manifest.py --dataset mose --root data/processed/mose --split train --output data/processed/mose_manifest.json
python scripts/build_manifest.py --dataset replica --root data/processed/replica --split eval --output data/processed/replica_manifest.json
```

`build_manifest.py --dataset scannetpp` now requires `instances.json` and rejects full-frame singleton masks by default. Use `--allow-missing-instances` only for legacy/debug manifests, not for paper-aligned training.

The default full reproduction config sets `datasets.scannetpp.require_instance_label_maps: true` and `training.sam2_point_pseudo_masks.mode: off`. The SAM2 random point pseudo-mask fallback is useful only as an emergency diagnostic when inspecting broken data; it is not the ScanNet++ training target used by the paper.

## Important limitations

This is a faithful reimplementation scaffold, not an official 3AM release. Full numeric reproduction requires connecting upstream SAM2 training internals and MUSt3R intermediate-feature APIs after those dependencies are installed in the target CUDA environment.
