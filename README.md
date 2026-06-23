# 3AM Unofficial Reproduction

This repository is an unofficial reproduction scaffold for **3AM: 3egment Anything with Geometric Consistency in Videos**.

The official 3AM project code is not released at the time this scaffold was created, so this repository provides a faithful reimplementation boundary around:

- SAM2 / SAM2.1 from `facebookresearch/sam2`
- MUSt3R from `naver/must3r`
- the training, sampling, evaluation, and notebook workflow described in the paper

## Quick Start

```bash
python -m pip install -e .[dev]
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/train_3am.py --config configs/full_reproduction.yaml --smoke --iterations 10
```

## Single ScanNet++ Scene With Continuous Sampling

For a prepared scene at `processed/scannetpp/<scene_id>` containing `images/`,
`masks/`, and `instances.json`, use `configs/scannetpp_continuous.yaml`. This
configuration disables FoV sampling, samples contiguous 8-frame clips, writes
loss/tracking curves under `outputs/training_metrics/scannetpp_continuous`, and
exports 8-frame tracking visualizations under
`outputs/visualizations/scannetpp_continuous`.

Reference objects are filtered before a clip is accepted. The default policy
rejects tiny noisy regions, objects covering more than 10% of the reference
image, and objects not visible in at least two frames. Structural categories
such as wall, floor, and ceiling are excluded when `instances.json` contains
category metadata (`category`, `label`, `class_name`, etc.). If the file only
contains object IDs, add confirmed structural IDs to
`datasets.scannetpp.instance_sampling.excluded_instance_ids`.

Build the manifest and precompute the frozen MUSt3R features before training:

```bash
PYTHONPATH=src python scripts/build_manifest.py \
  --dataset scannetpp \
  --root processed/scannetpp \
  --split train \
  --format normalized \
  --output processed/scannetpp_manifest.json

PYTHONPATH=src python scripts/precompute_must3r_features.py \
  --config configs/scannetpp_continuous.yaml \
  --manifest processed/scannetpp_manifest.json \
  --output-dir outputs/must3r_features \
  --memory-window 8

PYTHONPATH=src python scripts/train_3am.py \
  --config configs/scannetpp_continuous.yaml \
  --device cuda
```

## ScanNet++ Smoke Data

After generating ScanNet++ `obj_ids/<scene_id>/*.pth` with the official toolbox, prepare a small DSLR-based training subset with:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_training_data.py \
  --obj-id-root /path/to/scannetpp/obj_ids \
  --data-root data/raw/scannetpp \
  --scene-list data/processed/scannetpp_smoke_scenes.txt \
  --precompute-must3r
```

This writes per-frame instance-id label maps, `instances.json`, `data/processed/scannetpp_manifest.json`, and optional MUSt3R feature cache files for strict 3AM training.

## Main Files

- `configs/full_reproduction.yaml` — full reproduction settings from the paper.
- `src/three_am/models/feature_merger.py` — core SAM2 + MUSt3R feature fusion module.
- `scripts/` — install, download, preprocess, train, and evaluate entrypoints.
- `notebooks/3am_reproduction_tutorial.ipynb` — step-by-step usage tutorial.
- `docs/reproduction.md` — detailed reproduction notes.

## Important Limitations

This is not an official 3AM implementation. Full numeric reproduction requires connecting upstream SAM2 training internals and MUSt3R intermediate-feature extraction APIs after installing their official repositories.

Large datasets, checkpoints, generated outputs, and external repos are intentionally ignored by Git.
