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

## Main Files

- `configs/full_reproduction.yaml` — full reproduction settings from the paper.
- `src/three_am/models/feature_merger.py` — core SAM2 + MUSt3R feature fusion module.
- `scripts/` — install, download, preprocess, train, and evaluate entrypoints.
- `notebooks/3am_reproduction_tutorial.ipynb` — step-by-step usage tutorial.
- `docs/reproduction.md` — detailed reproduction notes.

## Important Limitations

This is not an official 3AM implementation. Full numeric reproduction requires connecting upstream SAM2 training internals and MUSt3R intermediate-feature extraction APIs after installing their official repositories.

Large datasets, checkpoints, generated outputs, and external repos are intentionally ignored by Git.
