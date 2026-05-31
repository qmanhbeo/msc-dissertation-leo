# Repository Contract (Active Canon)

This document defines the hard-cut active conventions for code, data, outputs,
and reproducibility.

## 1) Code Layout Contract

Active scripts live only under:
- `code/0_fetch/`
- `code/1_preprocess/`
- `code/2_embed/`
- `code/3_main_analysis/`
- `code/4_visualization/`
- `code/shared_utils.py`

Legacy code must remain under `_legacy/` and must not be used by active runs.

## 2) Data Layout Contract

Active data layers are:
- `data/0_raw/`
- `data/1_preprocessed/`
- `data/2_embedded/`
- `data/3_scored/`

Run outputs live under:
- `outputs/<run_name>/` with `figures/` and `tables/`.

No active script should write new analysis outputs under `data/`.

## 3) Reproducibility Contract (Root Canon)

Canonical reproducibility files are root-level:
- `requirements.txt`
- `requirements.lock.txt`
- `conda-env-dissertation.yml`
- `conda-explicit-dissertation.txt`
- `runtime_snapshot_2026-05-26.md`

These files are the authoritative source for dependency and environment replay.

## 4) Runtime Baseline

Reference runtime snapshot (`runtime_snapshot_2026-05-26.md`) captures:
- OS: Ubuntu 24.04.3 LTS on WSL2
- Python: 3.11.15
- Torch: 2.5.1+cu121
- CUDA runtime: 12.1 (torch), driver-reported CUDA 12.3
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- NVIDIA driver: 546.30

For full-corpus embedding, use CUDA (`--device cuda`) in the dissertation env.
