# Dissertation Reproducibility Guide

## What this repository contains

This repository contains the dissertation code, manuscript source, committed outputs, and the canonical reproducibility entrypoint for the submitted MSc project. The dissertation measures research-policy divergence in AI-for-SDG discourse using coverage and semantic-gap outputs. The README is operational only; the manuscript explains the research design, findings, and limitations.

## Quick start for markers

```bash
git clone https://github.com/qmanhbeo/dissertation-bham.git
cd dissertation-bham
conda env create -f conda-env-dissertation.yml
conda activate dissertation
python main.py --warm-replay --overwrite
```

In the submitted repository, `4_outputs/` is committed for direct inspection. The command above rebuilds the canonical outputs and dissertation PDF from the frozen curated data snapshot. `--overwrite` is included because canonical outputs already exist; without it, `main.py` fails closed to prevent accidental replacement.

If `2_data/` is missing, `python main.py --warm-replay --overwrite` should fetch the curated snapshot automatically. If automatic fetch fails, hydrate `2_data/` explicitly and then rerun warm replay:

```bash
python main.py --fetch-data-snapshot curated
python main.py --warm-replay --overwrite
```

## What the replay produces

Warm replay rebuilds:
- `4_outputs/dissertation.pdf`
- canonical machine-readable outputs under `4_outputs/main/data/`
- manuscript tables and figures under `4_outputs/main/tables/` and `4_outputs/main/figures/`

## Tracked vs not tracked

Tracked in Git:
- `main.py`
- `1_code/`
- `3_writing/`
- environment files
- `README.md`
- committed `4_outputs/`

Not tracked in Git:
- `2_data/`

`2_data/` is hydrated from the frozen curated snapshot. `4_outputs/` is committed for marker inspection but can be regenerated from the snapshot and source code.

## Environment assumptions

- Canonical setup path: `conda-env-dissertation.yml`
- Python version in the canonical environment: `3.11.15`
- `requirements.txt` is provided as a pip dependency reference or fallback, but the conda environment remains the preferred setup path
- `requirements.lock.txt` and `conda-explicit-dissertation.txt` are retained as audit/lock artifacts
- Linux or WSL is the assumed/tested environment; native Windows is not the main target
- PDF build requires LaTeX tools including `latexmk`, `pdflatex`, and `biber`
- CPU is sufficient for `--warm-replay`
- Network access is needed for initial environment setup and data snapshot download; once `2_data/` is hydrated, warm replay does not need network access

## Additional optional commands

```bash
python main.py
```
Read-only status check for warm-replay readiness and current artifact presence.

```bash
python main.py --build-pdf --overwrite
```
Rebuild only the dissertation PDF from existing tables and figures.

```bash
python main.py --register-adjustment --overwrite
```
Run the Appendix B register-adjustment robustness suite from existing embedded/scored data.

```bash
python main.py --sample-stability --overwrite
```
Run only the sample-stability robustness stage from existing canonical analysis outputs.

```bash
python main.py --fetch-data-snapshot curated
```
Explicitly hydrate the default marker-facing curated snapshot into `2_data/`.

```bash
python main.py --fetch-data-snapshot full
```
Hydrate the full data snapshot for audit or broader reconstruction; this is optional and not required for normal marking.

The underlying snapshot utilities remain under `1_code/data_backup_and_fetch/` for debugging or audit use.

## Reproducibility boundaries

The canonical reproducibility target is warm replay from the frozen curated snapshot. Live-source refetching is not the marker-facing target. `--full-pipeline`, OpenAlex refetching, and policy-source refreshes are heavier and not expected to be byte-identical to the submitted snapshot state. The curated snapshot is the submitted marker-facing data state; the full snapshot is retained only for broader audit or reconstruction if needed.

## Repository layout

- `main.py`: canonical entrypoint
- `1_code/`: pipeline and analysis code
- `3_writing/`: manuscript source and PDF build script
- `4_outputs/`: committed dissertation outputs
- `2_data/`: hydrated snapshot data after fetch
- `5_notes/`: working notes retained in the repository
