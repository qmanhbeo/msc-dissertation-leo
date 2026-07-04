# Dissertation Reproducibility Guide

## What this repository contains

This repository contains the dissertation code, manuscript source, committed outputs, and the canonical reproducibility entrypoint for the submitted MSc project. The dissertation measures research-policy divergence in AI-for-SDG discourse using coverage and semantic-gap outputs. The README is operational only; the manuscript explains the research design, findings, and limitations.

## Quick start for markers

"Warm replay" means running the full analysis pipeline from pre-computed
embeddings, skipping the expensive fetch and embed stages. It is the canonical
reproducibility target — deterministic and snapshot-based.

```bash
git clone https://github.com/qmanhbeo/dissertation-bham.git
cd dissertation-bham
conda env create -f environment.yml
conda activate dissertation
# optional: if you have an NVIDIA GPU with CUDA 12.1:
# pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

python main.py --warm-replay --overwrite
```

This rebuilds main text outputs and the dissertation PDF from the frozen curated
data snapshot. Appendix outputs are already committed in the repo and do not need
regeneration. `--overwrite` is included because canonical outputs already exist
in the repo; without it `main.py` fails closed to prevent accidental replacement.
If the snapshot download fails, run `python main.py --fetch-data-snapshot curated`
then retry.

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

### Environment notes

- `environment.yml` is the canonical rebuild path. Pins 13 core Python packages;
  platform-specific libraries (CUDA, Linux libs) are handled by conda/pip per-OS.
- `requirements.txt` is a human-edited core reference (13 packages). Not needed
  for rebuild — `environment.yml` already covers the pip layer.
- Python version: `3.11`. Tested on Linux/WSL and Windows.
- PDF build requires LaTeX tools including `latexmk`, `pdflatex`, and `biber`
- CPU is sufficient for `--warm-replay`
- Network access is needed for initial environment setup and data snapshot download; once `2_data/` is hydrated, warm replay does not need network access

## Additional optional commands

| Command | What it does |
|---|---|
| `python main.py` | Read-only status check |
| `python main.py --warm-replay --overwrite` | Rebuild main text analysis + PDF from snapshot |
| `python main.py --main-text --overwrite` | Rebuild main text analysis only (no appendix, no PDF) |
| `python main.py --full-pipeline --overwrite` | Full live-source pipeline (not snapshot-reproducible) |
| `python main.py --appendix-all --overwrite` | Run all appendix stages (A1–A3, B1–B4, C) standalone |
| `python main.py --appendix-a1-source --overwrite` | Run A.1 Per-SDG Source Comparison |
| `python main.py --appendix-a2-family --overwrite` | Run A.2 Policy Source-Family Sensitivity |
| `python main.py --appendix-a3-sdg4 --overwrite` | Run A.3 SDG 4 Lexical Artefact Audit |
| `python main.py --appendix-b1-pca --overwrite` | Run B.1 Combined Research-Policy PCA Landscape |
| `python main.py --appendix-b2-centroid --overwrite` | Run B.2 Within-Corpus Centroid Structure |
| `python main.py --appendix-b3-interpret --overwrite` | Run B.3 Lexical Illustration of the Semantic Gap |
| `python main.py --appendix-b4-softmax --overwrite` | Run B.4 Softmax Multi-label SDG |
| `python main.py --appendix-c-register --overwrite` | Run C Register-Adjustment Robustness |
| `python main.py --refresh-policy-corpus --overwrite` | Re-fetch, re-embed, re-score policy corpus |
| `python main.py --sample-stability --overwrite` | Run sample-stability robustness stage |
| `python main.py --build-pdf --overwrite` | Rebuild PDF from existing outputs |
| `python main.py --clean-canon --overwrite` | Remove manuscript output artifacts |
| `python main.py --fetch-data-snapshot curated` | Hydrate curated snapshot into `2_data/` |
| `python main.py --fetch-data-snapshot full` | Hydrate full snapshot for audit |
| `python main.py --backup-data-snapshot curated` | Create and upload a curated snapshot archive |
| `python main.py --backup-data-snapshot full` | Create and upload a full snapshot archive |
| `python main.py --backup-data-snapshot both` | Create and upload both snapshot archives |

## Reproducibility boundaries

The canonical reproducibility target is warm replay from the frozen curated snapshot. Live-source refetching is not the marker-facing target. `--full-pipeline`, OpenAlex refetching, and policy-source refreshes are heavier and not expected to be byte-identical to the submitted snapshot state. The curated snapshot is the submitted marker-facing data state; the full snapshot is retained only for broader audit or reconstruction if needed.

## Repository layout

The repository uses a numbered directory convention. Each prefix indicates
the directory's role in the workflow:

- `0_literature/` — cited and consulted sources, organised by paper content
- `1_code/` — all pipeline and analysis code
- `2_data/` — frozen data snapshot (hydrated from archive, gitignored)
- `3_writing/` — manuscript source (`dissertation.tex`, `references.bib`, build script)
- `4_outputs/` — committed outputs (`dissertation.pdf`, `main/` figures and tables, `appendix/` analyses)
- `5_notes/` — working notes, assumptions, and workflow logs

Entrypoint and environment files at root:
- `main.py` — single reproducibility entrypoint
- `environment.yml` — full conda + pip environment specification
- `requirements.txt` — lightweight human-readable dependency reference
- `README.md` — this file
