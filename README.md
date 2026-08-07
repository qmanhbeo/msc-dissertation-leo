# Dissertation Reproducibility Guide

## What this repository contains

This repository contains the dissertation code, manuscript source, committed outputs, and the canonical reproducibility entrypoint for the submitted MSc project. The dissertation measures research-policy divergence in AI-for-SDG discourse using coverage and semantic-gap outputs. The README is operational only; the manuscript explains the research design, findings, and limitations.

## Prerequisites

| Requirement | Details |
|---|---|
| **Disk space** | Embedded snapshot: ~14 GB archive. Raw snapshot: ~3.7 GB archive |
| **Platform** | Full pipeline tested end-to-end on WSL (Ubuntu). On Windows (native): `--warm-replay-without-appendix` / `--warm-replay-with-appendix` and `--appendix-all` work. `--cold-replay` was not tested on bare Windows (OpenAlex re-fetch would cost too much). `--build-pdf` requires bash (WSL/Linux only) |
| **Conda** | Required — environment is defined in `environment.yml` |
| **RAM / VRAM** | 10 GB RAM + 4 GB VRAM is sufficient for the full pipeline (warm replay and cold replay). CPU-only warm replay works on the same RAM budget |
| **Network** | Required for `conda env create` (packages) and `--fetch-data-snapshot` (archive download). For `--cold-replay`, a **live OpenAlex re-fetch** needs the OpenAlex API (see Credentials); with the frozen **raw snapshot** hydrated, cold replay is offline/deterministic and only needs the HuggingFace model download (one-time, cached). Warm replay is fully offline once Conda exists and the snapshot is hydrated |
| **LaTeX** | `latexmk` + `pdflatex` + `biber` for `--build-pdf` |
| **rclone** | Required for `--backup-data-snapshot` only (maintainer tool). Override remote via `--remote-root` or `DISSERTATION_SNAPSHOT_REMOTE_ROOT`. Not needed for warm/cold replay |
| **OpenAlex key(s)** | `.env` with `OPENALEX_MAILTO` + `OPENALEX_API_KEY` — only for a **live** `--cold-replay` OpenAlex re-fetch. The full OpenAlex re-fetch cycles through up to 4 parallel API keys and takes approximately 1 week. A `--cold-replay` from the frozen **raw snapshot** needs **no** key (deterministic/offline) |
| **Embedding runtimes** (one-time, cold replay only) | Segmentation is **canonical and shared** — one ~17h pass at 384 tokens produces segments reused by every encoder. MPNet (`all-mpnet-base-v2`, primary) embeds the full corpus (~17h on 4 GB VRAM). MiniLM and SciBERT embed only the shared 100k-paper subset (~minutes each). SciBERT also loads as a raw BERT wrapped with mean pooling |
| **Git** | Required for cloning and pulling updates |

All other dependencies (Python packages, LaTeX packages) are handled by Conda.

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

# One-time: warm the HuggingFace encoder models (MPNet + MiniLM + SciBERT) into the local cache.
python main.py --fetch-encoder-models

python main.py --warm-replay-without-appendix --overwrite
```

This rebuilds main text outputs from the frozen embedded data snapshot.
Appendix outputs are already committed in the repo and do not need
regeneration. `--overwrite` is included because canonical outputs already exist
in the repo; without it `main.py` fails closed to prevent accidental replacement.
If the snapshot download fails, run `python main.py --fetch-data-snapshot embedded`
then retry.

```bash
# One-time: warm the HuggingFace encoder models (MPNet + MiniLM + SciBERT) into the local cache.
python main.py --fetch-encoder-models

# NOTE: --cold-replay rebuilds embeddings from scratch using the frozen raw
# snapshot. GPU embedding is not bit-deterministic across hardware (CUDA
# determinism flags were tested but hurt runtime too much), so results may
# differ at the last mantissa bits. Use --warm-replay (above) for exact
# reproduction of the submitted outputs.
python main.py --cold-replay --overwrite
```

## What the replay produces

Warm replay rebuilds:
- canonical machine-readable outputs under `4_outputs/{model}/data/`
- manuscript tables and figures under `4_outputs/{model}/tables/` and `4_outputs/{model}/figures/`

To build the dissertation PDF from warm-replay outputs, run `python main.py --build-pdf --overwrite` (requires bash — WSL/Linux only). If you have your own LaTeX distribution, you can also compile `3_writing/dissertation.tex` directly with `latexmk`, `pdflatex` + `biber`, or your preferred compiler.

## Pipeline architecture

The analysis pipeline measures research-policy divergence in AI-for-SDG discourse
using **three method axes** that share a unified data-preparation pipeline:

### Unified stages (shared by all axes)

```
Preprocess (8 scripts) → Segment (canonical, 7 corpora ONCE) → Embed (8 reference/policy shared + research: full for primary, 100k subset for sensitivity)
```

- **Labeled corpora** (ground-truth SDG labels): osdg, benchmark, sdg_knowledge_hub, sdgi, aurora
- **Unlabeled corpora** (to be assigned): research (OpenAlex), policy_scrape, policy_manual, ungdc_sdg
- osdg and benchmark are **not segmented** (embedded directly from preprocessed JSONL)
- sdgi plays a **dual role**: labeled training corpus AND merged into `policy.npy`
- **All sensitivity encoders reuse the canonical (MPNet) segmented texts.** Segmentation runs ONCE at 384 tokens; every model embeds those identical segments so the only varying factor is the encoder. Research: MPNet embeds the full corpus; **MiniLM and SciBERT embed a shared 100k-paper subset** (`research_subset`, deterministic seed-42 draw). Reference/policy: all models embed the canonical segments (`--seg-model all-mpnet-base-v2`). This removes the earlier per-model segmentation (and its chunking confound) plus the ~20GB+/model of redundant segmented text.

The shared training-data prep (`0_prepare_data.py`) writes `embeddings.npy`, `labels.npy`,
`sources.npy`, and train/test split indices to `2_data/4_supervised_model_results/{model}/`.

### Three method axes

| Axis | Classifier | Centroid built | Consumed by | Role |
|---|---|---|---|---|
| **A — Supervised LR** | Logistic Regression (C=10, lbfgs) | `research_centroids.npy` (from LR-assigned research papers) in `5_supervised_scored/{model}/` | `1_semantic_gap`, `0_coverage_gap`, `3_generate_cross_sensitivity_table` | **PRIMARY** — reported in dissertation main text |
| **B — Supervised MLP** | 4-layer MLP (384-hidden) | `mlp_research_centroids.npy` in `5_supervised_scored/{model}/mlp_scores/` | `3_generate_cross_sensitivity_table` | Sensitivity axis |
| **C — Zeroshot** | Nearest-centroid assignment | `sdg_centroids.npy` (reference, from labeled train split), `zeroshot/research_centroids.npy`, `zeroshot/policy_centroids.npy` in `4_outputs/{model}/zeroshot/` | `3_generate_cross_sensitivity_table`, `0_check_centroid_consistency`, `1_build_centroid_similarity_matrix` | Sensitivity axis |

The **reference centroids** (`sdg_centroids.npy`) are built once from the labeled
training split and serve as:
- the assignment target for zeroshot scoring,
- the input to the centroid similarity matrix, and
- the basis for PCA semantic-landscape visualisation.

The **supervised research centroids** (`research_centroids.npy` from `score_supervised --lr --research`)
are the primary input to the semantic-gap analysis; the zeroshot produces its own
research/policy centroids in a separate namespace for cross-method comparison.

### Build-order constraint

```mermaid
flowchart TD
    subgraph Sources[Data sources]
        L["Labeled<br>osdg, benchmark, KH, sdgi, aurora"]
        U["Unlabeled<br>research, policy_scrape, policy_manual, ungdc_sdg"]
    end

    subgraph Prep["Preprocess → Segment (canonical, ONCE) → Embed"]
        PSE["8 preprocess scripts<br>1 canonical 1_segment_corpus run (shared by all models)<br>8 embed + merge policy<br>embed_paper_shards (full for primary, 100k subset for MiniLM/SciBERT)"]
    end

    subgraph Train[Shared training data]
        PD["0_prepare_data.py<br>→ 4_supervised_model_results/{model}/<br>embeddings.npy, labels.npy<br>sources.npy, indices/"]
    end

    subgraph LR[Axis A: Supervised LR — PRIMARY]
        R1["2_retrain_full_data LR<br>→ sdg_classifier.joblib"]
        S1["score_supervised --lr --research<br>→ 5_supervised_scored/{model}/<br>research_centroids.npy<br>score_supervised --lr --policy<br>→ policy_scores.npy"]
    end

    subgraph MLP[Axis B: Supervised MLP — sensitivity]
        R2["2_retrain_full_data MLP<br>→ mlp_retrained.joblib"]
        S2["score_supervised --mlp<br>→ mlp_research_centroids.npy"]
    end

    subgraph ZS[Axis C: Zeroshot nearest-centroid — sensitivity]
        RC["0_build_sdg_reference_centroids<br>→ 5_supervised_scored/{model}/<br>sdg_centroids.npy"]
        ZSO["score_zeroshot<br>→ 4_outputs/{model}/zeroshot/<br>research_centroids.npy<br>policy_centroids.npy"]
    end

    subgraph Analysis[Downstream]
        G1["0_coverage_gap<br>← LR policy_scores.npy"]
        G2["1_semantic_gap<br>← supervised research_centroids.npy"]
        XT["3_generate_cross_sensitivity_table<br>← LR + MLP + zeroshot artifacts"]
        SM["1_build_centroid_similarity_matrix<br>← sdg_centroids.npy"]
    end

    L --> PSE
    U --> PSE
    PSE --> PD
    PD --> R1
    PD --> R2
    PD --> RC
    R1 --> S1
    R2 --> S2
    RC --> ZSO
    S1 --> G1
    S1 --> G2
    S1 --> XT
    S2 --> XT
    ZSO --> XT
    RC --> SM

    style LR fill:#dae8fc,stroke:#2c6e9c
    style MLP fill:#e8d4f1,stroke:#8e2c9c
    style ZS fill:#f1e6d4,stroke:#9c6e2c
```

**Order constraint:** `0_prepare_data.py` MUST run before both
`0_build_sdg_reference_centroids.py` and `2_retrain_full_data.py`, because both
read the `embeddings.npy` / `labels.npy` files that `0_prepare_data.py` writes
to `2_data/4_supervised_model_results/{model}/`.

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

`2_data/` is hydrated from the frozen embedded snapshot. `4_outputs/` is committed for marker inspection but can be regenerated from the snapshot and source code.

### Environment notes

- `environment.yml` is the canonical rebuild path. Pins 14 core Python packages;
  platform-specific libraries (CUDA, Linux libs) are handled by conda/pip per-OS.
- **nltk data is NOT bundled with the conda install.** The register-validation
  appendix (`--appendix-a1-register-validation`) needs the `punkt` and
  `averaged_perceptron_tagger_eng` data resources. On a fresh environment, fetch
  them once:
  `python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger_eng')"`
  (the stage fails closed with this hint if they are missing).
- `requirements.txt` is a human-edited core reference (14 packages). Not needed
  for rebuild — `environment.yml` already covers the pip layer.
- Python version: `3.11`.
- CPU is sufficient for `--warm-replay-without-appendix`

## Additional optional commands

| Command | What it does |
|---|---|
| `python main.py` | Read-only status check |
| `python main.py --warm-replay-without-appendix --overwrite` | Rebuild main text analysis from snapshot (no PDF, no appendix) |
| `python main.py --warm-replay-with-appendix --overwrite` | Rebuild main text + all appendix analyses from snapshot (no PDF) |
| `python main.py --cold-replay --overwrite` | Full pipeline from the raw frozen snapshot: rebuilds **all three encoder tracks** (MPNet + MiniLM + SciBERT) deterministically in one run. No OpenAlex credentials needed when the raw snapshot is hydrated (see [§Reproducibility boundaries](#reproducibility-boundaries)). |
| `python main.py --appendix-all --overwrite` | Run all appendix stages (A2, A3, B2, C, C1, C0, D1, H1, I1, F2, J1, K1) standalone (no PDF) |
| `python main.py --appendix-a1-register-validation --overwrite` | Run Register-Removal Validation against Independent Linguistic Register Markers (Appendix G; canonical MPNet only; needs nltk `punkt` + `averaged_perceptron_tagger_eng`, see [Environment](#environment)) |
| `python main.py --appendix-a2-family --overwrite` | Run A.2 Policy Source-Family Sensitivity |
| `python main.py --appendix-a3-sdg4 --overwrite` | Run A.3 SDG 4 Lexical Artefact Audit |
| `python main.py --appendix-b2-interpret --overwrite` | Run B.2 Lexical Illustration of the Semantic Gap |
| `python main.py --appendix-c-sample-stability --overwrite` | Run C Sample-Stability Robustness |
| `python main.py --appendix-c1-balanced-subset --overwrite` | Run C.1 Balanced-Subset Rank-Stability |
| `python main.py --appendix-c0-corpus-split --overwrite` | Run C.0 Corpus Split Macro Export |
| `python main.py --appendix-d1-model-selection --overwrite` | Run D.1 Model-Selection CV Macros |
| `python main.py --appendix-h1-cross-method --overwrite` | Run H.1 Cross-Method Gap Values |
| `python main.py --appendix-i1-assignment-method --overwrite` | Run I.1 Assignment Method Comparison |
| `python main.py --appendix-g-distributional --overwrite` | OPT-IN main-result table: distributional semantic-gap robustness. NOT run by warm replay or `--appendix-all`; run this before `--build-pdf`. |
| `python main.py --build-pdf --overwrite` | Build PDF from existing outputs (WSL/Linux only — requires bash) |
| `python main.py --fetch-data-snapshot embedded` | Hydrate embedded snapshot into `2_data/` |
| `python main.py --fetch-data-snapshot raw` | Hydrate raw snapshot for cold-replay rebuilds |
| `python main.py --backup-data-snapshot {raw\|embedded\|both}` | Create and upload a snapshot archive (maintainer-only — requires rclone on WSL) |

## Reproducibility boundaries

### Warm replay (canonical target)

Deterministic from the frozen embedded snapshot. No network needed after
hydration. Byte-identical across runs and platforms.

### Full cold-replay pipeline

```mermaid
flowchart LR
    Fetch["1. Fetch<br><em>live sources</em>"] --> Preproc["2. Preprocess<br><em>deterministic</em>"]
    Preproc --> Seg["3. Segment<br><em>canonical, once</em>"]
    Seg --> Embed["4. Embed<br><em>deterministic</em>"]
    Embed --> Analyse["5. Analyse<br><em>deterministic</em>"]
    Fetch -.-|"will drift"| Note["OpenAlex changes daily,<br>policy URLs fragile,<br>manual PDFs not automatable"]
```

Four deterministic stages produce identical results given frozen inputs.
The fetch stage cannot be reproduced because its sources change continuously.

### What drifts on live re-fetch

| Source | Drift mechanism |
|---|---|
| OpenAlex API | Papers added/deleted daily; abstracts and SDG classifications change |
| Policy scrape URLs | ~44 hardcoded HTTP links; many unconfirmed; PDFs may 403/redirect |
| Manual policy supplement | 65 PDFs from non-API sources — not automatable |
| GitHub benchmark | `SDGClassification/benchmark@main` — moving branch |
| HF dataset / Dataverse | `UNDP/sdgi-corpus` and UNGDC may be versioned |

### Credentials

A **live** `--cold-replay` OpenAlex re-fetch requires OpenAlex API credentials.
Copy `.env.example` to `.env` and fill in your key (free at
https://openalex.org/keys). Without these the live fetch stage raises
`RuntimeError`. The 3 rate-limit fallback keys are optional — only
`OPENALEX_MAILTO` + `OPENALEX_API_KEY` are required. If provided, they enable
parallel API key rotation during the full re-fetch. **Note:** a `--cold-replay`
run from the frozen **raw snapshot** is deterministic and offline — it needs
**no** OpenAlex credentials. The raw snapshot is auto-fetched if missing.

### Snapshot scope

- **Embedded snapshot**: contains `3_embedded/` (frozen embeddings) plus
  `3a_warm_replay_texts/` (gzipped copies of exactly the segment text the
  appendix analyses A3/B2 read: research shards + `policy.jsonl` for the
  default model). For warm-replay analysis.
  Warm replay from embedded is the canonical reproducibility target.
  No network needed after hydration. Byte-identical across runs and platforms.
  Canonical plain-text segment files (`2_segmented/`) are a producer-side
  artifact regenerated during cold replay; analysis code prefers them when
  present and falls back to `3a_warm_replay_texts/` otherwise.
- **Raw snapshot**: contains only `0_raw/`. For cold-replay rebuilds.
  Cold replay from raw will re-run preprocessing, segmentation, embedding, and
  training — outputs will differ from the submitted state due to OpenAlex live changes.

## Repository layout

The repository uses a numbered directory convention. Each prefix indicates
the directory's role in the workflow:

- `0_literature/` — cited and consulted sources, organised by paper content
- `1_code/` — all pipeline and analysis code
- `2_data/` — frozen data snapshot (hydrated from archive, gitignored)
- `3_writing/` — manuscript source (`dissertation.tex`, `references.bib`, build script)
- `4_outputs/` — committed outputs (`dissertation.pdf`, `main/{model}/` figures and tables, `appendix/{model}/` analyses)
- `5_notes/` — working notes, assumptions, and workflow logs

Entrypoint and environment files at root:
- `main.py` — single reproducibility entrypoint
- `environment.yml` — full conda + pip environment specification
- `requirements.txt` — lightweight human-readable dependency reference
- `README.md` — this file
