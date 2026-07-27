# AGENTS.md

Operational guidance for working in this dissertation reproducibility repo.
The README (`README.md`) is the authoritative source; this file captures the
non-obvious facts that are easy to miss.

## Single entrypoint

- `main.py` is the only reproducibility entrypoint. There is no test/lint/typecheck
  suite, build system, or package manager beyond Conda — do not look for one.
- **Default mode is read-only.** Running `python main.py` with no action flag only
  prints repo status. Any command that writes outputs requires `--overwrite`,
  otherwise it fails closed to protect committed `4_outputs/`.
- The canonical reproducibility target is **warm replay** (deterministic, offline
  after snapshot hydration):
  `python main.py --warm-replay-without-appendix --overwrite`
  Use `--warm-replay-appendix` to also rebuild appendix analyses.

## Environment

- Build with Conda from `environment.yml` (`conda env create -f environment.yml`,
  then `conda activate dissertation`). Python 3.11.
- `requirements.txt` is a human reference only — `environment.yml` is the real
  build path. Do not rely on `pip install -r requirements.txt` for a clean rebuild.
- `--cold-replay` (full live pipeline) needs `.env` with `OPENALEX_MAILTO` +
  `OPENALEX_API_KEY` (copy from `.env.example`). It takes ~1 week and drifts
  because OpenAlex/policy URLs change — avoid unless explicitly asked.

## Data and outputs

- `2_data/` is **gitignored** and hydrated from a frozen snapshot, not committed.
  Hydrate with `python main.py --fetch-data-snapshot embedded` (or `raw` for
  cold replay). Warm replay needs the embedded snapshot present first.
- `4_outputs/` is committed for inspection but regenerable. Outputs are
  namespaced by embedding model under `4_outputs/main/{model}/...`.
- `--build-pdf --overwrite` requires **bash (WSL/Linux)** — not supported on bare
  Windows. It compiles `3_writing/dissertation.tex` via `latexmk`/`pdflatex`/`biber`.

## Models and stages

- Default embedding model is `all-mpnet-base-v2`. Override with `--embed-model`
  (alias `--model`); this changes the `{model}` output namespace.
- `--stage <fetch|preprocess|segment|embed|train|infer|centroids|analysis>` runs a
  single stage assuming upstream outputs exist. `--snapshot-profile` defaults to
  `embedded`.
- Appendix stages (`--appendix-*`, `--appendix-all`) require existing main-text
  outputs first.

## Layout

Numbered directories encode workflow role: `0_literature/`, `1_code/` (only active
pipeline code, see `1_code/README.md`), `2_data/` (gitignored), `3_writing/`
(manuscript source), `4_outputs/` (committed), `5_notes/` (working notes).
