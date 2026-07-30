# AGENTS.md

Operational guidance for working in this dissertation reproducibility repo.
The README (`README.md`) is the authoritative source; this file captures the
non-obvious facts that are easy to miss.

## Engineering standards (non-negotiable)

- **Incremental write/save is the default for everything.** Long-running
  fetch/embed/score scripts must write+flush every record (or every batch)
  and save checkpoints — never accumulate-then-dump. Pair with **resume-safety**:
  on restart, detect the last written artifact and continue from there, so an
  abort loses nothing already written. (A session once lost a full embed because
  the script only saved at the end — see Gotchas.)
- **Verify, don't trust.** Treat every result, tool output, and prior
  agent's claim as unproven until reproduced. Re-run the smallest possible
  test; try to break it before declaring success. Prefer synthetic data over
  production data. Assume you know nothing about this codebase until you've
  read the actual code; assume any subagent's output is wrong until checked.
- **Reproducibility is the top priority.** Deterministic seeds for all
  randomness — record the seed in the OUTPUT, not just the source. Never
  overwrite without an explicit `--overwrite`; prefer cached/reused artifacts over
  recomputation. Every long-running script must be resume-safe (safe
  checkpoints). Pin and record the environment (deps/versions) with each output.
- **No magic numbers.** No undocumented constants, thresholds, or default
  hyperparameters. Surface them as named constants or CLI flags with documented
  rationale, and record any result-affecting one in the run's output. Prefer
  validation over arbitrary picks; document the choice. Fail closed — never
  silently swallow or default-past errors that affect results.
- **Follow repo conventions exactly.** Match existing naming; if unclear, ask.
  Numbered prefixes encode dependencies: same prefix = independent; N depends on
  N-1's outputs. Code only in `1_code/` (+ `main.py`); outputs only in
  `4_outputs/`; data only in `2_data/` (except final manuscript-ready artifacts).
- **Disciplined changes.** One concern per commit; clear standalone message;
  re-verify the affected stage (compile/run) before committing. Stop and hand
  off immediately on anything suspicious or needing judgment. Handoff = a
  self-sufficient message usable by a fresh agent with zero context: (1) where
  we are, (2) key facts to pick up without re-reading, (3) actions/decisions +
  files changed + why, (4) what remains + why, (5) concerns, (6) exactly
  what was interrupted. Delegate only cleanly-scoped subtasks to subagents;
  always verify their output.
- **Keep it simple.** Don't overcomplicate research design, wording, or code.

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
- The embedded snapshot ships `3_embedded/` + `3a_warm_replay_texts/` (gzipped
  A3/B2 appendix text for the default model, built only by
  `build_warm_replay_texts.py` during backup). Appendix readers resolve
  canonical `2_segmented/` first, then fall back to `3a_warm_replay_texts/`
  (`model_utils.resolve_research_text_path` / `resolve_policy_text_path`).
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

## Gotchas (learned the hard way)

- **Verify exact directory names before referencing them in code.** The
  robustness scratch dir is `__test_queries/` (single trailing underscore).
  A `__test_queries__/` (double) typo in a `Path(...) / "name"` that
  doesn't exist is `mkdir(parents=True)`'d **silently**, and because the
  expected cache file is then absent the script re-fetches from scratch
  (wasted hours + API calls). Always `ls` the exact path first.
- **Long jobs are killed at the ~120s tool timeout** (the harness kills the
  whole process group). Launch with `setsid <cmd> > log 2>&1 & disown`;
  poll via `pgrep -af <name>`, never the wrapper PID.
- **Standalone/robustness checks are scratch-only.** Write ONLY to
  `5_notes/scratch/` or `__test_queries/` (or `/tmp`); never `2_data/` or
  `4_outputs/`. Reuse the already-trained LR
  (`2_data/4_supervised_model_results/<model>/model/sdg_classifier_retrained.joblib`);
  never retrain.
- **Verify citations before use.** A prior session fabricated
  "Carraud & Gault, OECD STI WP" (two guessed DOIs 404'd). Real sources for
  the AI-query-term check: OECD.AI methodological note
  (`oecd.ai/en/partner-data-methodological-note`), Elsevier (2018)
  *Artificial Intelligence: How knowledge is created, transferred, and used*,
  and CSET (2023) Schoeberl, Toney & Dunham, *Identifying AI Research*,
  DOI **10.51593/20230** (independently resolved). Resolve a DOI before citing.
- **OpenAlex filter discipline.** `concepts.id` is a valid `works` filter;
  `concepts.display_name` is NOT. Free-text uses `search=<term>`. SDG filter:
  `sustainable_development_goals.id:https://metadata.un.org/sdg/N`.
- **Embedding text must match the baseline.** Research-corpus text =
  `f"{title}. {abstract}"` (`1_code/1_preprocess/0_preprocess_papers_streaming.py`).
  Any subset must embed the same string or the LR scores a different
  representation.
