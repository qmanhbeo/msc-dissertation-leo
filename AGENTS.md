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
- `--cold-replay --overwrite` rebuilds **all three encoder tracks**
  (MPNet + MiniLM + SciBERT) from the raw snapshot in one run — no OpenAlex
  credentials needed when the raw snapshot is hydrated. `--embed-model` is
  ignored by `--cold-replay`. Single-model rebuilds stay available via
  `--stage`.
- The embedded snapshot ships `3_embedded/` + `3a_warm_replay_texts/` (gzipped
  A3/B2 appendix text for the default model, built only by
  `build_warm_replay_texts.py` during backup). Appendix readers resolve
  canonical `2_segmented/` first, then fall back to `3a_warm_replay_texts/`
  (`model_utils.resolve_research_text_path` / `resolve_policy_text_path`).
- `4_outputs/` is committed for inspection but regenerable. Outputs are
  namespaced by embedding model under `4_outputs/{model}/...` (flattened
  from the previous `main/{model}/` layout). Appendix outputs live under
  `4_outputs/appendix/{model}/...`. Pipeline-intermediate `.npy` score
  files are kept in `2_data/5_supervised_scored/{model}/zeroshot/` (they
  are NOT publishable artifacts).
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

## Manuscript scope decisions

- **Zero-shot is a single comparison, not an axis.** The zero-shot
  nearest-centroid method is scoped to ONE comparison: canonical supervised
  (LR; MLP as robustness) vs nearest-centroid. It appears in the manuscript
  ONLY as the single zero-shot column under the MPNet group of the
  encoder-sensitivity tables and in Appendix I.1 (supervised vs nearest-
  centroid assignment comparison). It must NOT span encoders, policy-source
  families, segment caps, or retrieval strategies in any manuscript-facing
  table or prose, even though the pipeline scripts can compute it across those
  axes via flags. MLP keeps spanning encoders (it is a supervised robustness
  check); only zero-shot is restricted.
- `3_generate_cross_sensitivity_table.py` gates zero-shot columns to
  `DEFAULT_EMBED_MODEL` only; `h1_cross_method_gap_values.py` emits zero-shot
  only for MPNet (7 data columns + concept LR). Do not re-add ZS columns for
  MiniLM/SciBERT without revoking this decision.

## Pipeline invalidation model

Two distinct mechanisms protect different failure modes:

- **Per-shard/batch resume** covers the expensive frontier stages (embed,
  segment, score_supervised LR shard loop). These stages track completed shards
  in a manifest; on restart or snapshot re-hydration they skip completed work
  without re-running.  Do NOT add mtime-based fingerprinting here —
  `2_data/` re-hydration resets all mtimes, which would force a full expensive
  re-run and destroy the resume benefit.
- **Fingerprint-gated skip** (via `shared_utils.should_skip` /
  `record_fingerprint`) covers the 14 analysis/appendix scripts (Tier B).
  Each stage fingerprints its *direct inputs*; if an upstream re-runs, the
  fingerprint changes and the downstream re-derives.  These stages are fast
  enough that re-running on re-hydration is acceptable.
- **Existence-skip + `--overwrite`** covers centroids build,
  merge_policy, score_zeroshot, retrain_full_data, and plot_figures.
  If the output file(s) exist and `--overwrite` is not passed, the stage is
  skipped.  Cheap enough that re-running on re-hydration is acceptable.

`--overwrite` is designed for whole-replay use (`--cold-replay --overwrite`,
`--warm-replay-without-appendix --overwrite`). Piecemeal `--overwrite` of
a single mid-pipeline script invalidates its direct downstream's fingerprint,
but the pipeline does not implement a full make-style DAG — stale outputs can
persist if you `--overwrite` one stage in isolation while its downstream
outputs already exist from a prior run. In practice this is safe because
warm replay freezes inputs and cold replay re-runs everything, so the
staleness gap only opens on non-standard piecemeal workflows.

### Checkpoint / Overwrite inventory

| Stage | Resume mechanism | `--overwrite` | Atomic writes? | Comment |
|---|---|---|---|---|
| fetch (all) | incremental I/O per record | not forwarded | N/A | gitignored raw data; resume from last checkpoint per data source |
| **preprocess** (8 small scripts) | `_resume.py` checkpoint: `rows_done` + `out_offset` | `--reset` forwarded | truncate-to-offset (heals torn tail) | just-added (2026-07-30) |
| preprocess (research streaming) | `state.json` + shard manifest | `--reset` forwarded | `tmp.replace` per shard | already present |
| segment (sharded / `--all`) | per-shard file-existence | forwarded | `tmp.replace` per shard | |
| segment (single-file) | existence-skip | forwarded | `tmp.replace` (just-added) | |
| embed (paper shards) | shard + chunk manifest | forwarded | `tmp.replace` per chunk/batch | |
| embed (ref/policy corpora) | per-batch manifest + skip | forwarded | `tmp.replace` per batch; `atomic_write_npy` at concat | bug fix (2026-07-30): `--overwrite` now rmtree stale batches unconditionally |
| merge_policy (.npy + ids) | existence-skip | forwarded | `.npy` tmp+replace; `atomic_write_json` for ids (just-added) | |
| score_supervised (LR path) | shard manifest | forwarded | `atomic_write_npy` for centroids (just-added); shard scores already atomic | |
| score_supervised (MLP path) | whole-output gate (no per-shard) | forwarded | `atomic_write_npy` (just-added) | no per-shard manifest — kill mid-loop loses MLP pass |
| score_zeroshot | existence-skip | forwarded (just-added) | bare `np.save` | just-added `--overwrite` + skip guard |
| centroids build / consistency / similarity | existence-skip | forwarded | `atomic_write_npy` (just-added); similarity CSV already atomic | |
| retrain_full_data (LR + MLP) | existence-skip | forwarded (just-added) | `atomic_write_joblib` (just-added) | no per-epoch checkpoint; kill mid-MLP loses epochs |
| plot_figures | existence-skip | forwarded (just-added) | N/A (matplotlib writes) | |
| analysis / appendix (14 scripts) | `shared_utils.should_skip` / `record_fingerprint` | forwarded | tex,json via `open("w")` (small, fast) | Tier B — all verified correct (2026-07-30) |

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
- **CUDA allocator watermark trap (small-VRAM cards).** If any corpus segment
  exceeds the embedder's window (~512 tokens — e.g. XML/math-junk abstracts up
  to 920 tokens), a single spike batch ratchets the caching-allocator watermark
  toward card capacity; every later batch then pays CPU-side eviction churn
  (~20 texts/s vs ~110 on a 4GB card, GPU otherwise healthy). The paper-shards
  embedder (`0_embed_paper_shards.py`) calls `torch.cuda.empty_cache()` after
  every batch to prevent this — values are unaffected. Don't "fix" throughput
  by re-running with `--overwrite`; a running embed is safe to let finish.
