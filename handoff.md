# HANDOFF — `main.py` architecture refactor (main-refactor branch, COMPLETE)

> All three phases are done and committed on `main-refactor`. The only remaining
> action is merging `main-refactor` → `main`, which requires **explicit user
> approval** (no push/merge without instruction).

## Status

- **Phase 1 — committed `06b4c34`**: single appendix registry (`APPENDIX_SPECS`)
  + fix of the C0 no-op bug (dispatch now iterates the registry).
- **Phase 2 — committed `cfd8553`**: shared stage builders
  (`_preprocess_steps`/`_segment_steps`/`_embed_model_steps`/`_concept_track_steps`),
  unified concept track, single `_run_analysis_poststeps` call by all three
  consumers (warm/cold/`--stage analysis`).
- **Phase 3 — committed `bec3a7d`**: `--stage analysis` is now single-model
  (compose `_run_main_analysis_steps` for `--embed-model` only, then
  `_run_analysis_poststeps`); 3-encoder aggregation is cold-replay-only; dropped
  the unused `_run_analysis_only`; fixed a **silent double-run** (Phase 2 left the
  cross-sensitivity+figures tail inside `_run_main_analysis_steps`, so every
  default-model replay ran them twice — now once via poststeps); updated
  module docstring + PIPELINE.md.

## Verification

- Warm replay (`--warm-replay-without-appendix --overwrite`) ran **green** with
  canonical correlations unchanged (MPNet LR raw=−0.078, adj=0.475, reg=−0.493).
- Command-list capture confirms cross-sensitivity + figures now run **exactly
  once** for `--stage analysis` (was twice). All three consumers call
  `_run_analysis_poststeps` after `_run_main_analysis_steps`.
- `py_compile` of `main.py` passes; read-only `python main.py` status works.
- **Caveat:** the authoritative warm replay that proved green was captured
  *during the double-run*, so it validated output correctness, not the single-run
  path. The single-run path is a pure removal of a duplicated idempotent command,
  so outputs are unchanged by construction — but a clean warm replay after the
  Phase 3 fix has **not** been re-run. Recommended before merge (optional, ~20 min).

## Key invariants (preserved)

- `APPENDIX_SPECS` (in `analysis_orchestrator.py`) is the single source of truth;
  never re-introduce a hardcoded appendix flag list in `action_requested`/guard/
  dispatch.
- `_run_main_analysis_steps` does **not** produce cross-sensitivity+figures;
  `_run_analysis_poststeps` (gated to default model) does, exactly once.
- Cold-replay embed passes `--local-files-only` (offline-determinism; needs HF
  cache present).

## Remaining

1. **Merge `main-refactor` → `main` only on explicit user approval.** Do NOT push.
2. Optional: re-run warm replay after `bec3a7d` to confirm single-run green (above).
3. `5_notes/scratch/` is scratch-only — never commit it.
