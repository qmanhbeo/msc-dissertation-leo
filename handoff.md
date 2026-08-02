# HANDOFF — dissertation reproducibility repo: LR/MLP/ZS + concept-track parallelism (A–E, F1–F4, E.12 all DONE)

> Audience: a fresh agent picking this up cold. Everything needed to understand
> state, continue, or verify is below. This supersedes and replaces all prior
> `handoff-2.md` / `handoff-3.md` (those files have been deleted). `handoff.md`
> is itself a working doc kept only locally (see §4.5).

---

## 1) Context — where we are

This is the `dissertation-bham` reproducibility repo. Single entrypoint `main.py`,
Conda env `dissertation`, Python 3.11, **no test/lint suite** (per AGENTS.md).
Reproducibility target is a deterministic **warm replay** on a hydrated embedded
snapshot:
`python main.py --warm-replay-without-appendix --overwrite`.

The pipeline scores research/policy corpora for the 17 SDGs via **three
assignment methods**:
- **LR** — canonical supervised (C=10, L2, lbfgs).
- **MLP** — supervised robustness check (4-layer/384, lr=3e-4).
- **ZS** — zero-shot nearest-centroid, **MPNet-group-only** (per AGENTS.md axis
  restriction: do NOT re-add ZS columns for MiniLM/SciBERT).

A prior session audited the three routes for structural parallelism
(`handoff-2.md`) → approved 5-part plan A–E. A later session found the
**concept-retrieval track** still non-parallel (`handoff-3.md`) → plan F1–F4.
This session verified both audits against code, executed everything, and also
closed the one deferred item (**E.12**: persist MLP per-shard research scores so
`i1` reports research MLP-vs-ZS directly instead of proxying it).

**Current branch state:** `main` is at `f985058`
(`feat(E.12): persist MLP per-shard research scores; report research MLP-vs-ZS
directly in i1`). All work is **committed and pushed to `origin/main`**. No
branch other than `main` exists. Nothing is mid-flight; all tmux jobs have
exited.

Commits on `main` (most recent last):
- `ee2b42e` — A–E source fixes (7 files)
- `4763893` — regenerate outputs + re-derive dissertation prose
- `0647865` — rebuild PDF after A–E rank-text edits
- `45d9105` — concept-track fix (F1–F4) + fp_paths hardening
- `f985058` — E.12 (MLP research shards persisted; i1 direct research MLP-vs-ZS)

---

## 2) Key known facts (verified — do not re-derive unless in doubt)

### The three routes, as wired (after A–E + F1–F4 + E.12)
- **Shared inputs**: `0_prepare_data.py` makes one 85/15 stratified,
  document-grouped split (seed 42) → train-split-only labels/embeddings.
- **LR route**: `3_retrain_full_data.py` (C=10/l2/lbfgs) →
  `score_supervised.py --classifier lr --corpus research` →
  `research_centroids.npy` + `policy_scores.npy`; semantic gap via
  `1_semantic_gap.py` → `4_outputs/{model}/data/4_3_semantic_gap_distances.json`
  (capped per (doc, SDG) at 50).
- **MLP route**: `3_retrain_full_data.py --classifier-type mlp`
  → `score_supervised.py --classifier mlp` → `mlp_scores/`
  (`mlp_research_centroids.npy`, `mlp_policy_scores.npy`,
  `mlp_policy_vs_research.npy` (write-only), `mlp_summary.json` = **coverage
  only** — the `semantic_gaps` key was removed in A). **E.12 added:**
  `mlp_scores/mlp_research_scores_shards/part-NNNN.npy` (per-shard
  `predict_proba`, all 3 encoders; concept path `mlp_scores_concept/` too).
  Semantic gap via `1_semantic_gap.py --classifier mlp` →
  `4_3_mlp_semantic_gap_distances.json` (**capped, single source of truth**).
- **ZS route**: `0_build_sdg_reference_centroids.py` → `score_zeroshot.py`
  → `zeroshot/` npy + `data/semantic_gap_distances.json` (raw) and
  `data/adjusted/semantic_gap_distances.json`. Policy cap is per **(SDG, doc)**
  at 50 (B). Reliability via `MIN_CLUSTER_SIZE`(10) + norm<0.5 guard; each
  per-SDG record has `unreliable` + `unreliable_reason` (D).
- **Concept variant (MPNet only)**: concept-retrieved research corpus embedded
  in `2_data/3_embedded/mpnet/research_concept/`; scored by LR/MLP/ZS; gaps
  under `4_outputs/mpnet/data/concept/` (raw) and `concept/adjusted/`
  (adjusted). **F1–F4 fix** ensures the adjusted concept runs write to
  `concept/adjusted/` (not clobber the raw slot).

### Consumers (all read the correct, capped, raw/adjusted-split files)
- `3_generate_cross_sensitivity_table.py` → `tab_encoder_sensitivity_semantic.tex`,
  `num_encoder_sensitivity_semantic.tex` (feeds `\MlpSemanticRho`,
  `\ZeroShotSemanticRho`, `\SciBERTMeanGap`, etc.) — reads `4_3_mlp_*`
  (capped) + concept LR/MLP files; concept LR raw file added to `fp_paths`.
- `h1_cross_method_gap_values.py` → appendix cross-method table; reads concept
  LR/MLP files (added to `fp_paths`).
- `h1_register_correlation_table.py` — concept raw + adjusted LR/MLP files in
  `fp_paths`.
- `i1_assignment_method_comparison.py` (MPNet-only) — LR vs ZS + policy MLP vs
  ZS + **research MLP vs ZS (E.12, direct)**.

### Key numbers (all regenerated, MPNet unless noted)
- MLP raw gap SDG3 = **0.458** (capped; was 0.429 from the old uncapped
  `mlp_summary.json`).
- ZS SDG16 capped policy count = **3747** (was 3195 under the old global cap;
  LR is 4775 — genuine method divergence).
- Cross-sensitivity rank correlations (auto-updated): `\MlpSemanticRho` = 0.86,
  `\ZeroShotSemanticRho` = 0.44, `\MiniLMSemanticRho` = 0.65,
  `\SciBERTSemanticRho` = 0.39.
- Concept: `\ConceptSemanticGapRho` = **0.88** (was 0.34 from the corrupt
  F1 data), `\ConceptCoverageGapRho` = 0.88.
- i1 research agreement (MPNet, n=3.1M segments): LR-vs-ZS = **0.623**,
  MLP-vs-ZS = **0.616** (E.12, direct). Policy MLP-vs-ZS = 0.797 (segment) /
  0.804 (doc).
- Concept register-correlation cancellation pattern (genuine raw data):
  Concept LR raw ρ=0.02 (≈0), adj ρ=+0.25, register ρ=−0.13.

### Environment / operational facts
- `2_data/` is gitignored and hydrated from a frozen embedded snapshot
  (`python main.py --fetch-data-snapshot embedded`); warm replay needs it present.
- `4_outputs/` is committed and regenerable (regen needs `--overwrite`).
- Long jobs run in tmux (they exceed the 120s tool timeout); poll the log,
  never the wrapper PID. Logs in `/tmp/opencode/`.
- `5_notes/scratch/` is untracked scratch — never commit it.
- `handoff.md` is a working doc, currently modified locally (this file) — not
  committed by design.

---

## 3) Actions / decisions made this session, and files changed, and why

**Decision (user):** implement A–E + F1–F4 + E.12; merge, build PDF, commit,
push; delete `handoff-2.md`/`handoff-3.md`; keep `handoff.md` as local working
doc.

**Step A — Merge A–E (carry-over from prior handoff):**
- Fast-forwarded `main` to the `mlp-zs-parallel` branch tip
  (`git branch -f main mlp-zs-parallel` then `git checkout main`) — preserves
  the uncommitted `handoff.md` edit and untracked docs. Pushed.
- Rebuilt `dissertation.pdf` (`--build-pdf --overwrite`, clean) → commit
  `0647865`.

**Step B — Concept-track fix (handoff-3 / F1–F4):**
- `1_code/7_main_analysis/1_main_text/1_semantic_gap.py:191` — changed
  `adj_data_dir = Path(args.out_data_dir)` →
  `Path(args.out_data_dir) / "adjusted"`. Fixes F1 (adjusted concept runs no
  longer clobber the raw slot), F3 (no reliance on step ordering), F4 (distinct
  fingerprint sidecars). `score_zeroshot.py` already routed adjusted correctly,
  so the bug was contained to this one line.
- `3_generate_cross_sensitivity_table.py` — added
  `concept/4_3_semantic_gap_distances.json` (LR raw) to `fp_paths`.
- `h1_cross_method_gap_values.py` — added concept LR/MLP raw + adjusted +
  coverage files to `fp_paths` (register-correlation table already had them).
- Regenerated the MPNet concept track (4 `1_semantic_gap.py` concept variants)
  then full MPNet analysis+appendix via `python main.py --stage analysis
  --embed-model all-mpnet-base-v2 --overwrite`.
- Verified: `concept/4_3_semantic_gap_distances.json` → `"embeddings":"raw"`
  (was "adjusted"); `\ConceptSemanticGapRho` 0.34 → **0.88**; concept
  register-correlation shows expected cancellation with genuine raw data.
- Rebuilt PDF → commit `45d9105`, pushed. Deleted `handoff-2.md`/`handoff-3.md`.

**Step C — E.12 (persist MLP per-shard research scores):**
- `1_code/5_supervised_model_infer/score_supervised.py` `run_mlp` — inside the
  research shard loop, now saves `scores` (per-shard `predict_proba`) to
  `mlp_scores/mlp_research_scores_shards/part-NNNN.npy` (atomic tmp+replace),
  mirroring `paper_scores_shards`. Applies to both main (`mlp_scores/`) and
  concept (`mlp_scores_concept/`) paths. No model retrain (reuses
  `mlp_retrained.joblib`).
- `1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py`:
  - Reuses `check_research` with `mlp_research_scores_shards` to compute
    research MLP-vs-ZS **directly** (no longer proxied by LR-vs-MLP gap-rank).
  - Adds `research_mlp_vs_zs` to the output JSON and a new **"Res MLP %"**
    column to the appendix table.
  - Adds the persisted shard `.npy` files (via glob) to `fp_paths`.
  - Updated module docstring (proxy no longer needed).
- Regenerated MLP research shards for all 3 encoders:
  `python 1_code/5_supervised_model_infer/score_supervised.py --embed-model
  {all-mpnet-base-v2,all-MiniLM-L6-v2,scibert} --classifier mlp --corpus
  research --overwrite` (tmux). Note: MiniLM/SciBERT research manifests have 1
  shard (50000 rows) vs MPNet's 26 shards (3.1M rows) — that is **preexisting
  embed-snapshot data**, not a bug; functionally fine.
- Ran `python main.py --appendix-i1-assignment-method --overwrite` → research
  MLP-vs-ZS = **0.616** (n=3.1M).
- `3_writing/dissertation.tex:483` — added one sentence noting the MLP check
  agrees with zero-shot at the research assignment level (Appendix I.1).
- Rebuilt PDF (clean) → commit `f985058`, pushed.

**Not committed:** `handoff.md` (this working doc, intentionally left modified
locally per user choice) and `5_notes/scratch/`.

---

## 4) What remains and why

1. **Nothing code/analysis remains.** Every item in `handoff-2.md` (A–E),
   `handoff-3.md` (F1–F4), and the deferred E.12 is implemented, verified,
   committed, and pushed.
2. **Housekeeping only:** `handoff.md` is a local modified working doc (this
   message). It is intentionally NOT committed (user chose "leave as-is").
   `5_notes/scratch/` is untracked and must never be committed.
3. **No open branch, no open PR, no interrupted job.**

---

## 5) Concerns to emphasize

- **Determinism:** the per-SDG capping in `score_zeroshot.py` reuses the same
  seeded RNG across SDG iterations (matching `compute_sdg_semantic_gaps` in
  `semantic_gap_shared.py`). Do **not** reorder SDG iteration or reseed
  mid-loop. Every seed is recorded in outputs.
- **ZS stays MPNet-group-only** (AGENTS.md). Never re-add ZS columns for
  MiniLM/SciBERT in `3_generate_cross_sensitivity_table.py` or
  `h1_cross_method_gap_values.py`.
- **Do NOT "fix" the adjusted-space asymmetry** (ZS re-assigns on projected
  embeddings; LR/MLP keep raw-space clusters, only projecting vectors). It is
  intentional (`PLAN_register_topic_decomposition.md` §6.1) and documented in
  the ZS JSON `note` + code comments. This is NOT a bug.
- **Single source of truth for MLP gaps** is
  `4_3_mlp_semantic_gap_distances.json` (raw + `adjusted/` + `concept/`).
  `mlp_summary.json` carries coverage only; never re-introduce a second MLP gap
  source.
- **2_data is gitignored / hydrated.** Any warm replay needs the embedded
  snapshot present first (`python main.py --fetch-data-snapshot embedded`).
- **`main.py` is read-only by default**; regen requires `--overwrite`.
- **MiniLM/SciBERT research manifests have 1 shard (50000 rows)** vs MPNet's 26
  (3.1M). This is preexisting snapshot data — do not "correct" it unless the
  user explicitly asks; it does not affect MPNet-only `i1` or the manuscript.
- **`handoff-2.md` / `handoff-3.md` are deleted.** Do not reference them; this
  file is the authoritative handoff. Prior audit text is background only.
- **Verify-don't-trust:** if you re-run any stage, re-check the manuscript
  numbers (every transcribed MLP/ZS/concept figure) against regenerated tables
  before treating them as final.

---

## 6) The whole comprehensive plan (as executed)

### A–E (LR/MLP/ZS main-track parallelism) — DONE
- **A. Single-source capped MLP gap:** `main.py` stage 9 runs raw
  `1_semantic_gap.py --classifier mlp` for every model + concept-MLP (MPNet);
  `3_generate_cross_sensitivity_table.py` and `h1_cross_method_gap_values.py`
  read `4_3_mlp_semantic_gap_distances.json`; `h1_register_correlation_table.py`
  dropped the legacy `mlp_summary` fallback; `score_supervised.run_mlp` removed
  the divergent `semantic_gaps` key (coverage only).
- **B. ZS policy cap per (SDG, doc):** `score_zeroshot.py` uses
  `cap_policy_indices_per_doc` per SDG (raw + adjusted + concept, one function);
  local `cap_indices_per_doc` deleted.
- **C. Adjusted-space rule documented, not changed:** `note` field in ZS JSON +
  comments in `score_zeroshot.py` and `1_semantic_gap.py` referencing PLAN §6.1.
- **D. Reliability alignment:** ZS uses `MIN_CLUSTER_SIZE`(10);
  `unreliable`/`unreliable_reason` fields; norm<0.5 kept as secondary guard.
- **E. Robustness guards:** `run_mlp` uses
  `resolve_manifest_path(allowed_dirs=(embed_root,))` + asserts
  `emb_dim == first_layer.in_features`; `score_zeroshot` validates
  `policy_emb.shape[0] == len(policy_ids)`. E.12 (persist MLP per-shard research
  scores) was the only deferred sub-item — now DONE (Step C above).

### F1–F4 (concept-retrieval track parallelism) — DONE
- **F1 (critical):** `1_semantic_gap.py:191` now writes adjusted concept output
  to `out_data_dir/adjusted` (was clobbering the raw slot).
- **F2 (stale adjusted files):** regen overwrites `concept/adjusted/` with fresh
  adjusted runs.
- **F3 (ordering fragility):** collision removed, so step order no longer
  matters.
- **F4 (sidecar collision):** raw/adjusted now distinct files → distinct
  sidecars.
- **Step 6 (fp_paths hardening):** concept LR raw/adjusted/coverage added to
  `fp_paths` in `3_generate_cross_sensitivity_table.py` and
  `h1_cross_method_gap_values.py` so tables re-derive when concept gaps change.

### E.12 (direct research MLP-vs-ZS in i1) — DONE
- Persist per-shard MLP research scores (all 3 encoders) in `score_supervised.py`.
- `i1` computes research MLP-vs-ZS directly (reusing `check_research`), adds it
  to JSON + a "Res MLP %" table column, includes the shards in `fp_paths`.
- `dissertation.tex` notes the MLP check agrees with ZS at the research
  assignment level. PDF rebuilt.

---

## 7) What I was doing but interrupted

**Nothing is interrupted.** All work in this session completed end-to-end and
is committed + pushed (`f985058`). All tmux sessions
(`conceptregen`, `analysis`, `buildpdf2`, `mlpshards`, `i1`, `buildpdf3`) have
exited cleanly; their logs remain in `/tmp/opencode/`.

For historical context only (already resolved, not live):
- The original A–E warm replay **crashed** on a `gap_dict` NameError in
  `run_mlp` (uncaught `print(gap_dict)`); fixed in `ee2b42e`. The authoritative
  re-run (`warm_mlpzs2.log`) was green.
- The concept-track regen was run as a **full `--stage analysis`** (MPNet only)
  rather than a minimal regen, because the downstream tables depend on the full
  analysis flow. It took ~15 min (dominated by the INLP `register_adjust`
  iterations, capped at `ITERATIVE_MAX_K=200`, and the H.1 1100-draw
  replication), but completed without error. A truly minimal regen would instead
  re-run only the 4 concept `1_semantic_gap.py` variants + the specific consumer
  scripts, but that path is more error-prone; the full `--stage analysis` is the
  safe canonical choice.

If a fresh agent needs to resume anything, there is **no half-finished edit and
no interrupted long-running job** — the repo is in a clean, fully-committed,
pushed state. The only deliberate local-only artifacts are `handoff.md` (this
doc) and `5_notes/scratch/`.

---

## Quick-reference commands

```bash
# Regenerate everything (needs hydrated embedded snapshot; long — use tmux)
python main.py --warm-replay-without-appendix --overwrite

# Regenerate MLP per-shard research scores (E.12 artifact) for one encoder
python 1_code/5_supervised_model_infer/score_supervised.py \
  --embed-model all-mpnet-base-v2 --classifier mlp --corpus research --overwrite

# Regenerate just the i1 appendix table (fingerprint-gated on the MLP shards)
python main.py --appendix-i1-assignment-method --overwrite

# Rebuild the PDF (bash/WSL/Linux only; may exceed 120s — use tmux)
python main.py --build-pdf --overwrite

# Status / verify
git status
git log --oneline -5
python3 -c "import json; d=json.load(open('4_outputs/appendix/mpnet/i1_assignment_method_comparison/data/assignment_method_comparison.json')); print('research MLP-vs-ZS:', d['research_mlp_vs_zs']['overall_agreement'])"
```
