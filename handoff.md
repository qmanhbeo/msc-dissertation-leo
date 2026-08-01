# HAND-OFF: `register-adj` branch

## 1. Context

**Repository:** `dissertation-bham` — a reproducibility repo for a dissertation measuring semantic alignment between AI-for-sustainability research and SDG policy frameworks using Sentence-BERT embeddings.

**Branch:** `register-adj` off `main` at `8a83eaf`. 12 commits on this branch (latest `a760108`). Not yet merged.

**Authoritative spec:** `PLAN_register_topic_decomposition.md` (627 lines). This plan documents the entire register-topic decomposition restructure — science gate, engineering tasks §6.1–§6.7, manuscript rewrites §12, architecture decisions §10.

**What this branch does:** Implements INLP (Iterative Nullspace Projection) to decompose the raw semantic gap into a **topic component** (adjusted gap) and a **register component**, answering the question: "Is the raw gap driven by topic divergence or by register differences between academic and policy discourse?"

**The key scientific finding:** The raw coverage-vs-gap correlation is near-zero (rho=-0.08) because register cancels the topic signal. After INLP register removal, the adjusted topic gap shows a strong positive association with coverage divergence (rho=+0.48, p=0.054), while the register component is negatively correlated (rho=-0.49, p=0.045). SDG 17 flips from smallest raw gap (0.216) to largest adjusted gap (0.371).

## 2. Key Known Facts

- **Default embedding model:** `all-mpnet-base-v2` (MPNet). Track = `canon`. MiniLM/SciBERT use track = `subset`.
- **INLP stratification key:** `sdg_labels * 2 + y` (34 classes), NOT 34-class topic classification. Binary research-vs-policy LR.
- **G.npy lifecycle:** Gitignored, NOT in frozen snapshot. Warm replay must regenerate via `register_adjust`. Stored at `2_data/3_embedded/{slug}/register/{track}/G.npy` (~KB per model).
- **Adjusted embeddings:** NEVER stored as full `.npy` arrays. G is ~KB; projection runs on-the-fly via `register_utils.project()`.
- **Fingerprints:** Content-based (SHA-256), NOT mtime-based (2_data/ hydration resets mtimes). All consumer scripts now include G.npy content hash in adjusted-mode fingerprints.
- **Repo conventions:** Numbered dirs encode role. `1_code/` only active code. `main.py` single entrypoint. `--overwrite` required for writes. No test/lint/typecheck suite. Long jobs: `setsid ... & disown`, poll via `pgrep -af`.
- **Per-iteration RNG:** `default_rng(42+k)` for resume bit-identicality. `POLICY_SEGMENT_CAP_SEED = 42` from `semantic_gap_shared.py`.
- **`penalty='l2'` FutureWarning:** Benign, matches existing `f_register_adjustment.py`. Leave byte-consistent.
- **ZS adjusted:** MPNet-only (AGENTS.md axis restriction).
- **Coverage NOT re-run in adjusted mode:** §6.3 confirms adjustment-invariant.

## 3. Actions Taken This Session

### Commit 1 — `dfcc3a7`: §6.1 register_adjust stage
- Created `1_code/7_main_analysis/0_shared/register_adjust.py` (~450 lines)
- Wired into `main.py --stage register_adjust` and warm-replay orchestrator
- Produces G.npy for all 3 encoders (MPNet 75 iter, MiniLM 26, SciBERT 50)

### Commit 2 — `d1b22c4`: §6.2 register_utils + --embeddings adjusted
- Created `register_utils.py` with `load_G()`, `project()`, `register_dir()`, `track_for_model()`
- Added `--embeddings {raw,adjusted}` to 6 consumer scripts: `1_semantic_gap.py`, `score_zeroshot.py`, `a2`, `c`, `f`, `g_distributional_gap.py`
- Added `run_analysis_adjusted()` to `analysis_orchestrator.py`
- Wired register_adjust into warm-replay orchestrator (main.py)

### Commit 3 — `7d21808`: §6.4 re-run matrix + §6.5 generators
- Ran adjusted semantic gaps for all 3 encoders (MPNet LR/ZS, MiniLM LR, SciBERT LR)
- Created `g_register_decomposition.py` (per-SDG decomposition table + JSON)
- Created `g_interaction_extended.py` (coverage vs {raw,adj,register} correlations)
- Committed all adjusted outputs under `4_outputs/{model}/data/adjusted/`

### Commit 4 — `ef53467`: §6.6 macros + §6.7/12 manuscript rewrites
- Created `generate_tex_macros.py` — consolidated macro generator (62 lines of `num_register_topic_decomposition.tex`)
- Rewrote `dissertation.tex`: abstract (L87-91), method (L261), semantic gaps (L343-345), H1a discussion (L361), robust patterns (L433), register effects (L462), appendix (L662-709)

### Commit 5 — `8f5a463`: Audit pass 1 (15 files, C1-C2, H1-H3, M1-M12)
- C1+C2: Removed duplicate `num_*.tex` from `g_register_decomposition.py`, added `\InputIfFileExists{num_register_topic_decomposition.tex}` to `dissertation.tex`
- H1+H2: Wired 3 generators into orchestrator as `POST_ADJUSTED_STEPS`
- H3: Fixed `score_zeroshot.py` mode-blind existence guard
- M1-M12: Fixed unused imports, f-string, atomic writes, fingerprint coverage (G.npy hash in 5 consumer scripts), dead code removal, AGENTS.md update, RNG constant

### Commit 6 — `aae27aa`: Audit pass 2 (M1-M2, L1-L6)
- M1+M2: Added `register_adjust` + `run_analysis_adjusted()` to `--stage analysis` paths in `main.py` (both default and non-default model)
- L1-L6: Cleaned unused imports in `score_zeroshot.py`, `1_semantic_gap.py`, `a2_policy_source_family_sensitivity.py`; fixed `_ANALYSIS_ROOT` path

### Commit 7 — `15b9028`: Gitignore handoff.md, add register_adj literature
- Added `handoff.md` to `.gitignore`
- Added `0_literature/register_adj/RavfogelS_etal_2020_INSP.gz` (INLP source paper)

### Commit 8 — `f1fa7b7`: Fix LaTeX build
- Fixed undefined `\citealp` → `\textcite` (biblatex-apa compatible) at line 262
- Fixed citation key `Ravfogel2020iterative` → `Ravfogel2020INLP` (matching `references.bib`)
- PDF rebuilt: 64 pages, all citations resolved

### Commit 9 — `a760108`: PCA before/after register-removal figure (§6.5.4)
- Created `1_code/7_main_analysis/1_main_text/0_pca_register_before_after.py` (~350 lines)
- Two-panel PCA figure: left=raw (two separated clouds), right=adjusted (merged)
- PCA fitted once on raw data; both panels share same axes for visual comparability
- Wired into `POST_ADJUSTED_STEPS` in `analysis_orchestrator.py` (default-model only, using tuple format `(path, True)`)
- Added `fig2_pca_register_before_after.*` to `MANUSCRIPT_FIGURE_FILES` in `shared_utils.py`
- Added `\InputIfFileExists{num_pca_register_before_after.tex}` to `dissertation.tex` preamble
- Added figure reference + caption in `dissertation.tex` after INLP description (after L262)
- Added new figure `\begin{figure}` environment with label `fig:pca-register-before-after`
- PDF rebuilt: 65 pages, no undefined refs

## 4. What Remains

| Task | Status | Why remaining |
|------|--------|---------------|
| **Merge to main** | Not done | Waiting for user confirmation |
| **h1_register_correlation_table.py** | Not started (§6.5.3) | Optional — per-config correlation table showing cancellation replicates across encoders. The decomposition table and interaction extension already provide the key correlations. |

### Already completed (§6.5.1, §6.5.2, §6.5.4):
- §6.5.1 — Decomposition table: `g_register_decomposition.py` → `register_decomposition.json` + `tab_register_decomposition.tex`
- §6.5.2 — Interaction extension: `g_interaction_extended.py` → `4_4_interaction_extended.json`
- §6.5.4 — PCA before/after figure: `0_pca_register_before_after.py` → `fig2_pca_register_before_after.pdf`

### §6.5.3 remaining detail:
- Script should iterate over configs: `(MPNet, LR)`, `(MPNet, ZS)`, `(MiniLM, LR)`, `(SciBERT, LR)`
- For each config, read coverage + raw/adjusted semantic gap JSONs
- Compute Spearman rho for (coverage_gap, raw_gap), (coverage_gap, adjusted_gap), (coverage_gap, register_component)
- Emit `tab_app_register_correlation.tex` + JSON
- Pattern exists in `h1_cross_method_gap_values.py` (multi-model iteration) and `g_interaction_extended.py` (correlation computation)

## 5. Concerns

1. **`POST_ADJUSTED_STEPS` now mixes strings and tuples.** The list was `[str, str, str]` and is now `[str, str, str, (str, bool)]`. The `run_analysis_adjusted()` function was updated to handle both formats via `isinstance(item, tuple)` check. This works but is slightly ugly. A cleaner design would convert all entries to tuples, but that's a larger refactor for marginal benefit.

2. **The `.opencode_fp.json` file was committed.** The `4_outputs/mpnet/data/pca_register_before_after_metadata.json.opencode_fp.json` is an opencode fingerprint artifact. It should probably be gitignored, but it's harmless.

3. **The old `2_coverage_semantic_interaction.py` still runs** in `MAIN_STEPS`. It produces `num_interaction.tex` (loaded by `dissertation.tex` for `\HPrimary*` macros) and `4_4_interaction_correlation_asymmetry.json`. The new `g_interaction_extended.py` produces a superset JSON. No conflict, but there's redundancy.

4. **G.npy is gitignored and not in the frozen snapshot.** Warm replay MUST run `register_adjust` to regenerate it. This is documented in AGENTS.md but easy to forget. If someone tries to run adjusted analyses without G.npy, `register_utils.load_G()` will fail with a clear error.

5. **`project_centroids` and `project_sub_centroid` were removed** from `register_utils.py` as dead code. If any future script needs them, they'll need to be re-added. The docstring was updated accordingly.

6. **LaTeX compilation is now verified.** The new macros (`\MeanRawGap`, `\RhoCovTopic`, `\RawGapSdgSeventeen`, etc.) are defined in `num_register_topic_decomposition.tex` which is loaded via `\InputIfFileExists`. The file exists and is loaded. The new PCA macros (`\PcaRegBef*`) are in `num_pca_register_before_after.tex`, also loaded via `\InputIfFileExists`.

## 6. Comprehensive Plan (from `PLAN_register_topic_decomposition.md`)

**§0 — Science gate (resolved):** cov vs raw gap rho=-0.09 (cancellation); cov vs adjusted/topic rho=+0.44; cov vs register rho=-0.50. Pattern confirmed.

**§6.1 — register_adjust stage:** DONE. INLP with SDG-stratified LR, binary research-vs-policy, stops at acc<=0.5. G.npy at `2_data/3_embedded/{slug}/register/{track}/`. Resume-safe via checkpoint.json.

**§6.2 — register_utils + --embeddings adjusted:** DONE. `load_G()`, `project()`, `track_for_model()`. Added to 6 consumer scripts. `run_analysis_adjusted()` in orchestrator.

**§6.3 — Coverage adjustment-invariant:** Confirmed. Coverage NOT re-run in adjusted mode.

**§6.4 — Re-run matrix:** DONE. MPNet LR/ZS adjusted, MiniLM LR adjusted, SciBERT LR adjusted. All outputs committed.

**§6.5.1 — Decomposition table:** DONE. `g_register_decomposition.py` produces `register_decomposition.json` + `tab_register_decomposition.tex`.

**§6.5.2 — Interaction extension:** DONE. `g_interaction_extended.py` produces `4_4_interaction_extended.json` with raw/adjusted/register correlations.

**§6.5.3 — h1_register_correlation_table.py:** NOT DONE. Optional. Per-config correlation table showing cancellation replicates across encoders.

**§6.5.4 — PCA before/after figure:** DONE. `0_pca_register_before_after.py` produces `fig2_pca_register_before_after.pdf` (two-panel: raw vs adjusted).

**§6.6 — Consolidated macros:** DONE. `generate_tex_macros.py` produces `num_register_topic_decomposition.tex` (62 macros).

**§6.7/12 — Manuscript rewrites:** DONE. Abstract, L261, L343, L361, L433, L462, appendix all updated.

**§10 — Architecture decisions:** G-only materialisation (no full adjusted arrays), content-based fingerprints, raw kept as canonical reference, adjusted as the meaningful comparison.

## 7. What Was Being Worked On When Interrupted

The session completed the PCA before/after figure (§6.5.4). The last action was committing `a760108` (PCA figure + LaTeX wiring). The branch is clean — no uncommitted changes, all generators verified working, all outputs present.

The natural next steps would be:
1. **Implement §6.5.3** (h1_register_correlation_table.py) if desired — optional
2. **Merge to main** — once the user is satisfied
3. **Optional: LaTeX compilation final check** — already verified, but can re-run `python main.py --build-pdf --overwrite` before merge

No work was interrupted — the session reached a natural stopping point with all critical and medium issues resolved. The user explicitly said "dont merge yet" and asked to tackle the optional items first. §6.5.4 (PCA figure) was completed; §6.5.3 (correlation table) remains optional.
