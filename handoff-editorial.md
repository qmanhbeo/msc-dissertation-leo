# Handoff: Statistical-framing fix (permutation p-values) — editorial session

Status: **main work COMPLETE and pushed**; follow-up seed-hardening session also COMPLETE
and pushed (see §8). This file is a self-sufficient handoff for a fresh agent. Read
`AGENTS.md` first for repo conventions; the numbered-directory layout, engineering standards,
and gotchas there are assumed known.

---

## 1) Context: where we are

This is a UK MSc dissertation reproducibility repo ("Measuring the Gap: Semantic Alignment
Between AI-for-Sustainability Research and SDG Policy Frameworks"). The manuscript is
`3_writing/dissertation.tex`, compiled via `python main.py --build-pdf --overwrite` into
`4_outputs/dissertation.pdf`. The repo is on GitHub (`qmanhbeo/dissertation-bham`, branch
`main`), currently private for examination.

The dissertation is in an examiner-review feedback loop. A reviewer recently submitted a
comment (quoted in full below) flagging a **statistical framing tension**: §3.9 of the
manuscript claims the 17 SDGs are the complete, fully enumerated population of the SDG
framework (so "sampling-based power reasoning does not directly apply"), yet the paper reports
Pearson/Spearman p-values throughout (including the abstract headline ρ=0.544, p=0.024) and
even runs an 80%-power Fisher-z minimal-detectable-effect analysis. The reviewer said this is
internally inconsistent and would be caught by a stats-savvy examiner.

**Decision made (user-confirmed): keep "17 = population"** (it is factually correct — the 17
SDGs ARE the complete framework), and make the stochastic element the *measurement process*
(classifier assignment, corpus construction, embeddings), not SDG sampling. All correlation
p-values are re-derived as **two-sided Monte Carlo permutation p-values** (100,000 resamples,
fixed seed 42). This was implemented, tables regenerated for all three encoders, the
manuscript edited, PDF rebuilt, and pushed in two commits (`311d18c`, `6c5befc`).

The reviewer's comment, verbatim:

> The thing I'd flag hardest — statistical framing tension in 3.9: You argue the 17 SDGs are
> "the complete set of the framework," not a sample, so "sampling-based power reasoning does
> not directly apply" — but then you go on to report Pearson/Spearman correlations with
> p-values throughout the whole paper (including the abstract's headline ρ=0.544, p=0.024).
> That's internally inconsistent: p-values are inferential statements about a
> data-generating process/sampling distribution. If you're treating n=17 as a fixed,
> fully-enumerated population, you shouldn't be reporting p-values at all — you'd just report
> the population correlation as a descriptive fact with no inferential claim attached. If you
> are reporting p-values, you're implicitly treating each SDG's gap as a draw from some
> broader stochastic process (measurement noise, corpus sampling, etc.), which contradicts
> "population not sample." Examiners who know stats will catch this immediately. You need to
> pick one framing and be consistent, or explicitly justify why p-values are meaningful under
> a "complete population" framing (e.g., treating the classifier and corpus construction as
> the stochastic element, not the SDGs themselves) — I don't see that argument made yet.

## 2) Key known facts (so you don't have to re-derive them)

### The statistical issue
- **The 17 SDGs genuinely are the complete, enumerated SDG framework.** Claiming they are a
  sample from a super-population would be wrong. The population framing stays.
- **A permutation test is fully compatible with population framing**: it holds the observed
  values fixed, permutes one vector, and tests the null "no association between the *measured*
  quantities". It requires NO superpopulation/sampling assumption — which is exactly why it
  resolves the reviewer's tension. The scipy analytic t-approximations (previously used) are
  sampling-theory approximations of that same null and silently import the sampling language
  the manuscript rejects.
- **Before the fix, the manuscript contradicted ITSELF on the same page**: §3.9 (L303) said
  "sampling-based power reasoning does not directly apply", while the Results (L420) ran an
  80%-power Fisher-z MDE analysis (`\HPrimaryMinDetectableR = 0.63`, with a full MDE footnote)
  and the Discussion (L463) repeated "powered only to detect very large correlations". This
  was the reviewer's exact point.
- **The permutation p-values barely move the numbers.** On real MPNet data (seed 42, 100k
  draws): headline cov-vs-adjusted Spearman p 0.0239→0.0253 (still <0.05); cov-vs-register
  p 0.1220→0.1204 (still n.s.); cov-vs-raw p 0.9628→0.9593 (still null). No prose claim
  inverted; this is a framing/consistency fix with minimal numeric movement. **All rho values
  are byte-identical to before** (the helper computes the statistic exactly as scipy does).

### Pipeline / architecture facts
- Default embedding model: `all-mpnet-base-v2` (aliased `mpnet`); MiniLM (`minilm`) and
  SciBERT (`scibert`) are S1-subset-only robustness tracks. Main-text tables exist for all
  three under `4_outputs/{model}/tables/`; appendix analyses are **MPNet-only**
  (`4_outputs/appendix/mpnet/...`) — there are no committed minilm/scibert appendix outputs.
- All analysis scripts are Tier-B, fingerprint-gated (`shared_utils.should_skip` /
  `record_fingerprint`). Editing the script does NOT change input fingerprints, so
  **`--overwrite` is mandatory** when re-running after code changes (else they skip).
- Fingerprint files (`*.fingerprint.json`) are **mtime-sensitive by design**
  (`_file_fp`: size + mtime_ns + first 64KB). They refreshed during this session even though
  input *content* is frozen — harmless, by design (catches snapshot re-hydration). Commit them.
- `scipy 1.17.1`: `pearsonr` supports `method=PermutationMethod`, but **`spearmanr` does NOT**
  (no `method=` kwarg) — hence the custom shared helper.
- The 120s tool-timeout rule: long jobs MUST go in `tmux` (`tmux new-session -d -s <name>
  "<cmd> > log 2>&1; touch log.DONE"`, poll `ls log.DONE`). `a1_register_validation.py`
  takes ~15+ minutes (heavy sampling + 500 bootstraps; it ran ~13 min in this session at
  ~50-110% CPU) — always tmux it.
- **`h1_register_correlation_table.py` is RETIRED** (commit `763b446` removed it from the
  orchestrator and deleted its outputs). Its reader functions are still imported by
  `2_coverage_semantic_interaction.py` (gap loaders), but do NOT re-run its main and do NOT
  regenerate its outputs. (This session initially ran it and committed nothing; outputs were
  removed.)

### The p-value architecture after the fix
- **One shared helper**: `permutation_p(x, y, kind="spearman"|"pearson", n_resamples=100_000,
  seed=42)` in `1_code/7_main_analysis/0_shared/shared_utils.py` (lazy numpy/scipy imports).
  Returns `(stat, p)` with `p = (count+1)/(n_resamples+1)`; vectorized via
  `np.argsort(rng.random(...))` rank/value permutations; observed stat via scipy (unchanged
  numbers); deterministic for a given numpy version (PCG64 stream).
- Constants `PERMUTATION_N_RESAMPLES = 100_000`, `PERMUTATION_SEED = 42` in shared_utils.
- **Provenance is recorded in outputs**: every affected JSON now carries a
  `"p_value": {"method": "monte_carlo_permutation", "n_resamples": 100000, "seed": 42}` block,
  and every affected generated `.tex` carries a header comment
  `% p-values: two-sided Monte Carlo permutation (100,000 resamples, seed 42)`.
- Manuscript-facing correlation p-values now ALL come from `permutation_p`. The only p-values
  still analytic are: (a) pooled-OLS regression appendix (`k1_regression_semantic_gap.py`,
  L943: p=0.069/0.040/0.003/0.007/0.003) — deliberately left (different inferential
  framework, config×SDG panel, not flagged by reviewer); (b) binomial tests + Wilson CIs in
  `a1_register_validation` (accuracy tables) — different inference, left.

### Current headline values (post-fix, MPNet)
- `\RhoCovTopic = 0.544`, `\RhoCovTopicP = 0.025` (was 0.024) — abstract "only the adjusted
  association reaches significance" still true.
- `\RhoCovRegister = -0.390`, `\RhoCovRegisterP = 0.120` (was 0.122) — n.s.
- `\SpearmanCovRaw = -0.012`, `\SpearmanCovRawP = 0.959` (was 0.963) — null.
- MiniLM num5: 0.115 / 0.063 / 0.346. SciBERT num5: 0.392 / 0.296 / 0.613.
- RegVal (a1 appendix): only 3 values moved at 3 decimals (DrawFortyFiveP 0.072→0.070,
  MegaExclDrawOneP 0.091→0.090, MegaExclDrawTwoP 0.053→0.054); `\RegValPerSdgSig = 0`;
  hardcoded "n.s." annotations all still true.

### Two incidental findings this session (both resolved/handled)
1. **Stale `\ConceptLRCovgapAdjRho` in minilm/scibert num4 files**: committed HEAD had
   0.439 in minilm/scibert but 0.434 in mpnet. The Concept grid rows read MPNet-only concept
   data (hardcoded `output_dir_for_model("all-mpnet-base-v2", ...)`), so all three must
   agree; minilm/scibert were stale from before the Concept-row fix (commit `7cdbb8d`
   regenerated only mpnet). The session's rerun propagated the correct 0.434 to all three.
   Prose uses the macro, so it auto-updated.
2. **Stale-macro bug at Results L430**: prose cited `\HExclFourResearchSpearmanRho/P`
   (0.606, p=0.015 — significant!) while claiming the correlation was "null". At the commit
   where the sentence was written (`6e7f44d`) the research cell was 0.341/p=0.196 (null); a
   later regeneration changed the number without updating prose. The sentence's own logic
   ("the raw null" = the H1a coverage-gap null) requires the covgap macros. **Fixed**: now
   cites `\HExclFourCovgapSpearmanRho/P` (0.088, p=0.744 — null ✓) and says "coverage-gap".
   Note: the ex-SDG4 *research*-share raw correlation genuinely flipped to significant
   (0.606, p=0.015) — nobody decided yet whether that deserves a prose mention anywhere; it
   is currently not mentioned.

## 3) Actions / decisions made + files changed this session + why

### Decision: Option 1 (permutation p-values), user-confirmed
Three options were considered and the user chose the full coherent package:
- Option 1 (CHOSEN): keep 17=population; reframe §3.9; swap analytic p → exact/permutation p
  in the pipeline; delete the Fisher-z power/MDE passages.
- Option 2 (rejected): keep analytic p + verification footnote — structurally weaker, a
  determined examiner can still ask "why report the sampling-theory number at all?".
- Option 3 (rejected): text-only — papers over the contradiction; the pipeline would still
  run the power code and t-approximations.

### Commit 1 — `311d18c` "analysis: use Monte Carlo permutation p-values for SDG-level correlations" (46 files)
Code (7 files):
- `1_code/7_main_analysis/0_shared/shared_utils.py`: added `PERMUTATION_N_RESAMPLES`,
  `PERMUTATION_SEED`, `permutation_p()` (vectorized MC permutation, docstring explains the
  population-framing rationale).
- `0_shared/g_interaction_extended.py`: `pearson_spearman()` now uses `permutation_p` for
  both Pearson and Spearman p; output JSON gets `"p_value"` provenance block.
- `0_shared/g_register_decomposition.py`: headline `cov vs adjusted` and `cov vs register`
  Spearman p via `permutation_p`; JSON provenance block.
- `1_main_text/2_coverage_semantic_interaction.py`: `pearson_and_spearman()` and
  `_spearman_dict()` use `permutation_p`; **deleted the Fisher-z 80%-power MDE computation
  and the `\HPrimaryMinDetectableR` macro emission** (coordinated with manuscript edit);
  num4 tex header now carries the permutation provenance comment.
- `2_appendix/a1_register_validation.py`: `spearman()` and `partial_spearman()` return
  `permutation_p` results; removed now-unused `spearmanr` import; num_a1 header comment.
- `2_appendix/a2_policy_source_family_sensitivity.py`: `_pearson_and_spearman()` via
  `permutation_p`; num_a2 header comment.
- `2_appendix/j1_raw_value_correlation.py`: `_pearson_dict()` via `permutation_p`;
  tab_j1 header comment; JSON provenance block.

Regenerated outputs (all three encoders + MPNet-only appendix):
- `4_outputs/{mpnet,minilm,scibert}/data/{interaction_extended,register_decomposition,
  interaction_h25}.json` (+ fingerprint files), `tables/{num4,tab4,num5}.tex`.
- `4_outputs/appendix/mpnet/a1_register_validation/{data/register_validation.json,
  tables/num_a1_register_validation.tex}`, `a2_source_family_sensitivity/...`,
  `j1_raw_value_correlation/...`.

NOT changed (deliberately): `h1_register_correlation_table.py` (retired script — my initial
edit was **reverted**, and the untracked outputs it regenerated were **deleted**),
`c1_subset_balanced_stability.py` (rho only, no p), `g_distributional_gap.py` (p stored in
JSON but never emitted to tex), `h1_cross_method_gap_values.py` (no correlation p),
`k1_regression_semantic_gap.py` (regression framework), binomial/Wilson-CI code in a1.

### Commit 2 — `6c5befc` "writing: align statistical framing with permutation inference in §3.9" (2 files)
`3_writing/dissertation.tex`:
1. **§3.9 (sec:coverage-semantic-interaction, ~L303)**: kept the population sentence; replaced
   "sampling-based power reasoning does not directly apply" with: the only stochastic element
   is the measurement process (classifier assignment, corpus construction, embeddings); all
   reported p-values are two-sided Monte Carlo permutation p-values (100,000 resamples, fixed
   seed) conditioning on the measured values, so no sampling-based power reasoning is
   involved. Citations `\parencite{BerkWesternWeiss1995, AbadieAtheyImbensWooldridge2020}`
   retained.
2. **Results (~L420)**: deleted "With only 17 SDG-level observations the design can reliably
   detect only very large correlations (|ρ| ≥ \HPrimaryMinDetectableR at 80% power, α=0.05)"
   AND its entire Fisher-z MDE footnote. Kept "this is evidence against a strong monotonic
   relationship, not proof of independence".
3. **Discussion (~L463)**: deleted "with only n=\HPrimaryN SDG-level observations the design
   is powered only to detect very large correlations (...)"; reflowed to "the null result is
   consistent with either no effect or a moderate one and is evidence against a strong
   monotonic relationship, not proof of exact independence (the complete SDG set is a fully
   enumerated population...)".
4. **Appendix J.1 (~L886)**: item (A) of "Possible causes of the SciBERT divergence"
   rewritten without MDE/power: "At n=17 the Pearson estimates are noisy and the permutation
   null is only rejected for large magnitudes: both SciBERT adjusted-gap values are small in
   absolute terms and do not reach significance, so the rank/magnitude sign flip is within
   estimation error."
5. **Table note L905** (tab:raw-value-correlation): "significance stars use the two-sided
   Monte Carlo permutation $p$-value".
6. **Table note L935** (tab:policy-source-family-h25): "Spearman ρ (with two-sided Monte
   Carlo permutation p, 100,000 resamples, fixed seed)".
7. **Results L430** (the incidental stale-macro fix, see §2): Research macros →
   `\HExclFourCovgapSpearmanRho/P`; "research--gap" → "coverage-gap".
- `4_outputs/dissertation.pdf` rebuilt (exit 0, no undefined refs/citations).

### Verification performed this session
- Helper unit-tested: headline p reproduces (0.0253/0.1204/0.9593), deterministic across
  calls, timing 0.03s @n=17, ~2s @n=204 (n=204 1.9s spearman / 0.41s pearson), 0.03s @n=12.
- All 7 edited scripts `py_compile` clean; remaining raw `spearmanr/pearsonr` calls verified
  to discard p (`_`).
- Regenerated-table diff audit: every rho identical; only p-macros changed (plus the two
  star-level moves in tab_j1: MiniLM MLP H1d adj †→*, SciBERT LR H1d adj ***→** — neither
  cell is quoted in prose) plus the Concept-row correction (§2 finding 1).
- Prose-truth audit of every p-value cite in the manuscript (abstract L93, intro L114,
  results L420-422, L430, L432, discussion L463, appendix L705/L760/L762/L886):
  - L420's H1d star claims verified: MPNet LR +0.589*, Concept LR +0.510*, SciBERT LR
    +0.605* (p<0.05), MPNet MLP +0.428† (p<0.10). "positive in 8/9 configs" macro correct.
  - L421 "would not survive a correction" still true (0.025 > 0.0167).
  - L760/L762 register-validation claims all hold: TwoBOrigPooled p=0.040 (<0.05);
    hardcoded n.s. annotations verified from JSON: opp pooled 0.063, within_res 0.541,
    within_pol 0.604; draw p's 0.063/0.949/0.070 all n.s.; MegaExclDrawOne 0.090 n.s.,
    DrawTwo 0.054 n.s., DrawThree 0.003 significant; PerSdgSig=0.
- PDF rebuild exit 0; grep of `3_writing/artifact/dissertation.log` for undefined control
  sequences/undefined refs → none; pdftotext sanity: new p-values rendered (0.025/0.120/
  0.959), zero "80% power"/"minimal detectable"/MDE residue in the compiled PDF.

### Regeneration commands used (for future re-runs, in dependency order)
```
python 1_code/7_main_analysis/0_shared/g_register_decomposition.py --embed-model <M> --output-dir 4_outputs --overwrite
python 1_code/7_main_analysis/0_shared/g_interaction_extended.py --embed-model <M> --output-dir 4_outputs --overwrite
python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py --embed-model <M> --output-dir 4_outputs --overwrite
python 1_code/7_main_analysis/0_shared/generate_tex_macros.py --embed-model <M> --output-dir 4_outputs --overwrite   # AFTER the two g_* scripts
```
for `<M>` in mpnet, minilm, scibert; then MPNet-only:
```
python 1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py --embed-model mpnet --output-dir 4_outputs --overwrite
python 1_code/7_main_analysis/2_appendix/j1_raw_value_correlation.py --embed-model mpnet --output-dir 4_outputs --overwrite
# a1 in tmux (~15 min):
tmux new-session -d -s a1 "python 1_code/7_main_analysis/2_appendix/a1_register_validation.py --embed-model mpnet --output-dir 4_outputs --overwrite > /tmp/a1.log 2>&1; touch /tmp/a1.DONE"
```
Then `python main.py --build-pdf --overwrite`.

## 4) What remains + why

1. **Reply to the reviewer** (not yet drafted — the fix is done; the user has not asked for a
   reply text yet). Suggested content: 17 SDGs are the complete framework (population claim
   kept); the stochastic element is now explicitly the measurement process; all correlation
   p-values are now two-sided Monte Carlo permutation p-values (100,000 resamples, fixed
   seed) which condition on the observed values and need no superpopulation assumption; the
   sampling-based power/MDE passages were removed; numbers move negligibly (headline
   p=0.024→0.025) so no substantive claim changes.
2. **Decide on the §3.9 "All reported $p$-values" overbreadth** (see §5 concern A): either
   scope the sentence to correlation tests or leave as-is. Small edit, needs a judgment call.
3. **Fix the stale "7/9" count in Appendix J.1 prose** (see §5 concern B): `dissertation.tex`
   L876 says "positive in 7/9 and 9/9 configs respectively under Pearson"; the current
   tab_j1 table shows H1a adjusted is positive in **8/9** (only SciBERT MLP -0.189
   negative). Pre-existing (rho signs did not change this session), found during the audit,
   NOT fixed because it is a sign-count claim outside the p-value scope. Fix is a one-word
   edit ("7/9" → "8/9") after confirming intent (the H1d "9/9" is correct).
4. **Optional: one-line AGENTS.md addition** recording the convention "all correlation
   p-values are two-sided Monte Carlo permutation (seed 42, 100k resamples) — the 17 SDGs
   are a fully enumerated population; analytic t-approximations and power/MDE reasoning are
   out of scope". This is a non-obvious convention exactly of the kind AGENTS.md exists for.
5. **Optional: retire `\HPrimaryN`-adjacent dead macros** — `HPrimary*CiLower/CiUpper`
   (Fisher-z CIs) are still emitted by `2_coverage_semantic_interaction.py` into num4 but are
   NOT used anywhere in the manuscript (verified by grep). Left untouched this session to
   minimise diff. They are inert; a purist could argue they import sampling language into
   outputs. Consider deleting their emission.
6. **Optional: decide on the ex-SDG4 research-share flip** (0.606, p=0.015, significant when
   SDG4 is removed — currently unmentioned in prose; see §2 finding 2 tail).
7. **Not required**: no further regeneration needed; nothing about this fix is stale or
   incomplete. Repo is clean and pushed.

## 5) Concerns to emphasize

- **A. "All reported $p$-values..." overbreadth.** The new §3.9 sentence says "All reported
  $p$-values are two-sided Monte Carlo permutation $p$-values". Strictly, the pooled-OLS
  regression appendix (L943) still reports analytic regression p-values (0.069, 0.040, 0.003,
  0.007, 0.003), and the a1 accuracy tables report binomial p-values. An extremely picky
  examiner could read "all" as global. Mitigation options: (a) scope to "All correlation
  $p$-values reported in this dissertation..." — regression/binomial are different inference
  types, so the sentence remains true and precise; or (b) leave as-is and argue context makes
  it obviously about the correlation tests. Recommendation: (a).
- **B. Stale "7/9" count in J.1 prose (L876)** — the current Pearson table shows 8/9 positive
  for H1a adjusted. It predates this session (signs unchanged) but is the exact kind of
  numbers-vs-prose drift an examiner could catch. Fix it.
- **C. Monte Carlo noise on borderline p-values.** The permutation p is an MC estimate with
  standard error ≈ sqrt(p(1-p)/100000) ≈ 0.0015 for p≈0.05. Borderline values could cross a
  threshold under a different seed/numpy version: `\RegValMegaExclDrawTwoP = 0.054` backs the
  hardcoded "n.s." at L762 (a different seed could give p<0.05); `\HExclFourCovgapAdjSpearmanP
  = 0.016` and `\HExclFourPolicyAdjPearsonP = 0.048` are near 0.05. If any of these flip in a
  future re-run, the prose annotations must be re-audited. (Seed 42/100k are fixed and
  recorded, so the current outputs are reproducible as committed.)
- **D. Determinism across numpy versions**: the helper uses `np.random.default_rng` (PCG64);
  PCG64 streams are stable in practice, but if the environment's numpy is upgraded, a re-run
  could shift p at the 4th decimal. Recorded seed + resamples make any such shift detectable
  and explainable. `environment.yml` is the real build path (Python 3.11).
- **E. Do NOT re-run `h1_register_correlation_table.py`** — retired script; its outputs are
  not committed and should stay gone. Only its imported reader functions are live.
- **F. `a1_register_validation.py` is slow (~15 min)** — tmux it; it prints little until the
  end (final self-check gates "[PASS] ..."). Wait for `log.DONE`.
- **G. The L500 "underpowered" sentence was deliberately KEPT** — it refers to the 94-document
  curated policy family (a corpus-size/precision statement), NOT SDG-sampling power
  reasoning. Do not "fix" it; it does not contradict the new framing.
- **H. Fingerprint files changed in commit 1** — expected (mtime-sensitive by design), not a
  sign of input drift.

## 6) The whole comprehensive plan (as executed, and as the future baseline)

### 6.1 The plan (approved, executed 2026-08-06)
The plan document `.opencode/plans/permutation-pvalues-framing.md` was the working plan; it
was followed except where noted below. Its content, in full:

**Goal**: resolve the reviewer's statistical-framing inconsistency by (1) keeping the
"17 SDGs = fully enumerated population" claim, (2) making the stochastic element the
measurement process, (3) replacing analytic correlation p-values with two-sided Monte Carlo
permutation p-values (seed 42, 100k resamples) in every manuscript-facing correlation site,
(4) deleting all sampling-based power/MDE reasoning from manuscript and pipeline, (5)
regenerating all tables for all three encoders and the MPNet-only appendix, (6) rebuilding
the PDF, (7) committing code+tables and manuscript separately.

**Steps**:
1. Add `permutation_p` + constants to `shared_utils.py` (lazy imports; vectorized
   rank/value permutation; observed stat via scipy so rho unchanged; `p=(count+1)/
   (n_resamples+1)`).
2. Convert the 7 script sites (g_interaction_extended `pearson_spearman`,
   g_register_decomposition L147-148, 2_coverage_semantic_interaction
   `pearson_and_spearman`+`_spearman_dict`, a2 `_pearson_and_spearman`, j1 `_pearson_dict`,
   a1 `spearman`+`partial_spearman`); add provenance blocks/comments to outputs.
3. Delete Fisher-z MDE block + `\HPrimaryMinDetectableR` emission from
   2_coverage_semantic_interaction.py (keep `_fisher_ci`/CiLower/CiUpper emission — inert).
4. Regenerate with `--overwrite` in dependency order for mpnet/minilm/scibert (+ MPNet-only
   a2/j1/a1), then `--build-pdf`.
5. Manuscript edits: §3.9 reframe; delete MDE passages (L420+footnote, L463, L886-A);
   update table notes (L905, L935).
6. Verify: rho unchanged / p changed only; headline p matches; `\HPrimaryMinDetectableR`
   fully retired; prose-truth audit; PDF exit 0, no undefined refs; provenance visible.
7. Two commits + push.

**Deviations from the plan doc (all deliberate, all logged)**:
- `h1_register_correlation_table.py` was in the plan's script list, but commit `763b446` had
  retired it; the edit was reverted and its regenerated outputs deleted (see §2, §5-E).
- The plan's provenance for h1_register_correlation_table (JSON block) was therefore not
  applied; provenance went to the six live scripts instead.
- The plan's manuscript step gained a 7th edit: the L430 stale-macro fix (see §2 finding 2).
- The plan's verification step surfaced the J.1 "7/9" staleness (see §4 item 3 / §5-B) —
  recorded but not fixed, being outside scope.
- The audit confirmed the a1 hardcoded "n.s." annotations remain true post-conversion.

### 6.2 Future baseline (what a fresh agent should treat as canonical)
- Correlation inference = permutation (seed 42, 100k); rho/r via scipy (unchanged);
  provenance in every output JSON + tex header.
- No power/MDE language anywhere except the deliberate L500 corpus-size sentence.
- `\HPrimaryMinDetectableR` does not exist anymore (macro + emissions deleted); do not
  reintroduce.
- The p-value values in the committed tables are reproducible as-is; a full re-run needs the
  regeneration commands in §3 and a fresh prose audit (§5-C borderline cells).

## 7) Interrupted-work details

**Nothing was interrupted mid-task.** This session's assigned work (Option 1 implementation:
helper, six-script conversion, MDE removal, regeneration for all encoders, MPNet-only
appendix incl. the ~15-min a1 run, manuscript edits, PDF rebuild, verification, two commits,
push) was **completed and pushed** before the user asked to stop. The git HEAD is `6c5befc`,
the tree is clean, and the PDF is current.

The exact point of interruption: immediately after the push, the assistant's final summary
mentioned the optional AGENTS.md follow-up line. The next actions in queue were the items in
§4 (reviewer reply draft, the §3.9 "all p-values" scoping decision, the "7/9" fix, the
optional AGENTS.md line) — none of which were started.

For completeness, the session's earlier stopping point (same day): the first implementation
attempt was blocked by plan-mode read-only permissions; the comprehensive plan was written to
`.opencode/plans/permutation-pvalues-framing.md` instead, and the user then approved
execution ("implement the plan").

## 8) Follow-up session (same day): seed-hardening audit + fixes (commit `76b27ae`)

**Task**: "ensure in the new logic, anything random must be seeded" — audit every random
draw in the permutation p-value machinery and the pipeline, fix any gaps.

### Audit result (the new logic was already fully seeded)
- `permutation_p` (`shared_utils.py:352,363`) uses `np.random.default_rng(seed)` with
  `seed=PERMUTATION_SEED=42` default; **all 6 call sites** (`g_interaction_extended`,
  `g_register_decomposition`, `2_coverage_semantic_interaction`, `a1` `spearman`/
  `partial_spearman`, `a2`, `j1`) pass no seed → default 42. Determinism verified
  empirically (two identical calls → identical result).
- `a1` module `_rng = default_rng(SEED)`, `boot_diff_pvals` `default_rng(SEED+100)`,
  draw-instability fresh seeds 43/44/45, `StratifiedKFold`/`LogisticRegression`
  `random_state=` — all fixed. `a2:147` cap sampling `default_rng(RANDOM_SEED + sdg_idx)` —
  fixed.
- Whole pipeline: no bare `default_rng()` calls anywhere; every other script
  (`0_pca_*`, `1_semantic_gap`, `g_distributional_gap`, `register_adjust`, `k1`,
  `c_sample_stability`, `a3`, `b2`, `score_zeroshot`, `2_sample_segments`,
  `0_prepare_data`, torch training) is seeded via `default_rng(seed)` /
  `random.Random(seed)` / `random_state=` / `torch.manual_seed` + deterministic algorithms.
  The **only unseeded random in the repo** is `fetch_aurora.py:300`
  `random.uniform(0, 0.02)` — fetch-stage rate-limit sleep jitter, result-neutral, left as-is.

### Fixes made (user chose Items 1+2; Item 3 JSON-provenance parity was DECLINED)
1. `g_register_decomposition.py:247`: magic literal `np.random.default_rng(42)` →
   `np.random.default_rng(PERMUTATION_SEED)` (named constant, value-preserving). Added
   `% seed: 42 (PERMUTATION_SEED) — policy per-document cap sampling` provenance comment to
   the `tab12_register_check.tex` and `num12_register_check.tex` headers (the iterative
   diagnostic's seed was previously unrecorded; it feeds the `RegisterIter*`/`RegIterGap*`
   macros used by the manuscript).
2. `shared_utils.py` `permutation_p`: fail-closed guard raising `ValueError` when
   `seed is None` or non-int, so a future caller can never silently get an entropy-seeded
   `default_rng(None)`.
3. NOT done (declined by user): `p_value` JSON provenance blocks for `interaction_h25.json`,
   `a2` h25.json, `a1` `register_validation.json` (only their tex headers carry the note).
   `fetch_aurora` jitter left unseeded (deliberate throttle, result-neutral).

### Regeneration + verification
- `g_register_decomposition.py --embed-model {mpnet,minilm,scibert} --overwrite` re-run
  (scibert exceeded the 120s tool timeout → ran via tmux, completed: 79 iterations,
  rho=0.6324). Diff vs committed: **only** the two seed-comment lines added to
  tab12/num12 per model; `register_decomposition.json`, num5, tab5 byte-identical
  (fingerprint JSONs changed — expected mtime refresh).
- `py_compile` clean on both edited files; guard rejects `seed=None` and non-int seeds.
- Committed as `76b27ae` (code + regenerated outputs) with this handoff update; pushed;
  working tree clean.

### Remaining items (unchanged from §4)
Still open: reviewer reply draft (§4.1), §3.9 "all reported p-values" scoping (§4.2/§5-A),
J.1 "7/9"→"8/9" prose fix (§4.3/§5-B), optional AGENTS.md convention line (§4.4), optional
dead `CiLower/CiUpper` macro retirement (§4.5), ex-SDG4 research-share flip decision (§4.6).
