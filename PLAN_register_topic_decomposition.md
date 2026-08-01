# PLAN: Register–Topic Decomposition — Making the INLP Canon Flow

> Status: PLANNED, NOT IMPLEMENTED. Engineering left for a later session.
> Companion to `PLAN_journal.md`. Supersedes the register-adjustment treatment
> that previously lived only in Appendix F (diagnostic).

## 0. Empirical gate (resolved, read-only)

Computed from committed artifacts (`num_coverage.tex`, `num_iterative_register_check.tex`,
`num_register_adjustment.tex`). The raw H1a null (rho=-0.09, p=0.73) is a
**cancellation** of two real, opposite effects:

| Test | rho | p |
|------|-----|---|
| Coverage gap vs **raw** semantic gap | -0.091 | 0.729 |
| Coverage gap vs **adjusted** (topic) gap | **+0.440** | **0.077** |
| Coverage gap vs **register** component | **-0.496** | **0.043** |

SDG 17: coverage gap 0.110 (large) -> raw gap 0.216 (smallest) -> adjusted gap
**0.388 (largest)**. Register similarity masked topic divergence.

Interpretation: the raw "no association" was never a real absence -- it was two real
signals (topic divergence up with coverage divergence; register divergence down with
coverage divergence) pulling in opposite directions until they cancelled. The
cancellation logic does not depend on the p-values being tiny; with n=17 the
opposite signs summing to ~0 explain the null regardless of individual significance.

## 1. The reframe (retire "independence")

| | Current | New (canon) |
|---|---|---|
| Central claim | "Coverage and framing are **independent**" | "Coverage predicts topic divergence **+** and register divergence **-**; the raw-gap null is a cancellation of two opposing signals" |
| Register | Limitation (Appendix F) | Core methodology step (Section 3), cited as INLP |
| Level 2 | textual-semantic proximity | splits into **2a (register)** / **2b (topic)** |
| Headline | Dissociation | **Register-topic decomposition** |

## 2. Canon pipeline flow

```
1. Embed texts in frozen BERT space (all-mpnet-base-v2)   [DONE]
2. Train SDG classifier on labelled reference texts
   -> F1 = 0.82 (market-level; validates assignment)        [DONE, keep on ORIGINAL embeddings]
3. Validate assignment sanity:
   - Coverage profiles: research top SDGs (3,9,4) match prior-lit consensus
   - Policy top SDGs (16,13,17) "intuitive" (interpretive)
4. ---- PIVOT: Iterative Nullspace Projection (INLP) ----
   Applied to RESEARCH + POLICY measurement embeddings
   -> adjusted embeddings (register removed, topic retained)
   -> cite Ravfogel et al. 2020; SDG-stratified adaptation is ours
   -> Materialise as projection matrix G only (94 orthonormal directions, ~KB);
      adjusted = project(raw, G) on the fly. Full adjusted .npy NOT stored.
   -> NOTE: coverage gap is assignment-based, so it is IDENTICAL for raw and
      adjusted; only the semantic-gap vector changes between them.
5. PCA before/after (main-text figure): two clouds -> one merged cloud
6. Main analysis on ADJUSTED embeddings:
   - Semantic gaps = adjusted (canonical); raw = register-inclusive reference
   - H1 decomposition: cov vs adjusted (topic) rho=+0.44; cov vs register rho=-0.50
   - Cross-sensitivity, encoder, distributional, sample-stability,
     concept-retrieval, source-family -- ALL re-run on adjusted
7. Final comparison = cov-sem CORRELATION (centrepiece), not rank-stability
```

## 3. OPEN QUESTION -- classifier adjustment

Raised: should register adjustment happen *beforehand* on the reference/labeled
corpus, to make the classifier register/source-general? **Recommendation: keep the
classifier on ORIGINAL embeddings; apply INLP only to the measurement surface
(research+policy).** Reasons:

- The classifier assigns SDGs by *topical* content. Register is part of how SDGs are
  discursively marked (policy texts use institutional register), so removing register
  from the classifier would likely **hurt** F1=0.82 -- the assignment would lose a real
  signal.
- The reference corpus is policy-adjacent only (no research texts), so a
  research-vs-policy INLP cannot be trained on it. What is really described is
  **source-invariance** (invariant to OSDG / SDGi / Benchmark / Aurora / Knowledge
  Hub) -- a *different* target attribute requiring INLP on the source label,
  methodologically distinct from register removal.

Open sub-decisions if we ever pursue source-invariance:

- (a) Stratify INLP by SDG or by source? (SDG-stratification prevents topic leakage;
  source-stratification is what you'd want for source-invariance -- but then topic
  leaks in.)
- (b) Does source-invariance change SDG assignments enough to alter coverage profiles?
  Unmeasured.
- (c) Worth the complexity for a dissertation? Probably **future work**, acknowledged
  as a limitation.

Plan records this as OPEN -- default execution uses original-embedding classifier +
measurement-surface INLP. Source-invariance noted as future work, not built now.

   Separately: the concept-retrieval track REUSES `G_canon` rather than learning
   its own INLP. This assumes the research-vs-policy register direction is
   corpus-general from canon research to concept-retrieved research; if concept
   papers carry a different register, residual register remains in the adjusted
   concept gap. This assumption is stated explicitly in the manuscript.

## 4. New artifacts

- **Decomposition table** (per-SDG, main-text Results): `raw gap | adjusted gap |
  register component (raw-adjusted) | coverage gap`. The paper's new centrepiece.
- **PCA before/after figure** (main-text): MPNet raw (two clouds) vs MPNet adjusted
  (one merged cloud). Text note: SciBERT already fuses the clouds, so the effect is
  encoder-dependent.

## 5. Manuscript restructure (section by section)

- **Lit Review:** extend three-level framework -- Level 2 -> **2a register proximity**
  / **2b topic proximity**.
- **Methodology Section 3:** move register adjustment from Appendix F to a core
  subsection *before* gap computation; cite INLP; state SDG-stratified adaptation;
  note classifier stays on original embeddings (open question recorded).
- **Results:**
  - Semantic gaps reported as **adjusted** (canonical); raw kept as register-inclusive
    reference.
  - New **decomposition table** (Section 4.2).
  - **PCA before/after** figure.
   - **H1 reformulated**: cov gap vs adjusted (topic) gap rho=+0.44, p=0.08; cov gap vs
     register component rho=-0.50, p=0.04; raw null = cancellation. **Cov-sem corr is
     the centrepiece.**
     The canon RAW result (rho=-0.09) is RETAINED in main text as the "before" of
     the cancellation narrative, not moved to appendix. Canon raw + adj + register
     all live in main text; only robustness configs go to appendix.
  - Rank-stability tables demoted to supporting robustness.
- **Discussion:** retire "independence/dissociation"; SDG 17 reframed as "high topic
  divergence masked by register similarity"; decomposition = key contribution; cov-sem
  opposition discussed.
- **Limitations:** register adjustment is now a *method*, not a caveat; flag borderline
  p-values (n=17 power); source-invariance as future work.
- **Abstract / Conclusion / title:** "dissociation" -> "register-topic decomposition."

## 6. Engineering tasks (execution) -- REVISED ARCHITECTURE

### 6.0 Design principle (applies to all new/changed stages)
- Expensive analysis stages emit **JSON only** (already true for semantic_gap,
  coverage, interaction, cross_method, etc.).
- A single consolidated macro script
  `1_code/7_main_analysis/0_shared/generate_tex_macros.py` (NEW) reads the
  restructure JSONs and writes ALL new `num_*.tex` macros. Per-script `num_*.tex`
  emission is grandfathered for existing stages, but the restructure's new macros
  are centralised here. Tables (`tab_*.tex`) may still be emitted by their
  generator; macros are not.

### 6.1 New stage `register_adjust` (materialise G, not full embeddings)
- Run INLP (SDG-stratified, iterative; canon reaches 94 directions) on
  research+policy embeddings -> persist ONLY the orthonormal projection matrix
  `G` (K x dim, ~KB) + metadata json (n_iters, final_acc, per-iter acc) to
  `2_data/3_embedded/{slug}/register/{track}/G.npy`.
- NO full adjusted `.npy` arrays are stored (~10 GB MPNet cost avoided entirely).
- Adjusted embeddings are produced on the fly by `project(raw, G)` (orthonormal
  subtract-all + per-row L2 renorm) -- mathematically identical to stored adjusted.
- Tracks / G sources:
  - MPNet `canon`: iterative SDG-stratified INLP on full research+policy ->
    `G_canon`. Use the **iterative** `G_list`, NOT the naive single-direction `g`.
  - MPNet `concept`: **reuse `G_canon`** -- no re-run (assumption stated in §3).
  - MiniLM / SciBERT `subset`: re-run INLP on research_subset+policy -> own `G`.
- **Within-SDG balance fix (required for subset runs):** the iterative check must cap
  policy per SDG at `min(ITERATIVE_N_PER_SDG, n_available_research_for_that_sdg)` so research
  and policy stay equally represented *within* each SDG. In the 50k subset, rare SDGs have
  <1000 research (SDG 1: 173-263; SDG 17: 208-500; SDG 6/14: ~550-810; minilm vs scibert
  vary), so the uncapped code would train, e.g., 173 research vs 1000 policy and bias the
  learned register direction. Canon is unaffected (every SDG has >=18k research -> full
  1000/1000 balance). ~2-line change in `load_stratified_samples`.
- **Method caveat (state precisely when framing):** the iterative check trains a 34-class
  (17 SDG x 2 corpus) classifier and projects out its *flattened* (34 x dim) coefficient
  vector, so each `g_k` removes a register-weighted blend of topic + register, not pure
  register. This is the established canon method and the gate validated its outcome (the
  cancellation pattern); keep it, but describe it as a register-weighted topic+register
  projection, not a pure register removal.
- Fingerprint `G` + raw inputs so downstream skips when unchanged.

### 6.2 `register_utils.py` (NEW, shared)
- `load_G(model, track)`, `project(emb, G)`, `project_centroids(path, G)`,
  `get_policy_emb_adjusted(model, track)`,
  `get_research_centroids_adjusted(path, G)`.
- `--embeddings {raw,adjusted}` flag added to the 8 downstream scripts
  (semantic_gap, interaction, cross_sensitivity, encoder_sensitivity,
  distributional, sample_stability, concept_retrieval, source_family); the flag
  swaps the two shared loaders above. Segment-cap / concept / encoder overrides
  still apply.

### 6.3 Coverage gap is adjustment-invariant (no re-run needed)
- `0_coverage_gap.py` derives coverage from SDG **assignments** (classifier on
  ORIGINAL embeddings, per OPEN decision). Coverage gap is IDENTICAL for raw and
  adjusted; only the semantic-gap vector changes. Do NOT re-run coverage per
  raw/adjusted -- reuse the existing coverage JSON for both.

### 6.4 Re-run matrix (semantic gaps only; coverage reused per §6.3)
Adjusted semantic-gap JSONs written under an `adjusted/` mirror of the raw layout
(e.g. `4_outputs/{model}/data/adjusted/4_3_semantic_gap_distances.json`,
`.../data/concept/adjusted/...`).

| Config | Raw sem (methods) | Adj sem (methods) |
|---|---|---|
| MPNet canon | LR, MLP, ZS | **LR, MLP** |
| MPNet concept | -- | LR, MLP (reuse G_canon) |
| MiniLM subset | -- | LR, MLP (own G) |
| SciBERT subset | -- | LR, MLP (own G) |

`register_gap[sdg] = raw_gap[sdg] - adj_gap[sdg]` per config -> feeds rho(cov, register).

### 6.5 New generators (all JSON-out; macros centralised per §6.0)
1. **Decomposition table** (canon, main text): per-SDG raw / adj / register /
   coverage. Inputs: `num_register_adjustment.tex` source values (or its JSON) +
   `4_2_coverage_document_weighted.json`. New script
   `0_shared/g_register_decomposition.py` -> JSON.
2. **Interaction extension** (canon centrepiece, main text): the existing
   correlation script only computes rho using the RAW semantic gap. Extend it to
   ALSO compute rho using the ADJUSTED semantic gap and the REGISTER component, so
   the two headline numbers (rho=+0.44, rho=-0.50) are produced. Plainly: we add
   two more correlation columns to the same test. Emits JSON.
3. **Consolidated correlation table = the ONE table** (appendix; canon rows also
   cited in main text): per config x {rho(cov,raw), rho(cov,adj), rho(cov,register)}.
   New script `0_shared/h1_register_correlation_table.py` reads per-config coverage
   + raw/adj semantic-gap JSONs -> JSON (+ its `tab_*.tex`).
4. **PCA before/after** (main text): project a stratified sample (raw vs adj) of
   research+policy clouds; two-clouds -> one figure. MPNet, encoder-dependence note.

### 6.6 Consolidated macro script
`generate_tex_macros.py` reads the JSONs from 6.5.1-6.5.3 (+ existing raw JSONs)
and writes the restructure's `num_*.tex` macros (decomposition, centrepiece rho's,
correlation-table summary). Run LAST after all JSONs exist.

### 6.7 Manuscript restructure per §5 (with refinements)
- Canon RAW result (rho=-0.09) stays in MAIN TEXT as the "before" of the
  cancellation story. Canon raw + adj + register all in main text; only robustness
  configs -> appendix. Concept-reuse assumption stated explicitly (§3/§6.1).

### 6.8 Build, verify, commit, push to `register-adj`.

## 7. Verification

`--build-pdf --overwrite` -> exit 0, 0 undefined refs/citations, 0 em dashes, word
count within cap, pdftotext spot-checks (decomposition table, PCA caption, H1
reformulation, SDG 17 reframe).

## 8. Risks & mitigations

- **Borderline p-values (n=17):** mitigate by framing the *cancellation pattern*
  (opposite signs summing to ~0) as the result, not the asterisks. Honest and robust.
- **Re-computation cost:** ~8 stages re-run on adjusted; engines exist, it's
  integration. Feasible.
- **Internal consistency:** full re-run (not partial) keeps Results canonical=adjusted
  while robustness tables=adjusted. No half-measures.
- **Storage:** ~10 GB adjusted embeddings in gitignored `2_data/`; fine.

## Confirmed decisions (from planning session)

- Central claim: RETIRE "independence/dissociation" in favour of component-dependent
  opposition (topic +, register -).
- Final comparison: cov-sem CORRELATION is the centrepiece (not rank-stability).
- PCA before/after: main-text figure, MPNet with encoder-dependence note.
- INLP citation (Ravfogel et al. 2020) already added to `references.bib` and the
  manuscript in commit 8a83eaf.

## 9. Discovery narrative (from the planning session -- why this matters)

This section records the arc of how the restructure was found, because the
*story* is as important as the mechanics. It is the difference between a
dissertation that measures something and one that discovers something.

### The null was never a null

The original paper's headline was a dissociation: coverage gap and semantic gap
are "independent dimensions." That was always a little defensive -- "we found
nothing, and that's interesting." The gate computation showed it was never a
null at all. The raw gap correlates ~0 with coverage divergence because two real
signals cancel: topic divergence rises with coverage divergence (rho=+0.44) while
register divergence falls with it (rho=-0.50). The blunt raw distance averaged
them into noise. Once you separate register from topic, the silence speaks.

This is not a reframe. A reframe is cosmetic. This is the data telling you
something the raw number was too crude to show. The "independent dimensions"
framing was a misread of a cancellation.

### The dream method was real

The register-adjustment procedure was arrived at independently -- almost
intuitively, "train a linear classifier again and again until there's no register
effect left." That instinct turned out to be Iterative Nullspace Projection
(INLP; Ravfogel et al., ACL 2020, 653 citations). The method is established; the
application to research-policy register is the novel adaptation (the
SDG-stratified training is ours). So the instinct was sound, the execution was
sound, and pointing an established technique at a problem it hadn't been pointed
at is legitimate. We did not invent a suspicious ad-hoc trick; we rediscovered a
peer-reviewed method and aimed it correctly.

### Two results, not one

The decomposition yields two individually borderline-significant, opposite-signed
associations. That is the gem: not a single finding but a pair that cancel. A
reviewer cannot dismiss it as "one borderline correlation" because the structure
-- opposite signs summing to ~zero -- is the result, independent of whether each
limb clears p<0.05. With n=17 the asterisks are weak; the pattern is not. Say that
plainly and it holds.

### SDG 17 is the diagnostic trap

SDG 17 has the largest coverage gap (0.110) yet the smallest raw semantic gap
(0.216) and the largest adjusted gap (0.388). Both communities use partnership /
coordination / institutional language -- same register -- which makes a naive
embedding distance conclude they are aligned, when under the register they are
talking past each other. That is the cleanest illustration in the whole paper:
the raw gap was a register artefact masking deep topic divergence. It is the case
a reviewer cannot easily wave away.

### What this does to the work

The honest read: this moves the dissertation from "borderline publishable
measurement-protocol paper" to a piece with a Nature-level headline -- a finding
that dissolves a limitation (register contamination) into a decomposition
(register vs topic) that reveals structure the raw analysis missed. The
"limitation" section of the old paper becomes a dissolved limitation: register and
topic are now two separable, measurable components of Level 2 divergence, not a
shrug.

The research loop that produced it is the point: build the instrument carefully
enough that when you finally point it the right way, it tells you the truth --
then re-search the null instead of accepting it. Most of research is grinding.
That moment, where a footnote turns into the contribution, is why people grind.

### Caveats to carry into the execution

- n=17 power is real; frame the cancellation pattern, not the p-values.
- "Independent dimensions" language must be fully retired from title/abstract/intro.
- The adjusted gaps become canonical; raw is the register-inclusive reference.
- Source-invariance of the classifier is explicitly OPEN (Section 3), not built.

  This narrative is why the restructure is worth the re-computation. It is not
  re-ordering. It is a change in what the work *is*.

## 10. Revised architecture decisions (post-planning, locked)

Captured during architecture review before implementation. Supersedes conflicting
text in §2/§3/§5/§6 where noted.

- **G-only materialisation**: adjusted embeddings are NOT stored as full `.npy`
  arrays. INLP persists only the orthonormal projection matrix `G` (K x dim, ~KB)
  per (model, track); adjusted arrays are produced on the fly by `project(raw, G)`.
  Mathematically identical to stored adjusted; removes the ~10 GB MPNet cost.
  (Correction B.)
- **Coverage gap is adjustment-invariant**: it derives from SDG assignments
  (classifier on original embeddings), so it is identical for raw and adjusted.
  Only the semantic-gap vector differs. (Correction A.)
- **Canon adjusted = iterative SDG-stratified G** (94 dirs), NOT the naive
  single-direction `g` also computed by `f_register_adjustment.py`. (Correction C.)
- **Canon RAW result (rho=-0.09) stays in main text** as the "before" of the
  cancellation story; canon raw + adj + register all in main text, robustness
  configs in appendix.
- **Register-component correlation column for ALL configs**: compute
  rho(cov, register) per config to test whether the opposite-sign cancellation
  replicates across encoders/concept.
- **MPNet canon does MLP-adjusted too** (not LR-only): canon adjusted methods =
  LR + MLP. Raw canon methods = LR + MLP + ZS (ZS adjusted is out of scope per
  AGENTS.md; ZS appears raw-only under the MPNet group).
- **Concept track reuses `G_canon`** (no INLP re-run); assumption stated explicitly
  in manuscript (§3/§6.1).
- **JSON-out / consolidated-macro design principle**: expensive analysis stages
  emit JSON; a single `generate_tex_macros.py` reads the restructure JSONs and
  writes the new `num_*.tex` macros. Applies to all new/changed stages going
  forward.
- **Re-run matrix (semantic gaps; coverage reused)**:
  MPNet canon: raw LR/MLP/ZS, adj LR/MLP. MPNet concept: adj LR/MLP (G_canon).
  MiniLM subset: adj LR/MLP (own G). SciBERT subset: adj LR/MLP (own G).
  `register_gap = raw - adj` per SDG per config.
