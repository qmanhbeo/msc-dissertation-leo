# PLAN: Journal-submission readiness (venue-agnostic)

Operational plan for shaping `3_writing/dissertation.tex` into a submission-ready
measurement-protocol article. Authoritative companion to `AGENTS.md`.

## Framing and conventions

- **Target identity:** a *measurement-protocol* paper, not an empirical SDG
  finding. This framing is already in place: the Abstract leads with the
  protocol contribution, the Introduction states the framework contribution,
  and the Discussion names the framework as the main contribution.
- **Single source of truth:** `3_writing/dissertation.tex` (already
  `\documentclass{article}`). No parallel journal file. Venue template/format is
  the final mechanical step and is deliberately deferred.
- **Voice:** every new prose edit uses first-person singular **I**, matching the
  existing manuscript ("I advance", "I operationalise", "I treat"). Keep this
  consistent.

## Completed (prior sessions + journal pass)

- Discussion tightened (2,771 -> 1,691 words); Limitations rebalanced to credit
  mitigations and to cite sample-stability and each appendix.
- Introduction + Literature Review audited (six citation fixes, verified by
  subagents reading the PDFs in `0_literature/`).
- Em dashes fully purged (0 unicode U+2014 and 0 three-hyphen `---`).
- **P1** Abstract opens with the measurement-protocol contribution; reports the
  H1a confidence interval.
- **P2** Introduction contribution sentence names the robustness machinery
  (cross-sensitivity grid, sample-stability ladder, register-adjustment).
- **P3** Fisher-z 95% CIs for all four H1 predictors plus a minimal detectable
  effect (`\HPrimaryMinDetectableR` = 0.63 at 80% power, alpha=0.05, n=17) added
  to `1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py`;
  `num4_interaction_h25.tex` regenerated; CIs and the power caveat reported in
  Abstract, Results, and Discussion.
- **P4** Keywords + Data and Code Availability (GitHub; private until grade
  release) + competing-interests statement.
- **P5** Conclusion trimmed (297 -> 195 words); redundancy with Discussion
  removed.
- Verified: build clean (0 undefined refs/citations), main text 8,882 words.

## Item 2 — balanced research-subset stability check (DONE, 2026-07-31)

**Concern raised by reviewer pass.** The research corpus (3,105,144 segments from
2,536,771 abstracts) dwarfs the policy corpus (40,597 segments), so the within-SDG
semantic gap
partly reflects estimation precision on the policy side; the existing
sample-stability ladder subsamples the research corpus only.

**What was done.** Rather than re-embed a standalone `research_subset` for MPNet
(the ladder already subsamples the MPNet research embeddings to 50k across 100
seeded draws, which is statistically stronger than one fixed subset), a new
appendix stage reuses the existing draws:

- New script `1_code/7_main_analysis/2_appendix/c1_subset_balanced_stability.py`
  (new stage **C.1**): for each sampled tier it computes the Spearman rank
  correlation between each draw's within-SDG semantic-gap ranking and the
  full-corpus ranking from `semantic_gap_distances_lr.json`. Cheap, reads no
  embeddings, fingerprint-gated like other Tier B scripts.
- Result at the balanced ~50k tier (vs ~40,597 policy segments):
  **ρ = 0.983 ± 0.009** over 100 draws (17 SDGs), already 0.951 at 10k and
  converging to 1.000 at the 2M full corpus. Macros:
  `\SubsetGapRhoFiftyK` / `\SubsetGapRhoStdFiftyK` /
  `\SubsetGapRhoFiftyKN` (+ per-tier `\SubsetGapRho*`),
  emitted to `4_outputs/appendix/mpnet/c1_subset_balanced_stability/tables/num_c1_subset_stability.tex`.
- Wiring: added to `analysis_orchestrator.py::APPENDIX_STEPS` (after
  c_sample_stability), `main.py` flag `--appendix-c1-balanced-subset` +
  `--appendix-all`, and `shared_utils.py` manifest lists.
- Manuscript: one first-person Results sentence (Semantic Gap subsection) plus an
  appendix paragraph under the Sample-Stability appendix
  (`\label{app:balanced-subset-stability}`): the research-side
  over-representation does not drive which SDGs show large versus small gaps.

**Magnitude note.** Gap *magnitudes* are slightly elevated at 50k
(mean 0.354 vs 0.352 full) because smaller research centroids are noisier; the
ranking — which the dissociation rests on — is preserved. Both are reported.

## Item 1 — corpus-asymmetry confound (Option (b) DONE, 2026-07-31; Option (a) future work)

**Concern.** Research corpus = AI cap SDG; policy corpus = broader union. The
headline dissociation (coverage perpendicular framing) could partly be an
artifact of comparing two differently-constructed corpora.

**What was done — Option (b), the symmetric curated control (offline, no OpenAlex).**
The "94-document curated AI/SDG set" in the original plan is the existing
`curated_ai_sdg` policy source family (policy_scrape 31 + policy_manual 64
docs). `a2_policy_source_family_sensitivity.py` was extended to replicate the
canonical H25 interaction test (four coverage predictors vs within-SDG semantic
gap, Spearman + Pearson, with/without SDG 4) for each policy source family,
using the already-embedded full policy corpus. The full-corpus row exactly
reproduces the canonical 4_4 values (research ρ=0.216, covgap ρ=-0.078),
validating the replication. Result for the symmetric AI/SDG-vs-AI/SDG
construction (curated family, n=17): research share vs gap ρ = -0.17 (p=0.51),
coverage gap vs gap ρ = -0.41 (p=0.10) — the dissociation survives, and research
share is not in the hypothesised positive direction. Emits
`tab_a2_policy_source_family_h25.tex` + `num_a2_policy_source_family_h25.tex`
(`\CuratedHPrimary*`, `\FullPolicyHPrimary*` macros).

- Manuscript: one first-person Results sentence in the Coverage--Semantic
  Interaction subsection (`\label{sec:interaction}` added) plus an appendix
  paragraph + `tab:policy-source-family-h25` table under
  `app:policy-source-data`; Limitations acknowledge that "AI and the SDGs" is
  not yet a mainstream policy category, so the curated AI/SDG set is the
  closest symmetric control fieldable.
- Option (a) — broadening the research corpus to all-SDG (drop the AI-term
  intersection) — is left as future work, stated in the manuscript; it needs
  new OpenAlex retrieval + re-embedding (~1 week, API keys).

## Minor / cosmetic (optional, venue-agnostic)

- **M5** Methodology: phrase `\NResearchPapers` as "abstracts (each treated as
  one segment)" instead of "abstract-derived text segments". DONE (2026-07-31).
- **M4** Flag the SDG-4 artefact in the Abstract (one clause) for reviewer
  fairness. DONE (already present in Abstract, verified 2026-07-31).
- **M7** Add 1-2 DOI-verified 2025-26 AI-SDG bibliometric citations to Related
  Work (resolve each DOI before citing, per `AGENTS.md`). DONE (2026-07-31):
  Gohr et al. 2025, *Nature Sustainability* 8:970-978,
  doi:10.1038/s41893-025-01598-6 (review of 792 AI-for-SDG articles);
  Yin et al. 2025, *Array* 27:100419, doi:10.1016/j.array.2025.100419
  (AI-based SDG publication mapping). Both DOIs resolved via Crossref before
  citing.
- **M6** Figure filename renumber (`fig1_` duplicated, no `fig2`) — DEFERRED to
  venue (cosmetic for upload; in-PDF captions already number correctly).
- **M8** Double-blind anonymisation — DEFERRED to venue.
- **M9** Cut a GitHub release tag at grade release so the private-to-public
  switch is version-pinned.

## Deferred venue-specific step (mechanical, after target journal chosen)

Swap `\documentclass` to the journal class; biblatex style (apa <-> numeric);
structured/annotated abstract; section rename (e.g. Background); figure
dpi/format; cover letter; author/affiliation block.

## Verification checklist (every edit pass)

1. `python main.py --build-pdf --overwrite` -> exit 0.
2. Build log: 0 undefined references/citations.
3. 0 em dashes (unicode U+2014 and three-hyphen `---`) in `dissertation.tex`.
4. `python 5_notes/word_count.py` within the journal cap.
5. `pdftotext` spot-check of newly added content.
