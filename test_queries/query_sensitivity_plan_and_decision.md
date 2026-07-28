# AI-Query-Term Sensitivity — Plan & Locked Decision Tree (advisor review item 1, path #1)

This file is the recorded discipline for the bounded robustness check implemented in
`1_code/9_robustness_checks/0_query_sensitivity.py`. It is written BEFORE running the
script, per the advisor's approval condition (2): the rho-low contingency must exist
before the number is seen, not after.

## Condition (1): source verification — DONE
- OECD.AI, "Data from partners: a methodological note"
  (oecd.ai/en/partner-data-methodological-note), section "Determining Artificial
  Intelligence papers for OECD.AI" — fetched and read directly; REAL, citable.
- Elsevier (2018), "Artificial Intelligence: How knowledge is created, transferred,
  and used" — 700-keyword Fingerprint Engine(tm) taxonomy; the closer analogue to the
  current keyword method. Treated as real/confident per advisor.
- CSET Georgetown — Schoeberl, C., Toney, A. & Dunham, J. (2023), "Identifying AI
  Research", DOI 10.51593/20220030. Independently RESOLVED this session via the DOI
  (title/authors/venue confirmed) before being allowed into the manuscript. REAL.
- The earlier "Carraud & Gault, OECD STI WP" reference was fabricated/guessed (two
  guessed DOIs 404'd; user confirmed it does not exist). It is NOT used.

## Design
- Path #1 (additive, scratch-only). Path #2 (corpus replacement) rejected: week-scale,
  invalidates every committed number, and forfeits the sensitivity claim.
- Two variants, both write only to `5_notes/scratch/`:
  (b) concept-tag subset via OpenAlex `concepts.id` (ML/DL/NN/AI) — OECD-aligned,
      non-arbitrary retrieval contrast.
  (a) free-text expanded term superset — the reviewer's literal ask.
- Each: fetch bounded subset (~50-100k) under the SAME sdg/year/abstract filters,
  embed "title. abstract" with cached MPNet, score with the ALREADY-TRAINED LR,
  compute per-SDG hard-assignment shares, correlate (Spearman rho, Kendall tau) vs the
  committed baseline `research_profile_hard`.
- No 2_data/ or 4_outputs/ touched. The trained LR is valid (trained on SDG
  reference-labelled data, not the research corpus).

## LOCKED DECISION TREE (applied to actual results, not reverse-engineered)
Let rho = Spearman correlation of each variant's per-SDG share vector vs baseline.

- BRANCH H (rho >= 0.9, at least for concept_tag):
  Corpus is robust to the AI-query-term choice. Apply Step 3 happy path:
    * add a small robustness appendix entry citing OECD.AI note + Elsevier (2018) +
      CSET (2023);
    * soften (NOT remove) the Section 3.1 limitation: cite the schema, state the
      bounded check empirically validates robustness, but KEEP the "conditional on term
      selection" limitation statement (it remains true).
- BRANCH L-CT (concept_tag rho < 0.9):
  Possible real defect — INVESTIGATE, never write around it:
    (i) Pipeline sanity: correct SDG filter? correct dedup? embeddings/scoring identical
        to baseline (same classifier, same "title. abstract" text, normalize=True)?
    (ii) If pipeline correct -> genuine, reportable retrieval-strategy sensitivity
        ("our corpus is robust to free-text term expansion but not to concept-taxonomy
        retrieval"). Report transparently as a sensitivity/limitation finding; DO NOT
        soften Section 3.1 to claim validation. Strengthen the limitation instead.
    (iii) If a pipeline bug is found -> fix, re-run, then apply the correct branch.
- BRANCH L-FT (free_text rho < 0.9 only):
  Same discipline: investigate pipeline correctness first; if correct, this confirms the
  reviewer's literal concern (term sensitivity is real). Report transparently; DO NOT
  claim validation.

In every branch, the Section 3.1 limitation statement is preserved (the term selection
remains a stated scope limitation). The check ADDS evidence; it never deletes the caveat.

## Scope discipline
Only advisor item 1 is in scope. Items 2-7 (within-register baseline, EMD/Wasserstein,
SciBERT/SPECTER encoder check, bootstrap/target-level H1a test, soft-assignment, full
TF-IDF) remain parked as separate, separately-scoped investigations.

## Methodology incident log (part of the investigate discipline)
- FIRST RUN used a per-(concept, SDG) cap of 600. concept_tag returned rho=0.8431
  (<0.9) -> triggered BRANCH L-CT. Investigation found the low rho was a SAMPLING
  ARTIFACT, not genuine sensitivity: capping each (concept, SDG) pair equally truncates
  high-SDG goals (SDG3/9/16) while leaving low-SDG goals uncapped, flattening the per-SDG
  distribution toward uniformity and depressing the rank correlation. FIXED by capping per
  CONCEPT / per TERM and enforcing the baseline's SDG scope as a post-filter on returned
  records (keep papers OpenAlex tags with >=1 SDG). This preserves the natural per-SDG
  composition and isolates the retrieval-STRATEGY effect. Rerun in progress.
- After the fix, re-evaluate both variants against the BRANCH H / L decision tree above.
  If concept_tag rho is still <0.9 post-fix, that is a genuine retrieval-strategy
  sensitivity finding (report transparently), not an artifact.
