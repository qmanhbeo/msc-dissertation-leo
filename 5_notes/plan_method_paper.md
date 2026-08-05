# Plan: Method-led framing pass + Appendix G second-encoder strengthening

Status: rough plan (2026-08-05). No code or framing changes made yet — this
document is the intent + feasibility map. Reference: `5_notes/report_editorial_suggestions_ignore.md` (architecture
verification audit) and the PI consensus in that thread.

---

## 1. Goal

Make the **measurement framework** (coverage/framing separation + register
decomposition) the paper's headline, with the AI–SDG analysis as a high-stakes
proof-of-concept. Two tracks:

- **Track A — framing/consistency pass** on `3_writing/dissertation.tex`
  (no analysis, no numbers changed; kill the internal headline contradiction).
- **Track B — strengthen Appendix G** with a second encoder (MiniLM) so the
  register decomposition — now load-bearing — has cross-encoder validity
  evidence, not just MPNet.

## 2. Track A — framing pass (edit locations, dissertation.tex)

| Location | Change | Effect |
|---|---|---|
| Abstract (lines 89–95) | Lead with framework contribution (incl. INLP register decomposition as part of the method); cancellation becomes hedged diagnostic-warning sentence; p-values as footnotes; name SDG 17/15 | Reader gets the real headline in 30s |
| Intro para "contribution" (line 111) | Scope "advance AI–SDG mapping from Level 1 to Level 2" to *research–policy alignment measurement*; add register decomposition to contribution statement | Removes novelty overreach (Gjorgjevikj2025 / McCarthy2026 are cited in the same text) |
| Intro para 4 (line 113) | Reframe result-preview as "applying the framework reveals a diagnostic warning" (shared labels can mask a register–topic cancellation) | Finding reads as proof-of-concept, not discovery |
| Intro roadmap (line 115) | Name the decomposition | Telegraphs the method early |
| Lit review §2.4 (lines 160–164) | Add one bridge sentence: the study will *measure and remove* register, not merely flag it | Foreshadows §3.8 as designed |
| Methodology §3.8 opening (line 259) | Reframe from corrective ("the semantic gap is not a clean measure…") to centerpiece ("the framework's second instrument separates topic from register…"); add one-sentence validation summary | Instrument reads as designed, not patched |
| Methodology §3.9 (line 285) | Rename to expose hypotheses (e.g. "Hypotheses: The Coverage–Semantic Interaction") | Hypotheses findable in ToC |
| Results §4.2 | Make SDG 17 inversion + SDG 15 the explicit center; one-sentence pointer to Appendix G validation | The finding lands where earned |
| Results §4.3 (lines 381–408) | Reframe from "A Possible Cancellation" to diagnostic framing (without the decomposition the raw test is null) | Same numbers, method-led meaning |
| Discussion §5.1 (line 435) | **Highest-impact edit**: "The headline result is a possible cancellation" → "The application demonstrates the framework's value: raw measurement would conclude independence; the decomposition reveals a cancellation" | Kills the internal headline contradiction |
| Discussion §5.4 | Keep n=17 power limit but frame it as bounding the *application*, not the method | Protects the contribution |
| Conclusion para 1 (lines 488–494) | Lead with framework; "primary result" → "key result of the application" | Matches lines 439/496/498 |
| Appendices | Compress A.3 (→ §3.4 sentence), C.3 (→ §5.4), F.1 (dedupe vs §4.2), G.5 (results-shaped prose) | Removes "patchwork" fingerprints |

## 3. Track B — Appendix G second encoder (MiniLM)

### 3.1 Facts from feasibility check (verified read-only)

- `a1_register_validation.py` is **canonical-MPNet-only**: `run()` returns early
  for any `model != DEFAULT_EMBED_MODEL` (line ~618), and the file docstring /
  gates call it MPNet-canonical.
- G checkpoints exist for all three encoders:
  - `2_data/3b_register/mpnet/canon/G.npy` (k=62, complete)
  - `2_data/3b_register/minilm/subset/G.npy` (k=29, complete)
  - `2_data/3b_register/scibert/subset/G.npy` (k=71, complete)
- MiniLM track data exists:
  - embeddings: `2_data/3_embedded/minilm/` (policy.npy 40597×384; **1** research
    shard part-00001.npy)
  - scored: `2_data/5_supervised_scored/minilm/` (policy_scores.npy 40597×17;
    paper_scores_shards/metadata/part-00001_ids.jsonl)
  - **MiniLM is the 100k deterministic subset (seed 42), not the full 3.1M corpus**
    (only 1 research shard vs MPNet's 26). Consistent with the rest of the
    paper's MiniLM/SciBERT treatment.
- Register score features are computed from **text**, so the six-feature
  register score itself is encoder-independent (same segments); only the
  discrimination / selectivity / projection steps depend on embeddings + G.
- Text path risk (OPEN): `resolve_research_text_path(model, part-00001)` points
  at canonical `2_segmented/research/part-00001.jsonl` (full-corpus shard), but
  MiniLM's part-00001 is the 100k **subset**. The subset text likely lives in
  `2_data/2_segmented/research_subset/part-00001.jsonl` (exists). **Must verify
  row alignment** (subset text rows ↔ MiniLM embedding rows) before trusting
  register-score/text features for MiniLM.

### 3.2 Implementation sketch (later)

1. Generalise `a1_register_validation.py` to accept MiniLM (remove early-return;
   resolve subset text path correctly; keep all statistics identical).
2. Outputs are already model-scoped:
   `4_outputs/appendix/{model}/a1_register_validation/` — so MiniLM writes its
   own `data/`, `tables/`, `num_*.tex` without touching MPNet outputs.
3. New/parallel appendix table (or extended G) reporting MiniLM corpus
   discrimination before/after, topic selectivity, and residual-register checks
   — mirroring the MPNet G structure.
4. New `num_*` macros for MiniLM validation; wire into dissertation.tex appendix
   text (add a G subsection "cross-encoder replication: MiniLM").

### 3.3 Risks / decisions

- MiniLM subset scope: validation on 100k subset (not full corpus) — state this
  plainly in the appendix; it matches how MiniLM is used everywhere else.
- Sampling: 12/SDG/corpus needs enough per-SDG research rows in the subset —
  verify per-SDG counts before committing.
- Seed discipline: keep the single continuous seed-42 stream + fresh 43/44/45
  draw-stability pattern; do not re-seed per call (script comment warns).
- If MiniLM register score / discrimination is much weaker than MPNet, decide:
  (a) report as-is with honest limits, or (b) only SciBERT… (likely keep MiniLM;
  SciBERT optional later).

## 4. Sequencing

1. Track A first (fast, text-only) → rebuild PDF → verify refs/labels intact.
2. Track B (code + compute, hours) → run MiniLM validation → new table/macros →
   wire into appendix G → rebuild PDF.
3. Optionally: consider SciBERT as a third validation point later.

## 5. Verification

- `./3_writing/build_pdf.sh` (bash/WSL) after each track; check no orphaned
  `\ref`, no broken `\input`.
- Track B acceptance: MiniLM discrimination/selectivity tables produced, gates
  pass (MPNet numbers must remain byte-identical to today).

## 6. Do NOT change

- All numbers, tables, figures, pipeline, H1a–H1d definitions.
- Three-level framework (§2.2); results arc; raw/adjusted panel convention.
- Cross-sensitivity grid (becomes more central, not less).
