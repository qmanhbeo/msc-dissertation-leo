# Go/No-Go: Does the INLP "Register" Interpretation Hold Up?

**Status:** Diagnostic first-pass check. NOT an appendix draft. MPNet (`all-mpnet-base-v2`, canon track, G(62,768)) only. Seed 42 throughout. Units are **segments** (~384-token chunks), because that is the unit the pipeline scores, embeds, and register-adjusts.

Run script: `5_notes/scratch/register_validation_check.py` · full log: `5_notes/scratch/regcheck_full.log` · artifacts: `5_notes/scratch/regcheck_arrays.npz`.

---

## Step 0 — What I found in the data

| Need | Location | Alignment |
|---|---|---|
| Research raw text (embedded) | `2_data/2_segmented/research/part-*.jsonl` — `text` field | line `i` == embedding row `i` |
| Policy raw text (embedded) | `2_data/2_segmented/policy.jsonl` — `text` field | line `i` == embedding row `i` |
| Research raw embeddings | `2_data/3_embedded/mpnet/research_shards/part-*.npy` | row == segment |
| Policy raw embeddings | `2_data/3_embedded/mpnet/policy.npy` (40597×768) | row == segment |
| INLP projection matrix G | `2_data/3b_register/{model}/{track}/G.npy` | MPNet canon (62,768); MiniLM subset (29,384); SciBERT subset (71,768) |
| **Adjusted embeddings** | **Never materialised** — projected on the fly via `register_utils.project()` (orthonormal G, `x' = P(x)` then L2-renorm) | |
| SDG label (research segments) | `2_data/5_supervised_scored/mpnet/paper_scores_shards/metadata/part-*_ids.jsonl` → `assigned_sdg` | row == segment |
| SDG label (policy segments) | `2_data/5_supervised_scored/mpnet/policy_scores.npy` → `argmax` | row == segment |
| Concept encoder | **Not an encoder.** `research_concept/` is the concept-retrieved *corpus* embedded with MPNet. It has **no** register artifacts (nothing under `3b_register` for it). Not usable for this check. | |

Critical Step-0 facts:
- **"Documents" = segments, not papers/docs.** Each research paper splits into multiple segments; one paper's segments can even carry different `assigned_sdg` (label is per-segment from the LR classifier). The INLP training (register_adjust.py) and the gap analysis all operate at segment level. My "200 documents per corpus" is therefore 204 segments per corpus (12/SDG × 17).
- **All alignments are positional** (row index), and I verified them: score-shard `row_in_shard` == embedding row == segmented line; `policy_scores.npy` row == `policy.jsonl` line == `policy_ids.json` entry. The only ID mismatch worth knowing: `policy.jsonl`'s `id` field is the *source_doc*, not the segment id (`segment_id` matches `policy_ids.json`).
- Encoders with complete artifacts (embeddings + G + matching text): **MPNet (full research), MiniLM + SciBERT (100k-paper subset)**. MPNet canon is the correct single-encoder target.

## Step 1 — Register features and combined score

Sample: 12 segments per SDG per corpus → **204 research + 204 policy = 408 segments**, all with ≥20 words (no attrition at this threshold), distinct papers / distinct policy source-docs within each SDG. Balanced 12/12/17 → majority-class baseline = 0.5 for corpus, 1/17 ≈ 0.059 for SDG.

Features (nltk punkt tokenizer + averaged-perceptron POS tagger), rates per 1000 words:
hedge (may/might/could/suggests/appears/potentially/likely), deontic (must/shall/should/will), passive (VBN after be-form), first-person pronouns, nominalization (‑tion/‑ment/‑ness), and mean sentence length (words/sentence).

**Combined score choice:** first principal component of the z-scored features, oriented so that positive = longer sentences. Reported because it is the dominant data-driven axis (22% of feature variance). **Caveat: the choice matters** — see §Discussion. I also computed an a-priori "institutional" z-sum; results differ (below).

Mean features (research vs policy):

| feature | research | policy | std mean diff (d) |
|---|---|---|---|
| hedge_rate | 1.427 | 0.854 | −0.155 |
| deontic_rate | 0.780 | **3.130** | **+0.539** |
| passive_rate | 10.303 | 8.453 | −0.200 |
| mean_sent_len | 37.0 | **63.8** | +0.154 |
| first_person_rate | 6.901 | 6.463 | −0.035 |
| nominal_rate | 36.4 | 39.0 | +0.118 |

Policy segments are deontic-heavy and long-sentence-heavy; research is passive-heavy (typical of abstracts). Notably **passive and hedge point toward research, not policy** — a naive "policy = formal/institutional" orientation of the score is wrong for this corpus pair.

## Step 2 — Quick correlation check (n=408, Spearman unless noted)

### 2b — Register score vs magnitude of the removed component ‖x − x′‖

| Contrast | ρ | p |
|---|---|---|
| Pooled | 0.102 | 0.040 |
| Within research (n=204) | **0.212** | 0.002 |
| Within policy (n=204) | **0.191** | 0.006 |

Per-feature vs ‖x−x′‖ (pooled): deontic **0.241** (p<1e‑6), mean_sent_len **0.241** (p<1e‑6), passive **−0.194** (p=8e‑5), nominal 0.117 (p=0.018), hedge −0.037 (ns), first_person −0.015 (ns).

**Reading:** the amount INLP removes does track the policy-coded features (deontic modality, sentence length) even *within* a corpus, so the removed subspace is register-*like*. But note **passive correlates negatively** — the removed direction is better described as "institutional modality/stance" than "academic formality."

### 2c — Register score vs distance-to-SDG-centroid (raw vs adjusted)

| centroid definition | RAW ρ (p) | ADJ ρ (p) |
|---|---|---|
| Pooled research+policy SDG centroid | 0.126 (0.011) | **0.247 (4e‑7)** |
| Own-corpus SDG centroid | 0.150 (0.002) | **0.245 (6e‑7)** |
| Partial Spearman controlling corpus | 0.130 (0.009) | **0.253 (2e‑7)** |
| Register score w/o mean_sent_len | — | 0.256 |
| Opposite-corpus centroid (gap-like) | 0.076 (ns) | **0.205** |

Mean distance: research 0.596→0.588, policy 0.353→0.475. Per-SDG research–policy centroid distance: raw 0.477 → adjusted 0.406.

**Reading — the red flag.** The register score correlates **more strongly** with distance-to-centroid in the *adjusted* space than in the raw space (0.25 vs 0.13), and this survives every robustness variant (own-corpus centroid, partialling out corpus, dropping the junk-sensitive sentence-length feature, within-corpus splits: research 0.160→0.301, policy 0.100→0.204). Under a clean "register fully removed" hypothesis, this should drop toward 0, not rise. There is **residual register-like structure inside SDGs even after INLP**.

### 2d — Corpus classifier (research=0 / policy=1), 5-fold CV accuracy

| Predictor | Acc |
|---|---|
| Register score (PC1) only | 0.456 |
| **Raw embeddings** | **0.909** |
| **Adjusted embeddings** | **0.505** |

Best single feature alone: deontic_rate acc=0.625; all others ≈0.50–0.56. A-priori z-sum acc=0.544.

**Reading:** the adjusted embeddings are effectively **indistinguishable from chance** for corpus identity (0.505), while raw is 0.909 — INLP destroys essentially all *linear* corpus-separable signal. That is exactly what it is designed to do and is consistent with the "removed = corpus/register-like" claim. **However**, the 6-feature register score alone reaches only ~0.46–0.54: these cheap features capture a small slice of what INLP actually removed. "Removed = *my 6 Biber proxies*" is only weakly supported; "removed = a corpus-separable subspace" is strongly supported.

## Step 3 — 17-way SDG classifier (selectivity spot-check)

| Space | LR (C=10) | kNN(5) | chance |
|---|---|---|---|
| Raw embeddings | **0.691** | 0.554 | 0.059 |
| Adjusted embeddings | **0.672** | 0.578 | 0.059 |

**No red flag here.** Adjusted SDG accuracy is essentially unchanged (LR Δ −0.019; kNN slightly up +0.024). INLP is **not** deleting topic signal at this sample size — the residual space still carries SDG structure.

---

## Plain-language verdict

**Leaning in favour of "INLP removes a register-like, not topic-like, component" — but not cleanly, and with one genuine warning.**

Evidence *for* the register interpretation:
1. Corpus separability is destroyed by adjustment (0.909 → 0.505, ~chance) while SDG/topic separability is preserved (0.691 → 0.672). This is the strongest, most direct result and it is robust.
2. The removed magnitude tracks policy-coded register features (deontic modality ρ=0.24, sentence length ρ=0.24) *within* each corpus — the removed subspace is not arbitrary.

Evidence *against* / cautions:
3. **Residual register structure survives adjustment (Step 2c).** High-register segments stay farther from their SDG centroid in adjusted space, and the association strengthens after adjustment. So INLP removes the *global* corpus direction but leaves per-SDG register-like offset (per-SDG research–policy centroid distance is still 0.41 after adjustment).
4. The cheap features barely capture what was removed (register-score-only corpus classifier ≈0.46–0.54 vs 0.909 raw). Any full validation appendix will need a much richer Biber-style feature set — 6 proxies are not enough to adjudicate "the removed subspace is register."
5. **The combined-score operationalization changes the answer.** PC1 (data-driven prose axis) shows the effects above; an a-priori "institutional" z-sum gives null for 2b (ρ=0.007) and *reversed* sign for 2c (−0.24). A validation claim must pin down *which* register operationalization is meant.

**Bottom line:** *GO* to a fuller validation appendix — the interpretation is not refuted and the selectivity checks pass — but the appendix must (a) use a substantially richer, corpus-linguistically grounded feature set (ideally a Biber MD dimension score), and (b) confront the residual within-SDG register structure (Step 2c) rather than only showing corpus-classifier collapse. At n=408 the confidence intervals are wide; these are direction-finding numbers, not confirmatory.

## Blockers / things that seemed off

1. **Adjusted embeddings are never stored** — everything projects on the fly from G. Not a blocker (G is small and `register_utils.project` is fast), but any downstream validation must reuse `register_utils.load_G`/`project`, not hunt for an `.npy` that doesn't exist.
2. **"Concept encoder" is a misnomer** — it is a corpus track (`research_concept/`) embedded with MPNet and has no register artifacts. Cannot be used as an independent encoder check.
3. **Policy `mean_sentence_length` is inflated by PDF-extraction junk** (63.8 words/sentence is implausible; segments like `"AAAA-cover.indd 1 10/08/2015 2:02:22 PM"` and banner text break sentence-splitting). The 2c result survives dropping this feature, but any appendix must pre-clean policy text or drop this feature.
4. **Policy `policy.jsonl` `id` is the source_doc, not the segment id** — matching must go through positional row index or `segment_id`. I verified the positional alignment, but it is a trap for scripts that join on `id`.
5. **Research `assigned_sdg` is per-segment, not per-paper** — the same paper can be assigned different SDGs across its segments. Sampling "documents" therefore means sampling segments; if a validation wants paper-level register scores it must aggregate segments, which is a separate design decision.
6. **Passive voice points the "wrong" way** (research > policy), consistent with abstract-writing conventions. Any register-score construction that assumes policy is simply "more formal" on every Biber feature will be mis-specified.
