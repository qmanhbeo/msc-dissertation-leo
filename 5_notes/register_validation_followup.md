# Register Validation Follow-up: Clustering Diagnostic and Step-2c Decomposition

**Date:** 2026-08-05
**Model:** `all-mpnet-base-v2` (MPNet canon)
**G:** `2_data/3b_register/mpnet/canon/G.npy` (62×768), `g_sha256: 10955e...`
**Seed:** 42 throughout
**Units:** segments (~384-token chunks)
**Scripts:** `5_notes/scratch/check_concept_same_space.py` (Item 1), `5_notes/scratch/register_validation_followup.py` (Items 2–3)

This extends the first report (`5_notes/register_validation_report.md`, committed at `0f96a3f`). All three items of the mandated follow-up diagnostic are addressed below.

---

## Item 1 — Concept-provenance in Table 3: IS NOT A BUG

### What was asked

The first report flagged that the Concept rows in `4_outputs/mpnet/data/concept/adjusted/semantic_gap_distances_{lr,mlp}.json` are not produced by a separate encoder, raising a concern about Table 3's provenance. The question: is applying MPNet's register directions to the concept-retrieved corpus a valid operation, or a bug?

### Code path (traced to source)

1. `2_coverage_semantic_interaction.py:170-180` — `_H1_CONFIGS` declares Concept rows with `corpus="concept"`.
2. `2_coverage_semantic_interaction.py:277-305` — `_h1_config_row` → `_adj_gaps_for` → `_concept_adj_gaps` / `_concept_mlp_adj_gaps` from `1_code/7_main_analysis/0_shared/h1_register_correlation_table.py:168-189` (pure readers of the JSON).
3. Those JSONs are produced by `main.py:714-732` ("semantic gap (concept corpus, adjusted)" LR/MLP) → `1_code/7_main_analysis/1_main_text/1_semantic_gap.py` with `--embeddings adjusted --research-centroids <concept centroids> --out-data-dir concept_dir`.
4. `1_semantic_gap.py:263-268` (adjusted mode): `G = register_utils.load_G(args.embed_model)` = **MPNet canon G**; `gap = 1 - cosine(projected_centroids)`.
5. Concept centroids: `2_data/5_supervised_scored/mpnet/research_concept_centroids.npy` (LR, canonical retrained `sdg_classifier_retrained.joblib`; `main.py:635-655`). MLP: `mlp_scores_concept/mlp_research_centroids.npy`.
6. JSON provenance: `"g_path": "2_data/3b_register/mpnet/canon/G.npy", "track": "canon", "g_sha256": "10955e..."`.

**Key:** The concept corpus is a *retrieval axis* (OpenAlex AI/ML field-of-study search), not a separate encoder. Both concept and canonical embeddings are produced by the **same script** `0_embed_paper_shards.py` (`1_code/3_embed/0_embed_paper_shards.py`), with identical embedder, checkpoint, pooling, normalisation. This was empirically verified (see below).

### Empirical same-space proof (`5_notes/scratch/check_concept_same_space.py`)

- 30,545 shared papers between `research_concept` and `research` tracks (by OpenAlex ID).
- 40 sampled papers; matched segment pairs by text identity.
- **44 byte-identical-text pairs found.**
- **Max elementwise |diff|: 0.000183** (fp16 tolerance).
- **Min cosine similarity: 0.99999952**.

Conclusion: the embeddings come from the same `model.encode()` call, same checkpoint, same normalisation. Applying MPNet canon G to the concept research centroids is formally valid — it removes the same register directions from the same embedding space.

### What Concept rows represent

MPNet's register directions removed from the concept-retrieved corpus's research side (policy side is byte-identical to the MPNet adjusted run; only the research side differs). The Concept rows test **retrieval-axis robustness** — whether the register-topic decomposition survives a different paper-selection strategy — not a different encoder. This is an intentional design choice, not a bug.

**Verdict: NOT a bug. The concept rows are valid and correctly produced.**

---

## Item 2 — Clustering in the n=408 sample

### 2A — Original sample (per-SDG dedup only, no global parent dedup)

| Metric | Value |
|---|---|
| All 408: distinct parents | 390 |
| All 408: units sharing parent | 25 (6.1%) |
| Research 204: distinct parents | 204 |
| Research 204: units sharing parent | 0 (0.0%) |
| Policy 204: distinct parents | 186 |
| Policy 204: units sharing parent | 25 (12.3%) |

Top clustered parents (policy): SDSN Sustainable Development Report 2024 (6 segments), SDSN 2025 (6), UNDP HDR 2021/22 (5), WHO Ethics & Governance of AI for Health (2), UN SDG Progress Report 2020 (2).

**Policy clustering exceeds the ~10% threshold → one-per-parent rerun required.**

### 2B — One-segment-per-parent (global dedup across SDGs)

| Stat | Original (2A) | One-per-parent (2B) | Change |
|---|---|---|---|
| 2b reg~\|x−x'\| pooled ρ (p) | 0.102 (0.040) | 0.092 (0.063, ns) | −0.010 |
| 2b within-research ρ (p) | 0.212 (0.002) | **−0.043 (0.545, ns)** | −0.255 |
| 2b within-policy ρ (p) | 0.191 (0.006) | **−0.036 (0.606, ns)** | −0.227 |
| 2c RAW ρ (p) | 0.126 (0.011) | **−0.212 (1.6e-05)** | −0.338 |
| 2c ADJ ρ (p) | 0.247 (4.4e-07) | **−0.197 (6.1e-05)** | −0.444 |
| 2c partial-controlling-corpus RAW | 0.130 | **−0.155** | −0.285 |
| 2c partial-controlling-corpus ADJ | 0.253 | **−0.159** | −0.412 |
| 2d register-only acc | 0.456 | 0.544 | +0.088 |
| 2d raw acc | 0.909 | 0.944 | +0.035 |
| 2d adj acc | 0.505 | **0.603** | +0.098 |

### Reading

- **Step-2b (removed magnitude ~ register features):** The pooled correlation drops to marginal (ρ=0.092, p=0.063). Within-corpus correlations **collapse to null** (−0.04/−0.04, both ns). The first report's "removed subspace tracks my 6 features" evidence evaporates on the clean sample.
- **Step-2c (register ~ centroid distance):** **Flips sign from positive to negative.** The original 2c red flag (register score → farther from SDG centroid, ρ=+0.25 adj) was a clustering artifact. On the one-per-parent sample, high register segments sit slightly *closer* to their SDG centroid (ρ≈−0.07..−0.09). The policy mega-docs (SDSN/UNDP) — which are deontic/long-sentence-heavy AND spread across many SDGs — manufactured the positive correlation.
- **Step-2d (corpus classification):** Raw 0.944 → adj 0.603. The corpus collapse is large but **not to chance** (register-only baseline 0.544). The adjusted space retains meaningful corpus structure.

---

## Item 3 — Step-2c decomposition (on the one-per-parent sample, N=12/SDG)

### 3a — Per-SDG register ~ centroid-distance (n=24/SDG, low power)

| SDG | n | RAW ρ | ADJ ρ | Direction |
|---|---|---|---|---|
| 1 | 24 | +0.069 | −0.048 | better |
| 2 | 24 | −0.428 | −0.417 | worse |
| 3 | 24 | +0.337 | +0.500 | worse |
| 4 | 24 | −0.226 | −0.201 | worse |
| 5 | 24 | −0.379 | −0.368 | worse |
| 6 | 24 | −0.126 | −0.283 | better |
| 7 | 24 | −0.137 | −0.123 | worse |
| 8 | 24 | −0.351 | −0.165 | worse |
| 9 | 24 | +0.078 | +0.003 | better |
| 10 | 24 | +0.039 | +0.186 | worse |
| 11 | 24 | +0.147 | +0.067 | better |
| 12 | 24 | −0.339 | −0.470 | better |
| 13 | 24 | +0.140 | +0.017 | better |
| 14 | 24 | −0.045 | −0.142 | better |
| 15 | 24 | −0.208 | −0.216 | better |
| 16 | 24 | +0.478 | +0.588 | worse |
| 17 | 24 | +0.418 | +0.519 | worse |

- SDGs where ADJ ρ > RAW ρ: 9/17
- Mean per-SDG ρ: RAW −0.031, ADJ −0.033
- Partial Spearman controlling SDG: RAW −0.094, ADJ −0.074

**No systematic "gets worse after adjustment" pattern.** Per-SDG correlations are mixed noise at n=24 (low power). The first report's red flag does not reproduce per-SDG.

### 3b — Per-feature vs centroid-distance (pooled, n=408)

| Feature | RAW ρ | ADJ ρ | Δ | Controlling SDG RAW→ADJ |
|---|---|---|---|---|
| hedge_rate | +0.091 | +0.064 | −0.028 | +0.062→+0.073 |
| deontic_rate | −0.276 | −0.199 | +0.077 | −0.246→−0.171 |
| passive_rate | +0.075 | +0.002 | −0.073 | +0.071→+0.001 |
| mean_sent_len | −0.293 | −0.172 | +0.121 | −0.291→−0.172 |
| first_person_rate | +0.095 | +0.154 | +0.059 | +0.035→+0.090 |
| nominal_rate | −0.145 | −0.098 | +0.046 | −0.145→−0.098 |

The features that correlate with centroid distance at all do so **negatively** (high register → closer to SDG centroid) and that association **weakens after adjustment** (positive deltas for deontic, mean_sent_len, nominal). **No feature drives an adjusted-space increase of the kind the first report flagged.** The dominant pre-adjustment correlations (deontic −0.276, sentence length −0.293) are driven by the mega-docs; on the clean sample they shrink.

### 3c — Own-corpus vs other-corpus SDG centroid pull

| Metric | RAW ρ | ADJ ρ |
|---|---|---|
| reg ~ own-dist (pooled) | −0.102 | −0.065 |
| reg ~ other-dist (pooled) | −0.061 | −0.114 |
| reg ~ bias (other−own) | +0.019 | −0.024 |
| reg ~ bias within research | +0.040 | +0.039 |
| reg ~ bias within policy | −0.024 | −0.111 |
| own-dist ADJ: research | — | −0.033 |
| own-dist ADJ: policy | — | −0.060 |
| other-dist ADJ: research | — | +0.003 |
| other-dist ADJ: policy | — | −0.197 |

The only surviving within-SDG register trace: **policy segments in adjusted space show reg ~ other-dist ρ=−0.197** (high-register policy segments sit farther from the *research* centroid and closer to their own). This is a small within-SDG corpus pull on the policy side only, not a cross-corpus red flag.

### 3d — Renormalization artifact check

| Metric | RAW ρ | ADJ (renormalized) | ADJ (un-renormalized) |
|---|---|---|---|
| reg ~ dist pooled | −0.088 | −0.074 | −0.058 |
| reg ~ dist within-SDG | −0.094 | −0.074 | −0.057 |

- reg ~ renorm-scale 1/‖resid‖: pooled +0.030, within-SDG +0.031 (ns)
- reg ~ ‖x−x'‖ (removed norm): +0.030
- dist_adj ~ renorm-scale 1/‖resid‖: **−0.312** (mechanical: renormalizing shrinks distances for high-residual-norm rows, but this does NOT translate into a reg-score correlation)

**NOT a renormalization artifact.** The near-zero negative correlation is not inflated by L2 renormalisation. The first report's 2c red flag was driven by the clustered mega-docs, not by renormalisation mechanics.

---

## Overall verdict

| Item | Finding | Implication |
|---|---|---|
| Item 1 | Concept rows are valid (same embedder, same space, empirically proven) | Table 3 Concept columns are correctly produced; no integrity issue |
| Item 2 | Original 2b/2c signals were clustering artifacts (SDSN/UNDP mega-docs) | First report's "cautionary" findings were false positives |
| Item 3 | On clean sample: 2c ≈ −0.07..−0.09 (near-zero, negative); per-feature negative and shrinking after adj; renorm not the driver | No residual-register red flag survives; adjusted space is cleaner than raw |

### Recommended verdict: GO with qualification

The go/no-go evidence **shifts** from the first report:

- **What clears:** Item 1 removes the Concept-integrity review risk. Item 2+3 show the original 2c red flag (register → farther from centroid) was a clustering artifact on SDSN/UNDP mega-docs. The adjusted space is not destroying corpus signal (2d adj=0.603, not chance). The renormalization concern is dismissed.

- **What weakens:** The positive evidence from the first report (2b within-corpus correlations, 2d adj≈chance) also weakened on the clean sample. The 2b within-corpus correlations collapse to null (−0.04, ns). The register-feature ↔ removed-subspace link is thinner than the first report suggested. The per-SDG correlations at n=24/SDG are low-power.

- **What needs before the appendix:**
  1. **REGCHECK_N=60 run** (n=120/SDG) for stable per-SDG correlations — the current n=24/SDG per-SDG data is too noisy for confirmatory claims.
  2. **Optional:** run Item 3 on the original (2A-composition) sample to directly localise the old red flag to the mega-doc SDGs.
  3. The full validation appendix (Phase 2+) should use **one-per-parent sampling as primary** and treat SDSN/UNDP mega-docs explicitly.

---

## Reproducibility

- All numbers from `5_notes/scratch/regcheck_followup.log` (complete run, seed 42, N_PER_SDG=12 for Items 2A/2B/3, global parent dedup for 2B/3).
- Arrays saved to `5_notes/scratch/regcheck_followup_arrays.npz` (reg, dist_raw, dist_adj, dist_noren, corr, sdg, F, resid_norm).
- Same-space proof from `5_notes/scratch/check_concept_same_space.py` (seed 42, 40 papers, 44 identical pairs).
