"""
Build per-SDG centroid embeddings from the labelled corpora.

SDG centroids are the core measurement instrument for this dissertation. Every downstream
analysis (coverage gap, semantic gap, H25 correlation) scores texts against these centroids
via cosine similarity. Centroid quality must be validated before use — run validate_centroids.py
after this script.

Sources:
  SDGs 1–16 → OSDG Community Dataset (30,534 texts, agreement ≥ 0.5)
  SDG 17    → SDG Classification Benchmark (31 expert-labelled texts)
             Note: no OSDG texts exist for SDG 17; this is the only available source.

Normalisation:
  Input embeddings are L2-normalised (‖v‖ = 1). The mean of unit vectors has norm < 1
  (measured 0.47–0.59 across SDGs). We normalise each centroid to a unit vector before saving
  so that downstream dot products equal cosine similarities. The raw centroid norm (centroid
  cohesion) is preserved in metadata — it reflects intra-SDG cluster tightness.

Assumption flags (see notes/ASSUMPTIONS.md for full details):
  A3  — Equal weighting: each OSDG text is weighted equally regardless of agreement score.
  A6  — Centroid validity: cohesion < 0.50 is flagged, but cohesion alone is not the best
         risk indicator. Use per-SDG F1 from validate_centroids.py instead (see A6 revision).
  A-NORM — Centroid normalisation: raw centroid norm is recorded, unit centroid is saved.
  A-SDG17 — SDG 17 contamination: benchmark texts used for both building and validating
             the SDG-17 centroid. Primary validation in validate_centroids.py excludes SDG 17.
  A15 — OSDG circularity: not diagnosed here. Deferred to downstream alignment scoring steps.

Row ordering convention (critical for ALL downstream scripts):
  centroids[i] = centroid for SDG (i + 1)
  i.e., row 0 → SDG 1, row 1 → SDG 2, ..., row 16 → SDG 17

Inputs:
  data/2_embedded/osdg.npy           (30534, 384) float32, L2-normalised
  data/2_embedded/metadata/osdg_ids.json      list of {id, text, sdg} — sdg in 1..16
  data/2_embedded/benchmark.npy      (616, 384)   float32, L2-normalised
  data/2_embedded/metadata/benchmark_ids.json list of {id, text, sdg} — sdg in 1..17

Outputs:
  data/3_scored/sdg_centroids.npy      (17, 384) float32, unit-normalised
  data/3_scored/metadata/sdg_centroid_meta.json list of 17 dicts with per-SDG diagnostics

Run from project root:
    python code/2_embed/sdg_centroids.py
"""

import json
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = Path("data/2_embedded")
EMBED_METADATA_DIR = EMBEDDINGS_DIR / "metadata"
OUTPUT_DIR = Path("data/3_scored")
SCORED_METADATA_DIR = OUTPUT_DIR / "metadata"

OSDG_EMB   = EMBEDDINGS_DIR / "osdg.npy"
OSDG_IDS   = EMBED_METADATA_DIR / "osdg_ids.json"
BENCH_EMB  = EMBEDDINGS_DIR / "benchmark.npy"
BENCH_IDS  = EMBED_METADATA_DIR / "benchmark_ids.json"

OUT_CENTROIDS = OUTPUT_DIR / "sdg_centroids.npy"
OUT_META      = SCORED_METADATA_DIR / "sdg_centroid_meta.json"

# Cohesion threshold for the high-variance flag (Assumption A6).
# "Cohesion" = mean cosine similarity of an SDG's member vectors to its own unit centroid.
# A value of 0.50 was chosen as a round number below the observed range for 'normal' SDGs
# (0.50–0.59); SDG 8 (0.486) and SDG 16 (0.477) fall below it.
# IMPORTANT: cohesion alone is not the authoritative risk indicator. After running
# validate_centroids.py, use per-SDG F1 < 0.55 as the operational threshold instead.
# Rationale: a cluster can be internally diffuse yet still discriminable from other clusters
# (SDG 16 was flagged here but achieved F1=0.857 at validation). See A6 in ASSUMPTIONS.md.
COHESION_WARN_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_centroid(emb: np.ndarray, idxs: list[int], sdg: int, source: str) -> tuple:
    """
    Compute a unit-normalised centroid for one SDG and return diagnostic metadata.

    Args:
        emb:    Full embedding matrix (all texts), shape (N, 384), L2-normalised.
        idxs:   Row indices into `emb` that belong to this SDG.
        sdg:    SDG number (1–17), used only for error messages and metadata.
        source: "osdg" or "benchmark" — which corpus supplied these texts.

    Returns:
        (unit_centroid, meta_dict)
        unit_centroid: (384,) float32 unit vector — the normalised direction of the mean.
        meta_dict: diagnostic fields including raw_centroid_norm and mean_cos_to_centroid.
    """
    vecs = emb[idxs]   # (n, 384) — subset of L2-normalised embeddings for this SDG

    # ASSUMPTION (A3 — equal weighting): every text in `idxs` contributes equally to the mean.
    # We do NOT weight by annotator agreement score, even though OSDG provides it.
    # Rationale: texts were already pre-filtered at agreement ≥ 0.5 in preprocess_osdg.py,
    # so minimum reliability is ensured. The agreement scores above that floor are noisy
    # estimates from ~9 annotators, and differential weighting would add complexity without
    # a principled basis for the weight function. The filtering step is the cleaner mitigation.
    raw = vecs.mean(axis=0)   # (384,) — arithmetic mean; NOT a unit vector

    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        # Near-zero norm would mean the embeddings are near-uniformly distributed on the sphere
        # (every direction cancels out). This should never happen with a real SDG corpus; if it
        # does, the embedding data or SDG labels are corrupt.
        raise ValueError(f"SDG {sdg}: near-zero centroid norm — check embedding data")

    # ASSUMPTION (A-NORM — centroid normalisation): we divide by the norm to produce a unit
    # vector. This gives us the *direction* of the mean, not the mean itself. The two are
    # mathematically distinct: the unit centroid is the spherical mean (Fréchet mean on S^(d-1)).
    # Why normalise? Because all input embeddings are unit vectors and downstream scripts use
    # dot product for cosine similarity. A non-unit centroid would require dividing by its norm
    # at every similarity computation — error-prone and easy to forget. Normalising once here
    # eliminates that risk. The information lost (cluster spread / cohesion) is preserved in
    # `raw_centroid_norm` below, so nothing is truly discarded.
    unit = (raw / norm).astype(np.float32)

    # Cohesion diagnostic: mean cosine similarity of member vectors to the unit centroid.
    # ASSUMPTION (embeddings.py): input vecs are L2-normalised, so dot product = cosine sim.
    # If embeddings were not normalised, `vecs @ unit` would give inner products, not cosine
    # similarities — and this metric would be meaningless.
    # The mean cosine similarity equals `norm` for unit input vectors:
    #   mean(vecs @ unit) = (sum of vecs) @ unit / n = raw @ unit / n = norm * unit @ unit / n
    #   = norm / n * n = norm  ... so raw_centroid_norm IS the cohesion for unit inputs.
    # We compute it directly anyway for clarity and to future-proof against non-unit inputs.
    mean_cos = float((vecs @ unit).mean())

    # Flag if cohesion is below the warning threshold (see COHESION_WARN_THRESHOLD comments).
    # This flag is informational only — do not skip or discard flagged centroids.
    high_variance = mean_cos < COHESION_WARN_THRESHOLD

    meta = {
        "sdg": sdg,
        "n": len(idxs),
        "source": source,
        # raw_centroid_norm: the norm of the un-normalised centroid.
        # Ranges 0–1 for unit inputs. Higher = tighter cluster = more cohesive SDG.
        # Equivalent to mean_cos_to_centroid for unit input vectors (see derivation above).
        "raw_centroid_norm": round(norm, 6),
        "mean_cos_to_centroid": round(mean_cos, 6),
        "high_variance_flag": high_variance,
    }
    return unit, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load embeddings and ID metadata ----
    log.info("Loading OSDG embeddings: %s", OSDG_EMB)
    osdg_emb = np.load(OSDG_EMB)   # (30534, 384) float32 — produced by code/2_embed/embeddings.py
    osdg_ids = load_json(OSDG_IDS)  # list of {id, text, sdg} with sdg in 1..16
    log.info("  shape=%s  dtype=%s", osdg_emb.shape, osdg_emb.dtype)

    log.info("Loading benchmark embeddings: %s", BENCH_EMB)
    bench_emb = np.load(BENCH_EMB)   # (616, 384) float32 — SDGs 1..17 (including 17)
    bench_ids = load_json(BENCH_IDS)  # list of {id, text, sdg}
    log.info("  shape=%s  dtype=%s", bench_emb.shape, bench_emb.dtype)

    # ---- Verify L2 normalisation ----
    # ASSUMPTION (embeddings.py): all embeddings were produced with normalize_embeddings=True
    # in SentenceTransformer.encode(). If this is violated, build_centroid's cohesion metric
    # and all downstream dot-product-as-cosine-similarity computations will be wrong.
    # We check a sample (not all 30K rows) for speed; the sample is sufficient to catch
    # a systematic normalisation failure.
    sample_norms = np.linalg.norm(osdg_emb[:20], axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-4):
        log.warning("OSDG embeddings may not be L2-normalised (norms: %s)", sample_norms[:5])
    else:
        log.info("Embedding norms verified ≈ 1.0 (L2-normalised)")

    # ---- Build per-SDG index maps ----
    # Group row indices by SDG label so build_centroid can slice the embedding matrix.
    osdg_by_sdg: dict[int, list[int]] = {}
    for i, r in enumerate(osdg_ids):
        sdg = r["sdg"]
        osdg_by_sdg.setdefault(sdg, []).append(i)

    bench_by_sdg: dict[int, list[int]] = {}
    for i, r in enumerate(bench_ids):
        sdg = r["sdg"]
        bench_by_sdg.setdefault(sdg, []).append(i)

    # Sanity check: OSDG must cover exactly SDGs 1–16 (no 17, none missing).
    # OSDG was filtered from a version that lacks SDG 17 labels. If this fails, the
    # data preprocessing step (preprocess_osdg.py) produced unexpected output.
    osdg_sdgs = sorted(osdg_by_sdg.keys())
    if osdg_sdgs != list(range(1, 17)):
        log.warning("Unexpected OSDG SDG labels: %s", osdg_sdgs)
    else:
        log.info("OSDG SDG coverage confirmed: 1–16")

    # ASSUMPTION (A-SDG17 — SDG 17 source): OSDG has no SDG 17 labels, so we must source
    # the SDG-17 centroid from the benchmark corpus (31 expert-labelled texts).
    # Consequence: the SDG-17 centroid is built from a different corpus and annotation process
    # than SDGs 1–16. Its style may lean academic (benchmark texts are from academic/policy
    # abstracts) rather than policy-style (OSDG texts). This could make the SDG-17 centroid
    # slightly less representative of policy framing than the other centroids.
    # Additionally: the same 31 texts are later used to evaluate the SDG-17 centroid in
    # validate_centroids.py — a contamination that inflates SDG-17 classification accuracy.
    # Mitigation: validate_centroids.py reports SDG-17 accuracy as an upper bound only and
    # uses SDGs 1–16 for the primary (uncontaminated) macro-F1 score.
    if 17 not in bench_by_sdg:
        raise RuntimeError("No SDG-17 texts found in benchmark — cannot build SDG-17 centroid")
    log.info("Benchmark SDG-17 pool: n=%d texts", len(bench_by_sdg[17]))

    # ---- Build centroids ----
    log.info("")
    log.info("Building centroids...")
    log.info("%-8s %-7s %-7s %-10s %-10s %s",
             "SDG", "n", "source", "raw_norm", "cohesion", "variance_flag")
    log.info("-" * 60)

    centroid_vectors = []
    centroid_meta = []

    # SDGs 1–16: source from OSDG (30,534 texts total, varying counts per SDG)
    for sdg in range(1, 17):
        idxs = osdg_by_sdg[sdg]
        vec, meta = build_centroid(osdg_emb, idxs, sdg, source="osdg")
        centroid_vectors.append(vec)
        centroid_meta.append(meta)

        flag = " [HIGH VARIANCE — A6 risk]" if meta["high_variance_flag"] else ""
        level = logging.WARNING if meta["high_variance_flag"] else logging.INFO
        log.log(level, "SDG %2d | n=%5d | osdg      | norm=%.4f | cohesion=%.4f%s",
                sdg, meta["n"], meta["raw_centroid_norm"], meta["mean_cos_to_centroid"], flag)

    # SDG 17: source from benchmark (see A-SDG17 comment above)
    sdg17_idxs = bench_by_sdg[17]
    vec17, meta17 = build_centroid(bench_emb, sdg17_idxs, 17, source="benchmark")
    centroid_vectors.append(vec17)
    centroid_meta.append(meta17)

    flag = " [HIGH VARIANCE — A6 risk]" if meta17["high_variance_flag"] else ""
    level = logging.WARNING if meta17["high_variance_flag"] else logging.INFO
    log.log(level,
            "SDG 17 | n=%5d | benchmark | norm=%.4f | cohesion=%.4f%s  "
            "[NOTE: centroid from benchmark; validate_centroids.py SDG-17 results are contaminated]",
            meta17["n"], meta17["raw_centroid_norm"], meta17["mean_cos_to_centroid"], flag)

    # ---- Stack into (17, 384) array ----
    # ROW ORDERING CONVENTION: centroids[i] = centroid for SDG (i+1).
    # Row 0 → SDG 1, row 1 → SDG 2, ..., row 16 → SDG 17.
    # This convention is assumed by every downstream script that loads sdg_centroids.npy.
    # Changing it here without updating downstream scoring/analysis scripts, including coverage_gap.py, and
    # semantic_gap.py would silently corrupt all downstream SDG assignments.
    centroids = np.stack(centroid_vectors, axis=0)  # (17, 384)
    assert centroids.shape == (17, 384), f"Unexpected centroid shape: {centroids.shape}"

    # Final normalisation check — each row should be a unit vector after build_centroid.
    norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        log.warning("Post-normalisation centroid norms not all ≈ 1.0: %s", norms)
    else:
        log.info("\nAll 17 centroid norms ≈ 1.0 ✓")

    # ---- Save ----
    np.save(OUT_CENTROIDS, centroids)
    log.info("Saved: %s  shape=%s  dtype=%s", OUT_CENTROIDS, centroids.shape, centroids.dtype)

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(centroid_meta, f, indent=2)
    log.info("Saved: %s", OUT_META)

    # ---- Summary ----
    high_var = [m["sdg"] for m in centroid_meta if m["high_variance_flag"]]
    if high_var:
        log.warning(
            "\nHigh-variance SDGs (cohesion < %.2f): %s\n"
            "  → These SDG centroids are internally diffuse but may still be discriminable\n"
            "    from other centroids (see A6 in ASSUMPTIONS.md). Run validate_centroids.py\n"
            "    and use per-SDG F1 < 0.55 as the authoritative risk threshold.",
            COHESION_WARN_THRESHOLD, high_var
        )
    else:
        log.info("\nNo high-variance SDGs detected at threshold %.2f", COHESION_WARN_THRESHOLD)

    log.info(
        "\nRow ordering: centroids[i] = SDG (i+1)  "
        "(row 0 → SDG 1, row 16 → SDG 17)\n"
        "Next step: python code/2_embed/validate_centroids.py"
    )


if __name__ == "__main__":
    main()
