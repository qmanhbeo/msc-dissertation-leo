"""
Validate the SDG centroid measurement instrument against the expert-labelled benchmark.

This script treats the 17 SDG centroids as a nearest-neighbour classifier and evaluates
how well they recover ground-truth SDG labels on the SDG Classification Benchmark (616 texts).
This is the only independent quality check available before running the full analysis. If the
instrument is too noisy, all downstream coverage/semantic gap findings are unreliable.

Evaluation design:
  All 17 SDGs (n = 616) are evaluated in a single held-out evaluation.
  Every centroid is built from corpora independent of the benchmark (OSDG, Knowledge Hub, SDGi),
  so no contamination occurs — the benchmark is genuinely independent for all 17 SDGs.

Interpretation guide (macro-F1 on SDGs 1–17):
  < 0.25   FAIL  — serious concern; consider domain-adapted model; flag in limitations
  0.25–0.50 WARN  — usable signal but moderate noise; acknowledge in methodology
  > 0.50   PASS  — good instrument; proceed with confidence

Note on expected accuracy:
  SDGs have substantial semantic overlap. Perfect separation is not expected. The centroid
  captures the *modal direction* of each SDG cluster, which still provides discriminative
  signal even when pairwise overlaps are high.

Outputs:
  4_outputs/main/data/4_1_validation_results.json         structured metrics + instrument flag
  4_outputs/main/data/4_1_confusion_matrix.csv            17x17 (rows = true SDG, cols = predicted SDG)
  4_outputs/main/data/4_1_centroid_similarity_matrix.csv  17x17 pairwise cosine similarities between centroids
  4_outputs/main/tables/*.tex                    generated LaTeX macros/tables

Run from project root (after sdg_centroids.py):
    python 1_code/2_embed/reference/2_validate_centroids.py
"""

import csv
import json
import logging
import numpy as np
import argparse
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from shared_utils import ensure_canonical_outputs
ANALYSIS_DIR = CODE_ROOT / "3_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_slug_utils import embed_dir_for_model, scored_dir_for_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = Path("4_outputs")

# Macro-F1 thresholds for the instrument pass/warn/fail flag (SDGs 1–16, uncontaminated).
# These are judgment calls (Assumption A-THRESH in the implementation plan):
#   - Random baseline = 1/17 ≈ 5.9%. THRESH_FAIL at 0.25 is ~4× above random — a minimal
#     bar for "better than noise." Any result below it means the embedding space does not
#     separate SDG semantics reliably enough for corpus-level analysis.
#   - THRESH_PASS at 0.50 is the conventional threshold for acceptable multi-class
#     classification. SDGs have genuine semantic overlap (H14: SDG 1 ↔ SDG 10 centroid
#     sim = 0.887), so 0.60+ would be an unrealistically high bar for this domain.
#   - Source: these thresholds were set before running validation (see plan file) to avoid
#     post-hoc adjustment. Actual result: macro-F1 = 0.733 → PASS.
THRESH_FAIL = 0.25
THRESH_PASS = 0.50

# Random baseline for a 17-class classifier with uniform class distribution.
# Actual benchmark classes are not perfectly uniform (n=27–50 per SDG), so this is an
# approximation. The true random baseline for a majority-class classifier would be
# max(class_count) / total ≈ 50/616 ≈ 8.1%, slightly higher. We use 1/17 as the
# conservative (lower) baseline.
RANDOM_BASELINE = 1 / 17   # ≈ 0.0588

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_json(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_csv_matrix(matrix: np.ndarray, labels: list, path: Path) -> None:
    """Save a square matrix as a CSV with row/column headers labelled SDG1..SDG17."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"SDG{l}" for l in labels])
        for i, row in enumerate(matrix):
            writer.writerow([f"SDG{labels[i]}"] + [f"{v:.4f}" for v in row])


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SDG centroids into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model", default="all-MiniLM-L6-v2", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    model = args.model
    embed_dir = embed_dir_for_model(model)
    scored_dir = scored_dir_for_model(model)
    EMBEDDINGS_DIR = embed_dir
    EMBED_METADATA_DIR = embed_dir / "metadata"
    SCORED_DIR = scored_dir
    SCORED_METADATA_DIR = scored_dir / "metadata"

    CENTROIDS_PATH   = SCORED_DIR / "sdg_centroids.npy"
    META_PATH        = SCORED_METADATA_DIR / "sdg_centroid_meta.json"
    BENCH_EMB        = EMBEDDINGS_DIR / "benchmark.npy"
    BENCH_IDS        = EMBED_METADATA_DIR / "benchmark_ids.json"

    layout = ensure_canonical_outputs(Path(args.output_dir))
    out_results = layout.data_dir / "4_1_validation_results.json"
    out_confusion = layout.data_dir / "4_1_confusion_matrix.csv"
    out_centroid_sim = layout.data_dir / "4_1_centroid_similarity_matrix.csv"
    tables_dir = layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    # ---- Load centroids ----
    log.info("Loading centroids: %s", CENTROIDS_PATH)
    centroids = np.load(CENTROIDS_PATH)   # (17, 384) float32, unit-normalised
    meta = load_json(META_PATH)           # list of 17 dicts from sdg_centroid_meta.json
    log.info("  shape=%s", centroids.shape)

    # Verify centroid normalisation before computing dot products.
    # ASSUMPTION: sdg_centroids.py saved unit-normalised centroids. If this fails, the
    # dot product below is NOT cosine similarity — it is a raw inner product biased toward
    # centroids with larger norms (i.e., tighter clusters would dominate purely because of
    # cohesion, not because of topical similarity). The analysis would be invalid.
    norms = np.linalg.norm(centroids, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        log.warning("Centroid norms not all ≈ 1.0 — dot product ≠ cosine sim: %s", norms)
    else:
        log.info("Centroid norms verified ≈ 1.0")

    # ---- Load benchmark embeddings ----
    log.info("Loading benchmark embeddings: %s", BENCH_EMB)
    bench_emb = np.load(BENCH_EMB)   # (616, 384) float32, unit-normalised
    bench_ids = load_json(BENCH_IDS)  # list of {id, text, sdg} with sdg in 1..17
    log.info("  shape=%s", bench_emb.shape)

    # Extract true SDG labels as an integer array for sklearn metric functions.
    true_sdgs = np.array([r["sdg"] for r in bench_ids], dtype=int)   # (616,)

    # ---- Nearest-centroid classification ----
    # ASSUMPTION: both bench_emb and centroids are L2-normalised unit vectors (verified above).
    # For unit vectors: dot(a, b) = cos(θ) = cosine similarity.
    # scores[i, j] = cosine similarity of benchmark text i to centroid for SDG (j+1).
    # This is the *same operation* applied in downstream scoring stages for papers and policy
    # segments — validating it here confirms the instrument before it is used at scale.
    scores = bench_emb @ centroids.T   # (616, 17)

    # Predicted SDG = the centroid with the highest cosine similarity.
    # argmax returns 0-indexed position (0..16); +1 converts to 1-indexed SDG label (1..17).
    # This must match the row ordering convention in sdg_centroids.npy:
    # centroids[i] = SDG (i+1), so argmax=0 → SDG 1, argmax=16 → SDG 17.
    pred_sdgs = scores.argmax(axis=1) + 1   # (616,) int, values 1..17

    # ---- Evaluation: All 17 SDGs, n = 616, no contamination ----
    # ASSUMPTION (A-SDG17): every centroid is built from corpora independent of the benchmark.
    # The benchmark is fully held out — no centroid uses benchmark texts. This evaluation is
    # genuinely independent for all 17 SDGs.
    labels_17 = list(range(1, 18))

    # labels= is passed explicitly so all 17 SDGs appear in the output arrays even if the
    # classifier never predicts a given class. Without this, sklearn would silently drop
    # classes with no predictions, making the per-SDG F1 array mis-indexed.
    # zero_division=0: return 0 (not a warning/error) for any SDG with zero true or predicted
    # instances. Possible for low-resource SDGs (e.g. SDG 12, n=43 in benchmark, could have
    # zero predictions if the classifier always prefers a neighbouring centroid).
    acc = float(accuracy_score(true_sdgs, pred_sdgs))
    mf1 = float(f1_score(true_sdgs, pred_sdgs, average="macro",
                         labels=labels_17, zero_division=0))
    per_sdg_f1 = f1_score(true_sdgs, pred_sdgs, average=None,
                          labels=labels_17, zero_division=0)   # shape (17,)

    # ---- Instrument flag ----
    if mf1 >= THRESH_PASS:
        flag = "PASS"
    elif mf1 >= THRESH_FAIL:
        flag = "WARN"
    else:
        flag = "FAIL"

    # ---- Inter-centroid similarity matrix ----
    # ASSUMPTION: centroids are unit-normalised (verified above), so this matrix product gives
    # pairwise cosine similarities. The diagonal will be ≈ 1.0 (each centroid with itself).
    # Off-diagonal values reveal semantic overlap between SDG concepts in SBERT space.
    # High inter-centroid similarity predicts classification confusion AND informs interpretation
    # of coverage gap results:
    #   - SDG 1 ↔ SDG 10 (0.887): these SDGs will be hard to distinguish (supports A26)
    #   - SDG 13 ↔ SDG 17 (0.860): "partnerships" language is close to "climate" (drives H35)
    #   - SDG 9 ↔ SDG 17 (0.813): "innovation" and "partnerships" overlap (context for H36)
    centroid_sim = centroids @ centroids.T   # (17, 17)

    # ---- Console output ----
    log.info("")
    log.info("=" * 60)
    log.info("CENTROID VALIDATION RESULTS (all 17 SDGs, n=%d, no contamination)", len(true_sdgs))
    log.info("=" * 60)
    log.info("")
    log.info("  Accuracy : %.4f  (random baseline: %.4f)", acc, RANDOM_BASELINE)
    log.info("  Macro-F1 : %.4f  → %s", mf1, flag)
    if flag == "FAIL":
        log.warning("  FAIL: Macro-F1 < %.2f — instrument too noisy for reliable analysis.", THRESH_FAIL)
        log.warning("         Consider domain-adapted model (e.g., Aurora-M, SDG-BERT).")
    elif flag == "WARN":
        log.warning("  WARN: Macro-F1 %.2f–%.2f — usable signal but moderate noise, acknowledge in methodology.",
                    THRESH_FAIL, THRESH_PASS)
    else:
        log.info("  PASS: Macro-F1 ≥ %.2f — instrument validated for analysis.", THRESH_PASS)

    log.info("")
    log.info("PER-SDG F1 (SDGs 1–17, all n=%d):", len(true_sdgs))
    log.info("  %-6s  %-8s  %-6s  %s", "SDG", "F1", "n_true", "variance_flag")
    log.info("  " + "-" * 45)
    for i, sdg in enumerate(labels_17):
        n_true = int((true_sdgs == sdg).sum())
        # meta is a 0-indexed list from sdg_centroid_meta.json.
        # meta[0] = SDG 1, meta[1] = SDG 2, ..., meta[16] = SDG 17.
        m = meta[sdg - 1]
        vflag = "[HIGH VAR]" if m["high_variance_flag"] else ""
        log.info("  SDG %2d   %.4f   n=%3d   %s", sdg, per_sdg_f1[i], n_true, vflag)

    # ---- Nearest centroid neighbours ----
    # These top-2 nearest neighbours per centroid predict the most common classification
    # errors and pre-register which SDG pairs are most likely to be confused in alignment
    # scoring. Useful for interpreting coverage gap results (e.g. if SDG 10 coverage
    # appears low, check whether SDG 1 and SDG 8 are absorbing its signal — see A26).
    log.info("")
    log.info("CENTROID NEAREST NEIGHBOURS (top-2, excluding self):")
    for i in range(17):
        sim_row = centroid_sim[i].copy()
        # Set self-similarity to -1 so argsort does not rank the centroid as its own
        # nearest neighbour. Cosine sim ranges -1..1, so -1 is the minimum possible value.
        sim_row[i] = -1.0
        top2_idx = np.argsort(sim_row)[::-1][:2]
        top2_str = ", ".join(f"SDG{j+1} ({centroid_sim[i,j]:.3f})" for j in top2_idx)
        log.info("  SDG %2d ← nearest: %s", i + 1, top2_str)

    # ---- Save outputs ----
    results = {
        "evaluation": {
            "n": len(true_sdgs),
            "accuracy": round(acc, 6),
            "macro_f1": round(mf1, 6),
            "contaminated": False,
            "note": (
                "All centroids built from corpora independent of the benchmark "
                "(OSDG, Knowledge Hub, SDGi). No contamination."
            ),
        },
        "per_sdg_f1": {str(sdg): round(float(per_sdg_f1[i]), 6)
                       for i, sdg in enumerate(labels_17)},
        "instrument_flag": flag,
        "random_baseline": round(RANDOM_BASELINE, 6),
        "thresholds": {"fail_below": THRESH_FAIL, "pass_above": THRESH_PASS},
    }

    with out_results.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("\nSaved: %s", out_results)

    # Confusion matrix over all 17 SDGs (rows = true, cols = predicted).
    # No contamination: all centroids built from corpora independent of the benchmark.
    # Examining off-diagonal mass reveals which SDG pairs are most confused — this directly
    # informs which per-SDG coverage gap findings should carry extra caveats (A6, A26).
    cm = confusion_matrix(true_sdgs, pred_sdgs, labels=labels_17)
    save_csv_matrix(cm.astype(float), labels_17, out_confusion)
    log.info("Saved: %s  (17×17, rows=true, cols=predicted)", out_confusion)

    # Centroid similarity matrix: how close each pair of SDG centroids is in SBERT space.
    # Used downstream to: (1) predict which SDG pairs will have leakage in coverage scoring,
    # (2) interpret H14 (SDG 1 ↔ SDG 10), (3) contextualise H35 (SDG 17 ↔ SDG 13),
    # (4) flag A26 (SDG 1-8-10 cluster collinearity).
    save_csv_matrix(centroid_sim, labels_17, out_centroid_sim)
    log.info("Saved: %s  (17×17 pairwise centroid cosine sim)", out_centroid_sim)

    log.info("\nNext step: run the active scoring path (shard scoring or bridge materialisation).")

    # ---- Write LaTeX generated outputs ----
    _sdg_num_words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
        15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
    }
    _sdg_names_all = {
        1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-Being",
        4: "Quality Education", 5: "Gender Equality",
        6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
        8: "Decent Work and Economic Growth",
        9: "Industry, Innovation and Infrastructure",
        10: "Reduced Inequalities", 11: "Sustainable Cities and Communities",
        12: "Responsible Consumption and Production", 13: "Climate Action",
        14: "Life Below Water", 15: "Life on Land",
        16: "Peace, Justice and Strong Institutions",
        17: "Partnerships for the Goals",
    }

    gen_dir = tables_dir

    # Per-SDG centroid similarities needed in dissertation prose.
    # SDG 11 (idx 10) vs SDG 9 (idx 8)
    sim_11_9 = float(centroid_sim[10, 8])
    # SDG 1/8/10 cluster pairwise sims (0-indexed: 0,7,9)
    cluster_sims = [
        float(centroid_sim[0, 7]),   # SDG1-SDG8
        float(centroid_sim[0, 9]),   # SDG1-SDG10
        float(centroid_sim[7, 9]),   # SDG8-SDG10
    ]

    # num_validation.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/2_embed/reference/2_validate_centroids.py — do not edit manually",
        rf"\newcommand{{\MacroFOne}}{{{mf1:.3f}}}",
        rf"\newcommand{{\ValidationAccuracy}}{{{acc:.3f}}}",
        rf"\newcommand{{\RandomBaselineSeventeenClass}}{{{RANDOM_BASELINE:.3f}}}",
        rf"\newcommand{{\ValidationVsRandomMultiple}}{{{(mf1 / RANDOM_BASELINE):.1f}}}",
        rf"\newcommand{{\CentroidSimThirteenSeventeen}}{{{centroid_sim[12, 16]:.3f}}}",
        rf"\newcommand{{\CentroidSimElevenNine}}{{{sim_11_9:.3f}}}",
        rf"\newcommand{{\CentroidSimOneEightTenMin}}{{{min(cluster_sims):.3f}}}",
        rf"\newcommand{{\CentroidSimOneEightTenMax}}{{{max(cluster_sims):.3f}}}",
    ]
    for i, sdg in enumerate(labels_17):
        word = _sdg_num_words[sdg]
        num_lines.append(rf"\newcommand{{\FiSdg{word}}}{{{per_sdg_f1[i]:.3f}}}")
    (gen_dir / "num_validation.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num_validation.tex")

    # tab_validation.tex — full tabular block
    tab_lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"SDG & Description & F1 \\",
        r"\midrule",
    ]
    for i, sdg in enumerate(labels_17):
        word = _sdg_num_words[sdg]
        name = _sdg_names_all[sdg]
        footnote = r"$^\ddagger$" if sdg == 4 else ""
        tab_lines.append(rf"SDG {sdg:2d} & {name}{footnote} & \FiSdg{word} \\")
    tab_lines.extend([
        r"\midrule",
        r"\multicolumn{2}{l}{Macro-F1 (SDGs 1--17)} & \textbf{\MacroFOne} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (gen_dir / "tab_validation.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab_validation.tex")


if __name__ == "__main__":
    main()
