"""
Register-adjustment sensitivity analysis for the semantic gap.

Learns one global research-vs-policy direction in embedding space via binary
logistic regression, subtracts its projection from all embeddings, and
recomputes within-SDG semantic gaps using the adjusted centroids.

Raw gaps are loaded from the canonical pipeline output
(4_3_semantic_gap_distances.json) so the "raw" column exactly matches the
main manuscript.  Adjusted gaps are recomputed under the same segment-cap
and random-seed settings.

Outputs:
  4_outputs/appendix/f_register_adjustment/tables/tab_register_adjusted_semgap.tex
  4_outputs/appendix/f_register_adjustment/tables/num_register_adjustment.tex

Run from project root:
    python 1_code/7_main_analysis/3_appendix/f_register_adjustment.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    N_SDG,
    SDG_NAMES,
    embed_dir_for_model,
    scored_dir_for_model,
)
import semantic_gap_shared
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    MIN_CLUSTER_SIZE,
    RANDOM_SEED as POLICY_SEGMENT_CAP_SEED,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    get_cluster_assignments,
    load_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAMPLE_SIZE_PER_CLASS = 40_000

CANONICAL_SEMANTIC_JSON = ROOT / "4_outputs" / "main" / "data" / "4_3_semantic_gap_distances.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run register-adjustment sensitivity analysis.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return p.parse_args()


def load_research_sample(
    model: str, n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Load a random sample of research embeddings from shards."""
    shard_dir = embed_dir_for_model(model) / "research_shards"
    shard_files = sorted(shard_dir.glob("part-*.npy"))
    if not shard_files:
        raise FileNotFoundError(f"No research shard files in {shard_dir}")

    n_per_shard = max(1, n_samples // len(shard_files))
    samples: list[np.ndarray] = []
    total = 0

    for shard_path in shard_files:
        needed = n_samples - total
        if needed <= 0:
            break
        n_take = min(n_per_shard, needed)
        emb = np.load(shard_path, mmap_mode="r")
        n_rows = emb.shape[0]
        if n_rows == 0:
            continue
        n_take = min(n_take, n_rows)
        idx = rng.choice(n_rows, size=n_take, replace=False)
        samples.append(emb[idx].copy())
        total += n_take
        log.info("  Sampled %d from %s (total %d)", n_take, shard_path.name, total)

    if total == 0:
        raise RuntimeError("Could not load any research embeddings")
    return np.concatenate(samples, axis=0).astype(np.float32)


def subtract_direction(emb: np.ndarray, g_dir: np.ndarray) -> np.ndarray:
    proj = np.dot(emb, g_dir)[:, np.newaxis] * g_dir
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def main() -> None:
    args = parse_args()
    model = args.model
    rng = np.random.default_rng(POLICY_SEGMENT_CAP_SEED)

    # ------------------------------------------------------------------
    # 1. Load canonical raw gaps
    # ------------------------------------------------------------------
    log.info("Loading canonical raw gaps from %s", CANONICAL_SEMANTIC_JSON)
    canonical = load_json(CANONICAL_SEMANTIC_JSON)
    canonical_raw = {}
    for entry in canonical["per_sdg"]:
        sdg = entry["sdg"]
        gap = entry.get("semantic_gap")
        canonical_raw[sdg] = gap
    n_loaded = len(canonical_raw)
    log.info("  Loaded %d raw gap values", n_loaded)

    # Sanity check — the canonical raw gaps must be complete and valid
    assert n_loaded == N_SDG, (
        f"Expected {N_SDG} canonical gap values, got {n_loaded}"
    )
    assert all(0 < v < 1 for v in canonical_raw.values()), (
        f"Canonical gaps not in (0,1): { {k: v for k, v in canonical_raw.items() if not 0 < v < 1} }"
    )

    if canonical["segment_cap"] != SEGMENT_CAP_PRIMARY:
        log.warning(
            "Canonical segment cap (%d) differs from expected (%d) — data may be stale",
            canonical["segment_cap"], SEGMENT_CAP_PRIMARY,
        )

    # ------------------------------------------------------------------
    # 2. Load data using shared loaders
    # ------------------------------------------------------------------
    log.info("Loading policy.npy ...")
    policy_emb = np.load(semantic_gap_shared.get_policy_emb(model)).astype(np.float32)
    n_policy = policy_emb.shape[0]
    log.info("  policy.npy: %s", policy_emb.shape)

    log.info("Loading policy scores & IDs ...")
    policy_scores = np.load(semantic_gap_shared.get_policy_scores(model))
    policy_ids = load_json(semantic_gap_shared.get_policy_ids(model))
    policy_assignments = get_cluster_assignments(policy_scores)

    log.info("Loading research centroids & meta ...")
    research_centroids = np.load(semantic_gap_shared.get_research_centroids(model)).astype(np.float32)
    research_meta = load_json(semantic_gap_shared.get_research_centroid_meta(model))
    research_cohesions = np.array([float(r["mean_cos_to_centroid"]) for r in research_meta], dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Train binary research-vs-policy classifier
    # ------------------------------------------------------------------
    log.info("Sampling %d research embeddings ...", SAMPLE_SIZE_PER_CLASS)
    research_sample = load_research_sample(model, SAMPLE_SIZE_PER_CLASS, rng)
    log.info("  research sample shape: %s", research_sample.shape)

    n_train = min(SAMPLE_SIZE_PER_CLASS, n_policy)
    policy_sample_idx = rng.choice(n_policy, size=n_train, replace=False)
    policy_sample = policy_emb[policy_sample_idx].copy()

    X_train = np.concatenate([research_sample[:n_train], policy_sample], axis=0)
    if X_train.shape[0] < 2 * n_train:
        log.warning("research_sample has fewer than n_train rows; using all available.")
        X_train = np.concatenate([research_sample, policy_sample], axis=0)
    y_train = np.concatenate([
        np.zeros(len(research_sample[:n_train]), dtype=np.int32),
        np.ones(len(policy_sample), dtype=np.int32),
    ])

    log.info("Training binary LR (research=0, policy=1) on %d samples ...", len(y_train))
    clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    log.info("  Training accuracy: %.4f", train_acc)

    coef = clf.coef_.astype(np.float32).flatten()
    coef_norm = float(np.linalg.norm(coef))
    g = (coef / coef_norm).astype(np.float32) if coef_norm > 1e-12 else coef
    log.info("  Learned direction norm: %.6f", coef_norm)

    # ------------------------------------------------------------------
    # 4. Adjust all policy embeddings
    # ------------------------------------------------------------------
    log.info("Adjusting policy embeddings ...")
    policy_adj = subtract_direction(policy_emb, g)

    # ------------------------------------------------------------------
    # 5. Compute adjusted policy centroids per SDG (segment-capped)
    # ------------------------------------------------------------------
    log.info("Computing adjusted policy centroids (segment cap = %d) ...", SEGMENT_CAP_PRIMARY)
    adj_pol_centroids = np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float32)
    adj_pol_cohesions = np.zeros(N_SDG, dtype=np.float32)

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]
        if not policy_idxs:
            adj_pol_centroids[sdg_idx] = 0
            adj_pol_cohesions[sdg_idx] = 0.0
            continue

        idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        centroid, cohesion = build_sub_centroid(policy_adj, idxs_capped)
        if centroid is not None:
            adj_pol_centroids[sdg_idx] = centroid
            adj_pol_cohesions[sdg_idx] = cohesion
        else:
            adj_pol_centroids[sdg_idx] = 0
            adj_pol_cohesions[sdg_idx] = 0.0

        log.info(
            "  SDG %2d: policy idxs %4d → %4d (capped), cohesion %.4f",
            sdg, len(policy_idxs), len(idxs_capped), adj_pol_cohesions[sdg_idx],
        )

    # ------------------------------------------------------------------
    # 6. Compute adjusted research centroids per SDG
    # ------------------------------------------------------------------
    log.info("Adjusting research centroids ...")
    adj_res_centroids = np.zeros((N_SDG, research_centroids.shape[1]), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        raw_mean = research_centroids[sdg_idx] * research_cohesions[sdg_idx]
        proj = float(np.dot(raw_mean, g)) * g
        adj_raw = raw_mean - proj
        norm = float(np.linalg.norm(adj_raw))
        if norm > 1e-8:
            adj_res_centroids[sdg_idx] = (adj_raw / norm).astype(np.float32)
        else:
            adj_res_centroids[sdg_idx] = research_centroids[sdg_idx]

    # ------------------------------------------------------------------
    # 7. Compute adjusted gaps — compare against canonical raw gaps
    # ------------------------------------------------------------------
    log.info("Computing adjusted semantic gaps ...")
    results: list[dict] = []
    raw_gaps_list: list[float] = []
    adj_gaps_list: list[float] = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        canonical_val = canonical_raw.get(sdg)
        if canonical_val is None:
            log.warning("SDG %d: canonical raw gap missing, skipping", sdg)
            continue

        r_adj = adj_res_centroids[sdg_idx]
        p_adj = adj_pol_centroids[sdg_idx]

        adj_gap: float | None = None
        if float(np.linalg.norm(r_adj)) > 1e-8 and float(np.linalg.norm(p_adj)) > 1e-8:
            adj_sim = float(np.dot(r_adj, p_adj))
            adj_gap = 1.0 - adj_sim

        delta = (adj_gap - canonical_val) if adj_gap is not None else None

        results.append({
            "sdg": sdg,
            "name": SDG_NAMES[sdg],
            "raw_gap": canonical_val,
            "adj_gap": round(adj_gap, 6) if adj_gap is not None else None,
            "delta": round(delta, 6) if delta is not None else None,
        })
        raw_gaps_list.append(canonical_val)
        if adj_gap is not None:
            adj_gaps_list.append(adj_gap)

    # Verify: the canonical raw gaps we loaded should be self-consistent
    # (no cross-pipeline mismatch possible since we are not recomputing them).
    # Log the values for manual inspection.
    log.info("\n%-4s  %-45s  %8s  %8s  %8s", "SDG", "Description", "Raw gap", "Adj gap", "Delta")
    log.info("-" * 78)
    for r in sorted(results, key=lambda x: x["sdg"]):
        raw_s = f"{r['raw_gap']:.4f}" if r['raw_gap'] is not None else "N/A"
        adj_s = f"{r['adj_gap']:.4f}" if r['adj_gap'] is not None else "N/A"
        del_s = f"{r['delta']:.4f}" if r['delta'] is not None else "N/A"
        log.info("SDG %-2d  %-45s  %8s  %8s  %8s", r["sdg"], r["name"], raw_s, adj_s, del_s)

    mean_raw = float(np.mean(raw_gaps_list))
    mean_adj = float(np.mean(adj_gaps_list)) if adj_gaps_list else 0.0
    mean_delta = mean_adj - mean_raw

    # ------------------------------------------------------------------
    # 8. Write LaTeX output
    # ------------------------------------------------------------------
    out_root = Path(args.output_dir) / "appendix" / "f_register_adjustment"
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_train_total = len(y_train)
    n_per_class = min(SAMPLE_SIZE_PER_CLASS, n_policy)

    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/3_appendix/f_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\RegisterClassifierTrainAcc}}{{{train_acc:.3f}}}",
        rf"\newcommand{{\RegisterClassifierCoefNorm}}{{{coef_norm:.3f}}}",
        rf"\newcommand{{\RegisterClassifierNTrain}}{{{n_train_total:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\RegisterClassifierNPerClass}}{{{n_per_class:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\MeanRawSemanticGap}}{{{mean_raw:.3f}}}",
        rf"\newcommand{{\MeanAdjustedSemanticGap}}{{{mean_adj:.3f}}}",
        rf"\newcommand{{\MeanSemanticGapDelta}}{{{mean_delta:.3f}}}",
    ]
    for r in results:
        sdg_word = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                     "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
                     "Fifteen", "Sixteen", "Seventeen"][r["sdg"] - 1]
        if r["raw_gap"] is not None:
            num_lines.append(rf"\newcommand{{\RegRawGapSdg{sdg_word}}}{{{r['raw_gap']:.4f}}}")
        if r["adj_gap"] is not None:
            num_lines.append(rf"\newcommand{{\RegAdjGapSdg{sdg_word}}}{{{r['adj_gap']:.4f}}}")
        if r["delta"] is not None:
            num_lines.append(rf"\newcommand{{\RegDeltaSdg{sdg_word}}}{{{r['delta']:.4f}}}")
    (tables_dir / "num_register_adjustment.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", tables_dir / "num_register_adjustment.tex")

    tab_lines = [
        "% Auto-generated by 1_code/7_main_analysis/3_appendix/f_register_adjustment.py — do not edit manually",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"SDG & Description & Raw gap & Adjusted gap & $\Delta$ \\",
        r"\midrule",
    ]
    for r in sorted(results, key=lambda x: x["raw_gap"] or 0, reverse=True):
        raw_str = f"{r['raw_gap']:.4f}" if r["raw_gap"] is not None else "N/A"
        adj_str = f"{r['adj_gap']:.4f}" if r["adj_gap"] is not None else "N/A"
        delta_str = f"{r['delta']:.4f}" if r["delta"] is not None else "N/A"
        tab_lines.append(rf"SDG {r['sdg']:2d} & {r['name']} & {raw_str} & {adj_str} & {delta_str} \\")
    tab_lines.extend([
        r"\midrule",
        rf"\multicolumn{{2}}{{l}}{{Mean}} & {mean_raw:.4f} & {mean_adj:.4f} & {mean_delta:.4f} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (tables_dir / "tab_register_adjusted_semgap.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", tables_dir / "tab_register_adjusted_semgap.tex")

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "register_adjustment_results.json").open("w") as f:
        json.dump({
            "n_train": n_train_total,
            "train_accuracy": float(train_acc),
            "direction_norm": float(coef_norm),
            "per_sdg": results,
            "mean_raw_gap": round(mean_raw, 4),
            "mean_adj_gap": round(mean_adj, 4),
            "mean_delta": round(mean_delta, 4),
            "note": "Raw gaps loaded from canonical 4_3_semantic_gap_distances.json; adjusted gaps use segment cap 50.",
        }, f, indent=2)
    log.info("Saved: %s", out_root / "register_adjustment_results.json")


if __name__ == "__main__":
    main()
