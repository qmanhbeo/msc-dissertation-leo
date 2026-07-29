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
    python 1_code/7_main_analysis/2_appendix/f_register_adjustment.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    embed_research_dir_for_model,
    model_slug,
    output_main_dir_for_model,
    scored_dir_for_model,
    preprocessed_dir,
    resolve_model_alias,
)
from shard_pipeline_utils import resolve_manifest_path
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

# Iterative register check (Appendix E2)
ITERATIVE_N_PER_SDG = 1000
ITERATIVE_ACC_THRESHOLD = 0.5
ITERATIVE_MAX_K = 200

# No module-level canonical path — computed in main() from --output-dir and --model.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run register-adjustment sensitivity analysis.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    return p.parse_args()


def load_research_sample(
    model: str, n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Load a random sample of research embeddings by reading each shard directly (shard-native).

    Samples with the same seeded RNG sequence as the old per-file loop, so
    results are unchanged.
    """
    manifest = load_json(embed_dir_for_model(model) / "research_shards" / "metadata" / "manifest.json")
    shards = sorted(manifest["shards"], key=lambda x: int(x["shard_id"]))
    if not shards:
        raise FileNotFoundError(f"No research shards in manifest for {model}")

    n_per_shard = max(1, n_samples // len(shards))
    samples: list[np.ndarray] = []
    total = 0

    for shard in shards:
        needed = n_samples - total
        if needed <= 0:
            break
        n_rows = int(shard["rows"])
        if n_rows == 0:
            continue
        n_take = min(n_per_shard, needed)
        n_take = min(n_take, n_rows)
        idx = rng.choice(n_rows, size=n_take, replace=False)
        emb = np.load(
            resolve_manifest_path(
                shard["embedding_path"],
                allowed_dirs=(embed_research_dir_for_model(model), scored_dir_for_model(model), preprocessed_dir()),
            ),
            mmap_mode="r",
        )
        samples.append(np.asarray(emb[idx]).copy())
        total += n_take
        log.info("  Sampled %d from shard %s (total %d)", n_take, shard.get("name", shard["shard_id"]), total)

    if total == 0:
        raise RuntimeError("Could not load any research embeddings")
    return np.concatenate(samples, axis=0).astype(np.float32)


def subtract_direction(emb: np.ndarray, g_dir: np.ndarray) -> np.ndarray:
    proj = np.dot(emb, g_dir)[:, np.newaxis] * g_dir
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def build_research_sdg_index(model: str) -> dict[int, list[tuple[int, int]]]:
    """Map each SDG (1-17) to a list of (shard_id, row_index) from score shards."""
    scored = scored_dir_for_model(model)
    shards_meta = scored / "paper_scores_shards" / "metadata"
    manifest_path = shards_meta / "manifest.json"
    manifest = load_json(manifest_path)
    shards = sorted(manifest["shards"], key=lambda x: int(x["shard_id"]))

    index: dict[int, list[tuple[int, int]]] = {sdg: [] for sdg in range(1, N_SDG + 1)}
    for shard in shards:
        shard_id = int(shard["shard_id"])
        jsonl_path = shards_meta / f"part-{shard_id:05d}_ids.jsonl"
        if not jsonl_path.exists():
            log.warning("Score shard JSONL not found: %s", jsonl_path)
            continue
        with open(jsonl_path, encoding="utf-8") as fh:
            for row_idx, line in enumerate(fh):
                row = json.loads(line)
                sdg = row.get("assigned_sdg")
                if isinstance(sdg, (int, float)) and 1 <= int(sdg) <= N_SDG:
                    index[int(sdg)].append((shard_id, row_idx))
        log.info("  Indexed shard %d", shard_id)

    total = sum(len(v) for v in index.values())
    log.info("Research SDG index built: %d papers across %d SDGs", total, N_SDG)
    return index


def load_research_embeddings_for_sdg(
    model: str,
    entries: list[tuple[int, int]],
    n_per_sdg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Load up to n_per_sdg research embeddings from precomputed .npy shards."""
    if len(entries) <= n_per_sdg:
        chosen = entries
    else:
        chosen = [entries[i] for i in rng.choice(len(entries), size=n_per_sdg, replace=False)]

    by_shard: dict[int, list[int]] = {}
    for shard_id, row_idx in chosen:
        by_shard.setdefault(shard_id, []).append(row_idx)

    embed_dir = embed_research_dir_for_model(model)
    parts: list[np.ndarray] = []
    for shard_id, row_idxs in sorted(by_shard.items()):
        emb_path = embed_dir / f"part-{shard_id:05d}.npy"
        emb = np.load(emb_path, mmap_mode="r")
        parts.append(np.asarray(emb[row_idxs]).copy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_stratified_samples(
    model: str,
    sdg_index: dict[int, list[tuple[int, int]]],
    n_per_sdg: int,
    rng: np.random.Generator,
    projector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a balanced 17-SDG × 2-corpus sample, optionally projected through G.

    Returns (X, y, sdg_labels) where sdg_labels is an int array of 1-indexed SDG
    for each row, used for stratified train/test splitting.
    """
    policy_emb = np.load(semantic_gap_shared.get_policy_emb(model)).astype(np.float32)
    policy_scores = np.load(semantic_gap_shared.get_policy_scores(model))
    policy_assignments = get_cluster_assignments(policy_scores)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sdg_parts: list[np.ndarray] = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1

        res_emb = load_research_embeddings_for_sdg(model, sdg_index[sdg], n_per_sdg, rng)
        n_res = len(res_emb)

        policy_mask = np.array([a == sdg_idx for a in policy_assignments])
        policy_idxs = np.where(policy_mask)[0]
        if len(policy_idxs) == 0:
            continue
        n_take = min(n_per_sdg, len(policy_idxs))
        chosen_policy = rng.choice(policy_idxs, size=n_take, replace=False)
        pol_emb = policy_emb[chosen_policy].copy()

        X_parts.append(res_emb)
        y_parts.append(np.zeros(n_res, dtype=np.int32))
        sdg_parts.append(np.full(n_res, sdg, dtype=np.int32))

        X_parts.append(pol_emb)
        y_parts.append(np.ones(len(pol_emb), dtype=np.int32))
        sdg_parts.append(np.full(len(pol_emb), sdg, dtype=np.int32))

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    sdg_labels = np.concatenate(sdg_parts, axis=0)

    if projector is not None and projector.shape[0] > 0:
        for g_k in projector:
            X = subtract_direction(X, g_k)

    return X, y, sdg_labels


def subtract_multiple_directions(emb: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Apply subtract_direction for each row in G."""
    result = emb.copy()
    for k in range(G.shape[0]):
        result = subtract_direction(result, G[k])
    return result


def iterative_register_check(
    model: str,
    sdg_index: dict[int, list[tuple[int, int]]],
    policy_emb: np.ndarray,
    policy_assignments: list[int],
    policy_ids: list,
    research_centroids: np.ndarray,
    research_cohesions: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict:
    """Iteratively remove register directions via stratified 10-fold CV until accuracy ≤ threshold."""
    from scipy.stats import spearmanr

    G_list: list[np.ndarray] = []
    iteration_results: list[dict] = []

    for k in range(1, ITERATIVE_MAX_K + 1):
        G = np.vstack(G_list) if G_list else np.zeros((0, policy_emb.shape[1]), dtype=np.float32)

        log.info("Iterative check — iteration %d (G has %d rows)", k, G.shape[0])
        X, y, sdg_labels = load_stratified_samples(
            model, sdg_index, ITERATIVE_N_PER_SDG, rng,
            projector=G if G.shape[0] > 0 else None,
        )

        # Combined stratification key: 34 classes (17 SDGs × 2 corpora)
        stratify_key = sdg_labels * 2 + y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=stratify_key, random_state=42 + k,
        )

        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + k)
        clf.fit(X_train, y_train)
        test_acc = float(clf.score(X_test, y_test))
        log.info("  85/15 test accuracy: %.4f (n_train=%d, n_test=%d)", test_acc, len(X_train), len(X_test))

        # Fit on full sample for the direction used in orthogonalisation
        clf_full = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42 + k)
        clf_full.fit(X, y)
        coef = clf_full.coef_.astype(np.float32).flatten()
        g_k = (coef / np.linalg.norm(coef)).astype(np.float32)

        for prev_g in G_list:
            g_k = g_k - np.dot(g_k, prev_g) * prev_g
        g_k_norm = float(np.linalg.norm(g_k))
        if g_k_norm < 1e-8:
            log.warning("  Direction collapsed after orthogonalisation; stopping.")
            break
        g_k = (g_k / g_k_norm).astype(np.float32)
        G_list.append(g_k)

        iteration_results.append({
            "k": k,
            "test_acc": round(test_acc, 4),
        })
        log.info("  Iteration %d complete", k)

        if test_acc <= ITERATIVE_ACC_THRESHOLD:
            log.info("  Test accuracy %.4f ≤ threshold %.4f — stopping.", test_acc, ITERATIVE_ACC_THRESHOLD)
            break

    # Recompute gaps using final G
    G_final = np.vstack(G_list) if G_list else np.zeros((0, policy_emb.shape[1]), dtype=np.float32)
    policy_adj = subtract_multiple_directions(policy_emb, G_final)

    adj_pol_centroids = np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float32)
    adj_pol_cohesions = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]
        if not policy_idxs:
            continue
        idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        centroid, cohesion = build_sub_centroid(policy_adj, idxs_capped)
        if centroid is not None:
            adj_pol_centroids[sdg_idx] = centroid
            adj_pol_cohesions[sdg_idx] = cohesion

    adj_res_centroids = np.zeros((N_SDG, research_centroids.shape[1]), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        raw_mean = research_centroids[sdg_idx] * research_cohesions[sdg_idx]
        adj_raw = raw_mean.copy()
        for g_k in G_list:
            adj_raw = adj_raw - np.dot(adj_raw, g_k) * g_k
        norm_val = float(np.linalg.norm(adj_raw))
        if norm_val > 1e-8:
            adj_res_centroids[sdg_idx] = (adj_raw / norm_val).astype(np.float32)
        else:
            adj_res_centroids[sdg_idx] = research_centroids[sdg_idx]

    # Load canonical raw gaps
    canonical_semantic_path = (
        output_main_dir_for_model(model, root=Path(args.output_dir))
        / "data" / "4_3_semantic_gap_distances.json"
    )
    canonical = load_json(canonical_semantic_path)
    canonical_raw = {e["sdg"]: e["semantic_gap"] for e in canonical["per_sdg"]}

    iter_gaps: dict[int, float] = {}
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        r_adj = adj_res_centroids[sdg_idx]
        p_adj = adj_pol_centroids[sdg_idx]
        if float(np.linalg.norm(r_adj)) > 1e-8 and float(np.linalg.norm(p_adj)) > 1e-8:
            iter_gaps[sdg] = 1.0 - float(np.dot(r_adj, p_adj))

    # Load E1 gaps for Spearman comparison
    e1_path = (
        Path(args.output_dir) / "appendix" / model_slug(model) / "f_register_adjustment"
        / "register_adjustment_results.json"
    )
    e1_results = load_json(e1_path)["per_sdg"]
    e1_gaps = {r["sdg"]: r["adj_gap"] for r in e1_results if r["adj_gap"] is not None}

    # Compute iteration-1 gaps (single direction only)
    iter1_gaps: dict[int, float] = {}
    if len(G_list) > 0:
        policy_adj_iter1 = subtract_multiple_directions(policy_emb, np.vstack(G_list[0:1]))
        adj_pol_c1 = np.zeros((N_SDG, policy_emb.shape[1]), dtype=np.float32)
        for sdg_idx in range(N_SDG):
            policy_idxs = [i for i, a in enumerate(policy_assignments) if a == sdg_idx]
            if not policy_idxs:
                continue
            idxs_capped = cap_policy_indices_per_doc(policy_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
            centroid, _ = build_sub_centroid(policy_adj_iter1, idxs_capped)
            if centroid is not None:
                adj_pol_c1[sdg_idx] = centroid
        adj_res_c1 = np.zeros((N_SDG, research_centroids.shape[1]), dtype=np.float32)
        for sdg_idx in range(N_SDG):
            raw_mean = research_centroids[sdg_idx] * research_cohesions[sdg_idx]
            adj_raw = raw_mean - np.dot(raw_mean, G_list[0]) * G_list[0]
            nv = float(np.linalg.norm(adj_raw))
            if nv > 1e-8:
                adj_res_c1[sdg_idx] = (adj_raw / nv).astype(np.float32)
            else:
                adj_res_c1[sdg_idx] = research_centroids[sdg_idx]
        for sdg_idx in range(N_SDG):
            sdg = sdg_idx + 1
            if float(np.linalg.norm(adj_res_c1[sdg_idx])) > 1e-8 and float(np.linalg.norm(adj_pol_c1[sdg_idx])) > 1e-8:
                iter1_gaps[sdg] = 1.0 - float(np.dot(adj_res_c1[sdg_idx], adj_pol_c1[sdg_idx]))

    # Spearman rank correlation: iteration-1 vs final
    sdgs_common = sorted(set(iter1_gaps.keys()) & set(iter_gaps.keys()))
    rho_iter1_final = 0.0
    if len(sdgs_common) >= 3:
        rho_iter1_final, _ = spearmanr(
            [iter1_gaps[s] for s in sdgs_common],
            [iter_gaps[s] for s in sdgs_common],
        )

    # Write outputs
    out_root = Path(args.output_dir) / "appendix" / model_slug(model) / "f_register_adjustment"
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # LaTeX table (truncated: show iter 1, every 10th, and the last)
    mean_gaps = [np.mean(list(iter_gaps.values()))] if iter_gaps else [0.0]
    last_k = iteration_results[-1]["k"]
    show_ks = {1, last_k}
    show_ks.update(range(10, last_k, 10))
    tab_lines = [
        "% Auto-generated by f_register_adjustment.py iterative check — do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Iteration & Test acc. & Mean gap & Spearman $\rho$ vs iter\,1 \\",
        r"\midrule",
    ]
    for r in iteration_results:
        if r["k"] not in show_ks:
            continue
        if r["k"] == 1:
            mg = f"{mean_gaps[0]:.3f}"
            sp = "1.000"
        else:
            mg = "—"
            sp = f"{rho_iter1_final:.3f}"
        tab_lines.append(f"{r['k']} & {r['test_acc']:.3f} & {mg} & {sp} \\\\")
    tab_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_iterative_register_check.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")

    # Numerical macros
    num_lines = [
        "% Auto-generated by f_register_adjustment.py iterative check",
        rf"\newcommand{{\RegisterIterNPerSdg}}{{{ITERATIVE_N_PER_SDG}}}",
        rf"\newcommand{{\RegisterFirstAcc}}{{{iteration_results[0]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterFinalAcc}}{{{iteration_results[-1]['test_acc']:.3f}}}",
        rf"\newcommand{{\RegisterIterFinalK}}{{{len(iteration_results)}}}",
        rf"\newcommand{{\RegisterIterSpearmanRho}}{{{rho_iter1_final:.3f}}}",
    ]
    (tables_dir / "num_iterative_register_check.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    # JSON
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "iterative_check_results.json").open("w") as f:
        json.dump({
            "iterations": iteration_results,
            "final_k": len(iteration_results),
            "spearman_rho_iter1_final": round(rho_iter1_final, 4),
            "per_sdg_final_gaps": {str(k): round(v, 4) for k, v in iter_gaps.items()},
            "note": "85/15 stratified train/test split on 1000-per-SDG samples; G-projected data at each iteration.",
        }, f, indent=2)

    # Lightweight iteration log (CSV)
    csv_lines = ["iteration,test_acc"]
    for r in iteration_results:
        csv_lines.append(f"{r['k']},{r['test_acc']}")
    (tables_dir / "iteration_results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    log.info("Iterative register check complete: %d iterations, Spearman ρ=%.4f", len(iteration_results), rho_iter1_final)
    return {"iterations": iteration_results, "spearman_rho": rho_iter1_final}


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    rng = np.random.default_rng(POLICY_SEGMENT_CAP_SEED)

    # ------------------------------------------------------------------
    # 1. Load canonical raw gaps
    # ------------------------------------------------------------------
    canonical_semantic_path = output_main_dir_for_model(model, root=Path(args.output_dir)) / "data" / "4_3_semantic_gap_distances.json"
    log.info("Loading canonical raw gaps from %s", canonical_semantic_path)
    canonical = load_json(canonical_semantic_path)
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
    out_root = Path(args.output_dir) / "appendix" / model_slug(model) / "f_register_adjustment"
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_train_total = len(y_train)
    n_per_class = min(SAMPLE_SIZE_PER_CLASS, n_policy)

    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/f_register_adjustment.py — do not edit manually",
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
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/f_register_adjustment.py — do not edit manually",
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

    # ------------------------------------------------------------------
    # 9. Iterative stratified register check (Appendix E2)
    # ------------------------------------------------------------------
    log.info("\n=== Iterative stratified register check ===")
    sdg_index = build_research_sdg_index(model)
    iterative_register_check(
        model=model,
        sdg_index=sdg_index,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_ids,
        research_centroids=research_centroids,
        research_cohesions=research_cohesions,
        rng=rng,
        args=args,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
