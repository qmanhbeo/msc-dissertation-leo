"""
register_adjust stage: materialise the INLP projection matrix G (plan §6.1).

Runs iterative, SDG-stratified INLP (Iterative Nullspace Projection) — a
*binary* research-vs-policy logistic-regression classifier, SDG used only to
stratify sampling and the train/test split (never as a classification target) —
on research + policy embeddings, and persists ONLY the accumulated orthonormal
projection matrix ``G`` (K x dim, ~KB) plus a checkpoint/meta json.  Adjusted
embeddings are NEVER materialised (avoids ~10 GB MPNet arrays); downstream
consumers project raw embeddings on the fly via ``register_utils``.

Tracks are derived from ``--embed-model`` (no ``--track`` flag):
    all-mpnet-base-v2            -> canon   (full research shards)
    all-MiniLM-L6-v2 / scibert   -> subset  (research_subset, 100k papers)

Each iteration k uses its OWN deterministic RNG
``default_rng(POLICY_SEGMENT_CAP_SEED + k)`` for BOTH the stratified sampling
and (downstream) gap computation, so iteration k depends only on (frozen
inputs, ``G[:k-1]``, k).  Resume-from-k is therefore bit-identical to a full
uninterrupted run.

Resume / checkpoint protocol (iteration-level):
  * Persist after EVERY iteration:
      G.npy            accumulated orthonormal rows 1..k (atomic_write_npy)
      checkpoint.json  schema_version, completed_k, iterations [{k, test_acc}],
                       config + inputs fingerprints, run config values
                       (doubles as meta.json on completion)
      status/          heartbeat via update_stage_status
  * Startup:
      1. --overwrite            -> rmtree register/{track}/, start fresh
      2. checkpoint exists +
         G.npy row count == completed_k  -> interrupted run. Recompute config +
         inputs fingerprints NOW and compare. Match -> resume at
         completed_k + 1 using loaded G. Mismatch -> FAIL CLOSED (RuntimeError;
         require --overwrite).
      3. checkpoint.complete    -> stage complete; skip (existence-skip)
      4. G.npy row count != completed_k -> corrupt -> FAIL CLOSED.

Fingerprints are CONTENT-based (sha256 over manifest files / arrays), NOT
mtime-based — 2_data/ re-hydration resets mtimes and must not force a re-run.

Outputs (gitignored 2_data/, never 4_outputs/):
    2_data/3b_register/{slug}/{track}/G.npy
    2_data/3b_register/{slug}/{track}/checkpoint.json
    2_data/3b_register/{slug}/{track}/status/

Run from project root:
    python 1_code/7_main_analysis/0_shared/register_adjust.py --embed-model mpnet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
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
    N_SDG,
    embed_dir_for_model,
    embed_research_dir_for_model,
    register_dir_for_model as register_root_dir_for_model,
    resolve_model_alias,
    scored_dir_for_model,
)
from semantic_gap_shared import (
    RANDOM_SEED as POLICY_SEGMENT_CAP_SEED,
    SEGMENT_CAP_PRIMARY,
    get_cluster_assignments,
    get_policy_emb,
    get_policy_ids,
    get_policy_scores,
    get_research_centroids,
    get_research_centroid_meta,
)
from shard_pipeline_utils import (
    atomic_write_json,
    atomic_write_npy,
    load_json,
    sha256_file,
    update_stage_status,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---- Named constants (result-affecting; recorded in the config fingerprint) ----
SCRIPT_VERSION = "2"
SCHEMA_VERSION = 1

# Retired as a sampling cap (plan §6.1 fix): the per-SDG-per-corpus count is now
# the GLOBAL MIN over all (SDG, corpus) of available counts — a single number for
# the whole track, with NO fixed ceiling. Kept only as a documented reference;
# ``n_target`` (data-derived) supersedes it. MiniLM/SciBERT -> 263, MPNet -> 1207
# under the frozen embedded snapshot.
ITERATIVE_N_PER_SDG = 1000
ITERATIVE_ACC_THRESHOLD = 0.5
ITERATIVE_MAX_K = 200
TEST_SIZE = 0.15
# Classifier hyperparameters (binary research=0 / policy=1 LR).
LR_C = 1.0
LR_PENALTY = "l2"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
# Orthonormalisation guard: below this norm a candidate direction is deemed
# collinear with the already-accumulated space and the loop stops.
DIRECTION_COLLAPSE_EPS = 1e-8

TRACK_CANON = "canon"
TRACK_SUBSET = "subset"

# Within-SDG balance-cap rule version (plan §6.1, FIXED): every SDG/corpus pair
# contributes exactly ``n_target = min(min_sdg avail_research, min_sdg
# avail_policy)`` texts — ONE number for the whole track, no fixed per-SDG cap.
# This removes the prior across-SDG skew where SDGs with more research coverage
# contributed more rows to the pooled INLP classifier and were thus more strongly
# gap-adjusted. Versioned string so a change bumps the config fingerprint.
WITHIN_SDG_BALANCE_CAP_RULE = "global_min_over_sdg_corpus_no_cap"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialise INLP projection matrix G (register_adjust stage, plan §6.1)."
    )
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def track_for_model(model: str) -> str:
    """Track is derived from --embed-model: canon for the default encoder, subset otherwise."""
    return TRACK_CANON if model == DEFAULT_EMBED_MODEL else TRACK_SUBSET


def register_dir_for_model(model: str, track: str) -> Path:
    return register_root_dir_for_model(model) / track


# --------------------------------------------------------------------------- #
# Loaders (canonical source of truth for the INLP engine; this stage is the only
# INLP trainer — downstream scripts load G via register_utils.load_G())
# --------------------------------------------------------------------------- #


def build_research_sdg_index(model: str) -> dict[int, list[tuple[int, int]]]:
    """Map each SDG (1-17) to a list of (shard_id, row_index) from score shards."""
    scored = scored_dir_for_model(model)
    shards_meta = scored / "paper_scores_shards" / "metadata"
    manifest = load_json(shards_meta / "manifest.json")
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
        log.info("  Indexed score shard %d", shard_id)

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


def subtract_direction(emb: np.ndarray, g_dir: np.ndarray) -> np.ndarray:
    """Subtract the projection onto a unit direction and L2-renormalise per row."""
    proj = np.dot(emb, g_dir)[:, np.newaxis] * g_dir
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def load_stratified_samples(
    model: str,
    sdg_index: dict[int, list[tuple[int, int]]],
    n_target: int,
    rng: np.random.Generator,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    projector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a balanced 17-SDG x 2-corpus sample, optionally projected through G.

    Global-min rule (plan §6.1, FIXED): every SDG/corpus pair contributes exactly
    ``n_target`` texts, where ``n_target = min(min_sdg avail_research,
    min_sdg avail_policy)`` is ONE number for the whole track — no fixed per-SDG
    cap. This removes the prior across-SDG skew where SDGs with more research
    coverage contributed more rows to the pooled INLP classifier and were thus
    more strongly gap-adjusted. Because n_target is the global minimum available
    count, both corpora are equally AND uniformly represented within and across
    every SDG (MiniLM/SciBERT -> 263, MPNet -> 1207 under the frozen snapshot).

    Returns (X, y, sdg_labels) where sdg_labels is an int array of 1-indexed SDG
    per row, used for stratified train/test splitting.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sdg_parts: list[np.ndarray] = []
    taken: list[tuple[int, int, int]] = []

    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        entries = sdg_index[sdg]
        n_res = min(n_target, len(entries))
        if n_res == 0:
            continue
        res_emb = load_research_embeddings_for_sdg(model, entries, n_res, rng)

        policy_mask = np.array([a == sdg_idx for a in policy_assignments])
        policy_idxs = np.where(policy_mask)[0]
        n_pol = min(n_target, len(policy_idxs))
        if n_pol == 0:
            continue
        chosen_policy = rng.choice(policy_idxs, size=n_pol, replace=False)
        pol_emb = policy_emb[chosen_policy].copy()

        X_parts.append(res_emb)
        y_parts.append(np.zeros(n_res, dtype=np.int32))
        sdg_parts.append(np.full(n_res, sdg, dtype=np.int32))

        X_parts.append(pol_emb)
        y_parts.append(np.ones(n_pol, dtype=np.int32))
        sdg_parts.append(np.full(n_pol, sdg, dtype=np.int32))
        taken.append((sdg, n_res, n_pol))

    log.info("  stratified per-SDG: %d SDGs, n_target=%d", len(taken), n_target)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    sdg_labels = np.concatenate(sdg_parts, axis=0)

    if projector is not None and projector.shape[0] > 0:
        for g_k in projector:
            X = subtract_direction(X, g_k)

    return X, y, sdg_labels


# --------------------------------------------------------------------------- #
# Fingerprints (content-based, NOT mtime-based)
# --------------------------------------------------------------------------- #


def config_fingerprint(n_target: int) -> str:
    cfg = {
        "script_version": SCRIPT_VERSION,
        "within_sdg_balance_cap_rule": WITHIN_SDG_BALANCE_CAP_RULE,
        "n_target": n_target,
        "iterative_acc_threshold": ITERATIVE_ACC_THRESHOLD,
        "iterative_max_k": ITERATIVE_MAX_K,
        "segment_cap": SEGMENT_CAP_PRIMARY,
        "seed": POLICY_SEGMENT_CAP_SEED,
        "classifier": {"C": LR_C, "penalty": LR_PENALTY, "solver": LR_SOLVER, "max_iter": LR_MAX_ITER},
        "test_size": TEST_SIZE,
        "split_stratify": "sdg_labels * 2 + y",
    }
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _input_files(model: str) -> list[Path]:
    return [
        embed_dir_for_model(model) / "research_shards" / "metadata" / "manifest.json",
        scored_dir_for_model(model) / "paper_scores_shards" / "metadata" / "manifest.json",
        get_policy_emb(model),
        get_policy_scores(model),
        get_policy_ids(model),
        get_research_centroids(model),
        get_research_centroid_meta(model),
    ]


def inputs_fingerprint(model: str) -> str:
    h = hashlib.sha256()
    for path in _input_files(model):
        h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest()


def config_values(n_target: int) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "within_sdg_balance_cap_rule": WITHIN_SDG_BALANCE_CAP_RULE,
        "n_target": n_target,
        "iterative_acc_threshold": ITERATIVE_ACC_THRESHOLD,
        "iterative_max_k": ITERATIVE_MAX_K,
        "segment_cap": SEGMENT_CAP_PRIMARY,
        "seed": POLICY_SEGMENT_CAP_SEED,
        "classifier": {"C": LR_C, "penalty": LR_PENALTY, "solver": LR_SOLVER, "max_iter": LR_MAX_ITER},
        "test_size": TEST_SIZE,
        "split_stratify": "sdg_labels * 2 + y",
    }


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #


def _checkpoint_payload(
    *,
    model: str,
    track: str,
    complete: bool,
    stopped_reason: str | None,
    completed_k: int,
    iterations: list[dict],
    config_fp: str,
    inputs_fp: str,
    n_target: int,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "track": track,
        "complete": complete,
        "stopped_reason": stopped_reason,
        "completed_k": completed_k,
        "n_iters": len(iterations),
        "iterations": iterations,
        "final_acc": iterations[-1]["test_acc"] if iterations else None,
        "config_fingerprint": config_fp,
        "inputs_fingerprint": inputs_fp,
        "config": config_values(n_target),
    }


def _write_checkpoint(
    register_dir: Path,
    G_list: list[np.ndarray],
    payload: dict,
) -> None:
    G = np.vstack(G_list).astype(np.float32) if G_list else np.zeros((0, 0), dtype=np.float32)
    atomic_write_npy(register_dir / "G.npy", G)
    atomic_write_json(register_dir / "checkpoint.json", payload)


def preflight(model: str) -> None:
    """Fail closed if any stage input is missing (per-model policy emb incl. subset)."""
    required = [
        ("policy.npy", get_policy_emb(model)),
        ("policy_scores.npy", get_policy_scores(model)),
        ("policy_ids.json", get_policy_ids(model)),
        ("research_centroids.npy", get_research_centroids(model)),
        ("research_centroid_meta.json", get_research_centroid_meta(model)),
        ("research embed manifest", embed_dir_for_model(model) / "research_shards" / "metadata" / "manifest.json"),
        ("research score manifest", scored_dir_for_model(model) / "paper_scores_shards" / "metadata" / "manifest.json"),
    ]
    missing = [f"{name} ({path})" for name, path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "register_adjust pre-flight failed: missing inputs:\n  " + "\n  ".join(missing)
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    track = track_for_model(model)
    register_dir = register_dir_for_model(model, track)
    g_path = register_dir / "G.npy"
    checkpoint_path = register_dir / "checkpoint.json"
    status_dir = register_dir / "status"

    preflight(model)

    # Derive the global per-SDG-per-corpus target BEFORE the fingerprint check so
    # n_target is part of config_fingerprint. n_target = min over all (SDG, corpus)
    # of available counts — ONE number for the whole track, no fixed cap.
    policy_emb = np.load(get_policy_emb(model)).astype(np.float32)
    policy_scores = np.load(get_policy_scores(model))
    policy_assignments = get_cluster_assignments(policy_scores)
    sdg_index = build_research_sdg_index(model)
    res_avail = {sdg: len(sdg_index[sdg]) for sdg in range(1, N_SDG + 1)}
    pol_avail = np.bincount(policy_assignments, minlength=N_SDG)
    n_target = int(min(min(res_avail.values()), int(pol_avail[1:].min())))
    log.info(
        "Global per-SDG-per-corpus target n_target=%d (res_min=%d, pol_min=%d)",
        n_target, min(res_avail.values()), int(pol_avail[1:].min()),
    )

    config_fp = config_fingerprint(n_target)
    inputs_fp = inputs_fingerprint(model)
    log.info("register_adjust: model=%s track=%s", model, track)
    log.info("  config fingerprint: %s", config_fp[:16])
    log.info("  inputs fingerprint: %s", inputs_fp[:16])

    update_stage_status(status_dir, "register_adjust", "running", {"model": model, "track": track})

    # -- Startup / resume resolution ---------------------------------------- #
    if args.overwrite:
        if register_dir.exists():
            log.info("--overwrite: removing %s", register_dir)
            shutil.rmtree(register_dir)
        completed_k = 0
        G_list: list[np.ndarray] = []
        iterations: list[dict] = []
    elif checkpoint_path.exists():
        ckpt = load_json(checkpoint_path)
        if ckpt.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint.json schema_version {ckpt.get('schema_version')} != {SCHEMA_VERSION}. "
                "Recompute with --overwrite."
            )
        if ckpt.get("config_fingerprint") != config_fp or ckpt.get("inputs_fingerprint") != inputs_fp:
            raise RuntimeError(
                "checkpoint.json fingerprints do NOT match current inputs/config. "
                "Refusing to resume on changed inputs — use --overwrite for a clean recompute."
            )
        if ckpt.get("complete"):
            if not g_path.exists():
                raise RuntimeError(
                    f"checkpoint.json is complete but G.npy is missing ({g_path}). Corrupt state — use --overwrite."
                )
            log.info("Stage already complete (%d iterations, reason=%s) — skipping.", ckpt["n_iters"], ckpt.get("stopped_reason"))
            update_stage_status(status_dir, "register_adjust", "completed", {"model": model, "track": track, "completed_k": ckpt["completed_k"]})
            return

        completed_k = int(ckpt["completed_k"])
        iterations = list(ckpt.get("iterations", []))
        if g_path.exists():
            G = np.load(g_path)
            if G.shape[0] != completed_k:
                raise RuntimeError(
                    f"G.npy has {G.shape[0]} rows but checkpoint says completed_k={completed_k}. "
                    "Corrupt/inconsistent state — use --overwrite."
                )
            G_list = [np.asarray(G[i]).astype(np.float32) for i in range(completed_k)]
        elif completed_k > 0:
            raise RuntimeError(
                f"checkpoint.json says completed_k={completed_k} but G.npy is missing. Use --overwrite."
            )
        else:
            G_list = []
        log.info("Resuming at iteration %d (completed_k=%d)", completed_k + 1, completed_k)
    else:
        completed_k = 0
        G_list = []
        iterations = []

    # -- Iterative INLP loop (per-iteration deterministic RNG) ---------------- #
    # policy_emb / policy_assignments / sdg_index already loaded above.
    started_complete = False
    for k in range(completed_k + 1, ITERATIVE_MAX_K + 1):
        iteration_rng = np.random.default_rng(POLICY_SEGMENT_CAP_SEED + k)
        G_prev = np.vstack(G_list).astype(np.float32) if G_list else None

        log.info("Iteration %d (G has %d rows)", k, G_prev.shape[0] if G_prev is not None else 0)
        X, y, sdg_labels = load_stratified_samples(
            model, sdg_index, n_target, iteration_rng,
            policy_emb=policy_emb, policy_assignments=policy_assignments,
            projector=G_prev,
        )

        # Combined stratification key: 34 classes (17 SDGs x 2 corpora)
        stratify_key = sdg_labels * 2 + y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=stratify_key, random_state=POLICY_SEGMENT_CAP_SEED + k,
        )

        clf = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER, random_state=POLICY_SEGMENT_CAP_SEED + k)
        clf.fit(X_train, y_train)
        test_acc = float(clf.score(X_test, y_test))
        log.info("  85/15 test accuracy: %.4f (n_train=%d, n_test=%d)", test_acc, len(X_train), len(X_test))

        # Fit on the full sample for the direction used in orthogonalisation
        clf_full = LogisticRegression(C=LR_C, solver=LR_SOLVER, max_iter=LR_MAX_ITER, random_state=POLICY_SEGMENT_CAP_SEED + k)
        clf_full.fit(X, y)
        coef = clf_full.coef_.astype(np.float32).flatten()
        g_k = (coef / np.linalg.norm(coef)).astype(np.float32)

        for prev_g in G_list:
            g_k = g_k - np.dot(g_k, prev_g) * prev_g
        g_k_norm = float(np.linalg.norm(g_k))
        if g_k_norm < DIRECTION_COLLAPSE_EPS:
            log.warning("  Direction collapsed after orthogonalisation; stopping.")
            started_complete = True
            stopped_reason = "direction_collapse"
            payload = _checkpoint_payload(
                model=model, track=track, complete=True, stopped_reason=stopped_reason,
                completed_k=len(G_list), iterations=iterations,
                config_fp=config_fp, inputs_fp=inputs_fp, n_target=n_target,
            )
            _write_checkpoint(register_dir, G_list, payload)
            update_stage_status(status_dir, "register_adjust", "completed", {"model": model, "track": track, "completed_k": len(G_list), "reason": stopped_reason})
            break
        g_k = (g_k / g_k_norm).astype(np.float32)
        G_list.append(g_k)
        iterations.append({"k": k, "test_acc": round(test_acc, 4)})

        complete = test_acc <= ITERATIVE_ACC_THRESHOLD
        stopped_reason = "threshold" if complete else None
        payload = _checkpoint_payload(
            model=model, track=track, complete=complete, stopped_reason=stopped_reason,
            completed_k=len(G_list), iterations=iterations,
            config_fp=config_fp, inputs_fp=inputs_fp, n_target=n_target,
        )
        _write_checkpoint(register_dir, G_list, payload)
        update_stage_status(
            status_dir, "register_adjust", "running",
            {"model": model, "track": track, "completed_k": len(G_list), "test_acc": test_acc},
        )
        log.debug("  Iteration %d complete (checkpoint written)", k)

        if complete:
            started_complete = True
            log.info("  Test accuracy %.4f <= threshold %.4f — stopping.", test_acc, ITERATIVE_ACC_THRESHOLD)
            break

    # Loop exhausted at ITERATIVE_MAX_K without the threshold being met.
    if not started_complete:
        log.info("Reached ITERATIVE_MAX_K=%d without threshold; marking complete.", ITERATIVE_MAX_K)
        payload = _checkpoint_payload(
            model=model, track=track, complete=True, stopped_reason="max_k",
            completed_k=len(G_list), iterations=iterations,
            config_fp=config_fp, inputs_fp=inputs_fp, n_target=n_target,
        )
        _write_checkpoint(register_dir, G_list, payload)
        update_stage_status(status_dir, "register_adjust", "completed", {"model": model, "track": track, "completed_k": len(G_list), "reason": "max_k"})

    final = iterations[-1]["test_acc"] if iterations else None
    log.info("register_adjust complete: %d directions, final test acc %s", len(G_list), final)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
