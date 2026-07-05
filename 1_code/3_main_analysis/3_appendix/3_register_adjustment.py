"""
Register-adjustment robustness and diagnostics for semantic-gap analysis.

This stage is intentionally additive. The raw within-SDG semantic gap remains
the dissertation's primary estimand and canonical result. The procedures here
probe whether broad research-vs-policy register differences contribute to that
raw gap, and how aggressively different subtraction schemes begin to remove the
within-SDG contrast itself.

Outputs:
  4_outputs/appendix/d_register_adjustment/data/*.json
  4_outputs/appendix/d_register_adjustment/data/*.csv
  4_outputs/appendix/d_register_adjustment/data/*.npy
  4_outputs/appendix/d_register_adjustment/register_confidence_checks/*
  4_outputs/appendix/d_register_adjustment/tables/*.tex
  4_outputs/appendix/d_register_adjustment/figures/*.pdf
  4_outputs/appendix/d_register_adjustment/figures/*.png
  4_outputs/appendix/d_register_adjustment/README_register_adjustment.md
  4_outputs/appendix/d_register_adjustment/register_direction_interpretation.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dissertation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared_utils import DissertationOutputs
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    MIN_CLUSTER_SIZE,
    N_SDG,
    POLICY_EMB,
    POLICY_IDS,
    POLICY_SCORES,
    RANDOM_SEED,
    RESEARCH_CENTROID_META,
    RESEARCH_CENTROIDS,
    build_sub_centroid,
    cap_policy_indices_per_doc,
    compute_sdg_semantic_gaps,
    get_cluster_assignments,
)


DEFAULT_OUTPUT_ROOT = Path("4_outputs")
DEFAULT_CACHE_ROOT = Path("2_data/3_scored/register_adjustment_cache")
EMBED_MANIFEST = Path("2_data/2_embedded/research_shards/metadata/manifest.json")
SCORE_MANIFEST = Path("2_data/3_scored/paper_scores_shards/metadata/manifest.json")
TEXT_MANIFEST = Path("2_data/1_preprocessed/research_corpus/metadata/manifest.json")
POLICY_TEXT_IDS = Path("2_data/2_embedded/metadata/policy_ids.json")
SDG_CENTROIDS = Path("2_data/3_scored/sdg_centroids.npy")

DEFAULT_SAMPLE_PER_CLASS = 20_000
DEFAULT_TFIDF_SAMPLE_PER_CLASS = 5_000
DEFAULT_TFIDF_MAX_FEATURES = 20_000
DEFAULT_EXTREME_TOP_N = 25
ROBUSTNESS_OUTPUT_SUBDIR = Path("appendix") / "d_register_adjustment"
DEFAULT_GENRE_CONFIDENCE_SUBDIR = "register_confidence_checks"
DEFAULT_MULTI_DIRECTION_KS = (1, 2, 3, 5)
DEFAULT_TOPIC_MATCH_RESEARCH_PER_SDG = 5_000
DEFAULT_TOPIC_MATCH_TOP_K = 1
TOPIC_MATCH_EXAMPLES_PER_SDG = 5
TOPIC_MATCH_BATCH_SIZE = 512
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
DEFAULT_SDG_GENRE_METHOD = "both"
DEFAULT_SDG_GENRE_MIN_SAMPLES_PER_CLASS = 50
DEFAULT_SDG_GENRE_TEST_SIZE = 0.20
DEFAULT_SDG_GENRE_CLASSIFIER_TYPE = "logistic_regression_liblinear"
GENRE_PROJECTION_TOP_N = 100
GENRE_PROJECTION_REPORT_N = 20
GENRE_PROJECTION_PREVIEW_CHARS = 400
GENRE_TABLE_PREVIEW_CHARS = 100
REGRESSION_BASELINE_SDG = 1
CACHE_SCHEMA_VERSION = 1
HELDOUT_SDG_FOLDS = (
    (1, 6, 11, 16),
    (2, 7, 12, 17),
    (3, 8, 13),
    (4, 9, 14),
    (5, 10, 15),
)


def ensure_register_robustness_outputs(output_dir: Path) -> DissertationOutputs:
    """Create and return the appendix-style robustness output layout."""
    robustness_root = Path(output_dir) / ROBUSTNESS_OUTPUT_SUBDIR
    data_dir = robustness_root / "data"
    tables_dir = robustness_root / "tables"
    figures_dir = robustness_root / "figures"
    for path in (data_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return DissertationOutputs(root=robustness_root, tables_dir=tables_dir, figures_dir=figures_dir, data_dir=data_dir)


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchShard:
    shard_id: int
    name: str
    rows: int
    start: int
    stop: int
    emb_path: Path
    score_path: Path
    score_ids_path: Path
    text_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the appendix-style register-adjustment robustness suite. "
            "This stage does not replace the canonical raw semantic gap."
        )
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    p.add_argument("--sample-per-class", type=int, default=DEFAULT_SAMPLE_PER_CLASS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    p.add_argument("--segment-cap", type=int, default=SEGMENT_CAP_PRIMARY)
    p.add_argument("--extreme-top-n", type=int, default=DEFAULT_EXTREME_TOP_N)
    p.add_argument("--tfidf-sample-per-class", type=int, default=DEFAULT_TFIDF_SAMPLE_PER_CLASS)
    p.add_argument("--tfidf-max-features", type=int, default=DEFAULT_TFIDF_MAX_FEATURES)
    p.add_argument("--skip-tfidf-helper", action="store_true")
    p.add_argument(
        "--skip-register-confidence-checks",
        action="store_true",
        help="Skip the additional register-confidence robustness checks inside this stage.",
    )
    p.add_argument(
        "--method",
        choices=["sdg_balanced", "within_sdg", "both"],
        default=DEFAULT_SDG_GENRE_METHOD,
        help="Run SDG-aware register robustness methods: one global SDG-balanced classifier, within-SDG classifiers, or both.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional override seed for the SDG-aware register robustness methods. Defaults to --seed.",
    )
    p.add_argument(
        "--samples-per-cell",
        type=int,
        default=None,
        help="Optional cap for the number of research and policy observations sampled per SDG x register cell.",
    )
    p.add_argument(
        "--min-samples-per-class",
        type=int,
        default=DEFAULT_SDG_GENRE_MIN_SAMPLES_PER_CLASS,
        help="Minimum per-class count required before fitting a within-SDG classifier.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_SDG_GENRE_TEST_SIZE,
        help="Held-out test fraction for the SDG-aware register robustness classifiers.",
    )
    p.add_argument(
        "--classifier-type",
        choices=["logistic_regression_liblinear", "logistic_regression_saga"],
        default=DEFAULT_SDG_GENRE_CLASSIFIER_TYPE,
        help="Simple linear classifier to use for the SDG-aware register robustness methods.",
    )
    p.add_argument("--local-confidence-checks", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--local-check-subdir", default=DEFAULT_GENRE_CONFIDENCE_SUBDIR, help=argparse.SUPPRESS)
    p.add_argument("--multi-direction-ks", default="1,2,3,5")
    p.add_argument("--topic-match-research-per-sdg", type=int, default=DEFAULT_TOPIC_MATCH_RESEARCH_PER_SDG)
    p.add_argument("--topic-match-top-k", type=int, default=DEFAULT_TOPIC_MATCH_TOP_K)
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_manifest_path(stored_path: str, required_prefix: str) -> Path:
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    if not raw.as_posix().startswith(required_prefix):
        raise RuntimeError(
            f"Hard pivot violation: expected path under {required_prefix}, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def build_research_shards() -> tuple[list[ResearchShard], int]:
    emb_manifest = load_json(EMBED_MANIFEST)
    score_manifest = load_json(SCORE_MANIFEST)
    text_manifest = load_json(TEXT_MANIFEST)

    emb_shards = sorted(emb_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    score_shards = sorted(score_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))
    text_shards = sorted(text_manifest.get("shards", []), key=lambda x: int(x["shard_id"]))

    if not emb_shards or not score_shards or not text_shards:
        raise RuntimeError("One or more research manifests are empty.")
    if not (len(emb_shards) == len(score_shards) == len(text_shards)):
        raise RuntimeError(
            "Research manifests are misaligned: "
            f"emb={len(emb_shards)} score={len(score_shards)} text={len(text_shards)}"
        )

    shards: list[ResearchShard] = []
    offset = 0
    for emb_shard, score_shard, text_shard in zip(emb_shards, score_shards, text_shards):
        shard_id = int(emb_shard["shard_id"])
        if shard_id != int(score_shard["shard_id"]) or shard_id != int(text_shard["shard_id"]):
            raise RuntimeError("Research manifests do not align on shard_id.")
        name = emb_shard["name"]
        if name != score_shard["name"] or name != text_shard["name"]:
            raise RuntimeError("Research manifests do not align on shard name.")
        rows = int(emb_shard["rows"])
        if rows != int(score_shard["rows"]) or rows != int(text_shard["rows"]):
            raise RuntimeError(f"Research manifests do not align on row count for shard {name}.")
        shards.append(
            ResearchShard(
                shard_id=shard_id,
                name=name,
                rows=rows,
                start=offset,
                stop=offset + rows,
                emb_path=resolve_manifest_path(emb_shard["embedding_path"], "2_data/2_embedded/"),
                score_path=resolve_manifest_path(score_shard["score_path"], "2_data/3_scored/"),
                score_ids_path=resolve_manifest_path(score_shard["ids_path"], "2_data/3_scored/"),
                text_path=resolve_manifest_path(text_shard["data_path"], "2_data/1_preprocessed/"),
            )
        )
        offset += rows
    return shards, offset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    size = int(stat.st_size)
    with path.open("rb") as f:
        head = f.read(4096)
    content_digest = hashlib.sha256(head).hexdigest()[:16]
    return {
        "path": str(path),
        "size": size,
        "content_head_digest": content_digest,
    }


def stable_cache_key(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def cache_phase_dir(cache_root: Path, phase: str) -> Path:
    path = cache_root / phase
    ensure_dir(path)
    return path


def cache_npz_path(cache_root: Path, phase: str, key: str) -> Path:
    return cache_phase_dir(cache_root, phase) / f"{key}.npz"


def cache_npy_path(cache_root: Path, phase: str, key: str) -> Path:
    return cache_phase_dir(cache_root, phase) / f"{key}.npy"


def cache_json_path(cache_root: Path, phase: str, key: str) -> Path:
    return cache_phase_dir(cache_root, phase) / f"{key}.json"


def cache_csv_path(cache_root: Path, phase: str, key: str) -> Path:
    return cache_phase_dir(cache_root, phase) / f"{key}.csv"


def cache_input_signatures(required_paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {str(path): path_signature(path) for path in required_paths}


def write_cache_manifest(cache_root: Path, payload: dict[str, Any]) -> None:
    write_json(cache_root / "manifest.json", payload)


def parse_c_grid(raw: str) -> list[float]:
    vals = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError("C grid must contain at least one value.")
    return vals


def parse_int_list(raw: str) -> list[int]:
    vals = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def validate_split_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Train/validation/test fractions must sum to 1.0, got {train_frac} + {val_frac} + {test_frac} = {total}"
        )


def validate_test_size(test_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"Expected 0 < test_size < 1, got {test_size}")


def build_linear_classifier(classifier_type: str, seed: int, *, c_value: float = 1.0) -> LogisticRegression:
    solver = {
        "logistic_regression_liblinear": "liblinear",
        "logistic_regression_saga": "saga",
    }[classifier_type]
    max_iter = 2000 if solver == "saga" else 1000
    return LogisticRegression(
        C=c_value,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
        class_weight=None,
        l1_ratio=0,
    )


def metrics_for_binary_classifier(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 6)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def fit_binary_classifier_train_test(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str,
    seed: int,
) -> tuple[LogisticRegression, dict[str, Any]]:
    model = build_linear_classifier(classifier_type, seed)
    model.fit(X_train, y_train)
    scores = model.decision_function(X_test)
    preds = model.predict(X_test)
    metrics = metrics_for_binary_classifier(y_test, scores, preds)
    coef = model.coef_[0].astype(np.float32)
    coef_norm = float(np.linalg.norm(coef))
    if coef_norm < 1e-8:
        raise RuntimeError("Classifier coefficient norm is near zero.")
    metrics["coefficient_norm"] = round(coef_norm, 6)
    metrics["intercept"] = round(float(model.intercept_[0]), 6)
    return model, metrics


def sample_sorted_from_pool(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Sample size must be positive.")
    if n > pool.shape[0]:
        raise ValueError(f"Cannot sample {n} rows from pool of size {pool.shape[0]}")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False).astype(np.int64))


def normalize_unit_vector(vec: np.ndarray, *, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise RuntimeError(f"{label} has near-zero norm.")
    return (vec / norm).astype(np.float32)


def choose_global_indices(total_rows: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if sample_size > total_rows:
        raise ValueError(f"Requested sample_size={sample_size:,} exceeds available rows={total_rows:,}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=sample_size, replace=False).astype(np.int64))


def collect_research_embeddings(shards: list[ResearchShard], global_indices: np.ndarray) -> np.ndarray:
    out = np.empty((global_indices.shape[0], 384), dtype=np.float32)
    cursor = 0
    for shard in shards:
        left = int(np.searchsorted(global_indices, shard.start, side="left"))
        right = int(np.searchsorted(global_indices, shard.stop, side="left"))
        if left >= right:
            continue
        local = global_indices[left:right] - shard.start
        emb = np.load(shard.emb_path).astype(np.float32)
        out[cursor : cursor + local.shape[0]] = emb[local]
        cursor += local.shape[0]
    if cursor != global_indices.shape[0]:
        raise RuntimeError(
            f"Collected only {cursor:,} research embeddings for {global_indices.shape[0]:,} requested rows."
        )
    return out


def load_or_build_sample_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    shards: list[ResearchShard],
    total_research_rows: int,
    policy_rows: int,
    sample_per_class: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "sample",
        "base_signature": base_signature,
        "sample_per_class": sample_per_class,
        "seed": seed,
        "total_research_rows": total_research_rows,
        "policy_rows": policy_rows,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "sample", key)
    meta_path = cache_json_path(cache_root, "sample", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: sample embeddings (%s)", key)
        cached = np.load(cache_path)
        return (
            payload,
            cached["research_indices"].astype(np.int64),
            cached["policy_indices"].astype(np.int64),
            cached["research_embeddings"].astype(np.float32),
        )

    log.info("Cache miss: sample embeddings (%s)", key)
    research_indices = choose_global_indices(total_research_rows, sample_per_class, seed)
    policy_indices = np.sort(
        np.random.default_rng(seed + 17).choice(
            policy_rows,
            size=sample_per_class,
            replace=False,
        ).astype(np.int64)
    )
    research_embeddings = collect_research_embeddings(shards, research_indices)
    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        research_indices=research_indices.astype(np.int64),
        policy_indices=policy_indices.astype(np.int64),
        research_embeddings=research_embeddings.astype(np.float32),
    )
    write_json(meta_path, payload)
    return payload, research_indices, policy_indices, research_embeddings


def load_or_build_classifier_cache(
    *,
    cache_root: Path,
    sample_signature: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    c_grid: list[float],
    split_fracs: dict[str, float],
    split_sizes: dict[str, int],
    sample_class_balance: dict[str, int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray, float]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "classifier",
        "sample_signature": sample_signature,
        "c_grid": list(c_grid),
        "split_fracs": split_fracs,
        "split_sizes": split_sizes,
        "sample_class_balance": sample_class_balance,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "classifier", key)
    if cache_path.exists():
        log.info("Cache hit: classifier selection (%s)", key)
        cached = load_json(cache_path)
        coef = np.array(cached["coef"], dtype=np.float32)
        register_unit = np.array(cached["register_unit"], dtype=np.float32)
        intercept = float(cached["intercept"])
        return payload, cached["candidate_rows"], cached["metrics_payload"], coef, register_unit, intercept

    log.info("Cache miss: classifier selection (%s)", key)
    candidate_rows: list[dict[str, Any]] = []
    fitted_models: dict[float, LogisticRegression] = {}
    for c_value in c_grid:
        model, train_metrics, val_metrics = fit_and_score_model(X_train, y_train, X_val, y_val, c_value, seed)
        fitted_models[c_value] = model
        candidate_rows.append(
            {
                "C": c_value,
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "train_roc_auc": round(train_metrics["roc_auc"], 6),
                "train_f1": round(train_metrics["f1"], 6),
                "train_precision": round(train_metrics["precision"], 6),
                "train_recall": round(train_metrics["recall"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_roc_auc": round(val_metrics["roc_auc"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "val_precision": round(val_metrics["precision"], 6),
                "val_recall": round(val_metrics["recall"], 6),
            }
        )

    selected = select_best_candidate(candidate_rows)
    selected_c = float(selected["C"])
    selected_model = fitted_models[selected_c]
    test_scores = selected_model.decision_function(X_test)
    test_preds = selected_model.predict(X_test)
    test_metrics = metrics_for_scores(y_test, test_scores, test_preds)
    test_cm = confusion_matrix(y_test, test_preds, labels=[0, 1]).tolist()

    final_model = LogisticRegression(
        C=selected_c,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
        l1_ratio=0,
    )
    final_model.fit(X_train_val, y_train_val)
    coef = final_model.coef_[0].astype(np.float32)
    coef_norm = float(np.linalg.norm(coef))
    if coef_norm < 1e-8:
        raise RuntimeError("Final register classifier coefficient norm is near zero.")
    register_unit = (coef / coef_norm).astype(np.float32)
    intercept = float(final_model.intercept_[0])

    metrics_payload = {
        "seed": seed,
        "sample_class_balance": sample_class_balance,
        "split_fracs": split_fracs,
        "split_sizes": split_sizes,
        "candidate_c_grid": list(c_grid),
        "selected_model": selected,
        "selected_model_test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 6),
            "roc_auc": round(test_metrics["roc_auc"], 6),
            "f1": round(test_metrics["f1"], 6),
            "precision": round(test_metrics["precision"], 6),
            "recall": round(test_metrics["recall"], 6),
        },
        "selected_model_test_confusion_matrix": {
            "labels": {"0": "research", "1": "policy"},
            "matrix": test_cm,
        },
        "final_train_plus_validation_model": {
            "C": selected_c,
            "coefficient_norm": round(coef_norm, 6),
            "intercept": round(intercept, 6),
            "register_unit": [round(float(v), 10) for v in register_unit.tolist()],
        },
    }
    write_json(
        cache_path,
        {
            "payload": payload,
            "candidate_rows": candidate_rows,
            "metrics_payload": metrics_payload,
            "coef": [float(v) for v in coef.tolist()],
            "register_unit": [float(v) for v in register_unit.tolist()],
            "intercept": intercept,
        },
    )
    return payload, candidate_rows, metrics_payload, coef, register_unit, intercept


def load_or_build_adjusted_policy_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    classifier_signature: dict[str, Any],
    policy_emb: np.ndarray,
    register_unit: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "adjusted_policy",
        "base_signature": base_signature,
        "classifier_signature": classifier_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npy_path(cache_root, "adjusted_policy", key)
    meta_path = cache_json_path(cache_root, "adjusted_policy", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: adjusted policy embeddings (%s)", key)
        return payload, np.load(cache_path).astype(np.float32)

    log.info("Cache miss: adjusted policy embeddings (%s)", key)
    adjusted_policy_emb = project_out_direction_and_normalize(policy_emb, register_unit)
    ensure_dir(cache_path.parent)
    np.save(cache_path, adjusted_policy_emb.astype(np.float32))
    write_json(meta_path, payload)
    return payload, adjusted_policy_emb


def load_or_build_adjusted_research_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    classifier_signature: dict[str, Any],
    shards: list[ResearchShard],
    register_unit: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "adjusted_research",
        "base_signature": base_signature,
        "classifier_signature": classifier_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "adjusted_research", key)
    meta_path = cache_json_path(cache_root, "adjusted_research", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: adjusted research aggregates (%s)", key)
        cached = np.load(cache_path)
        return (
            payload,
            cached["centroids"].astype(np.float32),
            cached["counts"].astype(np.int64),
            cached["cohesions"].astype(np.float32),
        )

    log.info("Cache miss: adjusted research aggregates (%s)", key)
    centroids, counts, cohesions = build_adjusted_research_centroids(shards, register_unit)
    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        centroids=centroids.astype(np.float32),
        counts=counts.astype(np.int64),
        cohesions=cohesions.astype(np.float32),
    )
    write_json(meta_path, payload)
    return payload, centroids, counts, cohesions


def sanitize_text(text: str) -> str:
    cleaned: list[str] = []
    for ch in text or "":
        codepoint = ord(ch)
        if ch in "\n\r\t":
            cleaned.append(" ")
            continue
        if codepoint < 32 or 0x7F <= codepoint <= 0x9F:
            continue
        if 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFF) in {0xFFFE, 0xFFFF}:
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def strip_text(text: str, limit: int = 280) -> str:
    compact = " ".join(sanitize_text(text).split())
    return compact[:limit]


def compact_table_id(value: str, limit: int = 24) -> str:
    text = strip_text(value, limit=10_000)
    if text.startswith("https://openalex.org/"):
        text = text.rsplit("/", 1)[-1]
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_research_texts_for_indices(
    shards: list[ResearchShard],
    global_indices: np.ndarray,
) -> dict[int, dict[str, Any]]:
    want = {int(idx): None for idx in global_indices.tolist()}
    found = 0
    for shard in shards:
        left = int(np.searchsorted(global_indices, shard.start, side="left"))
        right = int(np.searchsorted(global_indices, shard.stop, side="left"))
        if left >= right:
            continue
        local_positions = {int(global_indices[pos] - shard.start): int(global_indices[pos]) for pos in range(left, right)}
        for row_idx, row in enumerate(iter_jsonl(shard.text_path)):
            global_idx = local_positions.get(row_idx)
            if global_idx is None:
                continue
            want[global_idx] = {
                "openalex_id": row["openalex_id"],
                "publication_year": row.get("publication_year"),
                "title": row.get("title") or "",
                "text": row.get("combined_text") or row.get("abstract") or row.get("title") or "",
            }
            found += 1
            if found == len(want):
                return want
    missing = [idx for idx, row in want.items() if row is None]
    if missing:
        raise RuntimeError(f"Could not resolve {len(missing)} research text rows for TF-IDF helper.")
    return want


def metrics_for_scores(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def fit_and_score_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[LogisticRegression, dict[str, float], dict[str, float]]:
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
        l1_ratio=0,
    )
    model.fit(X_train, y_train)
    train_scores = model.decision_function(X_train)
    train_preds = model.predict(X_train)
    eval_scores = model.decision_function(X_eval)
    eval_preds = model.predict(X_eval)
    return (
        model,
        metrics_for_scores(y_train, train_scores, train_preds),
        metrics_for_scores(y_eval, eval_scores, eval_preds),
    )


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No validation candidates were evaluated.")
    return sorted(
        rows,
        key=lambda row: (
            -row["val_roc_auc"],
            -row["val_f1"],
            -row["val_accuracy"],
            row["C"],
        ),
    )[0]


def project_out_direction_and_normalize(emb: np.ndarray, register_unit: np.ndarray) -> np.ndarray:
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb.shape}")
    proj = emb @ register_unit
    adjusted = emb - np.outer(proj, register_unit).astype(np.float32)
    norms = np.linalg.norm(adjusted, axis=1)
    bad = np.flatnonzero(norms < 1e-8)
    if bad.size:
        raise RuntimeError(
            f"Projection removal produced {bad.size} near-zero adjusted vectors; sample rows: {bad[:5].tolist()}"
        )
    adjusted = adjusted / norms[:, None]
    return adjusted.astype(np.float32)


def build_adjusted_research_centroids(
    shards: list[ResearchShard],
    register_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = register_unit.shape[0]
    sums = np.zeros((N_SDG, dim), dtype=np.float64)
    counts = np.zeros(N_SDG, dtype=np.int64)
    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        score = np.load(shard.score_path).astype(np.float32)
        assignments = score.argmax(axis=1)
        adjusted = project_out_direction_and_normalize(emb, register_unit)
        counts += np.bincount(assignments, minlength=N_SDG)
        for sdg_idx in np.unique(assignments):
            mask = assignments == sdg_idx
            sums[sdg_idx] += adjusted[mask].sum(axis=0)
    centroids = np.zeros((N_SDG, dim), dtype=np.float32)
    cohesions = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        if n == 0:
            continue
        raw = sums[sdg_idx] / float(n)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            raise RuntimeError(f"Adjusted research centroid for SDG {sdg_idx + 1} has near-zero norm.")
        centroids[sdg_idx] = (raw / norm).astype(np.float32)
        cohesions[sdg_idx] = float(norm)
    return centroids, counts, cohesions


def merge_gap_results(raw_results: list[dict[str, Any]], adjusted_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for raw_row, adj_row in zip(raw_results, adjusted_results):
        merged.append(
            {
                "sdg": raw_row["sdg"],
                "n_papers": raw_row["n_papers"],
                "n_policy_segments": raw_row["n_policy_segments"],
                "n_policy_segments_capped": raw_row["n_policy_segments_capped"],
                "n_policy_docs": raw_row["n_policy_docs"],
                "n_policy_docs_capped": raw_row["n_policy_docs_capped"],
                "segment_cap": raw_row["segment_cap"],
                "raw_similarity": raw_row["semantic_similarity"],
                "raw_gap": raw_row["semantic_gap"],
                "register_adjusted_similarity": adj_row["semantic_similarity"],
                "register_adjusted_gap": adj_row["semantic_gap"],
                "delta_gap": (
                    None
                    if raw_row["semantic_gap"] is None or adj_row["semantic_gap"] is None
                    else round(float(adj_row["semantic_gap"]) - float(raw_row["semantic_gap"]), 6)
                ),
                "raw_research_cohesion": raw_row["research_cohesion"],
                "raw_policy_cohesion": raw_row["policy_cohesion"],
                "register_adjusted_research_cohesion": adj_row["research_cohesion"],
                "register_adjusted_policy_cohesion": adj_row["policy_cohesion"],
                "unreliable": bool(adj_row["unreliable"]),
                "unreliable_reason": adj_row["unreliable_reason"],
            }
        )
    return merged


def write_combined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sdg",
                "n_papers",
                "n_policy_segments",
                "n_policy_segments_capped",
                "n_policy_docs",
                "n_policy_docs_capped",
                "segment_cap",
                "raw_similarity",
                "raw_gap",
                "raw_research_cohesion",
                "raw_policy_cohesion",
                "register_adjusted_similarity",
                "register_adjusted_gap",
                "register_adjusted_research_cohesion",
                "register_adjusted_policy_cohesion",
                "delta_gap",
                "unreliable",
                "unreliable_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def score_row_with_model(emb: np.ndarray, coef: np.ndarray, intercept: float) -> float:
    return float(np.dot(emb, coef) + intercept)


def push_highest(
    heap: list[tuple[float, str, dict[str, Any]]],
    limit: int,
    score: float,
    payload: dict[str, Any],
) -> None:
    item = (score, str(payload["item_id"]), payload)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def push_lowest(
    heap: list[tuple[float, str, dict[str, Any]]],
    limit: int,
    score: float,
    payload: dict[str, Any],
) -> None:
    item = (-score, str(payload["item_id"]), payload)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif -score > heap[0][0]:
        heapq.heapreplace(heap, item)


def collect_extreme_examples(
    policy_emb: np.ndarray,
    policy_text_ids: list[dict[str, Any]],
    policy_score_ids: list[dict[str, Any]],
    policy_assignments: np.ndarray,
    shards: list[ResearchShard],
    coef: np.ndarray,
    intercept: float,
    top_n: int,
) -> list[dict[str, Any]]:
    policy_like: list[tuple[float, str, dict[str, Any]]] = []
    research_like: list[tuple[float, str, dict[str, Any]]] = []

    for idx in range(policy_emb.shape[0]):
        score = score_row_with_model(policy_emb[idx], coef, intercept)
        payload = {
            "source_type": "policy",
            "item_id": policy_score_ids[idx]["id"],
            "source_doc_or_year": policy_score_ids[idx]["source_doc"],
            "assigned_sdg": int(policy_assignments[idx]) + 1,
            "register_score": score,
            "text": policy_text_ids[idx]["text"],
            "score_type": "decision_function",
        }
        push_highest(policy_like, top_n, score, payload)
        push_lowest(research_like, top_n, score, payload)

    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        score = np.load(shard.score_path).astype(np.float32)
        ids_rows = load_jsonl(shard.score_ids_path)
        if emb.shape[0] != len(ids_rows) or emb.shape[0] != score.shape[0]:
            raise RuntimeError(f"Research shard {shard.name} has mismatched embeddings/scores/ids.")
        decision_scores = emb @ coef + intercept
        assignments = score.argmax(axis=1)
        for row_idx in range(emb.shape[0]):
            payload = {
                "source_type": "research",
                "item_id": ids_rows[row_idx]["openalex_id"],
                "source_doc_or_year": ids_rows[row_idx].get("publication_year"),
                "assigned_sdg": int(assignments[row_idx]) + 1,
                "register_score": float(decision_scores[row_idx]),
                "score_type": "decision_function",
                "shard_name": shard.name,
                "row_in_shard": row_idx,
            }
            push_highest(policy_like, top_n, float(decision_scores[row_idx]), payload)
            push_lowest(research_like, top_n, float(decision_scores[row_idx]), payload)

    needed_research_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for _, _, payload in policy_like + research_like:
        if payload["source_type"] == "research":
            needed_research_rows[(payload["shard_name"], payload["row_in_shard"])] = payload

    for shard in shards:
        shard_targets = {
            row_idx: payload
            for (shard_name, row_idx), payload in needed_research_rows.items()
            if shard_name == shard.name
        }
        if not shard_targets:
            continue
        for row_idx, row in enumerate(iter_jsonl(shard.text_path)):
            payload = shard_targets.get(row_idx)
            if payload is None:
                continue
            payload["text"] = row.get("combined_text") or row.get("abstract") or row.get("title") or ""

    output_rows: list[dict[str, Any]] = []
    for rank, (_, _, payload) in enumerate(sorted(policy_like, key=lambda item: item[0], reverse=True), start=1):
        output_rows.append(
            {
                "rank_side": f"policy_like_{rank:02d}",
                "source_type": payload["source_type"],
                "item_id": payload["item_id"],
                "source_doc_or_year": payload["source_doc_or_year"],
                "assigned_sdg": payload["assigned_sdg"],
                "register_score": round(payload["register_score"], 6),
                "score_type": payload["score_type"],
                "text_snippet": strip_text(payload.get("text", "")),
            }
        )
    for rank, (_, _, payload) in enumerate(sorted(research_like, key=lambda item: -item[0], reverse=True), start=1):
        output_rows.append(
            {
                "rank_side": f"research_like_{rank:02d}",
                "source_type": payload["source_type"],
                "item_id": payload["item_id"],
                "source_doc_or_year": payload["source_doc_or_year"],
                "assigned_sdg": payload["assigned_sdg"],
                "register_score": round(payload["register_score"], 6),
                "score_type": payload["score_type"],
                "text_snippet": strip_text(payload.get("text", "")),
            }
        )
    return output_rows


def load_or_build_extreme_examples_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    classifier_signature: dict[str, Any],
    top_n: int,
    policy_emb: np.ndarray,
    policy_text_ids: list[dict[str, Any]],
    policy_score_ids: list[dict[str, Any]],
    policy_assignments: np.ndarray,
    shards: list[ResearchShard],
    coef: np.ndarray,
    intercept: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "extremes",
        "base_signature": base_signature,
        "classifier_signature": classifier_signature,
        "top_n": top_n,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "extremes", key)
    if cache_path.exists():
        log.info("Cache hit: extreme examples (%s)", key)
        return payload, load_json(cache_path)["rows"]

    log.info("Cache miss: extreme examples (%s)", key)
    rows = collect_extreme_examples(
        policy_emb=policy_emb,
        policy_text_ids=policy_text_ids,
        policy_score_ids=policy_score_ids,
        policy_assignments=policy_assignments,
        shards=shards,
        coef=coef,
        intercept=intercept,
        top_n=top_n,
    )
    write_json(cache_path, {"payload": payload, "rows": rows})
    return payload, rows


def write_extreme_examples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_side",
                "source_type",
                "item_id",
                "source_doc_or_year",
                "assigned_sdg",
                "register_score",
                "score_type",
                "text_snippet",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_tfidf_rows(
    research_rows: dict[int, dict[str, Any]],
    policy_text_ids: list[dict[str, Any]],
    policy_indices: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for idx in sorted(research_rows):
        row = research_rows[idx]
        texts.append(row["text"])
        labels.append(0)
    for idx in policy_indices.tolist():
        texts.append(policy_text_ids[int(idx)]["text"])
        labels.append(1)
    return texts, np.array(labels, dtype=np.int64)


def write_tfidf_terms_csv(path: Path, vectorizer: TfidfVectorizer, model: LogisticRegression, top_n: int = 50) -> None:
    terms = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    policy_idx = np.argsort(coef)[-top_n:][::-1]
    research_idx = np.argsort(coef)[:top_n]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["side", "term", "coefficient", "rank"])
        writer.writeheader()
        for rank, idx in enumerate(policy_idx, start=1):
            writer.writerow(
                {
                    "side": "policy",
                    "term": terms[idx],
                    "coefficient": round(float(coef[idx]), 6),
                    "rank": rank,
                }
            )
        for rank, idx in enumerate(research_idx, start=1):
            writer.writerow(
                {
                    "side": "research",
                    "term": terms[idx],
                    "coefficient": round(float(coef[idx]), 6),
                    "rank": rank,
                }
            )


def load_or_build_tfidf_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    sample_signature: dict[str, Any],
    selected_c: float,
    seed: int,
    tfidf_sample_per_class: int,
    tfidf_max_features: int,
    policy_text_ids: list[dict[str, Any]],
    research_sample_indices: np.ndarray,
    policy_sample_indices: np.ndarray,
    shards: list[ResearchShard],
) -> tuple[dict[str, Any], Path]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "tfidf",
        "base_signature": base_signature,
        "sample_signature": sample_signature,
        "selected_c": selected_c,
        "seed": seed,
        "tfidf_sample_per_class": tfidf_sample_per_class,
        "tfidf_max_features": tfidf_max_features,
    }
    key = stable_cache_key(payload)
    cache_path = cache_csv_path(cache_root, "tfidf", key)
    meta_path = cache_json_path(cache_root, "tfidf", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: tfidf interpretability terms (%s)", key)
        return payload, cache_path

    log.info("Cache miss: tfidf interpretability terms (%s)", key)
    tfidf_rng = np.random.default_rng(seed + 31)
    research_tfidf_indices = np.sort(
        tfidf_rng.choice(
            research_sample_indices,
            size=min(tfidf_sample_per_class, research_sample_indices.shape[0]),
            replace=False,
        ).astype(np.int64)
    )
    policy_tfidf_indices = np.sort(
        tfidf_rng.choice(
            policy_sample_indices,
            size=min(tfidf_sample_per_class, policy_sample_indices.shape[0]),
            replace=False,
        ).astype(np.int64)
    )
    research_tfidf_rows = load_research_texts_for_indices(shards, research_tfidf_indices)
    tfidf_texts, tfidf_labels = build_tfidf_rows(research_tfidf_rows, policy_text_ids, policy_tfidf_indices)
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, min_df=2)
    X_tfidf = vectorizer.fit_transform(tfidf_texts)
    tfidf_model = LogisticRegression(
        C=selected_c,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
        l1_ratio=0,
    )
    tfidf_model.fit(X_tfidf, tfidf_labels)
    write_tfidf_terms_csv(cache_path, vectorizer, tfidf_model)
    write_json(meta_path, payload)
    return payload, cache_path


def write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fit_classifier_bundle(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    fitted_models: dict[float, LogisticRegression] = {}
    for c_value in c_grid:
        model, train_metrics, val_metrics = fit_and_score_model(X_train, y_train, X_val, y_val, c_value, seed)
        fitted_models[c_value] = model
        candidate_rows.append(
            {
                "C": float(c_value),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "train_roc_auc": round(train_metrics["roc_auc"], 6),
                "train_f1": round(train_metrics["f1"], 6),
                "train_precision": round(train_metrics["precision"], 6),
                "train_recall": round(train_metrics["recall"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_roc_auc": round(val_metrics["roc_auc"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "val_precision": round(val_metrics["precision"], 6),
                "val_recall": round(val_metrics["recall"], 6),
            }
        )

    selected = select_best_candidate(candidate_rows)
    selected_c = float(selected["C"])
    selected_model = fitted_models[selected_c]
    test_scores = selected_model.decision_function(X_test)
    test_preds = selected_model.predict(X_test)
    test_metrics = metrics_for_scores(y_test, test_scores, test_preds)
    test_cm = confusion_matrix(y_test, test_preds, labels=[0, 1]).tolist()

    final_model = LogisticRegression(
        C=selected_c,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
        l1_ratio=0,
    )
    final_model.fit(X_train_val, y_train_val)
    coef = final_model.coef_[0].astype(np.float32)
    coef_norm = float(np.linalg.norm(coef))
    if coef_norm < 1e-8:
        raise RuntimeError("Classifier coefficient norm is near zero.")
    unit = (coef / coef_norm).astype(np.float32)
    intercept = float(final_model.intercept_[0])

    return {
        "candidate_rows": candidate_rows,
        "selected_model": selected,
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 6),
            "roc_auc": round(test_metrics["roc_auc"], 6),
            "f1": round(test_metrics["f1"], 6),
            "precision": round(test_metrics["precision"], 6),
            "recall": round(test_metrics["recall"], 6),
        },
        "test_confusion_matrix": {
            "labels": {"0": "research", "1": "policy"},
            "matrix": test_cm,
        },
        "coef": coef,
        "coefficient_norm": round(coef_norm, 6),
        "unit": unit,
        "intercept": round(intercept, 6),
    }


def project_out_multiple_directions_and_normalize(
    emb: np.ndarray,
    directions: list[np.ndarray],
) -> np.ndarray:
    adjusted = emb.astype(np.float32, copy=True)
    for direction in directions:
        adjusted = project_out_direction_and_normalize(adjusted, direction)
    return adjusted.astype(np.float32)


def build_policy_indices_by_sdg(policy_assignments: np.ndarray) -> dict[int, np.ndarray]:
    return {
        sdg_idx: np.flatnonzero(policy_assignments == sdg_idx).astype(np.int64)
        for sdg_idx in range(N_SDG)
    }


def load_or_build_research_indices_by_sdg_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    shards: list[ResearchShard],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "local_research_indices_by_sdg",
        "base_signature": base_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "local_research_indices_by_sdg", key)
    meta_path = cache_json_path(cache_root, "local_research_indices_by_sdg", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: research indices by SDG (%s)", key)
        cached = np.load(cache_path)
        return payload, {sdg_idx: cached[f"sdg_{sdg_idx}"].astype(np.int64) for sdg_idx in range(N_SDG)}

    log.info("Cache miss: research indices by SDG (%s)", key)
    parts: dict[int, list[np.ndarray]] = {sdg_idx: [] for sdg_idx in range(N_SDG)}
    for shard in shards:
        score = np.load(shard.score_path).astype(np.float32)
        assignments = score.argmax(axis=1)
        local_rows = np.arange(shard.rows, dtype=np.int64) + shard.start
        for sdg_idx in range(N_SDG):
            mask = assignments == sdg_idx
            if np.any(mask):
                parts[sdg_idx].append(local_rows[mask])

    arrays: dict[int, np.ndarray] = {}
    for sdg_idx in range(N_SDG):
        if parts[sdg_idx]:
            arrays[sdg_idx] = np.concatenate(parts[sdg_idx]).astype(np.int64)
        else:
            arrays[sdg_idx] = np.empty(0, dtype=np.int64)

    ensure_dir(cache_path.parent)
    np.savez(cache_path, **{f"sdg_{sdg_idx}": arr for sdg_idx, arr in arrays.items()})
    write_json(meta_path, payload)
    return payload, arrays


def resolve_samples_per_cell(
    *,
    research_indices_by_sdg: dict[int, np.ndarray],
    policy_indices_by_sdg: dict[int, np.ndarray],
    requested_samples_per_cell: int | None,
    min_samples_per_class: int,
) -> tuple[int, list[dict[str, int]]]:
    cell_rows: list[dict[str, int]] = []
    min_available: int | None = None
    for sdg_idx in range(N_SDG):
        research_n = int(research_indices_by_sdg[sdg_idx].shape[0])
        policy_n = int(policy_indices_by_sdg[sdg_idx].shape[0])
        cell_min = min(research_n, policy_n)
        cell_rows.append(
            {
                "sdg": sdg_idx + 1,
                "research_available": research_n,
                "policy_available": policy_n,
                "cell_min": cell_min,
            }
        )
        min_available = cell_min if min_available is None else min(min_available, cell_min)
    if min_available is None or min_available <= 0:
        raise RuntimeError("Could not determine a positive SDG x register cell minimum.")
    effective = min_available if requested_samples_per_cell is None else min(int(requested_samples_per_cell), min_available)
    if effective < min_samples_per_class:
        raise RuntimeError(
            f"Effective samples_per_cell={effective} is below min_samples_per_class={min_samples_per_class}."
        )
    for row in cell_rows:
        row["sampled_per_register"] = effective
    return effective, cell_rows


def load_or_build_sdg_balanced_sample_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    research_indices_by_sdg_signature: dict[str, Any],
    research_indices_by_sdg: dict[int, np.ndarray],
    policy_indices_by_sdg: dict[int, np.ndarray],
    requested_samples_per_cell: int | None,
    min_samples_per_class: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "sdg_balanced_sample",
        "base_signature": base_signature,
        "research_indices_by_sdg_signature": research_indices_by_sdg_signature,
        "requested_samples_per_cell": requested_samples_per_cell,
        "min_samples_per_class": min_samples_per_class,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "sdg_balanced_sample", key)
    if cache_path.exists():
        log.info("Cache hit: SDG-balanced sample plan (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: SDG-balanced sample plan (%s)", key)
    effective_samples_per_cell, cell_rows = resolve_samples_per_cell(
        research_indices_by_sdg=research_indices_by_sdg,
        policy_indices_by_sdg=policy_indices_by_sdg,
        requested_samples_per_cell=requested_samples_per_cell,
        min_samples_per_class=min_samples_per_class,
    )
    sampled_research: dict[str, list[int]] = {}
    sampled_policy: dict[str, list[int]] = {}
    for sdg_idx in range(N_SDG):
        sampled_research[f"sdg_{sdg_idx + 1}"] = sample_sorted_from_pool(
            research_indices_by_sdg[sdg_idx],
            effective_samples_per_cell,
            seed + 101 * (sdg_idx + 1),
        ).tolist()
        sampled_policy[f"sdg_{sdg_idx + 1}"] = sample_sorted_from_pool(
            policy_indices_by_sdg[sdg_idx],
            effective_samples_per_cell,
            seed + 101 * (sdg_idx + 1) + 1,
        ).tolist()
    payload_out = {
        "effective_samples_per_cell": effective_samples_per_cell,
        "cell_rows": cell_rows,
        "sampled_research_indices": sampled_research,
        "sampled_policy_indices": sampled_policy,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def assemble_sdg_balanced_dataset(
    *,
    shards: list[ResearchShard],
    policy_emb: np.ndarray,
    sampled_plan: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    strata_parts: list[np.ndarray] = []
    for sdg_idx in range(N_SDG):
        key = f"sdg_{sdg_idx + 1}"
        research_indices = np.array(sampled_plan["sampled_research_indices"][key], dtype=np.int64)
        policy_indices = np.array(sampled_plan["sampled_policy_indices"][key], dtype=np.int64)
        research_emb = collect_research_embeddings(shards, research_indices)
        X_parts.append(research_emb)
        y_parts.append(np.zeros(research_emb.shape[0], dtype=np.int64))
        strata_parts.append(np.full(research_emb.shape[0], sdg_idx * 2, dtype=np.int64))
        X_parts.append(policy_emb[policy_indices].astype(np.float32))
        y_parts.append(np.ones(policy_indices.shape[0], dtype=np.int64))
        strata_parts.append(np.full(policy_indices.shape[0], sdg_idx * 2 + 1, dtype=np.int64))
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    strata = np.concatenate(strata_parts)
    return X, y, strata


def load_or_build_sdg_balanced_dataset_cache(
    *,
    cache_root: Path,
    sample_signature: dict[str, Any],
    shards: list[ResearchShard],
    policy_emb: np.ndarray,
    sampled_plan: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "sdg_balanced_dataset",
        "sample_signature": sample_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "sdg_balanced_dataset", key)
    meta_path = cache_json_path(cache_root, "sdg_balanced_dataset", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: SDG-balanced dataset (%s)", key)
        cached = np.load(cache_path)
        return payload, cached["X"].astype(np.float32), cached["y"], cached["strata"]
    log.info("Cache miss: SDG-balanced dataset (%s)", key)
    X, y, strata = assemble_sdg_balanced_dataset(
        shards=shards,
        policy_emb=policy_emb,
        sampled_plan=sampled_plan,
    )
    ensure_dir(cache_path.parent)
    np.savez(cache_path, X=X, y=y, strata=strata)
    write_json(meta_path, payload)
    return payload, X, y, strata


def merge_method_gap_results(
    raw_results: list[dict[str, Any]],
    adjusted_results: list[dict[str, Any]],
    *,
    adjusted_gap_field: str,
    adjusted_similarity_field: str,
    skipped_sdgs: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    adjusted_map = {int(row["sdg"]): row for row in adjusted_results}
    skipped_sdgs = skipped_sdgs or {}
    merged_rows: list[dict[str, Any]] = []
    for raw_row in raw_results:
        sdg = int(raw_row["sdg"])
        skip_reason = skipped_sdgs.get(sdg)
        adjusted_row = adjusted_map.get(sdg)
        adjusted_similarity = None if adjusted_row is None else adjusted_row["semantic_similarity"]
        adjusted_gap = None if adjusted_row is None else adjusted_row["semantic_gap"]
        adjusted_research_cohesion = None if adjusted_row is None else adjusted_row["research_cohesion"]
        adjusted_policy_cohesion = None if adjusted_row is None else adjusted_row["policy_cohesion"]
        delta_gap = (
            None
            if raw_row["semantic_gap"] is None or adjusted_gap is None
            else round(float(raw_row["semantic_gap"]) - float(adjusted_gap), 6)
        )
        merged_rows.append(
            {
                "sdg": sdg,
                "n_papers": raw_row["n_papers"],
                "n_policy_segments": raw_row["n_policy_segments"],
                "n_policy_segments_capped": raw_row["n_policy_segments_capped"],
                "n_policy_docs": raw_row["n_policy_docs"],
                "n_policy_docs_capped": raw_row["n_policy_docs_capped"],
                "segment_cap": raw_row["segment_cap"],
                "raw_similarity": raw_row["semantic_similarity"],
                "raw_gap": raw_row["semantic_gap"],
                adjusted_similarity_field: adjusted_similarity,
                adjusted_gap_field: adjusted_gap,
                "delta_gap": delta_gap,
                "raw_research_cohesion": raw_row["research_cohesion"],
                "raw_policy_cohesion": raw_row["policy_cohesion"],
                f"{adjusted_gap_field}_research_cohesion": adjusted_research_cohesion,
                f"{adjusted_gap_field}_policy_cohesion": adjusted_policy_cohesion,
                "classifier_available": skip_reason is None,
                "skip_reason": skip_reason,
                "unreliable": bool(raw_row["unreliable"]) or (adjusted_row is not None and bool(adjusted_row["unreliable"])),
                "unreliable_reason": (
                    skip_reason
                    if skip_reason is not None
                    else (
                        adjusted_row["unreliable_reason"]
                        if adjusted_row is not None
                        else raw_row["unreliable_reason"]
                    )
                ),
            }
        )
    return merged_rows


def load_or_build_sdg_balanced_method_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    sample_signature: dict[str, Any],
    classifier_type: str,
    test_size: float,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "sdg_balanced_method",
        "base_signature": base_signature,
        "sample_signature": sample_signature,
        "classifier_type": classifier_type,
        "test_size": test_size,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "sdg_balanced_method", key)
    if cache_path.exists():
        log.info("Cache hit: SDG-balanced classifier (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: SDG-balanced classifier (%s)", key)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=strata,
        random_state=seed,
    )
    model, metrics = fit_binary_classifier_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifier_type=classifier_type,
        seed=seed,
    )
    coef = model.coef_[0].astype(np.float32)
    unit = normalize_unit_vector(coef, label="SDG-balanced register direction")
    payload_out = {
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "metrics": metrics,
        "coef": [float(v) for v in coef.tolist()],
        "register_unit": [float(v) for v in unit.tolist()],
        "intercept": round(float(model.intercept_[0]), 6),
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def build_within_sdg_adjusted_policy_embeddings(
    *,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    vectors_by_sdg: dict[int, np.ndarray],
) -> np.ndarray:
    adjusted = policy_emb.astype(np.float32, copy=True)
    for sdg_idx, direction in vectors_by_sdg.items():
        mask = policy_assignments == sdg_idx
        if not np.any(mask):
            continue
        adjusted[mask] = project_out_direction_and_normalize(adjusted[mask], direction)
    return adjusted.astype(np.float32)


def build_within_sdg_adjusted_research_centroids(
    *,
    shards: list[ResearchShard],
    vectors_by_sdg: dict[int, np.ndarray],
    raw_research_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not vectors_by_sdg:
        raise RuntimeError("No within-SDG register directions were learned.")
    dim = next(iter(vectors_by_sdg.values())).shape[0]
    sums = np.zeros((N_SDG, dim), dtype=np.float64)
    counts_for_compute = np.zeros(N_SDG, dtype=np.int64)
    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        score = np.load(shard.score_path).astype(np.float32)
        assignments = score.argmax(axis=1)
        for sdg_idx, direction in vectors_by_sdg.items():
            mask = assignments == sdg_idx
            if not np.any(mask):
                continue
            adjusted = project_out_direction_and_normalize(emb[mask], direction)
            sums[sdg_idx] += adjusted.sum(axis=0)
            counts_for_compute[sdg_idx] += adjusted.shape[0]
    centroids = np.zeros((N_SDG, dim), dtype=np.float32)
    cohesions = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx, count in enumerate(counts_for_compute.tolist()):
        if count == 0:
            continue
        raw = sums[sdg_idx] / float(count)
        centroids[sdg_idx] = normalize_unit_vector(raw.astype(np.float32), label=f"Within-SDG centroid for SDG {sdg_idx + 1}")
        cohesions[sdg_idx] = float(np.linalg.norm(raw))
    return centroids, counts_for_compute, cohesions


def load_or_build_within_sdg_adjusted_policy_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    within_signature: dict[str, Any],
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    vectors_by_sdg: dict[int, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "within_sdg_adjusted_policy",
        "base_signature": base_signature,
        "within_signature": within_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npy_path(cache_root, "within_sdg_adjusted_policy", key)
    meta_path = cache_json_path(cache_root, "within_sdg_adjusted_policy", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: within-SDG adjusted policy embeddings (%s)", key)
        return payload, np.load(cache_path).astype(np.float32)

    log.info("Cache miss: within-SDG adjusted policy embeddings (%s)", key)
    adjusted = build_within_sdg_adjusted_policy_embeddings(
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        vectors_by_sdg=vectors_by_sdg,
    )
    ensure_dir(cache_path.parent)
    np.save(cache_path, adjusted.astype(np.float32))
    write_json(meta_path, payload)
    return payload, adjusted


def load_or_build_within_sdg_adjusted_research_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    within_signature: dict[str, Any],
    shards: list[ResearchShard],
    vectors_by_sdg: dict[int, np.ndarray],
    raw_research_counts: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "within_sdg_adjusted_research",
        "base_signature": base_signature,
        "within_signature": within_signature,
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "within_sdg_adjusted_research", key)
    meta_path = cache_json_path(cache_root, "within_sdg_adjusted_research", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: within-SDG adjusted research aggregates (%s)", key)
        cached = np.load(cache_path)
        return (
            payload,
            cached["centroids"].astype(np.float32),
            cached["counts"].astype(np.int64),
            cached["cohesions"].astype(np.float32),
        )

    log.info("Cache miss: within-SDG adjusted research aggregates (%s)", key)
    centroids, counts, cohesions = build_within_sdg_adjusted_research_centroids(
        shards=shards,
        vectors_by_sdg=vectors_by_sdg,
        raw_research_counts=raw_research_counts,
    )
    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        centroids=centroids.astype(np.float32),
        counts=counts.astype(np.int64),
        cohesions=cohesions.astype(np.float32),
    )
    write_json(meta_path, payload)
    return payload, centroids, counts, cohesions


def load_or_build_within_sdg_method_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    research_indices_by_sdg_signature: dict[str, Any],
    research_indices_by_sdg: dict[int, np.ndarray],
    policy_indices_by_sdg: dict[int, np.ndarray],
    policy_emb: np.ndarray,
    shards: list[ResearchShard],
    requested_samples_per_cell: int | None,
    min_samples_per_class: int,
    test_size: float,
    classifier_type: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "within_sdg_method",
        "base_signature": base_signature,
        "research_indices_by_sdg_signature": research_indices_by_sdg_signature,
        "requested_samples_per_cell": requested_samples_per_cell,
        "min_samples_per_class": min_samples_per_class,
        "test_size": test_size,
        "classifier_type": classifier_type,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "within_sdg_method", key)
    if cache_path.exists():
        log.info("Cache hit: within-SDG classifiers (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: within-SDG classifiers (%s)", key)
    metrics_rows: list[dict[str, Any]] = []
    vectors = np.full((N_SDG, 384), np.nan, dtype=np.float32)
    vectors_by_sdg: dict[int, np.ndarray] = {}
    skipped_sdgs: dict[int, str] = {}
    for sdg_idx in range(N_SDG):
        log.info("Fitting within-SDG classifier SDG %d/%d", sdg_idx + 1, N_SDG)
        research_pool = research_indices_by_sdg[sdg_idx]
        policy_pool = policy_indices_by_sdg[sdg_idx]
        available_research = int(research_pool.shape[0])
        available_policy = int(policy_pool.shape[0])
        effective_samples = min(
            available_research,
            available_policy,
            requested_samples_per_cell if requested_samples_per_cell is not None else max(available_research, available_policy),
        )
        if effective_samples < min_samples_per_class:
            skip_reason = (
                f"insufficient samples for SDG {sdg_idx + 1}: "
                f"research={available_research}, policy={available_policy}, min_required={min_samples_per_class}"
            )
            skipped_sdgs[sdg_idx + 1] = skip_reason
            metrics_rows.append(
                {
                    "sdg": sdg_idx + 1,
                    "available_research": available_research,
                    "available_policy": available_policy,
                    "sampled_per_class": effective_samples,
                    "train_size": 0,
                    "test_size": 0,
                    "accuracy": None,
                    "macro_f1": None,
                    "roc_auc": None,
                    "coefficient_norm": None,
                    "intercept": None,
                    "classifier_available": False,
                    "skip_reason": skip_reason,
                }
            )
            log.warning("%s", skip_reason)
            continue
        research_indices = sample_sorted_from_pool(research_pool, effective_samples, seed + 5000 + sdg_idx * 10)
        policy_indices = sample_sorted_from_pool(policy_pool, effective_samples, seed + 5000 + sdg_idx * 10 + 1)
        X = np.vstack(
            [
                collect_research_embeddings(shards, research_indices),
                policy_emb[policy_indices].astype(np.float32),
            ]
        ).astype(np.float32)
        y = np.concatenate(
            [
                np.zeros(effective_samples, dtype=np.int64),
                np.ones(effective_samples, dtype=np.int64),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=seed + sdg_idx,
        )
        model, metrics = fit_binary_classifier_train_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            classifier_type=classifier_type,
            seed=seed + sdg_idx,
        )
        coef = model.coef_[0].astype(np.float32)
        unit = normalize_unit_vector(coef, label=f"Within-SDG register direction for SDG {sdg_idx + 1}")
        vectors[sdg_idx] = unit
        vectors_by_sdg[sdg_idx] = unit
        metrics_rows.append(
            {
                "sdg": sdg_idx + 1,
                "available_research": available_research,
                "available_policy": available_policy,
                "sampled_per_class": effective_samples,
                "train_size": int(X_train.shape[0]),
                "test_size": int(X_test.shape[0]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "coefficient_norm": metrics["coefficient_norm"],
                "intercept": metrics["intercept"],
                "classifier_available": True,
                "skip_reason": "",
            }
        )

    available_units = [vectors_by_sdg[idx] for idx in sorted(vectors_by_sdg)]
    if available_units:
        avg_direction = normalize_unit_vector(np.mean(np.vstack(available_units), axis=0), label="Average within-SDG register direction")
        cosine_global_vs_avg = None
    else:
        avg_direction = np.full(384, np.nan, dtype=np.float32)
        cosine_global_vs_avg = None
    payload_out = {
        "metrics_rows": metrics_rows,
        "vectors": vectors.tolist(),
        "available_sdgs": [sdg_idx + 1 for sdg_idx in sorted(vectors_by_sdg)],
        "skipped_sdgs": skipped_sdgs,
        "average_direction": avg_direction.tolist(),
        "cosine_global_vs_average": cosine_global_vs_avg,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def build_regression_cell_stats(
    *,
    shards: list[ResearchShard],
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((2, N_SDG), dtype=np.int64)
    sums = np.zeros((2, N_SDG, policy_emb.shape[1]), dtype=np.float64)

    policy_counts = np.bincount(policy_assignments, minlength=N_SDG).astype(np.int64)
    counts[1] = policy_counts
    for sdg_idx in range(N_SDG):
        mask = policy_assignments == sdg_idx
        if np.any(mask):
            sums[1, sdg_idx] = policy_emb[mask].sum(axis=0)

    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        score = np.load(shard.score_path).astype(np.float32)
        assignments = score.argmax(axis=1)
        counts[0] += np.bincount(assignments, minlength=N_SDG).astype(np.int64)
        for sdg_idx in np.unique(assignments):
            mask = assignments == sdg_idx
            sums[0, sdg_idx] += emb[mask].sum(axis=0)

    return counts, sums


def load_or_build_regression_cell_stats_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    shards: list[ResearchShard],
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "regression_cell_stats",
        "base_signature": base_signature,
        "n_policy_rows": int(policy_emb.shape[0]),
    }
    key = stable_cache_key(payload)
    cache_path = cache_npz_path(cache_root, "regression_cell_stats", key)
    meta_path = cache_json_path(cache_root, "regression_cell_stats", key)
    if cache_path.exists() and meta_path.exists():
        log.info("Cache hit: regression cell stats (%s)", key)
        cached = np.load(cache_path)
        return payload, cached["counts"].astype(np.int64), cached["sums"].astype(np.float64)

    log.info("Cache miss: regression cell stats (%s)", key)
    counts, sums = build_regression_cell_stats(
        shards=shards,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
    )
    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        counts=counts.astype(np.int64),
        sums=sums.astype(np.float64),
    )
    write_json(meta_path, payload)
    return payload, counts, sums


def solve_regression_vectors_from_cell_stats(
    counts: np.ndarray,
    sums: np.ndarray,
) -> dict[str, Any]:
    if counts.shape != (2, N_SDG):
        raise ValueError(f"Expected counts shape (2, {N_SDG}), got {counts.shape}")
    if sums.shape[:2] != (2, N_SDG):
        raise ValueError(f"Expected sums shape (2, {N_SDG}, dim), got {sums.shape}")
    if np.any(counts <= 0):
        bad = np.argwhere(counts <= 0)
        raise RuntimeError(
            "Regression register decomposition requires non-empty register x SDG cells; "
            f"missing cells: {[(('research' if g == 0 else 'policy'), int(sdg) + 1) for g, sdg in bad.tolist()]}"
        )

    means = sums / counts[:, :, None]
    research_means = means[0]
    policy_means = means[1]

    beta0 = research_means[REGRESSION_BASELINE_SDG - 1].astype(np.float32)
    beta_register = (policy_means[REGRESSION_BASELINE_SDG - 1] - research_means[REGRESSION_BASELINE_SDG - 1]).astype(np.float32)
    gamma = (research_means[1:] - research_means[REGRESSION_BASELINE_SDG - 1]).astype(np.float32)
    within_raw = (policy_means - research_means).astype(np.float32)
    delta = (within_raw[1:] - beta_register[None, :]).astype(np.float32)

    global_unit = normalize_unit_vector(beta_register, label="Regression global register direction")
    within_units = np.vstack(
        [
            normalize_unit_vector(within_raw[sdg_idx], label=f"Regression within-SDG register direction for SDG {sdg_idx + 1}")
            for sdg_idx in range(N_SDG)
        ]
    ).astype(np.float32)

    cell_rows = [
        {
            "sdg": sdg_idx + 1,
            "research_n": int(counts[0, sdg_idx]),
            "policy_n": int(counts[1, sdg_idx]),
        }
        for sdg_idx in range(N_SDG)
    ]
    return {
        "baseline_sdg": REGRESSION_BASELINE_SDG,
        "counts": counts.astype(np.int64),
        "means": means.astype(np.float32),
        "beta0": beta0,
        "beta_register": beta_register,
        "gamma": gamma,
        "delta": delta,
        "within_raw": within_raw,
        "global_unit": global_unit,
        "within_units": within_units,
        "cell_rows": cell_rows,
    }


def load_or_build_regression_method_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    cell_stats_signature: dict[str, Any],
    counts: np.ndarray,
    sums: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "regression_method",
        "base_signature": base_signature,
        "cell_stats_signature": cell_stats_signature,
        "baseline_sdg": REGRESSION_BASELINE_SDG,
    }
    key = stable_cache_key(payload)
    cache_npz = cache_npz_path(cache_root, "regression_method", key)
    cache_json = cache_json_path(cache_root, "regression_method", key)
    if cache_npz.exists() and cache_json.exists():
        log.info("Cache hit: regression register vectors (%s)", key)
        cached = np.load(cache_npz)
        meta = load_json(cache_json)
        return payload, {
            "baseline_sdg": int(meta["baseline_sdg"]),
            "cell_rows": meta["cell_rows"],
            "global_vector": cached["global_vector"].astype(np.float32),
            "global_unit": cached["global_unit"].astype(np.float32),
            "within_raw": cached["within_raw"].astype(np.float32),
            "within_units": cached["within_units"].astype(np.float32),
            "means": cached["means"].astype(np.float32),
            "counts": cached["counts"].astype(np.int64),
            "gamma": cached["gamma"].astype(np.float32),
            "delta": cached["delta"].astype(np.float32),
            "beta0": cached["beta0"].astype(np.float32),
            "global_vector_norm": float(meta["global_vector_norm"]),
        }

    log.info("Cache miss: regression register vectors (%s)", key)
    solved = solve_regression_vectors_from_cell_stats(counts, sums)
    ensure_dir(cache_npz.parent)
    np.savez(
        cache_npz,
        global_vector=solved["beta_register"].astype(np.float32),
        global_unit=solved["global_unit"].astype(np.float32),
        within_raw=solved["within_raw"].astype(np.float32),
        within_units=solved["within_units"].astype(np.float32),
        means=solved["means"].astype(np.float32),
        counts=solved["counts"].astype(np.int64),
        gamma=solved["gamma"].astype(np.float32),
        delta=solved["delta"].astype(np.float32),
        beta0=solved["beta0"].astype(np.float32),
    )
    write_json(
        cache_json,
        {
            "baseline_sdg": int(solved["baseline_sdg"]),
            "cell_rows": solved["cell_rows"],
            "global_vector_norm": round(float(np.linalg.norm(solved["beta_register"])), 6),
        },
    )
    return payload, {
        "baseline_sdg": int(solved["baseline_sdg"]),
        "cell_rows": solved["cell_rows"],
        "global_vector": solved["beta_register"].astype(np.float32),
        "global_unit": solved["global_unit"].astype(np.float32),
        "within_raw": solved["within_raw"].astype(np.float32),
        "within_units": solved["within_units"].astype(np.float32),
        "means": solved["means"].astype(np.float32),
        "counts": solved["counts"].astype(np.int64),
        "gamma": solved["gamma"].astype(np.float32),
        "delta": solved["delta"].astype(np.float32),
        "beta0": solved["beta0"].astype(np.float32),
        "global_vector_norm": float(np.linalg.norm(solved["beta_register"])),
    }


def build_regression_alignment_rows(
    *,
    classifier_alignment_rows: list[dict[str, Any]],
    regression_global_unit: np.ndarray,
    regression_within_units: np.ndarray,
    sdg_centroids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classifier_map = {int(row["sdg"]): row for row in classifier_alignment_rows}
    rows: list[dict[str, Any]] = []
    global_abs_vals: list[float] = []
    for sdg_idx in range(N_SDG):
        global_cos = float(np.dot(regression_global_unit, sdg_centroids[sdg_idx]))
        within_cos = float(np.dot(regression_within_units[sdg_idx], sdg_centroids[sdg_idx]))
        global_abs = abs(global_cos)
        global_abs_vals.append(global_abs)
        classifier_row = classifier_map.get(sdg_idx + 1)
        rows.append(
            {
                "sdg": sdg_idx + 1,
                "classifier_cosine_similarity": None if classifier_row is None else classifier_row["cosine_similarity"],
                "regression_cosine_similarity": round(global_cos, 6),
                "regression_abs_cosine_similarity": round(global_abs, 6),
                "regression_within_sdg_cosine_similarity": round(within_cos, 6),
                "regression_within_sdg_abs_cosine_similarity": round(abs(within_cos), 6),
            }
        )

    strongest = max(rows, key=lambda row: float(row["regression_abs_cosine_similarity"]))
    weakest = min(rows, key=lambda row: float(row["regression_abs_cosine_similarity"]))
    summary = {
        "mean_absolute_cosine": round(float(np.mean(global_abs_vals)), 6),
        "median_absolute_cosine": round(float(np.median(global_abs_vals)), 6),
        "max_absolute_cosine": round(float(np.max(global_abs_vals)), 6),
        "min_absolute_cosine": round(float(np.min(global_abs_vals)), 6),
        "strongest_alignment_sdg": int(strongest["sdg"]),
        "strongest_alignment_value": float(strongest["regression_cosine_similarity"]),
        "weakest_alignment_sdg": int(weakest["sdg"]),
        "weakest_alignment_value": float(weakest["regression_cosine_similarity"]),
    }
    return rows, summary


def build_regression_similarity_payload(
    *,
    regression_global_unit: np.ndarray,
    classifier_global_unit: np.ndarray,
    regression_alignment_summary: dict[str, Any],
    classifier_alignment_summary: dict[str, Any],
    baseline_sdg: int,
) -> dict[str, Any]:
    cosine = float(np.dot(regression_global_unit, classifier_global_unit))
    if cosine > 0.8:
        interpretation = "essentially_same"
    elif cosine < 0.4:
        interpretation = "substantially_different"
    else:
        interpretation = "partial_agreement"
    return {
        "analysis": "rodriguez_style_embedding_regression",
        "baseline_sdg": baseline_sdg,
        "global_regression_vs_classifier_cosine": round(cosine, 6),
        "agreement_band": interpretation,
        "classifier_vector_source": str(
            Path("4_outputs") / ROBUSTNESS_OUTPUT_SUBDIR / "sdg_balanced_register_vector.npy"
        ),
        "regression_vector_source": str(
            Path("4_outputs") / ROBUSTNESS_OUTPUT_SUBDIR / "regression_register_vector.npy"
        ),
        "classifier_alignment_summary": classifier_alignment_summary,
        "regression_alignment_summary": regression_alignment_summary,
    }


def build_regression_gap_comparison_rows(
    *,
    raw_rows: list[dict[str, Any]],
    classifier_global_rows: list[dict[str, Any]] | None,
    classifier_within_rows: list[dict[str, Any]] | None,
    regression_global_rows: list[dict[str, Any]],
    regression_within_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    classifier_global_map = {} if classifier_global_rows is None else {int(row["sdg"]): row for row in classifier_global_rows}
    classifier_within_map = {} if classifier_within_rows is None else {int(row["sdg"]): row for row in classifier_within_rows}
    regression_global_map = {int(row["sdg"]): row for row in regression_global_rows}
    regression_within_map = {int(row["sdg"]): row for row in regression_within_rows}

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        sdg = int(raw_row["sdg"])
        clf_global = classifier_global_map.get(sdg)
        clf_within = classifier_within_map.get(sdg)
        reg_global = regression_global_map[sdg]
        reg_within = regression_within_map[sdg]
        rows.append(
            {
                "sdg": sdg,
                "n_papers": raw_row["n_papers"],
                "n_policy_segments_capped": raw_row["n_policy_segments_capped"],
                "n_policy_docs_capped": raw_row["n_policy_docs_capped"],
                "raw_gap": raw_row["semantic_gap"],
                "classifier_global_adjusted_gap": None if clf_global is None else clf_global["sdg_balanced_adjusted_gap"],
                "classifier_global_delta_gap": None if clf_global is None else clf_global["delta_gap"],
                "regression_global_adjusted_gap": reg_global["regression_global_adjusted_gap"],
                "regression_global_delta_gap": reg_global["delta_gap"],
                "classifier_within_sdg_adjusted_gap": None if clf_within is None else clf_within["within_sdg_adjusted_gap"],
                "classifier_within_sdg_delta_gap": None if clf_within is None else clf_within["delta_gap"],
                "regression_within_sdg_adjusted_gap": reg_within["regression_within_sdg_adjusted_gap"],
                "regression_within_sdg_delta_gap": reg_within["delta_gap"],
                "unreliable": bool(raw_row["unreliable"]) or bool(reg_global["unreliable"]) or bool(reg_within["unreliable"]),
                "unreliable_reason": raw_row["unreliable_reason"] or reg_global["unreliable_reason"] or reg_within["unreliable_reason"],
            }
        )
    return rows


def write_regression_gap_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_csv(
        path,
        [
            "sdg",
            "n_papers",
            "n_policy_segments_capped",
            "n_policy_docs_capped",
            "raw_gap",
            "classifier_global_adjusted_gap",
            "classifier_global_delta_gap",
            "regression_global_adjusted_gap",
            "regression_global_delta_gap",
            "classifier_within_sdg_adjusted_gap",
            "classifier_within_sdg_delta_gap",
            "regression_within_sdg_adjusted_gap",
            "regression_within_sdg_delta_gap",
            "unreliable",
            "unreliable_reason",
        ],
        rows,
    )


def plot_regression_vs_classifier_alignment(
    figures_dir: Path,
    *,
    rows: list[dict[str, Any]],
    similarity_payload: dict[str, Any],
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["sdg"]))
    xs = np.arange(1, N_SDG + 1)
    classifier_vals = [float(row["classifier_cosine_similarity"]) for row in ordered]
    regression_vals = [float(row["regression_cosine_similarity"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.36
    ax.bar(xs - width / 2, classifier_vals, width=width, color="#BDBDBD", label="Classifier global")
    ax.bar(xs + width / 2, regression_vals, width=width, color="#D95F02", label="Regression global")
    ax.axhline(0.0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("SDG")
    ax.set_ylabel("Cosine similarity with SDG centroid")
    ax.set_title("Classifier vs regression register-direction alignment", fontsize=8.5, loc="left")
    ax.text(
        0.01,
        0.98,
        f"cos(g_reg, g_classifier) = {float(similarity_payload['global_regression_vs_classifier_cosine']):.3f}\n"
        f"agreement: {similarity_payload['agreement_band'].replace('_', ' ')}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "boxstyle": "round,pad=0.3"},
    )
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_regression_vs_classifier_alignment.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_regression_vs_classifier_alignment.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_regression_latex_outputs(
    tables_dir: Path,
    *,
    regression_alignment_rows: list[dict[str, Any]],
    regression_alignment_summary: dict[str, Any],
    similarity_payload: dict[str, Any],
    regression_gap_rows: list[dict[str, Any]],
) -> None:
    mean_regression_global_gap = mean_gap(regression_gap_rows, "regression_global_adjusted_gap")
    mean_regression_within_gap = mean_gap(regression_gap_rows, "regression_within_sdg_adjusted_gap")
    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/3_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\RegressionRegisterVsClassifierCosine}}{{{float(similarity_payload['global_regression_vs_classifier_cosine']):.3f}}}",
        rf"\newcommand{{\RegressionRegisterAgreementBand}}{{{similarity_payload['agreement_band'].replace('_', ' ')}}}",
        rf"\newcommand{{\RegressionRegisterMeanAbsCosine}}{{{float(regression_alignment_summary['mean_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegressionRegisterMedianAbsCosine}}{{{float(regression_alignment_summary['median_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegressionRegisterMaxAbsCosine}}{{{float(regression_alignment_summary['max_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegressionRegisterStrongestSdg}}{{SDG {int(regression_alignment_summary['strongest_alignment_sdg'])}}}",
        rf"\newcommand{{\RegressionRegisterWeakestSdg}}{{SDG {int(regression_alignment_summary['weakest_alignment_sdg'])}}}",
        rf"\newcommand{{\RegressionGlobalMeanGap}}{{{'' if mean_regression_global_gap is None else f'{mean_regression_global_gap:.3f}'}}}",
        rf"\newcommand{{\RegressionWithinMeanGap}}{{{'' if mean_regression_within_gap is None else f'{mean_regression_within_gap:.3f}'}}}",
    ]
    (tables_dir / "num_regression_register_alignment.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    table_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"SDG & $\cos(g_{\mathrm{clf}}, c_k)$ & $\cos(g_{\mathrm{reg}}, c_k)$ & $\cos(g_{\mathrm{reg},k}, c_k)$ \\",
        r"\midrule",
    ]
    for row in regression_alignment_rows:
        table_lines.append(
            rf"SDG {int(row['sdg'])} & "
            rf"{float(row['classifier_cosine_similarity']):.3f} & "
            rf"{float(row['regression_cosine_similarity']):.3f} & "
            rf"{float(row['regression_within_sdg_cosine_similarity']):.3f} \\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_regression_register_alignment.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")


def build_register_vector_cosine_rows(
    *,
    global_unit: np.ndarray,
    within_vectors: np.ndarray,
    available_sdgs: list[int],
    average_direction: np.ndarray,
) -> tuple[list[dict[str, Any]], float | None]:
    rows: list[dict[str, Any]] = []
    cosine_global_vs_avg: float | None = None
    if available_sdgs and np.all(np.isfinite(average_direction)):
        cosine_global_vs_avg = round(float(np.dot(global_unit, average_direction)), 6)
        rows.append(
            {
                "comparison_type": "global_vs_average",
                "left": "sdg_balanced_global",
                "right": "within_sdg_average",
                "cosine_similarity": cosine_global_vs_avg,
            }
        )
    for sdg in available_sdgs:
        unit = within_vectors[sdg - 1]
        rows.append(
            {
                "comparison_type": "global_vs_within_sdg",
                "left": "sdg_balanced_global",
                "right": f"sdg_{sdg}",
                "cosine_similarity": round(float(np.dot(global_unit, unit)), 6),
            }
        )
    for left_pos, left_sdg in enumerate(available_sdgs):
        left_vec = within_vectors[left_sdg - 1]
        for right_sdg in available_sdgs[left_pos + 1 :]:
            right_vec = within_vectors[right_sdg - 1]
            rows.append(
                {
                    "comparison_type": "pairwise_within_sdg",
                    "left": f"sdg_{left_sdg}",
                    "right": f"sdg_{right_sdg}",
                    "cosine_similarity": round(float(np.dot(left_vec, right_vec)), 6),
                }
            )
    return rows, cosine_global_vs_avg

def combine_index_pools(index_map: dict[int, np.ndarray], sdgs_one_based: tuple[int, ...]) -> np.ndarray:
    arrays = [index_map[sdg - 1] for sdg in sdgs_one_based if index_map[sdg - 1].size]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(arrays).astype(np.int64)


def sample_from_pool(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n > pool.shape[0]:
        raise ValueError(f"Cannot sample {n} rows from pool of size {pool.shape[0]}")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False).astype(np.int64))


def load_or_build_local_adjusted_reseparability(
    *,
    cache_root: Path,
    classifier_signature: dict[str, Any],
    register_unit: np.ndarray,
    split_fracs: dict[str, float],
    split_sizes: dict[str, int],
    sample_class_balance: dict[str, int],
    c_grid: list[float],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    raw_metrics_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "local_adjusted_classifier",
        "classifier_signature": classifier_signature,
        "c_grid": list(c_grid),
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "local_adjusted_classifier", key)
    if cache_path.exists():
        log.info("Cache hit: adjusted re-separability (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: adjusted re-separability (%s)", key)
    X_train_adj = project_out_direction_and_normalize(X_train, register_unit)
    X_val_adj = project_out_direction_and_normalize(X_val, register_unit)
    X_test_adj = project_out_direction_and_normalize(X_test, register_unit)
    X_train_val_adj = project_out_direction_and_normalize(X_train_val, register_unit)
    bundle = fit_classifier_bundle(
        X_train=X_train_adj,
        y_train=y_train,
        X_val=X_val_adj,
        y_val=y_val,
        X_test=X_test_adj,
        y_test=y_test,
        X_train_val=X_train_val_adj,
        y_train_val=y_train_val,
        c_grid=c_grid,
        seed=seed,
    )
    raw_test = raw_metrics_payload["selected_model_test_metrics"]
    adjusted_test = bundle["test_metrics"]
    metrics_payload = {
        "check": "adjusted_reseparability",
        "seed": seed,
        "sample_class_balance": sample_class_balance,
        "split_fracs": split_fracs,
        "split_sizes": split_sizes,
        "candidate_c_grid": list(c_grid),
        "raw_reference": {
            "selected_model": raw_metrics_payload["selected_model"],
            "test_metrics": raw_test,
        },
        "adjusted_model": {
            "selected_model": bundle["selected_model"],
            "test_metrics": bundle["test_metrics"],
            "test_confusion_matrix": bundle["test_confusion_matrix"],
            "final_train_plus_validation_model": {
                "C": float(bundle["selected_model"]["C"]),
                "coefficient_norm": bundle["coefficient_norm"],
                "intercept": bundle["intercept"],
            },
        },
        "delta_vs_raw_test": {
            metric: round(float(adjusted_test[metric]) - float(raw_test[metric]), 6)
            for metric in ("accuracy", "roc_auc", "f1", "precision", "recall")
        },
    }
    payload_out = {
        "payload": payload,
        "candidate_rows": bundle["candidate_rows"],
        "metrics_payload": metrics_payload,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def load_or_build_local_heldout_sdg_generalization(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    research_indices_by_sdg_signature: dict[str, Any],
    research_indices_by_sdg: dict[int, np.ndarray],
    policy_assignments: np.ndarray,
    policy_emb: np.ndarray,
    shards: list[ResearchShard],
    c_grid: list[float],
    split_fracs: dict[str, float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "local_heldout_sdg",
        "base_signature": base_signature,
        "research_indices_by_sdg_signature": research_indices_by_sdg_signature,
        "folds": [list(fold) for fold in HELDOUT_SDG_FOLDS],
        "c_grid": list(c_grid),
        "split_fracs": split_fracs,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "local_heldout_sdg", key)
    if cache_path.exists():
        log.info("Cache hit: held-out SDG generalization (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: held-out SDG generalization (%s)", key)
    policy_indices_by_sdg = build_policy_indices_by_sdg(policy_assignments)
    all_sdgs = tuple(range(1, N_SDG + 1))
    val_frac_within_trainval = split_fracs["validation"] / (split_fracs["train"] + split_fracs["validation"])

    fold_rows: list[dict[str, Any]] = []
    for fold_idx, heldout_sdgs in enumerate(HELDOUT_SDG_FOLDS, start=1):
        log.info("Fitting held-out fold %d/%d (held SDGs: %s)", fold_idx, len(HELDOUT_SDG_FOLDS), heldout_sdgs)
        train_sdgs = tuple(sdg for sdg in all_sdgs if sdg not in heldout_sdgs)
        research_train_pool = combine_index_pools(research_indices_by_sdg, train_sdgs)
        research_test_pool = combine_index_pools(research_indices_by_sdg, heldout_sdgs)
        policy_train_pool = np.concatenate([policy_indices_by_sdg[sdg - 1] for sdg in train_sdgs]).astype(np.int64)
        policy_test_pool = np.concatenate([policy_indices_by_sdg[sdg - 1] for sdg in heldout_sdgs]).astype(np.int64)

        train_per_class = min(int(research_train_pool.shape[0]), int(policy_train_pool.shape[0]))
        test_per_class = min(int(research_test_pool.shape[0]), int(policy_test_pool.shape[0]))
        if train_per_class <= 0 or test_per_class <= 0:
            raise RuntimeError(f"Held-out fold {fold_idx} has empty train or test pool.")

        research_train_indices = sample_from_pool(research_train_pool, train_per_class, seed + fold_idx * 100 + 1)
        policy_train_indices = sample_from_pool(policy_train_pool, train_per_class, seed + fold_idx * 100 + 2)
        research_test_indices = sample_from_pool(research_test_pool, test_per_class, seed + fold_idx * 100 + 3)
        policy_test_indices = sample_from_pool(policy_test_pool, test_per_class, seed + fold_idx * 100 + 4)

        X_train_val_fold = np.vstack(
            [
                collect_research_embeddings(shards, research_train_indices),
                policy_emb[policy_train_indices],
            ]
        ).astype(np.float32)
        y_train_val_fold = np.concatenate(
            [
                np.zeros(train_per_class, dtype=np.int64),
                np.ones(train_per_class, dtype=np.int64),
            ]
        )
        X_test_fold = np.vstack(
            [
                collect_research_embeddings(shards, research_test_indices),
                policy_emb[policy_test_indices],
            ]
        ).astype(np.float32)
        y_test_fold = np.concatenate(
            [
                np.zeros(test_per_class, dtype=np.int64),
                np.ones(test_per_class, dtype=np.int64),
            ]
        )

        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_val_fold,
            y_train_val_fold,
            test_size=val_frac_within_trainval,
            stratify=y_train_val_fold,
            random_state=seed + fold_idx,
        )
        bundle = fit_classifier_bundle(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            X_test=X_test_fold,
            y_test=y_test_fold,
            X_train_val=X_train_val_fold,
            y_train_val=y_train_val_fold,
            c_grid=c_grid,
            seed=seed + fold_idx,
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "heldout_sdgs": list(heldout_sdgs),
                "train_sdgs": list(train_sdgs),
                "train_per_class": train_per_class,
                "test_per_class": test_per_class,
                "selected_C": float(bundle["selected_model"]["C"]),
                "val_roc_auc": bundle["selected_model"]["val_roc_auc"],
                "val_f1": bundle["selected_model"]["val_f1"],
                "test_accuracy": bundle["test_metrics"]["accuracy"],
                "test_roc_auc": bundle["test_metrics"]["roc_auc"],
                "test_f1": bundle["test_metrics"]["f1"],
                "test_precision": bundle["test_metrics"]["precision"],
                "test_recall": bundle["test_metrics"]["recall"],
                "test_confusion_matrix": bundle["test_confusion_matrix"]["matrix"],
            }
        )

    mean_metrics = {
        metric: round(float(np.mean([row[metric] for row in fold_rows])), 6)
        for metric in ("test_accuracy", "test_roc_auc", "test_f1", "test_precision", "test_recall")
    }
    payload_out = {
        "payload": payload,
        "mean_test_metrics": mean_metrics,
        "fold_rows": fold_rows,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def learn_additional_directions(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    c_grid: list[float],
    seed: int,
    initial_direction: np.ndarray,
    max_k: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    if max_k < 1:
        raise ValueError("max_k must be at least 1")

    directions: list[np.ndarray] = [initial_direction.astype(np.float32)]
    direction_rows: list[dict[str, Any]] = [
        {
            "step": 1,
            "source": "canonical_register_direction",
        }
    ]
    residuals_by_k: dict[int, dict[str, np.ndarray]] = {
        1: {
            "train": project_out_direction_and_normalize(X_train, initial_direction),
            "val": project_out_direction_and_normalize(X_val, initial_direction),
            "train_val": project_out_direction_and_normalize(X_train_val, initial_direction),
        }
    }
    current_train = residuals_by_k[1]["train"]
    current_val = residuals_by_k[1]["val"]
    current_train_val = residuals_by_k[1]["train_val"]

    for step in range(2, max_k + 1):
        log.info("Fitting multi-direction step %d/%d", step, max_k)
        bundle = fit_classifier_bundle(
            X_train=current_train,
            y_train=y_train,
            X_val=current_val,
            y_val=y_val,
            X_test=current_val,
            y_test=y_val,
            X_train_val=current_train_val,
            y_train_val=y_train_val,
            c_grid=c_grid,
            seed=seed + step,
        )
        direction = bundle["unit"]
        directions.append(direction)
        direction_rows.append(
            {
                "step": step,
                "source": "residual_classifier",
                "selected_C": float(bundle["selected_model"]["C"]),
                "val_accuracy": bundle["selected_model"]["val_accuracy"],
                "val_roc_auc": bundle["selected_model"]["val_roc_auc"],
                "val_f1": bundle["selected_model"]["val_f1"],
                "coefficient_norm": bundle["coefficient_norm"],
                "intercept": bundle["intercept"],
            }
        )
        current_train = project_out_direction_and_normalize(current_train, direction)
        current_val = project_out_direction_and_normalize(current_val, direction)
        current_train_val = project_out_direction_and_normalize(current_train_val, direction)
        residuals_by_k[step] = {
            "train": current_train,
            "val": current_val,
            "train_val": current_train_val,
        }

    return directions, direction_rows, residuals_by_k


def build_multi_direction_research_outputs(
    *,
    shards: list[ResearchShard],
    directions: list[np.ndarray],
    ks: list[int],
    research_counts: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    dim = directions[0].shape[0]
    sums_by_k = {
        k: np.zeros((N_SDG, dim), dtype=np.float64)
        for k in ks
    }
    need_ks = set(ks)
    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        score = np.load(shard.score_path).astype(np.float32)
        assignments = score.argmax(axis=1)
        current = emb
        for step, direction in enumerate(directions, start=1):
            current = project_out_direction_and_normalize(current, direction)
            if step not in need_ks:
                continue
            for sdg_idx in np.unique(assignments):
                mask = assignments == sdg_idx
                sums_by_k[step][sdg_idx] += current[mask].sum(axis=0)

    outputs: dict[int, dict[str, np.ndarray]] = {}
    for k in ks:
        centroids = np.zeros((N_SDG, dim), dtype=np.float32)
        cohesions = np.zeros(N_SDG, dtype=np.float32)
        sums = sums_by_k[k]
        for sdg_idx in range(N_SDG):
            n = int(research_counts[sdg_idx])
            if n == 0:
                continue
            raw = sums[sdg_idx] / float(n)
            norm = float(np.linalg.norm(raw))
            if norm < 1e-8:
                raise RuntimeError(f"Multi-direction centroid has near-zero norm for SDG {sdg_idx + 1}, k={k}.")
            centroids[sdg_idx] = (raw / norm).astype(np.float32)
            cohesions[sdg_idx] = float(norm)
        outputs[k] = {"centroids": centroids, "cohesions": cohesions}
    return outputs


def load_or_build_local_multi_direction(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    classifier_signature: dict[str, Any],
    c_grid: list[float],
    seed: int,
    ks: list[int],
    raw_rows: list[dict[str, Any]],
    register_unit: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_ids: list[dict[str, Any]],
    shards: list[ResearchShard],
    research_counts: np.ndarray,
    segment_cap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "local_multi_direction",
        "base_signature": base_signature,
        "classifier_signature": classifier_signature,
        "c_grid": list(c_grid),
        "seed": seed,
        "ks": list(ks),
        "segment_cap": segment_cap,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "local_multi_direction", key)
    if cache_path.exists():
        log.info("Cache hit: multi-direction subtraction (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: multi-direction subtraction (%s)", key)
    max_k = max(ks)
    directions, direction_rows, residuals_by_k = learn_additional_directions(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        c_grid=c_grid,
        seed=seed,
        initial_direction=register_unit,
        max_k=max_k,
    )
    residual_test_by_k: dict[int, np.ndarray] = {}
    current_test = project_out_direction_and_normalize(X_test, directions[0])
    residual_test_by_k[1] = current_test
    for step, direction in enumerate(directions[1:], start=2):
        current_test = project_out_direction_and_normalize(current_test, direction)
        residual_test_by_k[step] = current_test

    research_by_k = build_multi_direction_research_outputs(
        shards=shards,
        directions=directions,
        ks=ks,
        research_counts=research_counts,
    )
    raw_by_sdg = {int(row["sdg"]): row for row in raw_rows}

    curve_rows: list[dict[str, Any]] = []
    per_sdg_rows: list[dict[str, Any]] = []
    current_policy = project_out_direction_and_normalize(policy_emb, directions[0])
    policy_by_k = {1: current_policy}
    for step, direction in enumerate(directions[1:], start=2):
        current_policy = project_out_direction_and_normalize(current_policy, direction)
        policy_by_k[step] = current_policy

    for k in ks:
        log.info("Fitting multi-direction k=%d (ks=%s)", k, ks)
        bundle = fit_classifier_bundle(
            X_train=residuals_by_k[k]["train"],
            y_train=y_train,
            X_val=residuals_by_k[k]["val"],
            y_val=y_val,
            X_test=residual_test_by_k[k],
            y_test=y_test,
            X_train_val=residuals_by_k[k]["train_val"],
            y_train_val=y_train_val,
            c_grid=c_grid,
            seed=seed + 500 + k,
        )
        adjusted_results = compute_sdg_semantic_gaps(
            research_centroids=research_by_k[k]["centroids"],
            research_counts=research_counts,
            research_cohesions=research_by_k[k]["cohesions"],
            policy_emb=policy_by_k[k],
            policy_assignments=policy_assignments,
            policy_ids=policy_ids,
            segment_cap=segment_cap,
            rng=np.random.default_rng(seed),
        )
        top_adjusted = sorted(
            [row for row in adjusted_results if row["semantic_gap"] is not None],
            key=lambda row: float(row["semantic_gap"]),
            reverse=True,
        )[:5]
        gap_vals = [float(row["semantic_gap"]) for row in adjusted_results if row["semantic_gap"] is not None]
        raw_gap_vals = [float(raw_by_sdg[row["sdg"]]["semantic_gap"]) for row in adjusted_results if row["semantic_gap"] is not None]
        mean_adjusted_gap = float(np.mean(gap_vals))
        mean_delta = float(np.mean([gap - raw for gap, raw in zip(gap_vals, raw_gap_vals)]))
        curve_rows.append(
            {
                "k": k,
                "test_accuracy": bundle["test_metrics"]["accuracy"],
                "test_roc_auc": bundle["test_metrics"]["roc_auc"],
                "test_f1": bundle["test_metrics"]["f1"],
                "test_precision": bundle["test_metrics"]["precision"],
                "test_recall": bundle["test_metrics"]["recall"],
                "mean_adjusted_gap": round(mean_adjusted_gap, 6),
                "mean_delta_vs_raw": round(mean_delta, 6),
                "top_adjusted_sdgs": [int(row["sdg"]) for row in top_adjusted],
            }
        )
        for row in adjusted_results:
            raw_row = raw_by_sdg[row["sdg"]]
            raw_gap = raw_row["semantic_gap"]
            adj_gap = row["semantic_gap"]
            per_sdg_rows.append(
                {
                    "k": k,
                    "sdg": int(row["sdg"]),
                    "raw_gap": raw_gap,
                    "adjusted_gap": adj_gap,
                    "delta_gap": (
                        None if raw_gap is None or adj_gap is None else round(float(adj_gap) - float(raw_gap), 6)
                    ),
                    "n_papers": int(row["n_papers"]),
                    "n_policy_docs_capped": int(row["n_policy_docs_capped"]),
                    "unreliable": bool(row["unreliable"]),
                }
            )

    payload_out = {
        "payload": payload,
        "direction_rows": direction_rows,
        "curve_rows": curve_rows,
        "per_sdg_rows": per_sdg_rows,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def build_capped_policy_indices_by_sdg(
    *,
    policy_assignments: np.ndarray,
    policy_ids: list[dict[str, Any]],
    segment_cap: int,
    seed: int,
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    out: dict[int, list[int]] = {}
    for sdg_idx in range(N_SDG):
        idxs = np.flatnonzero(policy_assignments == sdg_idx).astype(np.int64).tolist()
        out[sdg_idx] = cap_policy_indices_per_doc(idxs, policy_ids, segment_cap, rng)
    return out


def topk_raw_and_adjusted_matches(
    *,
    query_raw: np.ndarray,
    query_adj: np.ndarray,
    candidate_raw: np.ndarray,
    candidate_adj: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    n_query = query_raw.shape[0]
    mean_raw = np.empty(n_query, dtype=np.float32)
    mean_adj = np.empty(n_query, dtype=np.float32)
    best_idx = np.empty(n_query, dtype=np.int64)
    for start in range(0, n_query, TOPIC_MATCH_BATCH_SIZE):
        stop = min(start + TOPIC_MATCH_BATCH_SIZE, n_query)
        sims = query_raw[start:stop] @ candidate_raw.T
        local_top_k = min(top_k, candidate_raw.shape[0])
        top_idx = np.argpartition(-sims, kth=local_top_k - 1, axis=1)[:, :local_top_k]
        top_raw = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_raw, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_raw = np.take_along_axis(top_raw, order, axis=1)
        adj_scores = np.sum(
            query_adj[start:stop, None, :] * candidate_adj[top_idx],
            axis=2,
        )
        mean_raw[start:stop] = top_raw.mean(axis=1).astype(np.float32)
        mean_adj[start:stop] = adj_scores.mean(axis=1).astype(np.float32)
        best_idx[start:stop] = top_idx[:, 0].astype(np.int64)
    return mean_raw, mean_adj, best_idx


def load_or_build_local_topic_match(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    research_indices_by_sdg_signature: dict[str, Any],
    research_indices_by_sdg: dict[int, np.ndarray],
    policy_emb: np.ndarray,
    adjusted_policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_ids: list[dict[str, Any]],
    policy_text_ids: list[dict[str, Any]],
    shards: list[ResearchShard],
    register_unit: np.ndarray,
    segment_cap: int,
    topic_match_research_per_sdg: int,
    topic_match_top_k: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "local_topic_match",
        "base_signature": base_signature,
        "research_indices_by_sdg_signature": research_indices_by_sdg_signature,
        "segment_cap": segment_cap,
        "topic_match_research_per_sdg": topic_match_research_per_sdg,
        "topic_match_top_k": topic_match_top_k,
        "seed": seed,
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "local_topic_match", key)
    if cache_path.exists():
        log.info("Cache hit: topic-matched nearest-neighbor check (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: topic-matched nearest-neighbor check (%s)", key)
    capped_policy_by_sdg = build_capped_policy_indices_by_sdg(
        policy_assignments=policy_assignments,
        policy_ids=policy_ids,
        segment_cap=segment_cap,
        seed=seed,
    )
    per_sdg_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    deferred_examples: list[dict[str, Any]] = []
    all_needed_research_indices: set[int] = set()

    for sdg_idx in range(N_SDG):
        policy_idxs = capped_policy_by_sdg[sdg_idx]
        research_pool = research_indices_by_sdg[sdg_idx]
        if not policy_idxs or research_pool.size == 0:
            per_sdg_rows.append(
                {
                    "sdg": sdg_idx + 1,
                    "pair_count": 0,
                    "research_sample_n": 0,
                    "raw_mean_matched_cosine": None,
                    "adjusted_mean_matched_cosine": None,
                    "delta_mean_cosine": None,
                    "raw_mean_matched_gap": None,
                    "adjusted_mean_matched_gap": None,
                }
            )
            continue

        sample_n = min(topic_match_research_per_sdg, int(research_pool.shape[0]))
        sampled_indices = sample_from_pool(research_pool, sample_n, seed + 2000 + sdg_idx)
        research_emb = collect_research_embeddings(shards, sampled_indices)
        adjusted_research_emb = project_out_direction_and_normalize(research_emb, register_unit)
        query_raw = policy_emb[policy_idxs]
        query_adj = adjusted_policy_emb[policy_idxs]
        mean_raw, mean_adj, best_idx = topk_raw_and_adjusted_matches(
            query_raw=query_raw,
            query_adj=query_adj,
            candidate_raw=research_emb,
            candidate_adj=adjusted_research_emb,
            top_k=topic_match_top_k,
        )
        raw_mean = float(mean_raw.mean())
        adj_mean = float(mean_adj.mean())
        per_sdg_rows.append(
            {
                "sdg": sdg_idx + 1,
                "pair_count": len(policy_idxs),
                "research_sample_n": sample_n,
                "raw_mean_matched_cosine": round(raw_mean, 6),
                "adjusted_mean_matched_cosine": round(adj_mean, 6),
                "delta_mean_cosine": round(adj_mean - raw_mean, 6),
                "raw_mean_matched_gap": round(1.0 - raw_mean, 6),
                "adjusted_mean_matched_gap": round(1.0 - adj_mean, 6),
            }
        )

        top_example_order = np.argsort(-mean_raw)[:TOPIC_MATCH_EXAMPLES_PER_SDG]
        for rank, local_idx in enumerate(top_example_order, start=1):
            policy_row_idx = policy_idxs[int(local_idx)]
            research_global_idx = int(sampled_indices[int(best_idx[int(local_idx)])])
            all_needed_research_indices.add(research_global_idx)
            deferred_examples.append(
                {
                    "sdg": sdg_idx + 1,
                    "rank_within_sdg": rank,
                    "policy_item_id": policy_ids[policy_row_idx]["id"],
                    "policy_source_doc": policy_ids[policy_row_idx]["source_doc"],
                    "research_global_idx": research_global_idx,
                    "raw_cosine": round(float(mean_raw[int(local_idx)]), 6),
                    "adjusted_cosine": round(float(mean_adj[int(local_idx)]), 6),
                    "delta_cosine": round(float(mean_adj[int(local_idx)] - mean_raw[int(local_idx)]), 6),
                    "policy_text_snippet": strip_text(policy_text_ids[policy_row_idx]["text"]),
                }
            )

    research_text_rows = (
        load_research_texts_for_indices(
            shards,
            np.array(sorted(all_needed_research_indices), dtype=np.int64),
        )
        if all_needed_research_indices
        else {}
    )
    for row in deferred_examples:
        research_meta = research_text_rows[row["research_global_idx"]]
        example_rows.append(
            {
                "sdg": row["sdg"],
                "rank_within_sdg": row["rank_within_sdg"],
                "policy_item_id": row["policy_item_id"],
                "policy_source_doc": row["policy_source_doc"],
                "research_item_id": research_meta["openalex_id"],
                "research_publication_year": research_meta.get("publication_year"),
                "raw_cosine": row["raw_cosine"],
                "adjusted_cosine": row["adjusted_cosine"],
                "delta_cosine": row["delta_cosine"],
                "policy_text_snippet": row["policy_text_snippet"],
                "research_text_snippet": strip_text(research_meta["text"]),
            }
        )

    valid_rows = [row for row in per_sdg_rows if row["pair_count"] > 0]
    overall = {
        "raw_mean_matched_cosine": round(float(np.mean([row["raw_mean_matched_cosine"] for row in valid_rows])), 6),
        "adjusted_mean_matched_cosine": round(float(np.mean([row["adjusted_mean_matched_cosine"] for row in valid_rows])), 6),
        "raw_mean_matched_gap": round(float(np.mean([row["raw_mean_matched_gap"] for row in valid_rows])), 6),
        "adjusted_mean_matched_gap": round(float(np.mean([row["adjusted_mean_matched_gap"] for row in valid_rows])), 6),
    }
    payload_out = {
        "payload": payload,
        "overall": overall,
        "per_sdg_rows": per_sdg_rows,
        "example_rows": example_rows,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def build_local_confidence_summary(
    *,
    adjusted_reseparability: dict[str, Any],
    heldout_generalization: dict[str, Any],
    multi_direction: dict[str, Any],
    topic_match: dict[str, Any],
) -> dict[str, Any]:
    adjusted_test = adjusted_reseparability["metrics_payload"]["adjusted_model"]["test_metrics"]
    raw_test = adjusted_reseparability["metrics_payload"]["raw_reference"]["test_metrics"]
    heldout_mean = heldout_generalization["mean_test_metrics"]
    curve_rows = multi_direction["curve_rows"]
    k1_row = min(curve_rows, key=lambda row: int(row["k"]))
    later_rows = [row for row in curve_rows if int(row["k"]) > int(k1_row["k"])]
    max_mean_gap_change_after_k1 = 0.0
    if later_rows:
        max_mean_gap_change_after_k1 = max(
            abs(float(row["mean_adjusted_gap"]) - float(k1_row["mean_adjusted_gap"]))
            for row in later_rows
        )
    topic_overall = topic_match["overall"]
    return {
        "adjusted_reseparability": {
            "raw_test_roc_auc": raw_test["roc_auc"],
            "adjusted_test_roc_auc": adjusted_test["roc_auc"],
            "relative_roc_auc_drop": round(
                (float(adjusted_test["roc_auc"]) - float(raw_test["roc_auc"])) / float(raw_test["roc_auc"]),
                6,
            ),
            "adjusted_separability_still_high": bool(float(adjusted_test["roc_auc"]) >= 0.90),
        },
        "heldout_sdg_generalization": {
            "mean_test_roc_auc": heldout_mean["test_roc_auc"],
            "mean_test_f1": heldout_mean["test_f1"],
            "generalization_still_high": bool(float(heldout_mean["test_roc_auc"]) >= 0.90),
        },
        "multi_direction": {
            "k1_mean_adjusted_gap": k1_row["mean_adjusted_gap"],
            "max_abs_mean_gap_change_after_k1": round(float(max_mean_gap_change_after_k1), 6),
            "material_change_after_k1": bool(max_mean_gap_change_after_k1 >= 0.02),
        },
        "topic_matched": {
            "overall_adjusted_mean_matched_gap": topic_overall["adjusted_mean_matched_gap"],
            "overall_raw_mean_matched_gap": topic_overall["raw_mean_matched_gap"],
            "topic_matched_divergence_still_large": bool(float(topic_overall["adjusted_mean_matched_gap"]) >= 0.20),
        },
        "threshold_notes": {
            "high_separability_or_generalization": "ROC-AUC >= 0.90",
            "material_multi_direction_change": "absolute mean adjusted gap shift >= 0.02 relative to k=1",
            "large_topic_matched_divergence": "mean adjusted matched gap >= 0.20",
        },
    }


def format_sdg_label_list(sdgs: list[int]) -> str:
    if not sdgs:
        return "none"
    parts = [f"SDG {sdg}" for sdg in sdgs]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def write_register_confidence_latex_outputs(
    tables_dir: Path,
    summary: dict[str, Any],
    adjusted_reseparability: dict[str, Any],
    heldout_generalization: dict[str, Any],
    multi_direction: dict[str, Any],
    topic_match: dict[str, Any],
) -> None:
    raw_test = adjusted_reseparability["metrics_payload"]["raw_reference"]["test_metrics"]
    adjusted_test = adjusted_reseparability["metrics_payload"]["adjusted_model"]["test_metrics"]
    held_mean = heldout_generalization["mean_test_metrics"]
    held_rows = heldout_generalization["fold_rows"]
    min_held_roc_auc = min(float(row["test_roc_auc"]) for row in held_rows)
    max_held_roc_auc = max(float(row["test_roc_auc"]) for row in held_rows)
    curve_rows = sorted(multi_direction["curve_rows"], key=lambda row: int(row["k"]))
    k1_row = curve_rows[0]
    kmax_row = curve_rows[-1]
    top_stable = sorted(set(int(v) for v in k1_row["top_adjusted_sdgs"]) & set(int(v) for v in kmax_row["top_adjusted_sdgs"]))
    topic_overall = topic_match["overall"]

    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/3_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\RegisterConfidenceAdjustedResepRawRocAuc}}{{{float(raw_test['roc_auc']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceAdjustedResepAdjustedRocAuc}}{{{float(adjusted_test['roc_auc']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceAdjustedResepDeltaRocAuc}}{{{float(summary['adjusted_reseparability']['relative_roc_auc_drop']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceAdjustedResepAdjustedAccuracy}}{{{float(adjusted_test['accuracy']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceHeldoutMeanRocAuc}}{{{float(held_mean['test_roc_auc']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceHeldoutMeanFOne}}{{{float(held_mean['test_f1']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceHeldoutMinRocAuc}}{{{min_held_roc_auc:.3f}}}",
        rf"\newcommand{{\RegisterConfidenceHeldoutMaxRocAuc}}{{{max_held_roc_auc:.3f}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKOne}}{{{int(k1_row['k'])}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKMax}}{{{int(kmax_row['k'])}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKOneMeanGap}}{{{float(k1_row['mean_adjusted_gap']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKMaxMeanGap}}{{{float(kmax_row['mean_adjusted_gap']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKOneToMaxDelta}}{{{float(kmax_row['mean_adjusted_gap']) - float(k1_row['mean_adjusted_gap']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceMultiKMaxResidualRocAuc}}{{{float(kmax_row['test_roc_auc']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceStableTopGapSdgs}}{{{format_sdg_label_list(top_stable)}}}",
        rf"\newcommand{{\RegisterConfidenceTopicRawMatchedGap}}{{{float(topic_overall['raw_mean_matched_gap']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceTopicAdjustedMatchedGap}}{{{float(topic_overall['adjusted_mean_matched_gap']):.3f}}}",
        rf"\newcommand{{\RegisterConfidenceTopicDeltaMatchedGap}}{{{float(topic_overall['adjusted_mean_matched_gap']) - float(topic_overall['raw_mean_matched_gap']):.3f}}}",
    ]
    (tables_dir / "num_register_confidence_checks.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    tab_lines = [
        r"\begin{tabular}{p{0.24\textwidth}p{0.26\textwidth}p{0.40\textwidth}}",
        r"\toprule",
        r"Check & Headline estimate & Interpretation \\",
        r"\midrule",
        r"Adjusted re-separability & Raw ROC-AUC \RegisterConfidenceAdjustedResepRawRocAuc; adjusted \RegisterConfidenceAdjustedResepAdjustedRocAuc ($\Delta$ \RegisterConfidenceAdjustedResepDeltaRocAuc) & One removed direction leaves cross-corpus separability very high. \\",
        r"Held-out SDG generalization & Mean held-out ROC-AUC \RegisterConfidenceHeldoutMeanRocAuc; mean F1 \RegisterConfidenceHeldoutMeanFOne{} & The learned register axis transfers across unseen SDG blocks, so it is not merely within-SDG topic leakage. \\",
        r"Multi-direction subtraction & Mean adjusted gap \RegisterConfidenceMultiKOneMeanGap{} at $k=\RegisterConfidenceMultiKOne$ and \RegisterConfidenceMultiKMaxMeanGap{} at $k=\RegisterConfidenceMultiKMax$ & Additional discriminative directions continue to compress gap magnitude, although the largest-gap SDGs remain concentrated in \RegisterConfidenceStableTopGapSdgs. \\",
        r"Topic-matched nearest neighbor & Mean matched gap \RegisterConfidenceTopicRawMatchedGap{} raw and \RegisterConfidenceTopicAdjustedMatchedGap{} adjusted ($\Delta$ \RegisterConfidenceTopicDeltaMatchedGap) & Within-SDG nearest cross-corpus matches remain far apart after subtraction and become slightly less similar on average. \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (tables_dir / "tab_register_confidence_checks.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")


def plot_register_confidence_curve(figures_dir: Path, multi_direction: dict[str, Any]) -> None:
    rows = sorted(multi_direction["curve_rows"], key=lambda row: int(row["k"]))
    ks = [int(row["k"]) for row in rows]
    mean_gaps = [float(row["mean_adjusted_gap"]) for row in rows]
    roc_aucs = [float(row["test_roc_auc"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)

    ax_top.plot(ks, mean_gaps, marker="o", color="#2166AC", linewidth=1.8)
    ax_top.set_ylabel("Mean adjusted gap")
    ax_top.set_title(
        "Additional register-confidence checks: multi-direction subtraction sensitivity",
        fontsize=8.5,
        loc="left",
    )
    ax_top.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_bottom.plot(ks, roc_aucs, marker="o", color="#B2182B", linewidth=1.8)
    ax_bottom.set_xlabel("Number of removed directions (k)")
    ax_bottom.set_ylabel("Residual test ROC-AUC")
    ax_bottom.set_xticks(ks)
    ax_bottom.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_register_confidence_curve.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_register_confidence_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def build_sdg_name_maps() -> tuple[dict[int, str], dict[int, str]]:
    names = {
        1: "No Poverty",
        2: "Zero Hunger",
        3: "Good Health and Well-Being",
        4: "Quality Education",
        5: "Gender Equality",
        6: "Clean Water and Sanitation",
        7: "Affordable and Clean Energy",
        8: "Decent Work and Economic Growth",
        9: "Industry, Innovation and Infrastructure",
        10: "Reduced Inequalities",
        11: "Sustainable Cities and Communities",
        12: "Responsible Consumption and Production",
        13: "Climate Action",
        14: "Life Below Water",
        15: "Life on Land",
        16: "Peace, Justice and Strong Institutions",
        17: "Partnerships for the Goals",
    }
    num_words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
        11: "Eleven",
        12: "Twelve",
        13: "Thirteen",
        14: "Fourteen",
        15: "Fifteen",
        16: "Sixteen",
        17: "Seventeen",
    }
    return names, num_words


def mean_gap(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [float(row[field]) for row in rows if row.get(field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def write_gap_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    adjusted_similarity_field: str,
    adjusted_gap_field: str,
) -> None:
    fieldnames = [
        "sdg",
        "n_papers",
        "n_policy_segments",
        "n_policy_segments_capped",
        "n_policy_docs",
        "n_policy_docs_capped",
        "segment_cap",
        "raw_similarity",
        "raw_gap",
        "raw_research_cohesion",
        "raw_policy_cohesion",
        adjusted_similarity_field,
        adjusted_gap_field,
        f"{adjusted_gap_field}_research_cohesion",
        f"{adjusted_gap_field}_policy_cohesion",
        "delta_gap",
        "classifier_available",
        "skip_reason",
        "unreliable",
        "unreliable_reason",
    ]
    write_rows_csv(path, fieldnames, rows)


def plot_sdg_balanced_gap_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if row["sdg_balanced_adjusted_gap"] is not None]
    if not valid_rows:
        return
    names, _ = build_sdg_name_maps()
    valid_rows = sorted(valid_rows, key=lambda row: float(row["raw_gap"]), reverse=True)
    y = np.arange(len(valid_rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.barh(
        y - height / 2,
        [float(row["raw_gap"]) for row in valid_rows],
        height=height,
        color="#BDBDBD",
        label="Raw gap",
    )
    ax.barh(
        y + height / 2,
        [float(row["sdg_balanced_adjusted_gap"]) for row in valid_rows],
        height=height,
        color="#1B9E77",
        label="SDG-balanced adjusted gap",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"SDG {row['sdg']} {names[row['sdg']]}" for row in valid_rows], fontsize=7.5)
    ax.set_xlabel("Gap = 1 - cosine similarity")
    ax.set_title(
        "SDG-balanced global sensitivity check: raw vs adjusted semantic gap",
        fontsize=8.5,
        loc="left",
    )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "sdg_balanced_gap_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "sdg_balanced_gap_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_sdg_register_robustness_comparison(
    *,
    output_dir: Path,
    manuscript_figures_dir: Path,
    sdg_balanced_rows: list[dict[str, Any]] | None,
    within_sdg_rows: list[dict[str, Any]] | None,
) -> None:
    if not sdg_balanced_rows and not within_sdg_rows:
        return
    names, _ = build_sdg_name_maps()
    row_map: dict[int, dict[str, Any]] = {}
    for sdg in range(1, N_SDG + 1):
        raw_gap = None
        global_gap = None
        within_gap = None
        if sdg_balanced_rows is not None:
            row = next(row for row in sdg_balanced_rows if int(row["sdg"]) == sdg)
            raw_gap = row["raw_gap"]
            global_gap = row["sdg_balanced_adjusted_gap"]
        if within_sdg_rows is not None:
            row = next(row for row in within_sdg_rows if int(row["sdg"]) == sdg)
            raw_gap = row["raw_gap"] if raw_gap is None else raw_gap
            within_gap = row["within_sdg_adjusted_gap"]
        row_map[sdg] = {
            "raw_gap": raw_gap,
            "global_gap": global_gap,
            "within_gap": within_gap,
        }
    ordered_sdgs = sorted(
        row_map,
        key=lambda sdg: float(row_map[sdg]["raw_gap"]) if row_map[sdg]["raw_gap"] is not None else -1.0,
        reverse=True,
    )
    y = np.arange(len(ordered_sdgs))
    height = 0.24
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.barh(
        y - height,
        [float(row_map[sdg]["raw_gap"]) for sdg in ordered_sdgs],
        height=height,
        color="#BDBDBD",
        label="Raw gap",
    )
    ax.barh(
        y,
        [np.nan if row_map[sdg]["global_gap"] is None else float(row_map[sdg]["global_gap"]) for sdg in ordered_sdgs],
        height=height,
        color="#1B9E77",
        label="SDG-balanced adjusted",
    )
    ax.barh(
        y + height,
        [np.nan if row_map[sdg]["within_gap"] is None else float(row_map[sdg]["within_gap"]) for sdg in ordered_sdgs],
        height=height,
        color="#2166AC",
        label="Within-SDG adjusted",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"SDG {sdg} {names[sdg]}" for sdg in ordered_sdgs], fontsize=7.4)
    ax.set_xlabel("Gap = 1 - cosine similarity")
    ax.set_title(
        "SDG-aware register sensitivity and stress tests",
        fontsize=8.5,
        loc="left",
    )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_dir_pdf = output_dir / "raw_vs_global_vs_within_sdg_gap_comparison.pdf"
    output_dir_png = output_dir / "raw_vs_global_vs_within_sdg_gap_comparison.png"
    fig.savefig(output_dir_pdf, bbox_inches="tight")
    fig.savefig(output_dir_png, bbox_inches="tight", dpi=150)
    fig.savefig(manuscript_figures_dir / "fig_sdg_register_robustness_comparison.pdf", bbox_inches="tight")
    fig.savefig(manuscript_figures_dir / "fig_sdg_register_robustness_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_within_sdg_vector_similarity_heatmap(
    output_dir: Path,
    *,
    within_vectors: np.ndarray,
    available_sdgs: list[int],
) -> None:
    if not available_sdgs:
        return
    matrix = np.full((N_SDG, N_SDG), np.nan, dtype=np.float32)
    for left in available_sdgs:
        left_vec = within_vectors[left - 1]
        matrix[left - 1, left - 1] = 1.0
        for right in available_sdgs:
            if right <= left:
                continue
            cos = float(np.dot(left_vec, within_vectors[right - 1]))
            matrix[left - 1, right - 1] = cos
            matrix[right - 1, left - 1] = cos
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(N_SDG))
    ax.set_yticks(np.arange(N_SDG))
    ax.set_xticklabels([str(sdg) for sdg in range(1, N_SDG + 1)], fontsize=7)
    ax.set_yticklabels([str(sdg) for sdg in range(1, N_SDG + 1)], fontsize=7)
    ax.set_xlabel("SDG")
    ax.set_ylabel("SDG")
    ax.set_title("Within-SDG register-direction cosine similarity", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "within_sdg_vector_cosine_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "within_sdg_vector_cosine_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_sdg_register_adjustment_note(
    path: Path,
    *,
    method: str,
    seed: int,
    classifier_type: str,
    samples_per_cell: int | None,
    min_samples_per_class: int,
    test_size: float,
) -> None:
    note = [
        "# Register-adjustment robustness suite",
        "",
        "This folder contains appendix-style robustness and diagnostic checks.",
        "The raw within-SDG semantic gap remains the main estimand.",
        "",
        "1. `sdg_balanced` trains one global research-vs-policy classifier on an SDG-balanced sample.",
        "2. `within_sdg` trains separate research-vs-policy classifiers within each SDG.",
        "3. `regression` treats each embedding coordinate as an outcome and estimates register, SDG, and register x SDG effects by exact OLS cell contrasts.",
        "",
        "How to run:",
        f"- `python 1_code/3_main_analysis/3_appendix/3_register_adjustment.py --method {method}`",
        "",
        "Active configuration:",
        f"- `random_seed = {seed}`",
        f"- `classifier_type = {classifier_type}`",
        f"- `samples_per_cell = {samples_per_cell}`",
        f"- `min_samples_per_class = {min_samples_per_class}`",
        f"- `test_size = {test_size}`",
        "",
        "How to interpret:",
        "- Smaller adjusted gaps do not automatically mean a better estimate of the dissertation's target quantity.",
        "- The one-direction global subtraction and the SDG-balanced global subtraction are sensitivity checks for broad corpus-level register effects.",
        "- The within-SDG classifier and within-SDG regression procedures are stress tests: they can over-subtract by learning the very within-goal research-policy contrast the raw gap is meant to measure.",
        "- The regression method estimates register-associated embedding variation after controlling for SDG and compares that direction directly with the classifier-derived direction.",
        "- The register_vector_cosine_similarity.csv file shows whether the within-SDG directions are broadly stable or highly heterogeneous.",
    ]
    path.write_text("\n".join(note) + "\n", encoding="utf-8")


def write_sdg_register_latex_outputs(
    tables_dir: Path,
    *,
    sdg_balanced_metrics: dict[str, Any] | None,
    sdg_balanced_rows: list[dict[str, Any]] | None,
    within_metrics_rows: list[dict[str, Any]] | None,
    within_rows: list[dict[str, Any]] | None,
    cosine_rows: list[dict[str, Any]] | None,
) -> None:
    mean_balanced_gap = mean_gap(sdg_balanced_rows or [], "sdg_balanced_adjusted_gap")
    mean_within_gap = mean_gap(within_rows or [], "within_sdg_adjusted_gap")
    mean_raw_gap = mean_gap((sdg_balanced_rows or within_rows or []), "raw_gap")
    available_within = [row for row in (within_metrics_rows or []) if row["classifier_available"]]
    skipped_within = [int(row["sdg"]) for row in (within_metrics_rows or []) if not row["classifier_available"]]
    balanced_accuracy = ""
    balanced_macro_f1 = ""
    balanced_roc_auc = ""
    if sdg_balanced_metrics is not None:
        balanced_accuracy = f"{float(sdg_balanced_metrics['metrics']['accuracy']):.3f}"
        balanced_macro_f1 = f"{float(sdg_balanced_metrics['metrics']['macro_f1']):.3f}"
        if sdg_balanced_metrics["metrics"]["roc_auc"] is not None:
            balanced_roc_auc = f"{float(sdg_balanced_metrics['metrics']['roc_auc']):.3f}"
    within_mean_accuracy = ""
    within_mean_macro_f1 = ""
    within_mean_roc_auc = ""
    if available_within:
        within_mean_accuracy = f"{float(np.mean([row['accuracy'] for row in available_within])):.3f}"
        within_mean_macro_f1 = f"{float(np.mean([row['macro_f1'] for row in available_within])):.3f}"
        roc_auc_vals = [row["roc_auc"] for row in available_within if row["roc_auc"] is not None]
        if roc_auc_vals:
            within_mean_roc_auc = f"{float(np.mean(roc_auc_vals)):.3f}"
    global_vs_avg = None
    if cosine_rows:
        for row in cosine_rows:
            if row["comparison_type"] == "global_vs_average":
                global_vs_avg = row["cosine_similarity"]
                break

    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/3_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\SdgRegisterBalancedAccuracy}}{{{balanced_accuracy}}}",
        rf"\newcommand{{\SdgRegisterBalancedMacroFOne}}{{{balanced_macro_f1}}}",
        rf"\newcommand{{\SdgRegisterBalancedRocAuc}}{{{balanced_roc_auc}}}",
        rf"\newcommand{{\SdgRegisterBalancedMeanGap}}{{{'' if mean_balanced_gap is None else f'{mean_balanced_gap:.3f}'}}}",
        rf"\newcommand{{\SdgRegisterWithinMeanAccuracy}}{{{within_mean_accuracy}}}",
        rf"\newcommand{{\SdgRegisterWithinMeanMacroFOne}}{{{within_mean_macro_f1}}}",
        rf"\newcommand{{\SdgRegisterWithinMeanRocAuc}}{{{within_mean_roc_auc}}}",
        rf"\newcommand{{\SdgRegisterWithinMeanGap}}{{{'' if mean_within_gap is None else f'{mean_within_gap:.3f}'}}}",
        rf"\newcommand{{\SdgRegisterRawMeanGap}}{{{'' if mean_raw_gap is None else f'{mean_raw_gap:.3f}'}}}",
        rf"\newcommand{{\SdgRegisterWithinFittedCount}}{{{len(available_within)}}}",
        rf"\newcommand{{\SdgRegisterWithinSkippedCount}}{{{len(skipped_within)}}}",
        rf"\newcommand{{\SdgRegisterGlobalVsAverageCosine}}{{{'' if global_vs_avg is None else f'{float(global_vs_avg):.3f}'}}}",
    ]
    (tables_dir / "num_sdg_register_robustness.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    tab_lines = [
        r"\begin{tabular}{p{0.24\textwidth}p{0.23\textwidth}p{0.40\textwidth}}",
        r"\toprule",
        r"Method & Headline estimate & Interpretation \\",
        r"\midrule",
    ]
    if sdg_balanced_metrics is not None:
        tab_lines.append(
            r"SDG-balanced global classifier & Test accuracy \SdgRegisterBalancedAccuracy; macro-F1 \SdgRegisterBalancedMacroFOne; ROC-AUC \SdgRegisterBalancedRocAuc; mean adjusted gap \SdgRegisterBalancedMeanGap & One global research-policy direction is learned after equalising the SDG composition of the training sample. \\"
        )
    if within_metrics_rows is not None:
        tab_lines.append(
            r"Within-SDG classifiers & Mean test accuracy \SdgRegisterWithinMeanAccuracy; mean macro-F1 \SdgRegisterWithinMeanMacroFOne; mean ROC-AUC \SdgRegisterWithinMeanRocAuc; mean adjusted gap \SdgRegisterWithinMeanGap & SDG-specific register directions are fit where sufficient data exist (\SdgRegisterWithinFittedCount{} SDGs fit, \SdgRegisterWithinSkippedCount{} skipped). \\"
        )
    if cosine_rows is not None:
        tab_lines.append(
            r"Direction stability & Global-vs-average cosine \SdgRegisterGlobalVsAverageCosine & Higher cosine similarity means the within-SDG register directions point in roughly the same direction as the SDG-balanced global control. \\"
        )
    tab_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_sdg_register_robustness.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    escaped = sanitize_text(text or "")
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def build_box_stats(values: np.ndarray, label: str) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        raise ValueError(f"Cannot build box stats for empty values: {label}")
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return {
        "label": label,
        "whislo": float(vals.min()),
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "whishi": float(vals.max()),
        "fliers": [],
    }


def summarize_score_group(group_type: str, group_value: str, values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return {
            "group_type": group_type,
            "group_value": group_value,
            "count": 0,
            "mean_score": None,
            "median_score": None,
            "std_score": None,
            "min_score": None,
            "p25_score": None,
            "p75_score": None,
            "max_score": None,
        }
    q25, q75 = np.percentile(vals, [25, 75])
    return {
        "group_type": group_type,
        "group_value": group_value,
        "count": int(vals.size),
        "mean_score": round(float(vals.mean()), 6),
        "median_score": round(float(np.median(vals)), 6),
        "std_score": round(float(vals.std()), 6),
        "min_score": round(float(vals.min()), 6),
        "p25_score": round(float(q25), 6),
        "p75_score": round(float(q75), 6),
        "max_score": round(float(vals.max()), 6),
    }


def sorted_highest_heap(heap: list[tuple[float, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item[2] for item in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)]


def sorted_lowest_heap(heap: list[tuple[float, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item[2] for item in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)]


def materialize_projection_rows(
    *,
    shards: list[ResearchShard],
    rows: list[dict[str, Any]],
    preview_chars: int,
) -> list[dict[str, Any]]:
    needed_research_indices = sorted(
        {int(row["global_idx"]) for row in rows if row["corpus_type"] == "research" and "global_idx" in row}
    )
    research_text_map = (
        load_research_texts_for_indices(shards, np.array(needed_research_indices, dtype=np.int64))
        if needed_research_indices
        else {}
    )
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        title = row.get("title", "")
        text = row.get("text", "")
        if row["corpus_type"] == "research":
            meta = research_text_map[int(row["global_idx"])]
            title = meta.get("title", "")
            text = meta.get("text", "")
        out.append(
            {
                "rank": rank,
                "doc_id": row["doc_id"],
                "corpus_type": row["corpus_type"],
                "assigned_sdg": row["assigned_sdg"],
                "projection_score": round(float(row["projection_score"]), 6),
                "title": strip_text(title, limit=preview_chars),
                "text_preview": strip_text(text, limit=preview_chars),
                "source_doc_or_year": row.get("source_doc_or_year"),
            }
        )
    return out


def normalize_projection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                **row,
                "doc_id": strip_text(str(row.get("doc_id", "")), limit=10_000),
                "title": strip_text(str(row.get("title", "")), limit=GENRE_PROJECTION_PREVIEW_CHARS),
                "text_preview": strip_text(str(row.get("text_preview", "")), limit=GENRE_PROJECTION_PREVIEW_CHARS),
                "source_doc_or_year": strip_text(str(row.get("source_doc_or_year", "")), limit=128),
            }
        )
    return normalized


def load_or_build_register_projection_interpretability_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    sdg_balanced_method_signature: dict[str, Any],
    shards: list[ResearchShard],
    total_research_rows: int,
    policy_emb: np.ndarray,
    policy_assignments: np.ndarray,
    policy_score_ids: list[dict[str, Any]],
    policy_text_ids: list[dict[str, Any]],
    register_unit: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "register_projection_interpretability",
        "base_signature": base_signature,
        "sdg_balanced_method_signature": sdg_balanced_method_signature,
        "top_n": GENRE_PROJECTION_TOP_N,
        "report_n": GENRE_PROJECTION_REPORT_N,
        "preview_chars": GENRE_PROJECTION_PREVIEW_CHARS,
        "total_research_rows": total_research_rows,
        "policy_rows": int(policy_emb.shape[0]),
    }
    key = stable_cache_key(payload)
    cache_npz = cache_npz_path(cache_root, "register_projection_interpretability", key)
    cache_json = cache_json_path(cache_root, "register_projection_interpretability", key)
    if cache_npz.exists() and cache_json.exists():
        log.info("Cache hit: register projection interpretability (%s)", key)
        cached = np.load(cache_npz)
        meta = load_json(cache_json)
        return payload, {
            "research_scores": cached["research_scores"].astype(np.float32),
            "research_sdgs": cached["research_sdgs"].astype(np.uint8),
            "policy_scores": cached["policy_scores"].astype(np.float32),
            "policy_sdgs": cached["policy_sdgs"].astype(np.uint8),
            "top_policy_like_rows": normalize_projection_rows(meta["top_policy_like_rows"]),
            "top_research_like_rows": normalize_projection_rows(meta["top_research_like_rows"]),
            "report_policy_rows": normalize_projection_rows(meta["report_policy_rows"]),
            "report_research_rows": normalize_projection_rows(meta["report_research_rows"]),
        }

    log.info("Cache miss: register projection interpretability (%s)", key)
    policy_scores = (policy_emb @ register_unit).astype(np.float32)
    policy_sdgs = (policy_assignments + 1).astype(np.uint8)
    research_scores = np.empty(total_research_rows, dtype=np.float32)
    research_sdgs = np.empty(total_research_rows, dtype=np.uint8)

    top_policy_like_heap: list[tuple[float, str, dict[str, Any]]] = []
    top_research_like_heap: list[tuple[float, str, dict[str, Any]]] = []
    policy_report_heap: list[tuple[float, str, dict[str, Any]]] = []
    research_report_heap: list[tuple[float, str, dict[str, Any]]] = []

    for idx in range(policy_scores.shape[0]):
        score = float(policy_scores[idx])
        payload_row = {
            "item_id": policy_score_ids[idx]["id"],
            "doc_id": policy_score_ids[idx]["id"],
            "corpus_type": "policy",
            "assigned_sdg": int(policy_sdgs[idx]),
            "projection_score": score,
            "source_doc_or_year": policy_score_ids[idx]["source_doc"],
            "title": "",
            "text": policy_text_ids[idx]["text"],
        }
        push_highest(top_policy_like_heap, GENRE_PROJECTION_TOP_N, score, payload_row)
        push_lowest(top_research_like_heap, GENRE_PROJECTION_TOP_N, score, payload_row)
        push_highest(policy_report_heap, GENRE_PROJECTION_REPORT_N, score, payload_row)

    for shard in shards:
        emb = np.load(shard.emb_path).astype(np.float32)
        scores = (emb @ register_unit).astype(np.float32)
        score_rows = np.load(shard.score_path).astype(np.float32)
        assignments = score_rows.argmax(axis=1).astype(np.uint8) + 1
        ids_rows = load_jsonl(shard.score_ids_path)
        research_scores[shard.start : shard.stop] = scores
        research_sdgs[shard.start : shard.stop] = assignments
        for row_idx in range(shard.rows):
            global_idx = shard.start + row_idx
            payload_row = {
                "item_id": ids_rows[row_idx]["openalex_id"],
                "doc_id": ids_rows[row_idx]["openalex_id"],
                "corpus_type": "research",
                "assigned_sdg": int(assignments[row_idx]),
                "projection_score": float(scores[row_idx]),
                "source_doc_or_year": ids_rows[row_idx].get("publication_year"),
                "global_idx": global_idx,
            }
            push_highest(top_policy_like_heap, GENRE_PROJECTION_TOP_N, float(scores[row_idx]), payload_row)
            push_lowest(top_research_like_heap, GENRE_PROJECTION_TOP_N, float(scores[row_idx]), payload_row)
            push_lowest(research_report_heap, GENRE_PROJECTION_REPORT_N, float(scores[row_idx]), payload_row)

    top_policy_like_rows = materialize_projection_rows(
        shards=shards,
        rows=sorted_highest_heap(top_policy_like_heap),
        preview_chars=GENRE_PROJECTION_PREVIEW_CHARS,
    )
    top_research_like_rows = materialize_projection_rows(
        shards=shards,
        rows=sorted_lowest_heap(top_research_like_heap),
        preview_chars=GENRE_PROJECTION_PREVIEW_CHARS,
    )
    report_policy_rows = materialize_projection_rows(
        shards=shards,
        rows=sorted_highest_heap(policy_report_heap),
        preview_chars=GENRE_PROJECTION_PREVIEW_CHARS,
    )
    report_research_rows = materialize_projection_rows(
        shards=shards,
        rows=sorted_lowest_heap(research_report_heap),
        preview_chars=GENRE_PROJECTION_PREVIEW_CHARS,
    )

    ensure_dir(cache_npz.parent)
    np.savez(
        cache_npz,
        research_scores=research_scores.astype(np.float32),
        research_sdgs=research_sdgs.astype(np.uint8),
        policy_scores=policy_scores.astype(np.float32),
        policy_sdgs=policy_sdgs.astype(np.uint8),
    )
    write_json(
        cache_json,
        {
            "payload": payload,
            "top_policy_like_rows": top_policy_like_rows,
            "top_research_like_rows": top_research_like_rows,
            "report_policy_rows": report_policy_rows,
            "report_research_rows": report_research_rows,
        },
    )
    return payload, {
        "research_scores": research_scores,
        "research_sdgs": research_sdgs,
        "policy_scores": policy_scores,
        "policy_sdgs": policy_sdgs,
        "top_policy_like_rows": normalize_projection_rows(top_policy_like_rows),
        "top_research_like_rows": normalize_projection_rows(top_research_like_rows),
        "report_policy_rows": normalize_projection_rows(report_policy_rows),
        "report_research_rows": normalize_projection_rows(report_research_rows),
    }


def build_register_projection_summary_rows(
    *,
    research_scores: np.ndarray,
    research_sdgs: np.ndarray,
    policy_scores: np.ndarray,
    policy_sdgs: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(summarize_score_group("corpus", "research", research_scores))
    rows.append(summarize_score_group("corpus", "policy", policy_scores))
    for sdg in range(1, N_SDG + 1):
        combined = np.concatenate(
            [
                research_scores[research_sdgs == sdg],
                policy_scores[policy_sdgs == sdg],
            ]
        ).astype(np.float32)
        rows.append(summarize_score_group("sdg", f"SDG {sdg}", combined))
        rows.append(
            summarize_score_group(
                "corpus_by_sdg",
                f"research|SDG {sdg}",
                research_scores[research_sdgs == sdg],
            )
        )
        rows.append(
            summarize_score_group(
                "corpus_by_sdg",
                f"policy|SDG {sdg}",
                policy_scores[policy_sdgs == sdg],
            )
        )
    return rows


def write_register_direction_interpretation_report(
    path: Path,
    *,
    report_policy_rows: list[dict[str, Any]],
    report_research_rows: list[dict[str, Any]],
    vector_path: Path,
) -> None:
    lines = [
        "# Register Direction Interpretation",
        "",
        "This file is an automatic manual-inspection aid. It exposes raw examples aligned with the learned global register direction.",
        "",
        f"- Vector source: `{vector_path}`",
        f"- Projection score: `dot(x, g)` where `g` is the SDG-balanced global register unit vector",
        "- No LLM summarization was applied.",
        "",
        "## 20 Highest-Scoring Policy Examples",
        "",
    ]
    for row in report_policy_rows[:GENRE_PROJECTION_REPORT_N]:
        lines.extend(
            [
                f"### Rank {row['rank']}: {row['doc_id']}",
                f"- SDG: {row['assigned_sdg']}",
                f"- Score: {row['projection_score']}",
                f"- Source: {row['source_doc_or_year']}",
                f"- Title: {row['title'] or '(not available)'}",
                f"- Preview: {row['text_preview']}",
                "",
            ]
        )
    lines.extend(["## 20 Lowest-Scoring Research Examples", ""])
    for row in report_research_rows[:GENRE_PROJECTION_REPORT_N]:
        lines.extend(
            [
                f"### Rank {row['rank']}: {row['doc_id']}",
                f"- SDG: {row['assigned_sdg']}",
                f"- Score: {row['projection_score']}",
                f"- Year: {row['source_doc_or_year']}",
                f"- Title: {row['title'] or '(not available)'}",
                f"- Preview: {row['text_preview']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_register_projection_distribution(
    figures_dir: Path,
    *,
    research_scores: np.ndarray,
    research_sdgs: np.ndarray,
    policy_scores: np.ndarray,
    policy_sdgs: np.ndarray,
) -> None:
    names, _ = build_sdg_name_maps()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    fig, (ax_hist, ax_sdg, ax_corpus) = plt.subplots(
        3,
        1,
        figsize=(8.6, 10.5),
        gridspec_kw={"height_ratios": [1.1, 1.6, 0.9]},
    )

    all_scores = np.concatenate([research_scores, policy_scores]).astype(np.float32)
    bins = np.linspace(float(all_scores.min()), float(all_scores.max()), 60)
    ax_hist.hist(research_scores, bins=bins, alpha=0.6, color="#1B9E77", label="Research", density=True)
    ax_hist.hist(policy_scores, bins=bins, alpha=0.6, color="#2166AC", label="Policy", density=True)
    ax_hist.axvline(0.0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    ax_hist.set_title("Projection onto the SDG-balanced global register direction", fontsize=8.5, loc="left")
    ax_hist.set_xlabel("Projection score: dot(x, g)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper left")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    res_stats = [build_box_stats(research_scores[research_sdgs == sdg], f"R{sdg}") for sdg in range(1, N_SDG + 1)]
    pol_stats = [build_box_stats(policy_scores[policy_sdgs == sdg], f"P{sdg}") for sdg in range(1, N_SDG + 1)]
    centers = np.arange(1, N_SDG + 1)
    res_positions = centers - 0.18
    pol_positions = centers + 0.18
    res_bxp = ax_sdg.bxp(res_stats, positions=res_positions, widths=0.28, showfliers=False, patch_artist=True)
    pol_bxp = ax_sdg.bxp(pol_stats, positions=pol_positions, widths=0.28, showfliers=False, patch_artist=True)
    for patch in res_bxp["boxes"]:
        patch.set(facecolor="#1B9E77", alpha=0.55)
    for patch in pol_bxp["boxes"]:
        patch.set(facecolor="#2166AC", alpha=0.55)
    for median in res_bxp["medians"] + pol_bxp["medians"]:
        median.set(color="#222222", linewidth=1.0)
    ax_sdg.set_xticks(centers)
    ax_sdg.set_xticklabels([str(sdg) for sdg in range(1, N_SDG + 1)])
    ax_sdg.set_xlabel("Assigned SDG")
    ax_sdg.set_ylabel("Projection score")
    ax_sdg.set_title("Projection-score distribution by SDG and corpus", fontsize=8.5, loc="left")
    ax_sdg.legend(
        [res_bxp["boxes"][0], pol_bxp["boxes"][0]],
        ["Research", "Policy"],
        loc="lower left",
    )
    ax_sdg.spines["top"].set_visible(False)
    ax_sdg.spines["right"].set_visible(False)

    corpus_stats = [
        build_box_stats(research_scores, "Research"),
        build_box_stats(policy_scores, "Policy"),
    ]
    corpus_bxp = ax_corpus.bxp(corpus_stats, positions=[1, 2], widths=0.45, showfliers=False, patch_artist=True)
    corpus_bxp["boxes"][0].set(facecolor="#1B9E77", alpha=0.55)
    corpus_bxp["boxes"][1].set(facecolor="#2166AC", alpha=0.55)
    for median in corpus_bxp["medians"]:
        median.set(color="#222222", linewidth=1.0)
    ax_corpus.set_xticks([1, 2])
    ax_corpus.set_xticklabels(["Research", "Policy"])
    ax_corpus.set_ylabel("Projection score")
    ax_corpus.set_title("Projection-score distribution by corpus", fontsize=8.5, loc="left")
    ax_corpus.spines["top"].set_visible(False)
    ax_corpus.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_register_projection_distribution.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_register_projection_distribution.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def load_or_build_register_sdg_alignment_cache(
    *,
    cache_root: Path,
    base_signature: dict[str, Any],
    sdg_balanced_method_signature: dict[str, Any],
    sdg_centroids: np.ndarray,
    global_unit: np.ndarray,
    within_vectors: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "phase": "register_sdg_alignment",
        "base_signature": base_signature,
        "sdg_balanced_method_signature": sdg_balanced_method_signature,
        "has_within_vectors": bool(within_vectors is not None),
    }
    key = stable_cache_key(payload)
    cache_path = cache_json_path(cache_root, "register_sdg_alignment", key)
    if cache_path.exists():
        log.info("Cache hit: register-vs-SDG alignment (%s)", key)
        return payload, load_json(cache_path)

    log.info("Cache miss: register-vs-SDG alignment (%s)", key)
    alignment_rows: list[dict[str, Any]] = []
    for sdg_idx in range(N_SDG):
        cosine = float(np.dot(global_unit, sdg_centroids[sdg_idx]))
        alignment_rows.append(
            {
                "sdg": sdg_idx + 1,
                "cosine_similarity": round(cosine, 6),
                "abs_cosine_similarity": round(abs(cosine), 6),
            }
        )

    abs_vals = [float(row["abs_cosine_similarity"]) for row in alignment_rows]
    strongest = max(alignment_rows, key=lambda row: float(row["abs_cosine_similarity"]))
    weakest = min(alignment_rows, key=lambda row: float(row["abs_cosine_similarity"]))
    summary = {
        "mean_absolute_cosine": round(float(np.mean(abs_vals)), 6),
        "median_absolute_cosine": round(float(np.median(abs_vals)), 6),
        "max_absolute_cosine": round(float(np.max(abs_vals)), 6),
        "min_absolute_cosine": round(float(np.min(abs_vals)), 6),
        "strongest_alignment_sdg": int(strongest["sdg"]),
        "strongest_alignment_value": float(strongest["cosine_similarity"]),
        "weakest_alignment_sdg": int(weakest["sdg"]),
        "weakest_alignment_value": float(weakest["cosine_similarity"]),
    }

    within_rows: list[dict[str, Any]] = []
    if within_vectors is not None:
        for sdg_idx in range(N_SDG):
            vec = within_vectors[sdg_idx]
            available = bool(np.all(np.isfinite(vec)))
            cosine = None
            abs_cosine = None
            if available:
                raw_cos = float(np.dot(vec, sdg_centroids[sdg_idx]))
                cosine = round(raw_cos, 6)
                abs_cosine = round(abs(raw_cos), 6)
            within_rows.append(
                {
                    "sdg": sdg_idx + 1,
                    "cosine_similarity": cosine,
                    "abs_cosine_similarity": abs_cosine,
                    "classifier_available": available,
                }
            )

    payload_out = {
        "alignment_rows": alignment_rows,
        "summary": summary,
        "within_rows": within_rows,
    }
    write_json(cache_path, payload_out)
    return payload, payload_out


def plot_register_sdg_alignment(figures_dir: Path, alignment_rows: list[dict[str, Any]]) -> None:
    xs = [int(row["sdg"]) for row in alignment_rows]
    ys = [float(row["cosine_similarity"]) for row in alignment_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#2166AC" if y >= 0 else "#B2182B" for y in ys]
    ax.bar(xs, ys, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("SDG")
    ax.set_ylabel("Cosine similarity with global register direction")
    ax.set_title("Alignment between the SDG-balanced register direction and SDG centroids", fontsize=8.5, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_register_sdg_alignment.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig_register_sdg_alignment.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_register_interpretability_latex_outputs(
    tables_dir: Path,
    *,
    alignment_rows: list[dict[str, Any]],
    alignment_summary: dict[str, Any],
    within_alignment_rows: list[dict[str, Any]],
    report_policy_rows: list[dict[str, Any]],
    report_research_rows: list[dict[str, Any]],
) -> None:
    within_map = {int(row["sdg"]): row for row in within_alignment_rows}
    policy_mean = float(np.mean([float(row["projection_score"]) for row in report_policy_rows[:GENRE_PROJECTION_REPORT_N]]))
    research_mean = float(np.mean([float(row["projection_score"]) for row in report_research_rows[:GENRE_PROJECTION_REPORT_N]]))
    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/3_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\RegisterProjectionPolicyTopMean}}{{{policy_mean:.3f}}}",
        rf"\newcommand{{\RegisterProjectionResearchBottomMean}}{{{research_mean:.3f}}}",
        rf"\newcommand{{\RegisterAlignmentMeanAbsCosine}}{{{float(alignment_summary['mean_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegisterAlignmentMedianAbsCosine}}{{{float(alignment_summary['median_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegisterAlignmentMaxAbsCosine}}{{{float(alignment_summary['max_absolute_cosine']):.3f}}}",
        rf"\newcommand{{\RegisterAlignmentStrongestSdg}}{{SDG {int(alignment_summary['strongest_alignment_sdg'])}}}",
        rf"\newcommand{{\RegisterAlignmentWeakestSdg}}{{SDG {int(alignment_summary['weakest_alignment_sdg'])}}}",
    ]
    (tables_dir / "num_register_interpretability.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    align_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"SDG & $\cos(g, c_k)$ & $|\cos(g, c_k)|$ & $\cos(g_k, c_k)$ \\",
        r"\midrule",
    ]
    for row in alignment_rows:
        within_row = within_map.get(int(row["sdg"]))
        within_cell = ""
        if within_row and within_row["classifier_available"] and within_row["cosine_similarity"] is not None:
            within_cell = f"{float(within_row['cosine_similarity']):.3f}"
        align_lines.append(
            rf"SDG {int(row['sdg'])} & {float(row['cosine_similarity']):.3f} & "
            rf"{float(row['abs_cosine_similarity']):.3f} & {within_cell} \\"
        )
    align_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_register_sdg_alignment.tex").write_text("\n".join(align_lines) + "\n", encoding="utf-8")

    example_lines = [
        r"\begin{tabular}{llrp{0.18\textwidth}p{0.42\textwidth}}",
        r"\toprule",
        r"Side & SDG & Score & ID & Preview \\",
        r"\midrule",
    ]
    for row in report_policy_rows[:5]:
        example_lines.append(
            rf"Policy-like & SDG {int(row['assigned_sdg'])} & {float(row['projection_score']):.3f} & "
            rf"{latex_escape(compact_table_id(str(row['doc_id'])))} & {latex_escape(strip_text(row['text_preview'], GENRE_TABLE_PREVIEW_CHARS))} \\"
        )
    example_lines.append(r"\midrule")
    for row in report_research_rows[:5]:
        example_lines.append(
            rf"Research-like & SDG {int(row['assigned_sdg'])} & {float(row['projection_score']):.3f} & "
            rf"{latex_escape(compact_table_id(str(row['doc_id'])))} & {latex_escape(strip_text(row['text_preview'], GENRE_TABLE_PREVIEW_CHARS))} \\"
        )
    example_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tables_dir / "tab_register_projection_examples.tex").write_text("\n".join(example_lines) + "\n", encoding="utf-8")


def write_latex_outputs(
    tables_dir: Path,
    merged_rows: list[dict[str, Any]],
    metrics_payload: dict[str, Any],
) -> None:
    names, num_words = build_sdg_name_maps()
    valid_raw = [float(row["raw_gap"]) for row in merged_rows if row["raw_gap"] is not None]
    valid_adj = [float(row["register_adjusted_gap"]) for row in merged_rows if row["register_adjusted_gap"] is not None]
    valid_delta = [float(row["delta_gap"]) for row in merged_rows if row["delta_gap"] is not None]
    mean_raw = float(np.mean(valid_raw))
    mean_adj = float(np.mean(valid_adj))
    mean_delta = float(np.mean(valid_delta))
    median_adj = float(np.median(valid_adj))

    shrink_row = min(
        (row for row in merged_rows if row["delta_gap"] is not None),
        key=lambda row: float(row["delta_gap"]),
    )
    largest_adj_row = max(
        (row for row in merged_rows if row["register_adjusted_gap"] is not None),
        key=lambda row: float(row["register_adjusted_gap"]),
    )
    selected = metrics_payload["selected_model"]
    test_metrics = metrics_payload["selected_model_test_metrics"]
    split_sizes = metrics_payload["split_sizes"]
    sample_balance = metrics_payload["sample_class_balance"]

    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/3_register_adjustment.py — do not edit manually",
        rf"\newcommand{{\RegisterClassifierSamplePerClass}}{{{sample_balance['research']:,}}}",
        rf"\newcommand{{\RegisterClassifierTrainN}}{{{split_sizes['train']}}}",
        rf"\newcommand{{\RegisterClassifierValN}}{{{split_sizes['validation']}}}",
        rf"\newcommand{{\RegisterClassifierTestN}}{{{split_sizes['test']}}}",
        rf"\newcommand{{\RegisterClassifierSelectedC}}{{{selected['C']}}}",
        rf"\newcommand{{\RegisterClassifierTestAccuracy}}{{{test_metrics['accuracy']:.3f}}}",
        rf"\newcommand{{\RegisterClassifierTestRocAuc}}{{{test_metrics['roc_auc']:.3f}}}",
        rf"\newcommand{{\RegisterClassifierTestFOne}}{{{test_metrics['f1']:.3f}}}",
        rf"\newcommand{{\RegisterClassifierTestPrecision}}{{{test_metrics['precision']:.3f}}}",
        rf"\newcommand{{\RegisterClassifierTestRecall}}{{{test_metrics['recall']:.3f}}}",
        rf"\newcommand{{\MeanRawSemanticGap}}{{{mean_raw:.3f}}}",
        rf"\newcommand{{\MeanAdjustedSemanticGap}}{{{mean_adj:.3f}}}",
        rf"\newcommand{{\MeanSemanticGapDelta}}{{{mean_delta:.3f}}}",
        rf"\newcommand{{\MedianAdjustedSemanticGap}}{{{median_adj:.3f}}}",
        rf"\newcommand{{\LargestGapShrinkSdg}}{{SDG {shrink_row['sdg']}}}",
        rf"\newcommand{{\LargestGapShrinkDelta}}{{{float(shrink_row['delta_gap']):.3f}}}",
        rf"\newcommand{{\LargestAdjustedGapSdg}}{{SDG {largest_adj_row['sdg']}}}",
        rf"\newcommand{{\LargestAdjustedGapValue}}{{{float(largest_adj_row['register_adjusted_gap']):.3f}}}",
    ]
    for sdg_num, word in num_words.items():
        row = merged_rows[sdg_num - 1]
        if row["raw_gap"] is not None:
            num_lines.append(rf"\newcommand{{\RawSemanticGapSdg{word}}}{{{float(row['raw_gap']):.3f}}}")
        if row["register_adjusted_gap"] is not None:
            num_lines.append(
                rf"\newcommand{{\AdjustedSemanticGapSdg{word}}}{{{float(row['register_adjusted_gap']):.3f}}}"
            )
        if row["delta_gap"] is not None:
            num_lines.append(rf"\newcommand{{\SemanticGapDeltaSdg{word}}}{{{float(row['delta_gap']):.3f}}}")
    (tables_dir / "num_register_adjustment.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    sorted_rows = sorted(
        [row for row in merged_rows if row["register_adjusted_gap"] is not None],
        key=lambda row: float(row["register_adjusted_gap"]),
        reverse=True,
    )
    tab_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"SDG & Description & Raw gap & Adjusted gap & $\Delta$ gap & n$_{\text{res}}$ & n$_{\text{pol docs}}$ \\",
        r"\midrule",
    ]
    for row in sorted_rows:
        tab_lines.append(
            rf"SDG {row['sdg']:2d} & {names[row['sdg']]} & {float(row['raw_gap']):.3f} & "
            rf"{float(row['register_adjusted_gap']):.3f} & {float(row['delta_gap']):.3f} & "
            rf"{int(row['n_papers']):,} & {int(row['n_policy_docs_capped']):,} \\"
        )
    tab_lines.extend(
        [
            r"\midrule",
            r"\multicolumn{2}{l}{Mean gap} & \MeanRawSemanticGap & \MeanAdjustedSemanticGap & \MeanSemanticGapDelta & & \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (tables_dir / "tab_register_adjusted_semgap.tex").write_text(
        "\n".join(tab_lines) + "\n",
        encoding="utf-8",
    )


def plot_gap_comparison(figures_dir: Path, merged_rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in merged_rows if row["register_adjusted_gap"] is not None]
    if not valid_rows:
        raise RuntimeError("No finite register-adjusted semantic-gap rows available for plotting.")

    names, _ = build_sdg_name_maps()
    valid_rows = sorted(valid_rows, key=lambda row: float(row["register_adjusted_gap"]), reverse=True)
    y = np.arange(len(valid_rows))
    height = 0.36

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        y - height / 2,
        [float(row["raw_gap"]) for row in valid_rows],
        height=height,
        color="#BDBDBD",
        alpha=0.9,
        label="Raw semantic gap",
    )
    ax.barh(
        y + height / 2,
        [float(row["register_adjusted_gap"]) for row in valid_rows],
        height=height,
        color="#2166AC",
        alpha=0.9,
        label="Register-adjusted semantic gap",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"SDG {row['sdg']} {names[row['sdg']]}" for row in valid_rows], fontsize=7.5)
    ax.set_xlabel("Gap = 1 - cosine similarity between research and policy sub-centroids")
    ax.set_title(
        "One-direction global register sensitivity check: raw vs adjusted semantic gap",
        fontsize=8.5,
        loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_register_adjusted_semantic_gap_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        figures_dir / "fig_register_adjusted_semantic_gap_comparison.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_split_fracs(args.train_frac, args.val_frac, args.test_frac)
    validate_test_size(args.test_size)
    c_grid = parse_c_grid(args.c_grid)
    multi_direction_ks = sorted(set(parse_int_list(args.multi_direction_ks)))
    if 1 not in multi_direction_ks:
        multi_direction_ks = [1] + multi_direction_ks
    if any(k <= 0 for k in multi_direction_ks):
        raise ValueError(f"All multi-direction k values must be positive, got {multi_direction_ks}")
    if args.topic_match_top_k <= 0:
        raise ValueError("--topic-match-top-k must be positive")
    if args.samples_per_cell is not None and args.samples_per_cell <= 0:
        raise ValueError("--samples-per-cell must be positive when provided.")
    if args.min_samples_per_class <= 1:
        raise ValueError("--min-samples-per-class must be greater than 1.")

    layout = ensure_register_robustness_outputs(Path(args.output_dir))
    cache_root = Path(args.cache_dir)
    ensure_dir(cache_root)
    out_combined_json = layout.data_dir / "register_adjusted_semantic_gaps.json"
    out_combined_csv = layout.data_dir / "register_adjusted_semantic_gaps.csv"
    out_metrics = layout.data_dir / "register_adjustment_classifier_metrics.json"
    out_val_grid = layout.data_dir / "register_adjustment_classifier_validation_grid.csv"
    out_extremes = layout.data_dir / "register_extreme_examples.csv"
    out_tfidf = layout.data_dir / "register_tfidf_terms.csv"
    log.info("Robustness output dir: %s", layout.root.parent)
    log.info("Register data dir: %s", layout.data_dir)
    log.info("Cache dir: %s", cache_root)

    required_paths = [
        EMBED_MANIFEST,
        SCORE_MANIFEST,
        TEXT_MANIFEST,
        POLICY_EMB,
        POLICY_IDS,
        POLICY_TEXT_IDS,
        POLICY_SCORES,
        SDG_CENTROIDS,
        RESEARCH_CENTROIDS,
        RESEARCH_CENTROID_META,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Register-adjustment stage is missing required inputs: " + ", ".join(missing))
    base_signature = {
        "required_inputs": cache_input_signatures(required_paths),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "script": "1_code/3_main_analysis/3_appendix/3_register_adjustment.py",
    }

    shards, total_research_rows = build_research_shards()
    log.info("Research shards aligned: %d shards, %d rows", len(shards), total_research_rows)

    policy_emb = np.load(POLICY_EMB).astype(np.float32)
    policy_score_ids = load_json(POLICY_IDS)
    policy_text_ids = load_json(POLICY_TEXT_IDS)
    if policy_emb.shape[0] != len(policy_score_ids) or policy_emb.shape[0] != len(policy_text_ids):
        raise RuntimeError(
            "Policy embeddings, policy score IDs, and policy text IDs must have matching row counts."
        )
    for idx, row in enumerate(policy_score_ids):
        if row["id"] != policy_text_ids[idx]["id"]:
            raise RuntimeError("Policy score IDs and policy text IDs are not aligned.")

    log.info("Preparing balanced classifier data: %d per class", args.sample_per_class)
    sample_signature, research_sample_indices, policy_sample_indices, research_sample_emb = load_or_build_sample_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        shards=shards,
        total_research_rows=total_research_rows,
        policy_rows=policy_emb.shape[0],
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    policy_sample_emb = policy_emb[policy_sample_indices]
    X = np.vstack([research_sample_emb, policy_sample_emb]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(args.sample_per_class, dtype=np.int64),
            np.ones(args.sample_per_class, dtype=np.int64),
        ]
    )

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=args.val_frac + args.test_frac,
        stratify=y,
        random_state=args.seed,
    )
    holdout_test_frac = args.test_frac / (args.val_frac + args.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=holdout_test_frac,
        stratify=y_holdout,
        random_state=args.seed,
    )

    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    split_fracs = {
        "train": args.train_frac,
        "validation": args.val_frac,
        "test": args.test_frac,
    }
    split_sizes = {
        "train": int(X_train.shape[0]),
        "validation": int(X_val.shape[0]),
        "test": int(X_test.shape[0]),
        "train_plus_validation": int(X_train_val.shape[0]),
    }
    sample_class_balance = {
        "research": int(args.sample_per_class),
        "policy": int(args.sample_per_class),
    }
    classifier_signature, candidate_rows, metrics_payload, coef, register_unit, intercept = load_or_build_classifier_cache(
        cache_root=cache_root,
        sample_signature=sample_signature,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        c_grid=c_grid,
        split_fracs=split_fracs,
        split_sizes=split_sizes,
        sample_class_balance=sample_class_balance,
        seed=args.seed,
    )
    selected_c = float(metrics_payload["selected_model"]["C"])

    with out_val_grid.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C",
                "train_accuracy",
                "train_roc_auc",
                "train_f1",
                "train_precision",
                "train_recall",
                "val_accuracy",
                "val_roc_auc",
                "val_f1",
                "val_precision",
                "val_recall",
            ],
        )
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(row)

    out_metrics.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    log.info("Saved classifier metrics: %s", out_metrics)

    research_centroids = np.load(RESEARCH_CENTROIDS).astype(np.float32)
    research_meta = load_json(RESEARCH_CENTROID_META)
    research_counts = np.array([int(row["n_papers_assigned"]) for row in research_meta], dtype=np.int64)
    research_cohesions = np.array([float(row["mean_cos_to_centroid"]) for row in research_meta], dtype=np.float32)
    policy_scores = np.load(POLICY_SCORES).astype(np.float32)
    policy_assignments = get_cluster_assignments(policy_scores)

    raw_rng = np.random.default_rng(args.seed)
    raw_results = compute_sdg_semantic_gaps(
        research_centroids=research_centroids,
        research_counts=research_counts,
        research_cohesions=research_cohesions,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_score_ids,
        segment_cap=args.segment_cap,
        rng=raw_rng,
    )

    _, adjusted_policy_emb = load_or_build_adjusted_policy_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        classifier_signature=classifier_signature,
        policy_emb=policy_emb,
        register_unit=register_unit,
    )
    _, adjusted_research_centroids, adjusted_research_counts, adjusted_research_cohesions = load_or_build_adjusted_research_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        classifier_signature=classifier_signature,
        shards=shards,
        register_unit=register_unit,
    )
    if not np.array_equal(research_counts, adjusted_research_counts):
        raise RuntimeError("Research counts changed after adjustment; assignments must remain fixed.")

    adjusted_rng = np.random.default_rng(args.seed)
    adjusted_results = compute_sdg_semantic_gaps(
        research_centroids=adjusted_research_centroids,
        research_counts=adjusted_research_counts,
        research_cohesions=adjusted_research_cohesions,
        policy_emb=adjusted_policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_score_ids,
        segment_cap=args.segment_cap,
        rng=adjusted_rng,
    )

    merged_rows = merge_gap_results(raw_results, adjusted_results)
    combined_payload = {
        "robustness_check": "register_adjusted_semantic_gap",
        "canonical_semantic_gap_remains": "raw semantic gap",
        "seed": args.seed,
        "segment_cap": args.segment_cap,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "classifier_metrics_file": out_metrics.name,
        "note": (
            "This is an additive robustness check. The existing raw semantic gap remains canonical. "
            "The outputs here report how much within-SDG gaps change after subtracting a learned research-vs-policy "
            "register/register direction. "
            "The register direction is estimated by logistic regression on a balanced research-vs-policy embedding sample, "
            "selected by train/validation/test evaluation without using the test split to fit the final direction. "
            "These adjusted gaps are sensitivity diagnostics, not replacement estimates of a pure substantive gap."
        ),
        "per_sdg": merged_rows,
        "reliable_sdgs": [row["sdg"] for row in merged_rows if not row["unreliable"]],
        "unreliable_sdgs": [row["sdg"] for row in merged_rows if row["unreliable"]],
    }
    out_combined_json.write_text(json.dumps(combined_payload, indent=2) + "\n", encoding="utf-8")
    write_combined_csv(out_combined_csv, merged_rows)
    log.info("Saved combined semantic gaps: %s", out_combined_json)

    _, extreme_rows = load_or_build_extreme_examples_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        classifier_signature=classifier_signature,
        top_n=args.extreme_top_n,
        policy_emb=policy_emb,
        policy_text_ids=policy_text_ids,
        policy_score_ids=policy_score_ids,
        policy_assignments=policy_assignments,
        shards=shards,
        coef=coef,
        intercept=intercept,
    )
    write_extreme_examples_csv(out_extremes, extreme_rows)
    log.info("Saved extreme examples: %s", out_extremes)

    if not args.skip_tfidf_helper:
        _, tfidf_cache_path = load_or_build_tfidf_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            sample_signature=sample_signature,
            selected_c=selected_c,
            seed=args.seed,
            tfidf_sample_per_class=args.tfidf_sample_per_class,
            tfidf_max_features=args.tfidf_max_features,
            policy_text_ids=policy_text_ids,
            research_sample_indices=research_sample_indices,
            policy_sample_indices=policy_sample_indices,
            shards=shards,
        )
        shutil.copyfile(tfidf_cache_path, out_tfidf)
        log.info("Saved TF-IDF interpretability terms: %s", out_tfidf)
    else:
        with out_tfidf.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["side", "term", "coefficient", "rank"])
            writer.writeheader()
        log.info("TF-IDF helper skipped; wrote empty placeholder CSV: %s", out_tfidf)

    write_latex_outputs(layout.tables_dir, merged_rows, metrics_payload)
    log.info("Saved LaTeX outputs to %s", layout.tables_dir)
    plot_gap_comparison(layout.figures_dir, merged_rows)
    log.info("Saved figure outputs to %s", layout.figures_dir)

    register_methods_seed = args.seed if args.random_seed is None else args.random_seed
    register_adjustment_dir = layout.root
    register_data_dir = layout.data_dir
    research_indices_by_sdg_signature, research_indices_by_sdg = load_or_build_research_indices_by_sdg_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        shards=shards,
    )
    policy_indices_by_sdg = build_policy_indices_by_sdg(policy_assignments)

    sdg_balanced_rows: list[dict[str, Any]] | None = None
    within_sdg_rows: list[dict[str, Any]] | None = None
    sdg_balanced_metrics_out: dict[str, Any] | None = None
    within_sdg_metrics_rows: list[dict[str, Any]] | None = None
    cosine_rows: list[dict[str, Any]] | None = None
    sdg_balanced_unit: np.ndarray | None = None

    sdg_balanced_sample_signature, sdg_balanced_sample = load_or_build_sdg_balanced_sample_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        research_indices_by_sdg_signature=research_indices_by_sdg_signature,
        research_indices_by_sdg=research_indices_by_sdg,
        policy_indices_by_sdg=policy_indices_by_sdg,
        requested_samples_per_cell=args.samples_per_cell,
        min_samples_per_class=args.min_samples_per_class,
        seed=register_methods_seed,
    )
    log.info(
        "SDG-balanced sample plan: %d per SDG x register cell",
        int(sdg_balanced_sample["effective_samples_per_cell"]),
    )
    _, sdg_balanced_X, sdg_balanced_y, sdg_balanced_strata = load_or_build_sdg_balanced_dataset_cache(
        cache_root=cache_root,
        sample_signature=sdg_balanced_sample_signature,
        shards=shards,
        policy_emb=policy_emb,
        sampled_plan=sdg_balanced_sample,
    )
    sdg_balanced_method_signature, sdg_balanced_method = load_or_build_sdg_balanced_method_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        sample_signature=sdg_balanced_sample_signature,
        classifier_type=args.classifier_type,
        test_size=args.test_size,
        seed=register_methods_seed,
        X=sdg_balanced_X,
        y=sdg_balanced_y,
        strata=sdg_balanced_strata,
    )
    sdg_balanced_unit = np.array(sdg_balanced_method["register_unit"], dtype=np.float32)

    if args.method in ("sdg_balanced", "both"):
        sdg_balanced_metrics_out = {
            "method": "sdg_balanced",
            "random_seed": register_methods_seed,
            "classifier_type": args.classifier_type,
            "requested_samples_per_cell": args.samples_per_cell,
            "effective_samples_per_cell": int(sdg_balanced_sample["effective_samples_per_cell"]),
            "min_samples_per_class": args.min_samples_per_class,
            "test_size": args.test_size,
            "cell_rows": sdg_balanced_sample["cell_rows"],
            "total_sample_size": int(sdg_balanced_X.shape[0]),
            "train_size": int(sdg_balanced_method["train_size"]),
            "test_size_n": int(sdg_balanced_method["test_size"]),
            "metrics": sdg_balanced_method["metrics"],
            "intercept": sdg_balanced_method["intercept"],
        }
        write_json(register_data_dir / "sdg_balanced_metrics.json", sdg_balanced_metrics_out)
        np.save(register_data_dir / "sdg_balanced_register_vector.npy", sdg_balanced_unit.astype(np.float32))

        _, sdg_balanced_policy_emb = load_or_build_adjusted_policy_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            classifier_signature=sdg_balanced_method_signature,
            policy_emb=policy_emb,
            register_unit=sdg_balanced_unit,
        )
        _, sdg_balanced_research_centroids, sdg_balanced_research_counts, sdg_balanced_research_cohesions = load_or_build_adjusted_research_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            classifier_signature=sdg_balanced_method_signature,
            shards=shards,
            register_unit=sdg_balanced_unit,
        )
        if not np.array_equal(research_counts, sdg_balanced_research_counts):
            raise RuntimeError("SDG-balanced adjustment changed research counts unexpectedly.")
        sdg_balanced_results = compute_sdg_semantic_gaps(
            research_centroids=sdg_balanced_research_centroids,
            research_counts=sdg_balanced_research_counts,
            research_cohesions=sdg_balanced_research_cohesions,
            policy_emb=sdg_balanced_policy_emb,
            policy_assignments=policy_assignments,
            policy_ids=policy_score_ids,
            segment_cap=args.segment_cap,
            rng=np.random.default_rng(register_methods_seed),
        )
        sdg_balanced_rows = merge_method_gap_results(
            raw_results,
            sdg_balanced_results,
            adjusted_gap_field="sdg_balanced_adjusted_gap",
            adjusted_similarity_field="sdg_balanced_adjusted_similarity",
        )
        write_gap_comparison_csv(
            register_data_dir / "sdg_balanced_gap_comparison.csv",
            sdg_balanced_rows,
            adjusted_similarity_field="sdg_balanced_adjusted_similarity",
            adjusted_gap_field="sdg_balanced_adjusted_gap",
        )
        plot_sdg_balanced_gap_comparison(layout.figures_dir, sdg_balanced_rows)

    if args.method in ("within_sdg", "both"):
        within_signature, within_method = load_or_build_within_sdg_method_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            research_indices_by_sdg_signature=research_indices_by_sdg_signature,
            research_indices_by_sdg=research_indices_by_sdg,
            policy_indices_by_sdg=policy_indices_by_sdg,
            policy_emb=policy_emb,
            shards=shards,
            requested_samples_per_cell=args.samples_per_cell,
            min_samples_per_class=args.min_samples_per_class,
            test_size=args.test_size,
            classifier_type=args.classifier_type,
            seed=register_methods_seed,
        )
        within_vectors = np.array(within_method["vectors"], dtype=np.float32)
        available_sdgs = [int(sdg) for sdg in within_method["available_sdgs"]]
        skipped_sdgs = {int(sdg): reason for sdg, reason in within_method["skipped_sdgs"].items()}
        log.info(
            "Within-SDG classifiers fit for %d SDGs; skipped %d",
            len(available_sdgs),
            len(skipped_sdgs),
        )
        within_vectors_by_sdg = {sdg - 1: within_vectors[sdg - 1] for sdg in available_sdgs}
        within_average_direction = np.array(within_method["average_direction"], dtype=np.float32)
        cosine_rows, cosine_global_vs_average = build_register_vector_cosine_rows(
            global_unit=sdg_balanced_unit,
            within_vectors=within_vectors,
            available_sdgs=available_sdgs,
            average_direction=within_average_direction,
        )
        within_method["cosine_global_vs_average"] = cosine_global_vs_average
        write_rows_csv(
            register_data_dir / "within_sdg_metrics.csv",
            [
                "sdg",
                "available_research",
                "available_policy",
                "sampled_per_class",
                "train_size",
                "test_size",
                "accuracy",
                "macro_f1",
                "roc_auc",
                "coefficient_norm",
                "intercept",
                "classifier_available",
                "skip_reason",
            ],
            within_method["metrics_rows"],
        )
        within_sdg_metrics_rows = within_method["metrics_rows"]
        np.save(register_data_dir / "within_sdg_register_vectors.npy", within_vectors.astype(np.float32))
        write_rows_csv(
            register_data_dir / "register_vector_cosine_similarity.csv",
            ["comparison_type", "left", "right", "cosine_similarity"],
            cosine_rows,
        )
        plot_within_sdg_vector_similarity_heatmap(
            layout.figures_dir,
            within_vectors=within_vectors,
            available_sdgs=available_sdgs,
        )

        _, within_policy_emb = load_or_build_within_sdg_adjusted_policy_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            within_signature=within_signature,
            policy_emb=policy_emb,
            policy_assignments=policy_assignments,
            vectors_by_sdg=within_vectors_by_sdg,
        )
        _, within_research_centroids, within_research_counts, within_research_cohesions = load_or_build_within_sdg_adjusted_research_cache(
            cache_root=cache_root,
            base_signature=base_signature,
            within_signature=within_signature,
            shards=shards,
            vectors_by_sdg=within_vectors_by_sdg,
            raw_research_counts=research_counts,
        )
        within_results = compute_sdg_semantic_gaps(
            research_centroids=within_research_centroids,
            research_counts=within_research_counts,
            research_cohesions=within_research_cohesions,
            policy_emb=within_policy_emb,
            policy_assignments=policy_assignments,
            policy_ids=policy_score_ids,
            segment_cap=args.segment_cap,
            rng=np.random.default_rng(register_methods_seed),
        )
        within_sdg_rows = merge_method_gap_results(
            raw_results,
            within_results,
            adjusted_gap_field="within_sdg_adjusted_gap",
            adjusted_similarity_field="within_sdg_adjusted_similarity",
            skipped_sdgs=skipped_sdgs,
        )
        write_gap_comparison_csv(
            register_data_dir / "within_sdg_gap_comparison.csv",
            within_sdg_rows,
            adjusted_similarity_field="within_sdg_adjusted_similarity",
            adjusted_gap_field="within_sdg_adjusted_gap",
        )

    write_sdg_register_adjustment_note(
        register_adjustment_dir / "README_register_adjustment.md",
        method=args.method,
        seed=register_methods_seed,
        classifier_type=args.classifier_type,
        samples_per_cell=args.samples_per_cell,
        min_samples_per_class=args.min_samples_per_class,
        test_size=args.test_size,
    )
    write_json(
        register_data_dir / "summary.json",
        {
            "method": args.method,
            "random_seed": register_methods_seed,
            "classifier_type": args.classifier_type,
            "samples_per_cell": args.samples_per_cell,
            "min_samples_per_class": args.min_samples_per_class,
            "test_size": args.test_size,
            "sdg_balanced_mean_adjusted_gap": None if sdg_balanced_rows is None else mean_gap(sdg_balanced_rows, "sdg_balanced_adjusted_gap"),
            "within_sdg_mean_adjusted_gap": None if within_sdg_rows is None else mean_gap(within_sdg_rows, "within_sdg_adjusted_gap"),
            "global_vs_within_average_cosine": None if cosine_rows is None else next(
                (row["cosine_similarity"] for row in cosine_rows if row["comparison_type"] == "global_vs_average"),
                None,
            ),
            "projection_report_file": "register_direction_interpretation.md",
            "register_sdg_alignment_file": "register_sdg_alignment.csv",
        },
    )
    write_sdg_register_latex_outputs(
        layout.tables_dir,
        sdg_balanced_metrics=sdg_balanced_metrics_out,
        sdg_balanced_rows=sdg_balanced_rows,
        within_metrics_rows=within_sdg_metrics_rows,
        within_rows=within_sdg_rows,
        cosine_rows=cosine_rows,
    )
    plot_sdg_register_robustness_comparison(
        output_dir=layout.figures_dir,
        manuscript_figures_dir=layout.figures_dir,
        sdg_balanced_rows=sdg_balanced_rows,
        within_sdg_rows=within_sdg_rows,
    )

    projection_signature, projection_artifacts = load_or_build_register_projection_interpretability_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        sdg_balanced_method_signature=sdg_balanced_method_signature,
        shards=shards,
        total_research_rows=total_research_rows,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        policy_score_ids=policy_score_ids,
        policy_text_ids=policy_text_ids,
        register_unit=sdg_balanced_unit,
    )
    top_policy_like_path = register_data_dir / "top_policy_like_texts.csv"
    top_research_like_path = register_data_dir / "top_research_like_texts.csv"
    register_projection_summary_path = register_data_dir / "register_projection_summary.csv"
    register_direction_report_path = register_adjustment_dir / "register_direction_interpretation.md"
    write_rows_csv(
        top_policy_like_path,
        ["rank", "doc_id", "corpus_type", "assigned_sdg", "projection_score", "title", "text_preview", "source_doc_or_year"],
        projection_artifacts["top_policy_like_rows"],
    )
    write_rows_csv(
        top_research_like_path,
        ["rank", "doc_id", "corpus_type", "assigned_sdg", "projection_score", "title", "text_preview", "source_doc_or_year"],
        projection_artifacts["top_research_like_rows"],
    )
    projection_summary_rows = build_register_projection_summary_rows(
        research_scores=projection_artifacts["research_scores"],
        research_sdgs=projection_artifacts["research_sdgs"],
        policy_scores=projection_artifacts["policy_scores"],
        policy_sdgs=projection_artifacts["policy_sdgs"],
    )
    write_rows_csv(
        register_projection_summary_path,
        ["group_type", "group_value", "count", "mean_score", "median_score", "std_score", "min_score", "p25_score", "p75_score", "max_score"],
        projection_summary_rows,
    )
    write_register_direction_interpretation_report(
        register_direction_report_path,
        report_policy_rows=projection_artifacts["report_policy_rows"],
        report_research_rows=projection_artifacts["report_research_rows"],
        vector_path=register_data_dir / "sdg_balanced_register_vector.npy",
    )
    plot_register_projection_distribution(
        layout.figures_dir,
        research_scores=projection_artifacts["research_scores"],
        research_sdgs=projection_artifacts["research_sdgs"],
        policy_scores=projection_artifacts["policy_scores"],
        policy_sdgs=projection_artifacts["policy_sdgs"],
    )

    within_vectors_for_alignment: np.ndarray | None = None
    if args.method in ("within_sdg", "both"):
        within_vectors_for_alignment = within_vectors.astype(np.float32)
    else:
        within_vectors_path = register_data_dir / "within_sdg_register_vectors.npy"
        if within_vectors_path.exists():
            within_vectors_for_alignment = np.load(within_vectors_path).astype(np.float32)

    sdg_centroids = np.load(SDG_CENTROIDS).astype(np.float32)
    alignment_signature, alignment_artifacts = load_or_build_register_sdg_alignment_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        sdg_balanced_method_signature=sdg_balanced_method_signature,
        sdg_centroids=sdg_centroids,
        global_unit=sdg_balanced_unit,
        within_vectors=within_vectors_for_alignment,
    )
    register_sdg_alignment_path = register_data_dir / "register_sdg_alignment.csv"
    register_sdg_alignment_summary_path = register_data_dir / "register_sdg_alignment_summary.json"
    within_sdg_register_centroid_alignment_path = register_data_dir / "within_sdg_register_centroid_alignment.csv"
    write_rows_csv(
        register_sdg_alignment_path,
        ["sdg", "cosine_similarity", "abs_cosine_similarity"],
        alignment_artifacts["alignment_rows"],
    )
    write_json(register_sdg_alignment_summary_path, alignment_artifacts["summary"])
    write_rows_csv(
        within_sdg_register_centroid_alignment_path,
        ["sdg", "cosine_similarity", "abs_cosine_similarity", "classifier_available"],
        alignment_artifacts["within_rows"],
    )
    plot_register_sdg_alignment(layout.figures_dir, alignment_artifacts["alignment_rows"])
    write_register_interpretability_latex_outputs(
        layout.tables_dir,
        alignment_rows=alignment_artifacts["alignment_rows"],
        alignment_summary=alignment_artifacts["summary"],
        within_alignment_rows=alignment_artifacts["within_rows"],
        report_policy_rows=projection_artifacts["report_policy_rows"],
        report_research_rows=projection_artifacts["report_research_rows"],
    )

    regression_cell_stats_signature, regression_counts, regression_sums = load_or_build_regression_cell_stats_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        shards=shards,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
    )
    regression_signature, regression_artifacts = load_or_build_regression_method_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        cell_stats_signature=regression_cell_stats_signature,
        counts=regression_counts,
        sums=regression_sums,
    )
    regression_global_unit = np.array(regression_artifacts["global_unit"], dtype=np.float32)
    regression_within_units = np.array(regression_artifacts["within_units"], dtype=np.float32)
    regression_global_vector_path = register_data_dir / "regression_register_vector.npy"
    regression_within_vectors_path = register_data_dir / "regression_within_sdg_register_vectors.npy"
    regression_alignment_path = register_data_dir / "regression_register_sdg_alignment.csv"
    regression_similarity_path = register_data_dir / "regression_vs_classifier_similarity.json"
    regression_gap_path = register_data_dir / "regression_gap_comparison.csv"
    np.save(regression_global_vector_path, regression_global_unit.astype(np.float32))
    np.save(regression_within_vectors_path, regression_within_units.astype(np.float32))

    regression_alignment_rows, regression_alignment_summary = build_regression_alignment_rows(
        classifier_alignment_rows=alignment_artifacts["alignment_rows"],
        regression_global_unit=regression_global_unit,
        regression_within_units=regression_within_units,
        sdg_centroids=sdg_centroids,
    )
    regression_similarity_payload = build_regression_similarity_payload(
        regression_global_unit=regression_global_unit,
        classifier_global_unit=sdg_balanced_unit,
        regression_alignment_summary=regression_alignment_summary,
        classifier_alignment_summary=alignment_artifacts["summary"],
        baseline_sdg=int(regression_artifacts["baseline_sdg"]),
    )
    write_rows_csv(
        regression_alignment_path,
        [
            "sdg",
            "classifier_cosine_similarity",
            "regression_cosine_similarity",
            "regression_abs_cosine_similarity",
            "regression_within_sdg_cosine_similarity",
            "regression_within_sdg_abs_cosine_similarity",
        ],
        regression_alignment_rows,
    )
    write_json(regression_similarity_path, regression_similarity_payload)

    _, regression_global_policy_emb = load_or_build_adjusted_policy_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        classifier_signature=regression_signature,
        policy_emb=policy_emb,
        register_unit=regression_global_unit,
    )
    _, regression_global_research_centroids, regression_global_research_counts, regression_global_research_cohesions = load_or_build_adjusted_research_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        classifier_signature=regression_signature,
        shards=shards,
        register_unit=regression_global_unit,
    )
    if not np.array_equal(research_counts, regression_global_research_counts):
        raise RuntimeError("Regression global adjustment changed research counts unexpectedly.")
    regression_global_results = compute_sdg_semantic_gaps(
        research_centroids=regression_global_research_centroids,
        research_counts=regression_global_research_counts,
        research_cohesions=regression_global_research_cohesions,
        policy_emb=regression_global_policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_score_ids,
        segment_cap=args.segment_cap,
        rng=np.random.default_rng(register_methods_seed + 701),
    )
    regression_global_rows = merge_method_gap_results(
        raw_results,
        regression_global_results,
        adjusted_gap_field="regression_global_adjusted_gap",
        adjusted_similarity_field="regression_global_adjusted_similarity",
    )

    regression_within_vectors_by_sdg = {
        sdg_idx: regression_within_units[sdg_idx]
        for sdg_idx in range(N_SDG)
    }
    _, regression_within_policy_emb = load_or_build_within_sdg_adjusted_policy_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        within_signature=regression_signature,
        policy_emb=policy_emb,
        policy_assignments=policy_assignments,
        vectors_by_sdg=regression_within_vectors_by_sdg,
    )
    _, regression_within_research_centroids, regression_within_research_counts, regression_within_research_cohesions = load_or_build_within_sdg_adjusted_research_cache(
        cache_root=cache_root,
        base_signature=base_signature,
        within_signature=regression_signature,
        shards=shards,
        vectors_by_sdg=regression_within_vectors_by_sdg,
        raw_research_counts=research_counts,
    )
    regression_within_results = compute_sdg_semantic_gaps(
        research_centroids=regression_within_research_centroids,
        research_counts=regression_within_research_counts,
        research_cohesions=regression_within_research_cohesions,
        policy_emb=regression_within_policy_emb,
        policy_assignments=policy_assignments,
        policy_ids=policy_score_ids,
        segment_cap=args.segment_cap,
        rng=np.random.default_rng(register_methods_seed + 702),
    )
    regression_within_rows = merge_method_gap_results(
        raw_results,
        regression_within_results,
        adjusted_gap_field="regression_within_sdg_adjusted_gap",
        adjusted_similarity_field="regression_within_sdg_adjusted_similarity",
    )

    regression_gap_rows = build_regression_gap_comparison_rows(
        raw_rows=raw_results,
        classifier_global_rows=sdg_balanced_rows,
        classifier_within_rows=within_sdg_rows,
        regression_global_rows=regression_global_rows,
        regression_within_rows=regression_within_rows,
    )
    write_regression_gap_comparison_csv(regression_gap_path, regression_gap_rows)
    plot_regression_vs_classifier_alignment(
        layout.figures_dir,
        rows=regression_alignment_rows,
        similarity_payload=regression_similarity_payload,
    )
    write_regression_latex_outputs(
        layout.tables_dir,
        regression_alignment_rows=regression_alignment_rows,
        regression_alignment_summary=regression_alignment_summary,
        similarity_payload=regression_similarity_payload,
        regression_gap_rows=regression_gap_rows,
    )

    write_json(
        register_data_dir / "summary.json",
        {
            "method": args.method,
            "random_seed": register_methods_seed,
            "classifier_type": args.classifier_type,
            "samples_per_cell": args.samples_per_cell,
            "min_samples_per_class": args.min_samples_per_class,
            "test_size": args.test_size,
            "sdg_balanced_mean_adjusted_gap": None if sdg_balanced_rows is None else mean_gap(sdg_balanced_rows, "sdg_balanced_adjusted_gap"),
            "within_sdg_mean_adjusted_gap": None if within_sdg_rows is None else mean_gap(within_sdg_rows, "within_sdg_adjusted_gap"),
            "regression_global_mean_adjusted_gap": mean_gap(regression_gap_rows, "regression_global_adjusted_gap"),
            "regression_within_sdg_mean_adjusted_gap": mean_gap(regression_gap_rows, "regression_within_sdg_adjusted_gap"),
            "global_vs_within_average_cosine": None if cosine_rows is None else next(
                (row["cosine_similarity"] for row in cosine_rows if row["comparison_type"] == "global_vs_average"),
                None,
            ),
            "regression_vs_classifier_cosine": regression_similarity_payload["global_regression_vs_classifier_cosine"],
            "projection_report_file": "register_direction_interpretation.md",
            "register_sdg_alignment_file": "register_sdg_alignment.csv",
            "regression_alignment_file": "regression_register_sdg_alignment.csv",
            "regression_gap_comparison_file": "regression_gap_comparison.csv",
        },
    )
    log.info("Saved SDG-aware register robustness outputs to %s", layout.root.parent)

    register_confidence_artifacts: dict[str, Any] | None = None
    if not args.skip_register_confidence_checks:
        confidence_dir = layout.root / args.local_check_subdir
        ensure_dir(confidence_dir)
        register_confidence_artifacts = {}

        adjusted_reseparability_signature, adjusted_reseparability = load_or_build_local_adjusted_reseparability(
            cache_root=cache_root,
            classifier_signature=classifier_signature,
            register_unit=register_unit,
            split_fracs=split_fracs,
            split_sizes=split_sizes,
            sample_class_balance=sample_class_balance,
            c_grid=c_grid,
            seed=args.seed,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            raw_metrics_payload=metrics_payload,
        )
        adjusted_resep_metrics_path = confidence_dir / "adjusted_reseparability_metrics.json"
        adjusted_resep_grid_path = confidence_dir / "adjusted_reseparability_validation_grid.csv"
        write_json(adjusted_resep_metrics_path, adjusted_reseparability["metrics_payload"])
        write_rows_csv(
            adjusted_resep_grid_path,
            [
                "C",
                "train_accuracy",
                "train_roc_auc",
                "train_f1",
                "train_precision",
                "train_recall",
                "val_accuracy",
                "val_roc_auc",
                "val_f1",
                "val_precision",
                "val_recall",
            ],
            adjusted_reseparability["candidate_rows"],
        )
        register_confidence_artifacts["adjusted_reseparability_metrics_json"] = str(adjusted_resep_metrics_path)
        register_confidence_artifacts["adjusted_reseparability_validation_grid_csv"] = str(adjusted_resep_grid_path)

        heldout_signature, heldout_generalization = load_or_build_local_heldout_sdg_generalization(
            cache_root=cache_root,
            base_signature=base_signature,
            research_indices_by_sdg_signature=research_indices_by_sdg_signature,
            research_indices_by_sdg=research_indices_by_sdg,
            policy_assignments=policy_assignments,
            policy_emb=policy_emb,
            shards=shards,
            c_grid=c_grid,
            split_fracs=split_fracs,
            seed=args.seed,
        )
        heldout_json_path = confidence_dir / "heldout_sdg_generalization.json"
        heldout_csv_path = confidence_dir / "heldout_sdg_generalization.csv"
        write_json(heldout_json_path, heldout_generalization)
        heldout_rows = []
        for row in heldout_generalization["fold_rows"]:
            heldout_rows.append(
                {
                    **row,
                    "heldout_sdgs": ",".join(str(s) for s in row["heldout_sdgs"]),
                    "train_sdgs": ",".join(str(s) for s in row["train_sdgs"]),
                    "test_confusion_matrix": json.dumps(row["test_confusion_matrix"]),
                }
            )
        write_rows_csv(
            heldout_csv_path,
            [
                "fold",
                "heldout_sdgs",
                "train_sdgs",
                "train_per_class",
                "test_per_class",
                "selected_C",
                "val_roc_auc",
                "val_f1",
                "test_accuracy",
                "test_roc_auc",
                "test_f1",
                "test_precision",
                "test_recall",
                "test_confusion_matrix",
            ],
            heldout_rows,
        )
        register_confidence_artifacts["heldout_sdg_generalization_json"] = str(heldout_json_path)
        register_confidence_artifacts["heldout_sdg_generalization_csv"] = str(heldout_csv_path)

        multi_signature, multi_direction = load_or_build_local_multi_direction(
            cache_root=cache_root,
            base_signature=base_signature,
            classifier_signature=classifier_signature,
            c_grid=c_grid,
            seed=args.seed,
            ks=multi_direction_ks,
            raw_rows=raw_results,
            register_unit=register_unit,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            policy_emb=policy_emb,
            policy_assignments=policy_assignments,
            policy_ids=policy_score_ids,
            shards=shards,
            research_counts=research_counts,
            segment_cap=args.segment_cap,
        )
        multi_json_path = confidence_dir / "multi_direction_gap_curve.json"
        multi_curve_csv_path = confidence_dir / "multi_direction_gap_curve.csv"
        multi_per_sdg_csv_path = confidence_dir / "multi_direction_per_sdg.csv"
        write_json(multi_json_path, multi_direction)
        curve_rows = []
        for row in multi_direction["curve_rows"]:
            curve_rows.append({**row, "top_adjusted_sdgs": ",".join(str(s) for s in row["top_adjusted_sdgs"])})
        write_rows_csv(
            multi_curve_csv_path,
            [
                "k",
                "test_accuracy",
                "test_roc_auc",
                "test_f1",
                "test_precision",
                "test_recall",
                "mean_adjusted_gap",
                "mean_delta_vs_raw",
                "top_adjusted_sdgs",
            ],
            curve_rows,
        )
        write_rows_csv(
            multi_per_sdg_csv_path,
            [
                "k",
                "sdg",
                "raw_gap",
                "adjusted_gap",
                "delta_gap",
                "n_papers",
                "n_policy_docs_capped",
                "unreliable",
            ],
            multi_direction["per_sdg_rows"],
        )
        register_confidence_artifacts["multi_direction_gap_curve_json"] = str(multi_json_path)
        register_confidence_artifacts["multi_direction_gap_curve_csv"] = str(multi_curve_csv_path)
        register_confidence_artifacts["multi_direction_per_sdg_csv"] = str(multi_per_sdg_csv_path)

        topic_signature, topic_match = load_or_build_local_topic_match(
            cache_root=cache_root,
            base_signature=base_signature,
            research_indices_by_sdg_signature=research_indices_by_sdg_signature,
            research_indices_by_sdg=research_indices_by_sdg,
            policy_emb=policy_emb,
            adjusted_policy_emb=adjusted_policy_emb,
            policy_assignments=policy_assignments,
            policy_ids=policy_score_ids,
            policy_text_ids=policy_text_ids,
            shards=shards,
            register_unit=register_unit,
            segment_cap=args.segment_cap,
            topic_match_research_per_sdg=args.topic_match_research_per_sdg,
            topic_match_top_k=args.topic_match_top_k,
            seed=args.seed,
        )
        topic_json_path = confidence_dir / "topic_matched_pair_summary.json"
        topic_csv_path = confidence_dir / "topic_matched_pair_summary.csv"
        topic_examples_csv_path = confidence_dir / "topic_matched_pair_examples.csv"
        write_json(topic_json_path, topic_match)
        write_rows_csv(
            topic_csv_path,
            [
                "sdg",
                "pair_count",
                "research_sample_n",
                "raw_mean_matched_cosine",
                "adjusted_mean_matched_cosine",
                "delta_mean_cosine",
                "raw_mean_matched_gap",
                "adjusted_mean_matched_gap",
            ],
            topic_match["per_sdg_rows"],
        )
        write_rows_csv(
            topic_examples_csv_path,
            [
                "sdg",
                "rank_within_sdg",
                "policy_item_id",
                "policy_source_doc",
                "research_item_id",
                "research_publication_year",
                "raw_cosine",
                "adjusted_cosine",
                "delta_cosine",
                "policy_text_snippet",
                "research_text_snippet",
            ],
            topic_match["example_rows"],
        )
        register_confidence_artifacts["topic_matched_pair_summary_json"] = str(topic_json_path)
        register_confidence_artifacts["topic_matched_pair_summary_csv"] = str(topic_csv_path)
        register_confidence_artifacts["topic_matched_pair_examples_csv"] = str(topic_examples_csv_path)

        register_confidence_summary = build_local_confidence_summary(
            adjusted_reseparability=adjusted_reseparability,
            heldout_generalization=heldout_generalization,
            multi_direction=multi_direction,
            topic_match=topic_match,
        )
        register_confidence_summary_path = confidence_dir / "register_confidence_summary.json"
        write_json(register_confidence_summary_path, register_confidence_summary)
        register_confidence_artifacts["register_confidence_summary_json"] = str(register_confidence_summary_path)

        write_register_confidence_latex_outputs(
            layout.tables_dir,
            register_confidence_summary,
            adjusted_reseparability,
            heldout_generalization,
            multi_direction,
            topic_match,
        )
        register_confidence_artifacts["latex_num_tex"] = str(layout.tables_dir / "num_register_confidence_checks.tex")
        register_confidence_artifacts["latex_tab_tex"] = str(layout.tables_dir / "tab_register_confidence_checks.tex")

        plot_register_confidence_curve(layout.figures_dir, multi_direction)
        register_confidence_artifacts["figure_pdf"] = str(layout.figures_dir / "fig_register_confidence_curve.pdf")
        register_confidence_artifacts["figure_png"] = str(layout.figures_dir / "fig_register_confidence_curve.png")

        register_confidence_manifest_path = confidence_dir / "manifest.json"
        write_json(
            register_confidence_manifest_path,
            {
                "check_bundle": "register_confidence_checks",
                "seed": args.seed,
                "c_grid": c_grid,
                "multi_direction_ks": multi_direction_ks,
                "topic_match_research_per_sdg": args.topic_match_research_per_sdg,
                "topic_match_top_k": args.topic_match_top_k,
                "heldout_sdg_folds": [list(fold) for fold in HELDOUT_SDG_FOLDS],
                "cache_keys": {
                    "research_indices_by_sdg": stable_cache_key(research_indices_by_sdg_signature),
                    "adjusted_reseparability": stable_cache_key(adjusted_reseparability_signature),
                    "heldout_sdg_generalization": stable_cache_key(heldout_signature),
                    "multi_direction": stable_cache_key(multi_signature),
                    "topic_matched": stable_cache_key(topic_signature),
                },
                "artifacts": register_confidence_artifacts,
            },
        )
        register_confidence_artifacts["manifest_json"] = str(register_confidence_manifest_path)
        log.info("Saved register-confidence robustness checks to %s", confidence_dir)

    write_cache_manifest(
        cache_root,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "base_signature": base_signature,
            "sample_signature": sample_signature,
            "classifier_signature": classifier_signature,
            "selected_c": selected_c,
            "robustness_output_dir": str(layout.root.parent),
            "register_data_dir": str(layout.root),
            "artifacts": {
                "combined_json": str(out_combined_json),
                "combined_csv": str(out_combined_csv),
                "metrics_json": str(out_metrics),
                "validation_grid_csv": str(out_val_grid),
                "extremes_csv": str(out_extremes),
                "tfidf_csv": str(out_tfidf),
                "sdg_register_dir": str(register_adjustment_dir),
                "register_readme_md": str(register_adjustment_dir / "README_register_adjustment.md"),
                "sdg_register_summary_json": str(register_data_dir / "summary.json"),
                "register_projection_policy_csv": str(top_policy_like_path),
                "register_projection_research_csv": str(top_research_like_path),
                "register_projection_summary_csv": str(register_projection_summary_path),
                "register_direction_interpretation_md": str(register_direction_report_path),
                "register_sdg_alignment_csv": str(register_sdg_alignment_path),
                "register_sdg_alignment_summary_json": str(register_sdg_alignment_summary_path),
                "within_sdg_register_centroid_alignment_csv": str(within_sdg_register_centroid_alignment_path),
                "regression_register_vector_npy": str(regression_global_vector_path),
                "regression_within_sdg_register_vectors_npy": str(regression_within_vectors_path),
                "regression_register_sdg_alignment_csv": str(regression_alignment_path),
                "regression_vs_classifier_similarity_json": str(regression_similarity_path),
                "regression_gap_comparison_csv": str(regression_gap_path),
            },
            "interpretability_cache_keys": {
                "register_projection_interpretability": stable_cache_key(projection_signature),
                "register_sdg_alignment": stable_cache_key(alignment_signature),
            },
            "regression_cache_keys": {
                "regression_cell_stats": stable_cache_key(regression_cell_stats_signature),
                "regression_method": stable_cache_key(regression_signature),
            },
            "register_confidence_artifacts": register_confidence_artifacts,
        },
    )


if __name__ == "__main__":
    main()
