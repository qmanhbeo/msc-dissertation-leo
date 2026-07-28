"""
Score the supervised (LR / MLP) classifier outputs against the research and policy corpora.

This is the consolidated successor to the three former scripts:
    0_score_research_shards.py  (LR research scoring)
    1_score_policy.py        (LR policy scoring)
    2_score_mlp.py          (MLP research+policy scoring)

Design note (hygiene fix): the old scripts exposed a `--classifier-type {lr,mlp}`
flag on the LR-path writers, which let an MLP run silently overwrite the LR
outputs in the shared `paper_scores_shards/` / `policy_scores*.npy` locations.
That footgun is structurally removed here: `--classifier` selects the model
family GLOBALLY, and the LR branches can only ever write to the legacy
LR paths while the MLP branch writes only to `mlp_scores/`. There is no
longer any code path where an MLP run can clobber LR output.

Outputs (paths UNCHANGED from the former scripts, so downstream loaders
in semantic_gap_shared.py / 0_coverage_gap.py / 0_check_centroid_consistency.py
need no edits):
    LR research:
        2_data/5_supervised_scored/{model}/paper_scores_shards/part-NNNN.npy
        2_data/5_supervised_scored/{model}/paper_scores_shards/metadata/{shard}_ids.jsonl
        2_data/5_supervised_scored/{model}/paper_scores_shards/metadata/manifest.json
        2_data/5_supervised_scored/{model}/research_centroids.npy
        2_data/5_supervised_scored/{model}/metadata/research_centroid_meta.json
    LR policy:
        2_data/5_supervised_scored/{model}/policy_scores.npy
        2_data/5_supervised_scored/{model}/policy_scores_vs_research.npy
        2_data/5_supervised_scored/{model}/metadata/policy_scores_ids.json
    MLP (research + policy, corpus flag ignored):
        2_data/5_supervised_scored/{model}/mlp_scores/mlp_research_centroids.npy
        2_data/5_supervised_scored/{model}/mlp_scores/mlp_policy_scores.npy
        2_data/5_supervised_scored/{model}/mlp_scores/mlp_policy_vs_research.npy
        2_data/5_supervised_scored/{model}/mlp_scores/mlp_summary.json

Run from project root:
    python 1_code/5_supervised_model_infer/score_supervised.py \
        --embed-model all-mpnet-base-v2 --classifier lr  --corpus research
    python 1_code/5_supervised_model_infer/score_supervised.py \
        --embed-model all-mpnet-base-v2 --classifier lr  --corpus policy
    python 1_code/5_supervised_model_infer/score_supervised.py \
        --embed-model all-mpnet-base-v2 --classifier mlp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from alignment_core import verify_unit_norms
from model_utils import (
    N_SDG,
    ZERO_NORM_EPS,
    NORM_EPS,
    embed_dir_for_model,
    embed_research_dir_for_model,
    model_results_dir_for_model,
    scored_dir_for_model,
    DEFAULT_EMBED_MODEL,
)
from shard_pipeline_utils import (
    atomic_write_json,
    ensure_dir,
    load_json,
    now_iso,
    read_json,
    resolve_manifest_path,
    sha256_file,
    update_stage_status,
)

log = logging.getLogger(__name__)
STATUS_STAGE = "supervised_sdg_scores"


# ---------------------------------------------------------------------------
# Shared model-unpickling machinery (must match training architecture)
# ---------------------------------------------------------------------------
class MultiLabelMLP(torch.nn.Module):
    """Must match the architecture used during training (1_retrain_full_data.py)."""

    def __init__(self, input_dim: int, n_layers: int = 4, hidden_size: int = 384,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_size, N_SDG))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _NetWrapper:
    def __init__(self, net, input_dim):
        self.net = net
        self.input_dim = input_dim

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X.astype(np.float32)))
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.float32)


def _load_model(model_root: Path, classifier_type: str, input_dim: int):
    """Load the retrained supervised classifier (sklearn LR or PyTorch MLP).

    Raises a clear error if the loaded model's input dimension does not match
    the embedding dimension recorded in the embedding manifest -- this is the
    failure mode that previously crashed MiniLM (384-dim) scoring with a
    shape error.
    """
    if classifier_type == "mlp":
        model_path = model_root / "model" / "mlp_retrained.joblib"
        model = joblib.load(model_path)
        _net = model
        while hasattr(_net, "net") and not isinstance(getattr(_net, "net", None), torch.nn.Sequential):
            _net = _net.net
        first_layer = _net.net[0]
        assert first_layer.in_features == input_dim, (
            f"MLP first layer in_features {first_layer.in_features} "
            f"!= embedding dim {input_dim}"
        )
        return model
    model_path = model_root / "model" / "sdg_classifier_retrained.joblib"
    clf = joblib.load(model_path)
    assert clf.coef_.shape[1] == input_dim, (
        f"Classifier n_features {clf.coef_.shape[1]} != embedding dim {input_dim}"
    )
    return clf


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < ZERO_NORM_EPS:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def load_ids(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, data: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(path)


def load_policy_doc_map(ids_path: Path) -> dict[str, dict]:
    with ids_path.open() as f:
        ids_meta = json.load(f)
    return {
        row["id"]: {
            "source_doc": row["source_doc"],
            "text": row["text"],
        }
        for row in ids_meta
    }


# ---------------------------------------------------------------------------
# LR research branch  (port of former 0_score_research_shards.py, classifier forced LR)
# ---------------------------------------------------------------------------
def run_research_lr(args) -> None:
    scored_root = scored_dir_for_model(args.embed_model)
    embed_root = embed_dir_for_model(args.embed_model)
    model_root = model_results_dir_for_model(args.embed_model)

    model_path_default = model_root / "model" / "sdg_classifier_retrained.joblib"
    embed_manifest_default = embed_research_dir_for_model(args.embed_model) / "metadata" / "manifest.json"
    out_dir_default = scored_root / "paper_scores_shards"
    metadata_dir_default = out_dir_default / "metadata"
    status_dir_default = embed_research_dir_for_model(args.embed_model) / "metadata"
    research_centroids_out_default = scored_root / "research_centroids.npy"
    research_meta_out_default = scored_root / "metadata" / "research_centroid_meta.json"

    model_path = Path(args.model_path) if args.model_path else model_path_default
    embed_manifest_path = Path(args.embedding_manifest) if args.embedding_manifest else embed_manifest_default
    out_dir = Path(args.out_dir) if args.out_dir else out_dir_default
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else metadata_dir_default
    status_dir = Path(args.status_dir) if args.status_dir else status_dir_default
    research_centroids_out = Path(args.research_centroids_out) if args.research_centroids_out else research_centroids_out_default
    research_meta_out = Path(args.research_meta_out) if args.research_meta_out else research_meta_out_default
    embed_dir = embed_manifest_path.parent.parent

    if (
        args.limit_shards > 0
        and not args.allow_partial_research_centroids
        and research_centroids_out == research_centroids_out_default
        and research_meta_out == research_meta_out_default
    ):
        raise RuntimeError(
            "Refusing to overwrite canonical research centroids from a partial shard run. "
            "Either remove --limit-shards, point outputs elsewhere, or pass --allow-partial-research-centroids."
        )

    ensure_dir(out_dir)
    ensure_dir(metadata_dir)
    ensure_dir(status_dir)

    emb_manifest = read_json(embed_manifest_path)
    input_dim = emb_manifest.get("embedding_dim", 768)
    log.info("Loading model: %s (classifier_type=lr, input_dim=%d)", model_path, input_dim)
    model = _load_model(model_root, "lr", input_dim)
    log.info("Model loaded (dims=%d)", input_dim)
    if not emb_manifest or "shards" not in emb_manifest:
        raise RuntimeError(f"Invalid embedding manifest: {embed_manifest_path}")

    out_manifest_path = metadata_dir / "manifest.json"
    out_manifest = read_json(out_manifest_path, default=None)
    if out_manifest is None:
        out_manifest = {
            "stage": STATUS_STAGE,
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "input_embedding_manifest": str(embed_manifest_path),
            "model_path": str(model_path),
            "shards": [],
            "totals": {"rows": 0, "shards": 0},
        }

    completed = {int(s["shard_id"]): s for s in out_manifest.get("shards", [])}
    if args.overwrite:
        completed = {}
        out_manifest["shards"] = []
        out_manifest["totals"]["rows"] = 0
        out_manifest["totals"]["shards"] = 0
        out_manifest["created_at_utc"] = now_iso()
    shards = emb_manifest["shards"][: args.limit_shards] if args.limit_shards > 0 else emb_manifest["shards"]

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "running",
        {
            "embedding_manifest": str(embed_manifest_path),
            "model_path": str(model_path),
        },
    )

    emb_dim = int(emb_manifest.get("embedding_dim") or emb_manifest["shards"][0]["dim"])
    d = emb_dim
    assert d == input_dim, f"Manifest dim {d} != model input_dim {input_dim}"
    sums = np.zeros((N_SDG, d), dtype=np.float64)
    counts = np.zeros(N_SDG, dtype=np.int64)

    for shard in shards:
        shard_id = int(shard["shard_id"])
        shard_name = shard["name"]
        emb_path = resolve_manifest_path(shard["embedding_path"], allowed_dirs=(embed_dir,))
        ids_in = resolve_manifest_path(shard["ids_path"], allowed_dirs=(embed_dir,))
        score_path = out_dir / f"{shard_name}.npy"
        ids_out = metadata_dir / f"{shard_name}_ids.jsonl"

        emb = np.load(emb_path).astype(np.float32)
        ids_rows = load_ids(ids_in)
        if emb.shape[0] != len(ids_rows):
            raise RuntimeError(f"Row mismatch in shard {shard_name}: emb={emb.shape[0]} ids={len(ids_rows)}")

        if not args.overwrite and shard_id in completed and score_path.exists() and ids_out.exists():
            log.info("Skip scoring shard %s (already complete)", shard_name)
            scored_ids = load_ids(ids_out)
            assigned = np.array([int(r["assigned_sdg"]) - 1 for r in scored_ids], dtype=np.int64)
        else:
            log.info("Scoring shard %s  shape=%s", shard_name, emb.shape)
            scores = model.predict_proba(emb).astype(np.float32)
            tmp_score = score_path.with_suffix(".npy.tmp")
            with tmp_score.open("wb") as f:
                np.save(f, scores)
            tmp_score.replace(score_path)

            assigned = scores.argmax(axis=1).astype(np.int64)
            scored_ids = []
            for i, row in enumerate(ids_rows):
                scored_ids.append(
                    {
                        "openalex_id": row["openalex_id"],
                        "publication_year": row.get("publication_year"),
                        "row_in_shard": i,
                        "assigned_sdg": int(assigned[i]) + 1,
                        "max_score": float(scores[i, assigned[i]]),
                    }
                )
            write_ids(ids_out, scored_ids)

            out_record = {
                "shard_id": shard_id,
                "name": shard_name,
                "score_path": str(score_path),
                "ids_path": str(ids_out),
                "rows": int(scores.shape[0]),
                "bytes": score_path.stat().st_size,
                "sha256": sha256_file(score_path),
                "ids_sha256": sha256_file(ids_out),
            }
            out_manifest["shards"] = [s for s in out_manifest["shards"] if int(s["shard_id"]) != shard_id]
            out_manifest["shards"].append(out_record)
            out_manifest["shards"].sort(key=lambda x: int(x["shard_id"]))
            out_manifest["totals"]["rows"] = int(sum(int(s["rows"]) for s in out_manifest["shards"]))
            out_manifest["totals"]["shards"] = int(len(out_manifest["shards"]))
            atomic_write_json(out_manifest_path, out_manifest)

        for sdg_idx in range(N_SDG):
            mask = assigned == sdg_idx
            if not np.any(mask):
                continue
            counts[sdg_idx] += int(mask.sum())
            sums[sdg_idx] += emb[mask].sum(axis=0)

        update_stage_status(
            status_dir,
            STATUS_STAGE,
            "running",
            {"last_completed_shard": shard_id, "rows_done": int(counts.sum())},
        )

    research_centroids = np.zeros((N_SDG, d), dtype=np.float32)
    meta: list[dict[str, Any]] = []
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        sdg = sdg_idx + 1
        if n == 0:
            meta.append(
                {
                    "sdg": sdg,
                    "n_papers_assigned": 0,
                    "raw_centroid_norm": 0.0,
                    "mean_cos_to_centroid": 0.0,
                    "zero_flag": True,
                }
            )
            continue
        raw = (sums[sdg_idx] / max(n, 1)).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        unit = normalize(raw)
        research_centroids[sdg_idx] = unit
        meta.append(
            {
                "sdg": sdg,
                "n_papers_assigned": n,
                "raw_centroid_norm": round(norm, 6),
                "mean_cos_to_centroid": round(norm, 6),
                "zero_flag": bool(norm < ZERO_NORM_EPS),
            }
        )

    ensure_dir(research_centroids_out.parent)
    with research_centroids_out.open("wb") as f:
        np.save(f, research_centroids)
    atomic_write_json(research_meta_out, meta)

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "completed",
        {
            "manifest_path": str(out_manifest_path),
            "research_centroids": str(research_centroids_out),
            "research_meta": str(research_meta_out),
            "rows_done": int(counts.sum()),
        },
    )
    log.info("Scoring complete. rows=%d", int(counts.sum()))


# ---------------------------------------------------------------------------
# LR policy branch  (port of former 1_score_policy.py, classifier forced LR)
# ---------------------------------------------------------------------------
def run_policy_lr(args) -> None:
    embed_root = embed_dir_for_model(args.embed_model)
    scored_root = scored_dir_for_model(args.embed_model)
    model_root = model_results_dir_for_model(args.embed_model)

    args.model_path = args.model_path or str(model_root / "model" / "sdg_classifier_retrained.joblib")
    args.policy_emb = args.policy_emb or str(embed_root / "policy.npy")
    args.policy_ids = args.policy_ids or str(embed_root / "metadata" / "policy_ids.json")
    args.research_centroids = args.research_centroids or str(scored_root / "research_centroids.npy")
    args.policy_scores_out = args.policy_scores_out or str(scored_root / "policy_scores.npy")
    args.policy_vs_research_out = args.policy_vs_research_out or str(scored_root / "policy_scores_vs_research.npy")
    args.policy_score_ids_out = args.policy_score_ids_out or str(scored_root / "metadata" / "policy_scores_ids.json")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    model_path = Path(args.model_path)
    policy_emb_path = Path(args.policy_emb)
    policy_ids_path = Path(args.policy_ids)
    research_centroids_path = Path(args.research_centroids)
    policy_scores_out = Path(args.policy_scores_out)
    policy_vs_research_out = Path(args.policy_vs_research_out)
    policy_score_ids_out = Path(args.policy_score_ids_out)

    if not args.overwrite and policy_scores_out.exists() and policy_vs_research_out.exists() and policy_score_ids_out.exists():
        log.info("Skip -- policy scores already exist at %s", policy_scores_out)
        return

    log.info("Loading policy embeddings: %s", policy_emb_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    input_dim = policy_emb.shape[1]

    log.info("Loading model: %s (classifier_type=lr, input_dim=%d)", model_path, input_dim)
    model = _load_model(model_root, "lr", input_dim)
    log.info("Model loaded (dims=%d)", input_dim)
    policy_ids = load_json(policy_ids_path)
    if policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            f"Policy embeddings / ID metadata mismatch: {policy_emb.shape[0]} vs {len(policy_ids)}"
        )
    verify_unit_norms(policy_emb, "policy embeddings")

    log.info("Loading research centroids: %s", research_centroids_path)
    research_centroids = np.load(research_centroids_path).astype(np.float32)
    if research_centroids.shape[0] != N_SDG:
        raise RuntimeError(f"Expected {N_SDG} research centroids, got {research_centroids.shape}")
    verify_unit_norms(research_centroids, "research centroids", n_sample=N_SDG)

    log.info("Indexing policy corpus metadata: %s", policy_ids_path)
    policy_doc_map = load_policy_doc_map(policy_ids_path)
    if len(policy_doc_map) == 0:
        raise RuntimeError(f"No policy corpus rows found in {policy_ids_path}")

    policy_score_ids = []
    missing_ids: list[str] = []
    text_mismatches: list[str] = []
    for row in policy_ids:
        segment_id = row["id"]
        joined = policy_doc_map.get(segment_id)
        if joined is None:
            missing_ids.append(segment_id)
            continue
        if row.get("text") and row["text"] != joined["text"]:
            text_mismatches.append(segment_id)
        policy_score_ids.append({"id": segment_id, "source_doc": joined["source_doc"]})

    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        raise RuntimeError(
            f"{len(missing_ids)} policy embedding IDs were not found in the active policy corpus. "
            f"Examples: {sample}"
        )

    if text_mismatches:
        sample = ", ".join(text_mismatches[:5])
        raise RuntimeError(
            f"{len(text_mismatches)} policy ID/text pairs do not match the active policy corpus. "
            f"Examples: {sample}"
        )

    log.info("Scoring %d policy segments with supervised LR", policy_emb.shape[0])
    policy_scores = model.predict_proba(policy_emb).astype(np.float32)

    log.info("Scoring %d policy segments against LR-based research centroids", policy_emb.shape[0])
    policy_vs_research = (policy_emb @ research_centroids.T).astype(np.float32)

    policy_scores_out.parent.mkdir(parents=True, exist_ok=True)
    with policy_scores_out.open("wb") as f:
        np.save(f, policy_scores)
        f.flush()
    log.info("Saved: %s  shape=%s", policy_scores_out, policy_scores.shape)

    policy_vs_research_out.parent.mkdir(parents=True, exist_ok=True)
    with policy_vs_research_out.open("wb") as f:
        np.save(f, policy_vs_research)
        f.flush()
    log.info("Saved: %s  shape=%s", policy_vs_research_out, policy_vs_research.shape)

    write_json(policy_score_ids_out, policy_score_ids)
    log.info("Saved: %s  n=%d", policy_score_ids_out, len(policy_score_ids))

    log.info(
        "Policy scoring complete. mean top SDG score=%.4f, mean top research score=%.4f",
        float(policy_scores.max(axis=1).mean()),
        float(policy_vs_research.max(axis=1).mean()),
    )


# ---------------------------------------------------------------------------
# MLP branch  (port of former 2_score_mlp.py: research + policy, corpus flag ignored)
# ---------------------------------------------------------------------------
def run_mlp(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    model_name = args.embed_model
    model_root = model_results_dir_for_model(model_name)
    embed_root = embed_dir_for_model(model_name)
    scored_root = scored_dir_for_model(model_name)

    out_dir = scored_root / "mlp_scores"
    summary_path = out_dir / "mlp_summary.json"
    centroids_out = out_dir / "mlp_research_centroids.npy"
    policy_scores_out = out_dir / "mlp_policy_scores.npy"
    pvr_out = out_dir / "mlp_policy_vs_research.npy"

    if summary_path.exists() and not args.overwrite:
        log.info("MLP scores already exist (use --overwrite to rebuild)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load MLP model --
    mlp_path = model_root / "model" / "mlp_retrained.joblib"
    log.info("Loading MLP from %s", mlp_path)
    model = joblib.load(mlp_path)
    _net = model
    while hasattr(_net, "net") and not isinstance(getattr(_net, "net", None), torch.nn.Sequential):
        _net = _net.net
    first_layer = _net.net[0]
    d = first_layer.in_features
    log.info("MLP loaded (type=%s, input_dim=%d)", type(model).__name__, d)

    # -- Score research shards --
    manifest_path = embed_root / "research_shards" / "metadata" / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    log.info("Scoring %d research shards with MLP...", len(manifest["shards"]))
    emb_dim = int(manifest.get("embedding_dim") or manifest["shards"][0]["dim"])
    d = emb_dim
    sums = np.zeros((N_SDG, d), dtype=np.float64)
    counts = np.zeros(N_SDG, dtype=np.int64)

    project_root = Path.cwd()
    for shard in manifest["shards"]:
        emb_rel = Path(shard["embedding_path"])
        emb_path = project_root / emb_rel if not emb_rel.is_absolute() else emb_rel

        emb = np.load(emb_path).astype(np.float32)
        log.info("  Scoring shard %s  shape=%s", shard["name"], emb.shape)
        scores = model.predict_proba(emb).astype(np.float32)
        assigned = scores.argmax(axis=1).astype(np.int64)

        for sdg_idx in range(N_SDG):
            mask = assigned == sdg_idx
            if not np.any(mask):
                continue
            counts[sdg_idx] += int(mask.sum())
            sums[sdg_idx] += emb[mask].sum(axis=0)

    # Research centroids
    mlp_research_centroids = np.zeros((N_SDG, d), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        if n == 0:
            continue
        raw = (sums[sdg_idx] / n).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm > ZERO_NORM_EPS:
            mlp_research_centroids[sdg_idx] = (raw / norm).astype(np.float32)

    with centroids_out.open("wb") as f:
        np.save(f, mlp_research_centroids)
    log.info("Research centroids -> %s", centroids_out)

    # Research coverage profile
    total = int(counts.sum())
    coverage = {int(sdg + 1): int(counts[sdg]) for sdg in range(N_SDG)}

    # -- Score policy corpus --
    log.info("Scoring policy corpus with MLP...")
    policy_emb = np.load(embed_root / "policy.npy").astype(np.float32)
    policy_scores = model.predict_proba(policy_emb).astype(np.float32)

    with policy_scores_out.open("wb") as f:
        np.save(f, policy_scores)
    log.info("Policy scores -> %s  shape=%s", policy_scores_out, policy_scores.shape)

    policy_vs_research = (policy_emb @ mlp_research_centroids.T).astype(np.float32)
    with pvr_out.open("wb") as f:
        np.save(f, policy_vs_research)
    log.info("Policy vs research centroids -> %s", pvr_out)

    # Policy coverage profile
    policy_assigned = policy_scores.argmax(axis=1)
    policy_counts = np.bincount(policy_assigned, minlength=N_SDG)
    policy_coverage = {int(sdg + 1): int(policy_counts[sdg]) for sdg in range(N_SDG)}

    # -- Semantic gaps --
    gap = np.zeros(N_SDG, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        mask = policy_assigned == sdg_idx
        if mask.sum() == 0:
            gap[sdg_idx] = np.nan
            continue
        pol_mean = policy_emb[mask].mean(axis=0)
        pol_norm = pol_mean / (np.linalg.norm(pol_mean) + NORM_EPS)
        gap[sdg_idx] = 1.0 - float(mlp_research_centroids[sdg_idx] @ pol_norm)

    gap_dict = {int(idx + 1): float(gap[idx]) for idx in range(N_SDG)}
    log.info("Semantic gaps computed. Range: %.4f-%.4f",
             min(gap_dict.values()), max(gap_dict.values()))

    # -- Save summary JSON --
    summary = {
        "model": "MLP",
        "research_coverage": coverage,
        "research_total": total,
        "policy_coverage": policy_coverage,
        "policy_total": int(policy_counts.sum()),
        "semantic_gaps": gap_dict,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary -> %s", summary_path)

    print(f"\n{'=' * 60}")
    print(f"  MLP scoring complete")
    print(f"  Research papers scored: {total}")
    print(f"  Policy segments scored: {int(policy_counts.sum())}")
    print(f"  Semantic gaps saved to: {summary_path}")
    print(f"{'=' * 60}")
    print(f"\n  Per-SDG semantic gaps:")
    for sdg_idx in range(1, N_SDG + 1):
        print(f"  SDG {sdg_idx:2d}: {gap_dict[sdg_idx]:.4f}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Score research/policy corpora with the supervised classifier (LR or MLP).")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--classifier", default="lr", choices=["lr", "mlp"],
                        help="Classifier family to score with (default: %(default)s)")
    parser.add_argument("--corpus", default="research", choices=["research", "policy"],
                        help="Corpus to score for the LR classifier (ignored for --classifier mlp, which scores both).")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--status-dir", default=None)
    parser.add_argument("--research-centroids-out", default=None)
    parser.add_argument("--research-meta-out", default=None)
    parser.add_argument("--policy-emb", default=None)
    parser.add_argument("--policy-ids", default=None)
    parser.add_argument("--research-centroids", default=None)
    parser.add_argument("--policy-scores-out", default=None)
    parser.add_argument("--policy-vs-research-out", default=None)
    parser.add_argument("--policy-score-ids-out", default=None)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--allow-partial-research-centroids", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rescore existing shards / outputs even if already complete.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if args.classifier == "mlp":
        run_mlp(args)
    else:
        if args.corpus == "research":
            run_research_lr(args)
        else:
            run_policy_lr(args)


if __name__ == "__main__":
    main()
