"""
Score paper embedding shards with the retrained single-label MLP model.

Inputs:
   2_data/3_embedded/{model}/research_shards/metadata/manifest.json
   2_data/4_supervised_model_results/{model}/model/sdg_classifier_retrained.joblib

Outputs:
   2_data/5_supervised_scored/{model}/paper_scores_shards/part-NNNNN.npy
   2_data/5_supervised_scored/{model}/paper_scores_shards/metadata/...
   2_data/5_supervised_scored/{model}/research_centroids.npy

Run from project root:
    python 1_code/5_supervised_model_infer/0_score_research_shards.py --model all-mpnet-base-v2
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

from model_utils import N_SDG, embed_dir_for_model, embed_research_dir_for_model, model_results_dir_for_model, scored_dir_for_model
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, resolve_manifest_path, sha256_file, update_stage_status

log = logging.getLogger(__name__)
STATUS_STAGE = "supervised_sdg_scores"


class _MultiLabelMLP(torch.nn.Module):
    """Must match the architecture used during training."""
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


class _ModelWrapper:
    def __init__(self, net):
        self.net = net
    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X.astype(np.float32)))
            probs = torch.sigmoid(logits)
        return probs.numpy()
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.float32)


def _load_model(model_root: Path, input_dim: int = 768):
    """Load retrained classifier — supports sklearn LR or PyTorch MLP."""
    model_path = model_root / "model" / "sdg_classifier_retrained.joblib"
    pt_path = model_path.with_suffix(".pt").parent / "sdg_classifier_retrained.pt"
    if pt_path.exists():
        net = _MultiLabelMLP(input_dim)
        net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        net.eval()
        return _ModelWrapper(net)
    log.info("No .pt found — loading sklearn classifier from %s", model_path)
    return joblib.load(model_path)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score paper shards with supervised MLP.")
    parser.add_argument("--model", default="all-mpnet-base-v2",
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--embedding-manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--status-dir", default=None)
    parser.add_argument("--research-centroids-out", default=None)
    parser.add_argument("--research-meta-out", default=None)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--allow-partial-research-centroids", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rescore existing shards even if already complete")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    scored_root = scored_dir_for_model(args.model)
    embed_root = embed_dir_for_model(args.model)
    model_root = model_results_dir_for_model(args.model)

    model_path_default = model_root / "model" / "sdg_classifier_retrained.joblib"
    embed_manifest_default = embed_research_dir_for_model(args.model) / "metadata" / "manifest.json"
    out_dir_default = scored_root / "paper_scores_shards"
    metadata_dir_default = out_dir_default / "metadata"
    status_dir_default = embed_research_dir_for_model(args.model) / "metadata"
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

    log.info("Loading model: %s", model_path)
    input_dim = 768  # MPNet embedding dimension (known a priori)
    model = _load_model(model_root, input_dim)
    log.info("Model loaded (dims=%d)", input_dim)

    emb_manifest = read_json(embed_manifest_path)
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

    d = int(emb_manifest["shards"][0].get("dims", 768))
    sums = np.zeros((17, d), dtype=np.float64)
    counts = np.zeros(17, dtype=np.int64)

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

        for sdg_idx in range(17):
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

    research_centroids = np.zeros((17, d), dtype=np.float32)
    meta: list[dict[str, Any]] = []
    for sdg_idx in range(17):
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
                "zero_flag": bool(norm < 1e-8),
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


if __name__ == "__main__":
    main()
