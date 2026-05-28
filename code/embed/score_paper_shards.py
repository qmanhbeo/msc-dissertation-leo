"""
Score paper embedding shards against SDG centroids.

Inputs:
  data/embeddings/papers_shards/manifest.json
  data/sdg_centroids.npy

Outputs:
  data/paper_scores_shards/part-00001.npy
  data/paper_scores_shards/part-00001_ids.jsonl
  data/paper_scores_shards/manifest.json
  data/paper_scores_shards/subset_index.sqlite
  data/research_centroids.npy
  data/research_centroid_meta.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from main_analysis.shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, sha256_file, update_stage_status


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-manifest", default="data/embeddings/papers_shards/manifest.json")
    p.add_argument("--centroids", default="data/sdg_centroids.npy")
    p.add_argument("--out-dir", default="data/paper_scores_shards")
    p.add_argument("--status-dir", default="artifacts/job_status")
    p.add_argument("--research-centroids-out", default="data/research_centroids.npy")
    p.add_argument("--research-meta-out", default="data/research_centroid_meta.json")
    p.add_argument("--limit-shards", type=int, default=0)
    return p.parse_args()


def open_index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subset_index (
            openalex_id TEXT PRIMARY KEY,
            publication_year INTEGER,
            assigned_sdg INTEGER,
            shard_id INTEGER,
            row_idx INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subset_year ON subset_index(publication_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subset_sdg ON subset_index(assigned_sdg)")
    conn.commit()
    return conn


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


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    emb_manifest_path = Path(args.embedding_manifest)
    centroids_path = Path(args.centroids)
    out_dir = Path(args.out_dir)
    status_dir = Path(args.status_dir)
    research_centroids_out = Path(args.research_centroids_out)
    research_meta_out = Path(args.research_meta_out)

    ensure_dir(out_dir)
    ensure_dir(status_dir)

    emb_manifest = read_json(emb_manifest_path)
    if not emb_manifest or "shards" not in emb_manifest:
        raise RuntimeError(f"Invalid embedding manifest: {emb_manifest_path}")

    centroids = np.load(centroids_path).astype(np.float32)
    if centroids.shape[0] != 17:
        raise RuntimeError(f"Expected 17 centroids, got shape {centroids.shape}")

    out_manifest_path = out_dir / "manifest.json"
    out_manifest = read_json(out_manifest_path, default=None)
    if out_manifest is None:
        out_manifest = {
            "stage": "score_paper_shards",
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "input_embedding_manifest": str(emb_manifest_path),
            "centroids_path": str(centroids_path),
            "shards": [],
            "totals": {"rows": 0, "shards": 0},
        }

    completed = {int(s["shard_id"]): s for s in out_manifest.get("shards", [])}
    shards = emb_manifest["shards"][: args.limit_shards] if args.limit_shards > 0 else emb_manifest["shards"]

    index_db = out_dir / "subset_index.sqlite"
    conn = open_index(index_db)

    update_stage_status(
        status_dir,
        "score_paper_shards",
        "running",
        {
            "embedding_manifest": str(emb_manifest_path),
            "centroids_path": str(centroids_path),
            "subset_index_db": str(index_db),
        },
    )

    d = int(centroids.shape[1])
    sums = np.zeros((17, d), dtype=np.float64)
    counts = np.zeros(17, dtype=np.int64)

    for shard in shards:
        shard_id = int(shard["shard_id"])
        shard_name = shard["name"]
        emb_path = Path(shard["embedding_path"])
        ids_in = Path(shard["ids_path"])
        score_path = out_dir / f"{shard_name}.npy"
        ids_out = out_dir / f"{shard_name}_ids.jsonl"

        emb = np.load(emb_path).astype(np.float32)
        ids_rows = load_ids(ids_in)
        if emb.shape[0] != len(ids_rows):
            raise RuntimeError(f"Row mismatch in shard {shard_name}: emb={emb.shape[0]} ids={len(ids_rows)}")

        if shard_id in completed and score_path.exists() and ids_out.exists():
            log.info("Skip scoring shard %s (already complete)", shard_name)
            scored_ids = load_ids(ids_out)
            assigned = np.array([int(r["assigned_sdg"]) - 1 for r in scored_ids], dtype=np.int64)
        else:
            log.info("Scoring shard %s", shard_name)
            scores = (emb @ centroids.T).astype(np.float32)
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

        payload = [
            (
                r["openalex_id"],
                int(r["publication_year"]) if r.get("publication_year") is not None else None,
                int(r["assigned_sdg"]),
                shard_id,
                int(r["row_in_shard"]),
            )
            for r in scored_ids
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO subset_index(
                openalex_id, publication_year, assigned_sdg, shard_id, row_idx
            ) VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()

        update_stage_status(
            status_dir,
            "score_paper_shards",
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

    conn.close()
    update_stage_status(
        status_dir,
        "score_paper_shards",
        "completed",
        {
            "manifest_path": str(out_manifest_path),
            "subset_index_db": str(index_db),
            "research_centroids": str(research_centroids_out),
            "research_meta": str(research_meta_out),
            "rows_done": int(counts.sum()),
        },
    )
    log.info("Scoring complete. rows=%d", int(counts.sum()))


if __name__ == "__main__":
    main()
