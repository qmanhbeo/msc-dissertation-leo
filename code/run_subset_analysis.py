"""
Run fast subset analysis from cached paper-score shards (no rebuild).

This script reuses:
  - data/paper_scores_shards/*.npy
  - data/paper_scores_shards/subset_index.sqlite
  - existing policy embeddings/scores

It materializes a subset and recomputes coverage/semantic/H25/H26 outputs under:
  data/subsets/<subset_name>/
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alignment_score import build_research_centroids
from coverage_gap import compute_coverage_gap, document_weighted_policy_profile, hard_assignment_profile, mean_score_profile
from coverage_semantic_interaction import pearson_and_spearman
from semantic_gap import compute_sdg_semantic_gaps
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset-name", required=True)
    p.add_argument("--scores-manifest", default="data/paper_scores_shards/manifest.json")
    p.add_argument("--index-db", default="data/paper_scores_shards/subset_index.sqlite")
    p.add_argument("--embedding-manifest", default="data/embeddings/papers_shards/manifest.json")
    p.add_argument("--policy-scores", default="data/policy_scores.npy")
    p.add_argument("--policy-ids", default="data/policy_scores_ids.json")
    p.add_argument("--policy-embeddings", default="data/embeddings/policy.npy")
    p.add_argument("--id-list", default="", help="Optional text file with one openalex_id per line.")
    p.add_argument("--year-from", type=int, default=0)
    p.add_argument("--year-to", type=int, default=0)
    p.add_argument("--assigned-sdgs", default="", help="Comma list like 3,9,13")
    p.add_argument("--chunk-cap", type=int, default=50)
    p.add_argument("--skip-semantic", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_sdgs(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_id_list(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                out.add(v)
    return out


def fetch_subset_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    conn = sqlite3.connect(args.index_db)
    conn.row_factory = sqlite3.Row

    id_list_values: set[str] = set()
    if args.id_list:
        id_list_values = load_id_list(Path(args.id_list))
        conn.execute("DROP TABLE IF EXISTS requested_ids")
        conn.execute("CREATE TEMP TABLE requested_ids(openalex_id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT OR IGNORE INTO requested_ids(openalex_id) VALUES (?)",
            [(v,) for v in id_list_values],
        )
        conn.commit()

    where = []
    params: list[Any] = []
    if args.year_from:
        where.append("publication_year >= ?")
        params.append(args.year_from)
    if args.year_to:
        where.append("publication_year <= ?")
        params.append(args.year_to)

    sdgs = parse_sdgs(args.assigned_sdgs)
    if sdgs:
        ph = ",".join("?" for _ in sdgs)
        where.append(f"assigned_sdg IN ({ph})")
        params.extend(sdgs)

    sql = "SELECT s.openalex_id, s.publication_year, s.assigned_sdg, s.shard_id, s.row_idx FROM subset_index s"
    if id_list_values:
        sql += " INNER JOIN requested_ids r ON s.openalex_id = r.openalex_id"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY shard_id, row_idx"
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    conn.close()
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    subset_dir = Path("data/subsets") / args.subset_name
    ensure_dir(subset_dir)

    score_manifest = load_json(Path(args.scores_manifest))
    score_map = {int(s["shard_id"]): s for s in score_manifest["shards"]}
    emb_manifest = load_json(Path(args.embedding_manifest))
    emb_map = {int(s["shard_id"]): s for s in emb_manifest["shards"]}

    selected = fetch_subset_rows(args)
    if not selected:
        raise RuntimeError("Subset selection is empty. Relax filters.")
    log.info("Selected rows: %d", len(selected))

    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_shard[int(row["shard_id"])].append(row)

    score_blocks: list[np.ndarray] = []
    emb_blocks: list[np.ndarray] = []
    ids_rows: list[dict[str, Any]] = []

    for shard_id in sorted(by_shard):
        rows = by_shard[shard_id]
        idxs = np.array([int(r["row_idx"]) for r in rows], dtype=np.int64)

        score_path = Path(score_map[shard_id]["score_path"])
        shard_scores = np.load(score_path, mmap_mode="r")
        score_blocks.append(np.asarray(shard_scores[idxs], dtype=np.float32))

        emb_path = Path(emb_map[shard_id]["embedding_path"])
        shard_emb = np.load(emb_path, mmap_mode="r")
        emb_blocks.append(np.asarray(shard_emb[idxs], dtype=np.float32))

        ids_rows.extend(
            {
                "id": r["openalex_id"],
                "publication_year": r["publication_year"],
                "assigned_sdg_index": int(r["assigned_sdg"]) - 1,
            }
            for r in rows
        )

    paper_scores = np.vstack(score_blocks).astype(np.float32)
    paper_emb = np.vstack(emb_blocks).astype(np.float32)

    with (subset_dir / "paper_scores.npy").open("wb") as f:
        np.save(f, paper_scores)
    with (subset_dir / "papers.npy").open("wb") as f:
        np.save(f, paper_emb)
    atomic_write_json(subset_dir / "paper_scores_ids.json", ids_rows)

    # Research centroids from subset
    research_centroids, research_meta = build_research_centroids(paper_emb, paper_scores, n_sdg=17)
    with (subset_dir / "research_centroids.npy").open("wb") as f:
        np.save(f, research_centroids)
    atomic_write_json(subset_dir / "research_centroid_meta.json", research_meta)

    # Coverage
    policy_scores = np.load(Path(args.policy_scores))
    policy_ids = load_json(Path(args.policy_ids))
    res_hard = hard_assignment_profile(paper_scores)
    res_soft = mean_score_profile(paper_scores)
    pol_raw_hard = hard_assignment_profile(policy_scores)
    pol_raw_soft = mean_score_profile(policy_scores)
    pol_dw_hard, pol_dw_soft, doc_meta = document_weighted_policy_profile(policy_scores, policy_ids)
    gap = compute_coverage_gap(res_hard, pol_dw_hard)

    coverage_payload = {
        "subset_name": args.subset_name,
        "n_research_papers": int(paper_scores.shape[0]),
        "n_policy_chunks": int(policy_scores.shape[0]),
        "research_profile_hard": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(res_hard)},
        "research_profile_soft": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(res_soft)},
        "policy_profile_hard_raw": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(pol_raw_hard)},
        "policy_profile_soft_raw": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(pol_raw_soft)},
        "policy_profile_hard_docweighted": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(pol_dw_hard)},
        "policy_profile_soft_docweighted": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(pol_dw_soft)},
        "coverage_gap_hard": {f"SDG{i+1}": round(float(v), 6) for i, v in enumerate(gap)},
        "policy_doc_assignments": doc_meta,
    }
    atomic_write_json(subset_dir / "coverage_gap.json", coverage_payload)

    semantic_payload = None
    h25_payload = None
    if not args.skip_semantic:
        policy_emb = np.load(Path(args.policy_embeddings)).astype(np.float32)
        paper_assign = paper_scores.argmax(axis=1)
        policy_assign = policy_scores.argmax(axis=1)
        rng = np.random.default_rng(42)
        per_sdg = compute_sdg_semantic_gaps(
            paper_emb=paper_emb,
            policy_emb=policy_emb,
            paper_assignments=paper_assign,
            policy_assignments=policy_assign,
            policy_ids=policy_ids,
            chunk_cap=args.chunk_cap,
            rng=rng,
        )
        semantic_payload = {
            "subset_name": args.subset_name,
            "chunk_cap": args.chunk_cap,
            "per_sdg": per_sdg,
        }
        atomic_write_json(subset_dir / "semantic_gap.json", semantic_payload)

        # H25 + H26
        sem_gap = np.array(
            [np.nan if r["semantic_gap"] is None else float(r["semantic_gap"]) for r in per_sdg],
            dtype=float,
        )
        sem_sim = np.array(
            [np.nan if r["semantic_similarity"] is None else float(r["semantic_similarity"]) for r in per_sdg],
            dtype=float,
        )
        res_prop = np.array([coverage_payload["research_profile_hard"][f"SDG{i}"] for i in range(1, 18)], dtype=float)
        pol_prop = np.array([coverage_payload["policy_profile_hard_docweighted"][f"SDG{i}"] for i in range(1, 18)], dtype=float)
        cov_abs = np.array([coverage_payload["coverage_gap_hard"][f"SDG{i}"] for i in range(1, 18)], dtype=float)
        dominance = res_prop - pol_prop

        valid_mask = ~np.isnan(sem_gap)
        if valid_mask.sum() < 3:
            corr_all = {
                "message": "Insufficient valid SDGs for correlation (need >=3 with numeric semantic_gap).",
                "n_valid": int(valid_mask.sum()),
            }
        else:
            corr_all = {
                "a_res_prop_vs_sem_gap": pearson_and_spearman(res_prop[valid_mask], sem_gap[valid_mask], "subset(a)"),
                "b_cov_gap_abs_vs_sem_gap": pearson_and_spearman(cov_abs[valid_mask], sem_gap[valid_mask], "subset(b)"),
                "c_res_dominance_vs_sem_gap": pearson_and_spearman(
                    dominance[valid_mask], sem_gap[valid_mask], "subset(c)"
                ),
            }

        paper_top = paper_scores.max(axis=1)
        policy_vs_res = (policy_emb @ research_centroids.T).astype(np.float32)
        policy_top = policy_vs_res.max(axis=1)
        h26 = {
            "paper_top_mean": round(float(paper_top.mean()), 6),
            "policy_vs_research_top_mean": round(float(policy_top.mean()), 6),
            "difference_policy_minus_paper": round(float(policy_top.mean() - paper_top.mean()), 6),
        }

        h25_payload = {
            "subset_name": args.subset_name,
            "n_sdg": 17,
            "correlations_all_sdgs": corr_all,
            "h26_asymmetry": h26,
            "per_sdg_rows": [
                {
                    "sdg": i + 1,
                    "research_prop": float(res_prop[i]),
                    "policy_prop_docweighted": float(pol_prop[i]),
                    "coverage_gap_abs": float(cov_abs[i]),
                    "semantic_gap": float(sem_gap[i]),
                    "semantic_similarity": float(sem_sim[i]),
                }
                for i in range(17)
            ],
        }
        atomic_write_json(subset_dir / "h25_correlation.json", h25_payload)
        with (subset_dir / "policy_scores_vs_research.npy").open("wb") as f:
            np.save(f, policy_vs_res)

    atomic_write_json(
        subset_dir / "subset_manifest.json",
        {
            "subset_name": args.subset_name,
            "created_at_utc": now_iso(),
            "n_papers": int(paper_scores.shape[0]),
            "filters": {
                "year_from": args.year_from,
                "year_to": args.year_to,
                "assigned_sdgs": parse_sdgs(args.assigned_sdgs),
                "id_list": args.id_list or None,
                "limit": args.limit,
            },
            "artifacts": {
                "paper_scores": str(subset_dir / "paper_scores.npy"),
                "paper_scores_ids": str(subset_dir / "paper_scores_ids.json"),
                "papers_embeddings": str(subset_dir / "papers.npy"),
                "research_centroids": str(subset_dir / "research_centroids.npy"),
                "research_meta": str(subset_dir / "research_centroid_meta.json"),
                "coverage_gap": str(subset_dir / "coverage_gap.json"),
                "semantic_gap": str(subset_dir / "semantic_gap.json") if semantic_payload else None,
                "h25_correlation": str(subset_dir / "h25_correlation.json") if h25_payload else None,
            },
        },
    )
    log.info("Subset analysis complete: %s", subset_dir)


if __name__ == "__main__":
    main()
