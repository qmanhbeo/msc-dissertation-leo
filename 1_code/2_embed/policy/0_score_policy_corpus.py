"""
Materialize active policy scoring artifacts used by downstream analysis.

Inputs:
  2_data/2_embedded/policy.npy
  2_data/2_embedded/metadata/policy_ids.json
  2_data/1_preprocessed/policy_all/policy_chunks_all.jsonl
  2_data/3_scored/sdg_centroids.npy
  2_data/3_scored/research_centroids.npy

Outputs:
  2_data/3_scored/policy_scores.npy
  2_data/3_scored/policy_scores_vs_research.npy
  2_data/3_scored/metadata/policy_scores_ids.json

Run from project root:
    python 1_code/2_embed/policy/0_score_policy_corpus.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "3_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from alignment_core import verify_unit_norms


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the active policy corpus against SDG and research centroids.")
    p.add_argument("--policy-emb", default="2_data/2_embedded/policy.npy")
    p.add_argument("--policy-ids", default="2_data/2_embedded/metadata/policy_ids.json")
    p.add_argument("--policy-corpus", default="2_data/1_preprocessed/policy_all/policy_chunks_all.jsonl")
    p.add_argument("--sdg-centroids", default="2_data/3_scored/sdg_centroids.npy")
    p.add_argument("--research-centroids", default="2_data/3_scored/research_centroids.npy")
    p.add_argument("--policy-scores-out", default="2_data/3_scored/policy_scores.npy")
    p.add_argument("--policy-vs-research-out", default="2_data/3_scored/policy_scores_vs_research.npy")
    p.add_argument("--policy-score-ids-out", default="2_data/3_scored/metadata/policy_scores_ids.json")
    return p.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_policy_doc_map(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path):
        mapping[row["chunk_id"]] = {
            "source_doc": row["source_doc"],
            "text": row["text"],
        }
    return mapping


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    policy_emb_path = Path(args.policy_emb)
    policy_ids_path = Path(args.policy_ids)
    policy_corpus_path = Path(args.policy_corpus)
    sdg_centroids_path = Path(args.sdg_centroids)
    research_centroids_path = Path(args.research_centroids)
    policy_scores_out = Path(args.policy_scores_out)
    policy_vs_research_out = Path(args.policy_vs_research_out)
    policy_score_ids_out = Path(args.policy_score_ids_out)

    log.info("Loading policy embeddings: %s", policy_emb_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    policy_ids = load_json(policy_ids_path)
    if policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            f"Policy embeddings / ID metadata mismatch: {policy_emb.shape[0]} vs {len(policy_ids)}"
        )
    verify_unit_norms(policy_emb, "policy embeddings")

    log.info("Loading SDG centroids: %s", sdg_centroids_path)
    sdg_centroids = np.load(sdg_centroids_path).astype(np.float32)
    if sdg_centroids.shape[0] != 17:
        raise RuntimeError(f"Expected 17 SDG centroids, got {sdg_centroids.shape}")
    verify_unit_norms(sdg_centroids, "sdg centroids", n_sample=17)

    log.info("Loading research centroids: %s", research_centroids_path)
    research_centroids = np.load(research_centroids_path).astype(np.float32)
    if research_centroids.shape[0] != 17:
        raise RuntimeError(f"Expected 17 research centroids, got {research_centroids.shape}")
    verify_unit_norms(research_centroids, "research centroids", n_sample=17)

    log.info("Indexing policy corpus metadata: %s", policy_corpus_path)
    policy_doc_map = load_policy_doc_map(policy_corpus_path)
    if len(policy_doc_map) == 0:
        raise RuntimeError(f"No policy corpus rows found in {policy_corpus_path}")

    policy_score_ids = []
    missing_ids: list[str] = []
    text_mismatches: list[str] = []
    for row in policy_ids:
        chunk_id = row["id"]
        joined = policy_doc_map.get(chunk_id)
        if joined is None:
            missing_ids.append(chunk_id)
            continue
        if row.get("text") and row["text"] != joined["text"]:
            text_mismatches.append(chunk_id)
        policy_score_ids.append({"id": chunk_id, "source_doc": joined["source_doc"]})

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

    log.info("Scoring %d policy chunks against SDG centroids", policy_emb.shape[0])
    policy_scores = (policy_emb @ sdg_centroids.T).astype(np.float32)
    log.info("Scoring %d policy chunks against research centroids", policy_emb.shape[0])
    policy_vs_research = (policy_emb @ research_centroids.T).astype(np.float32)

    policy_scores_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(policy_scores_out, policy_scores)
    log.info("Saved: %s  shape=%s", policy_scores_out, policy_scores.shape)

    policy_vs_research_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(policy_vs_research_out, policy_vs_research)
    log.info("Saved: %s  shape=%s", policy_vs_research_out, policy_vs_research.shape)

    write_json(policy_score_ids_out, policy_score_ids)
    log.info("Saved: %s  n=%d", policy_score_ids_out, len(policy_score_ids))

    log.info(
        "Policy scoring complete. mean top SDG score=%.4f, mean top research score=%.4f",
        float(policy_scores.max(axis=1).mean()),
        float(policy_vs_research.max(axis=1).mean()),
    )


if __name__ == "__main__":
    main()
