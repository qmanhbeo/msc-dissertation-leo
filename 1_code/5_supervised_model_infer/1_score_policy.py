"""
Score the active policy corpus with the retrained single-label MPNet MLP model.

Replaces centroid-based scoring with supervised sigmoid probabilities.

Inputs:
  2_data/2_embedded/policy.npy
  2_data/2_embedded/metadata/policy_ids.json
  2_data/1_preprocessed/policy_all/policy_segments_all.jsonl
  2_data/3c_scored_supervised/research_centroids.npy
  2_data/2b_supervised_singlelabel_mpnet/model/sdg_classifier_retrained.joblib

Outputs:
  2_data/3c_scored_supervised/policy_scores.npy
  2_data/3c_scored_supervised/policy_scores_vs_research.npy
  2_data/3c_scored_supervised/metadata/policy_scores_ids.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
CODE_ROOT = _THIS.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from alignment_core import verify_unit_norms

log = logging.getLogger(__name__)

PROJECT = _THIS.parents[2]
MODEL_PATH = PROJECT / "2_data/4_supervised_model_results/mpnet/model/sdg_classifier_retrained.joblib"
EMBED_DIR = PROJECT / "2_data/3_embedded/mpnet"
SCORED_DIR = PROJECT / "2_data/5_supervised_scored/mpnet"
POLICY_CORPUS = PROJECT / "2_data/1_preprocessed/policy_all/policy_segments_all.jsonl"

N_SDG = 17


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


def load_json(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=None, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(path)


def load_policy_doc_map(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["segment_id"]] = {
                "source_doc": row["source_doc"],
                "text": row["text"],
            }
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Score policy corpus with supervised MLP.")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--policy-emb", default=str(EMBED_DIR / "policy.npy"))
    parser.add_argument("--policy-ids", default=str(EMBED_DIR / "metadata" / "policy_ids.json"))
    parser.add_argument("--policy-corpus", default=str(POLICY_CORPUS))
    parser.add_argument("--research-centroids", default=str(SCORED_DIR / "research_centroids.npy"))
    parser.add_argument("--policy-scores-out", default=str(SCORED_DIR / "policy_scores.npy"))
    parser.add_argument("--policy-vs-research-out", default=str(SCORED_DIR / "policy_scores_vs_research.npy"))
    parser.add_argument("--policy-score-ids-out", default=str(SCORED_DIR / "metadata" / "policy_scores_ids.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    model_path = Path(args.model_path)
    policy_emb_path = Path(args.policy_emb)
    policy_ids_path = Path(args.policy_ids)
    policy_corpus_path = Path(args.policy_corpus)
    research_centroids_path = Path(args.research_centroids)
    policy_scores_out = Path(args.policy_scores_out)
    policy_vs_research_out = Path(args.policy_vs_research_out)
    policy_score_ids_out = Path(args.policy_score_ids_out)

    log.info("Loading model state dict: %s", model_path)
    input_dim = 768  # MPNet
    net = _MultiLabelMLP(input_dim)
    pt_path = model_path.with_suffix(".pt").parent / "sdg_classifier_retrained.pt"
    net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    net.eval()
    model = _ModelWrapper(net)
    log.info("Model loaded (dims=%d)", input_dim)

    log.info("Loading policy embeddings: %s", policy_emb_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    policy_ids = load_json(policy_ids_path)
    if policy_emb.shape[0] != len(policy_ids):
        raise RuntimeError(
            f"Policy embeddings / ID metadata mismatch: {policy_emb.shape[0]} vs {len(policy_ids)}"
        )
    verify_unit_norms(policy_emb, "policy embeddings")

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

    log.info("Scoring %d policy segments with supervised MLP", policy_emb.shape[0])
    policy_scores = model.predict_proba(policy_emb).astype(np.float32)

    log.info("Scoring %d policy segments against MLP-based research centroids", policy_emb.shape[0])
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
