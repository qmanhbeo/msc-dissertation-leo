"""
Score the active policy corpus with the retrained supervised classifier (LR or MLP).

Inputs:
    2_data/3_embedded/{model}/policy.npy
    2_data/3_embedded/{model}/metadata/policy_ids.json
    2_data/5_supervised_scored/{model}/research_centroids.npy
    2_data/4_supervised_model_results/{model}/model/sdg_classifier_retrained.joblib  (--classifier-type lr)
    2_data/4_supervised_model_results/{model}/model/mlp_retrained.joblib             (--classifier-type mlp)

Outputs:
    2_data/5_supervised_scored/{model}/policy_scores.npy
    2_data/5_supervised_scored/{model}/policy_scores_vs_research.npy
    2_data/5_supervised_scored/{model}/metadata/policy_scores_ids.json

Run from project root:
    python 1_code/5_supervised_model_infer/1_score_policy.py --embed-model all-mpnet-base-v2 [--classifier-type lr|mlp]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
from model_utils import N_SDG, embed_dir_for_model, model_results_dir_for_model, scored_dir_for_model, DEFAULT_EMBED_MODEL
from shard_pipeline_utils import load_json

log = logging.getLogger(__name__)


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
    the embedding dimension of the policy embeddings — this is the failure
    mode that previously crashed MiniLM (384-dim) scoring with a shape error.
    """
    if classifier_type == "mlp":
        model_path = model_root / "model" / "mlp_retrained.joblib"
        model = joblib.load(model_path)
        first_layer = model.net[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score policy corpus with the supervised classifier (LR or MLP).")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--classifier-type", default="lr", choices=["lr", "mlp"],
                        help="Classifier family to score with (default: %(default)s)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--policy-emb", default=None)
    parser.add_argument("--policy-ids", default=None)
    parser.add_argument("--research-centroids", default=None)
    parser.add_argument("--policy-scores-out", default=None)
    parser.add_argument("--policy-vs-research-out", default=None)
    parser.add_argument("--policy-score-ids-out", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute policy scores even if outputs exist")
    args = parser.parse_args()

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
        log.info("Skip — policy scores already exist at %s", policy_scores_out)
        return

    log.info("Loading policy embeddings: %s", policy_emb_path)
    policy_emb = np.load(policy_emb_path).astype(np.float32)
    input_dim = policy_emb.shape[1]

    log.info("Loading model: %s (classifier_type=%s, input_dim=%d)", model_path, args.classifier_type, input_dim)
    model = _load_model(model_root, args.classifier_type, input_dim)
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

    log.info("Scoring %d policy segments with supervised MLP", policy_emb.shape[0])
    policy_scores = model.predict_proba(policy_emb).astype(np.float32)

    log.info("Scoring %d policy segments against MLP-based research centroids", policy_emb.shape[0])
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


if __name__ == "__main__":
    main()
