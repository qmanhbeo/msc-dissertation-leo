"""
Score research + policy corpora with the retrained MLP model.
Produces separate MLP output files to avoid overwriting LR scores.
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
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, ZERO_NORM_EPS, NORM_EPS, embed_dir_for_model, model_results_dir_for_model, scored_dir_for_model

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Replicate classes needed to unpickle the retrained MLP
# ---------------------------------------------------------------------------
class MultiLabelMLP(nn.Module):
    """Must match the exact class name used during pickle saving."""
    def __init__(self, input_dim: int, n_layers: int = 4, hidden_size: int = 384,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, N_SDG))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _NetWrapper:
    """Must match the exact class name used during pickle saving."""
    def __init__(self, net, input_dim):
        self.net = net
        self.input_dim = input_dim
    def predict_proba(self, X):
        self.net.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(self.net(X_t))
        return probs.cpu().numpy()
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Score research + policy corpora with MLP.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help=f"Embed model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing MLP score outputs")
    args = parser.parse_args()

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

    # ── Load MLP model ────────────────────────────────────────────────
    mlp_path = model_root / "model" / "mlp_retrained.joblib"
    log.info("Loading MLP from %s", mlp_path)
    model = joblib.load(mlp_path)
    first_layer = model.net[0]
    assert first_layer.in_features == d, (
        f"MLP first layer in_features {first_layer.in_features} != embedding dim {d}"
    )
    log.info("MLP loaded (type=%s, input_dim=%d)", type(model).__name__, d)

    # ── Score research shards ─────────────────────────────────────────
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
    research_centroids = np.zeros((N_SDG, d), dtype=np.float32)
    for sdg_idx in range(N_SDG):
        n = int(counts[sdg_idx])
        if n == 0:
            continue
        raw = (sums[sdg_idx] / n).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm > ZERO_NORM_EPS:
            research_centroids[sdg_idx] = (raw / norm).astype(np.float32)

    with centroids_out.open("wb") as f:
        np.save(f, research_centroids)
    log.info("Research centroids -> %s", centroids_out)

    # Research coverage profile
    total = int(counts.sum())
    coverage = {int(sdg + 1): int(counts[sdg]) for sdg in range(N_SDG)}

    # ── Score policy corpus ────────────────────────────────────────────
    log.info("Scoring policy corpus with MLP...")
    policy_emb = np.load(embed_root / "policy.npy").astype(np.float32)
    policy_scores = model.predict_proba(policy_emb).astype(np.float32)

    with policy_scores_out.open("wb") as f:
        np.save(f, policy_scores)
    log.info("Policy scores -> %s  shape=%s", policy_scores_out, policy_scores.shape)

    policy_vs_research = (policy_emb @ research_centroids.T).astype(np.float32)
    with pvr_out.open("wb") as f:
        np.save(f, policy_vs_research)
    log.info("Policy vs research centroids -> %s", pvr_out)

    # Policy coverage profile
    policy_assigned = policy_scores.argmax(axis=1)
    policy_counts = np.bincount(policy_assigned, minlength=N_SDG)
    policy_coverage = {int(sdg + 1): int(policy_counts[sdg]) for sdg in range(N_SDG)}

    # ── Semantic gaps ──────────────────────────────────────────────────
    gap = np.zeros(17, dtype=np.float32)
    for sdg_idx in range(N_SDG):
        mask = policy_assigned == sdg_idx
        if mask.sum() == 0:
            gap[sdg_idx] = np.nan
            continue
        pol_mean = policy_emb[mask].mean(axis=0)
        pol_norm = pol_mean / (np.linalg.norm(pol_mean) + NORM_EPS)
        gap[sdg_idx] = 1.0 - float(research_centroids[sdg_idx] @ pol_norm)

    gap_dict = {int(idx + 1): float(gap[idx]) for idx in range(N_SDG)}
    log.info("Semantic gaps computed. Range: %.4f-%.4f",
             min(gap_dict.values()), max(gap_dict.values()))

    # ── Save summary JSON ──────────────────────────────────────────────
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

    print(f"\n{'='*60}")
    print(f"  MLP scoring complete")
    print(f"  Research papers scored: {total}")
    print(f"  Policy segments scored: {int(policy_counts.sum())}")
    print(f"  Semantic gaps saved to: {summary_path}")
    print(f"{'='*60}")
    print(f"\n  Per-SDG semantic gaps:")
    for sdg_idx in range(1, N_SDG + 1):
        print(f"  SDG {sdg_idx:2d}: {gap_dict[sdg_idx]:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
