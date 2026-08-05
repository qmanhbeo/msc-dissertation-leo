"""
Appendix I.1: Supervised vs nearest-centroid assignment comparison.

Zero-shot nearest-centroid is scoped to ONE comparison (AGENTS.md
"Manuscript scope decisions"): canonical supervised assignment (LR, with MLP
as robustness) vs zero-shot nearest-centroid on the SDG reference centroids.
This script dives into that comparison under the canonical MPNet encoder and
keyword retrieval only:

   - Research corpus: per-segment LR-vs-ZS assignment agreement, per-SDG
     agreement rates, and a 17x17 confusion matrix (row = LR assignment,
     column = ZS assignment). MLP research per-shard scores are now persisted
     (mlp_scores/mlp_research_scores_shards), so research MLP-vs-ZS agreement
     is reported directly (no longer proxied by the LR-vs-MLP gap-rank
     correlation).
  - Policy corpus: segment-level AND document-level LR-vs-ZS agreement, plus
    segment/document-level MLP-vs-ZS overall agreement.
  - Per-SDG |gap-rank delta| between the LR and ZS semantic-gap rankings,
    anchoring method-sensitive SDGs (e.g. SDG 17).

Assignment rules:
  LR:   argmax of paper_scores_shards/*.npy (research) / policy_scores.npy (policy)
  MLP:  mlp_scores/mlp_policy_scores.npy (policy only)
  ZS:   argmax(embeddings @ sdg_centroids.T)  (sdg_centroids = train-split
        reference centroids, same target score_zeroshot.py uses)

Inputs (canonical MPNet artifacts):
  2_data/3_embedded/{model}/research_shards/metadata/manifest.json
  2_data/3_embedded/{model}/policy.npy
  2_data/3_embedded/{model}/metadata/policy_ids.json
  2_data/5_supervised_scored/{model}/sdg_centroids.npy
  2_data/5_supervised_scored/{model}/paper_scores_shards/part-NNNN.npy
  2_data/5_supervised_scored/{model}/policy_scores.npy
  2_data/5_supervised_scored/{model}/mlp_scores/mlp_policy_scores.npy
  4_outputs/{model}/data/semantic_gap_distances_lr.json
  4_outputs/{model}/data/semantic_gap_distances_zeroshot.json

Outputs:
  4_outputs/appendix/{model}/i1_assignment_method_comparison/data/assignment_method_comparison.json
  4_outputs/appendix/{model}/i1_assignment_method_comparison/tables/tab_app_assignment_method_comparison.tex

Run from project root:
  python 1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, embed_dir_for_model, model_slug, output_dir_for_model, preprocessed_dir, scored_dir_for_model, resolve_model_alias
from shared_utils import fingerprint_of, should_skip, record_fingerprint
from shard_pipeline_utils import load_json, resolve_manifest_path
from semantic_gap_shared import doc_level_assignments

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SDG_NAMES = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health", 4: "Quality Education",
    5: "Gender Equality", 6: "Clean Water", 7: "Clean Energy",
    8: "Decent Work", 9: "Industry \\& Infra.", 10: "Reduced Inequalities",
    11: "Sustainable Cities", 12: "Responsible Cons.", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land", 16: "Peace \\& Justice",
    17: "Partnerships",
}


def agreement_stats(assign_a: np.ndarray, assign_b: np.ndarray) -> dict:
    """Per-SDG and overall agreement between two hard-assignment vectors.

    Returns per-SDG: n_a (assigned by A), n_b (assigned by B), n_agree
    (assigned to the same SDG by both), and both conditional agreement rates.
    """
    n_rows = int(len(assign_a))
    per_sdg = []
    for sdg_idx in range(N_SDG):
        mask_a = assign_a == sdg_idx
        mask_b = assign_b == sdg_idx
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        n_agree = int((mask_a & mask_b).sum())
        per_sdg.append({
            "sdg": sdg_idx + 1,
            "n_assigned_a": n_a,
            "n_assigned_b": n_b,
            "n_agree": n_agree,
            "agree_rate_a": round(n_agree / n_a, 6) if n_a else None,
            "agree_rate_b": round(n_agree / n_b, 6) if n_b else None,
        })
    n_agree_total = int((assign_a == assign_b).sum())
    return {
        "n_rows": n_rows,
        "n_agree": n_agree_total,
        "overall_agreement": round(n_agree_total / n_rows, 6) if n_rows else None,
        "per_sdg": per_sdg,
    }


def confusion(assign_a: np.ndarray, assign_b: np.ndarray) -> list[list[int]]:
    """17x17 confusion: row = A assignment, column = B assignment."""
    out = np.zeros((N_SDG, N_SDG), dtype=np.int64)
    for i in range(N_SDG):
        mask = assign_a == i
        if mask.any():
            out[i] = np.bincount(assign_b[mask], minlength=N_SDG)
    return out.tolist()


def doc_level_assignment(score_matrix: np.ndarray, policy_ids: list[dict]) -> np.ndarray:
    """Document-level assignment: mean score per source_doc, then argmax.

    Delegates to the single source of truth in semantic_gap_shared.
    """
    return doc_level_assignments(score_matrix, policy_ids)


def gap_ranks(path: Path) -> dict[int, int]:
    """Per-SDG semantic-gap ranks from a semantic_gap_distances_zeroshot.json (1 = largest)."""
    with open(path) as f:
        data = json.load(f)
    items = [(r["sdg"], r["semantic_gap"]) for r in data["per_sdg"] if r.get("semantic_gap") is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    return {sdg: rank + 1 for rank, (sdg, _) in enumerate(items)}


def check_research(model: str, manifest_path: Path, centroids: np.ndarray, scores_dir: Path) -> dict:
    """Per-shard LR-vs-ZS agreement over the research corpus."""
    manifest = load_json(manifest_path)
    shards = sorted(manifest["shards"], key=lambda x: int(x["shard_id"]))
    log.info("Research: scoring %d shards (LR vs zero-shot)...", len(shards))
    embed_dir = embed_dir_for_model(model)

    total_n = np.zeros(N_SDG, dtype=np.int64)
    total_matched = np.zeros(N_SDG, dtype=np.int64)
    total_confusion = np.zeros((N_SDG, N_SDG), dtype=np.int64)
    n_rows = 0
    n_agree = 0

    for shard in shards:
        shard_name = shard["name"]
        emb_path = resolve_manifest_path(
            shard["embedding_path"],
            allowed_dirs=(embed_dir, scored_dir_for_model(model), preprocessed_dir()),
        )
        score_path = scores_dir / f"{shard_name}.npy"
        if not score_path.exists():
            raise FileNotFoundError(f"LR shard scores missing: {score_path}")
        emb = np.load(emb_path).astype(np.float32)
        lr_scores = np.load(score_path).astype(np.float32)
        if emb.shape[0] != lr_scores.shape[0]:
            raise RuntimeError(f"Shard {shard_name}: emb rows {emb.shape[0]} != score rows {lr_scores.shape[0]}")

        lr_assigned = lr_scores.argmax(axis=1)
        zs_assigned = (emb @ centroids.T).argmax(axis=1)

        for sdg_idx in range(N_SDG):
            mask = lr_assigned == sdg_idx
            n = int(mask.sum())
            total_n[sdg_idx] += n
            if n > 0:
                total_matched[sdg_idx] += int((zs_assigned[mask] == sdg_idx).sum())
                total_confusion[sdg_idx] += np.bincount(zs_assigned[mask], minlength=N_SDG)

        n_rows += emb.shape[0]
        n_agree += int((lr_assigned == zs_assigned).sum())
        del emb, lr_scores, lr_assigned, zs_assigned

    per_sdg = []
    for sdg_idx in range(N_SDG):
        n = int(total_n[sdg_idx])
        matched = int(total_matched[sdg_idx])
        per_sdg.append({
            "sdg": sdg_idx + 1,
            "n_assigned_lr": n,
            "n_agree": matched,
            "agree_rate_lr": round(matched / n, 6) if n else None,
        })

    return {
        "n_rows": n_rows,
        "n_agree": n_agree,
        "overall_agreement": round(n_agree / n_rows, 6) if n_rows else None,
        "per_sdg": per_sdg,
        "confusion_matrix": total_confusion.tolist(),
    }


def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    if model != DEFAULT_EMBED_MODEL:
        log.warning("Appendix I.1 is scoped to the canonical encoder (%s); got %s. Proceeding anyway.", DEFAULT_EMBED_MODEL, model)
    root = Path(args.output_dir)

    out_root = root / "appendix" / model_slug(model) / "i1_assignment_method_comparison"
    data_out = out_root / "data" / "assignment_method_comparison.json"
    table_out = out_root / "tables" / "tab_app_assignment_method_comparison.tex"

    embed_root = embed_dir_for_model(model)
    scored_root = scored_dir_for_model(model)
    data_root = output_dir_for_model(model, root=root) / "data"

    manifest_path = embed_root / "research_shards" / "metadata" / "manifest.json"
    lr_gap_path = data_root / "semantic_gap_distances_lr.json"
    zs_gap_path = data_root / "semantic_gap_distances_zeroshot.json"

    fp_paths = [
        manifest_path,
        scored_root / "sdg_centroids.npy",
        scored_root / "paper_scores_shards" / "metadata" / "manifest.json",
        embed_root / "policy.npy",
        embed_root / "metadata" / "policy_ids.json",
        scored_root / "policy_scores.npy",
        scored_root / "mlp_scores" / "mlp_policy_scores.npy",
        lr_gap_path,
        zs_gap_path,
    ]
    # E.12: MLP per-shard research scores are now persisted; re-derive when they change.
    mlp_research_dir = scored_root / "mlp_scores" / "mlp_research_scores_shards"
    if mlp_research_dir.exists():
        fp_paths += sorted(mlp_research_dir.glob("*.npy"))
    fp = fingerprint_of(*fp_paths) + "1"
    if should_skip([data_out, table_out], fp, args.overwrite, table_out):
        print(f"Skipping {table_out} \u2014 inputs unchanged")
        return

    centroids_path = scored_root / "sdg_centroids.npy"
    log.info("Loading reference centroids: %s", centroids_path)
    centroids = np.load(centroids_path).astype(np.float32)
    assert centroids.shape[0] == N_SDG
    norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Centroids not unit: {norms}"

    # --- Research: LR vs zero-shot ---
    research = check_research(model, manifest_path, centroids, scored_root / "paper_scores_shards")
    log.info("Research overall LR-vs-ZS agreement: %.4f (n=%d)", research["overall_agreement"], research["n_rows"])

    # --- Research: MLP vs zero-shot (E.12: now persisted) ---
    research_mlp = None
    if mlp_research_dir.exists():
        log.info("Research: MLP vs zero-shot...")
        research_mlp = check_research(model, manifest_path, centroids, mlp_research_dir)
        log.info("Research overall MLP-vs-ZS agreement: %.4f (n=%d)", research_mlp["overall_agreement"], research_mlp["n_rows"])

    # --- Policy: LR vs zero-shot (segment- and document-level) ---
    log.info("Policy: LR vs zero-shot...")
    policy_emb = np.load(embed_root / "policy.npy").astype(np.float32)
    policy_ids = load_json(embed_root / "metadata" / "policy_ids.json")
    lr_policy_scores = np.load(scored_root / "policy_scores.npy").astype(np.float32)
    assert policy_emb.shape[0] == lr_policy_scores.shape[0] == len(policy_ids)

    zs_policy_scores = policy_emb @ centroids.T
    lr_pol = lr_policy_scores.argmax(axis=1)
    zs_pol = zs_policy_scores.argmax(axis=1)

    policy_segment = agreement_stats(lr_pol, zs_pol)
    lr_pol_doc = doc_level_assignment(lr_policy_scores, policy_ids)
    zs_pol_doc = doc_level_assignment(zs_policy_scores, policy_ids)
    policy_doc = agreement_stats(lr_pol_doc, zs_pol_doc)
    log.info("Policy segment overall agreement: %.4f (n=%d)", policy_segment["overall_agreement"], policy_segment["n_rows"])
    log.info("Policy doc    overall agreement: %.4f (n=%d)", policy_doc["overall_agreement"], policy_doc["n_rows"])

    # --- Policy: MLP vs zero-shot (robustness; segment- and document-level) ---
    mlp_path = scored_root / "mlp_scores" / "mlp_policy_scores.npy"
    mlp_policy = None
    if mlp_path.exists():
        log.info("Policy: MLP vs zero-shot...")
        mlp_policy_scores = np.load(mlp_path).astype(np.float32)
        mlp_pol = mlp_policy_scores.argmax(axis=1)
        mlp_segment = agreement_stats(mlp_pol, zs_pol)
        mlp_pol_doc = doc_level_assignment(mlp_policy_scores, policy_ids)
        mlp_doc = agreement_stats(mlp_pol_doc, zs_pol_doc)
        mlp_policy = {
            "segment": {"overall_agreement": mlp_segment["overall_agreement"], "n_rows": mlp_segment["n_rows"]},
            "document": {"overall_agreement": mlp_doc["overall_agreement"], "n_rows": mlp_doc["n_rows"]},
        }
        log.info("Policy MLP-vs-ZS segment overall: %.4f", mlp_segment["overall_agreement"])

    # --- Semantic-gap rank deltas (LR vs ZS) ---
    ranks_lr = gap_ranks(lr_gap_path)
    ranks_zs = gap_ranks(zs_gap_path)
    rank_deltas = {sdg: abs(ranks_lr[sdg] - ranks_zs[sdg]) for sdg in ranks_lr if sdg in ranks_zs}

    per_sdg_rows = []
    res_by_sdg = {r["sdg"]: r for r in research["per_sdg"]}
    res_mlp_by_sdg = {r["sdg"]: r for r in research_mlp["per_sdg"]} if research_mlp else {}
    pol_by_sdg = {r["sdg"]: r for r in policy_segment["per_sdg"]}
    for sdg_idx in range(N_SDG):
        sdg = sdg_idx + 1
        r = res_by_sdg[sdg]
        rm = res_mlp_by_sdg.get(sdg)
        p = pol_by_sdg[sdg]
        per_sdg_rows.append({
            "sdg": sdg,
            "research_n_lr": r["n_assigned_lr"],
            "research_agree_rate": r["agree_rate_lr"],
            "research_mlp_agree_rate": rm["agree_rate_lr"] if rm else None,
            "policy_n_lr": p["n_assigned_a"],
            "policy_agree_rate": p["agree_rate_a"],
            "gap_rank_lr": ranks_lr.get(sdg),
            "gap_rank_zs": ranks_zs.get(sdg),
            "gap_rank_delta": rank_deltas.get(sdg),
        })

    out = {
        "method": "supervised_vs_nearest_centroid",
        "scope": "MPNet only; keyword retrieval; assignment-level comparison",
        "embedding_model": model,
        "reference_centroids": str(centroids_path),
        "research": research,
        "research_mlp_vs_zs": research_mlp,
        "policy_lr_vs_zs": {"segment": policy_segment, "document": policy_doc},
        "policy_mlp_vs_zs": mlp_policy,
        "semantic_gap_rank_deltas": per_sdg_rows,
    }
    data_out.parent.mkdir(parents=True, exist_ok=True)
    with data_out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("Saved: %s", data_out)

    # --- LaTeX table ---
    rows_tex = []
    for row in per_sdg_rows:
        def fmt_pct(v):
            return f"{v * 100:.1f}" if v is not None else "--"
        rows_tex.append(
            f"SDG {row['sdg']:2d} ({SDG_NAMES[row['sdg']]}) & "
            f"{row['research_n_lr']:,} & {fmt_pct(row['research_agree_rate'])} & {fmt_pct(row['research_mlp_agree_rate'])} & "
            f"{row['policy_n_lr']:,} & {fmt_pct(row['policy_agree_rate'])} & "
            f"{row['gap_rank_lr']} & {row['gap_rank_zs']} & {row['gap_rank_delta']} \\\\"
        )

    pol_doc_rate = policy_doc["overall_agreement"]
    research_mlp_overall = research_mlp["overall_agreement"] if research_mlp else None
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/i1_assignment_method_comparison.py — do not edit manually",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Research (segments)} & \multicolumn{2}{c}{Policy (segments)} & \multicolumn{3}{c}{Semantic-gap rank} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
        r"SDG & $n_{\text{LR}}$ & Agree \% & Res MLP \% & $n_{\text{LR}}$ & Agree \% & LR & ZS & $|\Delta|$ \\",
        r"\midrule",
    ] + rows_tex + [
        r"\midrule",
        f"Overall & {research['n_rows']:,} & {research['overall_agreement'] * 100:.1f} & "
        f"{fmt_pct(research_mlp_overall)} & "
        f"{policy_segment['n_rows']:,} & {policy_segment['overall_agreement'] * 100:.1f} & & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", table_out)
    print(f"Research LR-vs-ZS agreement: {research['overall_agreement']:.4f}; "
          f"policy segment: {policy_segment['overall_agreement']:.4f}; "
          f"policy doc: {pol_doc_rate:.4f}")
    record_fingerprint([data_out, table_out], fp, table_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                        help="Embed model (default: %(default)s; I.1 is scoped to MPNet)")
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
