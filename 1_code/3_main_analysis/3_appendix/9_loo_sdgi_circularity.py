"""
Leave-one-source-out circularity check for the SDGi policy sub-corpus.

The canonical combined centroid draws from OSDG + Knowledge Hub + SDGi + Aurora.
The scored policy corpus also contains SDGi VNR/VLR segments. Because SDGi is the
one reference source that is both inside the centroid and inside the scored policy
corpus, the policy-side "A15" top-score (mean best cosine to the combined centroid)
could in principle be inflated by within-source self-similarity rather than real
semantic alignment. This script isolates that possibility.

Test (confirmed supervisor scope):
  Build an OSDG + Knowledge Hub + Aurora centroid (SDGi EXCLUDED). Rescore only the
  SDGi policy segments (the single sub-corpus that also feeds the centroid) against
  this SDGi-excluded centroid. Compute the policy-segment mean top score, the research
  mean top score, and their gap; compare to the canonical 0.328 (0.560 policy vs
  0.232 research) produced by 0_coverage_gap.py.

This isolates the one place circularity could appear: if the gap shrinks materially
when SDGi is removed from the centroid, the original 0.328 was partly a within-source
artefact. If the gap is unchanged, the A15 asymmetry is not explained by SDGi
circularity.

Convention notes:
  - Centroids are built exactly as 1_build_sdg_centroids.py: mean of unit vectors per
    SDG across sources, then L2-normalised to a unit vector. SDG 17 has no OSDG
    labels, so it is sourced from KH + Aurora here.
  - Each policy segment is scored by its max cosine to the 17 centroids (same
    operation as policy_scores.npy / a15_policy_top). Research papers scored the same
    way (same operation as research mean_top_overall).
  - SDGi segment isolation reuses the source-family map from
    4_policy_source_family_sensitivity.py (source_doc -> sdgi_vnr_vlr).

Outputs:
  4_outputs/appendix/a_loo_sdgi_circularity/data/loo_sdgi_circularity.json
  4_outputs/appendix/a_loo_sdgi_circularity/data/loo_sdgi_circularity.csv
  4_outputs/appendix/a_loo_sdgi_circularity/tables/num_loo_sdgi_circularity.tex

Run from project root:
    python 1_code/3_main_analysis/3_appendix/9_loo_sdgi_circularity.py
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
POLICY_PREPROCESSED_ROOT = ROOT / "2_data" / "1_preprocessed" / "policy_all"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, embed_dir_for_model, scored_dir_for_model

OUTPUT_SUBDIR = "a_loo_sdgi_circularity"
DATA_JSON = "loo_sdgi_circularity.json"
DATA_CSV = "loo_sdgi_circularity.csv"
NUM_TEX = "num_loo_sdgi_circularity.tex"

FAMILY_FILE_MAP = {
    "curated_ai_sdg": [
        POLICY_PREPROCESSED_ROOT / "policy_scrape" / "policy_scrape_segments.jsonl",
        POLICY_PREPROCESSED_ROOT / "policy_manual" / "policy_manual_segments.jsonl",
    ],
    "sdgi_vnr_vlr": [
        POLICY_PREPROCESSED_ROOT / "sdgi_corpus" / "sdgi_segments.jsonl",
    ],
    "ungdc_speeches": [
        POLICY_PREPROCESSED_ROOT / "ungdc_sdg" / "ungdc_sdg_segments.jsonl",
    ],
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-source-out (SDGi) circularity check.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--overwrite", action="store_true", help="Recompute from source.")
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help=argparse.SUPPRESS)
    return p.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_source_family_map() -> dict[str, str]:
    source_family: dict[str, str] = {}
    for family, paths in FAMILY_FILE_MAP.items():
        for path in paths:
            for row in iter_jsonl(path):
                source_doc = str(row["source_doc"])
                existing = source_family.get(source_doc)
                if existing is not None and existing != family:
                    raise RuntimeError(
                        f"source_doc '{source_doc}' appears in multiple families: {existing} vs {family}"
                    )
                source_family[source_doc] = family
    if not source_family:
        raise RuntimeError("No source-family assignments were built from policy preprocessed files.")
    return source_family


def build_centroid_matrix(emb: np.ndarray, ids: list[dict]) -> np.ndarray:
    """Per-SDG unit centroid = L2-normalised mean of unit vectors (matches 1_build_sdg_centroids.py)."""
    rows = []
    for sdg in range(1, N_SDG + 1):
        idxs = [i for i, r in enumerate(ids) if r["sdg"] == sdg]
        if not idxs:
            rows.append(np.full(emb.shape[1], np.nan, dtype=np.float32))
            continue
        vecs = emb[idxs].astype(np.float32)
        raw = vecs.mean(axis=0)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            rows.append(np.full(emb.shape[1], np.nan, dtype=np.float32))
        else:
            rows.append((raw / norm).astype(np.float32))
    return np.stack(rows, axis=0)


def main() -> None:
    args = parse_args()
    model = args.model
    embed_dir = embed_dir_for_model(model)
    scored_dir = scored_dir_for_model(model)
    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / OUTPUT_SUBDIR
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load reference embeddings (EXCLUDE SDGi) ----
    osdg_emb = np.load(embed_dir / "osdg.npy").astype(np.float32)
    osdg_ids = load_json(embed_dir / "metadata" / "osdg_ids.json")
    kh_emb = np.load(embed_dir / "sdg_knowledge_hub.npy").astype(np.float32)
    kh_ids = load_json(embed_dir / "metadata" / "sdg_knowledge_hub_ids.json")
    aurora_emb = np.load(embed_dir / "aurora.npy").astype(np.float32)
    aurora_ids = load_json(embed_dir / "metadata" / "aurora_ids.json")
    # SDGi deliberately NOT loaded.

    osdg_cent = build_centroid_matrix(osdg_emb, osdg_ids)
    kh_cent = build_centroid_matrix(kh_emb, kh_ids)
    aurora_cent = build_centroid_matrix(aurora_emb, aurora_ids)

    # Combine OSDG + KH + Aurora per SDG (SDGi excluded).
    loo_centroids = np.full((N_SDG, osdg_emb.shape[1]), np.nan, dtype=np.float32)
    loo_n_sources = np.zeros(N_SDG, dtype=int)
    for sdg in range(N_SDG):
        parts = []
        for src in (osdg_cent, kh_cent, aurora_cent):
            v = src[sdg]
            if not np.isnan(v[0]):
                parts.append(v)
        if not parts:
            continue
        stacked = np.stack(parts, axis=0)
        raw = stacked.mean(axis=0)
        norm = float(np.linalg.norm(raw))
        loo_centroids[sdg] = (raw / norm).astype(np.float32)
        loo_n_sources[sdg] = len(parts)

    valid = ~np.isnan(loo_centroids[:, 0])
    loo_centroids_clean = loo_centroids[valid]
    valid_indices = np.where(valid)[0]

    log.info("LOO (SDGi-excluded) centroid built. SDGs with centroids: %d/%d", int(valid.sum()), N_SDG)
    log.info("Per-SDG source count (OSDG+KH+Aurora): %s", {int(s + 1): int(loo_n_sources[s]) for s in valid_indices})

    # ---- Load policy embeddings + isolate SDGi segments ----
    policy_emb = np.load(embed_dir / "policy.npy").astype(np.float32)
    policy_ids = load_json(scored_dir / "metadata" / "policy_scores_ids.json")
    family_map = build_source_family_map()

    sdgi_idxs = [
        i for i, r in enumerate(policy_ids)
        if family_map.get(r.get("source_doc"), "") == "sdgi_vnr_vlr"
    ]
    log.info("SDGi policy segments isolated: %d of %d total policy segments", len(sdgi_idxs), len(policy_ids))

    sdgi_emb = policy_emb[sdgi_idxs]
    sdgi_scores = sdgi_emb @ loo_centroids_clean.T          # (n_sdgi, n_valid)
    sdgi_top = sdgi_scores.max(axis=1)
    policy_top_sdgi_excl = float(sdgi_top.mean())

    # Full policy corpus top score against LOO centroid (diagnostic context).
    full_scores = policy_emb @ loo_centroids_clean.T
    full_policy_top = float(full_scores.max(axis=1).mean())

    # ---- Research papers against LOO centroid ----
    research_shard_manifest = load_json(scored_dir / "paper_scores_shards" / "metadata" / "manifest.json")
    emb_manifest = load_json(embed_dir / "research_shards" / "metadata" / "manifest.json")
    score_shards = {int(r["shard_id"]): r for r in research_shard_manifest["shards"]}
    emb_shards = {int(r["shard_id"]): r for r in emb_manifest["shards"]}
    if sorted(score_shards) != sorted(emb_shards):
        raise RuntimeError("Research shard IDs misaligned.")

    top_sum = 0.0
    row_count = 0
    for shard_id in sorted(score_shards):
        emb_path = ROOT / emb_shards[shard_id]["embedding_path"]
        emb = np.load(emb_path).astype(np.float32)
        scores = emb @ loo_centroids_clean.T
        top_vals = scores.max(axis=1)
        top_sum += float(top_vals.sum())
        row_count += int(scores.shape[0])
    paper_top_sdgi_excl = top_sum / float(row_count)

    gap = policy_top_sdgi_excl - paper_top_sdgi_excl

    # ---- Canonical baseline (from num_coverage.tex) ----
    canonical = load_json(ROOT / "4_outputs" / "main" / "tables" / "num_coverage.tex") if False else None
    # Read the three canonical macros directly.
    num_cov = (ROOT / "4_outputs" / "main" / "tables" / "num_coverage.tex").read_text(encoding="utf-8")
    def _read_macro(name: str) -> float:
        for line in num_cov.splitlines():
            if name in line:
                return float(line.split("{")[-1].rstrip("}"))
        raise RuntimeError(f"macro {name} not found")
    canon_policy = _read_macro("AffFifteenPolicyScore")
    canon_paper = _read_macro("AffFifteenPaperScore")
    canon_diff = _read_macro("AffFifteenDiff")

    result = {
        "script": "1_code/3_main_analysis/3_appendix/9_loo_sdgi_circularity.py",
        "scope": "SDGi-excluded (OSDG+KH+Aurora) centroid; SDGi policy segments rescored.",
        "n_sdgi_segments": len(sdgi_idxs),
        "n_policy_segments_total": len(policy_ids),
        "n_research_papers": row_count,
        "loo_centroid_sources_per_sdg": {int(s + 1): int(loo_n_sources[s]) for s in valid_indices},
        "policy_top_sdgi_excl_centroid": round(policy_top_sdgi_excl, 4),
        "full_policy_top_sdgi_excl_centroid": round(full_policy_top, 4),
        "paper_top_sdgi_excl_centroid": round(paper_top_sdgi_excl, 4),
        "gap_sdgi_segments": round(gap, 4),
        "canonical_policy_top": canon_policy,
        "canonical_paper_top": canon_paper,
        "canonical_gap": canon_diff,
        "gap_change_vs_canonical": round(gap - canon_diff, 4),
    }

    (data_dir / DATA_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (data_dir / DATA_CSV).open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in result.items():
            f.write(f"{k},{v}\n")

    num_lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/9_loo_sdgi_circularity.py — do not edit manually",
        rf"\newcommand{{\LooSdgiNSegments}}{{{len(sdgi_idxs):,}}}",
        rf"\newcommand{{\LooSdgiPctCorpus}}{{{100.0 * len(sdgi_idxs) / len(policy_ids):.1f}}}",
        rf"\newcommand{{\LooSdgiPolicyTop}}{{{policy_top_sdgi_excl:.3f}}}",
        rf"\newcommand{{\LooSdgiPaperTop}}{{{paper_top_sdgi_excl:.3f}}}",
        rf"\newcommand{{\LooSdgiGap}}{{{gap:.3f}}}",
        rf"\newcommand{{\LooSdgiGapChange}}{{{gap - canon_diff:.3f}}}",
    ]
    (tables_dir / NUM_TEX).write_text("\n".join(num_lines) + "\n", encoding="utf-8")

    log.info("")
    log.info("=== LOO SDGi CIRCULARITY RESULT ===")
    log.info("  SDGi-excl policy seg top score : %.3f", policy_top_sdgi_excl)
    log.info("  SDGi-excl research paper top   : %.3f", paper_top_sdgi_excl)
    log.info("  LOO gap (policy - research)    : %.3f", gap)
    log.info("  Canonical gap (0.560-0.232)    : %.3f", canon_diff)
    log.info("  Gap change vs canonical        : %+.3f", gap - canon_diff)
    log.info("Saved: %s", data_dir / DATA_JSON)


if __name__ == "__main__":
    main()
