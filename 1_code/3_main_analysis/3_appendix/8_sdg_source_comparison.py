"""
Per-SDG source comparison: validation F1, centroid similarity, coverage, and semantic gap.

For each of five reference configurations (Combined, OSDG-only, SDGi-only,
Knowledge Hub-only, Aurora-only), this script reports:
  1. Per-SDG validation F1 against the SDG Classification Benchmark
  2. Cosine similarity of the source-only centroid to the canonical combined centroid
  3. Research coverage (papers assigned) per SDG
  4. Semantic gap (1 - cos between research and policy sub-centroids)

The Combined column provides the canonical baseline. OSDG, SDGi, KH,
and Aurora columns show how much each individual source deviates from the combined.

Expensive research and policy re-scoring results are cached in
  2_data/3_scored/source_comparison_cache/
and reused on re-runs unless input files change or --overwrite is passed.

Outputs:
  4_outputs/appendix/a1_sdg_source_comparison/data/
    comparison_summary.json          — all metrics
    comparison_table.csv             — raw CSV
  4_outputs/appendix/a1_sdg_source_comparison/tables/
    num_a1_source_comparison.tex    — LaTeX macros
    tab_a1_source_comparison_f1cos.tex — F1 + cosine table
    tab_a1_source_comparison_covgap.tex — coverage + gap table

Run from project root:
    python 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py
    python 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py --output-dir 4_outputs
    python 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from semantic_gap_shared import cap_policy_indices_per_doc

DEFAULT_OUTPUT_ROOT = Path("outputs")

SDG_CENTROIDS = Path("2_data/3_scored/sdg_centroids.npy")
BENCHMARK_EMB = Path("2_data/2_embedded/benchmark.npy")
BENCHMARK_IDS = Path("2_data/2_embedded/metadata/benchmark_ids.json")
OSDG_EMB = Path("2_data/2_embedded/osdg.npy")
OSDG_IDS = Path("2_data/2_embedded/metadata/osdg_ids.json")
KH_EMB = Path("2_data/2_embedded/sdg_knowledge_hub.npy")
KH_IDS = Path("2_data/2_embedded/metadata/sdg_knowledge_hub_ids.json")
SDGI_EMB = Path("2_data/2_embedded/sdgi.npy")
SDGI_IDS = Path("2_data/2_embedded/metadata/sdgi_ids.json")
AURORA_EMB = Path("2_data/2_embedded/aurora.npy")
AURORA_IDS = Path("2_data/2_embedded/metadata/aurora_ids.json")
POLICY_EMB = Path("2_data/2_embedded/policy.npy")
POLICY_IDS = Path("2_data/3_scored/metadata/policy_scores_ids.json")
RESEARCH_EMBED_MANIFEST = Path("2_data/2_embedded/research_shards/metadata/manifest.json")
RESEARCH_SCORE_MANIFEST = Path("2_data/3_scored/paper_scores_shards/metadata/manifest.json")

OUTPUT_SUBDIR = "a1_sdg_source_comparison"
SUMMARY_JSON = "comparison_summary.json"
TABLE_CSV = "comparison_table.csv"
NUM_TEX = "num_a1_source_comparison.tex"
TABLE_F1COS_TEX = "tab_a1_source_comparison_f1cos.tex"
TABLE_COVGAP_TEX = "tab_a1_source_comparison_covgap.tex"

CACHE_DIR = Path("2_data/3_scored/source_comparison_cache")
CACHE_RESEARCH_COUNTS = "{}_research_counts.npy"
CACHE_RESEARCH_SUMS = "{}_research_sums.npy"
CACHE_POLICY_COUNTS = "{}_policy_counts.npy"
CACHE_POLICY_SUMS = "{}_policy_sums.npy"
CACHE_MANIFEST = "{}_manifest.json"

N_SDG = 17
SEGMENT_CAP_PRIMARY = 50
MIN_CLUSTER_SIZE = 10
RANDOM_SEED = 42

SDG_NUM_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
    15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-SDG source comparison.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--overwrite", action="store_true", help="Recompute and overwrite cache.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_centroid_matrix(emb: np.ndarray, ids: list[dict]) -> np.ndarray:
    rows = []
    for sdg in range(1, N_SDG + 1):
        idxs = [i for i, r in enumerate(ids) if r["sdg"] == sdg]
        if not idxs:
            row = np.full(emb.shape[1], np.nan, dtype=np.float32)
        else:
            vecs = emb[idxs]
            raw = vecs.mean(axis=0)
            norm = float(np.linalg.norm(raw))
            if norm < 1e-8:
                row = np.full(emb.shape[1], np.nan, dtype=np.float32)
            else:
                row = (raw / norm).astype(np.float32)
        rows.append(row)
    return np.stack(rows, axis=0)


def count_per_sdg(ids: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r in ids:
        sdg = r["sdg"]
        counts[sdg] = counts.get(sdg, 0) + 1
    return counts


def build_per_sdg_index(ids: list[dict]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for i, r in enumerate(ids):
        sdg = r["sdg"]
        result.setdefault(sdg, []).append(i)
    return result


def resolve_manifest_data_path(stored_path: str) -> Path:
    raw = Path(stored_path)
    if raw.is_absolute():
        return raw
    return ROOT / raw


# ---------------------------------------------------------------------------
# Validation (F1)
# ---------------------------------------------------------------------------

def compute_validation_f1(centroids: np.ndarray, bench_emb: np.ndarray, bench_ids: list[dict]) -> np.ndarray:
    true_sdgs = np.array([r["sdg"] for r in bench_ids], dtype=int)
    valid_rows = ~np.isnan(centroids[:, 0])
    if not valid_rows.any():
        return np.full(N_SDG, np.nan, dtype=np.float64)
    centroids_clean = centroids[valid_rows]
    scores = bench_emb @ centroids_clean.T
    labels_present = np.where(valid_rows)[0] + 1
    pred_sdgs = np.full(len(true_sdgs), -1, dtype=int)
    pred_sdgs_valid = labels_present[scores.argmax(axis=1)]
    pred_sdgs[valid_rows.any()] = pred_sdgs_valid
    per_sdg_f1 = f1_score(true_sdgs, pred_sdgs, labels=list(range(1, N_SDG + 1)), average=None, zero_division=0)
    result = np.full(N_SDG, np.nan, dtype=np.float64)
    for i, sdg in enumerate(range(1, N_SDG + 1)):
        if valid_rows[sdg - 1]:
            result[sdg - 1] = per_sdg_f1[i]
    return result


# ---------------------------------------------------------------------------
# Full research scoring (coverage + sub-centroids)
# ---------------------------------------------------------------------------

def load_aligned_research_shards() -> list[dict[str, Any]]:
    score_manifest = load_json(RESEARCH_SCORE_MANIFEST)
    emb_manifest = load_json(RESEARCH_EMBED_MANIFEST)
    score_shards = {int(row["shard_id"]): row for row in score_manifest["shards"]}
    emb_shards = {int(row["shard_id"]): row for row in emb_manifest["shards"]}
    shard_ids = sorted(score_shards)
    if shard_ids != sorted(emb_shards):
        raise RuntimeError("Research score shard IDs do not align with research embedding shard IDs.")
    aligned = []
    for shard_id in shard_ids:
        score_row = score_shards[shard_id]
        emb_row = emb_shards[shard_id]
        aligned.append({
            "shard_id": shard_id,
            "score_path": resolve_manifest_data_path(str(score_row["score_path"])),
            "embedding_path": resolve_manifest_data_path(str(emb_row["embedding_path"])),
        })
    return aligned


def score_research_full(centroids: np.ndarray, shards: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Score all research papers against centroids.
    
    Returns:
        counts: (17,) int64 — papers assigned to each SDG
        sums:   (17, 384) float64 — sum of assigned-paper embeddings per SDG
    """
    dim = centroids.shape[1]
    counts = np.zeros(N_SDG, dtype=np.int64)
    sums = np.zeros((N_SDG, dim), dtype=np.float64)
    valid = ~np.isnan(centroids[:, 0])
    centroids_clean = centroids[valid]
    valid_indices = np.where(valid)[0]

    for shard in shards:
        emb = np.load(shard["embedding_path"]).astype(np.float32)
        scores = emb @ centroids_clean.T
        assignments = valid_indices[scores.argmax(axis=1)]
        for idx in range(len(valid_indices)):
            sdg_idx = valid_indices[idx]
            mask = assignments == sdg_idx
            n = int(mask.sum())
            if n > 0:
                counts[sdg_idx] += n
                sums[sdg_idx] += emb[mask].sum(axis=0).astype(np.float64)

    return counts, sums


# ---------------------------------------------------------------------------
# Full policy scoring (sub-centroids with segment cap)
# ---------------------------------------------------------------------------

def score_policy_full(centroids: np.ndarray, policy_emb: np.ndarray, policy_ids: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Score all policy segments against centroids, with per-document segment capping.
    
    Returns:
        counts: (17,) int64 — capped segments assigned to each SDG
        sums:   (17, 384) float64 — sum of capped-segment embeddings per SDG
    """
    dim = centroids.shape[1]
    valid = ~np.isnan(centroids[:, 0])
    centroids_clean = centroids[valid]
    valid_indices = np.where(valid)[0]

    scores = policy_emb @ centroids_clean.T
    assignments_full = valid_indices[scores.argmax(axis=1)]

    # Group by SDG before capping
    rng = np.random.default_rng(RANDOM_SEED)
    counts = np.zeros(N_SDG, dtype=np.int64)
    sums = np.zeros((N_SDG, dim), dtype=np.float64)

    for sdg_idx in valid_indices:
        mask = assignments_full == sdg_idx
        all_idxs = np.where(mask)[0].tolist()
        capped = cap_policy_indices_per_doc(all_idxs, policy_ids, SEGMENT_CAP_PRIMARY, rng)
        if not capped:
            continue
        emb_capped = policy_emb[capped]
        counts[sdg_idx] = len(capped)
        sums[sdg_idx] = emb_capped.sum(axis=0).astype(np.float64)

    return counts, sums


# ---------------------------------------------------------------------------
# Normalise sums to unit centroids
# ---------------------------------------------------------------------------

def normalise_sums(counts: np.ndarray, sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert embedding sums to unit vectors.
    
    Returns:
        centroids: (17, 384) float32 — NaN where count < MIN_CLUSTER_SIZE or near-zero norm
        valid:     (17,) bool — which SDGs have reliable centroids
    """
    dim = sums.shape[1]
    centroids = np.full((N_SDG, dim), np.nan, dtype=np.float32)
    for idx in range(N_SDG):
        if counts[idx] < MIN_CLUSTER_SIZE:
            continue
        norm = float(np.linalg.norm(sums[idx]))
        if norm < 1e-8:
            continue
        centroids[idx] = (sums[idx] / norm).astype(np.float32)
    valid = ~np.isnan(centroids[:, 0])
    return centroids, valid


# ---------------------------------------------------------------------------
# Semantic gap
# ---------------------------------------------------------------------------

def compute_semantic_gaps(research_centroids: np.ndarray, policy_centroids: np.ndarray) -> np.ndarray:
    """Per-SDG semantic gap = 1 - cos(research_centroid, policy_centroid).
    
    Returns (17,) float64 — NaN where either centroid is unreliable.
    """
    gaps = np.full(N_SDG, np.nan, dtype=np.float64)
    for idx in range(N_SDG):
        rc = research_centroids[idx]
        pc = policy_centroids[idx]
        if np.isnan(rc[0]) or np.isnan(pc[0]):
            continue
        sim = float(np.dot(rc.astype(np.float64), pc.astype(np.float64)))
        gaps[idx] = 1.0 - sim
    return gaps


# ---------------------------------------------------------------------------
# Cosine to combined per SDG
# ---------------------------------------------------------------------------

def compute_cosine_to_combined(source_centroids: np.ndarray, combined_centroids: np.ndarray) -> np.ndarray:
    result = np.full(N_SDG, np.nan, dtype=np.float64)
    for idx in range(N_SDG):
        s = source_centroids[idx]
        c = combined_centroids[idx]
        if not np.isnan(s[0]) and not np.isnan(c[0]):
            result[idx] = float(np.dot(s, c))
    return result


# ---------------------------------------------------------------------------
# Cache for research + policy scoring
# ---------------------------------------------------------------------------

def _cache_input_mtimes(research_shards: list[dict], policy_emb_path: Path, source_emb_path: Path | None) -> dict[str, float]:
    paths = [policy_emb_path]
    if source_emb_path is not None:
        paths.append(source_emb_path)
    for shard in research_shards:
        paths.append(Path(shard["embedding_path"]))
    return {str(p): os.path.getmtime(p) for p in paths}


def _cache_valid(source_name: str, input_mtimes: dict[str, float]) -> bool:
    manifest_path = CACHE_DIR / CACHE_MANIFEST.format(source_name)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("type") == "full" and manifest.get("input_mtimes") == input_mtimes
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def _cache_save(source_name: str, research_counts: np.ndarray, research_sums: np.ndarray,
                policy_counts: np.ndarray, policy_sums: np.ndarray,
                input_mtimes: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / CACHE_RESEARCH_COUNTS.format(source_name), research_counts)
    np.save(CACHE_DIR / CACHE_RESEARCH_SUMS.format(source_name), research_sums)
    np.save(CACHE_DIR / CACHE_POLICY_COUNTS.format(source_name), policy_counts)
    np.save(CACHE_DIR / CACHE_POLICY_SUMS.format(source_name), policy_sums)
    manifest = {
        "source": source_name,
        "type": "full",
        "input_mtimes": input_mtimes,
        "computed_at": datetime.now().isoformat(),
    }
    (CACHE_DIR / CACHE_MANIFEST.format(source_name)).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _cache_load(source_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rc = np.load(CACHE_DIR / CACHE_RESEARCH_COUNTS.format(source_name))
    rs = np.load(CACHE_DIR / CACHE_RESEARCH_SUMS.format(source_name))
    pc = np.load(CACHE_DIR / CACHE_POLICY_COUNTS.format(source_name))
    ps = np.load(CACHE_DIR / CACHE_POLICY_SUMS.format(source_name))
    return rc, rs, pc, ps


def compute_or_load_research_policy(
    source_name: str,
    centroids: np.ndarray,
    research_shards: list[dict],
    policy_emb: np.ndarray,
    policy_ids: list[dict],
    source_emb_path: Path | None,
    *,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (research_counts, research_sums, policy_counts, policy_sums).
    
    Uses cache unless overwrite=True or inputs have changed.
    """
    input_mtimes = _cache_input_mtimes(research_shards, POLICY_EMB, source_emb_path)

    if not overwrite and _cache_valid(source_name, input_mtimes):
        log.info("  %s: cache hit — loading cached research + policy scores", source_name)
        return _cache_load(source_name)

    log.info("  %s: computing research scores...", source_name)
    rc, rs = score_research_full(centroids, research_shards)
    log.info("  %s: computing policy scores...", source_name)
    pc, ps = score_policy_full(centroids, policy_emb, policy_ids)

    _cache_save(source_name, rc, rs, pc, ps, input_mtimes)
    log.info("  %s: cached to %s", source_name, CACHE_DIR)
    return rc, rs, pc, ps


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.3f}"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "--"
    return f"{value:,}"


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_num_tex(path: Path, results: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py",
    ]
    for row in results:
        sdg = row["sdg"]
        word = SDG_NUM_WORDS[sdg]
        for source in ["Combined", "OSDG", "SDGi", "KnowledgeHub", "Aurora"]:
            key = source.lower()
            nv = row.get(f"{key}_n")
            if nv is not None:
                lines.append(rf"\newcommand{{\Src{word}{source}N}}{{{fmt_int(nv)}}}")
            fv = row.get(f"{key}_f1")
            if fv is not None and not (isinstance(fv, float) and np.isnan(fv)):
                lines.append(rf"\newcommand{{\Src{word}{source}FOne}}{{{fmt_pct(fv)}}}")
            cv = row.get(f"{key}_cosine")
            if cv is not None and not (isinstance(cv, float) and np.isnan(cv)):
                lines.append(rf"\newcommand{{\Src{word}{source}Cosine}}{{{fmt_pct(cv)}}}")
            cov = row.get(f"{key}_coverage")
            if cov is not None and not (isinstance(cov, int) and cov == 0 and source != "Combined"):
                lines.append(rf"\newcommand{{\Src{word}{source}Cov}}{{{fmt_int(cov)}}}")
            gv = row.get(f"{key}_gap")
            if gv is not None and not (isinstance(gv, float) and np.isnan(gv)):
                lines.append(rf"\newcommand{{\Src{word}{source}Gap}}{{{fmt_pct(gv)}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell_val(row: dict, key: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def write_table_f1cos(path: Path, results: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"SDG & \multicolumn{2}{c}{Combined} & \multicolumn{2}{c}{OSDG} & \multicolumn{2}{c}{SDGi} & \multicolumn{2}{c}{Knowledge Hub} & \multicolumn{2}{c}{Aurora} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        r"& F1 & $\cos$ & F1 & $\cos$ & F1 & $\cos$ & F1 & $\cos$ & F1 & $\cos$ \\",
        r"\midrule",
    ]
    for row in results:
        sdg_label = f"SDG {row['sdg']}"
        lines.append(
            rf"{sdg_label} & {_cell_val(row, 'combined_f1')} & {_cell_val(row, 'combined_cosine')} & "
            rf"{_cell_val(row, 'osdg_f1')} & {_cell_val(row, 'osdg_cosine')} & "
            rf"{_cell_val(row, 'sdgi_f1')} & {_cell_val(row, 'sdgi_cosine')} & "
            rf"{_cell_val(row, 'knowledgehub_f1')} & {_cell_val(row, 'knowledgehub_cosine')} & "
            rf"{_cell_val(row, 'aurora_f1')} & {_cell_val(row, 'aurora_cosine')} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_covgap(path: Path, results: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"SDG & \multicolumn{2}{c}{Combined} & \multicolumn{2}{c}{OSDG} & \multicolumn{2}{c}{SDGi} & \multicolumn{2}{c}{Knowledge Hub} & \multicolumn{2}{c}{Aurora} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        r"& $n$ & gap & $n$ & gap & $n$ & gap & $n$ & gap & $n$ & gap \\",
        r"\midrule",
    ]
    for row in results:
        sdg_label = f"SDG {row['sdg']}"
        lines.append(
            rf"{sdg_label} & {_cell_val(row, 'combined_coverage')} & {_cell_val(row, 'combined_gap')} & "
            rf"{_cell_val(row, 'osdg_coverage')} & {_cell_val(row, 'osdg_gap')} & "
            rf"{_cell_val(row, 'sdgi_coverage')} & {_cell_val(row, 'sdgi_gap')} & "
            rf"{_cell_val(row, 'knowledgehub_coverage')} & {_cell_val(row, 'knowledgehub_gap')} & "
            rf"{_cell_val(row, 'aurora_coverage')} & {_cell_val(row, 'aurora_gap')} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / OUTPUT_SUBDIR
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PER-SDG SOURCE COMPARISON (with coverage and semantic gap)")
    log.info("=" * 60)

    # ---- Load canonical combined centroids ----
    log.info("Loading canonical combined centroids: %s", SDG_CENTROIDS)
    combined_centroids = np.load(SDG_CENTROIDS).astype(np.float32)
    log.info("  shape=%s", combined_centroids.shape)

    # ---- Load per-source embeddings ----
    log.info("Loading OSDG embeddings: %s", OSDG_EMB)
    osdg_emb = np.load(OSDG_EMB).astype(np.float32)
    osdg_ids = load_json(OSDG_IDS)
    log.info("  shape=%s (SDGs 1-16 only)", osdg_emb.shape)

    log.info("Loading SDGi embeddings: %s", SDGI_EMB)
    sdgi_emb = np.load(SDGI_EMB).astype(np.float32)
    sdgi_ids = load_json(SDGI_IDS)
    log.info("  shape=%s", sdgi_emb.shape)

    log.info("Loading Knowledge Hub embeddings: %s", KH_EMB)
    kh_emb = np.load(KH_EMB).astype(np.float32)
    kh_ids = load_json(KH_IDS)
    log.info("  shape=%s", kh_emb.shape)

    log.info("Loading Aurora embeddings: %s", AURORA_EMB)
    aurora_emb = np.load(AURORA_EMB).astype(np.float32)
    aurora_ids = load_json(AURORA_IDS)
    log.info("  shape=%s", aurora_emb.shape)
    n_with_text = sum(1 for r in aurora_ids if len(r.get("text", "").strip()) > 50)
    log.info("  texts with abstract-like text: %d / %d", n_with_text, len(aurora_ids))

    log.info("Loading benchmark: %s", BENCHMARK_EMB)
    bench_emb = np.load(BENCHMARK_EMB).astype(np.float32)
    bench_ids = load_json(BENCHMARK_IDS)
    log.info("  shape=%s", bench_emb.shape)

    # ---- Build per-source centroid matrices ----
    log.info("\nBuilding per-source centroids...")
    osdg_centroids = build_centroid_matrix(osdg_emb, osdg_ids)
    sdgi_centroids = build_centroid_matrix(sdgi_emb, sdgi_ids)
    kh_centroids = build_centroid_matrix(kh_emb, kh_ids)
    aurora_centroids = build_centroid_matrix(aurora_emb, aurora_ids)

    osdg_counts = count_per_sdg(osdg_ids)
    sdgi_counts = count_per_sdg(sdgi_ids)
    kh_counts = count_per_sdg(kh_ids)
    aurora_counts = count_per_sdg(aurora_ids)
    combined_counts = {
        sdg: osdg_counts.get(sdg, 0) + sdgi_counts.get(sdg, 0) + kh_counts.get(sdg, 0) + aurora_counts.get(sdg, 0)
        for sdg in range(1, N_SDG + 1)
    }

    log.info("  OSDG centroids:  %s (SDG 17 = NaN)", osdg_centroids.shape)
    log.info("  SDGi centroids:  %s", sdgi_centroids.shape)
    log.info("  KH centroids:    %s", kh_centroids.shape)
    log.info("  Aurora centroids:%s", aurora_centroids.shape)

    # ---- Validate per source ----
    log.info("\nValidating per source against benchmark...")
    combined_f1 = compute_validation_f1(combined_centroids, bench_emb, bench_ids)
    osdg_f1 = compute_validation_f1(osdg_centroids, bench_emb, bench_ids)
    sdgi_f1 = compute_validation_f1(sdgi_centroids, bench_emb, bench_ids)
    kh_f1 = compute_validation_f1(kh_centroids, bench_emb, bench_ids)
    aurora_f1 = compute_validation_f1(aurora_centroids, bench_emb, bench_ids)

    log.info("  Combined F1 (SDG 17): %.3f", combined_f1[16] if not np.isnan(combined_f1[16]) else -1)
    log.info("  OSDG F1 (SDG 16):     %.3f", osdg_f1[15] if not np.isnan(osdg_f1[15]) else -1)
    log.info("  SDGi F1 (SDG 17):     %.3f", sdgi_f1[16] if not np.isnan(sdgi_f1[16]) else -1)
    log.info("  KH F1 (SDG 17):       %.3f", kh_f1[16] if not np.isnan(kh_f1[16]) else -1)
    log.info("  Aurora F1 (SDG 17):   %.3f", aurora_f1[16] if not np.isnan(aurora_f1[16]) else -1)

    # ---- Compute cosine to combined ----
    log.info("\nComputing per-SDG cosine to combined centroid...")
    osdg_cos = compute_cosine_to_combined(osdg_centroids, combined_centroids)
    sdgi_cos = compute_cosine_to_combined(sdgi_centroids, combined_centroids)
    kh_cos = compute_cosine_to_combined(kh_centroids, combined_centroids)
    aurora_cos = compute_cosine_to_combined(aurora_centroids, combined_centroids)

    # ---- Load research shards and policy data ----
    log.info("\nLoading research shards...")
    research_shards = load_aligned_research_shards()
    log.info("  %d research shards", len(research_shards))

    log.info("Loading policy embeddings: %s", POLICY_EMB)
    policy_emb = np.load(POLICY_EMB).astype(np.float32)
    policy_ids = load_json(POLICY_IDS)
    log.info("  shape=%s", policy_emb.shape)

    # ---- Score research + policy for each source (cached) ----
    sources = [
        ("combined", combined_centroids, SDG_CENTROIDS),
        ("osdg", osdg_centroids, OSDG_EMB),
        ("sdgi", sdgi_centroids, SDGI_EMB),
        ("knowledgehub", kh_centroids, KH_EMB),
        ("aurora", aurora_centroids, AURORA_EMB),
    ]

    source_metrics = {}

    for source_name, centroids, source_emb_path in sources:
        log.info("\n--- %s ---", source_name)

        # F1
        f1_vec = {
            "combined": combined_f1, "osdg": osdg_f1,
            "sdgi": sdgi_f1, "knowledgehub": kh_f1, "aurora": aurora_f1,
        }[source_name]

        # Cosine to combined
        cos_vec = {
            "combined": np.ones(N_SDG, dtype=np.float64),
            "osdg": osdg_cos, "sdgi": sdgi_cos, "knowledgehub": kh_cos,
            "aurora": aurora_cos,
        }[source_name]

        # Research + policy scoring (cached)
        rc, rs, pc, ps = compute_or_load_research_policy(
            source_name, centroids, research_shards, policy_emb, policy_ids,
            source_emb_path if source_name != "combined" else None,
            overwrite=args.overwrite,
        )

        # Normalise to unit centroids
        res_centroids, res_valid = normalise_sums(rc, rs)
        pol_centroids, pol_valid = normalise_sums(pc, ps)

        # Coverage
        coverage_vec = rc.copy()

        # Semantic gap
        gap_vec = compute_semantic_gaps(res_centroids, pol_centroids)

        # Build per-SDG metrics
        per_sdg = []
        for sdg in range(1, N_SDG + 1):
            idx = sdg - 1
            per_sdg.append({
                "f1": float(f1_vec[idx]) if not np.isnan(f1_vec[idx]) else None,
                "cosine": float(cos_vec[idx]) if not np.isnan(cos_vec[idx]) else None,
                "coverage": int(coverage_vec[idx]),
                "gap": float(gap_vec[idx]) if not np.isnan(gap_vec[idx]) else None,
            })

        source_metrics[source_name] = per_sdg

        log.info("  F1 range:   %s — %s",
                 fmt_pct(min(f1 for f1 in f1_vec if not np.isnan(f1))),
                 fmt_pct(max(f1 for f1 in f1_vec if not np.isnan(f1))))
        log.info("  Coverage range: %s — %s",
                 fmt_int(int(max(coverage_vec))),
                 fmt_int(int(min(coverage_vec[coverage_vec > 0])) if any(coverage_vec > 0) else 0))
        gap_vals = [g for g in gap_vec if not np.isnan(g)]
        if gap_vals:
            log.info("  Gap range:  %s — %s", fmt_pct(min(gap_vals)), fmt_pct(max(gap_vals)))

    # ---- Build result rows ----
    results = []
    for sdg in range(1, N_SDG + 1):
        idx = sdg - 1
        row = {"sdg": sdg}

        for src_name, src_label in [("combined", "combined"), ("osdg", "osdg"), ("sdgi", "sdgi"), ("knowledgehub", "knowledgehub"), ("aurora", "aurora")]:
            m = source_metrics[src_name][idx]
            n_counts = {"combined": combined_counts, "osdg": osdg_counts, "sdgi": sdgi_counts, "knowledgehub": kh_counts, "aurora": aurora_counts}[src_name]
            row[f"{src_label}_n"] = n_counts.get(sdg, 0)
            row[f"{src_label}_f1"] = m["f1"]
            row[f"{src_label}_cosine"] = m["cosine"]
            row[f"{src_label}_coverage"] = m["coverage"]
            row[f"{src_label}_gap"] = m["gap"]

        results.append(row)

    # ---- Print summary ----
    log.info("\n%-4s  %-20s  %-20s  %-20s  %-20s  %-20s",
             "SDG", "Combined", "OSDG", "SDGi", "KH", "Aurora")
    log.info("-" * 110)
    for row in results:
        sdg = row["sdg"]
        cf = fmt_pct(row["combined_f1"])
        of = fmt_pct(row["osdg_f1"])
        sf = fmt_pct(row["sdgi_f1"])
        kf = fmt_pct(row["knowledgehub_f1"])
        af = fmt_pct(row["aurora_f1"])
        log.info("SDG %2d  CombF1=%-7s OSDGF1=%-7s SDGiF1=%-7s KHF1=%-7s AurF1=%-7s",
                 sdg, cf, of, sf, kf, af)

    # ---- Save CSV ----
    csv_fields = [
        "sdg",
        "combined_n", "combined_f1", "combined_cosine", "combined_coverage", "combined_gap",
        "osdg_n", "osdg_f1", "osdg_cosine", "osdg_coverage", "osdg_gap",
        "sdgi_n", "sdgi_f1", "sdgi_cosine", "sdgi_coverage", "sdgi_gap",
        "knowledgehub_n", "knowledgehub_f1", "knowledgehub_cosine", "knowledgehub_coverage", "knowledgehub_gap",
        "aurora_n", "aurora_f1", "aurora_cosine", "aurora_coverage", "aurora_gap",
    ]
    with (data_dir / TABLE_CSV).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ---- Save JSON ----
    summary = {
        "script": "1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py",
        "note": "Per-SDG comparison across reference sources: F1, centroid cosine, coverage, semantic gap.",
        "sources": [s[0] for s in sources],
        "results": results,
    }
    (data_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Save LaTeX outputs ----
    write_num_tex(tables_dir / NUM_TEX, results)
    write_table_f1cos(tables_dir / TABLE_F1COS_TEX, results)
    write_table_covgap(tables_dir / TABLE_COVGAP_TEX, results)

    log.info("\nSaved:")
    log.info("  %s", data_dir / SUMMARY_JSON)
    log.info("  %s", data_dir / TABLE_CSV)
    log.info("  %s", tables_dir / NUM_TEX)
    log.info("  %s", tables_dir / TABLE_F1COS_TEX)
    log.info("  %s", tables_dir / TABLE_COVGAP_TEX)

    print(f"\n{'='*60}")
    print("Per-SDG Source Comparison Complete")
    print(f"{'='*60}")
    for row in results[:5]:
        print(f"  SDG {row['sdg']:2d}  | cov_c={row['combined_coverage']:>6d}  gap_c={fmt_pct(row['combined_gap']):>6s}")
    print("  ...")
    print(f"\nSaved to: {out_root}")
    print(f"  {SUMMARY_JSON}")
    print(f"  {TABLE_CSV}")
    print(f"  {tables_dir / NUM_TEX}")
    print(f"  {tables_dir / TABLE_F1COS_TEX}")
    print(f"  {tables_dir / TABLE_COVGAP_TEX}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
