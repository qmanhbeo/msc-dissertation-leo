"""
Compute intra-SDG semantic gap between research and policy corpora.

The semantic gap measures whether research and policy texts assigned to the *same* SDG are
semantically similar to each other. A large semantic gap on SDG j means that even though both
corpora discuss SDG j, they do so in materially different semantic framings within that SDG.

Method:
  For each SDG j:
    1. Research cluster j  = paper embeddings assigned to SDG j (all papers; no cap needed since
                             papers are independently authored, not dominated by one document).
    2. Policy cluster j    = policy segment embeddings assigned to SDG j, with per-document segment cap.
    3. Research sub-centroid j = L2-normalised mean of research cluster j embeddings.
    4. Policy sub-centroid j   = L2-normalised mean of policy cluster j embeddings (segment-capped).
    5. semantic_similarity[j]  = cosine_sim(research_sub_centroid_j, policy_sub_centroid_j)
                               = dot product (both are unit vectors after normalisation)
    6. semantic_gap[j]         = 1 - semantic_similarity[j]

  Interpretation:
    semantic_gap = 0.0 → perfect semantic overlap; both corpora discuss SDG j identically
    semantic_gap = 1.0 → orthogonal; the corpora discuss SDG j in completely unrelated ways
    Typical range: 0.1–0.8 (for real-world policy/research text in SBERT space)

Per-document segment cap (Assumption A-CHUNKCAT):
  Without capping, SDSN 2024 (~3,179 segments) and SDGi VNR/VLR reports (31,941 total segments)
  would dominate the policy cluster centroids for whichever SDG they are assigned to.
  We cap at SEGMENT_CAP segments per source_doc per SDG. Random sampling is seeded for
  reproducibility.

  SEGMENT_CAP = 50 was chosen as a round number that:
    - Prevents any single document from contributing more than 50 segments to a policy cluster
    - Still allows documents to contribute substantively (a 50-segment sample = ~7,500 words)
    - Is conservative relative to median document size (~14 segments/document in the corpus)
  This is Assumption A-CHUNKCAT. Results with SEGMENT_CAP = 20 and SEGMENT_CAP = 100 are
  included as sensitivity checks.

Minimum cluster size:
  SDGs with fewer than MIN_CLUSTER_SIZE items in the research OR policy cluster are flagged
  as unreliable. The semantic gap estimate for these SDGs should not be reported as a finding.
  MIN_CLUSTER_SIZE = 10 is a conservative lower bound; sub-centroids built from < 10 items
  are dominated by noise.

  Note on coverage gap interaction:
   The lowest-coverage SDGs (SDG 1 and SDG 17 are the two smallest research clusters by
   proportion; SDG 10 is also low) are precisely the SDGs that appear "neglected" in coverage
   gap analysis. In absolute terms they are not empty: SDG 1 has ~18.3k papers in the full
    corpus (~340-520 in the 100k research subset) and SDG 10 ~38.4k, all well above
   MIN_CLUSTER_SIZE, so their semantic-gap estimates are not cluster-size limited. Residual
   noise is acknowledged in Assumption A-SPARSE.

Inputs:
  2_data/5_supervised_scored/{model}/research_centroids.npy       (17, dim)    float32
  2_data/5_supervised_scored/{model}/metadata/research_centroid_meta.json  list of 17 SDG centroid metadata rows
  2_data/5_supervised_scored/{model}/policy_scores.npy            float32 matrix with one row per policy segment
  2_data/5_supervised_scored/{model}/metadata/policy_scores_ids.json       list of {id, source_doc}
  2_data/3_embedded/{model}/policy.npy                 float32 matrix with one row per policy segment

Outputs:
  4_outputs/main/data/semantic_gap_distances_lr.json                  primary: semantic gap per SDG (SEGMENT_CAP=50)
  4_outputs/main/data/semantic_gap_robustness_caps_lr.json             sensitivity analysis at SEGMENT_CAP=20 and SEGMENT_CAP=100
  4_outputs/main/tables/*.tex                   generated LaTeX macros/tables

Run from project root:
    python 1_code/7_main_analysis/1_main_text/1_semantic_gap.py
"""

import logging
import argparse
import json
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

import semantic_gap_shared
import register_utils
from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, SDG_SHORT_NAMES, SDG_NUM_WORDS, N_SDG, embed_dir_for_model, resolve_model_alias, scored_dir_for_model
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint
from shard_pipeline_utils import sha256_file, load_json
from semantic_gap_shared import (
    SEGMENT_CAP_PRIMARY,
    SEGMENT_CAP_SENS_NONE,
    SEGMENT_CAP_SENS_LO,
    MIN_CLUSTER_SIZE,
    RANDOM_SEED,
    compute_sdg_semantic_gaps,
    get_cluster_assignments,
    latex_int,
    load_json,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute semantic gap outputs into the canonical output folder.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--segment-cap", type=int, default=SEGMENT_CAP_PRIMARY,
                   help="Max segments sampled per source_doc per SDG for the primary analysis (default: %(default)s)")
    p.add_argument("--research-centroids", default=None,
                   help="Override research centroids .npy (default: canonical per-model path). Used for the concept-retrieval variant.")
    p.add_argument("--research-centroid-meta", default=None,
                   help="Override research centroid metadata .json (default: canonical per-model path).")
    p.add_argument("--mlp-centroids", default=None,
                   help="Override MLP research centroids .npy (default: canonical per-model path). Used for the concept-retrieval variant.")
    p.add_argument("--mlp-policy-scores", default=None,
                   help="Override MLP policy scores .npy (default: canonical per-model path). Used for the concept-retrieval variant.")
    p.add_argument("--out-data-dir", default=None,
                   help="Override output data directory (default: canonical layout data_dir). Concept variant writes here.")
    p.add_argument("--out-tables-dir", default=None,
                   help="Override output tables directory (default: canonical layout tables_dir). Concept variant writes here.")
    p.add_argument("--embeddings", choices=["raw", "adjusted"], default="raw",
                   help="Use raw (default) or register-adjusted embeddings (project via G).")
    p.add_argument("--classifier", choices=["lr", "mlp"], default="lr",
                   help="Classifier for cluster assignments: lr (default) or mlp.")
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    is_mlp = args.classifier == "mlp"
    # Concept-retrieval variant reuses the canonical model's tables dir but writes
    # its JSON under a concept subdir; it must NOT overwrite the main manuscript's
    # num3_semantic_gap.tex / tab3_semantic_gap.tex (which describe the raw MPNet gap).
    is_concept = args.out_data_dir is not None

    # ---- Load data paths based on classifier ----
    if is_mlp:
        _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.embed_model)
        _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.embed_model)
        _POLICY_SCORES = Path(args.mlp_policy_scores) if args.mlp_policy_scores else semantic_gap_shared.get_mlp_policy_scores(args.embed_model)
        _RESEARCH_CENTROIDS = Path(args.mlp_centroids) if args.mlp_centroids else semantic_gap_shared.get_mlp_research_centroids(args.embed_model)
        # MLP centroids don't have a pre-built metadata JSON; build it on the fly.
        if args.mlp_centroids:
            # Concept variant: build metadata from the override centroids + mlp_summary.json
            # We need to find the right mlp_summary.json — it's in the same dir as the centroids.
            _mlp_dir = Path(args.mlp_centroids).parent
            _summary_path = _mlp_dir / "mlp_summary.json"
            if _summary_path.exists():
                _summary = load_json(_summary_path)
                centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
                research_meta = []
                for sdg_idx in range(N_SDG):
                    sdg = sdg_idx + 1
                    n_papers = int(_summary.get("research_coverage", {}).get(str(sdg), 0))
                    norm = float(np.linalg.norm(centroids[sdg_idx]))
                    research_meta.append({
                        "sdg": sdg,
                        "n_papers_assigned": n_papers,
                        "raw_centroid_norm": round(norm, 6),
                        "mean_cos_to_centroid": round(norm, 6),
                        "zero_flag": norm < 1e-8,
                    })
            else:
                research_meta = semantic_gap_shared.build_mlp_centroid_meta(args.embed_model)
        else:
            research_meta = semantic_gap_shared.build_mlp_centroid_meta(args.embed_model)
    else:
        _POLICY_EMB = semantic_gap_shared.get_policy_emb(args.embed_model)
        _POLICY_IDS = semantic_gap_shared.get_policy_ids(args.embed_model)
        _POLICY_SCORES = semantic_gap_shared.get_policy_scores(args.embed_model)
        _RESEARCH_CENTROIDS = Path(args.research_centroids) if args.research_centroids else semantic_gap_shared.get_research_centroids(args.embed_model)
        _RESEARCH_CENTROID_META = Path(args.research_centroid_meta) if args.research_centroid_meta else semantic_gap_shared.get_research_centroid_meta(args.embed_model)

    layout = ensure_canonical_outputs(Path(args.output_dir), model=args.embed_model)
    if args.out_data_dir:
        Path(args.out_data_dir).mkdir(parents=True, exist_ok=True)
    if args.out_tables_dir:
        Path(args.out_tables_dir).mkdir(parents=True, exist_ok=True)

    # Determine output paths and fingerprint based on --embeddings mode.
    is_adjusted = args.embeddings == "adjusted"
    sem_gap_name = "semantic_gap_distances_mlp.json" if is_mlp else "semantic_gap_distances_lr.json"
    sem_sens_name = "semantic_gap_robustness_caps_mlp.json" if is_mlp else "semantic_gap_robustness_caps_lr.json"
    if is_adjusted:
        if args.out_data_dir:
            adj_data_dir = Path(args.out_data_dir) / "adjusted"
        else:
            adj_data_dir = layout.data_dir / "adjusted"
        adj_data_dir.mkdir(parents=True, exist_ok=True)
        out_sem_gap = adj_data_dir / sem_gap_name
        out_sem_sens = adj_data_dir / sem_sens_name
    else:
        out_sem_gap = Path(args.out_data_dir).joinpath(sem_gap_name) if args.out_data_dir else layout.data_dir / sem_gap_name
        out_sem_sens = Path(args.out_data_dir).joinpath(sem_sens_name) if args.out_data_dir else layout.data_dir / sem_sens_name

    tables_dir = Path(args.out_tables_dir) if args.out_tables_dir else layout.tables_dir
    log.info("Canonical output dir: %s", layout.data_dir)

    SCRIPT_VERSION = "2"
    PRIMARY = out_sem_gap
    OUTPUTS = [out_sem_gap, out_sem_sens]

    # Fingerprint: include classifier-specific input files.
    if is_mlp:
        fp = fingerprint_of(
            semantic_gap_shared.get_mlp_research_centroids(args.embed_model),
            _POLICY_EMB, _POLICY_IDS, _POLICY_SCORES,
        )
    else:
        fp = fingerprint_of(_RESEARCH_CENTROIDS, _RESEARCH_CENTROID_META,
                            _POLICY_EMB, _POLICY_IDS, _POLICY_SCORES,
                            embed_dir_for_model(args.embed_model) / "policy.npy")
    fp += SCRIPT_VERSION
    if is_adjusted:
        g_path = register_utils.register_dir(args.embed_model) / "G.npy"
        fp += f"_adjusted_{args.classifier}_{register_utils.track_for_model(args.embed_model)}"
        fp += fingerprint_of(g_path)
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s — inputs unchanged", PRIMARY)
        return

    # ---- Load research centroids/meta ----
    log.info("Loading research centroids: %s", _RESEARCH_CENTROIDS)
    research_centroids = np.load(_RESEARCH_CENTROIDS).astype(np.float32)
    if is_mlp:
        # research_meta was built by build_mlp_centroid_meta() above.
        pass
    else:
        research_meta = load_json(_RESEARCH_CENTROID_META)
    if research_centroids.shape[0] != N_SDG:
        raise RuntimeError(f"Expected research centroids shape ({N_SDG}, d), got {research_centroids.shape}")
    if len(research_meta) != N_SDG:
        raise RuntimeError(f"Expected {N_SDG} research centroid meta rows, got {len(research_meta)}")
    research_counts = np.array([int(r["n_papers_assigned"]) for r in research_meta], dtype=np.int64)
    research_cohesions = np.array([float(r["mean_cos_to_centroid"]) for r in research_meta], dtype=np.float32)

    log.info("Loading policy embeddings: %s", _POLICY_EMB)
    policy_emb = np.load(_POLICY_EMB)
    policy_ids = load_json(_POLICY_IDS)

    # ---- Load score matrices for cluster assignments ----
    log.info("Loading score matrices...")
    policy_scores = np.load(_POLICY_SCORES)

    # Hard assignment (0-indexed SDG index).
    policy_assignments = get_cluster_assignments(policy_scores)

    # ---- Adjusted mode: project through G ----
    # Intentional design (PLAN_register_topic_decomposition.md §6.1): LR/MLP keep
    # their RAW-space hard assignments (policy_assignments above come from raw
    # scores) and only project the cluster CENTROIDS/VECTORS through G for the
    # gap. This differs from zero-shot, which RE-ASSIGNS on projected embeddings.
    if is_adjusted:
        G = register_utils.load_G(args.embed_model)
        log.info("Projecting research centroids through G (%d directions)...", G.shape[0])
        research_centroids = register_utils.project(research_centroids, G)
        log.info("Projecting policy embeddings through G...")
        policy_emb = register_utils.project(policy_emb, G)

    # ---- Provenance fingerprint (fail-closed guard) ----
    # Records the exact score-shard / embedding manifest / classifier artifact /
    # G checkpoint that produced these gaps, so a cross-epoch re-score (different
    # hydrated snapshot) or register-code change is visible instead of silently
    # shifting the raw/adjusted gaps.
    provenance = {
        "embedding_model": args.embed_model,
        "classifier": args.classifier,
        "embeddings": args.embeddings,
    }
    scored = scored_dir_for_model(args.embed_model)
    rs_manifest = scored / "paper_scores_shards" / "metadata" / "manifest.json"
    if rs_manifest.exists():
        rm = load_json(rs_manifest)
        provenance["research_score_manifest"] = {
            "path": str(rs_manifest),
            "input_embedding_manifest": rm.get("input_embedding_manifest"),
            "model_path": rm.get("model_path"),
            "shard_sha256": [s.get("sha256") for s in rm.get("shards", [])],
        }
    pol_scores = scored / "policy_scores.npy"
    pol_ids = scored / "metadata" / "policy_scores_ids.json"
    provenance["policy_score"] = {
        "ids_path": str(pol_ids),
        "scores_path": str(pol_scores),
        "scores_sha256": sha256_file(pol_scores) if pol_scores.exists() else None,
    }
    ed = embed_dir_for_model(args.embed_model)
    pol_emb = ed / "policy.npy"
    provenance["policy_embedding_sha256"] = sha256_file(pol_emb) if pol_emb.exists() else None
    # The concept / override route builds its centroids from the
    # concept-retrieved corpus (research_concept), NOT the full research corpus.
    # Record the actual embedding manifest so a future concept-embedding drift
    # is visible instead of silently masked behind the full-corpus manifest.
    is_override = args.research_centroids is not None or args.mlp_centroids is not None
    rmanifest = (
        ed / "research_concept" / "metadata" / "manifest.json"
        if is_override
        else ed / "research_shards" / "metadata" / "manifest.json"
    )
    if rmanifest.exists():
        provenance["research_embedding_manifest"] = (
            load_json(rmanifest).get("input_embedding_manifest")
            if load_json(rmanifest).get("input_embedding_manifest") else str(rmanifest)
        )
    if is_adjusted:
        g_path = register_utils.register_dir(args.embed_model) / "G.npy"
        track = register_utils.track_for_model(args.embed_model)
        chk = register_utils.register_dir(args.embed_model) / "checkpoint.json"
        reg = {"g_path": str(g_path), "track": track}
        if g_path.exists():
            reg["g_sha256"] = sha256_file(g_path)
        if chk.exists():
            cfg = load_json(chk).get("config", {})
            reg["script_version"] = cfg.get("script_version")
            reg["n_target"] = cfg.get("n_target")
        provenance["register"] = reg

    # ---- Primary analysis (SEGMENT_CAP = 50) ----
    log.info("")
    log.info("=" * 60)
    log.info("PRIMARY SEMANTIC GAP (segment cap = %d)", args.segment_cap)
    log.info("=" * 60)
    rng_primary = np.random.default_rng(RANDOM_SEED)
    primary_results, _ = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, args.segment_cap, rng_primary
    )

    # Summary: top gap + mean for console; full table saved to JSON/TeX.
    reliable = [r for r in primary_results if not r["unreliable"] and r["semantic_gap"] is not None]
    if reliable:
        top = max(reliable, key=lambda x: x["semantic_gap"])
        mean_gap = np.mean([r["semantic_gap"] for r in reliable])
        log.info("Semantic gap (cap=%d): top=SDG %d (%.4f)  mean=%.4f  n_reliable=%d",
                 args.segment_cap, top["sdg"], top["semantic_gap"], mean_gap, len(reliable))

    # ---- Sensitivity analyses (LR and MLP use the same computation) ----
    log.info("")
    log.info("=" * 60)
    log.info("SENSITIVITY: segment cap = %d", SEGMENT_CAP_SENS_LO)
    log.info("=" * 60)
    rng_lo = np.random.default_rng(RANDOM_SEED)
    sens_lo, _ = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, SEGMENT_CAP_SENS_LO, rng_lo
    )

    log.info("")
    log.info("=" * 60)
    log.info("SENSITIVITY: no segment cap (uncapped)")
    log.info("=" * 60)
    sens_none, _ = compute_sdg_semantic_gaps(
        research_centroids, research_counts, research_cohesions,
        policy_emb, policy_assignments,
        policy_ids, SEGMENT_CAP_SENS_NONE, rng_lo
    )

    # Check sensitivity: do rankings change substantially across caps?
    # A finding is robust if its gap rank is stable across caps.
    rank_changes = []
    for i in range(N_SDG):
        gaps = [sens_lo[i]["semantic_gap"], primary_results[i]["semantic_gap"], sens_none[i]["semantic_gap"]]
        if all(g is not None for g in gaps):
            ranks = np.argsort(np.argsort([-g for g in gaps]))
            rank_changes.append(int(np.max(np.abs(ranks - ranks[1]))))
    max_rank_change = max(rank_changes) if rank_changes else 0
    log.info("Sensitivity: max rank change across caps=%d  n_reliable=%d", max_rank_change, len(rank_changes))

    # ---- Build output JSON ----
    primary_out = {
        "method": "centroid_to_centroid",
        "embedding_model": args.embed_model,
        "classifier": args.classifier,
        "embeddings": args.embeddings,
        "segment_cap": args.segment_cap,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "random_seed": RANDOM_SEED,
        "note": (
            "semantic_gap[j] = 1 - cosine_sim(research_sub_centroid_j, policy_sub_centroid_j). "
            "Both sub-centroids are L2-normalised means of cluster embeddings. "
            "Policy clusters are segment-capped per source_doc to avoid SDSN/SDGi dominance (A19). "
            "SDGs flagged unreliable have fewer than MIN_CLUSTER_SIZE items in research or policy."
        ),
        "per_sdg": primary_results,
        "reliable_sdgs": [r["sdg"] for r in primary_results if not r["unreliable"]],
        "unreliable_sdgs": [r["sdg"] for r in primary_results if r["unreliable"]],
        "provenance": provenance,
    }

    sensitivity_out = {
        "method": "centroid_to_centroid",
        "random_seed": RANDOM_SEED,
        "provenance": provenance,
        "note": (
            "Sensitivity analysis: same computation as semantic_gap_distances_lr.json but with an alternative "
            "per-document segment cap (20) and an uncapped (none) run. Use to verify finding robustness. "
            "Rankings should be broadly stable if findings are robust."
        ),
        f"cap_{SEGMENT_CAP_SENS_LO}": sens_lo,
        "cap_none": sens_none,
    }

    with out_sem_gap.open("w", encoding="utf-8") as f:
        json.dump(primary_out, f, indent=2)
    log.info("Saved: %s", out_sem_gap)

    with out_sem_sens.open("w", encoding="utf-8") as f:
        json.dump(sensitivity_out, f, indent=2)
    log.info("Saved: %s", out_sem_sens)

    log.info("")
    log.info("Next step: python 1_code/7_main_analysis/1_main_text/2_coverage_semantic_interaction.py")

    # ---- Write LaTeX generated outputs (raw LR mode only; not concept variant) ----
    if is_adjusted or is_mlp or is_concept:
        log.info("Adjusted/MLP mode: skipping tex generation (JSON written to %s)", out_sem_gap)
        record_fingerprint(OUTPUTS, fp, PRIMARY)
        return

    gen_dir = tables_dir

    # Extract per-SDG values from primary_results (SDG order 1–17)
    per_sdg_map = {r["sdg"]: r for r in primary_results}
    gaps = [per_sdg_map[s]["semantic_gap"] for s in range(1, N_SDG + 1)]
    valid_gaps = [g for g in gaps if g is not None]
    mean_gap = float(np.mean(valid_gaps))
    sorted_gaps = sorted(valid_gaps)
    n = len(sorted_gaps)
    median_gap = float(
        sorted_gaps[n // 2] if n % 2 == 1
        else (sorted_gaps[n // 2 - 1] + sorted_gaps[n // 2]) / 2
    )

    # num3_semantic_gap.tex — macro definitions
    num_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/1_semantic_gap.py — do not edit manually",
        rf"\newcommand{{\MeanSemanticGap}}{{{mean_gap:.3f}}}",
        rf"\newcommand{{\MedianSemanticGap}}{{{median_gap:.3f}}}",
        rf"\newcommand{{\SemanticGapRange}}{{{max(valid_gaps) - min(valid_gaps):.3f}}}",
    ]
    for sdg_num, word in SDG_NUM_WORDS.items():
        row = per_sdg_map[sdg_num]
        g = row["semantic_gap"]
        if g is not None:
            num_lines.append(rf"\newcommand{{\SemanticGapSdg{word}}}{{{g:.3f}}}")
        num_lines.append(
            rf"\newcommand{{\NPapersSdg{word}}}{{{latex_int(int(row['n_papers']))}}}"
        )
        num_lines.append(
            rf"\newcommand{{\NPolicyDocsSdg{word}}}{{{latex_int(int(row['n_policy_docs_capped']))}}}"
        )
    (gen_dir / "num3_semantic_gap.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "num3_semantic_gap.tex")

    # tab3_semantic_gap.tex — full tabular block
    sorted_results = sorted(
        [r for r in primary_results if r["semantic_gap"] is not None],
        key=lambda x: x["semantic_gap"],
        reverse=True,
    )
    tab_lines = [
        "% Auto-generated by 1_code/7_main_analysis/1_main_text/1_semantic_gap.py — do not edit manually",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"SDG & Description & Sem. Gap & n$_{\text{res}}$ & n$_{\text{pol docs}}$ \\",
        r"\midrule",
    ]
    for r in sorted_results:
        sdg = r["sdg"]
        name = SDG_SHORT_NAMES[sdg].replace("&", r"\&")
        g = r["semantic_gap"]
        n_res = r["n_papers"]
        n_pol = r["n_policy_docs_capped"]
        tab_lines.append(
            rf"{sdg:2d} & {name} & {g:.3f} & {n_res:,} & {n_pol:,} \\"
        )
    tab_lines.extend([
        r"\midrule",
        r"\multicolumn{2}{l}{Mean semantic gap} & \MeanSemanticGap & & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (gen_dir / "tab3_semantic_gap.tex").write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", gen_dir / "tab3_semantic_gap.tex")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
