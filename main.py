"""Dissertation reproducibility pipeline — single entrypoint.

Architecture — three method axes sharing a unified preprocess→segment→embed stage:

  Axis A — Supervised LR (PRIMARY result):
    prepare_data → retrain LR → score_supervised --lr --research
    → supervised research_centroids.npy → 1_semantic_gap, 0_coverage_gap

  Axis B — Supervised MLP (sensitivity):
    retrain MLP → score_supervised --mlp
    → mlp_research_centroids.npy → cross-sensitivity table

  Axis C — Zeroshot nearest-centroid (sensitivity):
    build_sdg_reference_centroids → sdg_centroids.npy
    → score_zeroshot → zeroshot/research_centroids.npy, policy_centroids.npy
    → cross-sensitivity table

Labeled corpora: osdg, benchmark, sdg_knowledge_hub, sdgi, aurora
  (consolidated into reference corpus at preprocess time).
Unlabeled: research (OpenAlex), policy (consolidated from policy_scrape,
  policy_manual, ungdc_sdg, sdgi).
sdgi is dual-role: labeled training corpus (in reference) AND
  unlabeled policy corpus (in policy).
SciBERT reuses MPNet segmented texts (--seg-model all-mpnet-base-v2).

Build-order: 0_prepare_data MUST precede both build_reference_centroids
and retrain_full_data (both read prepare_data's output files).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CODE_ROOT = Path(__file__).resolve().parent / "1_code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shared_utils import (
    canonical_artifact_paths,
    canonical_artifact_status,
    require_output_files,
    require_pdf_inputs,
)
from model_utils import (
    CANONICAL_SEGMENT_MODEL,
    DEFAULT_EMBED_MODEL,
    embed_dir_for_model,
    embed_research_dir_for_model,
    raw_dir,
    research_preprocessed_dir,
    research_segmented_dir_for_model,
    research_subset_manifest,
    scored_dir_for_model,
    segmented_dir_for_model,
    resolve_model_alias,
)
from analysis_orchestrator import run_analysis


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "4_outputs"

EMBED_BATCH_SIZE = "64"
ALL_EMBED_CORPORA = ["reference", "policy"]


def base_warm_replay_requirements(model: str = "") -> list[Path]:
    embed_root = embed_dir_for_model(model)
    return [
        embed_root / "policy.npy",
        embed_root / "metadata" / "policy_ids.json",
        embed_root / "reference.npy",
        embed_root / "metadata" / "reference_ids.json",
        embed_research_dir_for_model(model) / "metadata" / "manifest.json",
    ]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical dissertation entrypoint. Default mode prints repo status only. "
            "Mutation requires an explicit action flag."
        )
    )
    p.add_argument(
        "--warm-replay-without-appendix",
        action="store_true",
        help=(
            "Rebuild main text analysis from frozen embeddings. "
            "Appendix outputs remain committed in the repo and are not regenerated. "
            "Auto-fetches the embedded snapshot if 2_data/ is missing."
        ),
    )
    p.add_argument(
        "--warm-replay-with-appendix",
        action="store_true",
        help=(
            "Rebuild main text + all appendix analyses from frozen embeddings. "
            "Auto-fetches the embedded snapshot if 2_data/ is missing."
        ),
    )
    p.add_argument("--cold-replay", action="store_true", help="Full pipeline from live data sources — fetch, preprocess, embed, analyse. Not recommended (long runtime; OpenAlex live changes may break reproducibility).")
    p.add_argument("--appendix-all", action="store_true", help="Run all appendix stages (A2, A3, B2, C, F, H.1) standalone (requires existing main-text outputs).")
    p.add_argument("--appendix-a2-family", action="store_true", help="Run A.2 Policy Source-Family Sensitivity.")
    p.add_argument("--appendix-a3-sdg4", action="store_true", help="Run A.3 SDG 4 Lexical Artefact Audit.")
    p.add_argument("--appendix-b2-interpret", action="store_true", help="Run B.1 Lexical Illustration of the Semantic Gap.")
    p.add_argument("--appendix-c-sample-stability", action="store_true", help="Run C Sample-Stability Robustness (appendix).")
    p.add_argument("--appendix-f-register", action="store_true", help="Run F Register-Adjustment Robustness.")
    p.add_argument("--appendix-h1-cross-method", action="store_true", help="Run H.1 Cross-Method Gap Values.")
    p.add_argument("--appendix-c0-corpus-split", action="store_true", help="Export reference-corpus split-size macros.")
    p.add_argument("--appendix-d1-model-selection", action="store_true", help="Export D.1 model-selection CV macros.")
    p.add_argument("--appendix-g-distributional", action="store_true", help="Run the distributional semantic-gap robustness (MAIN-RESULT Table; OPT-IN: not run by warm replay or --appendix-all; run before --build-pdf).")
    # Deprecated aliases (hidden, kept for backward compatibility)
    p.add_argument("--policy-source-family-sensitivity", action="store_true", dest="appendix_a2_family", help=argparse.SUPPRESS)
    p.add_argument("--sdg4-lexical-audit", action="store_true", dest="appendix_a3_sdg4", help=argparse.SUPPRESS)
    p.add_argument("--semantic-gap-interpretability", action="store_true", dest="appendix_b2_interpret", help=argparse.SUPPRESS)
    p.add_argument(
        "--fetch-data-snapshot",
        nargs="?",
        const="embedded",
        choices=["raw", "embedded"],
        help=(
            "Fetch and extract a frozen dissertation data snapshot into ./2_data/. "
            "Defaults to embedded; raw is for cold-replay rebuilds."
        ),
    )
    p.add_argument(
        "--backup-data-snapshot",
        nargs="?",
        const="embedded",
        choices=["raw", "embedded", "both"],
        help=(
            "Create a dissertation data snapshot archive via the backup utility. "
            "Defaults to embedded; 'both' runs raw then embedded."
        ),
    )
    p.add_argument(
        "--register-adjustment",
        action="store_true",
        dest="appendix_f_register",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--build-pdf", action="store_true", help="Build dissertation.pdf from existing manuscript outputs (requires bash — WSL/Linux only).")
    p.add_argument("--overwrite", action="store_true", help="Required before replacing existing manuscript outputs.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Manuscript output directory. Default: 4_outputs/")
    p.add_argument(
        "--snapshot-profile",
        choices=["raw", "embedded"],
        default="embedded",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for embed_paper_shards.py in --cold-replay mode.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embed_paper_shards.py in --cold-replay mode.")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                   help="Compute precision for embedding (fp16 ≈ 2x faster on Ampere GPUs).")
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer model name (default: %(default)s). Override for model sensitivity.",
    )
    p.add_argument(
        "--stage",
        choices=["fetch", "preprocess", "segment", "embed", "train", "infer", "centroids", "analysis"],
        help="Run a single pipeline stage (assumes upstream outputs exist).",
    )
    p.add_argument("--corpus",
                   choices=["all", "reference", "policy", "research"],
                   default="all",
                   help="Corpus to segment (default: all; only used with --stage segment).")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _appendix_output_dir(base_output_dir: Path, model: str) -> Path:
    return base_output_dir


def action_requested(args: argparse.Namespace) -> bool:
    if args.stage:
        return True
    return any(
        [
            args.warm_replay_without_appendix,
            args.warm_replay_with_appendix,
            args.cold_replay,
            args.appendix_all,
            args.appendix_a2_family,
            args.appendix_a3_sdg4,
            args.appendix_b2_interpret,
            args.appendix_f_register,
            args.appendix_d1_model_selection,
            args.appendix_h1_cross_method,
            args.appendix_c_sample_stability,
            args.appendix_g_distributional,
            args.fetch_data_snapshot,
            args.backup_data_snapshot,
            args.build_pdf,
        ]
    )


def run_step(label: str, cmd: list[str], step_id: str | None = None) -> None:
    header = f"[{step_id}] {label}" if step_id else f"[{label}]"
    sep = "=" * 70
    print(f"\n{sep}", file=sys.stderr)
    print(f"  {header}", file=sys.stderr)
    print(sep, file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(file=sys.stderr)


def missing_requirements(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not (ROOT / p).exists()]


def required_warm_replay_inputs(model: str = DEFAULT_EMBED_MODEL) -> list[Path]:
    return base_warm_replay_requirements(model)


def missing_manifest_shard_paths(manifest_path: Path, shard_fields: tuple[str, ...]) -> list[Path]:
    manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        return [manifest_path.relative_to(ROOT)]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [manifest_path.relative_to(ROOT)]

    shards = payload.get("shards")
    if not isinstance(shards, list):
        return [manifest_path.relative_to(ROOT)]

    missing: list[Path] = []
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        for field in shard_fields:
            value = shard.get(field)
            if isinstance(value, str):
                path = ROOT / value
                if not path.exists():
                    missing.append(path.relative_to(ROOT))
    return missing


def missing_warm_replay_requirements(model: str = DEFAULT_EMBED_MODEL) -> list[Path]:
    missing = missing_requirements(required_warm_replay_inputs(model=model))
    embed_manifest = embed_research_dir_for_model(model) / "metadata" / "manifest.json"
    for path in missing_manifest_shard_paths(
        embed_manifest,
        ("embedding_path", "ids_path"),
    ):
        if path not in missing:
            missing.append(path)
    return missing


def canonical_exists(output_dir: Path) -> bool:
    return any(path.exists() for path in canonical_artifact_paths(output_dir))


def print_status(output_dir: Path) -> None:
    print(f"Project root: {ROOT}")
    print(f"Manuscript output dir: {output_dir}")

    warm_missing = missing_warm_replay_requirements()
    print("")
    print(f"Warm replay readiness: {'yes' if not warm_missing else 'no'}")
    if warm_missing:
        for path in warm_missing:
            print(f"  missing: {rel(ROOT / path)}")

    status = canonical_artifact_status(output_dir)
    print("")
    print("Manuscript output status:")
    print(f"  present: {len(status['present'])}")
    print(f"  missing: {len(status['missing'])}")


def build_pdf(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    require_pdf_inputs(output_dir, model)
    run_step(
        "build pdf",
        ["bash", str(ROOT / "3_writing" / "build_pdf.sh"), str(output_dir / "dissertation.pdf")],
    )


def run_sample_stability(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    require_output_files(
        output_dir / "main" / "data",
        [
            "4_2_coverage_document_weighted.json",
            "4_3_semantic_gap_distances.json",
            "4_4_interaction_correlation_asymmetry.json",
        ],
    )
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/c_sample_stability.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("sample stability", cmd, step_id="C")




def run_policy_source_family_sensitivity(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/a2_policy_source_family_sensitivity.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("policy source-family sensitivity", cmd, step_id="A2")


def run_sdg4_lexical_audit(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/a3_sdg4_lexical_audit.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("SDG 4 lexical artefact audit", cmd, step_id="A3")






def run_semantic_gap_interpretability(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    require_output_files(output_dir / "main" / model / "data", ["4_3_semantic_gap_distances.json"])
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/b2_semantic_gap_text_interpretability.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("lexical illustration of the semantic gap", cmd, step_id="B2")


def run_register_adjustment(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/f_register_adjustment.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("register-adjustment robustness", cmd, step_id="F")


def run_distributional_gap(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    """Run the distributional semantic-gap robustness (main-result table, opt-in)."""
    require_output_files(
        output_dir / "main" / model / "data",
        ["4_3_semantic_gap_distances.json"],
    )
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/1_main_text/g_distributional_gap.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("distributional semantic-gap metrics", cmd, step_id="G")


def run_corpus_split_sizes(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    """Export reference-corpus split-size macros to num_reference_split.tex."""
    import importlib.util
    script_path = ROOT / "1_code" / "7_main_analysis" / "2_appendix" / "c0_export_corpus_split_sizes.py"
    spec = importlib.util.spec_from_file_location("c0_export_corpus_split_sizes", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run(model, output_dir, overwrite=overwrite)


def run_model_selection_nums(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    """Export grid-search CV macro-F1 values to num_model_selection.tex."""
    import importlib.util
    script_path = ROOT / "1_code" / "7_main_analysis" / "2_appendix" / "d1_export_model_selection_nums.py"
    spec = importlib.util.spec_from_file_location("d1_export_model_selection_nums", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run(model, output_dir, overwrite=overwrite)

def run_h1_cross_method_gap_values(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/2_appendix/h1_cross_method_gap_values.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step("cross-method gap values", cmd, step_id="H1")


def _overwrite_flag(overwrite: bool) -> list[str]:
    return ["--overwrite"] if overwrite else []


def _reset_flag(overwrite: bool) -> list[str]:
    """Preprocess scripts are resume-safe; --overwrite forces a clean rebuild."""
    return ["--reset"] if overwrite else []


def run_build_sdg_reference_centroids(model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    run_step(
        "build SDG reference centroids",
        [sys.executable, "1_code/6_calculate_centroids/0_build_sdg_reference_centroids.py",
         "--embed-model", model] + _overwrite_flag(overwrite),
        step_id="0a",
    )


def run_build_centroid_similarity_matrix(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    run_step(
        "build centroid similarity matrix",
        [sys.executable, "1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py",
         "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(overwrite),
        step_id="9a",
    )


def _run_main_analysis_steps(output_dir: Path, model: str, overwrite: bool = False, include_appendix: bool = False) -> None:
    """Run the main-text analysis steps for a given model (no input guard).

    Step ordering (all share the same frozen labelled data from prepare_data):

      0  prepare_data          → embeddings.npy, labels.npy, sources.npy, indices/
      1  retrain_full_data LR  → sdg_classifier.joblib
      0a build_sdg_reference_centroids → sdg_centroids.npy
      2  score_supervised --lr --research → research_centroids.npy (supervised, PRIMARY)
      3  score_supervised --lr --policy   → policy_scores.npy
      3b retrain_full_data MLP           → mlp_retrained.joblib
      3c score_supervised --mlp          → mlp_research_centroids.npy
      4  check_centroid_consistency      → policy_centroids.npy (diagnostic)
      9a build_centroid_similarity_matrix → similarity matrix (reads sdg_centroids.npy)

    Then run_analysis() invokes in-process: score_zeroshot (→ zeroshot/
    research_centroids.npy, policy_centroids.npy), coverage_gap, semantic_gap,
    interaction, cross-sensitivity table, and (default model only) PCA + figures.

    Three method axes—LR (PRIMARY), MLP (sensitivity), zeroshot (sensitivity)—
    each produce their own research/policy centroids in separate namespaces.
    """
    model_args = ["--embed-model", model]
    run_step("prepare training data", [sys.executable, "1_code/4_supervised_model_train/0_prepare_data.py"] + model_args, step_id="0")
    run_step("retrain full data", [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py"] + model_args + _overwrite_flag(overwrite), step_id="1")
    run_build_sdg_reference_centroids(model, overwrite=overwrite)
    run_step("score research shards", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py"] + model_args + ["--classifier", "lr", "--corpus", "research"] + _overwrite_flag(overwrite), step_id="2")
    run_step("score policy corpus", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py"] + model_args + ["--classifier", "lr", "--corpus", "policy"] + _overwrite_flag(overwrite), step_id="3")
    # MLP is scored for every encoder (not just the default), so the
    # cross-sensitivity table can carry an MLP sub-column per encoder.
    run_step(
        "retrain MLP",
        [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py",
         "--embed-model", model, "--classifier-type", "mlp"] + _overwrite_flag(overwrite),
        step_id="3b",
    )
    run_step(
        "score MLP",
        [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
         "--embed-model", model, "--classifier", "mlp"] + _overwrite_flag(overwrite),
        step_id="3c",
    )
    run_step(
        "check centroid consistency",
        [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--output-dir", str(output_dir)] + model_args + _overwrite_flag(overwrite),
        step_id="4",
    )
    run_build_centroid_similarity_matrix(output_dir, model, overwrite=overwrite)
    # Main-text (and optionally appendix) analyses, driven in-process by the
    # orchestrator. Each analysis reads the 27 research embedding/score shards
    # directly (shard-native, mmap); no consolidated array is built or cached.
    # Must run BEFORE plot figures, which consumes the analysis outputs.
    run_analysis(model, output_dir, include_appendix=include_appendix, overwrite=overwrite)
    # Figures are MPNet-centric (fixed main/figures/ paths), so only plot for
    # the default encoder; a second encoder's tree is a robustness artifact and
    # must not overwrite the canonical figures.
    if model == DEFAULT_EMBED_MODEL:
        run_step("plot figures", [sys.executable, "1_code/8_visualization/plot_figures.py", "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(overwrite), step_id="9")


def run_main_text(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    model: str = DEFAULT_EMBED_MODEL,
    include_appendix: bool = False,
) -> None:
    missing = missing_warm_replay_requirements(model=model)
    if missing:
        missing_str = ", ".join(rel(ROOT / p) for p in missing)
        raise RuntimeError(f"Main text replay is not ready. Missing required inputs: {missing_str}")
    _run_main_analysis_steps(output_dir, model, overwrite=args.overwrite, include_appendix=include_appendix)


def run_warm_replay(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    include_appendix: bool = False,
) -> None:
    model = args.embed_model
    run_main_text(output_dir, args, include_appendix=include_appendix)
    print(
        "Main text outputs rebuilt. To build the dissertation PDF, run:\n"
        "  python main.py --build-pdf --overwrite\n"
        "Note: --build-pdf requires bash (WSL/Linux) and is not supported on bare Windows."
    )


def run_cold_replay(output_dir: Path, args: argparse.Namespace) -> None:
    print("NOTE: --cold-replay rebuilds MPNet + MiniLM + SciBERT from the raw snapshot (frozen data) in ONE run.")
    print("      It is deterministic and reproducible; no OpenAlex credentials needed when the raw snapshot is hydrated.")
    if args.embed_model != DEFAULT_EMBED_MODEL:
        print(f"NOTE: --embed-model {args.embed_model!r} is ignored by --cold-replay (all three encoders are rebuilt).")
    # Input gate: cold replay rebuilds FROM the frozen raw snapshot (2_data/0_raw/),
    # not from existing 4_outputs/. Refusing is based on missing *inputs*, never on
    # the presence of prior derived outputs — so an interrupted run can simply be
    # re-invoked to resume (each stage is resume-safe / idempotent on its own).
    if not raw_dir().exists():
        print(
            "ERROR: raw snapshot not found at 2_data/0_raw/. "
            "Cold replay rebuilds from the frozen raw snapshot — run "
            "`python main.py --fetch-data-snapshot raw` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    COLD_REPLAY_MODELS = (
        CANONICAL_SEGMENT_MODEL,
        "all-MiniLM-L6-v2",
        "allenai/scibert_scivocab_uncased",
    )

    pre_steps = [
        # — PREPROCESS (clean and structure raw data into 1_preprocessed/) —
        ("preprocess policy", [sys.executable, "1_code/1_preprocess/0_preprocess_policy.py"] + _reset_flag(args.overwrite)),
        ("preprocess ungdc", [sys.executable, "1_code/1_preprocess/0_preprocess_ungdc_sdg.py"] + _reset_flag(args.overwrite)),
        ("preprocess osdg", [sys.executable, "1_code/1_preprocess/0_preprocess_osdg.py"] + _reset_flag(args.overwrite)),
        ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_benchmark.py"] + _reset_flag(args.overwrite)),
        ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py"] + _reset_flag(args.overwrite)),
        ("preprocess aurora", [sys.executable, "1_code/1_preprocess/0_preprocess_aurora.py"] + _reset_flag(args.overwrite)),
        ("preprocess sdgi unified", [sys.executable, "1_code/1_preprocess/0_preprocess_sdgi_unified.py"] + _reset_flag(args.overwrite)),
        ("preprocess research shards", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py"] + _reset_flag(args.overwrite)),
        ("preprocess concept corpus", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py", "--retrieval", "concept"] + _reset_flag(args.overwrite)),
        # — BUILD CONSOLIDATED CORPORA —
        ("build reference corpus", [sys.executable, "1_code/1_preprocess/1_build_reference_corpus.py"] + _overwrite_flag(args.overwrite)),
        ("build policy corpus", [sys.executable, "1_code/1_preprocess/1_build_policy_corpus.py"] + _overwrite_flag(args.overwrite)),
        # — SEGMENT (canonical, ONCE, shared by every encoder) —
        ("segment reference & policy", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--all", "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
        ("segment research corpus", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--sharded",
         "--input-glob", str(research_preprocessed_dir() / "part-*.jsonl"),
         "--output-dir", str(research_segmented_dir_for_model(CANONICAL_SEGMENT_MODEL)),
         "--text-field", "combined_text", "--id-field", "openalex_id",
         "--prefix", "paper", "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
        # — EMBED (encode each source separately) —
    ]
    # Shared 50k representative subset (consumed by MiniLM + SciBERT instead of
    # the full corpus); built once from the canonical segments.
    pre_steps.append((
        "build research 50k subset",
        [sys.executable, "1_code/2_segment/2_sample_segments.py"] + _overwrite_flag(args.overwrite),
    ))
    for label, cmd in pre_steps:
        run_step(label, cmd)

    # Per-model embed + analysis. Segments are canonical (shared); only the
    # encoder (and its native context window) varies. MiniLM/SciBERT embed the
    # shared 50k subset via --input-manifest; MPNet embeds the full corpus.
    for model in COLD_REPLAY_MODELS:
        model_args = ["--embed-model", model]
        for corpus in ALL_EMBED_CORPORA:
            run_step(
                f"embed {corpus}",
                [sys.executable, "1_code/3_embed/0_embed_reference_and_policy_corpora.py",
                 "--corpus", corpus, "--batch-size", EMBED_BATCH_SIZE,
                 ] + model_args + ["--seg-model", CANONICAL_SEGMENT_MODEL]
                  + _overwrite_flag(args.overwrite),
            )
        embed_cmd = [
            sys.executable,
            "1_code/3_embed/0_embed_paper_shards.py",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
        embed_cmd.extend(model_args)
        embed_cmd.extend(_overwrite_flag(args.overwrite))
        if model != CANONICAL_SEGMENT_MODEL:
            embed_cmd.extend(["--input-manifest", str(research_subset_manifest())])
        run_step("embed paper shards", embed_cmd)

        _run_main_analysis_steps(output_dir, model=model, overwrite=args.overwrite, include_appendix=True)

    # The encoder-axis (cross-sensitivity) tables in the canonical model's dir
    # were written during the first loop pass (MPNet) before MiniLM/SciBERT
    # outputs existed. Regenerate them once now that all three are present so
    # the PDF-consumed tables show the full 3-way encoder comparison.
    run_step(
        "regenerate canonical cross-sensitivity table (all 3 encoders)",
        [sys.executable, "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
         "--output-dir", str(output_dir), "--embed-model", CANONICAL_SEGMENT_MODEL],
    )

    print(
        "Cold replay complete. To build the dissertation PDF, run:\n"
        "  python main.py --build-pdf --overwrite\n"
        "Note: --build-pdf requires bash (WSL/Linux) and is not supported on bare Windows."
    )


def run_fetch_data_snapshot(args: argparse.Namespace, *, profile_name: str, overwrite_data: bool) -> None:
    cmd = [sys.executable, "1_code/data_backup_and_fetch/fetch_data_snapshot.py", "--profile", profile_name]
    if overwrite_data:
        cmd.append("--overwrite")
    run_step(f"fetch data snapshot ({profile_name})", cmd)


def run_backup_data_snapshot(*, profile_name: str) -> None:
    cmd = [sys.executable, "1_code/data_backup_and_fetch/backup_data_snapshot.py", "--profile", profile_name]
    run_step(f"backup data snapshot ({profile_name})", cmd)


def explicit_fetch_snapshot_profile(argv: list[str]) -> str | None:
    tokens = argv[1:]
    for index, token in enumerate(tokens):
        if token.startswith("--fetch-data-snapshot="):
            return token.split("=", 1)[1]
        if token == "--fetch-data-snapshot":
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                return tokens[index + 1]
            return None
    return None


def resolve_fetch_snapshot_profile(args: argparse.Namespace) -> str | None:
    requested_profile = args.fetch_data_snapshot
    legacy_profile = args.snapshot_profile
    if requested_profile is None:
        return None
    explicit_profile = explicit_fetch_snapshot_profile(sys.argv)
    if explicit_profile is None:
        return legacy_profile
    if explicit_profile != legacy_profile and legacy_profile != "embedded":
        raise RuntimeError(
            "Conflicting fetch snapshot profiles. Use either `--fetch-data-snapshot <profile>` "
            "or the legacy `--snapshot-profile <profile>`, but not different values for both."
        )
    return explicit_profile


def selected_backup_profiles(profile_name: str) -> list[str]:
    if profile_name == "both":
        return ["raw", "embedded"]
    return [profile_name]


def ensure_warm_replay_inputs(args: argparse.Namespace, *, model: str = DEFAULT_EMBED_MODEL) -> None:
    if model != DEFAULT_EMBED_MODEL:
        # Model-sensitivity runs use model-specific embed files that are not in the snapshot.
        return
    missing = missing_warm_replay_requirements(model=model)
    if not missing:
        return

    missing_str = ", ".join(rel(ROOT / p) for p in missing[:12])
    print(f"[info] warm replay inputs missing: {missing_str}")
    if len(missing) > 12:
        print(f"[info] ... and {len(missing) - 12} more")
    run_fetch_data_snapshot(args, profile_name="embedded", overwrite_data=(ROOT / "2_data").exists())


def _run_single_stage(stage: str, output_dir: Path, args: argparse.Namespace) -> None:
    model = args.embed_model
    model_is_nondefault = model != DEFAULT_EMBED_MODEL
    model_args = ["--embed-model", model] if model_is_nondefault else []
    seg_root = segmented_dir_for_model(model)
    embed_root = embed_dir_for_model(model)
    scored_root = scored_dir_for_model(model)

    if stage == "fetch":
        steps = [
            ("fetch policy", [sys.executable, "1_code/0_fetch/fetch_policy.py"]),
            ("convert policy manual", [sys.executable, "1_code/0_fetch/convert_policy_manual.py"]),
            ("fetch sdgi corpus", [sys.executable, "1_code/0_fetch/fetch_sdgi_corpus.py"]),
            ("fetch ungdc", [sys.executable, "1_code/0_fetch/fetch_ungdc.py"]),
            ("fetch osdg", [sys.executable, "1_code/0_fetch/fetch_osdg.py"]),
            ("fetch sdg benchmark", [sys.executable, "1_code/0_fetch/fetch_sdg_benchmark.py"]),
            ("fetch sdg knowledge hub", [sys.executable, "1_code/0_fetch/fetch_sdg_knowledge_hub.py"]),
            ("fetch aurora", [sys.executable, "1_code/0_fetch/fetch_aurora.py"]),
            ("fetch openalex", [sys.executable, "1_code/0_fetch/fetch_openalex.py"]),
            ("fetch openalex concept", [sys.executable, "1_code/0_fetch/fetch_openalex.py", "--retrieval", "concept"]),
        ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "preprocess":
        steps = [
            ("preprocess policy", [sys.executable, "1_code/1_preprocess/0_preprocess_policy.py"] + _reset_flag(args.overwrite)),
            ("preprocess ungdc", [sys.executable, "1_code/1_preprocess/0_preprocess_ungdc_sdg.py"] + _reset_flag(args.overwrite)),
            ("preprocess osdg", [sys.executable, "1_code/1_preprocess/0_preprocess_osdg.py"] + _reset_flag(args.overwrite)),
            ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_benchmark.py"] + _reset_flag(args.overwrite)),
            ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py"] + _reset_flag(args.overwrite)),
            ("preprocess aurora", [sys.executable, "1_code/1_preprocess/0_preprocess_aurora.py"] + _reset_flag(args.overwrite)),
            ("preprocess sdgi unified", [sys.executable, "1_code/1_preprocess/0_preprocess_sdgi_unified.py"] + _reset_flag(args.overwrite)),
            ("preprocess research shards", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py"] + _reset_flag(args.overwrite)),
            ("preprocess concept corpus", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py", "--retrieval", "concept"] + _reset_flag(args.overwrite)),
            ("build reference corpus", [sys.executable, "1_code/1_preprocess/1_build_reference_corpus.py"] + _overwrite_flag(args.overwrite)),
            ("build policy corpus", [sys.executable, "1_code/1_preprocess/1_build_policy_corpus.py"] + _overwrite_flag(args.overwrite)),
        ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "segment":
        corpus = args.corpus
        if corpus == "all":
            steps = [
                ("segment reference & policy",
                 [sys.executable, "1_code/2_segment/segment_corpus.py",
                  "--all", "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
                ("segment research corpus",
                 [sys.executable, "1_code/2_segment/segment_corpus.py",
                  "--corpus", "research", "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
                ("build research 50k subset",
                 [sys.executable, "1_code/2_segment/2_sample_segments.py"] + _overwrite_flag(args.overwrite)),
            ]
        elif corpus == "research":
            steps = [
                ("segment research corpus",
                 [sys.executable, "1_code/2_segment/segment_corpus.py",
                  "--corpus", "research", "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
                ("build research 50k subset",
                 [sys.executable, "1_code/2_segment/2_sample_segments.py"] + _overwrite_flag(args.overwrite)),
            ]
        else:
            steps = [
                (f"segment {corpus}",
                 [sys.executable, "1_code/2_segment/segment_corpus.py",
                  "--corpus", corpus, "--embed-model", CANONICAL_SEGMENT_MODEL] + _overwrite_flag(args.overwrite)),
            ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "embed":
        for corpus in ALL_EMBED_CORPORA:
            run_step(
                f"embed {corpus}",
                 [sys.executable, "1_code/3_embed/0_embed_reference_and_policy_corpora.py",
                 "--corpus", corpus, "--batch-size", EMBED_BATCH_SIZE, "--local-files-only",
                 "--precision", args.precision, "--normalize-embeddings"] + model_args + ["--seg-model", CANONICAL_SEGMENT_MODEL]
                 + _overwrite_flag(args.overwrite),
            )
        embed_cmd = [
            sys.executable, "1_code/3_embed/0_embed_paper_shards.py",
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--chunk-size", "8192", "--local-files-only",
            "--precision", args.precision,
            "--normalize-embeddings",
        ]
        embed_cmd.extend(model_args)
        embed_cmd.extend(_overwrite_flag(args.overwrite))
        if model != CANONICAL_SEGMENT_MODEL:
            embed_cmd.extend(["--input-manifest", str(research_subset_manifest())])
        run_step("embed paper shards", embed_cmd)

    elif stage == "train":
        run_step("prepare training data", [sys.executable, "1_code/4_supervised_model_train/0_prepare_data.py", "--embed-model", model])
        run_step("retrain full data (LR)", [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py", "--embed-model", model] + _overwrite_flag(args.overwrite))
        run_step("retrain full data (MLP)", [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py", "--embed-model", model, "--classifier-type", "mlp"] + _overwrite_flag(args.overwrite))

    elif stage == "infer":
        # Score both supervised assignment methods (LR + MLP) for the encoder,
        # plus zero-shot nearest-centroid assignment. The zero-shot step depends
        # on sdg_centroids.npy, so `centroids` must run before `infer`.
        run_step("score research shards (LR)", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py", "--embed-model", model, "--classifier", "lr", "--corpus", "research"] + _overwrite_flag(args.overwrite))
        run_step("score policy corpus (LR)", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py", "--embed-model", model, "--classifier", "lr", "--corpus", "policy"] + _overwrite_flag(args.overwrite))
        run_step("score MLP", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py", "--embed-model", model, "--classifier", "mlp"] + _overwrite_flag(args.overwrite))
        run_step("zero-shot nearest-centroid assignment", [sys.executable, "1_code/5_supervised_model_infer/score_zeroshot.py", "--embed-model", model, "--output-dir", str(output_dir)] + _overwrite_flag(args.overwrite))

    elif stage == "centroids":
        # Build the SDG reference centroids (sdg_centroids.npy) consumed by the
        # zero-shot + semantic-gap analyses, then the a4 similarity matrix.
        run_step("build SDG reference centroids", [sys.executable, "1_code/6_calculate_centroids/0_build_sdg_reference_centroids.py", "--embed-model", model] + _overwrite_flag(args.overwrite))
        run_step("check centroid consistency", [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--embed-model", model] + _overwrite_flag(args.overwrite))
        run_step("build centroid similarity matrix", [sys.executable, "1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py", "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(args.overwrite))

    elif stage == "analysis":
        _run_main_analysis_steps(output_dir, model, overwrite=args.overwrite, include_appendix=True)

    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = parse_args()
    args.embed_model = resolve_model_alias(args.embed_model)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    embed_dir_for_model(args.embed_model)

    fetch_profile = resolve_fetch_snapshot_profile(args)

    if not action_requested(args):
        print_status(output_dir)
        return

    if (
        # NOTE: cold-replay and warm-replay gate on their *inputs* (raw snapshot /
        # 3_embedded), not on prior 4_outputs/, so an interrupted run can be
        # re-invoked to resume. They are intentionally excluded from this
        # output-existence refuse guard. Appendix stages and build-pdf still
        # protect committed results and require --overwrite to replace.
        args.appendix_all
        or args.appendix_a2_family
        or args.appendix_a3_sdg4
        or args.appendix_b2_interpret
        or args.appendix_c_sample_stability
        or args.appendix_c0_corpus_split
        or args.appendix_f_register
        or args.appendix_d1_model_selection
        or args.appendix_h1_cross_method
        or args.appendix_g_distributional
        or args.build_pdf
    ) and canonical_exists(output_dir) and not args.overwrite:
        print("Outputs already exist — use --overwrite to replace them.", file=sys.stderr)
        sys.exit(1)

    if args.stage:
        _run_single_stage(args.stage, output_dir, args)
    elif fetch_profile is not None:
        run_fetch_data_snapshot(args, profile_name=fetch_profile, overwrite_data=args.overwrite)
    elif args.backup_data_snapshot:
        for profile_name in selected_backup_profiles(args.backup_data_snapshot):
            run_backup_data_snapshot(profile_name=profile_name)
    elif args.cold_replay:
        run_cold_replay(output_dir, args)
    elif args.appendix_all:
        model = args.embed_model
        run_policy_source_family_sensitivity(output_dir, model=model, overwrite=args.overwrite)
        run_sdg4_lexical_audit(output_dir, model=model, overwrite=args.overwrite)
        run_semantic_gap_interpretability(output_dir, model=model, overwrite=args.overwrite)
        run_sample_stability(output_dir, model=model, overwrite=args.overwrite)
        run_register_adjustment(output_dir, model=model, overwrite=args.overwrite)
        run_model_selection_nums(output_dir, model=model, overwrite=args.overwrite)
        run_corpus_split_sizes(output_dir, model=model, overwrite=args.overwrite)
        run_h1_cross_method_gap_values(output_dir, model=model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_a2_family:
        run_policy_source_family_sensitivity(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_a3_sdg4:
        run_sdg4_lexical_audit(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_b2_interpret:
        run_semantic_gap_interpretability(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_c_sample_stability:
        run_sample_stability(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_f_register:
        run_register_adjustment(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_c0_corpus_split:
        run_corpus_split_sizes(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_d1_model_selection:
        run_model_selection_nums(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_h1_cross_method:
        run_h1_cross_method_gap_values(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.appendix_g_distributional:
        run_distributional_gap(output_dir, model=args.embed_model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif args.warm_replay_without_appendix:
        ensure_warm_replay_inputs(args)
        run_warm_replay(output_dir, args, include_appendix=False)
    elif args.warm_replay_with_appendix:
        ensure_warm_replay_inputs(args)
        run_warm_replay(output_dir, args, include_appendix=True)
    elif args.build_pdf:
        build_pdf(output_dir, model=args.embed_model)


if __name__ == "__main__":
    main()
