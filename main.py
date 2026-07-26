from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    DEFAULT_EMBED_MODEL,
    embed_dir_for_model,
    embed_research_dir_for_model,
    research_preprocessed_dir,
    research_segmented_dir_for_model,
    scored_dir_for_model,
    segmented_dir_for_model,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "4_outputs"

EMBED_BATCH_SIZE = "64"
ALL_EMBED_CORPORA = ["osdg", "benchmark", "sdg_knowledge_hub", "sdgi", "aurora",
                     "policy_scrape", "policy_manual", "ungdc_sdg"]


def base_warm_replay_requirements(model: str = "") -> list[Path]:
    embed_root = embed_dir_for_model(model)
    return [
        embed_root / "policy.npy",
        embed_root / "metadata" / "policy_ids.json",
        embed_root / "osdg.npy",
        embed_root / "metadata" / "osdg_ids.json",
        embed_root / "benchmark.npy",
        embed_root / "metadata" / "benchmark_ids.json",
        embed_root / "sdg_knowledge_hub.npy",
        embed_root / "metadata" / "sdg_knowledge_hub_ids.json",
        embed_root / "sdgi.npy",
        embed_root / "metadata" / "sdgi_ids.json",
        embed_root / "aurora.npy",
        embed_root / "metadata" / "aurora_ids.json",
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
    p.add_argument("--appendix-all", action="store_true", help="Run all appendix stages (A2, A3, A3b, B2, C, F) standalone (requires existing main-text outputs).")
    p.add_argument("--appendix-a2-family", action="store_true", help="Run A.2 Policy Source-Family Sensitivity.")
    p.add_argument("--appendix-a3-sdg4", action="store_true", help="Run A.3 SDG 4 Lexical Artefact Audit.")
    p.add_argument("--appendix-a3b-circularity", action="store_true", help="Run A.3b SDGi Circularity Note.")
    p.add_argument("--appendix-b2-interpret", action="store_true", help="Run B.2 Lexical Illustration of the Semantic Gap.")
    p.add_argument("--appendix-c-sample-stability", action="store_true", help="Run C Sample-Stability Robustness (appendix).")
    p.add_argument("--appendix-f-register", action="store_true", help="Run F Register-Adjustment Robustness.")
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
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer model name (default: %(default)s). Override for model sensitivity.",
    )
    p.add_argument("--model", dest="embed_model", help=argparse.SUPPRESS)
    p.add_argument(
        "--stage",
        choices=["fetch", "preprocess", "segment", "embed", "train", "infer", "centroids", "analysis"],
        help="Run a single pipeline stage (assumes upstream outputs exist).",
    )
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
            args.appendix_a3b_circularity,
            args.appendix_b2_interpret,
            args.appendix_f_register,
            args.appendix_c_sample_stability,
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


def build_pdf(output_dir: Path) -> None:
    require_pdf_inputs(output_dir)
    run_step(
        "build pdf",
        ["bash", str(ROOT / "3_writing" / "build_pdf.sh"), str(output_dir / "dissertation.pdf")],
    )


def run_sample_stability(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    require_output_files(
        output_dir / "main" / "data",
        [
            "4_2_coverage_document_weighted.json",
            "4_3_semantic_gap_distances.json",
            "4_4_interaction_correlation_asymmetry.json",
        ],
    )
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/c_sample_stability.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("sample stability", cmd, step_id="C")




def run_policy_source_family_sensitivity(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/a2_policy_source_family_sensitivity.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("policy source-family sensitivity", cmd, step_id="A2")


def run_sdg4_lexical_audit(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/a3_sdg4_lexical_audit.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("SDG 4 lexical artefact audit", cmd, step_id="A3")


def run_loo_sdgi_circularity(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/a3b_loo_sdgi_circularity.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("LOO SDGi circularity check", cmd, step_id="A3b")





def run_semantic_gap_interpretability(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    require_output_files(output_dir / "main" / "data", ["4_3_semantic_gap_distances.json"])
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/b2_semantic_gap_text_interpretability.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("lexical illustration of the semantic gap", cmd, step_id="B3")


def run_register_adjustment(output_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> None:
    actual_output_dir = _appendix_output_dir(output_dir, model)
    cmd = [sys.executable, "1_code/7_main_analysis/3_appendix/f_register_adjustment.py", "--output-dir", str(actual_output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--model", model]
    run_step("register-adjustment robustness", cmd, step_id="E")


def _run_main_analysis_steps(output_dir: Path, model: str) -> None:
    """Run the main-text analysis steps for a given model (no input guard).

    Steps 0-1 train the classifier deterministically from frozen embeddings.
    Steps 2-3 run inference with the freshly-trained classifier.
    Steps 4+ run analysis.
    """
    model_args = ["--model", model]
    run_step("prepare training data", [sys.executable, "1_code/4_supervised_model_train/0_prepare_data.py"] + model_args, step_id="0")
    run_step("retrain full data", [sys.executable, "1_code/4_supervised_model_train/1_retrain_full_data.py"] + model_args, step_id="1")
    run_step("score research shards", [sys.executable, "1_code/5_supervised_model_infer/0_score_research_shards.py"] + model_args, step_id="2")
    run_step("score policy corpus", [sys.executable, "1_code/5_supervised_model_infer/1_score_policy.py"] + model_args, step_id="3")
    run_step(
        "check centroid consistency",
        [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--output-dir", str(output_dir)] + model_args,
        step_id="4",
    )
    if model == DEFAULT_EMBED_MODEL:
        run_step(
            "zero-shot nearest-centroid scoring",
            [sys.executable, "1_code/7_main_analysis/1_canonical/0_zeroshot_scoring.py",
             "--output-dir", str(output_dir)] + model_args,
            step_id="5",
        )
    run_step("coverage gap", [sys.executable, "1_code/7_main_analysis/1_canonical/0_coverage_gap.py", "--output-dir", str(output_dir)] + model_args, step_id="6")
    run_step("semantic gap", [sys.executable, "1_code/7_main_analysis/1_canonical/1_semantic_gap.py", "--output-dir", str(output_dir)] + model_args, step_id="7")
    run_step(
        "coverage semantic interaction",
        [sys.executable, "1_code/7_main_analysis/1_canonical/2_coverage_semantic_interaction.py", "--output-dir", str(output_dir)] + model_args,
        step_id="8",
    )
    run_step("plot figures", [sys.executable, "1_code/8_visualization/plot_figures.py", "--output-dir", str(output_dir)], step_id="9")
    if model == DEFAULT_EMBED_MODEL:
        run_step(
            "generate cross-sensitivity table",
            [sys.executable, "1_code/7_main_analysis/1_canonical/3_generate_cross_sensitivity_table.py",
             "--output-dir", str(output_dir)],
            step_id="10",
        )
    if model == DEFAULT_EMBED_MODEL:
        run_step(
            "PCA semantic landscape",
            [sys.executable, "1_code/7_main_analysis/1_canonical/0_pca_semantic_landscape.py",
             "--output-dir", str(output_dir)],
            step_id="11",
        )


def run_main_text(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    model: str = DEFAULT_EMBED_MODEL,
) -> None:
    missing = missing_warm_replay_requirements(model=model)
    if missing:
        missing_str = ", ".join(rel(ROOT / p) for p in missing)
        raise RuntimeError(f"Main text replay is not ready. Missing required inputs: {missing_str}")
    _run_main_analysis_steps(output_dir, model)


def run_warm_replay(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    include_appendix: bool = False,
) -> None:
    model = args.embed_model
    run_main_text(output_dir, args)
    if include_appendix:
        run_policy_source_family_sensitivity(output_dir, model=model)
        run_sdg4_lexical_audit(output_dir, model=model)
        run_loo_sdgi_circularity(output_dir, model=model)
        run_semantic_gap_interpretability(output_dir, model=model)
        run_sample_stability(output_dir, model=model)
        run_register_adjustment(output_dir, model=model)
    print(
        "Main text outputs rebuilt. To build the dissertation PDF, run:\n"
        "  python main.py --build-pdf --overwrite\n"
        "Note: --build-pdf requires bash (WSL/Linux) and is not supported on bare Windows."
    )


def run_cold_replay(output_dir: Path, args: argparse.Namespace) -> None:
    print("NOTE: --cold-replay rebuilds from the raw snapshot (frozen data).")
    print("      For fresh data, run: python main.py --fetch-data-snapshot --profile raw")
    print("      Cold replay from frozen snapshots is deterministic and reproducible.")

    model = args.embed_model
    model_is_nondefault = model != DEFAULT_EMBED_MODEL
    model_args = ["--model", model] if model_is_nondefault else []

    pre_steps = [
        # — PREPROCESS (clean and structure raw data into 1_preprocessed/) —
        ("preprocess policy", [sys.executable, "1_code/1_preprocess/0_preprocess_policy.py"]),
        ("filter ungdc", [sys.executable, "1_code/1_preprocess/0_filter_ungdc_sdg.py"]),
        ("preprocess osdg", [sys.executable, "1_code/1_preprocess/0_preprocess_osdg.py"]),
        ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_benchmark.py"]),
        ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py"]),
        ("preprocess aurora", [sys.executable, "1_code/1_preprocess/0_preprocess_aurora.py"]),
        ("preprocess sdgi unified", [sys.executable, "1_code/1_preprocess/0_preprocess_sdgi_unified.py"]),
        ("preprocess research shards", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py"]),
        # — SEGMENT (token-count-aware segmentation into 2_segmented/{model}/) —
        ("segment knowledge hub", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "sdg_knowledge_hub", "--model", model]),
        ("segment aurora", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "aurora", "--model", model]),
        ("segment policy scrape", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "policy_scrape", "--model", model]),
        ("segment policy manual", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "policy_manual", "--model", model]),
        ("segment ungdc", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "ungdc_sdg", "--model", model]),
        ("segment sdgi", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--corpus", "sdgi", "--model", model]),
        ("build policy corpus", [sys.executable, "1_code/1_preprocess/1_build_policy_corpus.py", "--model", model]),
        ("segment research corpus", [sys.executable, "1_code/2_segment/segment_corpus.py",
         "--sharded",
         "--input-glob", str(research_preprocessed_dir() / "part-*.jsonl"),
         "--output-dir", str(research_segmented_dir_for_model(model)),
         "--text-field", "combined_text", "--id-field", "openalex_id",
         "--prefix", "paper", "--model", model]),
        # — EMBED (encode each source separately, then merge policy) —
    ] + [
        (f"embed {corpus}", [
            sys.executable, "1_code/3_embed/0_embed_reference_and_policy_corpora.py",
            "--corpus", corpus, "--batch-size", EMBED_BATCH_SIZE,
        ] + model_args)
        for corpus in ALL_EMBED_CORPORA
    ] + [
        ("merge policy corpus", [sys.executable, "1_code/3_embed/1_merge_policy_corpus.py"] + model_args),
    ]
    for label, cmd in pre_steps:
        run_step(label, cmd)

    embed_cmd = [
        sys.executable,
        "1_code/3_embed/0_embed_paper_shards.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    embed_cmd.extend(model_args)
    run_step("embed paper shards", embed_cmd)

    _run_main_analysis_steps(output_dir, model=model)

    run_policy_source_family_sensitivity(output_dir, model=model)
    run_sdg4_lexical_audit(output_dir, model=model)
    run_loo_sdgi_circularity(output_dir, model=model)
    run_semantic_gap_interpretability(output_dir, model=model)
    run_sample_stability(output_dir, model=model)
    run_register_adjustment(output_dir, model=model)


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
    model_args = ["--model", model] if model_is_nondefault else []
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
        ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "preprocess":
        steps = [
            ("preprocess policy", [sys.executable, "1_code/1_preprocess/0_preprocess_policy.py"]),
            ("filter ungdc", [sys.executable, "1_code/1_preprocess/0_filter_ungdc_sdg.py"]),
            ("preprocess osdg", [sys.executable, "1_code/1_preprocess/0_preprocess_osdg.py"]),
            ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_benchmark.py"]),
            ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py"]),
            ("preprocess aurora", [sys.executable, "1_code/1_preprocess/0_preprocess_aurora.py"]),
            ("preprocess sdgi unified", [sys.executable, "1_code/1_preprocess/0_preprocess_sdgi_unified.py"]),
            ("preprocess research shards", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py"]),
        ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "segment":
        steps = [
            ("segment knowledge hub", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "sdg_knowledge_hub", "--model", model]),
            ("segment aurora", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "aurora", "--model", model]),
            ("segment policy scrape", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "policy_scrape", "--model", model]),
            ("segment policy manual", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "policy_manual", "--model", model]),
            ("segment ungdc", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "ungdc_sdg", "--model", model]),
            ("segment sdgi", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--corpus", "sdgi", "--model", model]),
            ("build policy corpus", [sys.executable, "1_code/1_preprocess/1_build_policy_corpus.py", "--model", model]),
            ("segment research corpus", [sys.executable, "1_code/2_segment/segment_corpus.py",
             "--sharded",
             "--input-glob", str(research_preprocessed_dir() / "part-*.jsonl"),
             "--output-dir", str(research_segmented_dir_for_model(model)),
             "--text-field", "combined_text", "--id-field", "openalex_id",
             "--prefix", "paper", "--model", model]),
        ]
        for label, cmd in steps:
            run_step(label, cmd)

    elif stage == "embed":
        for corpus in ALL_EMBED_CORPORA:
            run_step(
                f"embed {corpus}",
                [sys.executable, "1_code/3_embed/0_embed_reference_and_policy_corpora.py",
                 "--corpus", corpus, "--batch-size", EMBED_BATCH_SIZE] + model_args,
            )
        run_step(
            "merge policy corpus",
            [sys.executable, "1_code/3_embed/1_merge_policy_corpus.py"] + model_args,
        )
        embed_cmd = [
            sys.executable, "1_code/3_embed/0_embed_paper_shards.py",
            "--device", args.device, "--batch-size", str(args.batch_size),
        ]
        embed_cmd.extend(model_args)
        run_step("embed paper shards", embed_cmd)

    elif stage == "train":
        run_step("prepare training data", [sys.executable, "1_code/4_supervised_model_train/0_prepare_data.py", "--model", model])
        run_step("retrain full data", [sys.executable, "1_code/4_supervised_model_train/1_retrain_full_data.py", "--model", model])

    elif stage == "infer":
        run_step("score research shards", [sys.executable, "1_code/5_supervised_model_infer/0_score_research_shards.py", "--model", model])
        run_step("score policy corpus", [sys.executable, "1_code/5_supervised_model_infer/1_score_policy.py", "--model", model])

    elif stage == "centroids":
        run_step("check centroid consistency", [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--model", model])

    elif stage == "analysis":
        _run_main_analysis_steps(output_dir, model)
        run_policy_source_family_sensitivity(output_dir, model=model)
        run_sdg4_lexical_audit(output_dir, model=model)
        run_loo_sdgi_circularity(output_dir, model=model)
        run_semantic_gap_interpretability(output_dir, model=model)
        run_sample_stability(output_dir, model=model)
        run_register_adjustment(output_dir, model=model)

    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    embed_dir_for_model(args.embed_model)

    fetch_profile = resolve_fetch_snapshot_profile(args)

    if not action_requested(args):
        print_status(output_dir)
        return

    if (
        args.warm_replay_without_appendix
        or args.warm_replay_with_appendix
        or args.cold_replay
        or args.appendix_all
        or args.appendix_a2_family
        or args.appendix_a3_sdg4
        or args.appendix_a3b_circularity
        or args.appendix_b2_interpret
        or args.appendix_c_sample_stability
        or args.appendix_f_register
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
        run_policy_source_family_sensitivity(output_dir, model=model)
        run_sdg4_lexical_audit(output_dir, model=model)
        run_loo_sdgi_circularity(output_dir, model=model)
        run_semantic_gap_interpretability(output_dir, model=model)
        run_sample_stability(output_dir, model=model)
        run_register_adjustment(output_dir, model=model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a2_family:
        run_policy_source_family_sensitivity(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a3_sdg4:
        run_sdg4_lexical_audit(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a3b_circularity:
        run_loo_sdgi_circularity(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_b2_interpret:
        run_semantic_gap_interpretability(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_c_sample_stability:
        run_sample_stability(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_f_register:
        run_register_adjustment(output_dir, model=args.embed_model)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.warm_replay_without_appendix:
        ensure_warm_replay_inputs(args)
        run_warm_replay(output_dir, args, include_appendix=False)
    elif args.warm_replay_with_appendix:
        ensure_warm_replay_inputs(args)
        run_warm_replay(output_dir, args, include_appendix=True)
    elif args.build_pdf:
        build_pdf(output_dir)


if __name__ == "__main__":
    main()
