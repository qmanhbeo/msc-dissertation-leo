from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent / "1_code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared_utils import (
    canonical_artifact_paths,
    canonical_artifact_status,
    require_output_files,
    require_pdf_inputs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "4_outputs"

BASE_WARM_REPLAY_REQUIREMENTS = [
    Path("2_data/2_embedded/policy.npy"),
    Path("2_data/2_embedded/metadata/policy_ids.json"),
    Path("2_data/2_embedded/osdg.npy"),
    Path("2_data/2_embedded/metadata/osdg_ids.json"),
    Path("2_data/2_embedded/benchmark.npy"),
    Path("2_data/2_embedded/metadata/benchmark_ids.json"),
    Path("2_data/2_embedded/sdg_knowledge_hub.npy"),
    Path("2_data/2_embedded/metadata/sdg_knowledge_hub_ids.json"),
    Path("2_data/2_embedded/sdgi.npy"),
    Path("2_data/2_embedded/metadata/sdgi_ids.json"),
    Path("2_data/2_embedded/aurora.npy"),
    Path("2_data/2_embedded/metadata/aurora_ids.json"),
    Path("2_data/2_embedded/research_shards/metadata/manifest.json"),
    Path("2_data/3_scored/sdg_centroids.npy"),
    Path("2_data/3_scored/metadata/sdg_centroid_meta.json"),
    Path("2_data/3_scored/research_centroids.npy"),
    Path("2_data/3_scored/paper_scores_shards/metadata/manifest.json"),
    Path("2_data/3_scored/policy_scores.npy"),
    Path("2_data/3_scored/policy_scores_vs_research.npy"),
    Path("2_data/1_preprocessed/policy_all/policy_segments_all.jsonl"),
    Path("2_data/0_raw/policy_manual/artifact/convert_policy_manual_summary.json"),
    Path("3_writing/dissertation.tex"),
    Path("3_writing/references.bib"),
]

WARM_REPLAY_APPENDIX_EXTRA_REQUIREMENTS = [
    Path("2_data/1_preprocessed/research_corpus/metadata/manifest.json"),
    Path("2_data/1_preprocessed/research_corpus/part-00001.jsonl"),
]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical dissertation entrypoint. Default mode prints repo status only. "
            "Mutation requires an explicit action flag."
        )
    )
    p.add_argument(
        "--warm-replay",
        action="store_true",
        help=(
            "Rebuild main text analysis from frozen embeddings. "
            "Appendix outputs remain committed in the repo and are not regenerated. "
            "Auto-fetches the curated snapshot if 2_data/ is missing."
        ),
    )
    p.add_argument("--cold-replay", action="store_true", help="Full pipeline from live data sources — fetch, preprocess, embed, analyse. Not recommended (long runtime; OpenAlex live changes may break reproducibility).")
    p.add_argument("--appendix-all", action="store_true", help="Run all appendix stages (A1-A3, B1-B4, C, D) standalone (requires existing main-text outputs).")
    p.add_argument("--appendix-a1-source", action="store_true", help="Run A.1 Per-SDG Source Comparison.")
    p.add_argument("--appendix-a2-family", action="store_true", help="Run A.2 Policy Source-Family Sensitivity.")
    p.add_argument("--appendix-a3-sdg4", action="store_true", help="Run A.3 SDG 4 Lexical Artefact Audit.")
    p.add_argument("--appendix-b1-pca", action="store_true", help="Run B.1 Combined Research-Policy PCA Landscape.")
    p.add_argument("--appendix-b2-centroid", action="store_true", help="Run B.2 Within-Corpus Centroid Structure.")
    p.add_argument("--appendix-b3-interpret", action="store_true", help="Run B.3 Lexical Illustration of the Semantic Gap.")
    p.add_argument("--appendix-b4-softmax", action="store_true", help="Run B.4 Softmax Multi-label SDG.")
    p.add_argument("--appendix-d-register", action="store_true", help="Run D Register-Adjustment Robustness.")
    p.add_argument("--appendix-c-sample-stability", action="store_true", help="Run C Sample-Stability Robustness (appendix).")
    # Deprecated aliases (hidden, kept for backward compatibility)
    p.add_argument("--pca-semantic-landscape", action="store_true", dest="appendix_b1_pca", help=argparse.SUPPRESS)
    p.add_argument("--softmax-multilabel-sdg", action="store_true", dest="appendix_b4_softmax", help=argparse.SUPPRESS)
    p.add_argument("--policy-source-family-sensitivity", action="store_true", dest="appendix_a2_family", help=argparse.SUPPRESS)
    p.add_argument("--sdg4-lexical-audit", action="store_true", dest="appendix_a3_sdg4", help=argparse.SUPPRESS)
    p.add_argument("--sdg-source-comparison", action="store_true", dest="appendix_a1_source", help=argparse.SUPPRESS)
    p.add_argument("--semantic-gap-interpretability", action="store_true", dest="appendix_b3_interpret", help=argparse.SUPPRESS)
    p.add_argument("--within-corpus-centroid-structure", action="store_true", dest="appendix_b2_centroid", help=argparse.SUPPRESS)
    p.add_argument(
        "--fetch-data-snapshot",
        nargs="?",
        const="curated",
        choices=["curated", "full"],
        help=(
            "Fetch and extract a frozen dissertation data snapshot into ./2_data/. "
            "Defaults to curated; full is optional and audit-oriented."
        ),
    )
    p.add_argument(
        "--backup-data-snapshot",
        nargs="?",
        const="curated",
        choices=["curated", "full", "both"],
        help=(
            "Create a dissertation data snapshot archive via the backup utility. "
            "Defaults to curated; 'both' runs curated then full."
        ),
    )
    p.add_argument(
        "--register-adjustment",
        action="store_true",
        dest="appendix_d_register",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--skip-register-confidence-checks", action="store_true", help="Skip the additional register-confidence checks inside --register-adjustment.")
    p.add_argument(
        "--sdg-register-method",
        choices=["sdg_balanced", "within_sdg", "both"],
        default="both",
        help=(
            "Method subset for the SDG-aware register robustness checks inside --register-adjustment. "
            "The SDG-balanced method is a stronger global sensitivity check; the within-SDG method is an over-subtraction stress test."
        ),
    )
    p.add_argument("--sdg-register-random-seed", type=int, default=None, help="Optional seed override for the SDG-aware register robustness checks.")
    p.add_argument("--sdg-register-samples-per-cell", type=int, default=None, help="Optional cap for samples per SDG x register cell in the SDG-aware register robustness checks.")
    p.add_argument("--sdg-register-min-samples-per-class", type=int, default=50, help="Minimum per-class sample size required for a within-SDG classifier.")
    p.add_argument("--sdg-register-test-size", type=float, default=0.20, help="Held-out test fraction for the SDG-aware register robustness checks.")
    p.add_argument("--sdg-register-classifier-type", choices=["logistic_regression_liblinear", "logistic_regression_saga"], default="logistic_regression_liblinear", help="Linear classifier variant for the SDG-aware register robustness checks.")
    p.add_argument("--build-pdf", action="store_true", help="Build dissertation.pdf from existing manuscript outputs (requires bash — WSL/Linux only).")
    p.add_argument("--overwrite", action="store_true", help="Required before replacing existing manuscript outputs.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Manuscript output directory. Default: 4_outputs/")
    p.add_argument(
        "--snapshot-profile",
        choices=["curated", "full"],
        default="curated",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for embed_paper_shards.py in --cold-replay mode.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embed_paper_shards.py in --cold-replay mode.")
    p.add_argument("--local-files-only", action="store_true", help="Pass --local-files-only to embed_paper_shards.py in --cold-replay mode.")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def action_requested(args: argparse.Namespace) -> bool:
    return any(
        [
            args.warm_replay,
            args.cold_replay,
            args.appendix_all,
            args.appendix_a1_source,
            args.appendix_a2_family,
            args.appendix_a3_sdg4,
            args.appendix_b1_pca,
            args.appendix_b2_centroid,
            args.appendix_b3_interpret,
            args.appendix_b4_softmax,
            args.appendix_d_register,
            args.appendix_c_sample_stability,
            args.fetch_data_snapshot,
            args.build_pdf,
        ]
    )


def run_step(label: str, cmd: list[str], step_id: str | None = None) -> None:
    header = f"[{step_id}] {label}" if step_id else f"[{label}]"
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {header}")
    print(sep)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print()


def missing_requirements(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not (ROOT / p).exists()]


def required_warm_replay_inputs(*, include_appendix_extra: bool) -> list[Path]:
    required = list(BASE_WARM_REPLAY_REQUIREMENTS)
    if include_appendix_extra:
        required.extend(WARM_REPLAY_APPENDIX_EXTRA_REQUIREMENTS)
    return required


def missing_research_text_shards() -> list[Path]:
    manifest_path = ROOT / "2_data/1_preprocessed/research_corpus/metadata/manifest.json"
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
        data_path = shard.get("data_path")
        ids_path = shard.get("ids_path")
        for value in (data_path, ids_path):
            if isinstance(value, str):
                path = ROOT / value
                if not path.exists():
                    missing.append(path.relative_to(ROOT))
    return missing


def missing_manifest_shard_paths(manifest_path: Path, shard_fields: tuple[str, ...]) -> list[Path]:
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


def missing_warm_replay_requirements(*, include_appendix_extra: bool) -> list[Path]:
    missing = missing_requirements(required_warm_replay_inputs(include_appendix_extra=include_appendix_extra))
    for path in missing_manifest_shard_paths(
        ROOT / "2_data/2_embedded/research_shards/metadata/manifest.json",
        ("embedding_path", "ids_path"),
    ):
        if path not in missing:
            missing.append(path)
    if include_appendix_extra:
        for path in missing_research_text_shards():
            if path not in missing:
                missing.append(path)
    return missing


def canonical_exists(output_dir: Path) -> bool:
    return any(path.exists() for path in canonical_artifact_paths(output_dir))


def print_status(output_dir: Path) -> None:
    print(f"Project root: {ROOT}")
    print(f"Manuscript output dir: {output_dir}")

    warm_missing = missing_warm_replay_requirements(include_appendix_extra=False)
    print("")
    print(f"Warm replay readiness: {'yes' if not warm_missing else 'no'}")
    if warm_missing:
        for path in warm_missing:
            print(f"  missing: {rel(ROOT / path)}")

    warm_appendix_missing = missing_warm_replay_requirements(include_appendix_extra=True)
    print("")
    print(f"Warm replay + appendix readiness: {'yes' if not warm_appendix_missing else 'no'}")
    if warm_appendix_missing:
        for path in warm_appendix_missing[:12]:
            print(f"  missing: {rel(ROOT / path)}")
        if len(warm_appendix_missing) > 12:
            print(f"  ... and {len(warm_appendix_missing) - 12} more")

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


def run_sample_stability(output_dir: Path) -> None:
    require_output_files(
        output_dir / "main" / "data",
        [
            "4_2_coverage_document_weighted.json",
            "4_3_semantic_gap_distances.json",
            "4_4_interaction_correlation_asymmetry.json",
        ],
    )
    run_step(
        "sample stability",
        [sys.executable, "1_code/3_main_analysis/3_appendix/6_sample_stability.py", "--output-dir", str(output_dir)],
        step_id="C",
    )


def run_pca_semantic_landscape(output_dir: Path) -> None:
    run_step(
        "combined research-policy PCA landscape",
        [sys.executable, "1_code/3_main_analysis/3_appendix/0_pca_semantic_landscape.py", "--output-dir", str(output_dir)],
        step_id="B1",
    )


def run_within_corpus_centroid_structure(output_dir: Path) -> None:
    run_step(
        "within-corpus centroid structure",
        [sys.executable, "1_code/3_main_analysis/3_appendix/1_within_corpus_centroid_structure.py", "--output-dir", str(output_dir)],
        step_id="B2",
    )


def run_softmax_multilabel_sdg(output_dir: Path) -> None:
    run_step(
        "softmax multi-label SDG robustness",
        [sys.executable, "1_code/3_main_analysis/3_appendix/2_softmax_multilabel_sdg.py", "--output-dir", str(output_dir)],
        step_id="B4",
    )


def run_policy_source_family_sensitivity(output_dir: Path) -> None:
    run_step(
        "policy source-family sensitivity",
        [sys.executable, "1_code/3_main_analysis/3_appendix/4_policy_source_family_sensitivity.py", "--output-dir", str(output_dir)],
        step_id="A2",
    )


def run_sdg4_lexical_audit(output_dir: Path) -> None:
    run_step(
        "SDG 4 lexical artefact audit",
        [sys.executable, "1_code/3_main_analysis/3_appendix/5_sdg4_lexical_audit.py", "--output-dir", str(output_dir)],
        step_id="A3",
    )


def run_sdg_source_comparison(output_dir: Path) -> None:
    run_step(
        "per-SDG source comparison",
        [sys.executable, "1_code/3_main_analysis/3_appendix/8_sdg_source_comparison.py", "--output-dir", str(output_dir)],
        step_id="A1",
    )


def run_semantic_gap_interpretability(output_dir: Path) -> None:
    require_output_files(output_dir / "main" / "data", ["4_3_semantic_gap_distances.json"])
    run_step(
        "lexical illustration of the semantic gap",
        [sys.executable, "1_code/3_main_analysis/3_appendix/7_semantic_gap_text_interpretability.py", "--output-dir", str(output_dir)],
        step_id="B3",
    )


def run_register_adjustment(output_dir: Path, args: argparse.Namespace, *, include_register_confidence_checks: bool) -> None:
    cmd = [sys.executable, "1_code/3_main_analysis/3_appendix/3_register_adjustment.py", "--output-dir", str(output_dir)]
    if not include_register_confidence_checks:
        cmd.append("--skip-register-confidence-checks")
    cmd.extend(["--method", args.sdg_register_method])
    cmd.extend(["--test-size", str(args.sdg_register_test_size)])
    cmd.extend(["--classifier-type", args.sdg_register_classifier_type])
    cmd.extend(["--min-samples-per-class", str(args.sdg_register_min_samples_per_class)])
    if args.sdg_register_random_seed is not None:
        cmd.extend(["--random-seed", str(args.sdg_register_random_seed)])
    if args.sdg_register_samples_per_cell is not None:
        cmd.extend(["--samples-per-cell", str(args.sdg_register_samples_per_cell)])
    run_step(
        "register-adjustment robustness",
        cmd,
        step_id="D",
    )


def run_main_text(
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    missing = missing_warm_replay_requirements(include_appendix_extra=False)
    if missing:
        missing_str = ", ".join(rel(ROOT / p) for p in missing)
        raise RuntimeError(f"Main text replay is not ready. Missing required inputs: {missing_str}")

    run_step("rebuild sdg centroids", [sys.executable, "1_code/2_embed/reference/1_build_sdg_centroids.py"], step_id="1")
    run_step("validate centroids", [sys.executable, "1_code/2_embed/reference/2_validate_centroids.py", "--output-dir", str(output_dir)], step_id="2")
    run_step("rebuild research centroids", [sys.executable, "1_code/2_embed/research/1_score_paper_shards.py"], step_id="3")
    run_step("score policy corpus", [sys.executable, "1_code/2_embed/policy/0_score_policy_corpus.py"], step_id="4")
    run_step("coverage gap", [sys.executable, "1_code/3_main_analysis/1_canonical/0_coverage_gap.py", "--output-dir", str(output_dir)], step_id="5")
    run_step("semantic gap", [sys.executable, "1_code/3_main_analysis/1_canonical/1_semantic_gap.py", "--output-dir", str(output_dir)], step_id="6")
    run_step(
        "coverage semantic interaction",
        [sys.executable, "1_code/3_main_analysis/1_canonical/2_coverage_semantic_interaction.py", "--output-dir", str(output_dir)],
        step_id="7",
    )
    run_step("plot figures", [sys.executable, "1_code/4_visualization/plot_figures.py", "--output-dir", str(output_dir)], step_id="8")


def run_warm_replay(
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    run_main_text(output_dir, args)
    print(
        "Main text outputs rebuilt. To build the dissertation PDF, run:\n"
        "  python main.py --build-pdf --overwrite\n"
        "Note: --build-pdf requires bash (WSL/Linux) and is not supported on bare Windows."
    )


def run_cold_replay(output_dir: Path, args: argparse.Namespace) -> None:
    print("WARNING: live-source --cold-replay reruns are not expected to be identical to the frozen data snapshot.")
    print("WARNING: OpenAlex updates over time, policy source links may drift, and the manual policy supplement is not fully automatable from stable URLs.")
    pre_steps = [
        ("fetch policy", [sys.executable, "1_code/0_fetch/fetch_policy.py"]),
        ("convert policy manual", [sys.executable, "1_code/0_fetch/convert_policy_manual.py"]),
        ("fetch sdgi corpus", [sys.executable, "1_code/0_fetch/fetch_sdgi_corpus.py"]),
        ("fetch ungdc", [sys.executable, "1_code/0_fetch/fetch_ungdc.py"]),
        ("preprocess policy", [sys.executable, "1_code/1_preprocess/policy/0_preprocess_policy.py"]),
        ("integrate sdgi", [sys.executable, "1_code/1_preprocess/policy/0_integrate_sdgi.py"]),
        ("filter ungdc", [sys.executable, "1_code/1_preprocess/policy/0_filter_ungdc_sdg.py"]),
        ("build policy corpus", [sys.executable, "1_code/1_preprocess/policy/1_build_policy_corpus.py"]),
        ("fetch osdg", [sys.executable, "1_code/0_fetch/fetch_osdg.py"]),
        ("fetch sdg benchmark", [sys.executable, "1_code/0_fetch/fetch_sdg_benchmark.py"]),
        ("preprocess osdg", [sys.executable, "1_code/1_preprocess/preprocess_osdg.py"]),
        ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/preprocess_sdg_benchmark.py"]),
        ("fetch sdg knowledge hub", [sys.executable, "1_code/0_fetch/fetch_sdg_knowledge_hub.py"]),
        ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/preprocess_sdg_knowledge_hub.py"]),
        ("preprocess sdgi for embedding", [sys.executable, "1_code/1_preprocess/preprocess_sdgi_corpus.py"]),
        ("fetch aurora", [sys.executable, "1_code/0_fetch/fetch_aurora.py"]),
        (
            "embed reference corpora",
            [
                sys.executable,
                "1_code/2_embed/reference/0_embed_reference_corpora.py",
                "--corpora", "policy", "osdg", "benchmark", "sdg_knowledge_hub", "sdgi", "aurora",
                *(["--local-files-only"] if args.local_files_only else []),
            ],
        ),
        ("fetch openalex", [sys.executable, "1_code/0_fetch/fetch_openalex.py"]),
        ("preprocess research shards", [sys.executable, "1_code/1_preprocess/preprocess_papers_streaming.py"]),
    ]
    for label, cmd in pre_steps:
        run_step(label, cmd)

    embed_cmd = [
        sys.executable,
        "1_code/2_embed/research/0_embed_paper_shards.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.local_files_only:
        embed_cmd.append("--local-files-only")
    run_step("embed paper shards", embed_cmd)

    run_main_text(output_dir, args)
    run_pca_semantic_landscape(output_dir)
    run_within_corpus_centroid_structure(output_dir)
    run_softmax_multilabel_sdg(output_dir)
    run_policy_source_family_sensitivity(output_dir)
    run_sdg4_lexical_audit(output_dir)
    run_sdg_source_comparison(output_dir)
    run_semantic_gap_interpretability(output_dir)
    run_sample_stability(output_dir)
    run_register_adjustment(output_dir, args, include_register_confidence_checks=not args.skip_register_confidence_checks)


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
    if explicit_profile != legacy_profile and legacy_profile != "curated":
        raise RuntimeError(
            "Conflicting fetch snapshot profiles. Use either `--fetch-data-snapshot <profile>` "
            "or the legacy `--snapshot-profile <profile>`, but not different values for both."
        )
    return explicit_profile


def selected_backup_profiles(profile_name: str) -> list[str]:
    if profile_name == "both":
        return ["curated", "full"]
    return [profile_name]


def ensure_warm_replay_inputs(args: argparse.Namespace, *, include_appendix_extra: bool) -> None:
    missing = missing_warm_replay_requirements(include_appendix_extra=include_appendix_extra)
    if not missing:
        return

    missing_str = ", ".join(rel(ROOT / p) for p in missing[:12])
    print(f"[info] warm replay inputs missing: {missing_str}")
    if len(missing) > 12:
        print(f"[info] ... and {len(missing) - 12} more")
    run_fetch_data_snapshot(args, profile_name="curated", overwrite_data=(ROOT / "2_data").exists())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    fetch_profile = resolve_fetch_snapshot_profile(args)

    if not action_requested(args):
        print_status(output_dir)
        return

    if (
        args.warm_replay
        or args.cold_replay
        or args.appendix_all
        or args.appendix_a1_source
        or args.appendix_a2_family
        or args.appendix_a3_sdg4
        or args.appendix_b1_pca
        or args.appendix_b2_centroid
        or args.appendix_b3_interpret
        or args.appendix_b4_softmax
        or         args.appendix_d_register
        or args.appendix_c_sample_stability
        or args.build_pdf
    ) and canonical_exists(output_dir) and not args.overwrite:
        print("Outputs already exist — use --overwrite to replace them.", file=sys.stderr)
        sys.exit(1)

    if fetch_profile is not None:
        run_fetch_data_snapshot(args, profile_name=fetch_profile, overwrite_data=args.overwrite)
    elif args.backup_data_snapshot:
        for profile_name in selected_backup_profiles(args.backup_data_snapshot):
            run_backup_data_snapshot(profile_name=profile_name)
    elif args.cold_replay:
        run_cold_replay(output_dir, args)
    elif args.appendix_all:
        run_sdg_source_comparison(output_dir)
        run_policy_source_family_sensitivity(output_dir)
        run_sdg4_lexical_audit(output_dir)
        run_pca_semantic_landscape(output_dir)
        run_within_corpus_centroid_structure(output_dir)
        run_semantic_gap_interpretability(output_dir)
        run_softmax_multilabel_sdg(output_dir)
        run_sample_stability(output_dir)
        run_register_adjustment(output_dir, args, include_register_confidence_checks=not args.skip_register_confidence_checks)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a1_source:
        run_sdg_source_comparison(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a2_family:
        run_policy_source_family_sensitivity(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_a3_sdg4:
        run_sdg4_lexical_audit(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_b1_pca:
        run_pca_semantic_landscape(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_b2_centroid:
        run_within_corpus_centroid_structure(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_b3_interpret:
        run_semantic_gap_interpretability(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_b4_softmax:
        run_softmax_multilabel_sdg(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_d_register:
        run_register_adjustment(output_dir, args, include_register_confidence_checks=not args.skip_register_confidence_checks)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.appendix_c_sample_stability:
        run_sample_stability(output_dir)
        if args.build_pdf:
            build_pdf(output_dir)
    elif args.warm_replay:
        ensure_warm_replay_inputs(args, include_appendix_extra=False)
        run_warm_replay(output_dir, args)
    elif args.build_pdf:
        build_pdf(output_dir)


if __name__ == "__main__":
    main()
