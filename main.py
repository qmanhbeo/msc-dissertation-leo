from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from shared_utils import canonical_artifact_paths, canonical_artifact_status


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

WARM_REPLAY_REQUIREMENTS = [
    Path("data/2_embedded/policy.npy"),
    Path("data/2_embedded/metadata/policy_ids.json"),
    Path("data/2_embedded/osdg.npy"),
    Path("data/2_embedded/benchmark.npy"),
    Path("data/2_embedded/research_shards/metadata/manifest.json"),
    Path("data/3_scored/sdg_centroids.npy"),
    Path("data/3_scored/paper_scores_shards/metadata/manifest.json"),
    Path("data/1_preprocessed/policy_all/policy_chunks_extended.jsonl"),
    Path("writing/dissertation.tex"),
    Path("writing/references.bib"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical dissertation entrypoint. Default mode prints repo status only. "
            "Mutation requires an explicit action flag."
        )
    )
    p.add_argument("--warm-replay", action="store_true", help="Rebuild canonical analysis outputs and PDF from existing data/.")
    p.add_argument("--full-pipeline", action="store_true", help="Run the full active pipeline facade from fetch through PDF.")
    p.add_argument("--build-pdf", action="store_true", help="Build outputs/dissertation.pdf from existing canonical tables/figures.")
    p.add_argument("--clean-canon", action="store_true", help="Remove canonical outputs/ artifacts only.")
    p.add_argument("--overwrite", action="store_true", help="Required before replacing existing canonical outputs.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Canonical output directory. Default: outputs/")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for embed_paper_shards.py in --full-pipeline mode.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embed_paper_shards.py in --full-pipeline mode.")
    p.add_argument("--local-files-only", action="store_true", help="Pass --local-files-only to embed_paper_shards.py in --full-pipeline mode.")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def action_requested(args: argparse.Namespace) -> bool:
    return any([args.warm_replay, args.full_pipeline, args.build_pdf, args.clean_canon])


def run_step(label: str, cmd: list[str]) -> None:
    print(f"[run] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def missing_requirements(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not (ROOT / p).exists()]


def canonical_exists(output_dir: Path) -> bool:
    return any(path.exists() for path in canonical_artifact_paths(output_dir))


def remove_if_exists(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def clean_canonical_outputs(output_dir: Path) -> None:
    for path in canonical_artifact_paths(output_dir):
        if path.exists():
            remove_if_exists(path)
    for subdir in [output_dir / "tables", output_dir / "figures"]:
        if subdir.exists():
            try:
                subdir.rmdir()
            except OSError:
                pass


def print_status(output_dir: Path) -> None:
    print(f"Project root: {ROOT}")
    print(f"Canonical output dir: {output_dir}")

    warm_missing = missing_requirements(WARM_REPLAY_REQUIREMENTS)
    print("")
    print("Warm replay readiness:")
    if warm_missing:
        print("  ready: no")
        for path in warm_missing:
            print(f"  missing: {rel(ROOT / path)}")
    else:
        print("  ready: yes")

    status = canonical_artifact_status(output_dir)
    print("")
    print("Canonical output status:")
    print(f"  present: {len(status['present'])}")
    print(f"  missing: {len(status['missing'])}")
    if status["present"]:
        print("  sample present:")
        for item in status["present"][:6]:
            print(f"    - {item}")
    if status["missing"]:
        print("  sample missing:")
        for item in status["missing"][:6]:
            print(f"    - {item}")

    tex = (ROOT / "writing" / "dissertation.tex").read_text(encoding="utf-8")
    legacy_markers = [
        "../data/generated/",
        "_legacy/",
        "num_context.tex",
        "num_sdg4.tex",
        "tab_sdgindex.tex",
        "AnalysisSnapshotDate",
    ]
    active_only = not any(marker in tex for marker in legacy_markers)
    print("")
    print("Manuscript active-only contract:")
    print(f"  clean: {'yes' if active_only else 'no'}")
    if not active_only:
        for marker in legacy_markers:
            if marker in tex:
                print(f"  found legacy marker: {marker}")


def build_pdf(output_dir: Path) -> None:
    run_step(
        "build pdf",
        ["bash", str(ROOT / "writing" / "build_pdf.sh"), str(output_dir / "dissertation.pdf")],
    )


def run_warm_replay(output_dir: Path) -> None:
    missing = missing_requirements(WARM_REPLAY_REQUIREMENTS)
    if missing:
        missing_str = ", ".join(rel(ROOT / p) for p in missing)
        raise RuntimeError(f"Warm replay is not ready. Missing required inputs: {missing_str}")

    run_step("rebuild sdg centroids", [sys.executable, "code/2_embed/sdg_centroids.py"])
    run_step("validate centroids", [sys.executable, "code/2_embed/validate_centroids.py", "--output-dir", str(output_dir)])
    run_step("rebuild research centroids", [sys.executable, "code/2_embed/score_paper_shards.py"])
    run_step("score policy corpus", [sys.executable, "code/2_embed/score_policy_corpus.py"])
    run_step("coverage gap", [sys.executable, "code/3_main_analysis/coverage_gap.py", "--output-dir", str(output_dir)])
    run_step("semantic gap", [sys.executable, "code/3_main_analysis/semantic_gap.py", "--output-dir", str(output_dir)])
    run_step(
        "coverage semantic interaction",
        [sys.executable, "code/3_main_analysis/coverage_semantic_interaction.py", "--output-dir", str(output_dir)],
    )
    run_step("plot figures", [sys.executable, "code/4_visualization/plot_figures.py", "--output-dir", str(output_dir)])
    build_pdf(output_dir)


def run_full_pipeline(output_dir: Path, args: argparse.Namespace) -> None:
    pre_steps = [
        ("fetch un sdg", [sys.executable, "code/0_fetch/fetch_un_sdg.py"]),
        ("fetch policy", [sys.executable, "code/0_fetch/fetch_policy.py"]),
        ("fetch sdgi corpus", [sys.executable, "code/0_fetch/fetch_sdgi_corpus.py"]),
        ("fetch ungdc", [sys.executable, "code/0_fetch/fetch_ungdc.py"]),
        ("preprocess policy", [sys.executable, "code/1_preprocess/preprocess_policy.py"]),
        ("integrate sdgi", [sys.executable, "code/1_preprocess/integrate_sdgi.py"]),
        ("filter ungdc", [sys.executable, "code/1_preprocess/filter_ungdc_sdg.py"]),
        ("build policy corpus", [sys.executable, "code/1_preprocess/build_policy_corpus.py"]),
        ("fetch osdg", [sys.executable, "code/0_fetch/fetch_osdg.py"]),
        ("fetch sdg benchmark", [sys.executable, "code/0_fetch/fetch_sdg_benchmark.py"]),
        ("preprocess osdg", [sys.executable, "code/1_preprocess/preprocess_osdg.py"]),
        ("preprocess sdg benchmark", [sys.executable, "code/1_preprocess/preprocess_sdg_benchmark.py"]),
        ("embed policy/osdg/benchmark", [sys.executable, "code/2_embed/embeddings.py"]),
        ("fetch openalex", [sys.executable, "code/0_fetch/fetch_openalex.py"]),
        ("preprocess research shards", [sys.executable, "code/1_preprocess/preprocess_papers_streaming.py"]),
    ]
    for label, cmd in pre_steps:
        run_step(label, cmd)

    embed_cmd = [
        sys.executable,
        "code/2_embed/embed_paper_shards.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.local_files_only:
        embed_cmd.append("--local-files-only")
    run_step("embed paper shards", embed_cmd)

    run_warm_replay(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    if not action_requested(args):
        print_status(output_dir)
        return

    if args.clean_canon and not args.overwrite:
        raise RuntimeError("--clean-canon requires --overwrite.")

    if (args.warm_replay or args.full_pipeline or args.build_pdf) and canonical_exists(output_dir) and not args.overwrite:
        raise RuntimeError(
            "Canonical outputs already exist. Re-run with --overwrite to replace them, "
            "or run without action flags to inspect status."
        )

    if args.clean_canon:
        clean_canonical_outputs(output_dir)

    if args.full_pipeline:
        run_full_pipeline(output_dir, args)
    elif args.warm_replay:
        run_warm_replay(output_dir)
    elif args.build_pdf:
        build_pdf(output_dir)


if __name__ == "__main__":
    main()
