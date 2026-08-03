"""Dissertation reproducibility pipeline — single entrypoint.

Architecture — three method axes sharing a unified preprocess->segment->embed stage:

  Axis A — Supervised LR (PRIMARY result):
    prepare_data -> retrain LR -> score_supervised --lr --research
    -> supervised research_centroids.npy -> 1_semantic_gap, 0_coverage_gap

  Axis B — Supervised MLP (sensitivity):
    retrain MLP -> score_supervised --mlp
    -> mlp_research_centroids.npy -> cross-sensitivity table

  Axis C — Zeroshot nearest-centroid (sensitivity):
    build_sdg_reference_centroids -> sdg_centroids.npy
    -> score_zeroshot -> research_centroids.npy, policy_centroids.npy
    -> cross-sensitivity table

Labeled corpora: osdg, benchmark, sdg_knowledge_hub, sdgi, aurora
  (consolidated into reference corpus at preprocess time).
Unlabeled: research (OpenAlex), policy (consolidated from policy_scrape,
  policy_manual, ungdc_sdg, sdgi).
sdgi is dual-role: labeled training corpus (in reference) AND
  unlabeled policy corpus (in policy).
SciBERT reuses MPNet segmented texts (--seg-model all-mpnet-base-v2).

Build-order: 0_prepare_data MUST precede both build_reference_centroids
and retrain_full_data (both read prepare_data's output files).

Orchestration — replays are compositions, not bespoke scripts:

  Warm replay  : run_main_text  -> _run_main_analysis_steps(model)
                                      + _run_analysis_poststeps(model)
  Cold replay  : pre_steps (preprocess+segment via builders)
                  -> per-model _embed_model_steps + _run_main_analysis_steps
                  -> _run_analysis_poststeps once (after the encoder loop)
  --stage      : preprocess/segment/embed/train/infer/centroids/register_adjust
                 delegate to the same shared builders; `--stage analysis`
                 composes _run_main_analysis_steps for --embed-model only, then
                 _run_analysis_poststeps.

The cross-sensitivity table + figures are produced exactly once by
_run_analysis_poststeps (gated to the default model); they are intentionally
NOT part of _run_main_analysis_steps, so no consumer double-runs them.
Appendices are driven entirely by the APPENDIX_SPECS registry in
analysis_orchestrator.py — adding an appendix means adding one spec entry,
never re-wiring dispatch.
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
    COLD_REPLAY_MODELS,
    DEFAULT_EMBED_MODEL,
    embed_dir_for_model,
    embed_research_dir_for_model,
    output_dir_for_model,
    raw_dir,
    research_preprocessed_dir,
    research_segmented_dir_for_model,
    scored_dir_for_model,
    segmented_dir_for_model,
    resolve_model_alias,
)
from analysis_orchestrator import run_analysis, run_post_adjusted, APPENDIX_SPECS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "4_outputs"

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
    p.add_argument("--appendix-all", action="store_true", help="Run all appendix stages (A2, A3, B2, C, C1, C0, D1, H.1, I.1) standalone (requires existing main-text outputs).")
    # Appendix identities are registry-driven (analysis_orchestrator.APPENDIX_SPECS);
    # the deprecated aliases below are hidden and preserved for backward compatibility.
    for _spec in APPENDIX_SPECS:
        p.add_argument(f"--{_spec['flag']}", action="store_true", help=_spec["help"])
        for _alias in _spec.get("aliases", []):
            p.add_argument(f"--{_alias}", action="store_true", dest=_spec["flag"].replace("-", "_"), help=argparse.SUPPRESS)
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
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for ALL embedding stages (global fallback). Overridden per "
                        "group by --embed-batch-size-policy-and-reference / --embed-batch-size-research.")
    p.add_argument("--embed-batch-size-policy-and-reference", type=int, default=None,
                   help="Batch size for reference + policy embedding (overrides --batch-size).")
    p.add_argument("--embed-batch-size-research", type=int, default=None,
                   help="Batch size for research + research_concept embedding (overrides --batch-size).")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                   help="Compute precision for embedding (fp16 = 2x faster on Ampere GPUs).")
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer model name (default: %(default)s). Override for model sensitivity.",
    )
    p.add_argument(
        "--stage",
        choices=["fetch", "preprocess", "segment", "embed", "train", "infer", "centroids", "register_adjust", "analysis"],
        help="Run a single pipeline stage (assumes upstream outputs exist).",
    )
    p.add_argument("--corpus",
                   choices=["all", "reference", "policy", "research", "research_concept"],
                   default="all",
                   help="Corpus to segment (default: all; only used with --stage segment).")
    p.add_argument("--segment-workers", type=int, default=0,
                   help="Worker processes for sharded segmentation (default: respectful cap; "
                        "passed through to 1_code/2_segment/1_segment_corpus.py --workers).")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def action_requested(args: argparse.Namespace) -> bool:
    if args.stage:
        return True
    return any(
        [
            args.warm_replay_without_appendix,
            args.warm_replay_with_appendix,
            args.cold_replay,
            args.appendix_all,
            args.fetch_data_snapshot,
            args.backup_data_snapshot,
            args.build_pdf,
        ]
        + [getattr(args, spec["flag"].replace("-", "_")) for spec in APPENDIX_SPECS]
    )


def run_step(label: str, cmd: list[str], step_id: str | None = None, *, model: str = "") -> None:
    model_tag = f" [{model}]" if model else ""
    header = f"[{step_id}]{model_tag} {label}" if step_id else f"[{label}]{model_tag}"
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


def run_appendix_spec(spec: dict, output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    """Run a single appendix script from APPENDIX_SPECS.

    Registry-driven replacement for the 10 previously hand-written `run_*`
    wrappers. Reproduces their exact subprocess command (script path, --output-dir,
    optional --embed-model, --overwrite) so behaviour is unchanged.
    """
    _warn_non_default_model(model, spec["warn"])
    if spec.get("requires"):
        require_output_files(
            output_dir_for_model(model, root=output_dir) / "data",
            spec["requires"],
        )
    cmd = [sys.executable, "1_code/7_main_analysis/" + spec["script"], "--output-dir", str(output_dir)]
    if model != DEFAULT_EMBED_MODEL:
        cmd += ["--embed-model", model]
    cmd += _overwrite_flag(overwrite)
    run_step(spec["run_label"], cmd, step_id=spec["step_id"])


































def _overwrite_flag(overwrite: bool) -> list[str]:
    return ["--overwrite"] if overwrite else []


def _reset_flag(overwrite: bool) -> list[str]:
    """Preprocess scripts are resume-safe; --overwrite forces a clean rebuild."""
    return ["--reset"] if overwrite else []


def _warn_non_default_model(model: str, label: str) -> None:
    if model != DEFAULT_EMBED_MODEL:
        print(
            f"[{model}] is not the default model ({DEFAULT_EMBED_MODEL}). "
            f"'{label}' outputs are computed but NOT consumed by "
            "the paper's main tables or figures.",
            file=sys.stderr,
        )


def run_build_sdg_reference_centroids(model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    run_step(
        "build SDG reference centroids",
        [sys.executable, "1_code/6_calculate_centroids/0_build_sdg_reference_centroids.py",
         "--embed-model", model] + _overwrite_flag(overwrite),
        step_id="0a",
        model=model,
    )


def run_build_centroid_similarity_matrix(output_dir: Path, model: str = DEFAULT_EMBED_MODEL, overwrite: bool = False) -> None:
    run_step(
        "build centroid similarity matrix",
        [sys.executable, "1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py",
         "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(overwrite),
        step_id="9a",
        model=model,
    )


def _preprocess_steps(overwrite: bool) -> list[tuple[str, list[str]]]:
    reset = _reset_flag(overwrite)
    ow = _overwrite_flag(overwrite)
    return [
        ("preprocess policy", [sys.executable, "1_code/1_preprocess/0_preprocess_policy.py"] + reset),
        ("preprocess ungdc", [sys.executable, "1_code/1_preprocess/0_preprocess_ungdc_sdg.py"] + reset),
        ("preprocess osdg", [sys.executable, "1_code/1_preprocess/0_preprocess_osdg.py"] + reset),
        ("preprocess sdg benchmark", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_benchmark.py"] + reset),
        ("preprocess sdg knowledge hub", [sys.executable, "1_code/1_preprocess/0_preprocess_sdg_knowledge_hub.py"] + reset),
        ("preprocess aurora", [sys.executable, "1_code/1_preprocess/0_preprocess_aurora.py"] + reset),
        ("preprocess sdgi unified", [sys.executable, "1_code/1_preprocess/0_preprocess_sdgi_unified.py"] + reset),
        ("preprocess research shards", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py"] + reset),
        ("preprocess concept corpus", [sys.executable, "1_code/1_preprocess/0_preprocess_papers_streaming.py", "--retrieval", "concept"] + reset),
        ("build reference corpus", [sys.executable, "1_code/1_preprocess/1_build_reference_corpus.py"] + ow),
        ("build policy corpus", [sys.executable, "1_code/1_preprocess/1_build_policy_corpus.py"] + ow),
    ]


def _segment_steps(corpus: str, overwrite: bool, *, segment_workers: int = 0) -> list[tuple[str, list[str]]]:
    ow = _overwrite_flag(overwrite)
    worker_args = ["--workers", str(segment_workers)] if segment_workers and segment_workers > 0 else []
    if corpus == "all":
        steps = [
            ("segment reference & policy", [sys.executable, "1_code/2_segment/1_segment_corpus.py", "--all", "--embed-model", CANONICAL_SEGMENT_MODEL] + worker_args + ow),
            ("segment research corpus", [sys.executable, "1_code/2_segment/1_segment_corpus.py", "--corpus", "research", "--embed-model", CANONICAL_SEGMENT_MODEL] + worker_args + ow),
            ("segment concept research corpus", [sys.executable, "1_code/2_segment/1_segment_corpus.py", "--corpus", "research_concept", "--embed-model", CANONICAL_SEGMENT_MODEL] + worker_args + ow),
            ("build research 50k subset", [sys.executable, "1_code/2_segment/2_sample_segments.py"] + ow),
        ]
    elif corpus == "research":
        steps = [
            ("segment research corpus", [sys.executable, "1_code/2_segment/1_segment_corpus.py", "--corpus", "research", "--embed-model", CANONICAL_SEGMENT_MODEL] + worker_args + ow),
            ("build research 50k subset", [sys.executable, "1_code/2_segment/2_sample_segments.py"] + ow),
        ]
    else:
        steps = [
            (f"segment {corpus}", [sys.executable, "1_code/2_segment/1_segment_corpus.py", "--corpus", corpus, "--embed-model", CANONICAL_SEGMENT_MODEL] + worker_args + ow),
        ]
    return steps


def _embed_model_steps(
    model: str,
    *,
    overwrite: bool,
    batch_size: int | None,
    embed_batch_size_policy_and_reference: int | None,
    embed_batch_size_research: int | None,
    device: str,
    precision: str,
) -> list[tuple[str, list[str]]]:
    ow = _overwrite_flag(overwrite)
    model_args = ["--embed-model", model]
    # Per-group batch sizes; --batch-size is the global fallback. Defaults
    # preserve current behaviour (policy/reference=64, research=128) so nothing
    # is silently changed for a fresh clone.
    pol_ref_bs = embed_batch_size_policy_and_reference or batch_size or 64
    research_bs = embed_batch_size_research or batch_size or 128
    steps: list[tuple[str, list[str]]] = []
    for corpus in ALL_EMBED_CORPORA:
        steps.append((f"embed {corpus} ({model})", [
            sys.executable, "1_code/3_embed/0_embed_reference_and_policy_corpora.py",
            "--corpus", corpus, "--batch-size", str(pol_ref_bs),
            "--local-files-only", "--precision", precision, "--normalize-embeddings",
        ] + model_args + ow))
    embed_cmd = [
        sys.executable, "1_code/3_embed/0_embed_paper_shards.py",
        "--device", device, "--local-files-only", "--precision", precision,
        "--normalize-embeddings", "--batch-size", str(research_bs),
    ] + model_args + ow
    if model != CANONICAL_SEGMENT_MODEL:
        embed_cmd += ["--corpus", "research_subset"]
    steps.append((f"embed paper shards ({model})", embed_cmd))
    if model == CANONICAL_SEGMENT_MODEL:
        steps.append((f"embed concept research corpus ({model})", [
            sys.executable, "1_code/3_embed/0_embed_paper_shards.py",
            "--corpus", "research_concept", "--device", device,
            "--local-files-only", "--precision", precision, "--normalize-embeddings",
            "--batch-size", str(research_bs), "--embed-model", model,
        ] + ow))
    return steps


def _concept_retrieval_paths(model: str, output_dir: Path) -> dict:
    return dict(
        concept_embed_dir=embed_dir_for_model(model) / "research_concept",
        concept_scores_dir=scored_dir_for_model(model) / "paper_scores_shards_concept",
        concept_data_dir=output_dir_for_model(model, root=output_dir) / "data" / "concept",
        concept_centroids=scored_dir_for_model(model) / "research_concept_centroids.npy",
        concept_centroids_meta=scored_dir_for_model(model) / "metadata" / "research_concept_centroid_meta.json",
    )


def _concept_track_steps(model: str, output_dir: Path, overwrite: bool) -> list[tuple[str, list[str]]]:
    p = _concept_retrieval_paths(model, output_dir)
    ow = _overwrite_flag(overwrite)
    return [
        ("score concept research corpus (LR)", [
            sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
            "--embed-model", model, "--classifier", "lr", "--corpus", "research",
            "--embedding-manifest", str(p["concept_embed_dir"] / "metadata" / "manifest.json"),
            "--out-dir", str(p["concept_scores_dir"]),
            "--metadata-dir", str(p["concept_scores_dir"] / "metadata"),
            "--research-centroids-out", str(p["concept_centroids"]),
            "--research-meta-out", str(p["concept_centroids_meta"]),
        ] + ow),
        ("score concept research corpus (MLP)", [
            sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
            "--embed-model", model, "--classifier", "mlp", "--corpus", "research_concept",
        ] + ow),
        ("zero-shot concept research corpus", [
            sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
            "--embed-model", model,
            "--embedding-manifest", str(p["concept_embed_dir"] / "metadata" / "manifest.json"),
            "--out-dir", str(scored_dir_for_model(model) / "zeroshot_concept"),
            "--data-dir", str(p["concept_data_dir"]),
        ] + ow),
    ]


def _run_main_analysis_steps(output_dir: Path, model: str, overwrite: bool = False, include_appendix: bool = False) -> None:
    """Single explicit linear pipeline: score -> cov gap -> register -> sem gap -> correlation.

    Pipeline order (all share the same frozen labelled data from prepare_data):

      5  CLASSIFY / SCORE
         build centroids -> score research/policy LR -> retrain+score MLP
         -> centroid consistency -> centroid similarity -> zeroshot
         + concept corpus scoring (MPNet only)

      7  COVERAGE GAP (raw)
         0_coverage_gap + concept variant (MPNet only)

      8  REGISTER ADJUSTMENT (INLP -> G)
         register_adjust.py

      9  SEMANTIC GAP BEFORE & AFTER + PCA
         1_semantic_gap raw -> adjusted (LR+MLP) -> concept variants (MPNet)
         -> adjusted zeroshot (MPNet) -> PCA landscape + PCA register before/after

      10  CORRELATION + ROBUSTNESS
          interaction (in-process) -> register decomposition + correlation + macros
          (in-process, POST_ADJUSTED)
          + appendix analyses (if include_appendix)
          NOTE: cross-sensitivity table + figures are NOT produced here; they
          are emitted exactly once by _run_analysis_poststeps, which every
          consumer calls after this function.

    Three method axes—LR (PRIMARY), MLP (sensitivity), zeroshot (sensitivity)—
    each produce their own research/policy centroids in separate namespaces.
    """
    model_args = ["--embed-model", model]

    # ==== STAGE 5: CLASSIFY / SCORE ==========================================
    run_step("prepare training data", [sys.executable, "1_code/4_supervised_model_train/0_prepare_data.py"] + model_args, step_id="0", model=model)
    run_step("retrain full data", [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py"] + model_args + _overwrite_flag(overwrite), step_id="1", model=model)
    run_build_sdg_reference_centroids(model, overwrite=overwrite)
    run_step("score research shards", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py"] + model_args + ["--classifier", "lr", "--corpus", "research"] + _overwrite_flag(overwrite), step_id="2", model=model)
    run_step("score policy corpus", [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py"] + model_args + ["--classifier", "lr", "--corpus", "policy"] + _overwrite_flag(overwrite), step_id="3", model=model)
    run_step(
        "retrain MLP",
        [sys.executable, "1_code/4_supervised_model_train/3_retrain_full_data.py",
         "--embed-model", model, "--classifier-type", "mlp"] + _overwrite_flag(overwrite),
        step_id="3b",
        model=model,
    )
    run_step(
        "score MLP",
        [sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
         "--embed-model", model, "--classifier", "mlp"] + _overwrite_flag(overwrite),
        step_id="3c",
        model=model,
    )
    run_step(
        "check centroid consistency",
        [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--output-dir", str(output_dir)] + model_args + _overwrite_flag(overwrite),
        step_id="4",
        model=model,
    )
    run_build_centroid_similarity_matrix(output_dir, model, overwrite=overwrite)
    run_step(
        "zero-shot nearest-centroid assignment",
        [sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
         "--embed-model", model, "--output-dir", str(output_dir)] + _overwrite_flag(overwrite),
        step_id="5",
        model=model,
    )
    # Concept-retrieval robustness (MPNet only): score the concept-retrieved
    # research corpus with all three assignment methods (LR/MLP/ZS), feeding the
    # Retrieval column of the cross-sensitivity tables (stage 10).
    if model == DEFAULT_EMBED_MODEL:
        concept_embed_dir = embed_dir_for_model(model) / "research_concept"
        concept_scores_dir = scored_dir_for_model(model) / "paper_scores_shards_concept"
        concept_data_dir = output_dir_for_model(model, root=output_dir) / "data" / "concept"
        concept_centroids = scored_dir_for_model(model) / "research_concept_centroids.npy"
        concept_centroids_meta = scored_dir_for_model(model) / "metadata" / "research_concept_centroid_meta.json"
        run_step("score concept research corpus (LR)", [
            sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
            "--embed-model", model, "--classifier", "lr", "--corpus", "research",
            "--embedding-manifest", str(concept_embed_dir / "metadata" / "manifest.json"),
            "--out-dir", str(concept_scores_dir),
            "--metadata-dir", str(concept_scores_dir / "metadata"),
            "--research-centroids-out", str(concept_centroids),
            "--research-meta-out", str(concept_centroids_meta),
        ] + _overwrite_flag(overwrite), model=model)
        run_step("score concept research corpus (MLP)", [
            sys.executable, "1_code/5_supervised_model_infer/score_supervised.py",
            "--embed-model", model, "--classifier", "mlp",
            "--corpus", "research_concept",
        ] + _overwrite_flag(overwrite), model=model)
        run_step("zero-shot concept research corpus", [
            sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
            "--embed-model", model,
            "--embedding-manifest", str(concept_embed_dir / "metadata" / "manifest.json"),
            "--out-dir", str(scored_dir_for_model(model) / "zeroshot_concept"),
            "--data-dir", str(concept_data_dir),
        ] + _overwrite_flag(overwrite), model=model)

    # ==== STAGE 7: COVERAGE GAP (raw) ========================================
    run_step("coverage gap (raw)", [
        sys.executable, "1_code/7_main_analysis/1_main_text/0_coverage_gap.py",
        "--output-dir", str(output_dir), "--embed-model", model,
    ] + _overwrite_flag(overwrite), model=model)
    if model == DEFAULT_EMBED_MODEL:
        run_step("coverage gap (concept corpus)", [
            sys.executable, "1_code/7_main_analysis/1_main_text/0_coverage_gap.py",
            "--output-dir", str(output_dir), "--embed-model", model,
            "--paper-scores-manifest", str(concept_scores_dir / "metadata" / "manifest.json"),
            "--out-data-dir", str(concept_data_dir),
            "--out-tables-dir", str(concept_data_dir / "tables"),
        ] + _overwrite_flag(overwrite), model=model)

    # ==== STAGE 8: REGISTER ADJUSTMENT (INLP -> G) ===========================
    run_step(
        "register_adjust (INLP -> G)",
        [sys.executable, "1_code/7_main_analysis/0_shared/register_adjust.py",
         "--embed-model", model] + _overwrite_flag(overwrite),
        model=model,
    )

    # ==== STAGE 9: SEMANTIC GAP BEFORE & AFTER + PCA =========================
    # Raw semantic gap
    run_step("semantic gap (raw)", [
        sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
        "--output-dir", str(output_dir), "--embed-model", model,
    ] + _overwrite_flag(overwrite), model=model)
    # Raw MLP semantic gap (mirrors LR raw; capped, single source of truth).
    # Runs for every model so the cross-sensitivity table has a capped MLP gap
    # for all three encoders (replaces the uncapped mlp_summary.json value).
    run_step("semantic gap (MLP, raw)", [
        sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
        "--output-dir", str(output_dir), "--embed-model", model,
        "--classifier", "mlp",
    ] + _overwrite_flag(overwrite), model=model)
    if model == DEFAULT_EMBED_MODEL:
        run_step("semantic gap (concept corpus)", [
            sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
            "--output-dir", str(output_dir), "--embed-model", model,
            "--research-centroids", str(concept_centroids),
            "--research-centroid-meta", str(concept_centroids_meta),
            "--out-data-dir", str(concept_data_dir),
            "--out-tables-dir", str(concept_data_dir / "tables"),
        ] + _overwrite_flag(overwrite), model=model)
    # Adjusted semantic gap (LR + MLP) — needs G
    run_step("semantic gap (adjusted, LR)", [
        sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
        "--output-dir", str(output_dir), "--embed-model", model,
        "--embeddings", "adjusted",
    ] + _overwrite_flag(overwrite), model=model)
    run_step("semantic gap (adjusted, MLP)", [
        sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
        "--output-dir", str(output_dir), "--embed-model", model,
        "--classifier", "mlp", "--embeddings", "adjusted",
    ] + _overwrite_flag(overwrite), model=model)
    if model == DEFAULT_EMBED_MODEL:
        run_step("semantic gap (concept corpus, adjusted)", [
            sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
            "--output-dir", str(output_dir), "--embed-model", model,
            "--embeddings", "adjusted",
            "--research-centroids", str(concept_centroids),
            "--research-centroid-meta", str(concept_centroids_meta),
            "--out-data-dir", str(concept_data_dir),
            "--out-tables-dir", str(concept_data_dir / "tables"),
        ] + _overwrite_flag(overwrite), model=model)
        mlp_concept_dir = scored_dir_for_model(model) / "mlp_scores_concept"
        run_step("semantic gap (concept corpus, MLP adjusted)", [
            sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
            "--output-dir", str(output_dir), "--embed-model", model,
            "--classifier", "mlp", "--embeddings", "adjusted",
            "--mlp-centroids", str(mlp_concept_dir / "mlp_research_centroids.npy"),
            "--mlp-policy-scores", str(mlp_concept_dir / "mlp_policy_scores.npy"),
            "--out-data-dir", str(concept_data_dir),
            "--out-tables-dir", str(concept_data_dir / "tables"),
        ] + _overwrite_flag(overwrite), model=model)
        # Raw concept-MLP semantic gap (capped, single source of truth).
        run_step("semantic gap (concept corpus, MLP)", [
            sys.executable, "1_code/7_main_analysis/1_main_text/1_semantic_gap.py",
            "--output-dir", str(output_dir), "--embed-model", model,
            "--classifier", "mlp",
            "--mlp-centroids", str(mlp_concept_dir / "mlp_research_centroids.npy"),
            "--mlp-policy-scores", str(mlp_concept_dir / "mlp_policy_scores.npy"),
            "--out-data-dir", str(concept_data_dir),
            "--out-tables-dir", str(concept_data_dir / "tables"),
        ] + _overwrite_flag(overwrite), model=model)
        # Adjusted zeroshot (MPNet only)
        run_step(
            "zero-shot nearest-centroid (adjusted)",
            [sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
             "--embed-model", model, "--output-dir", str(output_dir),
             "--embeddings", "adjusted"] + _overwrite_flag(overwrite),
            model=model,
        )
        # Concept-retrieved adjusted zeroshot (MPNet only): mirrors the keyword
        # adjusted-ZS step but scores the concept-retrieved corpus via the
        # distinct embedding manifest / output dirs, restoring ZS raw+adjusted
        # symmetry across the retrieval axis.
        run_step(
            "zero-shot concept research corpus (adjusted)",
            [sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py",
             "--embed-model", model,
             "--embedding-manifest", str(concept_embed_dir / "metadata" / "manifest.json"),
             "--out-dir", str(scored_dir_for_model(model) / "zeroshot_concept"),
             "--data-dir", str(concept_data_dir),
             "--embeddings", "adjusted"] + _overwrite_flag(overwrite),
            model=model,
        )
    # PCA: semantic landscape + register before/after (MPNet only, fixed paths)
    if model == DEFAULT_EMBED_MODEL:
        run_step("PCA semantic landscape", [
            sys.executable, "1_code/7_main_analysis/1_main_text/0_pca_semantic_landscape.py",
            "--output-dir", str(output_dir), "--embed-model", model,
        ] + _overwrite_flag(overwrite), model=model)

    # ==== STAGE 10: CORRELATION + ROBUSTNESS =================================
    # In-process: interaction analysis (+ optional appendix)
    run_analysis(model, output_dir, include_appendix=include_appendix, overwrite=overwrite)
    # Post-adjusted: decomposition table, extended interaction, correlation, macros, PCA before/after
    run_post_adjusted(model, output_dir, overwrite=overwrite)
    # NOTE: cross-sensitivity table + figures are NOT produced here. They are
    # emitted exactly once by _run_analysis_poststeps (gated to the default
    # model), which every consumer of this function calls afterwards. Producing
    # them here too would double-run them (the original bug).


def _run_analysis_poststeps(output_dir: Path, model: str, overwrite: bool = False) -> None:
    """Canonical post-analyses (MPNet only): cross-sensitivity table + figures.

    Must run AFTER every encoder's per-model analyses so the encoder-axis
    tables (which read MiniLM/SciBERT outputs) see all three encoders' data.
    """
    if model != DEFAULT_EMBED_MODEL:
        return
    run_step("generate cross-sensitivity table",
             [sys.executable, "1_code/7_main_analysis/1_main_text/3_generate_cross_sensitivity_table.py",
              "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(overwrite),
             step_id="6",
             model=model)
    run_step("plot figures", [sys.executable, "1_code/8_visualization/plot_figures.py",
             "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(overwrite), step_id="9",
             model=model)


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
    _run_analysis_poststeps(output_dir, model, overwrite=args.overwrite)


def run_warm_replay(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    include_appendix: bool = False,
) -> None:
    # Regenerate ALL three encoder tracks (MPNet + MiniLM + SciBERT) in ONE run.
    # Process the non-canonical encoders first and the canonical (MPNet) LAST, so
    # its poststep cross-sensitivity / encoder-sensitivity tables are assembled
    # from the freshly regenerated MiniLM and SciBERT coverage / semantic /
    # correlation values (the adjusted columns depend on each model's own G).
    # --embed-model is intentionally ignored here (as in --cold-replay): warm
    # replay always rebuilds every track.
    run_step("fetch encoder models",
             [sys.executable, "1_code/0_fetch/fetch_encoder_models.py"])
    ordered = [m for m in COLD_REPLAY_MODELS if m != DEFAULT_EMBED_MODEL] + [DEFAULT_EMBED_MODEL]
    for model in ordered:
        sep = "=" * 70
        print(f"\n{sep}", file=sys.stderr)
        print(f"  Encoder track: [{model}]", file=sys.stderr)
        print(sep, file=sys.stderr)
        # Only the canonical (MPNet) track runs the appendix battery. The
        # non-canonical tracks run the CORE pipeline only (train -> classify ->
        # centroids -> register adjust -> coverage gap -> semantic gap ->
        # correlation); their G / coverage / semantic / correlation values are
        # what MPNet's cross-sensitivity table reads. This mirrors cold replay
        # (main.py:791), which also gates include_appendix to the canonical model.
        model_include_appendix = include_appendix and (model == DEFAULT_EMBED_MODEL)
        run_main_text(output_dir, args, model=model, include_appendix=model_include_appendix)
    print(
        "Main text + appendix outputs rebuilt for all encoder tracks. To build the dissertation PDF, run:\n"
        "  python main.py --build-pdf --overwrite\n"
        "Note: --build-pdf requires bash (WSL/Linux) and is not supported on bare Windows."
    )


def run_cold_replay(output_dir: Path, args: argparse.Namespace) -> None:
    print("NOTE: --cold-replay rebuilds MPNet + MiniLM + SciBERT from the raw snapshot (frozen data) in ONE run.")
    print("      It is deterministic and reproducible; no OpenAlex credentials needed when the raw snapshot is hydrated.")
    if args.embed_model != DEFAULT_EMBED_MODEL:
        print(f"NOTE: --embed-model {args.embed_model!r} is ignored by --cold-replay (all three encoders are rebuilt).")
    # Auto-fetch the raw snapshot if missing (matches warm replay's auto-fetch behaviour).
    ensure_cold_replay_inputs(args)

    pre_steps = []
    pre_steps += [("fetch encoder models",
                   [sys.executable, "1_code/0_fetch/fetch_encoder_models.py"])]
    pre_steps += _preprocess_steps(args.overwrite)
    pre_steps += _segment_steps("all", args.overwrite, segment_workers=args.segment_workers)
    for label, cmd in pre_steps:
        run_step(label, cmd)

    # Per-model embed + analysis. Segments are canonical (shared); only the
    # encoder (and its native context window) varies. MiniLM/SciBERT embed the
    # shared 50k subset via --input-manifest; MPNet embeds the full corpus.
    for model in COLD_REPLAY_MODELS:
        sep = "=" * 70
        print(f"\n{sep}", file=sys.stderr)
        print(f"  Encoder track: [{model}]", file=sys.stderr)
        print(sep, file=sys.stderr)
        for label, cmd in _embed_model_steps(model, overwrite=args.overwrite,
                                             batch_size=args.batch_size,
                                             embed_batch_size_policy_and_reference=args.embed_batch_size_policy_and_reference,
                                             embed_batch_size_research=args.embed_batch_size_research,
                                             device=args.device,
                                             precision=args.precision):
            run_step(label, cmd)
        _run_main_analysis_steps(output_dir, model=model, overwrite=args.overwrite,
                                 include_appendix=(model == CANONICAL_SEGMENT_MODEL))
    # Cross-sensitivity table + figures (MPNet-only, needs all 3 encoders' data)
    # are produced once now that every encoder pass has completed.
    _run_analysis_poststeps(output_dir, CANONICAL_SEGMENT_MODEL, overwrite=args.overwrite)
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


def ensure_cold_replay_inputs(args: argparse.Namespace) -> None:
    if raw_dir().exists():
        return
    print("[info] raw snapshot not found at 2_data/0_raw/ — fetching automatically")
    run_fetch_data_snapshot(args, profile_name="raw", overwrite_data=True)


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
        for label, cmd in _preprocess_steps(args.overwrite):
            run_step(label, cmd)

    elif stage == "segment":
        corpus = args.corpus
        for label, cmd in _segment_steps(corpus, args.overwrite, segment_workers=args.segment_workers):
            run_step(label, cmd)

    elif stage == "embed":
        # Embed ALL three encoders (MPNet -> MiniLM -> SciBERT). Segments are
        # canonical/shared; only the encoder varies. MPNet embeds the full
        # research corpus (27 shards); MiniLM/SciBERT embed the shared 50k
        # subset via --corpus research_subset (handled below by the
        # model != CANONICAL_SEGMENT_MODEL check).
        for embed_model in COLD_REPLAY_MODELS:
            for label, cmd in _embed_model_steps(embed_model, overwrite=args.overwrite,
                                                 batch_size=args.batch_size,
                                                 embed_batch_size_policy_and_reference=args.embed_batch_size_policy_and_reference,
                                                 embed_batch_size_research=args.embed_batch_size_research,
                                                 device=args.device,
                                                 precision=args.precision):
                run_step(label, cmd)

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
        run_step("zero-shot nearest-centroid assignment", [sys.executable, "1_code/6_calculate_centroids/score_zeroshot.py", "--embed-model", model, "--output-dir", str(output_dir)] + _overwrite_flag(args.overwrite))
        if model == DEFAULT_EMBED_MODEL:
            for label, cmd in _concept_track_steps(model, output_dir, args.overwrite):
                run_step(label, cmd)

    elif stage == "centroids":
        # Build the SDG reference centroids (sdg_centroids.npy) consumed by the
        # zero-shot + semantic-gap analyses, then the a4 similarity matrix.
        run_step("build SDG reference centroids", [sys.executable, "1_code/6_calculate_centroids/0_build_sdg_reference_centroids.py", "--embed-model", model] + _overwrite_flag(args.overwrite))
        run_step("check centroid consistency", [sys.executable, "1_code/6_calculate_centroids/0_check_centroid_consistency.py", "--embed-model", model] + _overwrite_flag(args.overwrite))
        run_step("build centroid similarity matrix", [sys.executable, "1_code/6_calculate_centroids/1_build_centroid_similarity_matrix.py", "--output-dir", str(output_dir), "--embed-model", model] + _overwrite_flag(args.overwrite))

    elif stage == "register_adjust":
        # New register-topic decomposition stage (PLAN_register_topic_decomposition.md §6.1):
        # run INLP on research+policy embeddings and materialise ONLY the orthonormal
        # projection matrix G to gitignored 2_data/. No --output-dir (never touches
        # 4_outputs/); track is derived from --embed-model inside the script.
        run_step(
            "register_adjust (INLP -> G)",
            [sys.executable, "1_code/7_main_analysis/0_shared/register_adjust.py",
             "--embed-model", model] + _overwrite_flag(args.overwrite),
        )

    elif stage == "analysis":
        # Single-model composition for --embed-model. Cross-sensitivity + figures
        # are produced (MPNet-only gate inside _run_analysis_poststeps) here; the
        # 3-encoder aggregation is cold-replay-only.
        _run_main_analysis_steps(output_dir, model, overwrite=args.overwrite,
                                 include_appendix=True)
        _run_analysis_poststeps(output_dir, model, overwrite=args.overwrite)

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
        or any(getattr(args, spec["flag"].replace("-", "_")) for spec in APPENDIX_SPECS)
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
        for spec in APPENDIX_SPECS:
            if spec["in_all"]:
                run_appendix_spec(spec, output_dir, model=model, overwrite=args.overwrite)
        if args.build_pdf:
            build_pdf(output_dir, model=args.embed_model)
    elif any(getattr(args, spec["flag"].replace("-", "_")) for spec in APPENDIX_SPECS):
        model = args.embed_model
        for spec in APPENDIX_SPECS:
            if getattr(args, spec["flag"].replace("-", "_")):
                run_appendix_spec(spec, output_dir, model=model, overwrite=args.overwrite)
                if args.build_pdf:
                    build_pdf(output_dir, model=args.embed_model)
                break
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
