"""
Re-visualize the full OpenAlex corpus using existing legacy analysis/plot scripts.

This bridge script materializes full-corpus compatibility artifacts from shard outputs
inside a timestamped run workspace, then executes existing scripts unchanged:
  - code/coverage_gap.py
  - code/semantic_gap.py
  - code/coverage_semantic_interaction.py
  - code/plot_figures.py

Key properties:
  - checkpoint-safe / resume-safe stages
  - canonical repo outputs are untouched
  - strict reproducibility bundle (env snapshot, command log, artifact hashes)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from alignment_score import build_research_centroids
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, sha256_file, update_stage_status


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES_MANIFEST = ROOT / "data" / "paper_scores_shards" / "manifest.json"
DEFAULT_EMB_MANIFEST = ROOT / "data" / "embeddings" / "papers_shards" / "manifest.json"
DEFAULT_CLEAN_MANIFEST = ROOT / "data" / "openalex" / "clean_shards" / "manifest.json"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-manifest", default=str(DEFAULT_SCORES_MANIFEST))
    p.add_argument("--embedding-manifest", default=str(DEFAULT_EMB_MANIFEST))
    p.add_argument("--clean-manifest", default=str(DEFAULT_CLEAN_MANIFEST))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--run-name", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--limit-shards", type=int, default=0, help="Optional smoke-mode shard limit.")
    p.add_argument("--skip-analysis", action="store_true", help="Only build bridge artifacts.")
    p.add_argument("--skip-plots", action="store_true", help="Run analysis scripts but skip plot_figures.py.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _run(cmd: list[str], cwd: Path, command_log: Path) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with command_log.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] cwd={cwd}\n")
        f.write("$ " + " ".join(cmd) + "\n\n")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _status_dir(run_dir: Path) -> Path:
    return run_dir / "job_status"


def _is_stage_done(run_dir: Path, stage: str) -> bool:
    payload = read_json(_status_dir(run_dir) / f"{stage}.json", default={})
    return payload.get("state") == "completed"


def _mark_stage(run_dir: Path, stage: str, state: str, extra: dict[str, Any] | None = None) -> None:
    update_stage_status(_status_dir(run_dir), stage, state, extra=extra)


def _select_shards(shards: list[dict[str, Any]], limit_shards: int) -> list[dict[str, Any]]:
    ordered = sorted(shards, key=lambda s: int(s["shard_id"]))
    if limit_shards > 0:
        return ordered[:limit_shards]
    return ordered


def _total_rows(shards: list[dict[str, Any]]) -> int:
    return int(sum(int(s["rows"]) for s in shards))


def stage_prepare_workspace(run_dir: Path) -> Path:
    ws = run_dir / "workspace"
    ensure_dir(ws)
    ensure_dir(ws / "data")
    ensure_dir(ws / "data" / "embeddings")
    ensure_dir(ws / "data" / "generated")
    ensure_dir(ws / "writing" / "figures")

    code_link = ws / "code"
    if not code_link.exists() and not code_link.is_symlink():
        os.symlink(ROOT / "code", code_link)

    # Policy artifacts reused via symlink/copy fallback.
    _safe_link(ROOT / "data" / "policy_scores.npy", ws / "data" / "policy_scores.npy")
    _safe_link(ROOT / "data" / "policy_scores_ids.json", ws / "data" / "policy_scores_ids.json")
    _safe_link(ROOT / "data" / "embeddings" / "policy.npy", ws / "data" / "embeddings" / "policy.npy")

    return ws


def stage_materialize_paper_scores(
    run_dir: Path,
    ws: Path,
    scores_manifest_path: Path,
    limit_shards: int,
) -> dict[str, Any]:
    manifest = read_json(scores_manifest_path)
    shards = _select_shards(manifest["shards"], limit_shards)
    n_rows = _total_rows(shards)

    scores_out = ws / "data" / "paper_scores.npy"
    ids_out = ws / "data" / "paper_scores_ids.json"

    first_scores = np.load(Path(shards[0]["score_path"]), mmap_mode="r")
    n_sdg = int(first_scores.shape[1])

    mmap = np.lib.format.open_memmap(scores_out, mode="w+", dtype=np.float32, shape=(n_rows, n_sdg))

    offset = 0
    first = True
    with ids_out.open("w", encoding="utf-8") as ids_f:
        ids_f.write("[")
        for shard in shards:
            score_path = Path(shard["score_path"])
            ids_path = Path(shard["ids_path"])
            arr = np.load(score_path, mmap_mode="r")
            rows = int(arr.shape[0])
            mmap[offset : offset + rows] = arr

            row_in_shard_expected = 0
            ids_written = 0
            for rec in _read_jsonl(ids_path):
                if rec.get("row_in_shard") is not None and int(rec["row_in_shard"]) != row_in_shard_expected:
                    raise RuntimeError(f"row_in_shard mismatch in {ids_path} at row {row_in_shard_expected}")
                row_in_shard_expected += 1
                openalex_id = rec.get("openalex_id")
                if not openalex_id:
                    raise RuntimeError(f"missing openalex_id in {ids_path}")
                if not first:
                    ids_f.write(",")
                ids_f.write(json.dumps(openalex_id))
                first = False
                ids_written += 1

            if ids_written != rows:
                raise RuntimeError(
                    f"ids count mismatch for shard {shard['shard_id']}: ids={ids_written} rows={rows}"
                )

            offset += rows

        ids_f.write("]")

    mmap.flush()
    del mmap

    if offset != n_rows:
        raise RuntimeError(f"row mismatch while materializing scores: wrote={offset} expected={n_rows}")

    return {
        "n_rows": n_rows,
        "n_sdg": n_sdg,
        "shards_used": len(shards),
        "scores_sha256": sha256_file(scores_out),
        "ids_sha256": sha256_file(ids_out),
    }


def stage_materialize_paper_embeddings(
    run_dir: Path,
    ws: Path,
    emb_manifest_path: Path,
    expected_rows: int,
    limit_shards: int,
) -> dict[str, Any]:
    manifest = read_json(emb_manifest_path)
    shards = _select_shards(manifest["shards"], limit_shards)
    n_rows = _total_rows(shards)
    if n_rows != expected_rows:
        raise RuntimeError(f"embedding rows ({n_rows}) != score rows ({expected_rows})")

    first_emb = np.load(Path(shards[0]["embedding_path"]), mmap_mode="r")
    emb_dim = int(first_emb.shape[1])

    emb_out = ws / "data" / "embeddings" / "papers.npy"
    mmap = np.lib.format.open_memmap(emb_out, mode="w+", dtype=np.float32, shape=(n_rows, emb_dim))

    offset = 0
    for shard in shards:
        emb_path = Path(shard["embedding_path"])
        arr = np.load(emb_path, mmap_mode="r")
        rows = int(arr.shape[0])
        mmap[offset : offset + rows] = arr
        offset += rows

    mmap.flush()
    del mmap

    if offset != n_rows:
        raise RuntimeError(f"row mismatch while materializing embeddings: wrote={offset} expected={n_rows}")

    return {
        "n_rows": n_rows,
        "embedding_dim": emb_dim,
        "shards_used": len(shards),
        "embeddings_sha256": sha256_file(emb_out),
    }


def stage_recompute_research_direction(ws: Path) -> dict[str, Any]:
    paper_scores = np.load(ws / "data" / "paper_scores.npy", mmap_mode="r")
    paper_emb = np.load(ws / "data" / "embeddings" / "papers.npy", mmap_mode="r")
    policy_emb = np.load(ws / "data" / "embeddings" / "policy.npy")

    research_centroids, research_meta = build_research_centroids(
        paper_emb,
        paper_scores,
        n_sdg=17,
    )

    np.save(ws / "data" / "research_centroids.npy", research_centroids)
    atomic_write_json(ws / "data" / "research_centroid_meta.json", research_meta)

    policy_vs_research = (policy_emb @ research_centroids.T).astype(np.float32)
    np.save(ws / "data" / "policy_scores_vs_research.npy", policy_vs_research)

    return {
        "research_centroids_shape": list(research_centroids.shape),
        "policy_vs_research_shape": list(policy_vs_research.shape),
        "research_centroids_sha256": sha256_file(ws / "data" / "research_centroids.npy"),
        "policy_vs_research_sha256": sha256_file(ws / "data" / "policy_scores_vs_research.npy"),
    }


def stage_run_legacy_analysis(ws: Path, py: str, command_log: Path, skip_plots: bool) -> dict[str, Any]:
    cmds = [
        [py, "code/coverage_gap.py"],
        [py, "code/semantic_gap.py"],
        [py, "code/coverage_semantic_interaction.py"],
    ]
    if not skip_plots:
        cmds.append([py, "code/plot_figures.py"])
    for cmd in cmds:
        _run(cmd, cwd=ws, command_log=command_log)
    return {
        "commands": [" ".join(c) for c in cmds],
    }


def _capture(cmd: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def stage_environment_snapshot(run_dir: Path, py: str) -> dict[str, Any]:
    env_dir = run_dir / "reproducibility"
    ensure_dir(env_dir)

    checks = {
        "uname": _capture(["uname", "-a"], ROOT),
        "os_release": _capture(["cat", "/etc/os-release"], ROOT),
        "conda_version": _capture(["conda", "--version"], ROOT),
        "python_version": _capture([py, "--version"], ROOT),
        "nvidia_smi": _capture([
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version,temperature.gpu,utilization.gpu,memory.total",
            "--format=csv,noheader",
        ], ROOT),
    }

    py_probe = (
        "import importlib\n"
        "import json\n"
        "import platform\n"
        "import sys\n"
        "mods=['torch','torchvision','torchaudio','sentence_transformers','transformers','tokenizers','numpy','scipy','sklearn','pandas','datasets']\n"
        "out={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{}}\n"
        "for m in mods:\n"
        "  try:\n"
        "    mod=importlib.import_module(m)\n"
        "    out['packages'][m]=getattr(mod,'__version__','unknown')\n"
        "  except Exception as e:\n"
        "    out['packages'][m]=f'MISSING: {e}'\n"
        "try:\n"
        "  import torch\n"
        "  out['torch_cuda_available']=bool(torch.cuda.is_available())\n"
        "  out['torch_cuda_version']=getattr(torch.version,'cuda',None)\n"
        "  out['torch_gpu_count']=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0\n"
        "  if torch.cuda.is_available():\n"
        "    out['torch_gpu_name']=torch.cuda.get_device_name(0)\n"
        "except Exception as e:\n"
        "  out['torch_probe_error']=str(e)\n"
        "print(json.dumps(out, ensure_ascii=False, indent=2))\n"
    )
    py_probe_out = _capture([py, "-c", py_probe], ROOT)

    atomic_write_json(env_dir / "environment_checks.json", {"checks": checks, "python_probe": py_probe_out})

    md_lines = [
        "# Runtime Snapshot",
        "",
        f"Captured at UTC: {now_iso()}",
        "",
        "## Core checks",
    ]
    for key, rec in checks.items():
        md_lines.append(f"### {key}")
        md_lines.append("```text")
        md_lines.append((rec.get("stdout") or "").strip() or "(no stdout)")
        if rec.get("stderr"):
            md_lines.append("--- stderr ---")
            md_lines.append(rec["stderr"])
        md_lines.append("```")
        md_lines.append("")

    md_lines.append("## Python package probe")
    md_lines.append("```json")
    md_lines.append(py_probe_out.get("stdout", ""))
    if py_probe_out.get("stderr"):
        md_lines.append("// stderr")
        md_lines.append(py_probe_out["stderr"])
    md_lines.append("```")
    md_lines.append("")

    (env_dir / "runtime_snapshot.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "environment_checks_path": str(env_dir / "environment_checks.json"),
        "runtime_snapshot_path": str(env_dir / "runtime_snapshot.md"),
    }


def stage_source_manifest_snapshot(
    run_dir: Path,
    scores_manifest: Path,
    emb_manifest: Path,
    clean_manifest: Path,
) -> dict[str, Any]:
    out_dir = run_dir / "source_manifests"
    ensure_dir(out_dir)

    name_map = {
        str(scores_manifest.resolve()): "paper_scores_shards_manifest.json",
        str(emb_manifest.resolve()): "papers_embeddings_shards_manifest.json",
        str(clean_manifest.resolve()): "openalex_clean_shards_manifest.json",
    }

    copied = {}
    for src in [scores_manifest, emb_manifest, clean_manifest]:
        if not src.exists():
            continue
        dst_name = name_map.get(str(src.resolve()), src.name)
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        copied[str(src)] = {
            "copied_to": str(dst),
            "sha256": sha256_file(dst),
        }

    atomic_write_json(out_dir / "manifest_snapshot_index.json", copied)
    return {"copied": copied}


def stage_artifact_hashes(run_dir: Path, ws: Path) -> dict[str, Any]:
    rel_targets = [
        "data/paper_scores.npy",
        "data/paper_scores_ids.json",
        "data/embeddings/papers.npy",
        "data/research_centroids.npy",
        "data/research_centroid_meta.json",
        "data/policy_scores_vs_research.npy",
        "data/coverage_gap.json",
        "data/coverage_gap_raw.json",
        "data/semantic_gap.json",
        "data/semantic_gap_sensitivity.json",
        "data/h25_correlation.json",
        "data/h25_scatter.csv",
        "writing/figures/fig1_coverage_profiles.pdf",
        "writing/figures/fig1_coverage_profiles.png",
        "writing/figures/fig2_semantic_gap.pdf",
        "writing/figures/fig2_semantic_gap.png",
        "writing/figures/fig3_coverage_semantic_scatter.pdf",
        "writing/figures/fig3_coverage_semantic_scatter.png",
    ]

    items = {}
    for rel in rel_targets:
        p = ws / rel
        if not p.exists():
            continue
        items[rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    for p in sorted((ws / "data" / "generated").glob("*.tex")):
        rel = str(p.relative_to(ws))
        items[rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }

    out = run_dir / "artifacts_manifest.json"
    atomic_write_json(out, {"generated_at_utc": now_iso(), "artifacts": items})
    return {"artifact_manifest": str(out), "n_artifacts": len(items)}


def _execute_stage(run_dir: Path, stage: str, fn, *args, **kwargs):
    if _is_stage_done(run_dir, stage):
        return read_json(_status_dir(run_dir) / f"{stage}.json", default={})
    _mark_stage(run_dir, stage, "running")
    try:
        meta = fn(*args, **kwargs)
    except Exception as exc:
        _mark_stage(run_dir, stage, "failed", {"error": str(exc)})
        raise
    _mark_stage(run_dir, stage, "completed", meta if isinstance(meta, dict) else None)
    return meta


def main() -> None:
    args = parse_args()

    scores_manifest = Path(args.scores_manifest)
    emb_manifest = Path(args.embedding_manifest)
    clean_manifest = Path(args.clean_manifest)
    out_root = Path(args.out_root)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"full_corpus_viz_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = out_root / run_name
    ensure_dir(run_dir)

    command_log = run_dir / "commands.log"
    (run_dir / "README.txt").write_text(
        "Run-local workspace for full-corpus bridge visualization.\n"
        "Canonical repo data/ and writing/ are untouched.\n",
        encoding="utf-8",
    )

    if not args.resume:
        # Non-resume mode requires a clean run directory.
        status = list(_status_dir(run_dir).glob("*.json")) if _status_dir(run_dir).exists() else []
        if status:
            raise RuntimeError(f"Run dir already has stage status files: {run_dir}")

    _execute_stage(run_dir, "prepare_workspace", stage_prepare_workspace, run_dir)
    ws = run_dir / "workspace"

    _execute_stage(
        run_dir,
        "source_manifest_snapshot",
        stage_source_manifest_snapshot,
        run_dir,
        scores_manifest,
        emb_manifest,
        clean_manifest,
    )

    _execute_stage(
        run_dir,
        "materialize_paper_scores",
        stage_materialize_paper_scores,
        run_dir,
        ws,
        scores_manifest,
        args.limit_shards,
    )

    scores_meta = read_json(_status_dir(run_dir) / "materialize_paper_scores.json", default={})
    n_rows = int(scores_meta.get("n_rows", 0))

    _execute_stage(
        run_dir,
        "materialize_paper_embeddings",
        stage_materialize_paper_embeddings,
        run_dir,
        ws,
        emb_manifest,
        n_rows,
        args.limit_shards,
    )

    _execute_stage(run_dir, "recompute_research_direction", stage_recompute_research_direction, ws)

    if not args.skip_analysis:
        _execute_stage(
            run_dir,
            "legacy_analysis",
            stage_run_legacy_analysis,
            ws,
            args.python,
            command_log,
            args.skip_plots,
        )

    _execute_stage(run_dir, "environment_snapshot", stage_environment_snapshot, run_dir, args.python)
    _execute_stage(run_dir, "artifact_hashes", stage_artifact_hashes, run_dir, ws)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "workspace": str(ws),
        "resume": args.resume,
        "limit_shards": args.limit_shards,
        "skip_analysis": args.skip_analysis,
        "skip_plots": args.skip_plots,
        "completed_at_utc": now_iso(),
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
