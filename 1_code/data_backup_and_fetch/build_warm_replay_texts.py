"""
Materialize 2_data/3a_warm_replay_texts/ — the gzipped warm-replay text fallback.

Canonical segment text lives in 2_data/2_segmented/{model}/ as plain .jsonl
(kept plain for cold replay and manual inspection). The embedded snapshot
instead ships a gzipped copy of exactly the files the warm-replay appendix
consumers read:

- a3_sdg4_lexical_audit.py and b2_semantic_gap_text_interpretability.py read
  every research shard listed in
  3_embedded/{model}/research_shards/metadata/manifest.json, plus
  2_segmented/{model}/policy.jsonl. No other warm-replay code reads segment
  text (verified 2026-07-29).

This builder derives the shard list from that same embed manifest, so
coverage matches the consumers by construction. Consumers resolve the
canonical plain path first and fall back to 3a_warm_replay_texts/*.jsonl.gz
(see model_utils.resolve_research_text_path / resolve_policy_text_path).

Design decisions (recorded, not magic):
- GZIP_LEVEL = 6: measured on shard part-00003 (201 MB) 2026-07-29 —
  level 6 gave 9.4x reduction at 80 MB/s vs 7.0x at 234 MB/s for level 1.
  The artifact is written once and shipped, so size wins.
- GZIP_HEADER_MTIME = 0 and empty embedded filename: makes the .gz output
  byte-deterministic across rebuilds (stable sha256).
- WARM_REPLAY_TEXT_MODELS = ("all-mpnet-base-v2",): a3/b2 run only for the
  chosen --embed-model (default mpnet). User decision 2026-07-29: the
  snapshot does not carry MiniLM appendix text. Override with --models.

Incremental / resume-safe: each output is written to a .tmp and renamed;
the per-model _build_manifest.json is rewritten after every file; files
whose recorded source size+mtime are unchanged are skipped unless --rebuild.

This script is invoked by backup_data_snapshot.py for the embedded profile.
Direct usage:
    python 1_code/data_backup_and_fetch/build_warm_replay_texts.py
    python 1_code/data_backup_and_fetch/build_warm_replay_texts.py --rebuild
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
MODEL_UTILS_DIR = WORKSPACE_ROOT / "1_code" / "7_main_analysis" / "0_shared"
if str(MODEL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_UTILS_DIR))

from model_utils import WARM_REPLAY_TEXTS_DIRNAME, model_slug  # noqa: E402

GZIP_LEVEL = 6
GZIP_HEADER_MTIME = 0
WARM_REPLAY_TEXT_MODELS = ("all-mpnet-base-v2",)
BUILD_MANIFEST_NAME = "_build_manifest.json"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [build-warm-replay-texts] {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_build_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _log(f"WARNING: unreadable build manifest {path}; rebuilding from scratch")
        return {}


def _write_build_manifest(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _compress_file(source: Path, dest: Path, gzip_level: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with source.open("rb") as fi, tmp.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw,
            compresslevel=gzip_level, mtime=GZIP_HEADER_MTIME,
        ) as fo:
            shutil.copyfileobj(fi, fo, 1024 * 1024)
        raw.flush()
        os.fsync(raw.fileno())
    tmp.replace(dest)


def _plan_for_model(source_dir: Path, model: str) -> list[tuple[Path, str]]:
    """Return [(source_path, dest_rel_name)] for one model. Fail closed."""
    slug = model_slug(model)
    embed_manifest_path = source_dir / "3_embedded" / slug / "research_shards" / "metadata" / "manifest.json"
    if not embed_manifest_path.exists():
        raise FileNotFoundError(f"embed manifest not found: {embed_manifest_path}")
    manifest = json.loads(embed_manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"embed manifest has no shards: {embed_manifest_path}")

    segmented_root = source_dir / "2_segmented" / slug
    plan: list[tuple[Path, str]] = []
    for shard in sorted(shards, key=lambda s: int(s["shard_id"])):
        name = str(shard["name"])
        src = segmented_root / "research" / f"{name}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"canonical research text missing: {src}")
        plan.append((src, f"research/{name}.jsonl.gz"))
    policy_src = segmented_root / "policy.jsonl"
    if not policy_src.exists():
        raise FileNotFoundError(f"canonical policy text missing: {policy_src}")
    plan.append((policy_src, "policy.jsonl.gz"))
    return plan


def build_warm_replay_texts(
    source_dir: Path,
    models: tuple[str, ...] = WARM_REPLAY_TEXT_MODELS,
    gzip_level: int = GZIP_LEVEL,
    rebuild: bool = False,
) -> list[Path]:
    source_dir = source_dir.resolve()
    built: list[Path] = []
    for model in models:
        slug = model_slug(model)
        plan = _plan_for_model(source_dir, model)
        out_root = source_dir / WARM_REPLAY_TEXTS_DIRNAME / slug
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / BUILD_MANIFEST_NAME
        payload = _load_build_manifest(manifest_path)
        files: dict = payload.get("files", {}) if isinstance(payload.get("files"), dict) else {}

        n_skipped = 0
        for idx, (src, rel_name) in enumerate(plan, start=1):
            dest = out_root / rel_name
            stat = src.stat()
            entry = files.get(rel_name)
            if (
                not rebuild
                and dest.exists()
                and isinstance(entry, dict)
                and entry.get("source_size") == stat.st_size
                and entry.get("source_mtime_ns") == stat.st_mtime_ns
            ):
                n_skipped += 1
                built.append(dest)
                continue
            _log(f"[{model}] compressing {idx}/{len(plan)}: {src.name} ({stat.st_size / 1e6:.0f} MB)")
            _compress_file(src, dest, gzip_level)
            files[rel_name] = {
                "source": str(src.relative_to(source_dir)),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "dest_size": dest.stat().st_size,
                "dest_sha256": _sha256_file(dest),
            }
            payload = {
                "generated_by": "1_code/data_backup_and_fetch/build_warm_replay_texts.py",
                "model": model,
                "gzip_level": gzip_level,
                "gzip_header_mtime": GZIP_HEADER_MTIME,
                "shard_count": len(plan) - 1,
                "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "files": files,
            }
            _write_build_manifest(manifest_path, payload)
            built.append(dest)

        stale = sorted(set(files) - {rel_name for _, rel_name in plan})
        if stale:
            raise RuntimeError(
                f"[{model}] build manifest lists files no longer in the plan: {stale}. "
                "Delete the 3a_warm_replay_texts model directory and rebuild."
            )
        total = sum(int(files[k]["dest_size"]) for k in files)
        _log(
            f"[{model}] done: {len(plan)} files ({n_skipped} up-to-date, "
            f"{len(plan) - n_skipped} compressed), {total / 1e9:.2f} GB gzipped"
        )
    return built


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 2_data/3a_warm_replay_texts/ (gzipped warm-replay appendix text).")
    ap.add_argument("--source-dir", type=Path, default=WORKSPACE_ROOT / "2_data",
                    help="2_data directory to read from and write into. Defaults to dissertation 2_data/.")
    ap.add_argument("--models", nargs="+", default=list(WARM_REPLAY_TEXT_MODELS),
                    help=f"Embed models to include. Default: {list(WARM_REPLAY_TEXT_MODELS)}")
    ap.add_argument("--gzip-level", type=int, default=GZIP_LEVEL,
                    help=f"gzip compression level (1-9). Default: {GZIP_LEVEL} (see module docstring).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Recompress every file even if the recorded source size+mtime are unchanged.")
    args = ap.parse_args()
    build_warm_replay_texts(
        source_dir=args.source_dir,
        models=tuple(args.models),
        gzip_level=args.gzip_level,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
