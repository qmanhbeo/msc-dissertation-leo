"""
Create and upload a timestamped snapshot of the dissertation data/ directory.

Adapted from ~/stocks-ecosystem/alpha-research-lab/backup_data_snapshot.py.

Features:
- creates either:
  - `dissertation-data-snapshot-YYYY-MM-DD-HHMMSS.tar.zst` for the full profile
  - `dissertation-data-snapshot-curated-YYYY-MM-DD-HHMMSS.tar.zst` for the curated profile
- writes a matching `.sha256` checksum file
- uploads both to Google Drive via `rclone`
- prunes old local and remote snapshot pairs, keeping the newest N
- embeds snapshot metadata under `data/_snapshot_metadata/`

Usage:
    python main.py --backup-data-snapshot curated
    python main.py --backup-data-snapshot both

    # direct utility usage:
    python 1_code/data_backup_and_fetch/backup_data_snapshot.py

    # dry run — build archive locally, skip upload:
    python 1_code/data_backup_and_fetch/backup_data_snapshot.py --no-upload

    # marker-facing replay snapshot without OpenAlex and rebuildable caches:
    python 1_code/data_backup_and_fetch/backup_data_snapshot.py --profile curated --no-upload

    # override remote:
    python 1_code/data_backup_and_fetch/backup_data_snapshot.py --remote-root gdrive:some/other/path

Environment:
    DISSERTATION_SNAPSHOT_REMOTE_ROOT may be used instead of --remote-root.

Requirements:
    pip install zstandard
    rclone must be installed and the remote configured (run `rclone config` if not).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import zstandard as zstd

from data_snapshot_profiles import (
    SNAPSHOT_METADATA_FILE,
    build_snapshot_metadata,
    get_snapshot_profile,
    should_exclude_data_path,
    snapshot_archive_prefix,
    write_snapshot_metadata_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent    # dissertation root
DEFAULT_KEEP = 7
DEFAULT_REMOTE_ROOT = "stocks-ecosystem-data-snapshots:dissertation-backup/data-snapshots/"

@dataclass(frozen=True)
class SnapshotPaths:
    snapshot_at: datetime
    profile_name: str
    archive_path: Path
    checksum_path: Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _snapshot_patterns(profile_name: str) -> tuple[str, re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    prefix = snapshot_archive_prefix(profile_name)
    stem_re = re.compile(rf"^{re.escape(prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})(?:-(\d{{6}}))?$")
    archive_re = re.compile(
        rf"^({re.escape(prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})(?:-(\d{{6}}))?)\.tar\.zst$"
    )
    checksum_re = re.compile(
        rf"^({re.escape(prefix)}-(\d{{4}}-\d{{2}}-\d{{2}})(?:-(\d{{6}}))?)\.tar\.zst\.sha256$"
    )
    return prefix, stem_re, archive_re, checksum_re


def _snapshot_name(profile_name: str, snapshot_at: datetime) -> str:
    prefix = snapshot_archive_prefix(profile_name)
    return f"{prefix}-{snapshot_at.strftime('%Y-%m-%d-%H%M%S')}.tar.zst"


def _snapshot_paths(output_dir: Path, profile_name: str, snapshot_at: datetime) -> SnapshotPaths:
    archive_name = _snapshot_name(profile_name, snapshot_at)
    archive_path = output_dir / archive_name
    checksum_path = output_dir / f"{archive_name}.sha256"
    return SnapshotPaths(
        snapshot_at=snapshot_at,
        profile_name=profile_name,
        archive_path=archive_path,
        checksum_path=checksum_path,
    )


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum(archive_path: Path, checksum_path: Path) -> None:
    digest = _compute_sha256(archive_path)
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")


def _iter_snapshot_members(source_dir: Path, profile_name: str):
    profile = get_snapshot_profile(profile_name)
    if not source_dir.exists():
        raise FileNotFoundError(f"snapshot source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise RuntimeError(f"snapshot source is not a directory: {source_dir}")
    if source_dir.is_symlink():
        raise RuntimeError(f"refusing to snapshot symlinked source directory: {source_dir}")

    yield source_dir
    for root, dirnames, filenames in os.walk(source_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dirnames.sort()
        filenames.sort()

        kept_dirnames: list[str] = []
        for dirname in dirnames:
            path = root_path / dirname
            rel_data_path = path.relative_to(source_dir)
            if should_exclude_data_path(rel_data_path, profile):
                continue
            kept_dirnames.append(dirname)
            if path.is_symlink():
                raise RuntimeError(f"refusing to snapshot symlink entry: {path}")
            yield path
        dirnames[:] = kept_dirnames

        for filename in filenames:
            path = root_path / filename
            rel_data_path = path.relative_to(source_dir)
            if should_exclude_data_path(rel_data_path, profile):
                continue
            if path.is_symlink():
                raise RuntimeError(f"refusing to snapshot symlink entry: {path}")
            if not path.is_file():
                raise RuntimeError(f"refusing to snapshot non-regular entry: {path}")
            yield path


def _build_archive(source_dir: Path, archive_path: Path, *, profile_name: str, zstd_level: int = 9) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    _log(f"building archive {archive_path.name} from {source_dir}")
    snapshot_version = archive_path.name.removesuffix(".tar.zst")
    metadata_payload = build_snapshot_metadata(
        workspace_root=WORKSPACE_ROOT,
        source_dir=source_dir,
        profile=get_snapshot_profile(profile_name),
        snapshot_version=snapshot_version,
    )
    compressor = zstd.ZstdCompressor(level=zstd_level, threads=-1)
    with tempfile.TemporaryDirectory(prefix="snapshot-meta-", dir=str(WORKSPACE_ROOT)) as tmpdir:
        metadata_path = Path(tmpdir) / SNAPSHOT_METADATA_FILE.name
        write_snapshot_metadata_file(metadata_path, metadata_payload)
        with tmp_path.open("wb") as raw:
            with compressor.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tf:
                    for path in _iter_snapshot_members(source_dir, profile_name):
                        arcname = Path("2_data") if path == source_dir else Path("2_data") / path.relative_to(source_dir)
                        tf.add(path, arcname=str(arcname), recursive=False)
                    tf.add(metadata_path, arcname=str(Path("2_data") / SNAPSHOT_METADATA_FILE), recursive=False)
    tmp_path.replace(archive_path)


def _snapshot_sort_key_from_stem(stem: str, stem_re: re.Pattern[str]) -> datetime | None:
    m = stem_re.fullmatch(stem)
    if not m:
        return None
    day = date.fromisoformat(m.group(1))
    time_part = m.group(2)
    if time_part:
        return datetime.strptime(f"{m.group(1)}-{time_part}", "%Y-%m-%d-%H%M%S")
    return datetime.combine(day, datetime.min.time())


def _snapshot_group_info(
    path: Path,
    *,
    stem_re: re.Pattern[str],
    archive_re: re.Pattern[str],
    checksum_re: re.Pattern[str],
) -> tuple[str, datetime] | None:
    m = archive_re.fullmatch(path.name)
    if m:
        stem = m.group(1)
        sort_key = _snapshot_sort_key_from_stem(stem, stem_re)
        if sort_key is not None:
            return stem, sort_key
    m = checksum_re.fullmatch(path.name)
    if m:
        stem = m.group(1)
        sort_key = _snapshot_sort_key_from_stem(stem, stem_re)
        if sort_key is not None:
            return stem, sort_key
    return None


def _local_snapshot_groups(root: Path, profile_name: str) -> dict[str, dict]:
    _, stem_re, archive_re, checksum_re = _snapshot_patterns(profile_name)
    groups: dict[str, dict] = {}
    for path in root.iterdir():
        info = _snapshot_group_info(path, stem_re=stem_re, archive_re=archive_re, checksum_re=checksum_re)
        if info is None:
            continue
        stem, sort_key = info
        bucket = groups.setdefault(stem, {"sort_key": sort_key})
        if archive_re.fullmatch(path.name):
            bucket["archive"] = path
        elif checksum_re.fullmatch(path.name):
            bucket["checksum"] = path
    return groups


def _prune_local_snapshots(root: Path, profile_name: str, keep: int) -> list[Path]:
    groups = _local_snapshot_groups(root, profile_name)
    ordered_stems = sorted(groups.keys(), key=lambda stem: groups[stem]["sort_key"], reverse=True)
    removed: list[Path] = []
    for stem in ordered_stems[keep:]:
        for path in groups[stem].values():
            if isinstance(path, Path) and path.exists():
                path.unlink()
                removed.append(path)
    return removed


def _remote_join(remote_root: str, filename: str) -> str:
    return remote_root.rstrip("/") + "/" + filename


def _check_rclone_remote(remote_root: str) -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is required but was not found on PATH")
    if ":" not in remote_root:
        raise RuntimeError(
            f"invalid remote root '{remote_root}'; expected something like 'gdrive:backups/data'"
        )
    remote_name = remote_root.split(":", 1)[0].strip()
    if not remote_name:
        raise RuntimeError(f"invalid remote root '{remote_root}'")

    result = subprocess.run(
        ["rclone", "listremotes"],
        check=True,
        capture_output=True,
        text=True,
    )
    remotes = {line.strip().rstrip(":") for line in result.stdout.splitlines() if line.strip()}
    if remote_name not in remotes:
        raise RuntimeError(
            f"rclone remote '{remote_name}' is not configured. Run `rclone config` first."
        )


def _run_rclone(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True)


def _upload_snapshot(remote_root: str, snapshot: SnapshotPaths) -> None:
    _log(f"uploading to {remote_root}")
    _run_rclone("mkdir", remote_root)
    _run_rclone("copyto", str(snapshot.archive_path), _remote_join(remote_root, snapshot.archive_path.name))
    _run_rclone("copyto", str(snapshot.checksum_path), _remote_join(remote_root, snapshot.checksum_path.name))


def _remote_snapshot_groups(remote_root: str, profile_name: str) -> dict[str, dict]:
    _, stem_re, archive_re, checksum_re = _snapshot_patterns(profile_name)
    result = _run_rclone("lsjson", remote_root)
    import json

    entries = json.loads(result.stdout or "[]")
    groups: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("Name") or entry.get("Path") or "").strip()
        info = _snapshot_group_info(
            Path(name),
            stem_re=stem_re,
            archive_re=archive_re,
            checksum_re=checksum_re,
        )
        if info is None:
            continue
        stem, sort_key = info
        bucket = groups.setdefault(stem, {"sort_key": sort_key})
        if archive_re.fullmatch(name):
            bucket["archive"] = name
        elif checksum_re.fullmatch(name):
            bucket["checksum"] = name
    return groups


def _prune_remote_snapshots(remote_root: str, profile_name: str, keep: int) -> list[str]:
    groups = _remote_snapshot_groups(remote_root, profile_name)
    ordered_stems = sorted(groups.keys(), key=lambda stem: groups[stem]["sort_key"], reverse=True)
    removed: list[str] = []
    for stem in ordered_stems[keep:]:
        for name in groups[stem].values():
            if not isinstance(name, str):
                continue
            _run_rclone("deletefile", _remote_join(remote_root, name))
            removed.append(name)
    return removed


def _create_snapshot(
    *,
    source_dir: Path,
    output_dir: Path,
    profile_name: str,
    snapshot_at: datetime,
    zstd_level: int,
) -> SnapshotPaths:
    snapshot = _snapshot_paths(output_dir, profile_name, snapshot_at)
    _build_archive(source_dir, snapshot.archive_path, profile_name=profile_name, zstd_level=zstd_level)
    _write_checksum(snapshot.archive_path, snapshot.checksum_path)
    return snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description="Backup dissertation data/ to Google Drive via rclone.")
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=WORKSPACE_ROOT / "2_data",
        help="Directory to snapshot. Defaults to dissertation 2_data/.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Where to write the local archive and checksum. Defaults to dissertation root.",
    )
    ap.add_argument(
        "--remote-root",
        type=str,
        default=os.environ.get("DISSERTATION_SNAPSHOT_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
        help=f"rclone destination, e.g. {DEFAULT_REMOTE_ROOT}",
    )
    ap.add_argument(
        "--profile",
        choices=["full", "curated"],
        default="full",
        help="Snapshot profile. 'full' preserves the literal data/ tree. 'curated' excludes large rebuildable caches.",
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        help="Keep this many newest snapshot pairs locally and on Drive. Default: 7.",
    )
    ap.add_argument(
        "--date",
        type=str,
        default="",
        help="Override the snapshot day (YYYY-MM-DD). Defaults to today.",
    )
    ap.add_argument(
        "--zstd-level",
        type=int,
        default=9,
        help="zstd compression level (1–22). Default: 9.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="Build the local archive only; skip rclone upload and remote pruning.",
    )
    args = ap.parse_args()

    if args.keep < 1:
        raise SystemExit("ERROR: --keep must be >= 1")

    if not args.no_upload and not args.remote_root:
        raise SystemExit(
            "ERROR: --remote-root or DISSERTATION_SNAPSHOT_REMOTE_ROOT is required unless --no-upload is set"
        )

    if not args.no_upload:
        _check_rclone_remote(args.remote_root)

    snapshot_day = date.fromisoformat(args.date) if args.date else datetime.now().date()
    snapshot_at = datetime.combine(snapshot_day, datetime.now().time().replace(microsecond=0))

    snapshot = _create_snapshot(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        profile_name=args.profile,
        snapshot_at=snapshot_at,
        zstd_level=args.zstd_level,
    )

    if not args.no_upload:
        _upload_snapshot(args.remote_root, snapshot)
        removed_remote = _prune_remote_snapshots(args.remote_root, args.profile, args.keep)
        if removed_remote:
            _log(f"pruned remote snapshot files: {removed_remote}")

    removed_local = _prune_local_snapshots(args.output_dir.resolve(), args.profile, args.keep)
    if removed_local:
        _log(f"pruned local snapshot files: {[str(p) for p in removed_local]}")

    _log(
        f"done  archive={snapshot.archive_path.name}  "
        f"checksum={snapshot.checksum_path.name}  "
        f"profile={args.profile}  "
        f"keep={args.keep}  upload={'yes' if not args.no_upload else 'no'}"
    )


if __name__ == "__main__":
    main()
