"""
Shared utilities for shard-based OpenAlex processing.

This module centralises:
  - atomic JSON writes
  - stage status updates (abort/resume visibility)
  - simple checksums
  - lightweight manifest helpers
  - JSON / JSONL I/O
  - hard-pivot manifest path resolution
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:  # joblib is only needed for atomic_write_joblib
    joblib = None
try:
    import numpy as np
except Exception:  # numpy is only needed for atomic_write_npy
    np = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_npy(path: Path, arr: Any) -> None:
    """Atomically write a NumPy array: write to a .tmp sibling, fsync, then replace.

    Prevents a torn/corrupt .npy from being accepted as 'complete' by downstream
    exists-skip checks after an interrupted run.
    """
    if np is None:
        raise RuntimeError("numpy is required for atomic_write_npy")
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_joblib(path: Path, obj: Any) -> None:
    """Atomically serialise an object with joblib: write to a .tmp sibling, then replace.

    Protects the canonical trained model (e.g. sdg_classifier_retrained.joblib) from
    corruption if the run is interrupted mid-dump.
    """
    if joblib is None:
        raise RuntimeError("joblib is required for atomic_write_joblib")
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path) -> Any:
    """Load JSON file. Raises FileNotFoundError if missing."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    """Yield dicts from a JSONL file, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_manifest_path(
    stored_path: str,
    *,
    allowed_dirs: tuple[Path, ...],
) -> Path:
    """Resolve a hard-pivot path from a manifest.

    If stored_path is absolute, verify it exists and return it.
    If relative, verify it starts with one of the allowed directories,
    prepend CWD, and verify existence.
    """
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    posix = raw.as_posix()
    allowed_prefixes = tuple(d.as_posix() + "/" for d in allowed_dirs)
    if not any(posix.startswith(p) for p in allowed_prefixes):
        raise RuntimeError(
            f"Hard pivot violation: expected path under {allowed_prefixes}, got: {stored_path}"
        )
    resolved = Path.cwd() / raw
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Manifest path does not exist: {stored_path} (resolved: {resolved})")


def now_unix() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_stage_status(
    status_dir: Path,
    stage_name: str,
    state: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Write stage status for checkpoint monitoring.

    state in {"pending", "running", "completed", "failed"}.
    """
    ensure_dir(status_dir)
    path = status_dir / f"{stage_name}.json"
    payload = {
        "stage": stage_name,
        "state": state,
        "heartbeat_unix": now_unix(),
        "heartbeat_utc": now_iso(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)

