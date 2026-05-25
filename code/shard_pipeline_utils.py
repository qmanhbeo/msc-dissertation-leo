"""
Shared utilities for shard-based OpenAlex processing.

This module centralises:
  - atomic JSON writes
  - stage status updates (abort/resume visibility)
  - simple checksums
  - lightweight manifest helpers
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


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
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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

