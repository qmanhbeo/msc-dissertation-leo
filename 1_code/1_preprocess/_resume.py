"""
Shared resume-safe checkpointing for the small preprocess scripts.

This helper lets the otherwise pure per-record map scripts
(0_preprocess_*.py, 0_preprocess_ungdc_sdg.py) gain true interrupt-and-resume
safety without changing their transform logic or output format.

Design (correctness over micro-optimisation):
  - Input is read ordered; a single global counter ``rows_done`` advances by
    exactly one for every input record yielded (including dropped records).
    This guarantees that streaming transforms which encode the input position
    into an id (e.g. sdgi ``f"sdgi_{idx:05d}"``) reproduce identical ids.
  - Kept records accumulate in a pending buffer; every ``chunk_size`` records
    the buffer is appended to the output file, fsync'd, and only THEN is the
    checkpoint state advanced. A crash mid-append leaves a torn tail that the
    next run heals by truncating the output back to the last committed byte
    offset before resuming.
  - The output file always equals the concatenation of the first ``rows_kept``
    records for the first ``rows_done`` input records. On resume we truncate
    any trailing junk and re-derive from ``rows_done`` => no duplicates, no
    loss, byte-identical to an uninterrupted run.

Only 1_build_policy_corpus.py does NOT use this helper: it overwrites its own
inputs in place and is implemented transactional-atomically instead.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shard_pipeline_utils import ensure_dir, read_json, update_stage_status

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _append_records(out_path: Path, records: list[dict], dumps: Callable[[dict], str]) -> int:
    """Append json lines to ``out_path``; return the number of bytes written."""
    bytes_written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for r in records:
            line = dumps(r) + "\n"
            f.write(line)
            bytes_written += len(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    return bytes_written


def resumable_records(
    *,
    stage: str,
    read_records: Callable[[], Iterable[Any]],
    transform: Callable[[Any], dict | None],
    out_path: Path,
    state_path: Path,
    status_dir: Path,
    chunk_size: int = 5000,
    reset: bool = False,
    finalize: Callable[[Path], None] | None = None,
    dumps: Callable[[dict], str] = json.dumps,
) -> dict:
    """Run a resume-safe per-record map from ``read_records`` -> ``out_path``.

    ``transform`` maps one raw input record to an output dict, or returns
    ``None`` to drop it. The checkpoint state at ``state_path`` records how far
    we got; re-invoking resumes from there.
    """
    ensure_dir(out_path.parent)
    ensure_dir(status_dir)

    if reset:
        for p in (state_path, out_path):
            if p.exists():
                p.unlink()

    state = read_json(
        state_path,
        default={
            "schema_version": SCHEMA_VERSION,
            "rows_done": 0,
            "rows_kept": 0,
            "rows_dropped": 0,
            "out_offset": 0,
        },
    )
    state.setdefault("schema_version", SCHEMA_VERSION)
    state.setdefault("rows_done", 0)
    state.setdefault("rows_kept", 0)
    state.setdefault("rows_dropped", 0)
    state.setdefault("out_offset", 0)

    # Heal a torn tail from a previous crash: the valid prefix is exactly
    # state["out_offset"] bytes. Truncate back to it, then resume.
    if out_path.exists():
        file_size = out_path.stat().st_size
        if file_size > state["out_offset"]:
            with out_path.open("r+b") as f:
                f.truncate(state["out_offset"])
        elif file_size < state["out_offset"]:
            # Catastrophic: a prefix was lost. Restart this stage from scratch
            # (the truncated file is consistent with rows_done=0 => no dup/loss).
            log.warning("Output shorter than recorded offset; restarting %s from 0.", stage)
            out_path.unlink()
            state = {
                "schema_version": SCHEMA_VERSION,
                "rows_done": 0,
                "rows_kept": 0,
                "rows_dropped": 0,
                "out_offset": 0,
            }

    update_stage_status(
        status_dir, stage, "running",
        {"output": str(out_path), "state": str(state_path), "rows_done": int(state["rows_done"])},
    )

    rows_done = int(state["rows_done"])
    rows_kept = int(state["rows_kept"])
    rows_dropped = int(state["rows_dropped"])
    pending: list[dict] = []

    def commit() -> None:
        nonlocal rows_kept, rows_dropped
        if not pending:
            return
        written = _append_records(out_path, pending, dumps)
        # State advances ONLY after a successful append + fsync.
        state["out_offset"] = int(state["out_offset"]) + written
        state["rows_done"] = rows_done
        state["rows_kept"] = rows_kept
        state["rows_dropped"] = rows_dropped
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, state_path)
        pending.clear()

    try:
        for raw in itertools.islice(read_records(), rows_done, None):
            rec = transform(raw)
            rows_done += 1
            if rec is None:
                rows_dropped += 1
            else:
                pending.append(rec)
                rows_kept += 1
            if len(pending) >= chunk_size:
                commit()
        commit()
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
        if finalize is not None:
            finalize(out_path)
        update_stage_status(
            status_dir, stage, "completed",
            {"rows_done": rows_done, "rows_kept": rows_kept, "rows_dropped": rows_dropped},
        )
    except Exception:
        update_stage_status(
            status_dir, stage, "failed",
            {"rows_done": rows_done, "rows_kept": rows_kept, "rows_dropped": rows_dropped},
        )
        raise

    log.info(
        "%s done: total_processed=%d kept=%d dropped=%d",
        stage, rows_done, rows_kept, rows_dropped,
    )
    return state


if __name__ == "__main__":
    pass
