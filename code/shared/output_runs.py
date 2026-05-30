from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class RunSelection:
    run_dir: Path
    run_name: str
    created_new: bool


def sanitize_label(label: str) -> str:
    """Make a filesystem-safe label segment for run folder names."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (label or "analysis").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "analysis"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_name(label: str) -> str:
    return f"{timestamp_slug()}_{sanitize_label(label)}"


def latest_run_dir(output_root: Path, required_files: list[str] | None = None) -> Path | None:
    """
    Return latest run directory in output_root satisfying required_files.

    Latest is determined by directory mtime (descending).
    """
    if not output_root.exists():
        return None

    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if required_files:
        req = [Path(r) for r in required_files]
        candidates = [p for p in candidates if all((p / r).exists() for r in req)]

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run_dir(
    output_root: Path,
    run_name: str | None,
    run_label: str,
    *,
    prefer_latest_if_missing: bool = False,
    prefer_latest_required_files: list[str] | None = None,
) -> RunSelection:
    """
    Resolve destination run directory.

    Priority:
      1) explicit run_name
      2) latest compatible run (if prefer_latest_if_missing=True)
      3) new timestamped run
    """
    output_root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = output_root / run_name
        created_new = not run_dir.exists()
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunSelection(run_dir=run_dir, run_name=run_dir.name, created_new=created_new)

    if prefer_latest_if_missing:
        latest = latest_run_dir(output_root, required_files=prefer_latest_required_files)
        if latest is not None:
            return RunSelection(run_dir=latest, run_name=latest.name, created_new=False)

    new_dir = output_root / default_run_name(run_label)
    new_dir.mkdir(parents=True, exist_ok=True)
    return RunSelection(run_dir=new_dir, run_name=new_dir.name, created_new=True)


def require_run_with_files(output_root: Path, run_name: str, required_files: list[str]) -> Path:
    run_dir = output_root / run_name
    missing = [r for r in required_files if not (run_dir / r).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Run '{run_name}' is missing required files: {missing_str}")
    return run_dir
