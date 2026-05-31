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


@dataclass(frozen=True)
class CanonicalOutputs:
    root: Path
    tables_dir: Path
    figures_dir: Path


CANONICAL_ROOT_FILES = [
    "coverage_gap.json",
    "coverage_gap_raw.json",
    "semantic_gap.json",
    "semantic_gap_sensitivity.json",
    "h25_correlation.json",
    "h25_scatter.csv",
    "validation_results.json",
    "confusion_matrix.csv",
    "centroid_similarity_matrix.csv",
    "dissertation.pdf",
]

CANONICAL_TABLE_FILES = [
    "num_validation.tex",
    "tab_validation.tex",
    "num_coverage.tex",
    "tab_coverage.tex",
    "num_semantic.tex",
    "tab_semgap.tex",
    "num_h25.tex",
    "tab_h25.tex",
]

CANONICAL_FIGURE_FILES = [
    "fig1_coverage_profiles.pdf",
    "fig1_coverage_profiles.png",
    "fig2_semantic_gap.pdf",
    "fig2_semantic_gap.png",
    "fig3_coverage_semantic_scatter.pdf",
    "fig3_coverage_semantic_scatter.png",
]


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


def ensure_canonical_outputs(output_dir: Path) -> CanonicalOutputs:
    """Create and return the flat canonical output layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return CanonicalOutputs(root=output_dir, tables_dir=tables_dir, figures_dir=figures_dir)


def require_output_files(output_dir: Path, required_files: list[str]) -> Path:
    missing = [r for r in required_files if not (output_dir / r).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Canonical output directory '{output_dir}' is missing: {missing_str}")
    return output_dir


def canonical_artifact_paths(output_dir: Path) -> list[Path]:
    root = Path(output_dir)
    files = [root / name for name in CANONICAL_ROOT_FILES]
    files.extend(root / "tables" / name for name in CANONICAL_TABLE_FILES)
    files.extend(root / "figures" / name for name in CANONICAL_FIGURE_FILES)
    return files


def canonical_artifact_status(output_dir: Path) -> dict[str, list[str]]:
    root = Path(output_dir)
    present: list[str] = []
    missing: list[str] = []
    for path in canonical_artifact_paths(root):
        rel = str(path.relative_to(root))
        if path.exists():
            present.append(rel)
        else:
            missing.append(rel)
    return {"present": present, "missing": missing}
