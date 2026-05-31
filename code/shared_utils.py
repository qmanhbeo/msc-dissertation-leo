from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalOutputs:
    root: Path
    tables_dir: Path
    figures_dir: Path


CANONICAL_ROOT_FILES = [
    "sdg_attention_distribution_document_weighted.json",
    "diagnostic_sdg_attention_distribution_unweighted_chunks.json",
    "sdg_conceptual_alignment_cosine_distances.json",
    "robustness_check_semantic_distances_by_chunk_cap.json",
    "statistical_tests_hypothesis_25_hypothesis_26_and_bias_calibration.json",
    "visualization_source_sdg_attention_vs_semantic_distance.csv",
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
