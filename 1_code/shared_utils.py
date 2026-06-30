from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DissertationOutputs:
    root: Path
    tables_dir: Path
    figures_dir: Path
    data_dir: Path


CanonicalOutputs = DissertationOutputs


MANUSCRIPT_ROOT_FILES = [
    "sdg_attention_distribution_document_weighted.json",
    "diagnostic_sdg_attention_distribution_unweighted_segments.json",
    "sdg_conceptual_alignment_cosine_distances.json",
    "robustness_check_semantic_distances_by_segment_cap.json",
    "statistical_tests_hypothesis_25_hypothesis_26_and_bias_calibration.json",
    "visualization_source_sdg_attention_vs_semantic_distance.csv",
    "sample_stability_summary.json",
    "sample_stability_draws.jsonl",
    "sample_stability_per_sdg.json",
    "sample_stability_table.csv",
    "validation_results.json",
    "confusion_matrix.csv",
    "centroid_similarity_matrix.csv",
    "dissertation.pdf",
]

MANUSCRIPT_EXTRA_FILES = [
    "appendix/source_family_sensitivity/policy_source_family_summary.csv",
    "appendix/source_family_sensitivity/policy_source_family_coverage.csv",
    "appendix/source_family_sensitivity/policy_source_family_semantic_gaps.csv",
    "appendix/sdg4_audit/sdg4_lexical_audit.csv",
    "appendix/sdg4_audit/sdg4_lexical_audit_summary.json",
    "appendix/semantic_gap_interpretability/semantic_gap_distinctive_terms.csv",
    "appendix/semantic_gap_interpretability/semantic_gap_interpretability_summary.json",
    "appendix/semantic_gap_interpretability/semantic_gap_representative_examples.csv",
]

MANUSCRIPT_TABLE_FILES = [
    "pca_semantic_landscape_metadata.json",
    "num_pca_semantic_landscape.tex",
    "within_corpus_centroid_structure_metrics.csv",
    "within_corpus_centroid_structure_summary.json",
    "num_within_corpus_centroid_structure.tex",
    "softmax_multilabel_coverage.csv",
    "softmax_multilabel_semantic_gaps.csv",
    "softmax_multilabel_comparison_summary.csv",
    "softmax_multilabel_metadata.json",
    "num_softmax_multilabel.tex",
    "tab_softmax_multilabel_summary.tex",
    "tab_policy_source_family_sensitivity.tex",
    "tab_sdg4_lexical_audit.tex",
    "num_validation.tex",
    "tab_validation.tex",
    "num_coverage.tex",
    "tab_coverage.tex",
    "num_semantic.tex",
    "tab_semgap.tex",
    "num_h25.tex",
    "tab_h25.tex",
    "num_sample_stability.tex",
    "tab_sample_stability.tex",
]

MANUSCRIPT_FIGURE_FILES = [
    "fig_pca_semantic_landscape.pdf",
    "fig_pca_semantic_landscape.png",
    "fig_within_corpus_research_sdg_pca.pdf",
    "fig_within_corpus_research_sdg_pca.png",
    "fig_within_corpus_policy_sdg_pca.pdf",
    "fig_within_corpus_policy_sdg_pca.png",
    "fig1_coverage_profiles.pdf",
    "fig1_coverage_profiles.png",
    "fig2_semantic_gap.pdf",
    "fig2_semantic_gap.png",
    "fig3_coverage_semantic_scatter.pdf",
    "fig3_coverage_semantic_scatter.png",
]


def ensure_dissertation_outputs(output_dir: Path) -> DissertationOutputs:
    """Create and return the nested dissertation output layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / "main"
    data_dir = root / "data"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    for path in (data_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return DissertationOutputs(root=root, tables_dir=tables_dir, figures_dir=figures_dir, data_dir=data_dir)


def ensure_canonical_outputs(output_dir: Path) -> DissertationOutputs:
    return ensure_dissertation_outputs(output_dir)


def require_output_files(output_dir: Path, required_files: list[str]) -> Path:
    missing = [r for r in required_files if not (output_dir / r).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Manuscript output directory '{output_dir}' is missing: {missing_str}")
    return output_dir


def require_pdf_inputs(output_dir: Path) -> Path:
    root = Path(output_dir)
    missing = []
    for name in MANUSCRIPT_TABLE_FILES:
        path = root / "main" / "tables" / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_FIGURE_FILES:
        path = root / "main" / "figures" / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Manuscript output directory '{root}' is missing PDF inputs: {missing_str}"
        )
    return root


def canonical_artifact_paths(output_dir: Path) -> list[Path]:
    root = Path(output_dir)
    files = []
    for name in MANUSCRIPT_ROOT_FILES:
        if name == "dissertation.pdf":
            files.append(root / name)
        else:
            files.append(root / "main" / "data" / name)
    files.extend(root / name for name in MANUSCRIPT_EXTRA_FILES)
    files.extend(root / "main" / "tables" / name for name in MANUSCRIPT_TABLE_FILES)
    files.extend(root / "main" / "figures" / name for name in MANUSCRIPT_FIGURE_FILES)
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
