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
    "4_1_validation_results.json",
    "4_1_confusion_matrix.csv",
    "4_1_centroid_similarity_matrix.csv",
    "4_2_coverage_document_weighted.json",
    "4_2_coverage_diagnostic_unweighted.json",
    "4_3_semantic_gap_distances.json",
    "4_3_semantic_gap_robustness_caps.json",
    "4_4_interaction_correlation_asymmetry.json",
    "4_4_interaction_scatter_data.csv",
    "dissertation.pdf",
]

MANUSCRIPT_EXTRA_FILES = [
    "appendix/a1_sdg_source_comparison/data/comparison_summary.json",
    "appendix/a1_sdg_source_comparison/data/comparison_table.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_summary.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_coverage.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_semantic_gaps.csv",
    "appendix/a3_sdg4_audit/data/sdg4_lexical_audit.csv",
    "appendix/a3_sdg4_audit/data/sdg4_lexical_audit_summary.json",
    "appendix/b1_pca_semantic_landscape/data/b1_pca_landscape_metadata.json",
    "appendix/b2_within_corpus_centroid/data/b2_within_corpus_metrics.csv",
    "appendix/b2_within_corpus_centroid/data/b2_within_corpus_summary.json",
    "appendix/b3_semantic_gap_interpretability/data/semantic_gap_distinctive_terms.csv",
    "appendix/b3_semantic_gap_interpretability/data/semantic_gap_interpretability_summary.json",
    "appendix/b3_semantic_gap_interpretability/data/semantic_gap_representative_examples.csv",
    "appendix/b4_softmax_multilabel_sdg/data/b4_softmax_multilabel_coverage.csv",
    "appendix/b4_softmax_multilabel_sdg/data/b4_softmax_multilabel_semantic_gaps.csv",
    "appendix/b4_softmax_multilabel_sdg/data/b4_softmax_multilabel_comparison_summary.csv",
    "appendix/b4_softmax_multilabel_sdg/data/b4_softmax_multilabel_metadata.json",
    "appendix/c_sample_stability/data/4_5_sample_stability_summary.json",
    "appendix/c_sample_stability/data/4_5_sample_stability_draws.jsonl",
    "appendix/c_sample_stability/data/4_5_sample_stability_per_sdg.json",
    "appendix/c_sample_stability/data/4_5_sample_stability_table.csv",
]

MANUSCRIPT_TABLE_FILES = [
    "num_validation.tex",
    "tab_validation.tex",
    "num_coverage.tex",
    "tab_coverage.tex",
    "num_semantic.tex",
    "tab_semantic_gap.tex",
    "num_interaction.tex",
    "tab_interaction.tex",
    "num_sample_stability.tex",
    "tab_sample_stability.tex",
]

MANUSCRIPT_FIGURE_FILES = [
    "fig1_coverage_profiles.pdf",
    "fig1_coverage_profiles.png",
    "fig2_semantic_gap.pdf",
    "fig2_semantic_gap.png",
    "fig3_coverage_semantic_scatter.pdf",
    "fig3_coverage_semantic_scatter.png",
]

MANUSCRIPT_APPENDIX_TABLE_FILES = [
    "appendix/a1_sdg_source_comparison/tables/num_a1_source_comparison.tex",
    "appendix/a1_sdg_source_comparison/tables/tab_a1_source_comparison_f1cos.tex",
    "appendix/a1_sdg_source_comparison/tables/tab_a1_source_comparison_covgap.tex",
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_covshare.tex",
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_gap.tex",
    "appendix/a3_sdg4_audit/tables/tab_a3_sdg4_lexical_audit.tex",
    "appendix/b1_pca_semantic_landscape/tables/num_b1_pca_landscape.tex",
    "appendix/b2_within_corpus_centroid/tables/num_b2_within_corpus_centroid.tex",
    "appendix/b3_semantic_gap_interpretability/tables/tab_b3_semantic_gap_interpret.tex",
    "appendix/b4_softmax_multilabel_sdg/tables/num_b4_softmax_multilabel.tex",
    "appendix/b4_softmax_multilabel_sdg/tables/tab_b4_softmax_summary.tex",
    "appendix/d_register_adjustment/tables/num_register_adjustment.tex",
    "appendix/d_register_adjustment/tables/num_register_confidence_checks.tex",
    "appendix/d_register_adjustment/tables/num_register_interpretability.tex",
    "appendix/d_register_adjustment/tables/num_regression_register_alignment.tex",
    "appendix/d_register_adjustment/tables/num_sdg_register_robustness.tex",
    "appendix/d_register_adjustment/tables/tab_register_adjusted_semgap.tex",
    "appendix/d_register_adjustment/tables/tab_register_confidence_checks.tex",
    "appendix/d_register_adjustment/tables/tab_register_projection_examples.tex",
    "appendix/d_register_adjustment/tables/tab_register_sdg_alignment.tex",
    "appendix/d_register_adjustment/tables/tab_regression_register_alignment.tex",
    "appendix/d_register_adjustment/tables/tab_sdg_register_robustness.tex",
]

MANUSCRIPT_APPENDIX_FIGURE_FILES = [
    "appendix/b1_pca_semantic_landscape/figures/fig_b1_pca_semantic_landscape.pdf",
    "appendix/b1_pca_semantic_landscape/figures/fig_b1_pca_semantic_landscape.png",
    "appendix/b2_within_corpus_centroid/figures/fig_b2_research_sdg_pca.pdf",
    "appendix/b2_within_corpus_centroid/figures/fig_b2_research_sdg_pca.png",
    "appendix/b2_within_corpus_centroid/figures/fig_b2_policy_sdg_pca.pdf",
    "appendix/b2_within_corpus_centroid/figures/fig_b2_policy_sdg_pca.png",
    "appendix/d_register_adjustment/figures/fig_register_adjusted_semantic_gap_comparison.pdf",
    "appendix/d_register_adjustment/figures/fig_register_adjusted_semantic_gap_comparison.png",
    "appendix/d_register_adjustment/figures/fig_register_confidence_curve.pdf",
    "appendix/d_register_adjustment/figures/fig_register_confidence_curve.png",
    "appendix/d_register_adjustment/figures/fig_register_projection_distribution.pdf",
    "appendix/d_register_adjustment/figures/fig_register_projection_distribution.png",
    "appendix/d_register_adjustment/figures/fig_register_sdg_alignment.pdf",
    "appendix/d_register_adjustment/figures/fig_register_sdg_alignment.png",
    "appendix/d_register_adjustment/figures/fig_regression_vs_classifier_alignment.pdf",
    "appendix/d_register_adjustment/figures/fig_regression_vs_classifier_alignment.png",
    "appendix/d_register_adjustment/figures/fig_sdg_register_robustness_comparison.pdf",
    "appendix/d_register_adjustment/figures/fig_sdg_register_robustness_comparison.png",
]


def ensure_dissertation_outputs(output_dir: Path, subdir: str = "main") -> DissertationOutputs:
    """Create and return the nested dissertation output layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / subdir
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
    for name in MANUSCRIPT_EXTRA_FILES:
        path = root / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_APPENDIX_TABLE_FILES:
        path = root / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_APPENDIX_FIGURE_FILES:
        path = root / name
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
    files.extend(root / name for name in MANUSCRIPT_APPENDIX_TABLE_FILES)
    files.extend(root / name for name in MANUSCRIPT_APPENDIX_FIGURE_FILES)
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
