from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DissertationOutputs:
    root: Path
    tables_dir: Path
    figures_dir: Path
    data_dir: Path


def _insert_model_in_rel(rel_path: str, model: str | None) -> str:
    """Insert *model* after the top-level namespace prefix (main|appendix).

    Examples:
        "main/data/file.json" + "all-mpnet-base-v2"  → "main/all-mpnet-base-v2/data/file.json"
        "appendix/c/data/f.json" + "all-mpnet-base-v2" → "appendix/all-mpnet-base-v2/c/data/f.json"
    """
    if model is None:
        return rel_path
    for prefix in ("main/", "appendix/"):
        if rel_path.startswith(prefix):
            ns = prefix.rstrip("/")
            rest = rel_path[len(prefix):]
            return f"{ns}/{model}/{rest}"
    return rel_path




MANUSCRIPT_ROOT_FILES = [
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
    "appendix/a2_source_family_sensitivity/data/policy_source_family_summary.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_coverage.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_semantic_gaps.csv",
    "appendix/a3_sdg4_audit/data/sdg4_lexical_audit.csv",
    "appendix/a3_sdg4_audit/data/sdg4_lexical_audit_summary.json",
    "main/data/pca_landscape_metadata.json",
    "appendix/b2_semantic_gap_interpretability/data/semantic_gap_distinctive_terms.csv",
    "appendix/b2_semantic_gap_interpretability/data/semantic_gap_interpretability_summary.json",
    "appendix/b2_semantic_gap_interpretability/data/semantic_gap_representative_examples.csv",
    "appendix/c_sample_stability/data/c_sample_stability_summary.json",
    "appendix/c_sample_stability/data/c_sample_stability_draws.jsonl",
    "appendix/c_sample_stability/data/c_sample_stability_per_sdg.json",
    "appendix/c_sample_stability/data/c_sample_stability_table.csv",
]

MANUSCRIPT_TABLE_FILES = [
    "num_validation.tex",
    "tab_validation.tex",
    "tab_cross_sensitivity_robustness.tex",
    "num_cross_sensitivity.tex",
    "num_coverage.tex",
    "tab_coverage.tex",
    "num_semantic.tex",
    "tab_semantic_gap.tex",
    "num_interaction.tex",
    "tab_interaction.tex",
]

MANUSCRIPT_FIGURE_FILES = [
    "fig1_pca_semantic_landscape.pdf",
    "fig1_pca_semantic_landscape.png",
    "fig3_coverage_profiles.pdf",
    "fig3_coverage_profiles.png",
    "fig4_semantic_gap.pdf",
    "fig4_semantic_gap.png",
    "fig5_coverage_semantic_scatter.pdf",
    "fig5_coverage_semantic_scatter.png",
]

MANUSCRIPT_APPENDIX_TABLE_FILES = [
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_covshare.tex",
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_gap.tex",
    "appendix/a3_sdg4_audit/tables/tab_a3_sdg4_lexical_audit.tex",
    "main/tables/num_pca_landscape.tex",
    "appendix/b2_semantic_gap_interpretability/tables/tab_b2_semantic_gap_interpret.tex",
    "appendix/f_register_adjustment/tables/num_register_adjustment.tex",
    "appendix/f_register_adjustment/tables/num_register_confidence_checks.tex",
    "appendix/f_register_adjustment/tables/num_register_interpretability.tex",
    "appendix/f_register_adjustment/tables/num_regression_register_alignment.tex",
    "appendix/f_register_adjustment/tables/num_sdg_register_robustness.tex",
    "appendix/f_register_adjustment/tables/tab_register_adjusted_semgap.tex",
    "appendix/f_register_adjustment/tables/tab_register_confidence_checks.tex",
    "appendix/f_register_adjustment/tables/tab_register_projection_examples.tex",
    "appendix/f_register_adjustment/tables/tab_register_sdg_alignment.tex",
    "appendix/f_register_adjustment/tables/tab_regression_register_alignment.tex",
    "appendix/f_register_adjustment/tables/tab_sdg_register_robustness.tex",
    "appendix/c_sample_stability/tables/num_sample_stability.tex",
    "appendix/c_sample_stability/tables/tab_sample_stability.tex",
]

MANUSCRIPT_APPENDIX_FIGURE_FILES = [
    "appendix/a4_centroid_similarity/figures/fig_a4_centroid_similarity_heatmap.pdf",
    "appendix/a4_centroid_similarity/figures/fig_a4_centroid_similarity_heatmap.png",
]


def ensure_dissertation_outputs(output_dir: Path, subdir: str = "main", model: str | None = None) -> DissertationOutputs:
    """Create and return the nested dissertation output layout.

    When *model* is provided the output root becomes::

        output_dir / subdir / {model} / data|tables|figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        parts = subdir.split("/", 1)
        if len(parts) == 2 and parts[0] in ("main", "appendix"):
            root = output_dir / parts[0] / model / parts[1]
        else:
            root = output_dir / subdir / model
    else:
        root = output_dir / subdir
    data_dir = root / "data"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    for path in (data_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return DissertationOutputs(root=root, tables_dir=tables_dir, figures_dir=figures_dir, data_dir=data_dir)


def ensure_canonical_outputs(output_dir: Path, model: str | None = None) -> DissertationOutputs:
    return ensure_dissertation_outputs(output_dir, subdir="main", model=model)


def require_output_files(output_dir: Path, required_files: list[str]) -> Path:
    missing = [r for r in required_files if not (output_dir / r).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Manuscript output directory '{output_dir}' is missing: {missing_str}")
    return output_dir


def require_pdf_inputs(output_dir: Path, model: str | None = None) -> Path:
    root = Path(output_dir)
    missing = []
    for name in MANUSCRIPT_TABLE_FILES:
        path = root / "main" / (model or "") / "tables" / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_FIGURE_FILES:
        path = root / "main" / (model or "") / "figures" / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_EXTRA_FILES:
        path = root / _insert_model_in_rel(name, model)
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_APPENDIX_TABLE_FILES:
        path = root / _insert_model_in_rel(name, model)
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_APPENDIX_FIGURE_FILES:
        path = root / _insert_model_in_rel(name, model)
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Manuscript output directory '{root}' is missing PDF inputs: {missing_str}"
        )
    return root


def canonical_artifact_paths(output_dir: Path, model: str | None = None) -> list[Path]:
    root = Path(output_dir)
    files = []
    for name in MANUSCRIPT_ROOT_FILES:
        if name == "dissertation.pdf":
            files.append(root / name)
        else:
            files.append(root / "main" / (model or "") / "data" / name)
    for name in MANUSCRIPT_EXTRA_FILES:
        files.append(root / _insert_model_in_rel(name, model))
    for name in MANUSCRIPT_TABLE_FILES:
        files.append(root / "main" / (model or "") / "tables" / name)
    for name in MANUSCRIPT_FIGURE_FILES:
        files.append(root / "main" / (model or "") / "figures" / name)
    for name in MANUSCRIPT_APPENDIX_TABLE_FILES:
        files.append(root / _insert_model_in_rel(name, model))
    for name in MANUSCRIPT_APPENDIX_FIGURE_FILES:
        files.append(root / _insert_model_in_rel(name, model))
    return files


def canonical_artifact_status(output_dir: Path, model: str | None = None) -> dict[str, list[str]]:
    root = Path(output_dir)
    present: list[str] = []
    missing: list[str] = []
    for path in canonical_artifact_paths(root, model=model):
        rel = str(path.relative_to(root))
        if path.exists():
            present.append(rel)
        else:
            missing.append(rel)
    return {"present": present, "missing": missing}
