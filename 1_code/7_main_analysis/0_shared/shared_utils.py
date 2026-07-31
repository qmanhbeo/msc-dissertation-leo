from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from model_utils import model_slug, output_dir_for_model

log = logging.getLogger(__name__)


def _file_fp(path: Path) -> str:
    """Return a content-sensitive fingerprint for a single file (size + mtime + first 64KB)."""
    if not path.exists():
        return "missing"
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    with path.open("rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def fingerprint_of(*paths: Path) -> str:
    """Compute a combined fingerprint of one or more file paths."""
    h = hashlib.sha256()
    for p in paths:
        h.update(_file_fp(p).encode())
    return h.hexdigest()


def _meta_path(primary: Path) -> Path:
    return Path(str(primary) + ".opencode_fp.json")


def should_skip(output_paths: list[Path], fp: str, overwrite: bool, primary: Path) -> bool:
    """Return True iff outputs exist, fingerprint matches, and --overwrite is not set."""
    if overwrite:
        return False
    if not all(p.exists() for p in output_paths):
        return False
    meta = _meta_path(primary)
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text())["fingerprint"] == fp
    except Exception:
        return False


def record_fingerprint(output_paths: list[Path], fp: str, primary: Path) -> None:
    """Write the fingerprint sidecar next to *primary*."""
    _meta_path(primary).write_text(json.dumps({"fingerprint": fp}, indent=2))
    log.info("Fingerprint saved: %s", _meta_path(primary))


@dataclass(frozen=True)
class DissertationOutputs:
    root: Path
    tables_dir: Path
    figures_dir: Path
    data_dir: Path


def _insert_model_in_rel(rel_path: str, model: str | None) -> str:
    """Insert *model* into a relative path under 4_outputs.

    The ``main/`` namespace is flattened — the model slug replaces it
    directly.  The ``appendix/`` namespace keeps the prefix so multiple
    appendix sub-analyses coexist.

    Examples:
        "main/data/file.json" + "mpnet"          → "mpnet/data/file.json"
        "appendix/c/data/f.json" + "mpnet"       → "appendix/mpnet/c/data/f.json"
    """
    if model is None:
        return rel_path
    model = model_slug(model)
    if rel_path.startswith("main/"):
        rest = rel_path[len("main/"):]
        return f"{model}/{rest}"
    if rel_path.startswith("appendix/"):
        rest = rel_path[len("appendix/"):]
        return f"appendix/{model}/{rest}"
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
    "main/data/g_distributional_gap_summary.json",
    "main/data/g_distributional_gap_records.jsonl",
]

MANUSCRIPT_TABLE_FILES = [
    "num_validation.tex",
    "tab_validation.tex",
    "tab_cross_sensitivity_robustness.tex",
    "tab_cross_sensitivity_coverage.tex",
    "num_cross_sensitivity.tex",
    "num_cross_sensitivity_coverage.tex",
    "num_concept_coverage.tex",
    "num_concept_semantic.tex",
    "tab_concept_coverage.tex",
    "num_coverage.tex",
    "tab_coverage.tex",
    "num_semantic.tex",
    "tab_semantic_gap.tex",
    "num_interaction.tex",
    "tab_interaction.tex",
    "num_distributional_gap.tex",
    "tab_distributional_gap.tex",
    "tab_encoder_sensitivity_semantic.tex",
    "tab_encoder_sensitivity_coverage.tex",
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
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_combined.tex",
    "appendix/a3_sdg4_audit/tables/tab_a3_sdg4_lexical_audit.tex",
    "main/tables/num_pca_landscape.tex",
    "appendix/b2_semantic_gap_interpretability/tables/tab_b2_semantic_gap_interpret_all.tex",
    "appendix/f_register_adjustment/tables/num_register_adjustment.tex",
    "appendix/f_register_adjustment/tables/tab_register_adjusted_semgap.tex",
    "appendix/c_sample_stability/tables/num_sample_stability.tex",
    "appendix/c_sample_stability/tables/tab_sample_stability.tex",
    "appendix/h1_cross_method_gap_values/tables/tab_app_cross_method_covgap.tex",
    "appendix/h1_cross_method_gap_values/tables/tab_app_cross_method_semgap.tex",
    "appendix/i1_assignment_method_comparison/tables/tab_app_assignment_method_comparison.tex",
    "main/tables/num_model_selection.tex",
    "main/tables/num_reference_split.tex",
]

MANUSCRIPT_APPENDIX_FIGURE_FILES = [
    "appendix/a4_centroid_similarity/figures/fig_a4_centroid_similarity_heatmap.pdf",
    "appendix/a4_centroid_similarity/figures/fig_a4_centroid_similarity_heatmap.png",
]


def ensure_dissertation_outputs(output_dir: Path, subdir: str = "main", model: str | None = None) -> DissertationOutputs:
    """Create and return the dissertation output layout.

    When *model* is provided the output root becomes::

        output_dir / {model} / data|tables|figures    (when subdir="main")
        output_dir / appendix / {model} / ...         (when subdir="appendix/...")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        parts = subdir.split("/", 1)
        if len(parts) == 2 and parts[0] == "appendix":
            root = output_dir / parts[0] / model_slug(model) / parts[1]
        elif subdir == "main":
            root = output_dir / model_slug(model)
        else:
            root = output_dir / subdir / model_slug(model)
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
        path = output_dir_for_model(model, root=root) / "tables" / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
    for name in MANUSCRIPT_FIGURE_FILES:
        path = output_dir_for_model(model, root=root) / "figures" / name
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
            files.append(output_dir_for_model(model, root=root) / "data" / name)
    for name in MANUSCRIPT_EXTRA_FILES:
        files.append(root / _insert_model_in_rel(name, model))
    for name in MANUSCRIPT_TABLE_FILES:
        files.append(output_dir_for_model(model, root=root) / "tables" / name)
    for name in MANUSCRIPT_FIGURE_FILES:
        files.append(output_dir_for_model(model, root=root) / "figures" / name)
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
