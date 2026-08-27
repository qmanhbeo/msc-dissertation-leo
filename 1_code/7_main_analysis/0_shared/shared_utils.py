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
    return primary.with_name(primary.stem + ".fingerprint.json")


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
    "confusion_matrix_lr.csv",
    "centroid_similarity_matrix.csv",
    "coverage_document_weighted.json",
    "coverage_diagnostic_unweighted.json",
    "semantic_gap_distances_lr.json",
    "semantic_gap_robustness_caps_lr.json",
    "interaction_h25.json",
    "interaction_scatter_data.csv",
    "dissertation.pdf",
]

MANUSCRIPT_EXTRA_FILES = [
    "appendix/a2_source_family_sensitivity/data/policy_source_family_summary.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_coverage.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_semantic_gaps.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_h25.csv",
    "appendix/a2_source_family_sensitivity/data/policy_source_family_h25.json",
    "appendix/a1_register_validation/data/register_validation.json",
    "appendix/a1_register_validation/data/register_validation.csv",
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
    "appendix/c1_subset_balanced_stability/data/c1_subset_balanced_stability.json",
]

MANUSCRIPT_TABLE_FILES = [
    "num1_classifier_performance.tex",
    "tab1_classifier_performance.tex",
    "tab6a_cross_sensitivity.tex",
    "tab6b_cross_sensitivity.tex",
    "tab8_coverage_sensitivity.tex",
    "num6_cross_sensitivity.tex",
    "num8_coverage_sensitivity.tex",
    "num10_concept_coverage.tex",
    "num11_concept_semantic.tex",
    "tab10_concept_coverage.tex",
    "num2_coverage_gap.tex",
    "tab2_coverage_gap.tex",
    "num3_semantic_gap.tex",
    "tab3_semantic_gap.tex",
    "num4_interaction_h25.tex",
    "tab4_interaction_h25.tex",
    "num13_distributional_gap.tex",
    "tab13_distributional_gap.tex",
    "num18_distributional_h1.tex",
    "tab14_distributional_h1.tex",
    "tab7a_encoder_sensitivity.tex",
    "tab7b_encoder_sensitivity.tex",
    "tab9_encoder_sensitivity_coverage.tex",
    "tab5_register_decomposition.tex",
    "num5_register_decomposition.tex",
]

MANUSCRIPT_FIGURE_FILES = [
    "fig7_pca_semantic_landscape.pdf",
    "fig7_pca_semantic_landscape.png",
    "fig3_pca_register_before_after.pdf",
    "fig3_pca_register_before_after.png",
    "fig2_coverage_profiles.pdf",
    "fig2_coverage_profiles.png",
    "fig4_semantic_gap.pdf",
    "fig4_semantic_gap.png",
    "fig9_h1a_scatter.pdf",
    "fig9_h1a_scatter.png",
    "fig9_h1b_scatter.pdf",
    "fig9_h1b_scatter.png",
    "fig9_h1c_scatter.pdf",
    "fig9_h1c_scatter.png",
    "fig9_h1d_scatter.pdf",
    "fig9_h1d_scatter.png",
]

MANUSCRIPT_APPENDIX_TABLE_FILES = [
    "appendix/a1_register_validation/tables/tab_a1_register_validation.tex",
    "appendix/a1_register_validation/tables/tab_a1_register_validation_selectivity.tex",
    "appendix/a1_register_validation/tables/num_a1_register_validation.tex",
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_combined.tex",
    "appendix/a2_source_family_sensitivity/tables/tab_a2_policy_source_family_h25.tex",
    "appendix/a2_source_family_sensitivity/tables/num_a2_policy_source_family_h25.tex",
    "appendix/a3_sdg4_audit/tables/tab_a3_sdg4_lexical_audit.tex",
    "main/tables/num14_pca_landscape.tex",
    "appendix/b2_semantic_gap_interpretability/tables/tab_b2_semantic_gap_interpret_all.tex",
    "main/tables/num12_register_check.tex",
    "main/tables/tab12_register_check.tex",
    "appendix/c_sample_stability/tables/num_c_sample_stability.tex",
    "appendix/c_sample_stability/tables/tab_c_sample_stability.tex",
    "appendix/c1_subset_balanced_stability/tables/num_c1_subset_stability.tex",
    "appendix/h1_cross_method_gap_values/tables/tab_app_cross_method_covgap.tex",
    "appendix/h1_cross_method_gap_values/tables/tab_app_cross_method_semgap.tex",
    "appendix/i1_assignment_method_comparison/tables/tab_app_assignment_method_comparison.tex",
    "appendix/j1_raw_value_correlation/tables/tab_j1_raw_value_correlation.tex",
    "main/tables/num16_model_selection.tex",
    "main/tables/num17_reference_split.tex",
]

MANUSCRIPT_APPENDIX_FIGURE_FILES = [
    "appendix/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.pdf",
    "appendix/a4_centroid_similarity/figures/fig8_centroid_similarity_heatmap.png",
]

# Model-independent conceptual figures (TikZ + matplotlib) generated by
# 1_code/8_visualization/build_conceptual_figs.py into 4_outputs/conceptual_figs/.
# These are NOT model-namespaced, so they are checked against the output root,
# not per-model dirs. declare_pdf_inputs() enforces them so that --build-pdf /
# --build-word fail closed with a clear message when the analysis has not run.
MANUSCRIPT_CONCEPTUAL_FIGURE_FILES = [
    "conceptual_figs/fig1_conceptual_framework.pdf",
    "conceptual_figs/fig1_conceptual_framework.png",
    "conceptual_figs/fig6_pipeline_flowchart.pdf",
    "conceptual_figs/fig6_pipeline_flowchart.png",
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


# The distributional-gap tables (Appendix G) are written by g_distributional_gap.py
# into 4_outputs/{model}/adjusted/tables/ (the adjusted embedding track), which is
# also where dissertation.tex reads them from. The guard below accepts either
# location so build-pdf does not require a duplicate copy under tables/.
DISTRIBUTIONAL_TABLES = {
    "num13_distributional_gap.tex", "tab13_distributional_gap.tex",
    "num18_distributional_h1.tex", "tab14_distributional_h1.tex",
}


def require_pdf_inputs(output_dir: Path, model: str | None = None) -> Path:
    root = Path(output_dir)
    missing = []
    for name in MANUSCRIPT_TABLE_FILES:
        path = output_dir_for_model(model, root=root) / "tables" / name
        if not path.exists():
            # distributional tables live under adjusted/tables
            if name in DISTRIBUTIONAL_TABLES and (
                output_dir_for_model(model, root=root) / "adjusted" / "tables" / name
            ).exists():
                continue
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
    for name in MANUSCRIPT_CONCEPTUAL_FIGURE_FILES:
        path = root / name
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
    mpnet_tables = output_dir_for_model(model, root=root) / "tables"
    adj_tables = output_dir_for_model(model, root=root) / "adjusted" / "tables"
    files = []
    for name in MANUSCRIPT_ROOT_FILES:
        if name == "dissertation.pdf":
            files.append(root / name)
        else:
            files.append(output_dir_for_model(model, root=root) / "data" / name)
    for name in MANUSCRIPT_EXTRA_FILES:
        files.append(root / _insert_model_in_rel(name, model))
    for name in MANUSCRIPT_TABLE_FILES:
        path = mpnet_tables / name
        # Distributional-gap tables are written by g_distributional_gap.py into
        # adjusted/tables (the adjusted track); accept either location so
        # status reporting matches require_pdf_inputs.
        if name in DISTRIBUTIONAL_TABLES and adj_tables.joinpath(name).exists():
            path = adj_tables / name
        files.append(path)
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


# ---------------------------------------------------------------------------
# Permutation (Monte Carlo) p-values for Pearson/Spearman correlations
# ---------------------------------------------------------------------------
# The 17 SDGs are a fully enumerated population, not a sample, so analytic
# t-approximations (which assume sampling from a superpopulation) do not fit
# the paper's framing. A permutation test holds one measured vector fixed,
# permutes the other, and conditions on the observed values: its null is "no
# linear/monotonic association between the measured quantities" and needs no
# sampling assumption. Deterministic: draws use np.random.default_rng(seed).
PERMUTATION_N_RESAMPLES = 100_000
PERMUTATION_SEED = 42


def permutation_p(x, y, kind: str = "spearman",
                  n_resamples: int = PERMUTATION_N_RESAMPLES,
                  seed: int = PERMUTATION_SEED):
    """Two-sided Monte Carlo permutation p-value for a Pearson/Spearman correlation.

    Holds ``x`` fixed and permutes ``y`` (ranks for Spearman, values for Pearson);
    the null is "no linear/monotonic association between the measured values",
    conditioning on the observed data. No superpopulation sampling assumption —
    fits the 17-SDG fully-enumerated-population framing.

    Returns ``(stat, p)``: ``stat`` is computed exactly as scipy computes it, so
    the reported statistics are unchanged; ``p = (count + 1) / (n_resamples + 1)``.
    Deterministic given ``seed``.
    """
    import numpy as np
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    n = x.shape[0]
    if n < 3:
        raise ValueError("need at least 3 observations")
    if seed is None or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an int (got {seed!r}) — permutation draws must be deterministic")

    if kind == "spearman":
        stat, _ = stats.spearmanr(x, y)
        rx = stats.rankdata(x)
        denom = n * (n * n - 1)
        rng = np.random.default_rng(seed)
        perm = np.argsort(rng.random((n_resamples, n)), axis=1) + 1
        rho_perm = 1.0 - 6.0 * ((rx[None, :] - perm) ** 2).sum(axis=1) / denom
        n_ext = int(np.sum(np.abs(rho_perm) >= abs(stat)))
    elif kind == "pearson":
        stat, _ = stats.pearsonr(x, y)
        xc = x - x.mean()
        yc = y - y.mean()
        denom = (n - 1) * x.std(ddof=1) * y.std(ddof=1)
        if denom == 0:
            raise ValueError("zero variance in x or y")
        rng = np.random.default_rng(seed)
        perm_y = y[np.argsort(rng.random((n_resamples, n)), axis=1)]
        r_perm = (perm_y - y.mean()) @ xc / denom
        n_ext = int(np.sum(np.abs(r_perm) >= abs(stat)))
    else:
        raise ValueError(f"kind must be 'spearman' or 'pearson', got {kind!r}")

    p = (n_ext + 1) / (n_resamples + 1)
    return stat, p
