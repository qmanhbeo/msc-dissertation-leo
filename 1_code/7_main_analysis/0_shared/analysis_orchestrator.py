"""
Analysis orchestrator: run in-process main-text and appendix analyses.

Each analysis was previously a separate `subprocess`, re-opening the 27-shard
research data.  The orchestrator drives each script's `main()` in-process, and
every script reads the research embedding/score shards directly (shard-native,
mmap) — so no consolidated array is built or cached.

Coverage gap, semantic gap, and PCA are now subprocess steps in the linear
pipeline (run_linear_pipeline in main.py) — NOT run here.  This orchestrator
handles:
  - MAIN_STEPS:        in-process interaction analysis
  - APPENDIX_STEPS:    in-process appendix analyses
  - POST_ADJUSTED:     in-process register decomposition, extended interaction,
                       correlation table, consolidated macros, PCA before/after
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from model_utils import DEFAULT_EMBED_MODEL

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]  # 7_main_analysis

# In-process main-text analyses (coverage_gap, semantic_gap, PCA moved to
# subprocess steps in the linear pipeline).
MAIN_STEPS = [
    ("1_main_text/2_coverage_semantic_interaction.py", False),
]

# Single source of truth for every appendix identity. Canonical order:
# A2, A3, B2, C, C1, C0, D1, H1, I1, G(opt-in). `in_all` flags whether the
# script participates in `--appendix-all` (G is opt-in only).
APPENDIX_SPECS = [
    {
        "flag": "appendix-a2-family",
        "aliases": ["policy-source-family-sensitivity"],
        "script": "2_appendix/a2_policy_source_family_sensitivity.py",
        "help": "Run A.2 Policy Source-Family Sensitivity.",
        "warn": "Policy source-family sensitivity (Appendix A.2)",
        "run_label": "policy source-family sensitivity",
        "step_id": "A2",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-a3-sdg4",
        "aliases": ["sdg4-lexical-audit"],
        "script": "2_appendix/a3_sdg4_lexical_audit.py",
        "help": "Run A.3 SDG 4 Lexical Artefact Audit.",
        "warn": "SDG 4 lexical artefact audit (Appendix A.3)",
        "run_label": "SDG 4 lexical artefact audit",
        "step_id": "A3",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-b2-interpret",
        "aliases": ["semantic-gap-interpretability"],
        "script": "2_appendix/b2_semantic_gap_text_interpretability.py",
        "help": "Run B.1 Lexical Illustration of the Semantic Gap.",
        "warn": "Semantic-gap interpretability (Appendix B.2)",
        "run_label": "lexical illustration of the semantic gap",
        "step_id": "B2",
        "in_all": True,
        "requires": ["4_3_semantic_gap_distances.json"],
    },
    {
        "flag": "appendix-c-sample-stability",
        "aliases": [],
        "script": "2_appendix/c_sample_stability.py",
        "help": "Run C Sample-Stability Robustness (appendix).",
        "warn": "Sample-stability robustness (Appendix C)",
        "run_label": "sample stability",
        "step_id": "C",
        "in_all": True,
        "requires": [
            "4_2_coverage_document_weighted.json",
            "4_3_semantic_gap_distances.json",
            "4_4_interaction_correlation_asymmetry.json",
        ],
    },
    {
        "flag": "appendix-c1-balanced-subset",
        "aliases": [],
        "script": "2_appendix/c1_subset_balanced_stability.py",
        "help": "Run C.1 Balanced-Subset Rank-Stability (consumes C sample-stability draws; appendix).",
        "warn": "Balanced-subset rank stability (Appendix C.1)",
        "run_label": "balanced-subset rank stability",
        "step_id": "C1",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-c0-corpus-split",
        "aliases": [],
        "script": "2_appendix/c0_export_corpus_split_sizes.py",
        "help": "Export reference-corpus split-size macros.",
        "warn": "Corpus split macro export (Appendix C.0)",
        "run_label": "corpus split macro export",
        "step_id": "C0",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-d1-model-selection",
        "aliases": [],
        "script": "2_appendix/d1_export_model_selection_nums.py",
        "help": "Export D.1 model-selection CV macros.",
        "warn": "Model-selection macro export (Appendix D.1)",
        "run_label": "model-selection macro export",
        "step_id": "D1",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-h1-cross-method",
        "aliases": [],
        "script": "2_appendix/h1_cross_method_gap_values.py",
        "help": "Run H.1 Cross-Method Gap Values.",
        "warn": "Cross-method gap values (Appendix H.1)",
        "run_label": "cross-method gap values",
        "step_id": "H1",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-i1-assignment-method",
        "aliases": [],
        "script": "2_appendix/i1_assignment_method_comparison.py",
        "help": "Run I.1 Supervised vs Nearest-Centroid Assignment Comparison.",
        "warn": "Assignment-method comparison (Appendix I.1)",
        "run_label": "assignment-method comparison",
        "step_id": "I1",
        "in_all": True,
        "requires": None,
    },
    {
        "flag": "appendix-g-distributional",
        "aliases": [],
        "script": "1_main_text/g_distributional_gap.py",
        "help": "Run the distributional semantic-gap robustness (MAIN-RESULT Table; OPT-IN: not run by warm replay or --appendix-all; run before --build-pdf).",
        "warn": "Distributional gap (Appendix G)",
        "run_label": "distributional semantic-gap metrics",
        "step_id": "G",
        "in_all": False,
        "requires": ["4_3_semantic_gap_distances.json"],
    },
]

APPENDIX_STEPS = [(spec["script"], False) for spec in APPENDIX_SPECS if spec["in_all"]]

# Post-adjusted generators: run AFTER adjusted JSONs exist.
# These read the raw + adjusted JSONs and produce decomposition tables,
# extended interaction JSONs, and consolidated LaTeX macros.
POST_ADJUSTED_STEPS = [
    "0_shared/g_register_decomposition.py",
    "0_shared/g_interaction_extended.py",
    "0_shared/h1_register_correlation_table.py",
    "0_shared/generate_tex_macros.py",
    ("1_main_text/0_pca_register_before_after.py", True),
]

_MODULE_CACHE: dict[str, object] = {}


def _load_module(rel_path: str):
    cached = _MODULE_CACHE.get(rel_path)
    if cached is not None:
        return cached
    path = ANALYSIS_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(Path(rel_path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    # Register under the script's module name so that scripts using
    # multiprocessing.Pool (e.g. a3_sdg4_lexical_audit) can pickle their
    # module-level worker functions: forked children re-import the module by
    # this name, which must resolve.
    sys.modules[Path(rel_path).stem] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[rel_path] = mod
    return mod


def _run_step(rel_path: str, model: str, output_dir: Path, *, overwrite: bool = False, embeddings: str = "raw", classifier: str = "lr") -> None:
    mod = _load_module(rel_path)
    argv = [str(ANALYSIS_ROOT / rel_path), "--embed-model", model, "--output-dir", str(output_dir)]
    if overwrite:
        argv.append("--overwrite")
    if embeddings != "raw":
        argv.extend(["--embeddings", embeddings])
    if classifier != "lr":
        argv.extend(["--classifier", classifier])
    sys.argv = argv
    mod.main()


def run_analysis(
    model: str,
    output_dir: Path,
    *,
    include_appendix: bool = False,
    overwrite: bool = False,
) -> None:
    """Run in-process interaction analysis (+ optional appendix) for `model`.

    Coverage gap, semantic gap, and PCA are run as subprocess steps in the
    linear pipeline BEFORE this function is called.
    """
    for rel_path, only_default in MAIN_STEPS:
        if not only_default or model == DEFAULT_EMBED_MODEL:
            _run_step(rel_path, model, output_dir, overwrite=overwrite)

    if include_appendix:
        for rel_path, _ in APPENDIX_STEPS:
            _run_step(rel_path, model, output_dir, overwrite=overwrite)


def run_post_adjusted(
    model: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run post-adjusted generators (decomposition, correlation, macros, PCA before/after).

    Must run AFTER adjusted semantic-gap JSONs exist under data/adjusted/.
    """
    for item in POST_ADJUSTED_STEPS:
        if isinstance(item, tuple):
            rel_path, only_default = item
            if only_default and model != DEFAULT_EMBED_MODEL:
                continue
        else:
            rel_path = item
        _run_step(rel_path, model, output_dir, overwrite=overwrite)
