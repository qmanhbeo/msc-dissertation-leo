"""
Analysis orchestrator: run every main-text and appendix analysis for a model
IN A SINGLE PROCESS.

Each analysis was previously a separate `subprocess`, re-opening the 27-shard
research data. `run_analysis` now drives each script's `main()` in-process, and
every script reads the research embedding/score shards directly (shard-native,
mmap) — so no consolidated array is built or cached, and a re-embed / re-score
is reflected immediately on the next run.

SCORE steps (zeroshot, cross-sensitivity table) are NOT managed here — they
live in _run_main_analysis_steps / _run_analysis_only in main.py. This
orchestrator handles the pure analytical steps: coverage_gap, semantic_gap,
interaction, and PCA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from model_utils import DEFAULT_EMBED_MODEL

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]  # 7_main_analysis

# (relative script path, only_when_default_model)
# PCA emits fixed main/figures/ + main/tables/ paths that are MPNet-centric
# and must not be overwritten by a second encoder, so it stays default-only.
# The other steps (coverage, semantic, interaction) write under main/{model}/
# and are namespaced per-encoder for the cross-sensitivity table.
MAIN_STEPS = [
    ("1_main_text/0_coverage_gap.py", False),
    ("1_main_text/1_semantic_gap.py", False),
    ("1_main_text/2_coverage_semantic_interaction.py", False),
    ("1_main_text/0_pca_semantic_landscape.py", True),
]
APPENDIX_STEPS = [
    ("2_appendix/a2_policy_source_family_sensitivity.py", False),
    ("2_appendix/a3_sdg4_lexical_audit.py", False),
    ("2_appendix/b2_semantic_gap_text_interpretability.py", False),
    ("2_appendix/c_sample_stability.py", False),
    ("2_appendix/c1_subset_balanced_stability.py", False),
    ("2_appendix/c0_export_corpus_split_sizes.py", False),
    ("2_appendix/d1_export_model_selection_nums.py", False),
    ("2_appendix/f_register_adjustment.py", False),
    ("2_appendix/h1_cross_method_gap_values.py", False),
    ("2_appendix/i1_assignment_method_comparison.py", False),
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
    """Run all analysis scripts for `model` in-process.

    Each script reads the 27 research embedding/score shards directly (shard-
    native, mmap), so no consolidated array is built or cached. Runs main-text
    analyses and, optionally, appendix analyses — each via its `main()` with no
    subprocess boundary.
    """
    main_steps = [s for s in MAIN_STEPS if (not s[1] or model == DEFAULT_EMBED_MODEL)]
    for rel_path, _ in main_steps:
        _run_step(rel_path, model, output_dir, overwrite=overwrite)

    if include_appendix:
        for rel_path, _ in APPENDIX_STEPS:
            _run_step(rel_path, model, output_dir, overwrite=overwrite)


# Steps that support --embeddings adjusted (produce adjusted JSON outputs).
# Each entry: (script_path, classifier). LR adjusted runs first, then MLP.
ADJUSTED_STEPS = [
    ("1_main_text/1_semantic_gap.py", "lr"),
    ("1_main_text/1_semantic_gap.py", "mlp"),
]

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


def run_analysis_adjusted(
    model: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Run adjusted analyses (register-adjusted embeddings) for `model`.

    Produces adjusted semantic-gap JSONs under data/adjusted/, then runs
    post-adjusted generators (decomposition table, interaction extension,
    consolidated macros).
    """
    for rel_path, classifier in ADJUSTED_STEPS:
        _run_step(rel_path, model, output_dir, overwrite=overwrite, embeddings="adjusted", classifier=classifier)
    for item in POST_ADJUSTED_STEPS:
        if isinstance(item, tuple):
            rel_path, only_default = item
            if only_default and model != DEFAULT_EMBED_MODEL:
                continue
        else:
            rel_path = item
        _run_step(rel_path, model, output_dir, overwrite=overwrite)
