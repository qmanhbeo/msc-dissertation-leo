"""
Analysis orchestrator: run every main-text and appendix analysis for a model
IN A SINGLE PROCESS, loading the consolidated research array ONCE.

Previously each analysis was a separate `subprocess`, so the 27-shard research
data was re-opened/re-materialised by every script. Now `run_analysis` drives
each script's `main()` in-process. Because `load_consolidated_embeddings`/
`load_consolidated_scores` are memoized (see research_embedding_shards), the
consolidated array is loaded once and shared by all scripts.

It also refreshes the consolidated cache first, so a re-embed / re-score is
never silently stale (this closes the stale-cache gap for the new artifact).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from consolidate_research_artifacts import consolidate_embeddings, consolidate_scores
from model_utils import DEFAULT_EMBED_MODEL

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]  # 7_main_analysis

# (relative script path, only_when_default_model)
# Mirrors the ordering/conditions previously in main.py _run_main_analysis_steps.
MAIN_STEPS = [
    ("1_canonical/0_zeroshot_scoring.py", True),
    ("1_canonical/0_coverage_gap.py", False),
    ("1_canonical/1_semantic_gap.py", False),
    ("1_canonical/2_coverage_semantic_interaction.py", False),
    ("1_canonical/3_generate_cross_sensitivity_table.py", True),
    ("1_canonical/0_pca_semantic_landscape.py", True),
]
APPENDIX_STEPS = [
    ("3_appendix/a2_policy_source_family_sensitivity.py", False),
    ("3_appendix/a3_sdg4_lexical_audit.py", False),
    ("3_appendix/a3b_loo_sdgi_circularity.py", False),
    ("3_appendix/b2_semantic_gap_text_interpretability.py", False),
    ("3_appendix/c_sample_stability.py", False),
    ("3_appendix/f_register_adjustment.py", False),
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


def _run_step(rel_path: str, model: str, output_dir: Path) -> None:
    mod = _load_module(rel_path)
    # Scripts parse argv via parse_args(); main() == run(parse_args()).
    sys.argv = [str(ANALYSIS_ROOT / rel_path), "--embed-model", model, "--output-dir", str(output_dir)]
    mod.main()


def run_analysis(
    model: str,
    output_dir: Path,
    *,
    include_appendix: bool = False,
    overwrite: bool = False,
) -> None:
    """Run all analysis scripts for `model` in-process.

    Refreshes the consolidated research cache first (skip/regenerate per the
    sha256 sidecar), then runs main-text analyses and, optionally, appendix
    analyses — each via its `main()` with no subprocess boundary.
    """
    consolidate_embeddings(model, overwrite=overwrite)
    consolidate_scores(model, overwrite=overwrite)

    main_steps = [s for s in MAIN_STEPS if (not s[1] or model == DEFAULT_EMBED_MODEL)]
    for rel_path, _ in main_steps:
        _run_step(rel_path, model, output_dir)

    if include_appendix:
        for rel_path, _ in APPENDIX_STEPS:
            _run_step(rel_path, model, output_dir)
