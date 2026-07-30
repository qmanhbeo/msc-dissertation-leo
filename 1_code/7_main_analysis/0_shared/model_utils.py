from __future__ import annotations

import gzip
import json
from pathlib import Path

DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
ALLOWED_MODELS = {"all-mpnet-base-v2", "all-MiniLM-L6-v2", "allenai/scibert_scivocab_uncased"}
VALID_DIMS = {384, 768}
# Raw (non-sentence-transformers) BERT checkpoints that lack a pooling head and
# must be wrapped with mean pooling by embed_loader.load_embedder(). Used by the
# domain-encoder sensitivity analysis (same-dimension scientific encoder).
RAW_BERT_MODELS = {"allenai/scibert_scivocab_uncased"}

# Short aliases for --embed-model so callers can write scibert/minilm/mpnet.
MODEL_ALIASES = {
    "mpnet": "all-mpnet-base-v2",
    "minilm": "all-MiniLM-L6-v2",
    "scibert": "allenai/scibert_scivocab_uncased",
}
N_SDG = 17
RANDOM_SEED = 42
# Numerical-stability / "degenerate centroid" thresholds used across scoring &
# centroid code. Centralised so the assumption is explicit and auditable.
ZERO_NORM_EPS = 1e-8
NORM_EPS = 1e-12
# Below this L2 norm a (policy/research) centroid is treated as degenerate and
# excluded from semantic-gap computation.
MIN_CENTROID_NORM = 0.5
# Canonical segmentation shared by every encoder in the architecture-sensitivity
# comparison. All models embed the SAME canonical segments, so the only varying
# factor is the encoder (architecture + native context window). 384 is
# all-mpnet-base-v2's native limit: it covers SciBERT (<=512) fully and MiniLM
# truncates internally to 256 — a documented model property, not a hidden text
# difference.
CANONICAL_SEGMENT_MODEL = "all-mpnet-base-v2"
CANONICAL_MAX_SEQ_LENGTH = 384
# Shared, deterministic 50k-paper representative research subset drawn (seed 42)
# from the canonical segments. Consumed by every non-primary (sensitivity)
# encoder so the architecture comparison is on identical papers.
RESEARCH_SUBSET_SEED = 42
RESEARCH_SUBSET_SIZE = 50_000
DEFAULT_OUTPUT_ROOT = Path("4_outputs")

DATA_ROOT = Path("2_data")

SDG_NAMES: dict[int, str] = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-Being",
    4: "Quality Education", 5: "Gender Equality",
    6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities", 11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production", 13: "Climate Action",
    14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}

SDG_NUM_WORDS: dict[int, str] = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
    11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
    15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
}


def _validate_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown embed model: {model!r}. Allowed: {sorted(ALLOWED_MODELS)}")


def model_slug(model: str) -> str:
    return model.replace("/", "_").lower()


def output_main_dir_for_model(model: str | None, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Canonical ``4_outputs/main/{slug}/`` directory for a model.

    All 4_outputs model-scoped paths must be derived through this helper so
    that the on-disk layout is consistent with the 2_data slug (e.g.
    ``allenai_scibert_scivocab_uncased``, not the nested slash form
    ``allenai/scibert_scivocab_uncased``). ``model_slug`` is the identity for
    models without a ``/`` (all-mpnet-base-v2, all-MiniLM-L6-v2), so this is
    backward-compatible for them.
    """
    if model is None:
        return root / "main"
    return root / "main" / model_slug(model)


def resolve_model_alias(name: str) -> str:
    """Map a short alias (mpnet/minilm/scibert) to its canonical model id.

    Idempotent for already-canonical names, so it is safe to apply
    unconditionally at argument-parse time in every script that accepts
    --embed-model.
    """
    if not name:
        return name
    return MODEL_ALIASES.get(name.strip().lower(), name)


def preprocessed_dir() -> Path:
    return DATA_ROOT / "1_preprocessed"


def individual_sources_dir() -> Path:
    return preprocessed_dir() / "individual_sources"


def individual_source_dir(source: str) -> Path:
    return individual_sources_dir() / source


def segmented_dir_for_model(model: str) -> Path:
    _validate_model(model)
    return DATA_ROOT / "2_segmented" / model_slug(model)


def embed_dir_for_model(model: str) -> Path:
    _validate_model(model)
    return DATA_ROOT / "3_embedded" / model_slug(model)


def model_results_dir_for_model(model: str) -> Path:
    _validate_model(model)
    return DATA_ROOT / "4_supervised_model_results" / model_slug(model)


def scored_dir_for_model(model: str) -> Path:
    _validate_model(model)
    return DATA_ROOT / "5_supervised_scored" / model_slug(model)


def centroids_dir() -> Path:
    return DATA_ROOT / "6_centroids"


def raw_dir() -> Path:
    return DATA_ROOT / "0_raw"


def research_preprocessed_dir() -> Path:
    return preprocessed_dir() / "research"


def research_segmented_dir_for_model(model: str) -> Path:
    return segmented_dir_for_model(model) / "research"


def canonical_research_segment_dir() -> Path:
    return research_segmented_dir_for_model(CANONICAL_SEGMENT_MODEL)


def research_subset_dir() -> Path:
    """Shared 50k-paper subset of the canonical research segments.

    Sensitivity encoders (MiniLM, SciBERT) embed these identical texts rather
    than the full corpus, so the architecture comparison is on identical papers.
    """
    return canonical_research_segment_dir() / "research_50k_subset"


def research_subset_manifest() -> Path:
    return research_subset_dir() / "metadata" / "manifest.json"


def research_concept_preprocessed_dir() -> Path:
    return preprocessed_dir() / "research_concept"


def research_concept_segmented_dir_for_model(model: str) -> Path:
    return segmented_dir_for_model(model) / "research_concept"


# ── Warm-replay text fallback (3a_warm_replay_texts/) ────────────────────
#
# Canonical text lives in 2_segmented/{model}/ (plain .jsonl). The embedded
# snapshot instead ships a gzipped copy of exactly the files the warm-replay
# appendix consumers (a3, b2) need, under 3a_warm_replay_texts/{model}/,
# built only by 1_code/data_backup_and_fetch/build_warm_replay_texts.py.
# Resolution order: canonical plain file first, then the .gz fallback,
# else fail closed.

WARM_REPLAY_TEXTS_DIRNAME = "3a_warm_replay_texts"


def warm_replay_texts_dir_for_model(model: str) -> Path:
    _validate_model(model)
    return DATA_ROOT / WARM_REPLAY_TEXTS_DIRNAME / model_slug(model)


def resolve_research_text_path(model: str, shard_name: str) -> Path:
    if model != CANONICAL_SEGMENT_MODEL:
        # Sensitivity encoders embed the shared canonical 50k subset; their
        # texts are the canonical segments for those papers, not their own
        # (now-absent) segmented dir.
        subset = research_subset_dir() / f"{shard_name}.jsonl"
        if subset.exists():
            return subset
        subset_gz = research_subset_dir() / f"{shard_name}.jsonl.gz"
        if subset_gz.exists():
            return subset_gz
    canonical = research_segmented_dir_for_model(model) / f"{shard_name}.jsonl"
    if canonical.exists():
        return canonical
    fallback = warm_replay_texts_dir_for_model(model) / "research" / f"{shard_name}.jsonl.gz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Research segment text for model {model!r} shard {shard_name!r} not found: "
        f"neither {canonical} nor {fallback} exists. "
        "Hydrate 2_segmented/ or fetch the embedded snapshot (3a_warm_replay_texts/)."
    )


def resolve_policy_text_path(model: str) -> Path:
    if model != CANONICAL_SEGMENT_MODEL:
        # Sensitivity encoders embed the canonical policy segments, so their
        # policy text is the canonical model's, not their own (absent) dir.
        canonical_policy = segmented_dir_for_model(CANONICAL_SEGMENT_MODEL) / "policy.jsonl"
        if canonical_policy.exists():
            return canonical_policy
        canonical_policy_gz = segmented_dir_for_model(CANONICAL_SEGMENT_MODEL) / "policy.jsonl.gz"
        if canonical_policy_gz.exists():
            return canonical_policy_gz
    canonical = segmented_dir_for_model(model) / "policy.jsonl"
    if canonical.exists():
        return canonical
    fallback = warm_replay_texts_dir_for_model(model) / "policy.jsonl.gz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Policy segment text for model {model!r} not found: "
        f"neither {canonical} nor {fallback} exists. "
        "Hydrate 2_segmented/ or fetch the embedded snapshot (3a_warm_replay_texts/)."
    )


def open_text(path: str | Path):
    """Open a text file for reading, transparently decompressing .gz."""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return p.open(encoding="utf-8")


def embed_research_dir_for_model(model: str) -> Path:
    return embed_dir_for_model(model) / "research_shards"


def scored_research_dir_for_model(model: str) -> Path:
    return scored_dir_for_model(model) / "paper_scores_shards"


# ── Grid-search log helpers ──────────────────────────────────────────────

def _cfg_key(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def _entry_key(e: dict) -> tuple:
    return (e.get("model"), _cfg_key(e["config"]))


def append_grid_log(
    grid_log_path: Path,
    model_tag: str,
    config: dict,
    cv_metrics: dict,
    n_train: int,
    input_dim: int,
) -> None:
    """Append one config entry to the durable grid-search log (dedup-aware).

    If a matching (model_tag, config) entry already exists with identical
    metrics it is silently skipped.  If metrics differ a warning is logged
    and the new entry is appended anyway.
    """
    import datetime
    import logging

    log = logging.getLogger(__name__)

    if grid_log_path.exists():
        with grid_log_path.open() as f:
            grid_log = json.load(f)
    else:
        grid_log = {"log": []}

    key = (model_tag, _cfg_key(config))
    existing = [e for e in grid_log["log"] if _entry_key(e) == key]

    if existing:
        for entry in existing:
            em = entry["cv_metrics"]
            if em.get("mean_f1") == cv_metrics.get("mean_f1") and em.get("std_f1") == cv_metrics.get("std_f1"):
                log.info("Config already logged with identical metrics — skipping: %s", config)
                return
        log.warning("Config %s already logged with different metrics — appending", config)

    entry = {
        "model": model_tag,
        "config": config,
        "cv_metrics": cv_metrics,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train": n_train,
        "input_dim": input_dim,
    }
    grid_log["log"].append(entry)

    tmp = grid_log_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(grid_log, f, indent=2, default=str)
    tmp.replace(grid_log_path)
