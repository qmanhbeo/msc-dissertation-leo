from __future__ import annotations

from pathlib import Path

DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
ALLOWED_MODELS = {"all-mpnet-base-v2", "all-MiniLM-L6-v2"}
VALID_DIMS = {384, 768}
N_SDG = 17
DEFAULT_OUTPUT_ROOT = Path("4_outputs")

DATA_ROOT = Path("2_data")


def _validate_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown embed model: {model!r}. Allowed: {sorted(ALLOWED_MODELS)}")


def model_slug(model: str) -> str:
    return model.replace("/", "_").lower()


def preprocessed_dir() -> Path:
    return DATA_ROOT / "1_preprocessed"


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
    return preprocessed_dir() / "research_corpus"


def research_segmented_dir_for_model(model: str) -> Path:
    return segmented_dir_for_model(model) / "research"


def policy_preprocessed_dir() -> Path:
    return preprocessed_dir() / "policy_all"


def embed_research_dir_for_model(model: str) -> Path:
    return embed_dir_for_model(model) / "research_shards"


def scored_research_dir_for_model(model: str) -> Path:
    return scored_dir_for_model(model) / "paper_scores_shards"
