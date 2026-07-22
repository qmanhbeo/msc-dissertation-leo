from __future__ import annotations

from pathlib import Path

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
ALLOWED_MODELS = {DEFAULT_EMBED_MODEL, "all-mpnet-base-v2"}
VALID_DIMS = {384, 768}
N_SDG = 17
DEFAULT_OUTPUT_ROOT = Path("4_outputs_legacy")


def _validate_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown embed model: {model!r}. Allowed: {sorted(ALLOWED_MODELS)}")


def embed_dir_for_model(model: str) -> Path:
    _validate_model(model)
    if model == "all-mpnet-base-v2":
        return Path("2_data/2b_embedded_mpnet")
    return Path("2_data/2_embedded")


def scored_dir_for_model(model: str) -> Path:
    _validate_model(model)
    if model == "all-mpnet-base-v2":
        return Path("2_data/3b_scored_mpnet")
    return Path("2_data/3_scored")
