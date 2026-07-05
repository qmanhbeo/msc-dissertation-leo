from __future__ import annotations

from pathlib import Path

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
VALID_DIMS = {384, 768}


def embed_dir_for_model(model: str) -> Path:
    if model == "all-mpnet-base-v2":
        return Path("2_data/2b_embedded_mpnet")
    return Path("2_data/2_embedded")


def scored_dir_for_model(model: str) -> Path:
    if model == "all-mpnet-base-v2":
        return Path("2_data/3b_scored_mpnet")
    return Path("2_data/3_scored")
