"""
Fetch the encoder models required by --cold-replay / --warm-replay into the
local HuggingFace cache.

Cold replay (and the embed/segment steps it drives) run fully offline via
``--local-files-only``. Nothing in the pipeline downloads the encoder weights,
so a fresh clone would fail at the first segmentation/embedding step unless the
three encoder models are already present in ``~/.cache/huggingface/hub``.

This script is the single, explicit network dependency: it warms the cache for
every encoder track (MPNet + MiniLM + SciBERT) and the canonical segment
tokenizer. It is idempotent — models already in the cache are skipped.

Usage (run from the project root):
    python 1_code/0_fetch/fetch_encoder_models.py
    python 1_code/0_fetch/fetch_encoder_models.py --force   # re-download
    python 1_code/0_fetch/fetch_encoder_models.py --models all-mpnet-base-v2 allenai/scibert_scivocab_uncased

After this runs once, ``python main.py --cold-replay`` proceeds fully offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import COLD_REPLAY_MODELS, RAW_BERT_MODELS  # noqa: E402


def _is_cached_sentence_transformer(model_name: str) -> bool:
    """Return True if the sentence-transformers checkpoint is already cached."""
    from sentence_transformers import SentenceTransformer

    try:
        SentenceTransformer(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _is_cached_raw_bert(repo_id: str) -> bool:
    """Return True if a raw (non-sentence-transformers) BERT repo is cached."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def _fetch_sentence_transformer(model_name: str) -> None:
    """Download (and verify) a sentence-transformers checkpoint into the cache."""
    from sentence_transformers import SentenceTransformer

    log.info("Downloading sentence-transformers model: %s", model_name)
    SentenceTransformer(model_name)
    log.info("Cached sentence-transformers model: %s", model_name)


def _fetch_raw_bert(repo_id: str) -> None:
    """Download a raw BERT repo (weights + tokenizer) into the cache."""
    from huggingface_hub import snapshot_download

    log.info("Downloading raw BERT model: %s", repo_id)
    snapshot_download(repo_id)
    log.info("Cached raw BERT model: %s", repo_id)


def ensure_model_cached(model_name: str, *, force: bool = False) -> None:
    """Warm the HF cache for ``model_name`` if absent (or if ``force``)."""
    is_raw = model_name in RAW_BERT_MODELS
    if not force:
        cached = (
            _is_cached_raw_bert(model_name)
            if is_raw
            else _is_cached_sentence_transformer(model_name)
        )
        if cached:
            log.info("Skip %s — already in HF cache", model_name)
            return
    if is_raw:
        _fetch_raw_bert(model_name)
    else:
        _fetch_sentence_transformer(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(COLD_REPLAY_MODELS),
        help="Encoder model ids to fetch (default: all cold-replay tracks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a model is already cached",
    )
    args = parser.parse_args()

    log.info("Ensuring encoder models are present in the HF cache")
    for model in args.models:
        ensure_model_cached(model, force=args.force)
    log.info("Done — encoder models ready.")


if __name__ == "__main__":
    main()
