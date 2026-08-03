"""
Fetch the encoder models required by --cold-replay / --warm-replay into the
local HuggingFace cache.

Cold replay (and the embed/segment steps it drives) run fully offline via
``--local-files-only``. Nothing in the pipeline downloads the encoder weights,
so a fresh clone would fail at the first segmentation/embedding step unless the
three encoder models are already present in ``~/.cache/huggingface/hub``.

This script is the single, explicit network dependency: it warms the cache for
every encoder track (MPNet + MiniLM + SciBERT) and the canonical segment
tokenizer, plus the NLTK ``punkt`` / ``punkt_tab`` tokenizer data used by
segmentation. It is idempotent — resources already in the cache are skipped.

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

from model_utils import COLD_REPLAY_MODELS  # noqa: E402


def _hf_repo_id(model_name: str) -> str:
    """Resolve a model name to its HuggingFace repo id.

    Bare sentence-transformers names (e.g. ``all-mpnet-base-v2``) live under the
    ``sentence-transformers/`` namespace; repo ids that already contain a
    namespace (e.g. ``allenai/scibert_scivocab_uncased``) are used as-is. This
    lets us probe/download the cache via ``huggingface_hub.snapshot_download``
    without ever constructing a torch model — which would otherwise load the
    weights into memory just to answer "is this cached?".
    """
    if "/" in model_name:
        return model_name
    return "sentence-transformers/" + model_name


def _is_cached(model_name: str) -> bool:
    """Return True if the model's HF repo is already present in the local cache."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(_hf_repo_id(model_name), local_files_only=True)
        return True
    except Exception:
        return False


def _fetch(model_name: str, *, force: bool = False) -> None:
    """Download (and verify) a model repo into the HF cache.

    Uses ``snapshot_download`` so the weights are fetched to disk but never
    loaded into a torch model — the downstream ``SentenceTransformer(...,
    local_files_only=True)`` reads them back offline later.
    """
    from huggingface_hub import snapshot_download

    log.info("Downloading model: %s", model_name)
    snapshot_download(_hf_repo_id(model_name), force_download=force)
    log.info("Cached model: %s", model_name)


def ensure_model_cached(model_name: str, *, force: bool = False) -> None:
    """Warm the HF cache for ``model_name`` if absent (or if ``force``)."""
    if not force and _is_cached(model_name):
        log.info("Skip %s — already in HF cache", model_name)
        return
    _fetch(model_name, force=force)


def _nltk_data_present() -> bool:
    """Return True if NLTK punkt / punkt_tab are already available offline."""
    try:
        from nltk.data import find

        find("tokenizers/punkt_tab")
        find("tokenizers/punkt")
        return True
    except Exception:
        return False


def ensure_nltk_data() -> None:
    """Warm NLTK tokenizer data into ~/nltk_data (idempotent, offline-safe)."""
    if _nltk_data_present():
        log.info("Skip NLTK punkt — already present")
        return
    import nltk

    log.info("Downloading NLTK punkt / punkt_tab tokenizer data")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    log.info("Cached NLTK tokenizer data")


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
    ensure_nltk_data()
    log.info("Done — encoder models and NLTK data ready.")


if __name__ == "__main__":
    main()
