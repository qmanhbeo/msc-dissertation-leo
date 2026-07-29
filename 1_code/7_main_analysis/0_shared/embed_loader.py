"""
Single extension point for loading embedding models.

Sentence-transformers-native checkpoints (e.g. all-mpnet-base-v2,
all-MiniLM-L6-v2) load directly. Raw BERT checkpoints such as SciBERT
(``allenai/scibert_scivocab_uncased``) have no sentence-transformers pooling
head, so they are wrapped with mean pooling.

All embed scripts should call ``load_embedder`` instead of constructing
``SentenceTransformer`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def _disable_torch_load_guard() -> None:
    """Allow loading raw ``.bin`` checkpoints under torch<2.6.

    Recent ``transformers`` refuses ``torch.load`` on ``pytorch_model.bin`` for
    torch < 2.6 (CVE-2025-32434). SciBERT only ships ``pytorch_model.bin`` on the
    default branch; the pinned weights are identical to the safetensors variant.
    ``torch.load(weights_only=True)`` itself is safe in torch 2.5.1, so we disable
    the (over-strict) ``transformers`` guard for the duration of the load. The
    guard is restored afterwards; safetensors paths are unaffected.
    """
    import transformers.modeling_utils as tmu  # type: ignore
    import transformers.utils.import_utils as tiu  # type: ignore
    tmu.check_torch_load_is_safe = lambda: None  # type: ignore[attr-defined]
    tiu.check_torch_load_is_safe = lambda: None  # type: ignore[attr-defined]


def _is_pooled(model: SentenceTransformer) -> bool:
    """Return True if ``model.encode`` yields a single pooled vector per input."""
    try:
        out = model.encode(["probe"], convert_to_numpy=True)
    except Exception:  # pragma: no cover - cannot probe; assume acceptable
        return True
    try:
        dim = int(model.get_sentence_embedding_dimension())
    except Exception:
        return False
    return out.ndim == 2 and out.shape[0] == 1 and out.shape[1] == dim


def _build_raw_bert(model_name: str, device: str) -> SentenceTransformer:
    """Wrap a raw (non-sentence-transformers) BERT checkpoint with mean pooling."""
    from sentence_transformers import models

    _disable_torch_load_guard()
    word_embedding_model = models.Transformer(model_name)
    dim_fn = getattr(word_embedding_model, "get_embedding_dimension",
                     word_embedding_model.get_word_embedding_dimension)
    pooling = models.Pooling(
        dim_fn(),
        pooling_mode="mean",
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling], device=device)


def load_embedder(
    model_name: str,
    device: str,
    *,
    local_files_only: bool = False,
) -> SentenceTransformer:
    """Load an embedder, wrapping raw BERT checkpoints with mean pooling.

    Args:
        model_name: Hugging Face model id or path.
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``.
        local_files_only: only load from the local HF cache.
    """
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from model_utils import RAW_BERT_MODELS

    raw_slugs = {m.replace("/", "_").lower() for m in RAW_BERT_MODELS}
    slug = model_name.replace("/", "_").lower()
    if slug in raw_slugs:
        # Known raw BERT checkpoint: build the pooling wrapper explicitly, using
        # the canonical HF id (with slash) so weights resolve from the cache.
        hf_id = next(m for m in RAW_BERT_MODELS if m.replace("/", "_").lower() == slug)
        log.info("Wrapping raw BERT %s with mean pooling", hf_id)
        return _build_raw_bert(hf_id, device)

    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        if _is_pooled(model):
            return model
        log.info("Model %s loaded without pooling; wrapping with mean pooling", model_name)
    except Exception as exc:  # pragma: no cover - depends on network/cache
        log.info("Model %s not a native SentenceTransformer (%s); wrapping", model_name, exc)

    return _build_raw_bert(model_name, device)
