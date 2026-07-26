"""
Canonical token-count-aware segmentation for the dissertation pipeline.

Uses the actual model tokenizer (not word-count proxy)
and NLTK sentence-boundary detection (not naive regex).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nltk
from nltk.tokenize import sent_tokenize

_nltk_ready = False


def _ensure_nltk_data() -> None:
    """Download NLTK tokenizer data on first use, not at import time."""
    global _nltk_ready
    if not _nltk_ready:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        _nltk_ready = True


if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MIN_SEGMENT_WORDS = 20


def count_content_tokens(text: str, tokenizer) -> int:
    """Count tokens excluding special tokens (CLS, SEP)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def segment_text(
    text: str,
    model: SentenceTransformer,
    margin: int = 10,
    min_words: int = MIN_SEGMENT_WORDS,
) -> list[str]:
    """
    Greedily segment text into token-count-aware segments at sentence boundaries.

    Args:
        text: Input text to segment.
        model: SentenceTransformer instance (used for tokenizer + max_seq_length).
        margin: Safety margin below max_seq_length.  Verified empirically
                to bring truncation to ~0%.
        min_words: Discard any segment with fewer than this many words.

    Returns:
        List of segment strings, each guaranteed to fit within
        model.max_seq_length - margin tokens when encoded (including specials).

    Edge cases:
        - Single sentence exceeding the token limit: kept whole
          (can't split mid-sentence).  Will still truncate on encode,
          but this is unavoidable without mid-sentence splitting.
        - No sentences: returns empty list.
        - Trailing segment below min_words: dropped.
    """
    max_tokens = model.max_seq_length - margin
    tokenizer = model.tokenizer

    _ensure_nltk_data()
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_tokens: int = 0

    for sent in sentences:
        ntoks = count_content_tokens(sent, tokenizer)
        projected = current_tokens + ntoks + 2

        if current and projected > max_tokens:
            combined = " ".join(current)
            if len(combined.split()) >= min_words:
                segments.append(combined)
            current = [sent]
            current_tokens = ntoks
        elif ntoks + 2 > max_tokens:
            if current:
                combined = " ".join(current)
                if len(combined.split()) >= min_words:
                    segments.append(combined)
            if len(sent.split()) >= min_words:
                segments.append(sent)
            current = []
            current_tokens = 0
        else:
            current.append(sent)
            current_tokens += ntoks

    if current:
        combined = " ".join(current)
        if len(combined.split()) >= min_words:
            segments.append(combined)

    return segments


def verify_truncation_rate(
    texts: list[str],
    model: SentenceTransformer,
    label: str = "",
) -> float:
    """Measure what fraction of texts truncate when encoded."""
    max_len = model.max_seq_length
    tokenizer = model.tokenizer
    truncated = 0
    for t in texts:
        n = len(tokenizer.encode(t))
        if n > max_len:
            truncated += 1
    rate = truncated / len(texts) if texts else 0.0
    log.info(
        "Truncation rate [%s]: %.1f%% (%d/%d) at max_seq_length=%d",
        label,
        100 * rate,
        truncated,
        len(texts),
        max_len,
    )
    return rate


def scan_segmented_truncation(
    segments: list[str],
    model: SentenceTransformer,
    label: str = "",
) -> float:
    """After segmentation, confirm near-zero truncation."""
    return verify_truncation_rate(segments, model, label=f"{label} (segmented)")
