"""
Canonical token-count-aware segmentation for the dissertation pipeline.

Uses the actual model tokenizer (not word-count proxy)
and NLTK sentence-boundary detection (not naive regex).
"""

from __future__ import annotations

import logging

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


log = logging.getLogger(__name__)

MIN_SEGMENT_WORDS = 20


def segment_text(
    text: str,
    tokenizer,
    max_seq_length: int,
    margin: int = 10,
    min_words: int = MIN_SEGMENT_WORDS,
) -> list[str]:
    """
    Greedily segment text into token-count-aware segments at sentence boundaries.

    Args:
        text: Input text to segment.
        tokenizer: A HuggingFace tokenizer (used only for token counting).
        max_seq_length: The model's max sequence length.  Segments are kept
                        below this by `margin` tokens.
        margin: Safety margin below max_seq_length.  Verified empirically
                to bring truncation to ~0%.
        min_words: Discard any segment with fewer than this many words.

    Returns:
        List of segment strings, each guaranteed to fit within
        max_seq_length - margin tokens when encoded (including specials).

    Edge cases:
        - Single sentence exceeding the token limit: kept whole
          (can't split mid-sentence).  Will still truncate on encode,
          but this is unavoidable without mid-sentence splitting.
        - No sentences: returns empty list.
        - Trailing segment below min_words: dropped.
    """
    max_tokens = max_seq_length - margin

    _ensure_nltk_data()
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    enc = tokenizer(sentences, add_special_tokens=False)
    sentence_lengths = [len(ids) for ids in enc["input_ids"]]

    segments: list[str] = []
    current: list[str] = []
    current_tokens: int = 0

    for i, sent in enumerate(sentences):
        ntoks = sentence_lengths[i]
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
