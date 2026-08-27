#!/usr/bin/env python3
"""Reproducible per-section word counter for the dissertation main text.

Counts words in ``3_writing/dissertation.tex`` per top-level ``\section`` and
per ``\subsection``, excluding the appendix/robustness sections (everything
from ``\section{Retrieval and Query Chain}`` onward) so the reported total
reflects the 8.8k main-text cap only.

The method matches the ad-hoc session counts: strip LaTeX comments, then
strip all ``\command[opts]{args}`` macros (keeping their brace text only where
it is visible prose), normalise punctuation, and count whitespace-separated
tokens. This is a review-time aid, not a pipeline stage.

IMPORTANT (2026-08-27): this is a REVIEW AID, not the declaration count. It
keeps captions and the table "Notes:" paragraphs and silently drops
macro-injected numbers, so its total (8,521 at HEAD bf9317e) is neither
Word-exact nor regulation-compliant. The declaration-grade canon count comes
from ``word_count_docx.py`` on the built DOCX: main-text prose + headings
including in-text citations, excluding table cells, captions/legends and
table "Notes:" paragraphs (canon 8,559 at HEAD bf9317e; cap 8,000 + 10% =
8,800). Never declare this script's number.

Non-prose macros are removed together with their brace argument so they do not
pollute the count: ``\input``/``\include``/``\includegraphics`` file paths,
``\label``/``\ref`` keys, citation keys (``\cite``/``\citep``/``\parencite``/
``\textcite``), and ``\footnote`` text. ``\caption`` text is kept here for
review only; the canon declaration count EXCLUDES captions/legends (UoB Reg
7.4.2(d)). Table bodies never enter the count because they live in separate
``\input`` files that this script does not read.

Usage:
    python 5_notes/word_count.py [path/to/dissertation.tex]
"""
import re
import sys
from pathlib import Path

DEFAULT_TEX = Path(__file__).resolve().parents[1] / "3_writing" / "dissertation.tex"

# Appendix boundary: prose after this section is excluded from the main cap.
APPENDIX_BOUNDARY = r"\\section\{Concept-based Research Retrieval\}"


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # remove everything from an unescaped % to end of line
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        out.append(stripped)
    return "\n".join(out)


def strip_latex(text: str) -> str:
    # Drop verbatim / include / input style commands entirely.
    text = re.sub(r"\\begin\{(?:table|figure|tabular|verbatim|lstlisting)[^}]*\}", " ", text)
    text = re.sub(r"\\end\{(?:table|figure|tabular|verbatim|lstlisting)[^}]*\}", " ", text)
    # Non-prose commands: drop the command AND its brace argument (file paths,
    # label/ref keys, citation keys, footnote text). \caption is handled
    # separately below so its prose is kept and counted.
    text = re.sub(
        r"\\(?:input|include|includegraphics|label|ref|citep|cite|parencite|textcite|footnote)\b"
        r"\*?(?:\s*\[[^\]]*\])?(?:\s*\{[^}]*\})?",
        " ", text,
    )
    # \caption: remove only the command name; the brace text is counted.
    text = re.sub(r"\\caption\*?\b", " ", text)
    # Remove macro definitions and structure commands, keeping brace contents
    # only when they are ordinary prose (we keep all brace text to be safe).
    text = re.sub(r"\\newcommand\b", " ", text)
    # Strip a macro name followed by optional [..] and/or {..} groups.
    # Iteratively remove \name[..]{..} / \name{..} while preserving inner text.
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", text)
    # Remove any remaining stray braces and backslashes.
    text = text.replace("{", " ").replace("}", " ").replace("\\", " ")
    # Normalise whitespace and punctuation to word boundaries.
    text = re.sub(r"[~—–—-]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def count_words(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEX
    raw = path.read_text(encoding="utf-8")
    raw = strip_comments(raw)

    # Split into top-level sections.
    parts = re.split(r"(\\section\{[^}]*\})", raw)
    # parts: [pre, \section{A}, bodyA, \section{B}, bodyB, ...]
    sections = []
    for i in range(1, len(parts), 2):
        title = re.search(r"\\section\{([^}]*)\}", parts[i]).group(1)
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((title, body))

    print(f"{'SECTION':<28}{'WORDS':>8}")
    print("-" * 36)
    total_main = 0
    in_appendix = False
    for title, body in sections:
        if re.match(APPENDIX_BOUNDARY, rf"\section{{{title}}}"):
            in_appendix = True
        words = count_words(strip_latex(body))
        if in_appendix:
            print(f"{title:<28}{words:>8}  (appendix, excluded)")
        else:
            print(f"{title:<28}{words:>8}")
            total_main += words
    print("-" * 36)
    print(f"{'MAIN TEXT TOTAL':<28}{total_main:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
