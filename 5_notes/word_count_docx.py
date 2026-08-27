#!/usr/bin/env python3
"""Declaration-grade word counter for the built dissertation DOCX.

Canon definition (user ground truth, 2026-08-27): the declared word count is
the main text (``1 Introduction`` through ``6 Conclusion`` inclusive) counting
prose and headings INCLUDING in-text citations, and EXCLUDING table cell
contents, captions/legends (``TableCaption``/``ImageCaption`` styles, figure
and table alike) and the table ``Notes:`` paragraphs. Front matter, the
bibliography and appendices are outside the range. Basis: UoB Regulation
7.4.2(d) excludes tables, figures (including associated legends), contents
pages, abstract, appendices and the bibliography from the count; in-text
citations are NOT excluded (they count). The Notes exclusion is the user's
ruling on top of the regulation. The main text contains no footnotes (the
eight tex-level ``\\footnote`` grep hits are all ``\\footnotesize`` font
switches), so the regulation's footnote exclusion is moot.

Word-exact tokenization: pandoc stores numbers as ADJACENT ``<w:t>`` runs with
no whitespace ("62" "," "173"), so joining runs with an inserted space
overcounts (~+270 in the main range). All ``<w:t>`` text within a paragraph is
joined with NO separator; ``<w:tab/>`` and ``<w:br/>`` become a space. Table
cells are counted per cell-paragraph. ``str.split()`` (Unicode whitespace,
non-breaking space included) is the tokenizer, matching the verified baselines.

Verified baselines at the trim commit following HEAD ``7bbefba`` (2026-08-27,
post reference-list move and LoF/LoT build); any rebuild shifts them, so re-run
after every manuscript edit:

    canon (declaration count) 8,289
    main-range prose+headings (incl. citations) 8,289
    main-range table cells 679; captions 143; table Notes 154
    main-range total 9,265
    whole document 24,927

Layout notes affecting these numbers: since ``e63b860`` the Word build numbers
captions ("Figure 1:"/"Table 1:", +2 words per caption vs the ``bf9317e``
build), and since ``7bbefba`` the bibliography sits before the appendix with a
``Reference list`` Heading1, which this counter treats as the main-range end
boundary. The pre-trim baselines at HEAD ``bf9317e`` were: canon 8,559;
captions 121; main total 9,513; whole document 24,461. The 25,197
whole-document figure quoted in the 2026-08-27 handoff came from an uncommitted
one-off script with different outside-main-range handling; it is not
reproducible and does not affect the canon figure.

Usage:
    python 5_notes/word_count_docx.py [path/to/dissertation.docx]
"""
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

DEFAULT_DOCX = Path(__file__).resolve().parents[1] / "4_outputs" / "dissertation.docx"

# First main-text Heading1 (exact text after the Lua heading-numbering filter).
MAIN_START = "1 Introduction"

CAPTION_STYLES = {"TableCaption", "ImageCaption"}
NOTES_PREFIX = "Notes:"

# User-confirmed programme cap (2026-08-27): 8,000 words + 10% allowance.
CAP_WORDS = 8_000
CAP_ALLOWANCE = 0.10


def paragraph_text(par):
    parts = []
    for node in par.iter():
        if node.tag == W + "t":
            parts.append(node.text or "")
        elif node.tag in (W + "tab", W + "br"):
            parts.append(" ")
    return "".join(parts)


def paragraph_style(el):
    if el.tag != W + "p":
        return None
    ppr = el.find(W + "pPr")
    if ppr is None:
        return None
    st = ppr.find(W + "pStyle")
    return st.get(W + "val") if st is not None else None


def is_heading1(el):
    return paragraph_style(el) == "Heading1"


def token_count(text):
    return len(text.split())


def table_word_count(tbl):
    total = 0
    for cell in tbl.iter(W + "tc"):
        for par in cell.findall(W + "p"):
            total += token_count(paragraph_text(par))
    return total


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCX
    if not path.exists():
        sys.exit(f"not found: {path}")
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    elements = list(root.find(W + "body"))

    if not any(is_heading1(el) and paragraph_text(el).strip() == MAIN_START
               for el in elements):
        sys.exit(f"anchor heading {MAIN_START!r} not found; document structure changed")

    in_main = False
    buckets = {"prose": 0, "tables": 0, "captions": 0, "notes": 0}
    whole = 0
    sections = []

    for el in elements:
        if is_heading1(el):
            text = paragraph_text(el).strip()
            if text == MAIN_START:
                in_main = True
            elif in_main and not text[:1].isdigit():
                in_main = False
            if in_main:
                n = token_count(text)
                buckets["prose"] += n
                sections.append([text, n])
        elif in_main:
            if el.tag == W + "tbl":
                n = table_word_count(el)
                buckets["tables"] += n
            else:
                style = paragraph_style(el) or ""
                text = paragraph_text(el)
                n = token_count(text)
                if style in CAPTION_STYLES:
                    buckets["captions"] += n
                elif text.lstrip().startswith(NOTES_PREFIX):
                    buckets["notes"] += n
                else:
                    buckets["prose"] += n
            if sections:
                sections[-1][1] += n
        if el.tag == W + "tbl":
            whole += table_word_count(el)
        elif el.tag == W + "p":
            whole += token_count(paragraph_text(el))

    cap = int(CAP_WORDS * (1 + CAP_ALLOWANCE))
    main_total = sum(buckets.values())
    print(f"Canon (declaration) word count: {buckets['prose']:,}")
    print(f"  cap {CAP_WORDS:,} + {CAP_ALLOWANCE:.0%} = {cap:,}; "
          f"margin {cap - buckets['prose']:,} words")
    print("Main range (Introduction..Conclusion):")
    print(f"  prose+headings incl. citations: {buckets['prose']:>6,}")
    print(f"  table cells (excluded):         {buckets['tables']:>6,}")
    print(f"  captions/legends (excluded):    {buckets['captions']:>6,}")
    print(f"  table Notes paragraphs (excl.): {buckets['notes']:>6,}")
    print(f"  main-range total:               {main_total:>6,}")
    print(f"Whole document: {whole:,}")
    print("Per section (canon-relevant prose only):")
    for heading, n in sections:
        print(f"  {heading:<55} {n:>6,}")


if __name__ == "__main__":
    main()
