#!/usr/bin/env python3
"""Post-process dissertation.docx table formatting after the pandoc step.

Pandoc's docx writer cannot express per-row keep-together properties, so this
script rewrites ``word/document.xml`` in place (stdlib only, same zip-surgery
pattern as ``5_notes/word_count_docx.py``):

  1. ``<w:cantSplit/>`` in every row's ``trPr``  -> a row never splits across
     pages (body rows get a fresh ``trPr``; header rows keep their existing
     one, ``cantSplit`` inserted before ``tblHeader`` per CT_TrPr order).
  2. ``<w:keepNext/>`` on every header-row cell paragraph (after ``pStyle``
     per CT_PPr order) -> the header row stays glued to the first body row.
  3. Header promotion: pandoc's LaTeX reader fails to mark header rows when
     they contain ``\shortstack`` or multi-row ``\multicolumn``/``\cmidrule``
     headers, so those tables carry the header text as an untagged first row.
     For every table that has a ``tblCaption`` but no ``tblHeader`` row, the
     first row is promoted to a header (``tblHeader`` repeat + bold via the
     style's firstRow rule + keepNext). Layout tables (pandoc's subfigure
     grids) have no ``tblCaption`` and are left untouched.
  4. Tables centered horizontally: ``tblPr`` ``jc`` ``start`` -> ``center``
     (covers layout tables too; per-cell text alignment is untouched).
  5. Notes paragraphs centered: any paragraph whose joined run text starts
     with ``Notes:`` (the classification used by ``5_notes/word_count_docx.py``)
     gets ``<w:jc w:val="center"/>`` unless it already has one.

Bold headers, single cell line spacing and cell padding are NOT handled here:
they live declaratively in ``custom_thesis_template.docx`` (``Table`` style
``firstRow`` rule + ``Compact`` style), which pandoc copies into the output.
Caption/figure centering is likewise declarative (``TableCaption`` /
``ImageCaption`` / ``CaptionedFigure`` styles). The script verifies that copy
and fails closed if the template lacks them.

Run exactly once per fresh pandoc output (``build_word.sh`` does this);
re-running on an already-processed docx fails closed rather than double-
applying. Does not touch any text content, so the word count is unaffected.

Usage:
    python3 style_tables_docx.py path/to/dissertation.docx
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail(f"usage: {sys.argv[0]} <docx>")
    path = Path(sys.argv[1])
    if not path.is_file():
        fail(f"not a file: {path}")

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    doc = blobs["word/document.xml"].decode("utf-8")
    styles = blobs["word/styles.xml"].decode("utf-8")

    # Fail closed on template state: bold firstRow + single-spacing Compact.
    # Pandoc re-serializes styles.xml (attribute order, "<w:b/>" -> "<w:b />"),
    # so these checks are normalization-tolerant.
    m = re.search(r'<w:style [^>]*w:styleId="Compact".*?</w:style>', styles, re.S)
    if not (m and 'w:line="240"' in m.group(0) and 'w:lineRule="auto"' in m.group(0)):
        fail("styles.xml Compact style lacks single spacing (w:line=\"240\") - "
             "stale custom_thesis_template.docx? Re-apply the template edits.")
    m = re.search(r'<w:tblStylePr w:type="firstRow">.*?</w:tblStylePr>', styles, re.S)
    if not (m and re.search(r"<w:rPr>\s*<w:b\s*/>", m.group(0))):
        fail("styles.xml Table style firstRow rule lacks <w:b/> - stale "
             "custom_thesis_template.docx? Re-apply the template edits.")
    for sid in ("TableCaption", "ImageCaption", "CaptionedFigure"):
        m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, styles, re.S)
        if not (m and '<w:jc w:val="center" />' in m.group(0)):
            fail(f"styles.xml {sid} style lacks center alignment - stale "
                 "custom_thesis_template.docx? Re-apply the template edits.")

    # Idempotency guard: refuse to double-apply.
    if "cantSplit" in doc or "keepNext" in doc:
        fail("document.xml already contains cantSplit/keepNext - refusing to "
             "double-apply; rebuild the docx with pandoc first.")

    n_tables = doc.count("<w:tbl>")
    n_body = doc.count("<w:tr><w:tc>")
    n_header = doc.count("<w:trPr><w:tblHeader")

    # 1. Body rows: fresh trPr with cantSplit (trPr must precede first tc).
    doc = doc.replace("<w:tr><w:tc>", "<w:tr><w:trPr><w:cantSplit /></w:trPr><w:tc>")

    # 2. Header rows: cantSplit before tblHeader (CT_TrPr order).
    doc = doc.replace("<w:trPr><w:tblHeader", "<w:trPr><w:cantSplit /><w:tblHeader")

    # 3+4. Per-table pass. (a) Header promotion for caption'd tables pandoc
    #    left without a header row (shortstack / multi-row multicolumn
    #    headers): tag their first row with tblHeader so it repeats, goes
    #    bold via the firstRow style rule, and receives keepNext below;
    #    tblLook firstRow must also flip to 1 or the bold rule never
    #    applies. (b) Horizontal centering: tblPr jc start -> center.
    n_promoted = 0
    n_centered = 0

    def style_table(table_match: "re.Match[str]") -> str:
        nonlocal n_promoted, n_centered
        tbl = table_match.group(0)
        if "<w:tblHeader" not in tbl and "w:tblCaption" in tbl:
            tbl = tbl.replace(
                "<w:tr><w:trPr><w:cantSplit /></w:trPr>",
                '<w:tr><w:trPr><w:cantSplit /><w:tblHeader w:val="true" /></w:trPr>',
                1,
            )
            if 'w:firstRow="0"' not in tbl:
                fail("promoted table lacks w:firstRow=\"0\" in tblLook - "
                     "unexpected pandoc output shape")
            tbl = tbl.replace('w:firstRow="0"', 'w:firstRow="1"', 1)
            n_promoted += 1
        m = re.search(r"<w:tblPr>.*?</w:tblPr>", tbl, re.S)
        if not m:
            fail("table without tblPr - unexpected pandoc output shape")
        tblpr = m.group(0)
        if '<w:jc w:val="start" />' in tblpr:
            tblpr = tblpr.replace(
                '<w:jc w:val="start" />', '<w:jc w:val="center" />', 1
            )
            n_centered += 1
        elif '<w:jc w:val="center" />' not in tblpr:
            fail("table tblPr lacks start/center jc - unexpected shape")
        return tbl[: m.start()] + tblpr + tbl[m.end() :]

    doc = re.sub(r"<w:tbl>.*?</w:tbl>", style_table, doc, flags=re.S)

    # 4. Header-row cell paragraphs: keepNext right after pStyle (CT_PPr
    #    order). Scoped to header rows (incl. promoted) so body cells are
    #    untouched.
    keep_next_count = 0

    def glue(row_match: "re.Match[str]") -> str:
        nonlocal keep_next_count
        row = row_match.group(0)
        row, n = re.subn(
            r'(<w:pPr><w:pStyle w:val="[^"]+" />)', r"\1<w:keepNext />", row
        )
        keep_next_count += n
        return row

    doc = re.sub(
        r"<w:trPr><w:cantSplit /><w:tblHeader[^/]*/></w:trPr>.*?</w:tr>",
        glue,
        doc,
        flags=re.S,
    )

    # 5. Notes paragraphs: center via the same joined-text prefix rule the
    #    word counter uses ("Notes:"). Only the pPr shapes pandoc actually
    #    emits are accepted (pStyle-only, or already centered); anything
    #    else fails closed so a silent wrong ordering cannot slip in.
    n_notes = 0

    def center_note(p_match: "re.Match[str]") -> str:
        nonlocal n_notes
        p = p_match.group(0)
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        if not text.startswith("Notes:") or "<w:jc " in p:
            return p
        if "<w:pPr>" not in p:
            p = p.replace(
                "<w:p>", '<w:p><w:pPr><w:jc w:val="center" /></w:pPr>', 1
            )
        else:
            m = re.match(
                r"<w:p><w:pPr>(<w:pStyle w:val=\"[^\"]+\" />)?</w:pPr>", p
            )
            if not m:
                fail(f"unrecognized Notes paragraph pPr shape: {p[:120]!r}")
            p = p[: m.end(1)] + '<w:jc w:val="center" />' + p[m.end(1) :]
        n_notes += 1
        return p

    doc = re.sub(r"<w:p>.*?</w:p>", center_note, doc, flags=re.S)

    # Fail-closed postconditions.
    if doc.count("<w:cantSplit />") != n_body + n_header:
        fail(f"cantSplit count {doc.count('<w:cantSplit />')} != rows "
             f"{n_body} body + {n_header} header")
    if doc.count("<w:tbl>") != n_tables or not n_tables:
        fail("table count changed during rewrite")
    if doc.count("<w:tblHeader") != n_header + n_promoted:
        fail(f"tblHeader count {doc.count('<w:tblHeader')} != "
             f"{n_header} pandoc headers + {n_promoted} promoted")
    uncentered = sum(
        1
        for m in re.finditer(r"<w:tblPr>.*?</w:tblPr>", doc, re.S)
        if '<w:jc w:val="center" />' not in m.group(0)
    )
    if uncentered:
        fail(f"{uncentered} tables left without centered tblPr")
    if not n_notes:
        fail("no Notes paragraphs found to center")
    if not keep_next_count:
        fail("no header-row cell paragraphs matched for keepNext")
    if n_header and not n_body:
        fail("header rows found but no body rows - unexpected document shape")

    # Well-formedness before writing anything.
    try:
        ET.fromstring(doc.encode("utf-8"))
    except ET.ParseError as exc:
        fail(f"rewritten document.xml is not well-formed: {exc}")

    blobs["word/document.xml"] = doc.encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:  # preserve original entry order
            z.writestr(n, blobs[n])
    tmp.replace(path)

    print(f"style_tables_docx: {n_tables} tables; cantSplit on "
          f"{n_body + n_header} rows ({n_body} body + {n_header} header); "
          f"{n_promoted} promoted header rows; "
          f"keepNext on {keep_next_count} header-cell paragraphs; "
          f"{n_centered} tables centered ({n_tables - n_centered} already); "
          f"{n_notes} Notes paragraphs centered")


if __name__ == "__main__":
    main()
