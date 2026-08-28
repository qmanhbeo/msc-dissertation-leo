#!/usr/bin/env python3
"""Post-process dissertation.docx table formatting after the pandoc step.

Pandoc's docx writer cannot express per-row keep-together properties, so this
script rewrites ``word/document.xml`` in place (stdlib only, same zip-surgery
pattern as ``5_notes/word_count_docx.py``):

  1. ``<w:cantSplit/>`` in every row's ``trPr``  -> a row never splits across
      pages (body rows get a fresh ``trPr``; header rows keep their existing
      one, ``cantSplit`` inserted before ``tblHeader`` per CT_TrPr order).
  2. Whole-table keep: ``<w:keepNext/>`` on every cell paragraph of every row
      except the last -> with ``cantSplit``, Word keeps a whole table on one
      page whenever it fits, and degrades gracefully when it cannot (breaks
      between rows; the repeated ``tblHeader`` row still shows). This is the
      canonical Word recipe; keepNext is best-effort, so oversized tables
      (worse than one page) are unaffected. The LAST row stays unglued so a
      table never drags the following Notes/heading onto its page. Header
      rows are always non-last in content tables, so header-to-first-body-row
      glue is subsumed.
  3. Header promotion: pandoc's LaTeX reader fails to mark header rows when
      they contain ``\shortstack`` or multi-row ``\multicolumn``/``\cmidrule``
      headers, so those tables carry the header text as an untagged first row.
      For every table that has a ``tblCaption`` but no ``tblHeader`` row, the
      first row is promoted to a header (``tblHeader`` repeat + bold via the
      style's firstRow rule + keepNext via item 2). Layout tables (pandoc's
      subfigure grids) have no ``tblCaption`` and are not promoted.
  4. Tables centered horizontally: ``tblPr`` ``jc`` ``start`` -> ``center``
      (covers layout tables too; per-cell text alignment is untouched).
  5. Caption glue: ``<w:keepNext/>`` on every ``TableCaption`` and
      ``ImageCaption`` paragraph -> a caption cannot orphan onto the previous
      page; it always stays with its table/figure below.
  6. Notes paragraphs centered AND single-spaced: any paragraph whose joined
       run text starts with ``Notes:`` (the classification used by
       ``5_notes/word_count_docx.py``) gets ``<w:jc w:val="center"/>`` and an
       explicit ``<w:spacing w:line="240" w:lineRule="auto"/>`` (single)
       unless it already has them. Notes use the ``BodyText`` style, which
       shares ``Normal``'s double line spacing with the intentionally-double
       main prose, so they must be fixed per-paragraph; captions/table text
       get single spacing declaratively (see item 10).
  7. Booktabs borders on content tables (``tblCaption`` present; Harvard
       three-line style — horizontal rules only, no verticals, no lines
       between data rows): 1.5pt (``w:sz="12"`` eighth-point units) top and
       bottom table rules via ``tblPr`` ``tblBorders`` (everything else
       ``none``), and a 1pt (``w:sz="8"``) bottom border on every cell of
       each flattened header-stack row and mid-table group/summary row.
       Pandoc flattens multi-row LaTeX headers into ordinary rows, so the
       stack is re-identified structurally (see EXPECTED_* census below);
       this also moves the header separator to its correct place (under the
       LAST stack row — the template's ``firstRow`` rule alone would draw it
       under the group row of 2/3-tier headers). Layout tables (pandoc's
       subfigure grids, no ``tblCaption``) are untouched.
   8. Cmidrule-debris excision: pandoc 3.1.3 cannot parse booktabs
        ``\cmidrule`` — it leaks the rule arguments as literal text into the
        leading cell of the next flattened header-stack row ("3-5 (lr)6-9",
        bare "2-8"), fusing them with any real header text there
        ("...8-10 It."). After stack-row classification, pure debris cells
        are blanked and fused cells keep only the real remainder
        ("It."/"SDG", matching the PDF render). Census: ``EXPECTED_JUNK_CELLS``
        cells across ``EXPECTED_JUNK_TABLES`` tables.
   9. Font cap: an explicit 2.5pt ``w:sz`` is set on every run of Table 34
        (the 24-column pooled-OLS grid, ``tab:k1-specification-grid``), the
        widest table in the manuscript. Without it the table inherits the
        default body size and renders far too large in Word. Word-build-only
        — the PDF sizes the same ``\input`` via ``\resizebox``. Captioned
        as ``TABLE34_CAPTION``; every other table keeps its inherited size.

10. Single spacing policy (user request): table/figure captions and table
        text and after-table/figure Notes are single line-spaced, and table
        text has no space before/after each paragraph. Done declaratively in
        ``custom_thesis_template.docx`` (which pandoc copies into the output
        and this script verifies): ``Caption`` gains ``w:line="240"``
        ``w:lineRule="auto"`` (covers ``TableCaption``/``ImageCaption`` via
        inheritance), and ``Compact`` (every table-cell paragraph) becomes
        ``w:before="0" w:after="0"``. Notes use the shared ``BodyText`` style
        and are fixed per-paragraph in step 6 because ``BodyText`` must stay
        double-spaced for the main prose.

Bold headers, single cell line spacing and cell padding are NOT handled here:
they live declaratively in ``custom_thesis_template.docx`` (``Table`` style
``firstRow`` rule + ``Compact`` style), which pandoc copies into the output.
Caption/figure centering and single line spacing (item 10) are likewise
declarative (``TableCaption``/``ImageCaption``/``CaptionedFigure`` +
``Caption`` + ``Compact`` styles). The script verifies that copy
and fails closed if the template lacks them.

Run exactly once per fresh pandoc output (``build_word.sh`` does this);
re-running on an already-processed docx fails closed rather than double-
applying. The only text-content change is the excision of pandoc-leaked
``\cmidrule`` debris from header cells (item 8 below); the declaration
(canon) word count is unaffected because table cells are excluded from it.

Usage:
    python3 style_tables_docx.py path/to/dissertation.docx
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# --- Booktabs border constants (step 7) --------------------------------------
# Border sizes in eighth-point units (Word's w:sz for borders).
RULE_HEAVY_SZ = 12  # 1.5pt — top/bottom table rules (booktabs \toprule/\bottomrule)
RULE_LIGHT_SZ = 8   # 1pt   — header-stack separators + mid-table group rules
# A markerless label row (flattened header row with no cmidrule debris and no
# gridSpan) is accepted as a stack row only if no cell holds a digit and every
# non-empty cell is at most LABEL_CELL_MAX chars. Rationale: the two such rows
# in the census (Table 12 "Cov. gap/Dominance/...", Table 34 "Adj. gap/...")
# max out at 20 chars, while the nearest digit-free prose rows (Table 35 "AI
# use declaration" body) run to hundreds — 24 sits far from both.
LABEL_CELL_MAX = 24
# cmidrule debris pandoc leaves inside flattened header cells, e.g. "2-4 (lr)"
# (\cmidrule(lr){2-4}) or bare "2-8" (\cmidrule{2-8}). Matched PER CELL: the
# joined row text of data rows can fuse adjacent values ("4.1" + "-4.1" ->
# "4.1-4.1") and would false-positive.
JUNK_CELL_RE = re.compile(r"\d+-\d+(?: \(lr\))?")

# pandoc 3.1.3's LaTeX reader cannot parse booktabs \cmidrule: it leaks the
# rule arguments as literal text runs into the leading cell of the next
# flattened header-stack row ("3-5 (lr)6-9" from \cmidrule(lr){3-5}
# \cmidrule(lr){6-9}; bare "2-8" from a lone \cmidrule(lr){2-8}), fusing the
# debris with any real header text in that cell ("...8-10 It." = Table 15's
# iteration label, "...7-9 SDG" = Table 30's first-column label; both
# verified against the tex sources and the PDF render). Excision runs AFTER
# stack-row classification above (which keys on JUNK_CELL_RE) and BEFORE
# bordering, restricted to rows 0..stack_end so data rows are never touched.
# Pure debris is blanked; fused cells keep only the real remainder, matching
# the PDF. Fail-closed census of the 2026-08-27 build: 20 debris cells across
# 17 content tables (18 pure + 2 fused).
EXPECTED_JUNK_CELLS = 20
EXPECTED_JUNK_TABLES = 17
JUNK_FULL_RE = re.compile(r"^\s*\d+-\d+(?: \(lr\)\d+-\d+)*\s*$")
JUNK_PREFIX_RE = re.compile(r"^\s*\d+-\d+(?: \(lr\)\d+-\d+)*\s+(\S.*)$")
# A fused remainder must be a short digit-free header label ("It.", "SDG");
# anything longer or digit-bearing is not safely excisable — fail closed
# rather than guess. Same 24-char rationale as LABEL_CELL_MAX above.
REMAINDER_MAX_CHARS = 24

# Fail-closed census of the 2026-08-28 build (36 content tables). Any
# manuscript table change that alters the header structure must update these:
#   - 36 caption'd tables — ALL tables are content tables (since fe3efeb the
#     H1 scatter panels are one generator image, so no caption-less layout
#     grid exists any more; the row pass below still guards on `not nested`);
#   - 69 bordered rows = 62 stack rows (16 flat x1 + 12 two-tier x2
#     + 5 three-tier x3 + 2 label-header tables x2) + 7 mid-table gridSpan
#     rows (Table 5 H1a-H1d panels x3, Table 6 Spearman line x1,
#     Table 31 x3);
#   - 2 markerless label rows (Table 12 row 1, Table 34 row 2).
EXPECTED_CONTENT_TABLES = 36
EXPECTED_BORDERED_ROWS = 69
EXPECTED_LABEL_ROWS = 2

# Font cap for the Pooled OLS table (tab:k1-specification-grid, "Pooled OLS ... across 24
# configurations"): the widest table in the manuscript (23 coefficient columns
# + a row-major stub column, 24 cells/row across ~70 rows). Everything else
# inherits the template's default size; without an explicit size this table
# renders far too large in Word (its runs carry no w:sz at all). The user
# requested exactly 2.5pt. Word stores font sizes in half-point units, so
# 2.5pt = w:sz w:val="5" (w:szCs mirrors complex-script runs, which pandoc
# never emits here, but is correct CT_RPr practice). New runs injected with a
# fresh rPr; runs that already carry an rPr (<w:bCs/><w:b/> bold headers,
# 4 in the census) get it appended before </w:rPr> (valid order after
# b/bCs). Applied below as step 9, Word build only — the PDF uses
# \resizebox{\textwidth}{!} on the same \input and is untouched.
# Matched by unique caption substring (the trailing number shifts whenever a
# table is added earlier in the manuscript, so we key on "Pooled OLS" rather
# than the hard-coded "Table 34:" that broke after the a1 promotion added a
# table before this one).
TABLE34_CAPTION = 'Pooled OLS'
TABLE34_SZ = 5            # w:sz half-point val for the requested 2.5pt

TBL_BORDERS_XML = (
    "<w:tblBorders>"
    f'<w:top w:val="single" w:sz="{RULE_HEAVY_SZ}" w:space="0" w:color="auto" />'
    '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto" />'
    f'<w:bottom w:val="single" w:sz="{RULE_HEAVY_SZ}" w:space="0" w:color="auto" />'
    '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto" />'
    '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto" />'
    '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto" />'
    "</w:tblBorders>"
)
CELL_BOTTOM_XML = (
    "<w:tcBorders>"
    f'<w:bottom w:val="single" w:sz="{RULE_LIGHT_SZ}" w:space="0" w:color="auto" />'
    "</w:tcBorders>"
)


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def excise_junk_cell(tc: str) -> "tuple[str, bool]":
    """Excise pandoc-leaked ``\\cmidrule`` debris from one table cell.

    Returns ``(cell_xml, changed)``. Pure debris (``JUNK_FULL_RE``) -> empty
    paragraph; fused debris (``JUNK_PREFIX_RE``) -> single run holding the
    real remainder. ``tcPr`` and the paragraph ``pPr`` are preserved.
    """
    joined = "".join(re.findall(r"<[wm]:t[^>]*>([^<]*)</[wm]:t>", tc))
    remainder = ""
    if JUNK_FULL_RE.match(joined):
        pass
    else:
        m = JUNK_PREFIX_RE.match(joined)
        if not m:
            return tc, False
        remainder = m.group(1)
        if (
            len(remainder) > REMAINDER_MAX_CHARS
            or any(ch.isdigit() for ch in remainder)
            or any(ch in remainder for ch in "<>&")
        ):
            fail(
                "fused cmidrule debris remainder not safely excisable: "
                f"{joined[:80]!r}"
            )
    m_cell = re.match(
        r"(<w:tc>(?:<w:tcPr\s*/>|<w:tcPr>.*?</w:tcPr>))"
        r"(<w:p>(?:<w:pPr>.*?</w:pPr>)?(?:<w:r>.*</w:r>)+</w:p>|<w:p/>|<w:p />)"
        r"(</w:tc>)$",
        tc,
        re.S,
    )
    if not m_cell:
        fail(f"unexpected cmidrule-debris cell shape: {tc[:120]!r}")
    head, para, tail = m_cell.groups()
    if para in ("<w:p/>", "<w:p />"):
        fail(f"cmidrule-debris cell already has an empty paragraph: {tc[:80]!r}")
    mp = re.match(
        r"<w:p>(<w:pPr>.*?</w:pPr>)?(?:<w:r>.*</w:r>)+</w:p>$", para, re.S
    )
    if not mp:
        fail(f"unexpected cmidrule-debris paragraph shape: {para[:120]!r}")
    ppr = mp.group(1) or ""
    if remainder:
        new_para = (
            f'<w:p>{ppr}<w:r>'
            f'<w:t xml:space="preserve">{remainder}</w:t></w:r></w:p>'
        )
    else:
        new_para = f"<w:p>{ppr}</w:p>"
    return head + new_para + tail, True


def excise_junk_row(row: str) -> "tuple[str, int]":
    """Excise debris from every qualifying cell of one ``<w:tr>`` region.

    Returns ``(row_xml, n_cells_changed)``. Non-debris cells pass through
    untouched (``excise_junk_cell`` only rewrites matching cells).
    """
    n = 0

    def _cell(m: "re.Match[str]") -> str:
        nonlocal n
        new, changed = excise_junk_cell(m.group(0))
        if changed:
            n += 1
        return new

    out = re.sub(r"<w:tc>.*?(?=<w:tc>|</w:tr>)", _cell, row, flags=re.S)
    return out, n


def size_table_runs(tbl: str) -> "tuple[str, int]":
    """Set an explicit ``w:sz`` on every run of one table region to 2.5pt.

    Returns ``(new_table, n_runs_sized)``. Runs with no ``rPr`` get a fresh
    ``<w:rPr>``; runs that already carry one (bold header cells) get the size
    appended before ``</w:rPr>`` (CT_RPr order after ``b``/``bCs``). Fails
    closed if any run already declares a ``w:sz``, so a re-run cannot double-
    size, and if it meets a run shape it does not recognise.
    """
    n = 0
    sz_elem = (f'<w:sz w:val="{TABLE34_SZ}" />'
               f'<w:szCs w:val="{TABLE34_SZ}" />')
    sz_open = f"<w:rPr>{sz_elem}</w:rPr>"

    def _run(m: "re.Match[str]") -> str:
        nonlocal n
        r = m.group(0)
        if "w:sz" in r:
            fail(f"run in Table 34 already carries a w:sz - double-size "
                 f"guard: {r[:80]!r}")
        if "<w:rPr>" in r:
            if "</w:rPr>" not in r:
                fail(f"run rPr not closed in Table 34: {r[:80]!r}")
            n += 1
            # Insert the size elements before the run's OWN </w:rPr> close.
            return r.replace("</w:rPr>", f"{sz_elem}</w:rPr>", 1)
        if r.startswith("<w:r><w:t"):
            n += 1
            return "<w:r>" + sz_open + r[len("<w:r>") :]
        fail(f"unexpected run shape in Table 34: {r[:80]!r}")

    # NOTE: 4th positional arg of re.sub is `count`, not `flags` — must use
    # the keyword or the integer value of re.S (16) is read as a max count.
    out = re.sub(r"<w:r>.*?</w:r>", _run, tbl, flags=re.S)
    return out, n


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

    # Fail closed on template state: bold firstRow, single-spacing + zero
    # before/after on Compact (table text), single line on Caption (covers
    # table/figure captions). Notes get single spacing per-paragraph in step
    # 6, not here (BodyText is shared with the deliberately-double prose).
    # Pandoc re-serializes styles.xml (attribute order, "<w:b/>" -> "<w:b />"),
    # so these checks are normalization-tolerant.
    m = re.search(r'<w:style [^>]*w:styleId="Compact".*?</w:style>', styles, re.S)
    if not (
        m
        and 'w:line="240"' in m.group(0)
        and 'w:lineRule="auto"' in m.group(0)
        and 'w:before="0"' in m.group(0)
        and 'w:after="0"' in m.group(0)
    ):
        fail("styles.xml Compact style lacks single spacing (w:line=\"240\") "
             "and/or zero before/after (w:before=\"0\" w:after=\"0\") - stale "
             "custom_thesis_template.docx? Re-apply the template edits.")
    m = re.search(r'<w:style [^>]*w:styleId="Caption".*?</w:style>', styles, re.S)
    if not (m and 'w:line="240"' in m.group(0) and 'w:lineRule="auto"' in m.group(0)):
        fail("styles.xml Caption style lacks single spacing (w:line=\"240\") "
             "- stale custom_thesis_template.docx? Re-apply the template edits.")
    m = re.search(r'<w:tblStylePr w:type="firstRow">.*?</w:tblStylePr>', styles, re.S)
    if not (m and re.search(r"<w:rPr>\s*<w:b\s*/>", m.group(0))):
        fail("styles.xml Table style firstRow rule lacks <w:b/> - stale "
             "custom_thesis_template.docx? Re-apply the template edits.")
    for sid in ("TableCaption", "ImageCaption", "CaptionedFigure"):
        m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, styles, re.S)
        if not (m and '<w:jc w:val="center" />' in m.group(0)):
            fail(f"styles.xml {sid} style lacks center alignment - stale "
                 "custom_thesis_template.docx? Re-apply the template edits.")

    # Idempotency guard: refuse to double-apply. Fresh pandoc output contains
    # none of these (borders live in the template style, not the document).
    if (
        "cantSplit" in doc
        or "keepNext" in doc
        or "w:tblBorders" in doc
        or "w:tcBorders" in doc
    ):
        fail("document.xml already contains cantSplit/keepNext/tblBorders/"
             "tcBorders - refusing to double-apply; rebuild the docx with "
             "pandoc first.")

    n_tables = doc.count("<w:tbl>")
    n_body = doc.count("<w:tr><w:tc>")
    n_header = doc.count("<w:trPr><w:tblHeader")

    # 1. Body rows: fresh trPr with cantSplit (trPr must precede first tc).
    doc = doc.replace("<w:tr><w:tc>", "<w:tr><w:trPr><w:cantSplit /></w:trPr><w:tc>")

    # 2. Header rows: cantSplit before tblHeader (CT_TrPr order).
    doc = doc.replace("<w:trPr><w:tblHeader", "<w:trPr><w:cantSplit /><w:tblHeader")

    # 3+4+5. Per-table pass. (a) Header promotion for caption'd tables pandoc
    #    left without a header row (shortstack / multi-row multicolumn
    #    headers): tag their first row with tblHeader so it repeats, goes
    #    bold via the firstRow style rule, and receives keepNext via item 2;
    #    tblLook firstRow must also flip to 1 or the bold rule never
    #    applies. (b) Horizontal centering: tblPr jc start -> center.
    #    (c) Whole-table keep (item 2): keepNext on every cell paragraph of
    #    every row except the last; see glue_row / glue_para below.
    n_promoted = 0
    n_centered = 0
    n_glue = 0          # keepNext insertions in non-last table rows
    n_glued_tables = 0  # tables that actually had >=2 rows to glue
    n_multirow_tables = 0  # tables with >1 row (these are the ones we glue)
    n_bordered_tables = 0  # content tables that received booktabs borders
    n_border_rows = 0      # rows given a 1pt bottom separator (stack + mid)
    n_border_cells = 0     # individual cells bordered
    n_label_rows = 0       # stack rows classified via the markerless-label rule
    n_junk_cells = 0       # pandoc-leaked \cmidrule debris cells excised
    n_junk_tables = 0      # content tables that had debris cells excised
    n_tab34_tables = 0     # tables explicit-sized to 2.5pt (expect: 1 = Table 34)
    n_tab34_runs = 0       # runs given the explicit 2.5pt size

    def row_cell_texts(row: str) -> "list[str]":
        return [
            "".join(re.findall(r"<[wm]:t[^>]*>([^<]*)</[wm]:t>", c))
            for c in re.findall(r"<w:tc>.*?(?=<w:tc>|</w:tr>)", row, re.S)
        ]

    def is_label_row(texts: "list[str]") -> bool:
        """Markerless flattened label row: no digit anywhere, all non-empty
        cells short (see LABEL_CELL_MAX)."""
        ne = [t for t in texts if t.strip()]
        return (
            bool(ne)
            and not any(re.search(r"\d", t) for t in texts)
            and all(len(t) <= LABEL_CELL_MAX for t in ne)
        )

    def border_row_cells(row: str) -> str:
        nonlocal n_border_cells

        def _cell(m: "re.Match[str]") -> str:
            cell = m.group(0)
            if "<w:tcBorders>" in cell:
                fail(f"cell already carries tcBorders - unexpected: {cell[:80]!r}")
            if "<w:tcPr />" in cell:
                return cell.replace(
                    "<w:tcPr />", f"<w:tcPr>{CELL_BOTTOM_XML}</w:tcPr>", 1
                )
            if "</w:tcPr>" in cell:
                return cell.replace("</w:tcPr>", f"{CELL_BOTTOM_XML}</w:tcPr>", 1)
            fail(f"cell without tcPr - unexpected shape: {cell[:80]!r}")

        bordered = re.sub(r"<w:tc>.*?(?=<w:tc>|</w:tr>)", _cell, row, flags=re.S)
        n_border_cells += len(re.findall(CELL_BOTTOM_XML, bordered))
        return bordered

    def glue_para(pm: "re.Match[str]") -> str:
        """Insert <w:keepNext/> into one table-cell paragraph (CT_PPr order)."""
        nonlocal n_glue
        p = pm.group(0)
        if p in ("<w:p/>", "<w:p />"):
            n_glue += 1
            return "<w:p><w:pPr><w:keepNext /></w:pPr></w:p>"
        if "<w:keepNext" in p:
            fail(f"cell paragraph already carries keepNext - unexpected: {p[:100]!r}")
        m2 = re.match(r'(<w:p><w:pPr><w:pStyle w:val="[^"]+" />)', p)
        if m2:
            n_glue += 1
            return p[: m2.end(1)] + "<w:keepNext />" + p[m2.end(1) :]
        if p.startswith("<w:p><w:pPr>"):
            n_glue += 1
            return "<w:p><w:pPr><w:keepNext />" + p[len("<w:p><w:pPr>") :]
        if p.startswith("<w:p>"):
            n_glue += 1
            return "<w:p><w:pPr><w:keepNext /></w:pPr>" + p[len("<w:p>") :]
        fail(f"unrecognized cell paragraph shape: {p[:100]!r}")

    def glue_row(row: str) -> str:
        return re.sub(r"<w:p>.*?</w:p>|<w:p/>|<w:p />", glue_para, row, flags=re.S)

    def style_table(tbl: str) -> str:
        nonlocal n_promoted, n_centered, n_glued_tables, n_multirow_tables
        nonlocal n_bordered_tables, n_border_rows, n_label_rows
        nonlocal n_junk_cells, n_junk_tables
        nonlocal n_tab34_tables, n_tab34_runs
        # Recurse into nested tables first, then style this table's own
        # properties. pandoc wraps the Figure 6 image-grid table in a
        # FigureTable layout table, so a region may contain another table;
        # a flat <w:tr> regex would mis-span rows across the boundary.
        body = tbl[len("<w:tbl>"):-len("</w:tbl>")]
        nested = "<w:tbl>" in body
        tbl = "<w:tbl>" + transform_tables(body) + "</w:tbl>"
        if not nested and "<w:tblHeader" not in tbl and "w:tblCaption" in tbl:
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
        # Step 7: booktabs borders (content tables only; layout grids have
        # no tblCaption and must stay untouched).
        if "w:tblCaption" in tbl:
            if TBL_BORDERS_XML not in tblpr:
                tblpr = tblpr.replace(
                    '<w:jc w:val="center" />',
                    '<w:jc w:val="center" />' + TBL_BORDERS_XML,
                    1,
                )
            n_bordered_tables += 1
        tbl = tbl[: m.start()] + tblpr + tbl[m.end() :]
        # Step 7 (continued): 1pt separators under every header-stack row and
        # every mid-table group/summary row. Row classification uses a flat
        # row regex, so it must only run on non-nested regions (a nested
        # wrapper's flat scan would mis-span rows across the inner boundary).
        if not nested and "w:tblCaption" in tbl:
            b_rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S)
            if "tblHeader" not in b_rows[0]:
                fail("content table without header row 0 - unexpected shape")
            stack_end = 0
            for i in range(1, len(b_rows)):
                ts = row_cell_texts(b_rows[i])
                if "gridSpan" in b_rows[i] or any(
                    JUNK_CELL_RE.search(t) for t in ts
                ):
                    stack_end = i
                elif is_label_row(ts):
                    stack_end = i
                    n_label_rows += 1
                else:
                    break
            # Step 7 preamble: excise pandoc-leaked \cmidrule debris from the
            # header-stack rows (0..stack_end). Runs AFTER classification so
            # the JUNK_CELL_RE stack detection above still sees the debris it
            # keys on, and BEFORE bordering (which only touches tcPr).
            n_junk_in_table = 0
            for i in range(stack_end + 1):
                b_rows[i], n_excised = excise_junk_row(b_rows[i])
                n_junk_in_table += n_excised
            if n_junk_in_table:
                n_junk_cells += n_junk_in_table
                n_junk_tables += 1
            border_idx = set(range(stack_end + 1))
            border_idx.update(
                i
                for i in range(stack_end + 1, len(b_rows))
                if "gridSpan" in b_rows[i]
            )
            bordered = {i: border_row_cells(b_rows[i]) for i in border_idx}
            seen_i = {"i": 0}

            def border_mapper(rm: "re.Match[str]") -> str:
                i = seen_i["i"]
                seen_i["i"] += 1
                # Non-bordered rows fall back to their (possibly debris-
                # excised) b_rows entry, not the raw match.
                return bordered.get(i, b_rows[i])

            tbl = re.sub(r"<w:tr>.*?</w:tr>", border_mapper, tbl, flags=re.S)
            n_border_rows += len(border_idx)
        # Whole-table keep: glue every row except the last. Non-nested regions
        # only: their inner tables were glued by the recursion above, and a
        # wrapper's single row is its own last row (the generic pass would
        # skip it) - but a figure caption detaching from its grid is exactly
        # what keep-with-next exists to prevent, so for a nested wrapper we
        # glue the wrapper cell's OWN paragraphs (those outside the grid) to
        # the caption paragraph that follows the table.
        if nested:
            inner_open = tbl.find("<w:tbl>", len("<w:tbl>"))
            depth = 1
            k = inner_open + len("<w:tbl>")
            while depth:
                nxt_open = tbl.find("<w:tbl>", k)
                nxt_close = tbl.find("</w:tbl>", k)
                if nxt_close == -1:
                    fail("unbalanced nested <w:tbl> in figure wrapper")
                if nxt_open != -1 and nxt_open < nxt_close:
                    depth += 1
                    k = nxt_open + len("<w:tbl>")
                else:
                    depth -= 1
                    k = nxt_close + len("</w:tbl>")
            head = tbl[len("<w:tbl>") : inner_open]
            tail = tbl[k : -len("</w:tbl>")]
            tbl = (
                "<w:tbl>"
                + glue_row(head)
                + tbl[inner_open:k]
                + glue_row(tail)
                + "</w:tbl>"
            )
        else:
            rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S)
            if len(rows) > 1:
                n_multirow_tables += 1
                seen = {"i": 0}
                n = len(rows)

                def keep_row(rm: "re.Match[str]") -> str:
                    seen["i"] += 1
                    return glue_row(rm.group(0)) if seen["i"] < n else rm.group(0)

                tbl = re.sub(r"<w:tr>.*?</w:tr>", keep_row, tbl, flags=re.S)
                n_glued_tables += 1
        # Step 9: explicit 2.5pt font size on every run of Table 34 (the one
        # 24-column pooled-OLS grid). Word-build-only; the PDF sizes the same
        # \input via \resizebox. Runs carry no w:sz otherwise, so without this
        # the table inherits the giant default body size.
        if TABLE34_CAPTION in tbl:
            if "/w:tblPr>" not in tbl:
                fail("Table 34 region lacks tblPr - unexpected shape")
            new_tbl, n_runs = size_table_runs(tbl)
            n_tab34_runs += n_runs
            n_tab34_tables += 1
            return new_tbl
        return tbl

    def transform_tables(doc: str) -> str:
        """Style every complete <w:tbl> region, innermost tables first."""
        out = []
        i = 0
        while True:
            j = doc.find("<w:tbl>", i)
            if j == -1:
                out.append(doc[i:])
                return "".join(out)
            depth = 1
            k = j + len("<w:tbl>")
            while depth:
                nxt_open = doc.find("<w:tbl>", k)
                nxt_close = doc.find("</w:tbl>", k)
                if nxt_close == -1:
                    fail("unbalanced <w:tbl> regions in document.xml")
                if nxt_open != -1 and nxt_open < nxt_close:
                    depth += 1
                    k = nxt_open + len("<w:tbl>")
                else:
                    depth -= 1
                    k = nxt_close + len("</w:tbl>")
            out.append(doc[i:j])
            out.append(style_table(doc[j:k]))
            i = k

    doc = transform_tables(doc)

    # 6. Caption paragraphs: keepNext right after pStyle (CT_PPr order) on
    #    every TableCaption / ImageCaption paragraph, so a caption cannot
    #    orphan onto the previous page and always stays with its table/figure.
    n_captions = 0
    caption_count = len(
        re.findall(r'<w:pStyle w:val="(?:TableCaption|ImageCaption)" />', doc)
    )

    def glue_caption(p_match: "re.Match[str]") -> str:
        nonlocal n_captions
        p = p_match.group(0)
        if "<w:keepNext" in p:
            fail(f"caption paragraph already carries keepNext - unexpected: {p[:100]!r}")
        m2 = re.match(
            r'(<w:p><w:pPr><w:pStyle w:val="(?:TableCaption|ImageCaption)" />)', p
        )
        if not m2:
            fail(f"unrecognized caption paragraph shape: {p[:100]!r}")
        n_captions += 1
        return p[: m2.end(1)] + "<w:keepNext />" + p[m2.end(1) :]

    doc = re.sub(
        r'<w:p><w:pPr><w:pStyle w:val="(?:TableCaption|ImageCaption)" />.*?</w:p>',
        glue_caption,
        doc,
        flags=re.S,
    )

    # 6. Notes paragraphs: center AND single-space via the same joined-text
    #    prefix rule the word counter uses ("Notes:"). Notes use the shared
    #    BodyText style (double-spaced for main prose), so the single line
    #    spacing must be set per-paragraph. Only the pPr shapes pandoc
    #    actually emits are accepted; anything else fails closed so a silent
    #    wrong ordering cannot slip in.
    n_notes = 0
    NOTE_SPACING = '<w:spacing w:line="240" w:lineRule="auto" />'
    NOTE_JC = '<w:jc w:val="center" />'

    def center_note(p_match: "re.Match[str]") -> str:
        nonlocal n_notes
        p = p_match.group(0)
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        if not text.startswith("Notes:"):
            return p
        if "<w:pPr>" not in p:
            p = p.replace(
                "<w:p>", f"<w:p><w:pPr>{NOTE_SPACING}{NOTE_JC}</w:pPr>", 1
            )
        else:
            m = re.match(
                r'<w:p><w:pPr>(<w:pStyle w:val="[^"]+" />)?(.*?)</w:pPr>',
                p,
                re.S,
            )
            if not m:
                fail(f"unrecognized Notes paragraph pPr shape: {p[:120]!r}")
            # Rebuild pPr children as pStyle + spacing + jc (CT_PPr order);
            # strip any pre-existing spacing/jc so a re-run cannot duplicate.
            inner = m.group(2)
            inner = re.sub(r"<w:spacing[^>]*/>", "", inner)
            inner = re.sub(r"<w:jc[^>]*/>", "", inner)
            p = (
                p[: m.start(2)]
                + NOTE_SPACING
                + NOTE_JC
                + inner
                + p[m.end(2) :]
            )
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
    if n_glued_tables != n_multirow_tables:
        fail(f"whole-table glue ran on {n_glued_tables} but {n_multirow_tables} "
             f"multi-row tables found - a table parser mismatch")
    if n_glued_tables and not n_glue:
        fail("tables had multiple rows but no cell paragraphs were glued")
    if n_captions != caption_count:
        fail(f"caption keepNext {n_captions} != caption paragraphs "
             f"{caption_count} - a caption parser mismatch")
    if doc.count("<w:keepNext />") != n_glue + n_captions:
        fail(f"keepNext count {doc.count('<w:keepNext />')} != "
             f"{n_glue} cell + {n_captions} caption")
    if n_header and not n_body:
        fail("header rows found but no body rows - unexpected document shape")

    # Step 7 postconditions: booktabs borders landed on exactly the census.
    if n_bordered_tables != EXPECTED_CONTENT_TABLES:
        fail(f"bordered {n_bordered_tables} content tables but expected "
             f"{EXPECTED_CONTENT_TABLES} - table census changed; update the "
             f"EXPECTED_* constants")
    if doc.count("<w:tblBorders>") != n_bordered_tables:
        fail(f"tblBorders count {doc.count('<w:tblBorders>')} != bordered "
             f"tables {n_bordered_tables}")
    bad_heavy = [
        b
        for b in re.findall(r"<w:tblBorders>.*?</w:tblBorders>", doc, re.S)
        if f'w:top w:val="single" w:sz="{RULE_HEAVY_SZ}"' not in b
        or f'w:bottom w:val="single" w:sz="{RULE_HEAVY_SZ}"' not in b
    ]
    if bad_heavy:
        fail(f"{len(bad_heavy)} tblBorders blocks lack 1.5pt top+bottom rules")
    if n_border_rows != EXPECTED_BORDERED_ROWS:
        fail(f"bordered {n_border_rows} header-stack/group rows but expected "
             f"{EXPECTED_BORDERED_ROWS} - header census changed; update the "
             f"EXPECTED_* constants")
    if doc.count(CELL_BOTTOM_XML) != n_border_cells:
        fail(f"tcBorders count {doc.count(CELL_BOTTOM_XML)} != cells bordered "
             f"{n_border_cells}")
    if n_label_rows != EXPECTED_LABEL_ROWS:
        fail(f"{n_label_rows} markerless label rows classified but expected "
             f"{EXPECTED_LABEL_ROWS} - header census changed; update the "
             f"EXPECTED_* constants")

    # Cmidrule-debris excision postconditions.
    if n_junk_cells != EXPECTED_JUNK_CELLS:
        fail(f"excised {n_junk_cells} cmidrule-debris cells but expected "
             f"{EXPECTED_JUNK_CELLS} - table census changed; update the "
             f"EXPECTED_* constants")
    if n_junk_tables != EXPECTED_JUNK_TABLES:
        fail(f"excised cmidrule debris in {n_junk_tables} tables but expected "
             f"{EXPECTED_JUNK_TABLES} - table census changed; update the "
             f"EXPECTED_* constants")
    if "(lr)" in doc:
        fail("cmidrule debris '(lr)' still present in document.xml after "
             "excision")
    # The (lr) scan above cannot see bare debris like "2-8" (a lone
    # \cmidrule(lr){2-8} leaks no "(lr)" token), so re-scan every cell.
    for tc in re.findall(r"<w:tc>.*?(?=<w:tc>|</w:tr>)", doc, re.S):
        joined = "".join(re.findall(r"<[wm]:t[^>]*>([^<]*)</[wm]:t>", tc))
        if JUNK_FULL_RE.match(joined) or JUNK_PREFIX_RE.match(joined):
            fail(f"cmidrule debris still present in a table cell: "
                 f"{joined[:80]!r}")

    # Table 34 explicit-font postconditions.
    if n_tab34_tables != 1:
        fail(f"{n_tab34_tables} tables explicit-sized but expected exactly 1 "
             f"(Table 34) - census changed; update the script")
    if not n_tab34_runs:
        fail("Table 34 explicit sizing ran but sized 0 runs - unexpected")
    # The document has no other explicit w:sz (sizes all come from the
    # template), so every val="5" must be a Table 34 run we just wrote.
    sz5 = f'w:sz w:val="{TABLE34_SZ}"'
    if doc.count(sz5) != n_tab34_runs:
        fail(f"explicit 2.5pt w:sz count {doc.count(sz5)} != sized runs "
             f"{n_tab34_runs} - a non-Table-34 table carries the size; "
             f"unexpected")

    # 11. Hanging indent on bibliography paragraphs (0.5 in = 720 twips).
    HANG_IND = '<w:ind w:firstLine="-720" w:left="720" />'
    doc = re.sub(
        r'(<w:pStyle w:val="Bibliography" />)',
        r'\1' + HANG_IND,
        doc,
    )

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
          f"keepNext on {n_glue} cell paragraphs of {n_glued_tables} multi-row "
          f"tables + {n_captions} caption paragraphs; "
          f"{n_centered} tables centered ({n_tables - n_centered} already); "
          f"{n_notes} Notes paragraphs centered; "
          f"booktabs borders on {n_bordered_tables} tables "
          f"({n_border_rows} separator rows, {n_border_cells} cells, "
          f"{n_label_rows} label rows); "
          f"excised {n_junk_cells} cmidrule-debris cells in {n_junk_tables} "
          f"tables; "
          f"set 2.5pt font on {n_tab34_runs} runs of {n_tab34_tables} "
          f"table")


if __name__ == "__main__":
    main()
