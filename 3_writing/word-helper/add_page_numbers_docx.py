#!/usr/bin/env python3
"""Add guide-compliant page numbering to the Word build (stdlib zip-surgery).

The pandoc/style_tables_docx.py chain produces a docx with NO footer, so the
Word file currently has no page numbers at all. The University of Birmingham
thesis guide requires the preliminary pages to be "numbered separately from the
main body of the thesis, or left unnumbered", with the page sequence starting
at the main body. To match the PDF (which already numbers pages bottom-right via
fancyhdr, and is switched to roman-prelim / arabic-body in dissertation.tex),
this script carves the document into three sections:

  1. Title page          -- unnumbered (no footer reference).
  2. Preliminaries       -- lowercase roman (i, ii, iii...), bottom-right.
     (abstract, Technical Summary, ToC, LoF, LoT)
  3. Main body (Ch1..)   -- arabic from 1, bottom-right.

The displayed number format (roman vs decimal) is driven by each section's
``w:pgNumType``; both footers carry an identical right-aligned PAGE field, so the
footer XML is shared and only the section's pgNumType differs.

Run exactly once per fresh pandoc output (build_word.sh does this after
style_tables_docx.py). Re-running on an already-processed docx fails closed
rather than double-applying.
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

FOOTER_ROMAN_PART = "word/footer_roman.xml"
FOOTER_DECIMAL_PART = "word/footer_decimal.xml"
RID_ROMAN = "rIdFtrRoman"
RID_DECIMAL = "rIdFtrDecimal"

FOOTER_CT = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"
)
FOOTER_REL_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"
)

# Identical footer body for both sections: a right-aligned PAGE field. The
# number format is supplied per-section by w:pgNumType, not by the footer.
FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<w:p><w:pPr><w:jc w:val="right" /></w:pPr>'
    '<w:r><w:fldChar w:fldCharType="begin" /></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="end" /></w:r>'
    "</w:p></w:ftr>"
)


def fail(msg: str) -> "object":
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

    if FOOTER_ROMAN_PART in names or FOOTER_DECIMAL_PART in names:
        fail("docx already contains footer parts - refusing to double-apply; "
             "rebuild the docx with pandoc first.")
    rels = blobs["word/_rels/document.xml.rels"].decode("utf-8")
    if RID_ROMAN in rels or RID_DECIMAL in rels:
        fail("docx rels already reference footer rIds - refusing to double-apply")
    ctypes = blobs["[Content_Types].xml"].decode("utf-8")
    if FOOTER_ROMAN_PART in ctypes or FOOTER_DECIMAL_PART in ctypes:
        fail("docx [Content_Types] already lists footers - refusing to double-apply")

    doc = blobs["word/document.xml"].decode("utf-8")

    # Fail-closed: refuse to double-apply if a prior run already added page
    # numbering (check BEFORE we inject anything, so it catches pre-existing
    # footers rather than the strings we are about to add).
    if "<w:footerReference" in doc or "w:pgNumType" in doc:
        fail("document.xml already carries footerReference/pgNumType - "
             "refusing to double-apply; rebuild the docx with pandoc first")

    # --- Capture the body-level pgMar (a shift-invariant string) so the new
    #     roman section inherits the same page geometry. The body-level sectPr
    #     is the LAST <w:sectPr> in the document (immediately before </w:body>);
    #     we recompute its position later, AFTER the roman sectPr is injected,
    #     because the injection shifts all following character offsets. ---
    _b = doc.rfind("<w:sectPr>")
    if _b == -1:
        fail("document.xml has no sectPr - unexpected pandoc output")
    _be = doc.find("</w:sectPr>", _b)
    if _be == -1:
        fail("body-level sectPr has no closing tag - unexpected pandoc output")
    _body_inner = doc[_b + len("<w:sectPr>"):_be]
    m_pgmar = re.search(r"<w:pgMar[^>]*/>", _body_inner)
    if not m_pgmar:
        fail("body-level sectPr lacks pgMar - unexpected pandoc output")
    pgmar = m_pgmar.group(0)

    # --- Section 1 (title page, unnumbered): inject an empty sectPr into the
    #     Title paragraph's pPr. No footerReference => nothing is displayed. ---
    title_open = "<w:pPr><w:pStyle w:val=\"Title\" /></w:pPr>"
    n_title = doc.count(title_open)
    if n_title != 1:
        fail(f"expected exactly 1 Title paragraph pPr but found {n_title} - "
             f"unexpected document shape")
    doc = doc.replace(title_open,
                      "<w:pPr><w:pStyle w:val=\"Title\" /><w:sectPr/></w:pPr>", 1)

    # --- Section 2 (roman preliminaries): inject a sectPr into the page-break
    #     paragraph immediately before the "1 Introduction" Heading1. That
    #     paragraph is the LAST paragraph of section 2. ---
    ch1_pat = (
        r'(<w:p><w:r><w:br w:type="page"/></w:r></w:p>)'
        r'(<w:bookmarkStart[^>]*/>)?'
        r'(<w:p><w:pPr><w:pStyle w:val="Heading1" /><w:keepNext /></w:pPr>'
        r'<w:r><w:t[^>]*>1 Introduction</w:t></w:r></w:p>)'
    )

    def _ch1(m: "re.Match[str]") -> str:
        roman_sectpr = (
            "<w:p><w:pPr>"
            f"<w:sectPr><w:pgNumType w:fmt=\"lowerRoman\" w:start=\"1\"/>"
            f"{pgmar}"
            f'<w:footerReference w:type="default" r:id="{RID_ROMAN}" />'
            "</w:sectPr></w:pPr>"
            '<w:r><w:br w:type="page"/></w:r></w:p>'
        )
        return roman_sectpr + m.group(2) + m.group(3)

    doc, n_ch1 = re.subn(ch1_pat, _ch1, doc, flags=re.S)
    if n_ch1 != 1:
        fail(f"chapter-1 page-break anchor matched {n_ch1} times (expected 1) - "
             f"the Heading1/section structure changed; update the matcher")

    # --- Section 3 (arabic body): the body-level sectPr already exists; add
    #     pgNumType (decimal, start 1) + a footerReference, preserving CT_SectPr
    #     child order. Recompute its position NOW (after the roman sectPr
    #     injection above shifted all following offsets). (The pre-existing
    #     footerReference/pgNumType guard already ran above, before injection.) ---
    b_idx = doc.rfind("<w:sectPr>")
    if b_idx == -1:
        fail("document.xml lost its body-level sectPr during injection")
    b_end = doc.find("</w:sectPr>", b_idx)
    if b_end == -1:
        fail("body-level sectPr has no closing tag after injection")
    new_body_inner = (
        f"<w:pgNumType w:fmt=\"decimal\" w:start=\"1\"/>"
        f"{pgmar}"
        f'<w:footerReference w:type="default" r:id="{RID_DECIMAL}" />'
    )
    doc = doc[: b_idx + len("<w:sectPr>")] + new_body_inner + doc[b_end:]

    # --- Wire the footer parts into the relationships and content types. ---
    rels = rels.replace(
        "</Relationships>",
        f'<Relationship Type="{FOOTER_REL_TYPE}" Id="{RID_ROMAN}" '
        f'Target="footer_roman.xml" />'
        f'<Relationship Type="{FOOTER_REL_TYPE}" Id="{RID_DECIMAL}" '
        f'Target="footer_decimal.xml" /></Relationships>',
        1,
    )
    ctypes = ctypes.replace(
        "</Types>",
        f'<Override PartName="/word/footer_roman.xml" ContentType="{FOOTER_CT}" />'
        f'<Override PartName="/word/footer_decimal.xml" ContentType="{FOOTER_CT}" '
        f"/></Types>",
        1,
    )
    if RID_ROMAN not in rels or RID_DECIMAL not in rels:
        fail("footer relationships were not inserted - rels parse mismatch")
    if FOOTER_ROMAN_PART not in ctypes or FOOTER_DECIMAL_PART not in ctypes:
        fail("footer content-type overrides were not inserted - parse mismatch")

    # --- Well-formedness before writing anything. ---
    try:
        ET.fromstring(doc.encode("utf-8"))
    except ET.ParseError as exc:
        fail(f"rewritten document.xml is not well-formed: {exc}")
    try:
        ET.fromstring(rels.encode("utf-8"))
    except ET.ParseError as exc:
        fail(f"rewritten document.xml.rels is not well-formed: {exc}")
    try:
        ET.fromstring(ctypes.encode("utf-8"))
    except ET.ParseError as exc:
        fail(f"rewritten [Content_Types].xml is not well-formed: {exc}")
    try:
        ET.fromstring(FOOTER_XML.encode("utf-8"))
    except ET.ParseError as exc:
        fail(f"footer XML is not well-formed: {exc}")

    blobs["word/document.xml"] = doc.encode("utf-8")
    blobs["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    blobs["[Content_Types].xml"] = ctypes.encode("utf-8")
    blobs[FOOTER_ROMAN_PART] = FOOTER_XML.encode("utf-8")
    blobs[FOOTER_DECIMAL_PART] = FOOTER_XML.encode("utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:  # preserve original entry order
            z.writestr(n, blobs[n])
        # New parts appended last (order does not matter for OOXML validity).
        z.writestr(FOOTER_ROMAN_PART, blobs[FOOTER_ROMAN_PART])
        z.writestr(FOOTER_DECIMAL_PART, blobs[FOOTER_DECIMAL_PART])
    tmp.replace(path)

    print(f"add_page_numbers_docx: added {FOOTER_ROMAN_PART} (lowerRoman) + "
          f"{FOOTER_DECIMAL_PART} (decimal) footers; 3 sections "
          f"(title unnumbered, preliminaries roman, body arabic from 1), "
          f"bottom-right.")


if __name__ == "__main__":
    main()
