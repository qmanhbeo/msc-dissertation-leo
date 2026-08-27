#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

output_docx="${1:-../4_outputs/dissertation.docx}"

cleanup() {
  rm -f _build_word_tmp.tex build_meta.tex
}
trap cleanup EXIT

human_stamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
file_stamp="$(date '+%Y-%m-%d_%H%M%S_%Z')"

cat > build_meta.tex <<EOF
\renewcommand{\DraftCompiledAt}{$human_stamp}
\renewcommand{\DraftVersionTag}{$file_stamp}
EOF

# pandoc cannot parse \InputIfFileExists; rewrite the conditional includes to
# plain \input for the working copy (original dissertation.tex is untouched).
# Also unwrap \resizebox{..}{..}{..} -> content so pandoc registers table
# \label s (otherwise \ref{tab:..} renders as the raw [tab:..] label).
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("dissertation.tex")
s = p.read_text(encoding="utf-8")
s = re.sub(r'\\InputIfFileExists\{([^}]*)\}\{\}\{\}', r'\\input{\1}', s)
s = s.replace("\\documentclass", "\\providecommand{\\resizebox}[3]{#3}\n\\documentclass", 1)
# pandoc flattens a figure float's trailing Notes paragraph into an extra cell
# of its image layout grid; relocate figure Notes after \end{figure} in this
# working copy only (the PDF keeps them glued inside the float, so they cannot
# drift across page breaks). Table Notes are untouched: the tempered guard
# blocks at \end{table}, so a table Notes block can never reach a \end{figure}.
moves = 0
def _relocate_fig_notes(m):
    global moves
    moves += 1
    return "\\end{figure}\n" + m.group(1)
s = re.sub(
    r"(\\par\\smallskip\\footnotesize\\emph\{Notes:\}"
    r"(?:(?!\\end\{(?:figure|table)\}|\\par(?![a-zA-Z])).)*?\\par)"
    r"(\s*\\end\{figure\})",
    _relocate_fig_notes,
    s,
    flags=re.S,
)
print(f"build_word: relocated {moves} figure-float Notes paragraph(s)")
pathlib.Path("_build_word_tmp.tex").write_text(s, encoding="utf-8")
PY

# word_section_numbers.lua owns heading numbering (levels 1-3 decimal, then
# appendix letters A..J matching the PDF) and rewrites resolved \ref contents
# from the same map, so headings and cross-refs stay mutually consistent.
# lof_lot.lua injects ToC/LoF/LoT after the abstract (pandoc's own --toc
# always lands at the very top of the docx and cannot be reordered) and
# prefixes body captions "Figure N:" / "Table N:" like the PDF.
pandoc _build_word_tmp.tex -o "$output_docx" \
  --citeproc \
  --csl word-helper/harvard-university-of-birmingham.csl \
  --resource-path=.:../4_outputs/mpnet/figures:../4_outputs/appendix/mpnet \
  --standalone \
  --lua-filter=word-helper/word_section_numbers.lua \
  --lua-filter=word-helper/move_bibliography.lua \
  --lua-filter=lof_lot.lua \
  --reference-doc=word-helper/custom_thesis_template.docx

# Table formatting that pandoc/the reference doc cannot express (row
# keep-together via cantSplit, header rows glued to the body via keepNext).
# Bold headers, single cell line spacing and cell padding live declaratively
# in custom_thesis_template.docx styles; the script verifies those survived
# the pandoc copy and fails closed otherwise. The text changes are
# excising pandoc-leaked \cmidrule debris from table header cells (item 8),
# setting an explicit 2.5pt size on the wide Table 34 grid (item 9), and
# single-spacing the Notes paragraphs (item 6). Captions/table text single
# spacing + zero paragraph spacing live in the template and are verified
# (item 10); see style_tables_docx.py docstring.
python3 word-helper/style_tables_docx.py "$output_docx"

printf 'Built %s\n' "$output_docx"
