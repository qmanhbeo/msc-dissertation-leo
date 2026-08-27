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
pathlib.Path("_build_word_tmp.tex").write_text(s, encoding="utf-8")
PY

pandoc _build_word_tmp.tex -o "$output_docx" \
  --citeproc \
  --csl harvard-university-of-birmingham.csl \
  --resource-path=.:../4_outputs/mpnet/figures:../4_outputs/appendix/mpnet \
  --standalone --toc \
  --reference-doc=custom_thesis_template.docx

printf 'Built %s\n' "$output_docx"
