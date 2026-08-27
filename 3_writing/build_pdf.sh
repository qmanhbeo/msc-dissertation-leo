#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

output_pdf="${1:-../4_outputs/dissertation.pdf}"
artifact_dir="${2:-artifact}"

human_stamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
file_stamp="$(date '+%Y-%m-%d_%H%M%S_%Z')"

cleanup() {
  rm -f build_meta.tex
}
trap cleanup EXIT

cat > build_meta.tex <<EOF
\renewcommand{\DraftCompiledAt}{$human_stamp}
\renewcommand{\DraftVersionTag}{$file_stamp}
EOF

mkdir -p "$artifact_dir" "$(dirname "$output_pdf")"

latexmk -g -pdf -interaction=nonstopmode -auxdir="$artifact_dir" -outdir="$artifact_dir" dissertation.tex

cp "$artifact_dir/dissertation.pdf" "$output_pdf"
rm -f "$artifact_dir/dissertation.pdf"

# Keep 3_writing/ source-only: move ONLY generated LaTeX aux files to artifact/.
# (Never move source assets such as .docx/.csl used by the Word build.)
while IFS= read -r -d '' f; do
  mv "$f" "$artifact_dir/$(basename "$f")"
done < <(find . -maxdepth 1 -type f \( \
  -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' \
  -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' \
  -o -name '*.synctex' -o -name '*.bcf' -o -name '*.run.xml' -o -name '*.loa' \
  -o -name '*.lof' -o -name '*.lot' -o -name '*.idx' -o -name '*.ilg' -o -name '*.ind' \
  -o -name '*.nav' -o -name '*.snm' -o -name '*.vrb' -o -name '*.4tc' -o -name '*.4ct' \
  -o -name '*.xref' -o -name '*.lg' \) -print0)

printf 'Built %s\n' "$output_pdf"
