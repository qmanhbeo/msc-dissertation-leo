#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

output_pdf="${1:-../outputs/dissertation.pdf}"
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

# Keep writing/ source-only apart from artifact/.
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  mv "$f" "$artifact_dir/$base"
done < <(find . -maxdepth 1 -type f \
  ! -name '*.tex' ! -name '*.bib' ! -name '*.sh' -print0)

printf 'Built %s\n' "$output_pdf"
