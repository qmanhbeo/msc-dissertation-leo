#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

human_stamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
file_stamp="$(date '+%Y-%m-%d_%H%M%S_%Z')"
artifact_dir="artifact"

cleanup() {
  rm -f build_meta.tex
}
trap cleanup EXIT

cat > build_meta.tex <<EOF
\renewcommand{\DraftCompiledAt}{$human_stamp}
\renewcommand{\DraftVersionTag}{$file_stamp}
EOF

mkdir -p "$artifact_dir" builds

latexmk -g -pdf -interaction=nonstopmode -auxdir="$artifact_dir" -outdir="$artifact_dir" dissertation.tex

cp "$artifact_dir/dissertation.pdf" dissertation.pdf

# Keep writing/ root minimal: move any newly generated loose artifacts back under artifact/.
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  mv "$f" "$artifact_dir/$base"
done < <(find . -maxdepth 1 -type f \
  ! -name '*.tex' ! -name '*.bib' ! -name '*.pdf' ! -name '*.sh' -print0)

cp dissertation.pdf "builds/dissertation_${file_stamp}.pdf"

printf 'Built %s\n' "builds/dissertation_${file_stamp}.pdf"
