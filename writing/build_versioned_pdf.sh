#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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

latexmk -g -pdf -interaction=nonstopmode dissertation.tex

mkdir -p builds
cp dissertation.pdf "builds/dissertation_${file_stamp}.pdf"

printf 'Built %s\n' "builds/dissertation_${file_stamp}.pdf"
