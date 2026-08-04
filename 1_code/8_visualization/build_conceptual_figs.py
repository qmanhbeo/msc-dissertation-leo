#!/usr/bin/env python3
"""Regenerate the manuscript's conceptual LaTeX figures (fig1, fig6).

These are hand-authored TikZ diagrams living in this folder
(``fig_conceptual_framework.tex``, ``fig_pipeline_flowchart.tex``), compiled
standalone with ``pdflatex``. They are model-independent, so they are emitted
once into ``4_outputs/conceptual_figs/`` rather than a model-namespaced dir.

Idempotent: a figure is skipped when its PDF already exists and the source
``.tex`` has not been modified since. Pass ``--overwrite`` to force a recompile.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "4_outputs" / "conceptual_figs"

# source base name -> manuscript figure file name (without extension)
FIGURES = {
    "fig_conceptual_framework": "fig1_conceptual_framework",
    "fig_pipeline_flowchart": "fig6_pipeline_flowchart",
}


def _needs_build(src: Path, out_pdf: Path, overwrite: bool) -> bool:
    if overwrite or not out_pdf.exists():
        return True
    return src.stat().st_mtime > out_pdf.stat().st_mtime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Recompile even if the PDF exists and is up to date.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for src_base, jobname in FIGURES.items():
        src = SRC_DIR / f"{src_base}.tex"
        out_pdf = OUT_DIR / f"{jobname}.pdf"
        if not src.exists():
            print(f"[conceptual-figs] SKIP {src_base}: source {src} missing", file=sys.stderr)
            continue
        if not _needs_build(src, out_pdf, args.overwrite):
            print(f"[conceptual-figs] UP-TO-DATE {jobname}.pdf")
            continue
        print(f"[conceptual-figs] Building {jobname}.pdf from {src.name}")
        try:
            subprocess.run(
                [
                    "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "-jobname", jobname,
                    "-output-directory", str(OUT_DIR), str(src),
                ],
                cwd=str(SRC_DIR), check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[conceptual-figs] FAILED to build {jobname}.pdf: {exc}", file=sys.stderr)
            rc = 1
        finally:
            # Clean auxiliary files; the PDF (if produced) is kept.
            for ext in (".aux", ".log", ".out"):
                aux = OUT_DIR / f"{jobname}{ext}"
                if aux.exists():
                    aux.unlink()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
