"""
Convert manually downloaded policy PDFs into text files.

Input:  2_data/0_raw/policy_manual/pdf/*.pdf
Output: 2_data/0_raw/policy_manual/texts/<pdf_stem>.txt

This mirrors the PDF-to-text extraction step used by fetch_policy.py, but
operates only on the manually downloaded PDFs in policy_manual/.

Run from project root:
    python 1_code/0_fetch/convert_policy_manual.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import raw_dir

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

PAGE_BREAK = "\n[PAGE BREAK]\n"


def extract_text(pdf_path: Path) -> str | None:
    """Extract text from a PDF using pdfplumber."""
    if not HAS_PDFPLUMBER:
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return PAGE_BREAK.join(pages) if pages else None
    except Exception as exc:
        print(f"  ✗ Text extraction error: {exc}")
        return None


def convert_pdf(pdf_path: Path, txt_path: Path, overwrite: bool) -> dict:
    """Convert one PDF to text and return a status record."""
    record = {
        "pdf": str(pdf_path),
        "txt": str(txt_path),
        "status": "unknown",
        "pages_with_text": 0,
        "chars": 0,
        "error": None,
    }

    if txt_path.exists() and not overwrite:
        record["status"] = "skipped_existing"
        return record

    text = extract_text(pdf_path)
    if not text:
        record["status"] = "empty_or_failed"
        record["error"] = "no extractable text"
        return record

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    record["status"] = "converted"
    record["pages_with_text"] = text.count(PAGE_BREAK) + 1
    record["chars"] = len(text)
    return record


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert manually downloaded policy PDFs into .txt files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=raw_dir() / "policy_manual" / "pdf",
        help="Directory containing manually downloaded PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=raw_dir() / "policy_manual" / "texts",
        help="Directory to write extracted .txt files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=raw_dir() / "policy_manual" / "artifact" / "convert_policy_manual_summary.json",
        help="Where to write a JSON conversion summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files instead of skipping them.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not HAS_PDFPLUMBER:
        raise SystemExit("pdfplumber is required. Install it with: pip install pdfplumber")

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    report_path: Path = args.report_path

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting policy manual PDFs")
    print(f"  input : {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  files : {len(pdfs)} PDFs")
    print(f"  mode  : {'overwrite' if args.overwrite else 'skip-existing'}")

    results: list[dict] = []
    start = datetime.now()

    for pdf_path in pdfs:
        txt_path = output_dir / f"{pdf_path.stem}.txt"
        print(f"\n[{pdf_path.name}]")
        record = convert_pdf(pdf_path, txt_path, overwrite=args.overwrite)
        results.append(record)

        if record["status"] == "converted":
            print(f"  ✓ converted -> {txt_path}")
        elif record["status"] == "skipped_existing":
            print("  (already exists - skipping)")
        else:
            print(f"  ✗ failed: {record['error']}")

    elapsed = (datetime.now() - start).total_seconds()
    counts = Counter(r["status"] for r in results)
    summary = {
        "started_at": start.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "pdf_count": len(pdfs),
        "converted": counts.get("converted", 0),
        "skipped_existing": counts.get("skipped_existing", 0),
        "failed": counts.get("empty_or_failed", 0),
        "results": results,
    }

    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary")
    print(f"  converted       : {summary['converted']}")
    print(f"  skipped existing : {summary['skipped_existing']}")
    print(f"  failed          : {summary['failed']}")
    print(f"  report          : {report_path}")


if __name__ == "__main__":
    main()
