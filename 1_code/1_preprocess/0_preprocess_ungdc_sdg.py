"""
Extract SDG-relevant policy passages from the UN General Debate Corpus (UNGDC).

Input:  2_data/0_raw/ungdc/TXT/Session <N> - <YEAR>/<ISO>_<session>_<year>.txt
        Sessions 70–80 (2015–2024) — post-SDG-adoption speeches only

Output: 2_data/1_preprocessed/individual_sources/ungdc_sdg/ungdc_sdg_clean.jsonl

Strategy:
  1. Read all speeches from sessions 70–80 (2015–2024)
  2. Split each speech into paragraphs
  3. Keep paragraphs that contain SDG-relevant keywords
  4. Concatenate the SDG-relevant paragraphs into a single document per speech
  5. Output documents (not segments) for downstream segmentation

Preserves paragraph-level SDG keyword filtering before segmentation.
Only the merge/segmentation step is now handled by segment_corpus.py.

Run from project root:
    python 1_code/1_preprocess/0_preprocess_ungdc_sdg.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import raw_dir, preprocessed_dir, individual_source_dir
from _resume import resumable_records

UNGDC_TXT_DIR = raw_dir() / "ungdc" / "TXT"
OUTPUT_DIR = individual_source_dir("ungdc_sdg")
OUTPUT_JSONL = OUTPUT_DIR / "ungdc_sdg_clean.jsonl"
STATE_PATH = OUTPUT_DIR / "ungdc_sdg_state.json"
STATUS_DIR = OUTPUT_DIR / "metadata"

# Sessions after SDG adoption (September 2015 = Session 70)
MIN_SESSION = 70
MAX_SESSION = 80

# Minimum paragraph length to consider (words)
MIN_PARA_WORDS = 20

# ---------------------------------------------------------------------------
# SDG relevance keywords — broad enough to catch policy discourse without
# being so broad we include generic political speech
# ---------------------------------------------------------------------------
SDG_PATTERNS = re.compile(
    r"\b("
    r"sustainable development goal[s]?"
    r"|SDG[s\s\d]"
    r"|2030 agenda"
    r"|agenda 2030"
    r"|sustainable development"
    r"|sustainability"
    r"|climate change"
    r"|climate action"
    r"|global warming"
    r"|net.?zero"
    r"|carbon emission[s]?"
    r"|renewable energy"
    r"|clean energy"
    r"|poverty reduction"
    r"|eradicating poverty"
    r"|food security"
    r"|hunger"
    r"|universal health"
    r"|quality education"
    r"|gender equality"
    r"|women.s rights"
    r"|clean water"
    r"|sanitation"
    r"|decent work"
    r"|economic inequality"
    r"|reduce inequalit"
    r"|inclusive growth"
    r"|sustainable infrastructure"
    r"|responsible consumption"
    r"|biodiversity"
    r"|deforestation"
    r"|ocean[s]? conservation"
    r"|peace and justice"
    r"|rule of law"
    r"|global partnership[s]?"
    r"|multilateralism"
    r"|artificial intelligence"
    r"|digital transformation"
    r"|technology for development"
    r"|fourth industrial revolution"
    r"|digital economy"
    r"|IPCC"
    r"|Paris Agreement"
    r"|COP\s*\d+"
    r"|net zero"
    r"|greenhouse gas"
    r")\b",
    re.IGNORECASE,
)


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines or single newlines; discard very short fragments.

    UNGDC files use single newlines between paragraphs (no blank lines),
    so we split on both blank-line boundaries and single newlines.
    """
    if "\n\n" in text:
        raw_paras = re.split(r"\n\s*\n", text)
    else:
        raw_paras = text.split("\n")

    result = []
    for p in raw_paras:
        p = p.strip()
        p = re.sub(r"\s{2,}", " ", p)
        if len(p.split()) >= MIN_PARA_WORDS:
            result.append(p)
    return result


def parse_session_dir(session_dir: Path) -> tuple[int, int] | None:
    """Extract (session_number, year) from folder name like 'Session 70 - 2015'."""
    m = re.match(r"Session\s+(\d+)\s+-\s+(\d+)", session_dir.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_speech_file(path: Path) -> tuple[str, int, int] | None:
    """Extract (iso3, session, year) from filename like 'AFG_70_2015.txt'."""
    m = re.match(r"([A-Z]{2,3})_(\d+)_(\d+)\.txt", path.name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def read_records():
    if not UNGDC_TXT_DIR.exists():
        return
    session_dirs = []
    for d in sorted(UNGDC_TXT_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_session_dir(d)
        if parsed and MIN_SESSION <= parsed[0] <= MAX_SESSION:
            session_dirs.append((parsed[0], parsed[1], d))
    for session_num, year, session_dir in session_dirs:
        for speech_path in sorted(session_dir.glob("*.txt")):
            parsed = parse_speech_file(speech_path)
            if not parsed:
                continue
            yield session_num, year, speech_path


def transform(payload) -> dict | None:
    session_num, year, speech_path = payload
    parsed = parse_speech_file(speech_path)
    if not parsed:
        return None
    iso3, _, _ = parsed

    try:
        text = speech_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    paragraphs = split_paragraphs(text)
    relevant = [p for p in paragraphs if SDG_PATTERNS.search(p)]
    if not relevant:
        return None

    speech_text = " ".join(relevant)
    source_doc = f"ungdc_{iso3}_{session_num}_{year}"

    return {
        "id": speech_path.stem,
        "text": speech_text,
        "source_doc": source_doc,
        "institution": f"{iso3} (UNGDC speech)",
        "year": year,
        "session": session_num,
        "source": "ungdc_sdg",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter UNGDC speeches for SDG content (resume-safe).")
    p.add_argument("--txt-dir", default=str(UNGDC_TXT_DIR))
    p.add_argument("--out-jsonl", default=str(OUTPUT_JSONL))
    p.add_argument("--state", default=str(STATE_PATH))
    p.add_argument("--status-dir", default=str(STATUS_DIR))
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reset", action="store_true", help="Delete checkpoint + output and start fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    global UNGDC_TXT_DIR, OUTPUT_JSONL, STATE_PATH, STATUS_DIR
    UNGDC_TXT_DIR = Path(args.txt_dir)
    OUTPUT_JSONL = Path(args.out_jsonl)
    STATE_PATH = Path(args.state)
    STATUS_DIR = Path(args.status_dir)

    if not UNGDC_TXT_DIR.exists():
        print(f"ERROR: {UNGDC_TXT_DIR} not found. Run fetch_ungdc.py first.")
        return

    resumable_records(
        stage="filter_ungdc_sdg",
        read_records=read_records,
        transform=transform,
        out_path=OUTPUT_JSONL,
        state_path=STATE_PATH,
        status_dir=STATUS_DIR,
        chunk_size=args.chunk_size,
        reset=args.reset,
        dumps=lambda r: json.dumps(r, ensure_ascii=False),
    )

    n = sum(1 for line in OUTPUT_JSONL.open(encoding="utf-8") if line.strip()) if OUTPUT_JSONL.exists() else 0
    print(f"\n{'='*60}")
    print(f"Total documents:              {n}")
    print(f"Output:                       {OUTPUT_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
