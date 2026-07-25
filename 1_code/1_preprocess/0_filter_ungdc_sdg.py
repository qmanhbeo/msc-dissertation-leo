"""
Extract SDG-relevant policy passages from the UN General Debate Corpus (UNGDC).

Input:  2_data/0_raw/ungdc/TXT/Session <N> - <YEAR>/<ISO>_<session>_<year>.txt
        Sessions 70–80 (2015–2024) — post-SDG-adoption speeches only

Output: 2_data/1_preprocessed/policy_all/ungdc_sdg/ungdc_sdg_clean.jsonl

Strategy:
  1. Read all speeches from sessions 70–80 (2015–2024)
  2. Split each speech into paragraphs
  3. Keep paragraphs that contain SDG-relevant keywords
  4. Concatenate the SDG-relevant paragraphs into a single document per speech
  5. Output documents (not segments) for downstream segmentation

Preserves paragraph-level SDG keyword filtering before segmentation.
Only the merge/segmentation step is now handled by segment_corpus.py.

Run from project root:
    python 1_code/1_preprocess/0_filter_ungdc_sdg.py
"""

import json
import re
from pathlib import Path

import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import raw_dir, preprocessed_dir

UNGDC_TXT_DIR = raw_dir() / "ungdc" / "TXT"
OUTPUT_DIR = preprocessed_dir() / "policy_all" / "ungdc_sdg"
OUTPUT_JSONL = OUTPUT_DIR / "ungdc_sdg_clean.jsonl"

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not UNGDC_TXT_DIR.exists():
        print(f"ERROR: {UNGDC_TXT_DIR} not found. Run fetch_ungdc.py first.")
        return

    # Collect all session directories in range
    session_dirs = []
    for d in sorted(UNGDC_TXT_DIR.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_session_dir(d)
        if parsed and MIN_SESSION <= parsed[0] <= MAX_SESSION:
            session_dirs.append((parsed[0], parsed[1], d))

    print(f"Found {len(session_dirs)} sessions ({MIN_SESSION}–{MAX_SESSION})")

    all_records: list[dict] = []
    speech_count = 0
    kept_speech_count = 0
    para_before_count = 0
    para_after_count = 0

    for session_num, year, session_dir in session_dirs:
        speech_files = sorted(session_dir.glob("*.txt"))
        print(f"  Session {session_num} ({year}): {len(speech_files)} speeches", end="", flush=True)

        session_docs = 0
        for speech_path in speech_files:
            parsed = parse_speech_file(speech_path)
            if not parsed:
                continue
            iso3, _, _ = parsed
            speech_count += 1

            try:
                text = speech_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            paragraphs = split_paragraphs(text)
            para_before_count += len(paragraphs)
            relevant = [p for p in paragraphs if SDG_PATTERNS.search(p)]
            if not relevant:
                continue

            para_after_count += len(relevant)
            kept_speech_count += 1
            speech_text = " ".join(relevant)
            source_doc = f"ungdc_{iso3}_{session_num}_{year}"

            all_records.append({
                "id": speech_path.stem,
                "text": speech_text,
                "source_doc": source_doc,
                "institution": f"{iso3} (UNGDC speech)",
                "year": year,
                "session": session_num,
            })
            session_docs += 1

        print(f"  → {session_docs} documents")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Speeches scanned:             {speech_count}")
    print(f"Speeches with SDG content:    {kept_speech_count} ({kept_speech_count/max(speech_count,1)*100:.1f}%)")
    print(f"Paragraphs before filter:     {para_before_count}")
    print(f"Paragraphs after filter:      {para_after_count}")
    print(f"Filter retention:             {para_after_count/max(para_before_count,1)*100:.1f}%")
    print(f"Total documents:              {len(all_records)}")
    print(f"Output:                       {OUTPUT_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
