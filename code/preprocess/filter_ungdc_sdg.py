"""
Extract SDG-relevant policy passages from the UN General Debate Corpus (UNGDC).

Input:  data/raw/ungdc/TXT/Session <N> - <YEAR>/<ISO>_<session>_<year>.txt
        Sessions 70–80 (2015–2024) — post-SDG-adoption speeches only

Output: data/preprocessed/policy_all/ungdc_sdg/ungdc_sdg_chunks.jsonl

Strategy:
  1. Read all speeches from sessions 70–80 (2015–2024)
  2. Split each speech into paragraphs
  3. Keep paragraphs that contain SDG-relevant keywords
  4. Merge adjacent kept paragraphs into ~150-word chunks
  5. Output in the standard policy_chunks format

These are official statements by heads of state/government at the UN General
Assembly — authentic policy discourse on international goals and commitments.

Run from project root:
    python code/preprocess/filter_ungdc_sdg.py
"""

import json
import re
from pathlib import Path

UNGDC_TXT_DIR = Path("data/raw/ungdc/TXT")
OUTPUT_DIR = Path("data/preprocessed/policy_all/ungdc_sdg")
OUTPUT_JSONL = OUTPUT_DIR / "ungdc_sdg_chunks.jsonl"

# Sessions after SDG adoption (September 2015 = Session 70)
MIN_SESSION = 70
MAX_SESSION = 80

# Minimum paragraph length to consider (words)
MIN_PARA_WORDS = 20
# Target chunk size (words)
TARGET_CHUNK_WORDS = 150
# Hard cap (words)
MAX_CHUNK_WORDS = 300

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
    # Try double-newline split first
    if "\n\n" in text:
        raw_paras = re.split(r"\n\s*\n", text)
    else:
        # Fall back to single newline (UNGDC format)
        raw_paras = text.split("\n")

    result = []
    for p in raw_paras:
        p = p.strip()
        p = re.sub(r"\s{2,}", " ", p)
        if len(p.split()) >= MIN_PARA_WORDS:
            result.append(p)
    return result


def merge_chunks(paragraphs: list[str], target: int, max_words: int) -> list[str]:
    """Greedily merge consecutive paragraphs into ~target-word chunks."""
    chunks, current, current_wc = [], [], 0
    for para in paragraphs:
        wc = len(para.split())
        if current_wc + wc > max_words and current:
            chunks.append(" ".join(current))
            current, current_wc = [para], wc
        else:
            current.append(para)
            current_wc += wc
        if current_wc >= target:
            chunks.append(" ".join(current))
            current, current_wc = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


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

    all_chunks: list[dict] = []
    chunk_idx = 0
    speech_count = 0
    kept_speech_count = 0

    for session_num, year, session_dir in session_dirs:
        speech_files = sorted(session_dir.glob("*.txt"))
        print(f"  Session {session_num} ({year}): {len(speech_files)} speeches", end="", flush=True)

        session_chunks = 0
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
            # Filter to SDG-relevant paragraphs
            relevant = [p for p in paragraphs if SDG_PATTERNS.search(p)]
            if not relevant:
                continue

            kept_speech_count += 1
            merged = merge_chunks(relevant, TARGET_CHUNK_WORDS, MAX_CHUNK_WORDS)

            source_doc = f"ungdc_{iso3}_{session_num}_{year}"
            for i, chunk_text in enumerate(merged):
                if len(chunk_text.split()) < MIN_PARA_WORDS:
                    continue
                all_chunks.append(
                    {
                        "chunk_id": f"ungdc_{chunk_idx:06d}",
                        "source_doc": source_doc,
                        "chunk_index": i,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "institution": f"{iso3} (UNGDC speech)",
                        "year": year,
                        "session": session_num,
                    }
                )
                chunk_idx += 1
                session_chunks += 1

        print(f"  → {session_chunks} chunks")

    # Write output
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n{'='*60}")
    print(f"Speeches scanned:  {speech_count}")
    print(f"Speeches with SDG content: {kept_speech_count} ({kept_speech_count/max(speech_count,1)*100:.1f}%)")
    print(f"Total chunks:      {len(all_chunks)}")
    print(f"Output:            {OUTPUT_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
