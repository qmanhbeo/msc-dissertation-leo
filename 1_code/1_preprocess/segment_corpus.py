"""
Segment a preprocessed corpus using token-count-aware segmentation.

Applies the canonical segment_text() from segment_utils.py to every record
in a JSONL corpus, generating model-specific outputs with consistent fields.

Preserves all original metadata fields. Adds: segment_id, source_doc,
segment_index, word_count (recalculated per segment).

Handles both sdg: int and sdgs: list[int] label fields transparently.

Run from project root:
    # Single-file corpora (KH, Aurora)
    python 1_code/1_preprocess/segment_corpus.py \
        --input 2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl \
        --output 2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_segmented_{model}.jsonl \
        --text-field text --id-field id --prefix kh --model all-mpnet-base-v2

    python 1_code/1_preprocess/segment_corpus.py \
        --input 2_data/1_preprocessed/aurora/aurora_texts.jsonl \
        --output 2_data/1_preprocessed/aurora/aurora_segmented_{model}.jsonl \
        --text-field text --id-field doi --prefix aurora --model all-mpnet-base-v2

    # Sharded corpora (Research)
    python 1_code/1_preprocess/segment_corpus.py \
        --sharded --input-glob '2_data/1_preprocessed/research_corpus/part-*.jsonl' \
        --output-dir 2_data/1_preprocessed/research_corpus/segmented_{model} \
        --text-field combined_text --id-field openalex_id --prefix paper \
        --model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sentence_transformers import SentenceTransformer

from segment_utils import segment_text, verify_truncation_rate

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MIN_WORDS = 20


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalise_sdgs(entry: dict, sdg_field: str) -> list[int] | None:
    raw = entry.get(sdg_field)
    if raw is None:
        alt = "sdgs" if sdg_field == "sdg" else "sdg"
        raw = entry.get(alt)
    if raw is None:
        return None
    if isinstance(raw, list):
        return sorted(int(s) for s in raw if 1 <= s <= 17)
    if isinstance(raw, (int, float)):
        v = int(raw)
        return [v] if 1 <= v <= 17 else None
    return None


def segment_records(
    records: list[dict],
    model: SentenceTransformer,
    text_field: str,
    id_field: str,
    prefix: str,
) -> list[dict]:
    segments_out: list[dict] = []
    doc_counts: dict[str, int] = {}

    for idx, rec in enumerate(records):
        text = rec.get(text_field, "")
        if not text or not isinstance(text, str) or len(text.split()) < MIN_WORDS:
            continue

        sub_texts = segment_text(text, model)
        if not sub_texts:
            continue

        source_doc = f"{prefix}_{rec.get(id_field, str(idx))}"
        if source_doc not in doc_counts:
            doc_counts[source_doc] = 0

        sdgs = _normalise_sdgs(rec, "sdgs") or _normalise_sdgs(rec, "sdg")

        for si, sub_text in enumerate(sub_texts):
            seg = dict(rec)
            seg["segment_id"] = f"{prefix}_{idx:07d}_{si}"
            seg["source_doc"] = source_doc
            seg["segment_index"] = doc_counts[source_doc]
            seg["text"] = sub_text
            seg["word_count"] = len(sub_text.split())
            if sdgs is not None:
                seg["sdgs"] = sdgs
            if "sdg" in seg and "sdgs" in seg:
                del seg["sdg"]
            doc_counts[source_doc] += 1
            segments_out.append(seg)

    return segments_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a preprocessed corpus.")
    parser.add_argument("--input", help="Single input JSONL path.")
    parser.add_argument("--output", help="Single output JSONL path (supports {model} placeholder).")
    parser.add_argument("--sharded", action="store_true", help="Sharded input mode.")
    parser.add_argument("--input-glob", help="Glob pattern for sharded input files.")
    parser.add_argument("--output-dir", help="Output dir for sharded mode (supports {model} placeholder).")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--prefix", default="doc", help="Prefix for segment_id and source_doc.")
    parser.add_argument("--model", default="all-mpnet-base-v2")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").lower()

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)

    if args.sharded:
        input_paths = sorted(Path(p) for p in glob.glob(args.input_glob))
        if not input_paths:
            log.error("No input files match: %s", args.input_glob)
            return
        output_dir = Path(args.output_dir.format(model=model_slug))
        output_dir.mkdir(parents=True, exist_ok=True)

        total_segments = 0
        all_seg_texts = []

        for in_path in input_paths:
            log.info("Processing: %s", in_path)
            records = _load_jsonl(in_path)
            segments = segment_records(records, model, args.text_field, args.id_field, args.prefix)
            all_seg_texts.extend(s["text"] for s in segments)

            out_path = output_dir / in_path.name
            with out_path.open("w", encoding="utf-8") as f:
                for s in segments:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            log.info("  %s -> %s (%d segments)", in_path.name, out_path.name, len(segments))
            total_segments += len(segments)

        log.info("Total segments: %d", total_segments)
    else:
        if not args.input or not args.output:
            log.error("Single-file mode requires --input and --output.")
            return
        input_path = Path(args.input)
        output_path = Path(args.output.format(model=model_slug))

        log.info("Loading: %s", input_path)
        records = _load_jsonl(input_path)
        segments = segment_records(records, model, args.text_field, args.id_field, args.prefix)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for s in segments:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        log.info("Wrote %d segments -> %s", len(segments), output_path)
        all_seg_texts = [s["text"] for s in segments]

    if all_seg_texts:
        verify_truncation_rate(all_seg_texts, model, label=f"{args.prefix}")
    print(f"\nDone. {len(all_seg_texts)} segments total.")


if __name__ == "__main__":
    main()
