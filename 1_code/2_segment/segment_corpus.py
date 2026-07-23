"""
Segment a preprocessed corpus using token-count-aware segmentation.

Applies the canonical segment_text() from segment_utils.py to every record
in a JSONL corpus, generating model-specific outputs with consistent fields.

Preserves all original metadata fields. Adds: segment_id, source_doc,
segment_index, word_count (recalculated per segment).

Handles both sdg: int and sdgs: list[int] label fields transparently.

Paths are resolved via model_utils helpers. From project root:
    # Single-file corpora (KH, Aurora)
    python 1_code/2_segment/segment_corpus.py \
        --input <preprocessed_dir() / corpus / ..._clean.jsonl> \
        --output <segmented_dir_for_model(model) / corpus.jsonl> \
        --text-field text --id-field id --prefix kh --model all-mpnet-base-v2

    # Sharded corpora (Research)
    python 1_code/2_segment/segment_corpus.py \
        --sharded \
        --input-glob <research_preprocessed_dir() / part-*.jsonl> \
        --output-dir <research_segmented_dir_for_model(model)> \
        --text-field combined_text --id-field openalex_id --prefix paper \
        --model all-mpnet-base-v2

See model_utils.py for all path helpers. Use --corpus for auto-derived paths.
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
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sentence_transformers import SentenceTransformer

from model_utils import preprocessed_dir, research_preprocessed_dir, research_segmented_dir_for_model, segmented_dir_for_model
from segment_utils import segment_text, verify_truncation_rate
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, sha256_file

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
    parser.add_argument("--corpus", choices=["sdg_knowledge_hub", "aurora", "research"],
                        help="Known corpus name; auto-derives input/output from model_utils (alternative to --input/--output).")
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

    if args.corpus == "research":
        args.sharded = True
        args.input_glob = str(research_preprocessed_dir() / "part-*.jsonl")
        args.output_dir = str(research_segmented_dir_for_model(args.model))
        args.text_field = "combined_text"
        args.id_field = "openalex_id"
        args.prefix = "paper"
    elif args.corpus == "sdg_knowledge_hub":
        if not args.input:
            args.input = str(preprocessed_dir() / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.jsonl")
        if not args.output:
            args.output = str(segmented_dir_for_model(args.model) / "sdg_knowledge_hub.jsonl")
        if not args.prefix or args.prefix == "doc":
            args.prefix = "kh"
        if not args.id_field or args.id_field == "id":
            args.id_field = "id"
    elif args.corpus == "aurora":
        if not args.input:
            args.input = str(preprocessed_dir() / "aurora" / "aurora_texts.jsonl")
        if not args.output:
            args.output = str(segmented_dir_for_model(args.model) / "aurora.jsonl")
        if not args.prefix or args.prefix == "doc":
            args.prefix = "aurora"
        if not args.id_field or args.id_field == "id":
            args.id_field = "doi"

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
        manifest_entries: list[dict] = []

        for shard_idx, in_path in enumerate(input_paths, start=1):
            out_path = output_dir / in_path.name

            if out_path.exists():
                existing = sum(1 for _ in open(out_path, encoding="utf-8") if _.strip())
                if existing > 0:
                    try:
                        with open(out_path, encoding="utf-8") as f_check:
                            json.loads(f_check.readline())
                    except (json.JSONDecodeError, StopIteration):
                        log.warning("Corrupt output for %s — re-processing", in_path.name)
                        existing = 0
                if existing > 0:
                    log.info("Skip %s — already exists (%d segments)", in_path.name, existing)
                    total_segments += existing
                    manifest_entries.append({
                        "shard_id": shard_idx,
                        "name": in_path.stem,
                        "rows": existing,
                        "bytes": out_path.stat().st_size,
                        "sha256": sha256_file(out_path),
                    })
                    continue

            log.info("Processing: %s", in_path)
            records = _load_jsonl(in_path)
            segments = segment_records(records, model, args.text_field, args.id_field, args.prefix)
            all_seg_texts.extend(s["text"] for s in segments)

            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for s in segments:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            tmp_path.replace(out_path)

            log.info("  %s -> %s (%d segments)", in_path.name, out_path.name, len(segments))
            total_segments += len(segments)
            manifest_entries.append({
                "shard_id": shard_idx,
                "name": in_path.stem,
                "rows": len(segments),
                "bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
            })

        metadata_dir = output_dir / "metadata"
        ensure_dir(metadata_dir)
        manifest = {
            "stage": "research_segmentation",
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "model": args.model,
            "shards": manifest_entries,
            "totals": {"rows": total_segments, "shards": len(manifest_entries)},
        }
        atomic_write_json(metadata_dir / "manifest.json", manifest)
        log.info("Wrote segment manifest: %s", metadata_dir / "manifest.json")
        log.info("Total segments: %d across %d shards", total_segments, len(manifest_entries))
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
