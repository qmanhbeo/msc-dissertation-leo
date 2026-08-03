"""
Segment a preprocessed corpus using token-count-aware segmentation.

Applies the canonical segment_text() from segment_utils.py to every record
in a JSONL corpus, generating model-specific outputs with consistent fields.

Preserves all original metadata fields. Adds: segment_id, source_doc,
segment_index, word_count (recalculated per segment).

Handles both sdg: int and sdgs: list[int] label fields transparently.

Paths are resolved via model_utils helpers. From project root:
    # Single-file corpora (KH, Aurora)
    python 1_code/2_segment/1_segment_corpus.py \
        --input <preprocessed_dir() / corpus / ..._clean.jsonl> \
        --output <segmented_dir_for_model(model) / corpus.jsonl> \
        --text-field text --id-field id --prefix kh --model all-mpnet-base-v2

    # Sharded corpora (Research)
    python 1_code/2_segment/1_segment_corpus.py \
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
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import CANONICAL_MAX_SEQ_LENGTH, model_slug, preprocessed_dir, research_concept_preprocessed_dir, research_concept_segmented_dir_for_model, research_preprocessed_dir, research_segmented_dir_for_model, segmented_dir_for_model, DEFAULT_EMBED_MODEL, resolve_model_alias
from segment_utils import segment_text, _ensure_nltk_data
from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, sha256_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


_WTOK = None
_WMAXLEN = None


def _worker_init(model_name: str, max_len: int) -> None:
    """Load a lightweight tokenizer per worker (no full model weights)."""
    global _WTOK, _WMAXLEN
    _WTOK = AutoTokenizer.from_pretrained(
        "sentence-transformers/" + model_name, local_files_only=True
    )
    _WMAXLEN = max_len
    _ensure_nltk_data()


def _segment_shard_worker(task):
    """Process one shard in a worker process and return a manifest entry."""
    shard_idx, in_path_str, out_path_str, text_field, id_field, prefix, overwrite = task
    in_path = Path(in_path_str)
    out_path = Path(out_path_str)

    if out_path.exists() and not overwrite:
        existing = sum(1 for line in open(out_path, encoding="utf-8") if line.strip())
        if existing > 0:
            try:
                with open(out_path, encoding="utf-8") as fc:
                    json.loads(fc.readline())
                return {
                    "shard_id": shard_idx, "skip": True, "name": in_path.stem,
                    "rows": existing, "bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                }
            except (json.JSONDecodeError, StopIteration):
                pass

    records = _load_records(in_path)
    segments = segment_records(records, _WTOK, _WMAXLEN, text_field, id_field, prefix)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)

    return {
        "shard_id": shard_idx, "skip": False, "name": in_path.stem,
        "rows": len(segments), "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
    }

MIN_WORDS = 20


def _load_records(path: Path) -> list[dict]:
    """Load records from JSONL or JSON array file."""
    with path.open(encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
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
    tokenizer,
    max_seq_length: int,
    text_field: str,
    id_field: str,
    prefix: str,
    min_words: int = MIN_WORDS,
) -> list[dict]:
    segments_out: list[dict] = []
    doc_counts: dict[str, int] = {}

    for idx, rec in enumerate(records):
        text = rec.get(text_field, "")
        if not text or not isinstance(text, str) or len(text.split()) < min_words:
            continue

        sub_texts = segment_text(text, tokenizer, max_seq_length)
        if not sub_texts:
            continue

        source_doc = rec.get("source_doc") or f"{prefix}_{rec.get(id_field, str(idx))}"
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
    parser.add_argument("--corpus", choices=["reference", "policy", "research", "research_concept"],
                        help="Known corpus name; auto-derives input/output from model_utils (alternative to --input/--output).")
    parser.add_argument("--input", help="Single input JSONL path.")
    parser.add_argument("--output", help="Single output JSONL path (supports {model} placeholder).")
    parser.add_argument("--sharded", action="store_true", help="Sharded input mode.")
    parser.add_argument("--input-glob", help="Glob pattern for sharded input files.")
    parser.add_argument("--output-dir", help="Output dir for sharded mode (supports {model} placeholder).")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--prefix", default="doc", help="Prefix for segment_id and source_doc.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS,
                        help="Drop texts (or segments) shorter than this many words (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-segment existing shards even if already complete")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes for sharded mode (default: os.cpu_count())")
    parser.add_argument("--all", action="store_true", dest="all_corpora",
                        help="Segment all non-research corpora in one model load.")
    args = parser.parse_args()

    if args.all_corpora:
        corpora = [
            ("reference", str(preprocessed_dir() / "reference.jsonl"), "id", "ref"),
            ("policy", str(preprocessed_dir() / "policy.jsonl"), "id", "pol"),
        ]
        any_work = any(
            not (segmented_dir_for_model(args.embed_model) / f"{name}.jsonl").exists()
            or args.overwrite for name, _, _, _ in corpora
        )
        if not any_work:
            log.info("All corpora already exist — nothing to do")
            return
        tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/" + args.embed_model, local_files_only=True
        )
        max_seq_length = CANONICAL_MAX_SEQ_LENGTH
        for corpus_name, input_str, id_field, prefix in corpora:
            input_path = Path(input_str)
            output_path = segmented_dir_for_model(args.embed_model) / f"{corpus_name}.jsonl"
            if output_path.exists() and not args.overwrite:
                log.info("Skip %s — already exists", corpus_name)
                continue
            log.info("Processing %s", corpus_name)
            records = _load_records(input_path)
            segments = segment_records(records, tok, max_seq_length, "text", id_field, prefix)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as f:
                for s in segments:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            log.info("  wrote %d segments -> %s", len(segments), output_path.name)
        return

    if args.corpus == "research":
        args.sharded = True
        args.input_glob = str(research_preprocessed_dir() / "part-*.jsonl")
        args.output_dir = str(research_segmented_dir_for_model(args.embed_model))
        args.text_field = "combined_text"
        args.id_field = "openalex_id"
        args.prefix = "paper"
    elif args.corpus == "research_concept":
        args.sharded = True
        args.input_glob = str(research_concept_preprocessed_dir() / "part-*.jsonl")
        args.output_dir = str(research_concept_segmented_dir_for_model(args.embed_model))
        args.text_field = "combined_text"
        args.id_field = "openalex_id"
        args.prefix = "paper"
    elif args.corpus == "reference":
        if not args.input:
            args.input = str(preprocessed_dir() / "reference.jsonl")
        if not args.output:
            args.output = str(segmented_dir_for_model(args.embed_model) / "reference.jsonl")
        if not args.prefix or args.prefix == "doc":
            args.prefix = "ref"
        if not args.id_field or args.id_field == "id":
            args.id_field = "id"
    elif args.corpus == "policy":
        if not args.input:
            args.input = str(preprocessed_dir() / "policy.jsonl")
        if not args.output:
            args.output = str(segmented_dir_for_model(args.embed_model) / "policy.jsonl")
        if not args.prefix or args.prefix == "doc":
            args.prefix = "pol"
        if not args.id_field or args.id_field == "id":
            args.id_field = "id"

    mslug = model_slug(args.embed_model)

    log.info("Loading tokenizer: %s", args.embed_model)
    tok = AutoTokenizer.from_pretrained(
        "sentence-transformers/" + args.embed_model, local_files_only=True
    )
    max_seq_length = CANONICAL_MAX_SEQ_LENGTH

    if args.sharded:
        input_paths = sorted(Path(p) for p in glob.glob(args.input_glob))
        if not input_paths:
            log.error("No input files match: %s", args.input_glob)
            return
        output_dir = Path(args.output_dir.format(model=mslug))
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            (shard_idx, str(in_path), str(output_dir / in_path.name),
             args.text_field, args.id_field, args.prefix, args.overwrite)
            for shard_idx, in_path in enumerate(input_paths, start=1)
        ]
        n_workers = args.workers if args.workers > 0 else max(1, min(os.cpu_count() or 2, len(tasks)))
        mp_ctx = multiprocessing.get_context("spawn")
        manifest_entries: list[dict] = []
        total_segments = 0
        log.info("Segmenting %d shards across %d workers", len(tasks), n_workers)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_worker_init,
            initargs=(args.embed_model, max_seq_length),
        ) as ex:
            for r in ex.map(_segment_shard_worker, tasks):
                manifest_entries.append({
                    "shard_id": r["shard_id"],
                    "name": r["name"],
                    "rows": r["rows"],
                    "bytes": r["bytes"],
                    "sha256": r["sha256"],
                })
                total_segments += r["rows"]
                if r.get("skip"):
                    log.info("Skip %s — already exists (%d segments)", r["name"], r["rows"])
                else:
                    log.info("  %s -> %d segments", r["name"], r["rows"])

        metadata_dir = output_dir / "metadata"
        ensure_dir(metadata_dir)
        manifest = {
            "stage": "research_segmentation",
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "model": args.embed_model,
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
        output_path = Path(args.output.format(model=mslug))

        if output_path.exists() and not args.overwrite:
            log.info("Skip %s — already exists", output_path.name)
            return

        log.info("Loading: %s", input_path)
        records = _load_records(input_path)
        segments = segment_records(records, tok, max_seq_length, args.text_field, args.id_field, args.prefix, args.min_words)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: stage to a .tmp sibling, then replace, so an interrupted
        # run never leaves a torn file that a later exists-skip would accept.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            for s in segments:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)

        log.info("Wrote %d segments -> %s", len(segments), output_path)


if __name__ == "__main__":
    main()
