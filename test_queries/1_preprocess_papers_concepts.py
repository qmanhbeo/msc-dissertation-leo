"""
Streaming OpenAlex cleaner with checkpoint + resume safety.

This stage reads the raw OpenAlex JSONL line-by-line and writes clean shards.
It is safe to interrupt and resume. Progress is persisted under:
  - 2_data/3_embedded/{model}/research_shards/metadata/openalex_papers_to_clean_shards.json
  - 2_data/1_preprocessed/research_corpus/metadata/state.json

Outputs:
  2_data/1_preprocessed/research_corpus/part-00001.jsonl
  2_data/1_preprocessed/research_corpus/metadata/part-00001_ids.jsonl
  2_data/1_preprocessed/research_corpus/metadata/manifest.json
  2_data/1_preprocessed/research_corpus/metadata/dedupe.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shard_pipeline_utils import atomic_write_json, ensure_dir, now_iso, read_json, sha256_file, update_stage_status
from model_utils import raw_dir, preprocessed_dir


log = logging.getLogger(__name__)
STATUS_STAGE = "openalex_papers_to_clean_shards"

_BOILERPLATE = [
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"https?://\S+"),
    re.compile(r"\S+@\S+\.\S+"),
    re.compile(r"\b(doi|DOI):\s*\S+"),
]
_MULTI_SPACE = re.compile(r"\s{2,}")
_LEADING_PUNCT = re.compile(r"^[\s\-–—,.:;]+")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _LEADING_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def cleaned_record(raw: dict[str, Any], min_abstract_chars: int) -> dict[str, Any] | None:
    title = clean_text(raw.get("title", ""))
    abstract = clean_text(raw.get("abstract", ""))
    if not abstract or len(abstract) < min_abstract_chars:
        return None
    combined_text = f"{title}. {abstract}" if title else abstract
    concepts_sorted = sorted(raw.get("concepts", []), key=lambda c: c.get("score", 0), reverse=True)
    top_concepts = [c["display_name"] for c in concepts_sorted[:3] if c.get("display_name")]
    return {
        "openalex_id": raw.get("openalex_id", ""),
        "title": title,
        "abstract": abstract,
        "combined_text": combined_text,
        "doi": raw.get("doi", ""),
        "publication_year": raw.get("publication_year"),
        "cited_by_count": raw.get("cited_by_count", 0),
        "author_count": raw.get("author_count", 0),
        "top_concepts": top_concepts,
        "source_url": raw.get("source_url", ""),
        "abstract_word_count": len(abstract.split()),
    }


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_ids (
            openalex_id TEXT PRIMARY KEY
        )
        """
    )
    conn.commit()
    return conn


def id_seen(conn: sqlite3.Connection, openalex_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_ids WHERE openalex_id = ? LIMIT 1", (openalex_id,)).fetchone()
    return row is not None


def mark_seen(conn: sqlite3.Connection, openalex_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO seen_ids(openalex_id) VALUES (?)", (openalex_id,))


def pending_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def read_pending(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate_file(path: Path) -> None:
    with path.open("w", encoding="utf-8"):
        pass


def append_pending(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_shard(
    out_dir: Path,
    metadata_dir: Path,
    manifest_path: Path,
    state_path: Path,
    state: dict[str, Any],
    pending_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not pending_rows:
        return state

    shard_idx = int(state["next_shard_idx"])
    shard_name = f"part-{shard_idx:05d}"
    shard_path = out_dir / f"{shard_name}.jsonl"
    ids_path = metadata_dir / f"{shard_name}_ids.jsonl"
    shard_tmp = shard_path.with_suffix(".jsonl.tmp")
    ids_tmp = ids_path.with_suffix(".jsonl.tmp")

    per_year: dict[str, int] = {}
    max_offset = 0

    with shard_tmp.open("w", encoding="utf-8") as f_data, ids_tmp.open("w", encoding="utf-8") as f_ids:
        for i, row in enumerate(pending_rows):
            rec = row["record"]
            year = rec.get("publication_year")
            year_key = str(year) if year is not None else "unknown"
            per_year[year_key] = per_year.get(year_key, 0) + 1
            max_offset = max(max_offset, int(row["input_offset_end"]))
            f_data.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_ids.write(
                json.dumps(
                    {
                        "openalex_id": rec["openalex_id"],
                        "publication_year": rec.get("publication_year"),
                        "row_in_shard": i,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    shard_tmp.replace(shard_path)
    ids_tmp.replace(ids_path)

    manifest = read_json(manifest_path, default=None)
    if manifest is None:
        manifest = {
            "stage": STATUS_STAGE,
            "schema_version": 1,
            "created_at_utc": now_iso(),
            "shards": [],
            "totals": {"rows": 0, "shards": 0},
            "counts_by_year": {},
        }

    rows = len(pending_rows)
    manifest["shards"].append(
        {
            "shard_id": shard_idx,
            "name": shard_name,
            "data_path": str(shard_path),
            "ids_path": str(ids_path),
            "rows": rows,
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
            "ids_sha256": sha256_file(ids_path),
            "per_year": per_year,
        }
    )
    manifest["totals"]["rows"] += rows
    manifest["totals"]["shards"] = len(manifest["shards"])
    for y, c in per_year.items():
        manifest["counts_by_year"][y] = manifest["counts_by_year"].get(y, 0) + c
    atomic_write_json(manifest_path, manifest)

    state["next_shard_idx"] = shard_idx + 1
    state["last_input_offset"] = max_offset
    state["rows_written"] = int(state["rows_written"]) + rows
    atomic_write_json(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(raw_dir() / "openalex" / "papers.jsonl"))
    p.add_argument("--out-dir", default=str(preprocessed_dir() / "research_corpus"))
    p.add_argument("--status-dir", default=str(preprocessed_dir() / "research_corpus" / "metadata"))
    p.add_argument("--metadata-dir", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--state", default="")
    p.add_argument("--db", default="")
    p.add_argument("--pending", default="")
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--min-abstract-chars", type=int, default=30)
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--limit", type=int, default=0, help="Stop after N kept rows (0 = no limit)")
    p.add_argument("--reset", action="store_true", help="Delete state + manifest + pending and start fresh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    status_dir = Path(args.status_dir)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else out_dir / "metadata"
    ensure_dir(out_dir)
    ensure_dir(status_dir)
    ensure_dir(metadata_dir)

    manifest_path = Path(args.manifest) if args.manifest else metadata_dir / "manifest.json"
    state_path = Path(args.state) if args.state else metadata_dir / "state.json"
    db_path = Path(args.db) if args.db else metadata_dir / "dedupe.sqlite"
    pending_path = Path(args.pending) if args.pending else metadata_dir / "pending.jsonl"

    if args.reset:
        for p in (manifest_path, state_path, db_path, pending_path):
            if p.exists():
                p.unlink()
        for p in out_dir.glob("part-*.jsonl"):
            p.unlink()
        for p in metadata_dir.glob("part-*_ids.jsonl"):
            p.unlink()

    state = read_json(
        state_path,
        default={
            "schema_version": 1,
            "last_input_offset": 0,
            "next_shard_idx": 1,
            "rows_written": 0,
            "rows_scanned": 0,
            "rows_kept": 0,
            "rows_dropped": 0,
            "rows_duplicate": 0,
        },
    )
    atomic_write_json(state_path, state)

    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "running",
        {
            "input": str(input_path),
            "output_dir": str(out_dir),
            "state_path": str(state_path),
            "manifest_path": str(manifest_path),
        },
    )

    conn = open_db(db_path)
    conn.execute("BEGIN")
    conn.commit()

    pending_rows_count = pending_count(pending_path)
    if pending_rows_count:
        log.info("Recovery: found %d pending rows, committing to next shard first.", pending_rows_count)
        state = write_shard(out_dir, metadata_dir, manifest_path, state_path, state, read_pending(pending_path))
        truncate_file(pending_path)
        pending_rows_count = 0
        update_stage_status(
            status_dir,
            STATUS_STAGE,
            "running",
            {"message": "Recovered pending rows into shard", "rows_written": state["rows_written"]},
        )

    kept_limit = args.limit if args.limit > 0 else None
    since_last_commit = 0

    with input_path.open(encoding="utf-8") as f:
        f.seek(int(state["last_input_offset"]))
        while True:
            line = f.readline()
            if not line:
                break
            offset_end = f.tell()
            line = line.strip()
            if not line:
                continue
            state["rows_scanned"] = int(state["rows_scanned"]) + 1

            raw = json.loads(line)
            year = raw.get("publication_year")
            if year is None or year < args.start_year or year > args.end_year:
                state["rows_dropped"] = int(state["rows_dropped"]) + 1
                continue

            oid = raw.get("openalex_id", "")
            if not oid:
                state["rows_dropped"] = int(state["rows_dropped"]) + 1
                continue

            if id_seen(conn, oid):
                state["rows_duplicate"] = int(state["rows_duplicate"]) + 1
                continue

            rec = cleaned_record(raw, min_abstract_chars=args.min_abstract_chars)
            if rec is None:
                state["rows_dropped"] = int(state["rows_dropped"]) + 1
                continue

            pending_row = {
                "input_offset_end": offset_end,
                "record": rec,
            }
            append_pending(pending_path, pending_row)
            mark_seen(conn, oid)
            since_last_commit += 1
            pending_rows_count += 1
            if since_last_commit >= 1_000:
                conn.commit()
                since_last_commit = 0

            state["rows_kept"] = int(state["rows_kept"]) + 1

            if kept_limit and int(state["rows_kept"]) >= kept_limit:
                log.info("Reached --limit=%d; stopping clean stage.", kept_limit)
                break

            if pending_rows_count >= args.shard_size:
                if since_last_commit:
                    conn.commit()
                    since_last_commit = 0
                state = write_shard(out_dir, metadata_dir, manifest_path, state_path, state, read_pending(pending_path))
                truncate_file(pending_path)
                pending_rows_count = 0
                update_stage_status(
                    status_dir,
                    STATUS_STAGE,
                    "running",
                    {
                        "rows_written": state["rows_written"],
                        "rows_scanned": state["rows_scanned"],
                        "rows_kept": state["rows_kept"],
                    },
                )

    if since_last_commit:
        conn.commit()
    if pending_rows_count:
        state = write_shard(out_dir, metadata_dir, manifest_path, state_path, state, read_pending(pending_path))
        truncate_file(pending_path)

    atomic_write_json(state_path, state)
    update_stage_status(
        status_dir,
        STATUS_STAGE,
        "completed",
        {
            "rows_written": state["rows_written"],
            "rows_scanned": state["rows_scanned"],
            "rows_kept": state["rows_kept"],
            "rows_dropped": state["rows_dropped"],
            "rows_duplicate": state["rows_duplicate"],
            "manifest_path": str(manifest_path),
        },
    )
    log.info(
        "Done. scanned=%s kept=%s dropped=%s duplicate=%s written=%s",
        state["rows_scanned"],
        state["rows_kept"],
        state["rows_dropped"],
        state["rows_duplicate"],
        state["rows_written"],
    )


if __name__ == "__main__":
    main()
