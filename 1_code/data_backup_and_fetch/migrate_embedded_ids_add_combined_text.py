"""
One-time migration: add combined_text to embedded research shard IDs.

Reads combined_text from 2_segmented/{model}/research/part-*.jsonl and adds it
to the corresponding 3_embedded/{model}/research_shards/metadata/part-*_ids.jsonl.

Row alignment between segmented and embedded files is verified by openalex_id.

Run from project root:
    python 1_code/data_backup_and_fetch/migrate_embedded_ids_add_combined_text.py
    python 1_code/data_backup_and_fetch/migrate_embedded_ids_add_combined_text.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, segmented_dir_for_model, embed_research_dir_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add combined_text to embedded research shard IDs.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embed model (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    return p.parse_args()


def migrate_shard(seg_path: Path, ids_path: Path, dry_run: bool) -> int:
    """Add combined_text from segmented JSONL to embedded IDs JSONL. Returns count migrated."""
    updated = 0
    tmp = ids_path.with_suffix(".jsonl.tmp")

    with seg_path.open(encoding="utf-8") as seg_f, \
         ids_path.open(encoding="utf-8") as ids_f, \
         tmp.open("w", encoding="utf-8") as out_f:

        for line_no, (seg_line, ids_line) in enumerate(zip(seg_f, ids_f), start=1):
            seg_row = json.loads(seg_line)
            ids_row = json.loads(ids_line)

            if seg_row.get("openalex_id") != ids_row.get("openalex_id"):
                raise RuntimeError(
                    f"Row {line_no} alignment mismatch in {ids_path.name}: "
                    f"seg={seg_row.get('openalex_id')} vs ids={ids_row.get('openalex_id')}"
                )

            if "combined_text" in ids_row:
                out_f.write(ids_line)
                continue

            ids_row["combined_text"] = seg_row.get("combined_text", "")
            out_f.write(json.dumps(ids_row, ensure_ascii=False) + "\n")
            updated += 1

    if not dry_run and updated > 0:
        tmp.replace(ids_path)

    tmp.unlink(missing_ok=True)
    return updated


def main() -> None:
    args = parse_args()
    seg_root = segmented_dir_for_model(args.embed_model) / "research"
    ids_root = embed_research_dir_for_model(args.embed_model) / "metadata"

    seg_manifest_path = seg_root / "metadata" / "manifest.json"
    if not seg_manifest_path.exists():
        log.error("Segmented manifest not found: %s", seg_manifest_path)
        sys.exit(1)

    seg_manifest = json.loads(seg_manifest_path.read_text(encoding="utf-8"))
    shards = sorted(seg_manifest["shards"], key=lambda x: int(x["shard_id"]))

    log.info("Model: %s", args.embed_model)
    log.info("Segmented dir: %s", seg_root)
    log.info("Embedded IDs dir: %s", ids_root)
    log.info("Shards to process: %d", len(shards))
    if args.dry_run:
        log.info("DRY RUN — no files will be modified")

    total_updated = 0
    for shard in shards:
        name = shard["name"]
        seg_path = seg_root / f"{name}.jsonl"
        ids_path = ids_root / f"{name}_ids.jsonl"

        if not seg_path.exists():
            log.warning("Missing segmented shard: %s", seg_path)
            continue
        if not ids_path.exists():
            log.warning("Missing embedded IDs: %s", ids_path)
            continue

        n = migrate_shard(seg_path, ids_path, dry_run=args.dry_run)
        total_updated += n
        if n > 0:
            log.info("  %s: %d rows updated", name, n)

    log.info("Done. Total rows updated: %d", total_updated)


if __name__ == "__main__":
    main()
