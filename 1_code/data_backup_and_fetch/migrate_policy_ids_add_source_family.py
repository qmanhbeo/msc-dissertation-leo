"""
One-time migration: add source_family to embedded policy metadata.

Reads source_doc -> family mapping from 2_segmented/{model}/*.jsonl and adds
source_family to 3_embedded/{model}/metadata/policy_ids.json.

Run from project root:
    python 1_code/data_backup_and_fetch/migrate_policy_ids_add_source_family.py
    python 1_code/data_backup_and_fetch/migrate_policy_ids_add_source_family.py --model all-MiniLM-L6-v2
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

from model_utils import DEFAULT_EMBED_MODEL, segmented_dir_for_model, embed_dir_for_model, resolve_model_alias

SOURCE_FAMILY_MAP = {
    "policy_scrape": "curated_ai_sdg",
    "policy_manual": "curated_ai_sdg",
    "ungdc_sdg": "ungdc_speeches",
    "sdgi": "sdgi_vnr_vlr",
}

POLICY_SOURCES = ["policy_scrape", "policy_manual", "ungdc_sdg", "sdgi"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add source_family to embedded policy metadata.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help="Embed model (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seg_root = segmented_dir_for_model(args.embed_model)
    ids_path = embed_dir_for_model(args.embed_model) / "metadata" / "policy_ids.json"

    if not ids_path.exists():
        log.error("Missing: %s", ids_path)
        sys.exit(1)

    # Build source_doc -> family mapping from segmented JSONL files
    source_family: dict[str, str] = {}
    for source in POLICY_SOURCES:
        family = SOURCE_FAMILY_MAP[source]
        jsonl_path = seg_root / f"{source}.jsonl"
        if not jsonl_path.exists():
            log.warning("Missing segmented file: %s — skipping", jsonl_path)
            continue
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc = str(row["source_doc"])
                existing = source_family.get(doc)
                if existing is not None and existing != family:
                    raise RuntimeError(
                        f"source_doc '{doc}' appears in multiple families: {existing} vs {family}"
                    )
                source_family[doc] = family

    if not source_family:
        log.error("No source-family assignments built from segmented files.")
        sys.exit(1)

    log.info("Source family map: %d documents", len(source_family))

    # Patch policy_ids.json
    with ids_path.open(encoding="utf-8") as f:
        policy_ids = json.load(f)

    updated = 0
    missing = 0
    for entry in policy_ids:
        if "source_family" in entry:
            continue
        doc = entry.get("source_doc")
        family = source_family.get(doc)
        if family is None:
            missing += 1
            continue
        entry["source_family"] = family
        updated += 1

    log.info("Would update: %d, already has field: %d, missing source_doc: %d",
             updated, len(policy_ids) - updated - missing, missing)

    if args.dry_run:
        log.info("DRY RUN — no files modified")
        return

    if updated > 0:
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(policy_ids, f, ensure_ascii=False, indent=2)
            f.write("\n")
        log.info("Wrote %s (%d entries, %d with source_family)", ids_path, len(policy_ids), updated)
    else:
        log.info("No changes needed")


if __name__ == "__main__":
    main()
