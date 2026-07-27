"""
Cross-dedup policy source segments before embedding.

Reads the four policy-relevant segmented files (sdgi, policy_scrape,
policy_manual, ungdc_sdg), identifies exact-text duplicates across
sources, and writes deduped versions to the segmented dir.

SDGi is kept first in the dedup order so it retains all rows (it is
shared with the training set). Only the three policy-specific sources
are rewritten — SDGi's file on disk is unchanged.

Output (per model in 2_segmented/{model}/):
    policy_scrape.jsonl    (deduped)
    policy_manual.jsonl    (deduped)
    ungdc_sdg.jsonl        (deduped)
    policy.jsonl           (merged in 1_merge_policy_corpus.py order, for scoring)

The caller then embeds each source file separately via
0_embed_reference_and_policy_corpora.py --corpus <name>.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import preprocessed_dir, segmented_dir_for_model, DEFAULT_EMBED_MODEL

MIN_WORD_COUNT = 20

# Must match 1_merge_policy_corpus.py POLICY_SOURCES
MERGED_ORDER = ["policy_scrape", "policy_manual", "ungdc_sdg", "sdgi"]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dedup policy sources pre-embedding."
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL,
        help="Embed model name (default: %(default)s)",
    )
    args = parser.parse_args()

    # Source order — sdgi first so it keeps all rows
    SOURCES = [
        ("sdgi", segmented_dir_for_model(args.embed_model) / "sdgi.jsonl", True),
        ("policy_scrape", segmented_dir_for_model(args.embed_model) / "policy_scrape.jsonl", False),
        ("policy_manual", segmented_dir_for_model(args.embed_model) / "policy_manual.jsonl", False),
        ("ungdc_sdg", segmented_dir_for_model(args.embed_model) / "ungdc_sdg.jsonl", False),
    ]

    output_dir = segmented_dir_for_model(args.embed_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter short segments
    source_segments: dict[str, list[dict]] = {}
    source_stats: dict[str, tuple[int, int]] = {}

    for name, path, _ in SOURCES:
        raw = load_jsonl(path)
        kept = []
        skipped_short = 0
        for s in raw:
            text = s.get("text", "").strip()
            if not text:
                continue
            wc = s.get("word_count", len(text.split()))
            if wc < MIN_WORD_COUNT:
                skipped_short += 1
                continue
            kept.append(s)
        source_segments[name] = kept
        source_stats[name] = (len(raw), len(kept), skipped_short)
        print(f"  {name}: {len(raw)} raw → {len(kept)} kept ({skipped_short} too short)")

    # Cross-source exact-text dedup
    # First source (sdgi) seeds the seen_texts set — no rows removed.
    # Subsequent sources lose text already seen in any earlier source.
    seen_texts: set[str] = set()
    deduped_segments: dict[str, list[dict]] = {}

    for i, (name, _, _) in enumerate(SOURCES):
        before = len(source_segments[name])
        kept = []
        for s in source_segments[name]:
            text_key = s.get("text", "").strip()
            if text_key in seen_texts and i > 0:
                continue
            seen_texts.add(text_key)
            kept.append(s)
        removed = before - len(kept)
        deduped_segments[name] = kept
        if removed:
            print(f"  {name}: {removed} cross-source duplicates removed ({before} → {len(kept)})")

    total_before = sum(len(source_segments[n]) for n, _, _ in SOURCES)
    total_after = sum(len(deduped_segments[n]) for n, _, _ in SOURCES)
    total_removed = total_before - total_after
    print(f"\nTotal: {total_before} → {total_after} ({total_removed} removed, {100*total_removed/total_before:.1f}%)")

    # Write deduped files (only the three policy-specific sources)
    for name, _, rewrite in SOURCES:
        if not rewrite:
            out_path = output_dir / f"{name}.jsonl"
            write_jsonl(deduped_segments[name], out_path)
            print(f"  → wrote {out_path} ({len(deduped_segments[name])} rows)")

    # Write merged policy.jsonl (order matches 1_merge_policy_corpus.py)
    merged_records = []
    for name in MERGED_ORDER:
        merged_records.extend(deduped_segments[name])
    merged_path = output_dir / "policy.jsonl"
    write_jsonl(merged_records, merged_path)
    print(f"  → wrote merged {merged_path} ({len(merged_records)} rows)")

    print("\nSource breakdown (post-dedup, policy-only):")
    total_pct = sum(len(deduped_segments[n]) for n, _, _ in SOURCES if n != "sdgi")
    for name, _, _ in SOURCES:
        if name != "sdgi":
            count = len(deduped_segments[name])
            pct = 100.0 * count / max(total_pct, 1)
            print(f"  {name}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
