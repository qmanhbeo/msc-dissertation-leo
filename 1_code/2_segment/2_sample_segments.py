"""
Build the shared 50k representative research subset from the CANONICAL segmented
research corpus, for every non-primary (sensitivity) encoder to embed.

The canonical research corpus is segmented ONCE (at CANONICAL_MAX_SEQ_LENGTH,
all-mpnet-base-v2) and every encoder — including MiniLM and SciBERT — embeds the
SAME canonical segments, so the only varying factor in the architecture
comparison is the encoder itself. The 50k subset is a deterministic seed-42
uniform sample of global segment indices over the canonical corpus; it is
decoupled from the appendix sample-stability stage (no forward dependency) and
is consumed by MiniLM and SciBERT instead of the full 3.1M-row corpus.

From project root:
    python 1_code/2_segment/2_sample_segments.py [--overwrite]
"""

from __future__ import annotations

import argparse
import datetime
import json as _json
import logging
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import numpy as np
from model_utils import (
    CANONICAL_SEGMENT_MODEL,
    RESEARCH_SUBSET_SEED,
    RESEARCH_SUBSET_SIZE,
    research_subset_dir,
    research_subset_manifest,
    segmented_dir_for_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Force rebuild even if subset already exists.",
    )
    return parser.parse_args()


def build_canonical_research_subset(overwrite: bool) -> Path:
    """Build the shared 50k representative research subset (a deterministic
    seed-42 uniform sample of global segment indices over the canonical
    research corpus) for sensitivity encoders to embed. Returns the path to the
    shared subset input manifest.

    The canonical corpus is segmented once for CANONICAL_SEGMENT_MODEL; this
    script simply samples from those already-produced segments, so it has no
    dependency on the appendix sample-stability stage.
    """
    canonical_seg = segmented_dir_for_model(CANONICAL_SEGMENT_MODEL) / "research"
    canonical_manifest = canonical_seg / "metadata" / "manifest.json"
    if not canonical_manifest.exists():
        raise FileNotFoundError(
            f"Canonical research segment manifest missing: {canonical_manifest}"
        )

    subset_dir = research_subset_dir()
    subset_meta = subset_dir / "metadata"
    subset_meta.mkdir(parents=True, exist_ok=True)
    subset_jsonl = subset_dir / "part-00001.jsonl"
    subset_manifest = research_subset_manifest()

    canonical_data = _json.loads(canonical_manifest.read_text(encoding="utf-8"))
    total = int(canonical_data.get("totals", {}).get("rows", 0))
    if total < RESEARCH_SUBSET_SIZE:
        raise RuntimeError(
            f"Canonical research has only {total} rows; need {RESEARCH_SUBSET_SIZE}."
        )
    rng = np.random.default_rng(RESEARCH_SUBSET_SEED)
    indices = np.sort(
        rng.choice(total, size=RESEARCH_SUBSET_SIZE, replace=False).astype(np.int64)
    )
    log.info(
        "Selected %d research rows (seed=%d) for shared 50k subset",
        len(indices),
        RESEARCH_SUBSET_SEED,
    )

    if subset_jsonl.exists() and subset_manifest.exists() and not overwrite:
        # Resume-safe: reuse a previously built subset instead of re-scanning the
        # canonical research corpus. Verify row count matches the manifest so a
        # partial/interrupted build (jsonl present, manifest missing or short)
        # is detected and rebuilt.
        try:
            _meta = _json.loads(subset_manifest.read_text(encoding="utf-8"))
            _expect = int(_meta.get("totals", {}).get("rows", -1))
            _have = sum(1 for _ in subset_jsonl.open(encoding="utf-8"))
            if _have == _expect:
                log.info("Shared 50k subset already built (%d rows); reusing.", _have)
                return subset_manifest
            log.warning("Subset present but row count %d != %d; rebuilding", _have, _expect)
        except Exception as exc:
            log.warning("Subset present but unverifiable (%s); rebuilding", exc)

    shards = sorted(canonical_data["shards"], key=lambda x: int(x["shard_id"]))
    need = set(int(x) for x in indices)
    offset = 0
    written = 0
    tmp_jsonl = subset_jsonl.with_suffix(".jsonl.tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as out:
        for shard in shards:
            name = shard["name"]
            rows = int(shard["rows"])
            in_path = canonical_seg / f"{name}.jsonl"
            if not in_path.exists():
                raise FileNotFoundError(f"Canonical research shard missing: {in_path}")
            with in_path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    g = offset + i
                    if g in need:
                        out.write(line if line.endswith("\n") else line + "\n")
                        written += 1
            offset += rows
    tmp_jsonl.replace(subset_jsonl)  # atomic publish

    if written != len(indices):
        raise RuntimeError(
            f"Subset row mismatch: expected {len(indices)}, wrote {written}"
        )

    manifest_data = {
        "stage": "research_subset_for_encoder_sensitivity",
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": CANONICAL_SEGMENT_MODEL,
        "source_segment_model": CANONICAL_SEGMENT_MODEL,
        "sample_seed": RESEARCH_SUBSET_SEED,
        "sample_size": written,
        "sample_method": "uniform_global_segment_indices",
        "shards": [{"shard_id": 1, "name": "part-00001", "rows": written}],
        "totals": {"rows": written, "shards": 1},
    }
    tmp = subset_manifest.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(manifest_data, indent=2))
    tmp.replace(subset_manifest)
    log.info("Built shared research subset manifest: %s (%d rows)", subset_manifest, written)
    return subset_manifest


def main() -> None:
    args = parse_args()
    build_canonical_research_subset(args.overwrite)


if __name__ == "__main__":
    main()
