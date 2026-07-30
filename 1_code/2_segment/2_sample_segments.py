"""
Build a 50k representative research subset from MPNet's segmented data for SciBERT.

This script reads MPNet's segmented research corpus and filters to a pre-computed
50k random sample (seed-42 draw), producing a subset manifest and JSONL for SciBERT
to embed. The identical texts ensure the encoder comparison isolates architecture
and domain alone.

From project root:
    python 1_code/2_segment/2_sample_segments.py \
        --slug allenai/scibert_scivocab_uncased \
        [--overwrite]
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
from model_utils import segmented_dir_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--slug",
        default="allenai/scibert_scivocab_uncased",
        help="Target model slug (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Force rebuild even if subset already exists.",
    )
    return parser.parse_args()


def sample_scibert_research_subset(slug: str, overwrite: bool) -> Path:
    """Build a 50k representative research subset (the *identical* MPNet-segmented
    texts) for SciBERT to embed, so the encoder comparison isolates architecture
    and domain alone. Returns the path to the subset input manifest.

    The 50k indices are the sample-stability draw (tier "50k", first draw). Global
    indices map onto the research embed manifest sorted by shard_id (cumulative
    rows), exactly as c_sample_stability.py and research_embedding_shards.py do.
    """
    mpnet_seg = segmented_dir_for_model("all-mpnet-base-v2") / "research"
    mpnet_manifest = mpnet_seg / "metadata" / "manifest.json"
    if not mpnet_manifest.exists():
        raise FileNotFoundError(f"MPNet research segment manifest missing: {mpnet_manifest}")

    base = segmented_dir_for_model(slug)
    subset_dir = base / "research_subset"
    subset_meta = subset_dir / "metadata"
    subset_meta.mkdir(parents=True, exist_ok=True)
    subset_jsonl = subset_dir / "part-00001.jsonl"
    subset_manifest = subset_meta / "manifest.json"

    sample_root = REPO_ROOT / "2_data" / "5_supervised_scored" / "all-mpnet-base-v2"
    candidates = sorted(sample_root.glob("paper_sample_seed_*/50k/draw_01_indices.npy"))
    if not candidates:
        raise FileNotFoundError(
            "No 50k sample draw found under 2_data/5_supervised_scored/all-mpnet-base-v2/"
        )
    draw_path = candidates[0]
    indices = np.sort(np.unique(np.load(draw_path).astype(np.int64)))
    log.info("Selected %d research rows for SciBERT subset (draw=%s)", len(indices), draw_path.name)

    if subset_jsonl.exists() and subset_manifest.exists() and not overwrite:
        # Resume-safe: reuse a previously built subset instead of re-scanning the
        # 3.1M-segment research corpus. Verify row count matches the manifest so a
        # partial/interrupted build (jsonl present, manifest missing or short) is
        # detected and rebuilt.
        try:
            _meta = _json.loads(subset_manifest.read_text(encoding="utf-8"))
            _expect = int(_meta.get("totals", {}).get("rows", -1))
            _have = sum(1 for _ in subset_jsonl.open(encoding="utf-8"))
            if _have == _expect:
                log.info("SciBERT research subset already built (%d rows); reusing.", _have)
                return subset_manifest
            log.warning("Subset present but row count %d != %d; rebuilding", _have, _expect)
        except Exception as exc:
            log.warning("Subset present but unverifiable (%s); rebuilding", exc)

    mpnet_data = _json.loads(mpnet_manifest.read_text(encoding="utf-8"))
    shards = sorted(mpnet_data["shards"], key=lambda x: int(x["shard_id"]))
    need = set(int(x) for x in indices)
    offset = 0
    written = 0
    tmp_jsonl = subset_jsonl.with_suffix(".jsonl.tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as out:
        for shard in shards:
            name = shard["name"]
            rows = int(shard["rows"])
            in_path = mpnet_seg / f"{name}.jsonl"
            if not in_path.exists():
                raise FileNotFoundError(f"MPNet research shard missing: {in_path}")
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
        "model": slug,
        "source_segment_model": "all-mpnet-base-v2",
        "draw": str(draw_path),
        "n_selected": written,
        "shards": [{"shard_id": 1, "name": "part-00001", "rows": written}],
        "totals": {"rows": written, "shards": 1},
    }
    tmp = subset_manifest.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(manifest_data, indent=2))
    tmp.replace(subset_manifest)
    log.info("Built SciBERT research subset manifest: %s (%d rows)", subset_manifest, written)
    return subset_manifest


def main() -> None:
    args = parse_args()
    sample_scibert_research_subset(args.slug, args.overwrite)


if __name__ == "__main__":
    main()
