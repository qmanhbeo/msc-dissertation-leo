"""
Concatenate the four policy-source embedding files into policy.npy.

Reads policy_scrape.npy, policy_manual.npy, ungdc_sdg.npy, and sdgi.npy
(and their corresponding metadata sidecars) from the embed directory,
concatenates them in that order, and writes policy.npy + policy_ids.json.

This is a fast in-memory operation — no checkpointing needed.

Output:
    embed_dir/{model}/policy.npy
    embed_dir/{model}/metadata/policy_ids.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, embed_dir_for_model, segmented_dir_for_model
from shard_pipeline_utils import atomic_write_json, atomic_write_npy

POLICY_SOURCES = ["policy_scrape", "policy_manual", "ungdc_sdg", "sdgi"]

SOURCE_FAMILY_MAP = {
    "policy_scrape": "curated_ai_sdg",
    "policy_manual": "curated_ai_sdg",
    "ungdc_sdg": "ungdc_speeches",
    "sdgi": "sdgi_vnr_vlr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge policy-source embeddings into policy.npy."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing policy.npy and policy_ids.json.",
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL,
        help="Embed model (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_dir = embed_dir_for_model(args.embed_model)
    emb_path = embed_dir / "policy.npy"
    meta_dir = embed_dir / "metadata"
    ids_path = meta_dir / "policy_ids.json"

    if emb_path.exists() and not args.overwrite:
        print(f"Skipping — {emb_path} already exists")
        return

    embed_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    embs = []
    ids_meta = []
    total_rows = 0

    for source in POLICY_SOURCES:
        src_emb = embed_dir / f"{source}.npy"
        src_ids = meta_dir / f"{source}_ids.json"

        if not src_emb.exists():
            raise FileNotFoundError(f"Missing: {src_emb}")
        if not src_ids.exists():
            raise FileNotFoundError(f"Missing: {src_ids}")

        arr = np.load(src_emb).astype(np.float32)
        with src_ids.open() as f:
            meta = json.load(f)

        if len(arr) != len(meta):
            raise RuntimeError(
                f"{source}: {len(arr)} embeddings != {len(meta)} metadata rows"
            )

        family = SOURCE_FAMILY_MAP[source]
        for entry in meta:
            entry["source_family"] = family
            entry.pop("text", None)

        embs.append(arr)
        ids_meta.extend(meta)
        total_rows += len(arr)
        print(f"  {source}: {arr.shape} ({family})")

    merged = np.concatenate(embs, axis=0)
    dim = merged.shape[1]
    print(f"\nMerged shape: {merged.shape}")

    tmp_emb = emb_path.with_suffix(".npy.tmp")
    with tmp_emb.open("wb") as f:
        np.save(f, merged)
        f.flush()
    if not tmp_emb.exists():
        raise RuntimeError(f"Failed to write {tmp_emb}")
    tmp_emb.replace(emb_path)

    atomic_write_json(ids_path, ids_meta)

    # Alignment check: verify policy.jsonl IDs == policy_ids.json IDs (position by position)
    seg_root = segmented_dir_for_model(args.embed_model)
    policy_jsonl_path = seg_root / "policy.jsonl"
    if not policy_jsonl_path.exists():
        print(f"  WARNING: {policy_jsonl_path} not found — skipping alignment check")
    else:
        jsonl_ids = []
        with policy_jsonl_path.open() as f:
            for line in f:
                r = json.loads(line)
                jsonl_ids.append(r.get("segment_id", ""))
        if len(jsonl_ids) != len(ids_meta):
            raise RuntimeError(
                f"Alignment mismatch: policy.jsonl has {len(jsonl_ids)} rows, "
                f"policy_ids.json has {len(ids_meta)} rows"
            )
        mismatches = [(i, jid, mid) for i, (jid, mid) in
                      enumerate(zip(jsonl_ids, [e["id"] for e in ids_meta])) if jid != mid]
        if mismatches:
            raise RuntimeError(
                f"ID alignment mismatch at {len(mismatches)} positions "
                f"(first: row {mismatches[0][0]}, jsonl={mismatches[0][1]!r}, "
                f"ids={mismatches[0][2]!r})"
            )
        print(f"  ✓ Alignment check: {len(jsonl_ids)} IDs match policy.jsonl")

    print(f"→ {emb_path}  ({total_rows} rows, {dim} dim)")
    print(f"→ {ids_path}")


if __name__ == "__main__":
    main()
