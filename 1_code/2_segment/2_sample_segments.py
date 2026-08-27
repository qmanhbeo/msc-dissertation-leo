"""
Build the shared 100k representative research subset from the CANONICAL segmented
research corpus, for every non-primary (sensitivity) encoder to embed.

The canonical research corpus is segmented ONCE (at CANONICAL_MAX_SEQ_LENGTH,
all-mpnet-base-v2) and every encoder — including MiniLM and SciBERT — embeds the
SAME canonical segments, so the only varying factor in the architecture
comparison is the encoder itself. The 100k subset is a deterministic seed-42
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
    RESEARCH_SUBSET_PAPERS,
    RESEARCH_SUBSET_SEED,
    research_subset_dir,
    research_subset_manifest,
    segmented_dir_for_model,
)
from research_score_shards import assert_papers_contiguous, paper_run_starts

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
    """Build the shared research subset for sensitivity encoders to embed.

    The subset is a deterministic seed-42 uniform draw of **papers** (abstracts)
    from the canonical segmented research corpus; EVERY segment of each drawn
    paper is included, so the subset preserves the per-paper segment structure
    (≈1.22 segments/abstract). Drawn papers are recorded as global paper
    ordinals (first-seen order over the canonical shards), so the same 100k
    papers are reproducible from the seed alone.

    Returns the path to the shared subset input manifest.
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
    paper_index_path = subset_meta / "paper_index.jsonl"

    canonical_data = _json.loads(canonical_manifest.read_text(encoding="utf-8"))
    shards = sorted(canonical_data["shards"], key=lambda x: int(x["shard_id"]))

    # ---- Pass 1: ordered unique paper ids (cached, resume-safe) ----
    n_papers = 0
    if paper_index_path.exists() and not overwrite:
        with paper_index_path.open(encoding="utf-8") as f:
            n_papers = sum(1 for _ in f)
        log.info("Reusing cached paper index (%d papers): %s", n_papers, paper_index_path)
    else:
        log.info("Building ordered paper index from %d canonical shards...", len(shards))
        prev_id: str | None = None
        tmp_index = paper_index_path.with_suffix(".jsonl.tmp")
        n_papers = 0
        with tmp_index.open("w", encoding="utf-8") as out:
            for shard in shards:
                in_path = canonical_seg / f"{shard['name']}.jsonl"
                if not in_path.exists():
                    raise FileNotFoundError(f"Canonical research shard missing: {in_path}")
                with in_path.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        pid = _json.loads(line).get("openalex_id")
                        if not pid:
                            raise RuntimeError(f"Row in {in_path} has no openalex_id")
                        if pid != prev_id:
                            out.write(f"{pid}\n")
                            n_papers += 1
                            prev_id = pid
        tmp_index.replace(paper_index_path)
        log.info("Built paper index: %d papers", n_papers)

    if n_papers < RESEARCH_SUBSET_PAPERS:
        raise RuntimeError(
            f"Canonical research has only {n_papers} papers; need {RESEARCH_SUBSET_PAPERS}."
        )

    # ---- Draw papers (global ordinals) ----
    rng = np.random.default_rng(RESEARCH_SUBSET_SEED)
    drawn = np.sort(
        rng.choice(n_papers, size=RESEARCH_SUBSET_PAPERS, replace=False).astype(np.int64)
    )
    drawnset = set(int(x) for x in drawn)
    log.info("Selected %d papers (seed=%d) for shared subset", len(drawn), RESEARCH_SUBSET_SEED)

    if (
        subset_jsonl.exists()
        and subset_manifest.exists()
        and not overwrite
    ):
        try:
            _meta = _json.loads(subset_manifest.read_text(encoding="utf-8"))
            # Reuse is validated ONLY against the recorded manifest
            # (self-consistency + file line count) — it does NOT re-check
            # that the canonical segmented corpus still matches. After a
            # re-segmentation, delete the subset dir or pass --overwrite.
            # NB: the chained comparison means rows == n_segments AND
            # n_segments != -1.
            if _meta.get("sample_method") == "uniform_paper_ordinals" and int(
                _meta.get("totals", {}).get("rows", -1)
            ) == int(_meta.get("n_segments", -1)) != -1:
                # Verify the cached subset actually contains the drawn papers.
                _have = sum(1 for _ in subset_jsonl.open(encoding="utf-8"))
                if _have == int(_meta.get("n_segments", -1)):
                    log.info("Shared paper-based subset already built (%d segments); reusing.", _have)
                    return subset_manifest
            log.warning("Subset present but stale; rebuilding")
        except Exception as exc:
            log.warning("Subset present but unverifiable (%s); rebuilding", exc)

    # ---- Pass 2: copy every segment of each drawn paper ----
    run_cursor = 0
    written = 0
    tmp_jsonl = subset_jsonl.with_suffix(".jsonl.tmp")
    with tmp_jsonl.open("w", encoding="utf-8") as out:
        for shard in shards:
            in_path = canonical_seg / f"{shard['name']}.jsonl"
            if not in_path.exists():
                raise FileNotFoundError(f"Canonical research shard missing: {in_path}")
            # Read line-by-line (NOT read_text().splitlines()) so control
            # characters inside a JSON string field do not fragment a record.
            paper_ids_run: list[str] = []
            line_buf: list[str] = []
            with in_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    pid = _json.loads(line).get("openalex_id")
                    paper_ids_run.append(pid)
                    line_buf.append(line if line.endswith("\n") else line + "\n")
            starts = paper_run_starts(paper_ids_run)
            assert_papers_contiguous(paper_ids_run, starts, shard["name"])
            seg_counts = np.diff(np.append(starts, len(line_buf))).astype(np.int64)
            for j, start in enumerate(starts):
                ordinal = run_cursor + j
                if ordinal in drawnset:
                    stop = start + int(seg_counts[j])
                    for k in range(start, stop):
                        out.write(line_buf[k])
                        written += 1
            run_cursor += len(starts)
    # Atomic publish deliberately precedes the invariant checks below: on a
    # failed check the manifest is never written, so the reuse gate above
    # (line count vs manifest n_segments) rejects the bad subset on the next
    # run. Keep this order — the gate is the backstop.
    tmp_jsonl.replace(subset_jsonl)  # atomic publish

    if run_cursor != n_papers:
        raise RuntimeError(f"Run cursor {run_cursor} != paper count {n_papers}")
    if written == 0:
        raise RuntimeError("Subset build produced 0 segments")

    manifest_data = {
        "stage": "research_subset_for_encoder_sensitivity",
        "schema_version": 1,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": CANONICAL_SEGMENT_MODEL,
        "source_segment_model": CANONICAL_SEGMENT_MODEL,
        "sample_seed": RESEARCH_SUBSET_SEED,
        "sample_method": "uniform_paper_ordinals",
        "n_papers": int(len(drawn)),
        "n_segments": int(written),
        "sample_size": int(len(drawn)),
        "drawn_ordinals": [int(x) for x in drawn],
        "shards": [{"shard_id": 1, "name": "part-00001", "rows": written}],
        "totals": {"rows": written, "shards": 1, "papers": int(len(drawn))},
    }
    tmp = subset_manifest.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(manifest_data, indent=2))
    tmp.replace(subset_manifest)
    log.info(
        "Built shared paper-based research subset: %s (%d papers, %d segments)",
        subset_manifest,
        len(drawn),
        written,
    )
    return subset_manifest


def main() -> None:
    args = parse_args()
    build_canonical_research_subset(args.overwrite)


if __name__ == "__main__":
    main()
