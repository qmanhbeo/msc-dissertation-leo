"""
Prepare training and test data for the single-label SDG classifier.

Loads the canon single-label embeddings from 2_data/2_embedded/, reads
the corresponding preprocessed JSONL, filters for single-label records
(multi-label texts are dropped at this boundary per design), builds
17D one-hot vectors, and performs a per-source stratified 85/15 split.

Sources: osdg, benchmark, sdg_knowledge_hub, sdgi_corpus, aurora
Excludes: policy (unlabeled), research_corpus (unlabeled)

Inputs:
   {embed_root}/{source}.npy
   2_data/1_preprocessed/{source_dir}/{file}.jsonl

Outputs (saved to {output_root}/):
  embeddings.npy       (N, dim) float32
  labels.npy           (N, 17) float32   — one-hot
  sources.npy          (N,) str
  indices/train.npy    int64
  indices/test.npy     int64
  split_report.txt

Run from project root (MiniLM default):
    python 1_code/2b_supervised_training_singlelabel/0_prepare_data.py
Run with MPNet:
    python 1_code/2b_supervised_training_singlelabel/0_prepare_data.py \
        --embed-root 2_data/2b_embedded_mpnet \
        --output-root 2_data/2b_supervised_singlelabel_mpnet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

N_SDG = 17
PREPROCESS_ROOT = Path("2_data/1_preprocessed")

CORPORA = [
    {
        "name": "osdg",
        "embed_file": "osdg.npy",
        "jsonl_path": PREPROCESS_ROOT / "osdg" / "osdg_clean.jsonl",
    },
    {
        "name": "benchmark",
        "embed_file": "benchmark.npy",
        "jsonl_path": PREPROCESS_ROOT / "sdg_benchmark" / "benchmark_clean.jsonl",
    },
    {
        "name": "sdg_knowledge_hub",
        "embed_file": "sdg_knowledge_hub.npy",
        "jsonl_path": PREPROCESS_ROOT / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.jsonl",
    },
    {
        "name": "sdgi",
        "embed_file": "sdgi.npy",
        "jsonl_path": PREPROCESS_ROOT / "sdgi_corpus" / "sdgi_clean.jsonl",
    },
    {
        "name": "aurora",
        "embed_file": "aurora.npy",
        "jsonl_path": PREPROCESS_ROOT / "aurora" / "aurora_texts.jsonl",
    },
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare single-label training data.")
    parser.add_argument("--embed-root", default="2_data/2_embedded",
                        help="Embedding root dir (default: 2_data/2_embedded)")
    parser.add_argument("--output-root", default="2_data/2b_supervised_singlelabel",
                        help="Output dir (default: 2_data/2b_supervised_singlelabel)")
    args = parser.parse_args()
    embed_root = Path(args.embed_root)
    output_dir = Path(args.output_root)
    log.info("Embed root: %s  Output: %s", embed_root, output_dir)

    all_embs, all_labels, all_sources = [], [], []

    for corpus in CORPORA:
        name = corpus["name"]
        emb_path = embed_root / corpus["embed_file"]
        jsonl_path = corpus["jsonl_path"]

        if not emb_path.exists() or not jsonl_path.exists():
            log.warning("Missing: %s or %s — skipping", emb_path, jsonl_path)
            continue

        embs = np.load(emb_path).astype(np.float32)
        rows = load_jsonl(jsonl_path)

        if len(embs) != len(rows):
            log.error(
                "Mismatch: %s embeddings (%d) vs JSONL (%d) — skipping",
                name, len(embs), len(rows),
            )
            continue

        kept_indices = []
        dropped_multi = 0

        for i, entry in enumerate(rows):
            sdg = entry.get("sdg")
            sdgs = entry.get("sdgs")
            is_single = (
                (sdg is not None and 1 <= sdg <= N_SDG)
                or (isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= N_SDG)
            )
            if not is_single:
                dropped_multi += 1
                continue

            kept_indices.append(i)

        kept_embs = embs[kept_indices]
        labels = np.zeros((len(kept_indices), N_SDG), dtype=np.float32)
        for j, i in enumerate(kept_indices):
            entry = rows[i]
            sdg = entry.get("sdg")
            sdgs = entry.get("sdgs")
            if sdg is not None and 1 <= sdg <= N_SDG:
                labels[j, sdg - 1] = 1.0
            elif isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= N_SDG:
                labels[j, sdgs[0] - 1] = 1.0

        all_embs.append(kept_embs)
        all_labels.append(labels)
        all_sources.extend([name] * len(kept_indices))

        log.info(
            "  %s: %d texts (dropped %d multi-label)",
            name, len(kept_indices), dropped_multi,
        )

    if not all_embs:
        log.error("No corpora loaded — nothing to do.")
        return

    embeddings = np.vstack(all_embs)
    labels = np.vstack(all_labels)
    sources = np.array(all_sources)

    log.info("Total: %d texts", len(embeddings))

    all_idx = np.arange(len(embeddings))
    train_pool_idx, test_idx = [], []

    for src in np.unique(sources):
        mask = sources == src
        src_idx = all_idx[mask]
        src_y = labels[mask]

        if len(src_idx) < 5:
            train_pool_idx.extend(src_idx.tolist())
            log.warning("  %s: only %d texts — kept entirely in train", src, len(src_idx))
            continue

        y_int = src_y.argmax(axis=1)

        s_train, s_test = train_test_split(
            src_idx, test_size=0.15, random_state=42, stratify=y_int,
        )
        train_pool_idx.extend(s_train.tolist())
        test_idx.extend(s_test.tolist())

        log.info(
            "  %s: %d train + %d test (%.1f%%)",
            src, len(s_train), len(s_test), 100 * len(s_test) / len(src_idx),
        )

    train_pool_idx = np.array(train_pool_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indices").mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "sources.npy", sources)
    np.save(output_dir / "indices" / "train.npy", train_pool_idx)
    np.save(output_dir / "indices" / "test.npy", test_idx)

    lines = ["=" * 70]
    lines.append("SPLIT REPORT — Per-source stratified 85/15")
    lines.append("=" * 70)
    lines.append(f"Total: {len(embeddings)} texts, {len(np.unique(sources))} sources\n")

    for name_split, name_idx in [("Train", train_pool_idx), ("Test", test_idx)]:
        lines.append(f"--- {name_split} ({len(name_idx)} texts) ---")
        for src in np.unique(sources):
            n = int((sources[name_idx] == src).sum())
            lines.append(f"  {src:20s}: {n}")
        lines.append("")

    lines.append("--- Per-SDG label counts (train | test) ---")
    for sdg in range(N_SDG):
        train_c = int(labels[train_pool_idx, sdg].sum())
        test_c = int(labels[test_idx, sdg].sum())
        total_c = train_c + test_c
        train_pct = train_c / total_c * 100 if total_c > 0 else 0
        lines.append(f"  SDG-{sdg+1:2d}: {train_c:5d} train ({train_pct:.1f}%) | {test_c:5d} test")

    lines.append("")
    lines.append(f"Total train: {len(train_pool_idx)}  test: {len(test_idx)}")

    report_path = output_dir / "split_report.txt"
    report_path.write_text("\n".join(lines))
    log.info("Saved split report → %s", report_path)

    print("\n".join(lines[-10:]))
    print(f"\nDone. Train: {len(train_pool_idx)}  Test: {len(test_idx)}")


if __name__ == "__main__":
    main()
