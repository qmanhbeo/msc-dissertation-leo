"""
Prepare training and test data for the single-label SDG classifier.

Loads the canon single-label embeddings from 2_data/2_embedded/, reads
the corresponding preprocessed JSONL, filters for single-label records
(multi-label texts are dropped at this boundary per design), builds
17D one-hot vectors, and performs a per-source stratified 85/15 split
with document-level grouping (all segments of a document stay together).

Sources: osdg, benchmark, sdg_knowledge_hub, sdgi_corpus, aurora
Excludes: policy (unlabeled), research_corpus (unlabeled)

Inputs:
   {embed_root}/{source}.npy
   2_data/1_preprocessed/{source_dir}/{file}.jsonl

Outputs (saved to {output_root}/):
  embeddings.npy       (N, dim) float32
  labels.npy           (N, 17) float32   — one-hot
  sources.npy          (N,) str
  source_docs.npy      (N,) str          — document-level grouping key
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
from collections import defaultdict
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
        "segmented": True,
        "segmented_path_template": "2_data/1_preprocessed/sdg_knowledge_hub/sdg_knowledge_hub_segmented_{model}.jsonl",
    },
    {
        "name": "sdgi",
        "embed_file": "sdgi.npy",
        "jsonl_path": PREPROCESS_ROOT / "sdgi_corpus" / "sdgi_clean.jsonl",
        "segmented": True,
        "segmented_path_template": "2_data/1_preprocessed/sdgi_corpus/sdgi_unified_{model}.jsonl",
    },
    {
        "name": "aurora",
        "embed_file": "aurora.npy",
        "jsonl_path": PREPROCESS_ROOT / "aurora" / "aurora_texts.jsonl",
        "segmented": True,
        "segmented_path_template": "2_data/1_preprocessed/aurora/aurora_segmented_{model}.jsonl",
    },
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _model_slug(model: str) -> str:
    return model.replace("/", "_").lower()


def _resolve_jsonl_path(corpus: dict, model_name: str) -> Path:
    if corpus.get("segmented"):
        return Path(corpus["segmented_path_template"].format(model=_model_slug(model_name)))
    return corpus["jsonl_path"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _group_by_source_doc(indices: np.ndarray, source_docs: np.ndarray, labels: np.ndarray) -> tuple[list[list[int]], list[int], list[int]]:
    doc_to_idxs: dict[str, list[int]] = defaultdict(list)
    doc_to_label: dict[str, int] = {}
    for i in indices:
        doc = source_docs[i]
        doc_to_idxs[doc].append(i)
        if doc not in doc_to_label:
            label = int(labels[i].argmax())
            doc_to_label[doc] = label
    doc_groups = list(doc_to_idxs.values())
    doc_labels = [doc_to_label[source_docs[group[0]]] for group in doc_groups]
    flat_idxs = [i for group in doc_groups for i in group]
    return doc_groups, doc_labels, flat_idxs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare single-label training data.")
    parser.add_argument("--embed-root", default="2_data/2_embedded",
                        help="Embedding root dir (default: 2_data/2_embedded)")
    parser.add_argument("--output-root", default="2_data/2b_supervised_singlelabel",
                        help="Output dir (default: 2_data/2b_supervised_singlelabel)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model (default: %(default)s)")
    args = parser.parse_args()
    embed_root = Path(args.embed_root)
    output_dir = Path(args.output_root)
    log.info("Embed root: %s  Output: %s  Model: %s", embed_root, output_dir, args.model)

    all_embs, all_labels, all_sources, all_source_docs = [], [], [], []

    for corpus in CORPORA:
        name = corpus["name"]
        emb_path = embed_root / corpus["embed_file"]
        jsonl_path = _resolve_jsonl_path(corpus, args.model)

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
        source_docs_corpus: list[str] = []
        for j, i in enumerate(kept_indices):
            entry = rows[i]
            sdg = entry.get("sdg")
            sdgs = entry.get("sdgs")
            if sdg is not None and 1 <= sdg <= N_SDG:
                labels[j, sdg - 1] = 1.0
            elif isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= N_SDG:
                labels[j, sdgs[0] - 1] = 1.0
            source_docs_corpus.append(entry.get("source_doc", f"{name}_{i}"))

        all_embs.append(kept_embs)
        all_labels.append(labels)
        all_sources.extend([name] * len(kept_indices))
        all_source_docs.extend(source_docs_corpus)

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
    source_docs = np.array(all_source_docs)

    log.info("Total: %d texts", len(embeddings))

    all_idx = np.arange(len(embeddings))
    train_pool_idx, test_idx = [], []

    for src in np.unique(sources):
        mask = sources == src
        src_idx = all_idx[mask]

        if len(src_idx) < 5:
            train_pool_idx.extend(src_idx.tolist())
            log.warning("  %s: only %d texts — kept entirely in train", src, len(src_idx))
            continue

        doc_groups, doc_labels, _ = _group_by_source_doc(src_idx, source_docs, labels)

        unique_doc_labels = list(set(doc_labels))
        if len(unique_doc_labels) < 2:
            train_pool_idx.extend([i for group in doc_groups for i in group])
            log.warning("  %s: only %d document-level SDG labels — kept entirely in train", src, len(unique_doc_labels))
            continue

        doc_group_indices = np.arange(len(doc_groups))
        doc_label_arr = np.array(doc_labels, dtype=np.int64)

        n_test_docs = max(1, int(len(doc_groups) * 0.15))
        try:
            train_doc_groups, test_doc_groups = train_test_split(
                doc_group_indices, test_size=n_test_docs / len(doc_groups),
                random_state=42, stratify=doc_label_arr,
            )
        except ValueError:
            log.warning("  %s: stratification failed — falling back to unstratified split", src)
            train_doc_groups, test_doc_groups = train_test_split(
                doc_group_indices, test_size=n_test_docs / len(doc_groups),
                random_state=42,
            )

        for gi in train_doc_groups:
            train_pool_idx.extend(doc_groups[gi])
        for gi in test_doc_groups:
            test_idx.extend(doc_groups[gi])

        n_train = sum(len(doc_groups[gi]) for gi in train_doc_groups)
        n_test = sum(len(doc_groups[gi]) for gi in test_doc_groups)
        log.info(
            "  %s: %d train (%d docs) + %d test (%d docs) (%.1f%%)",
            src, n_train, len(train_doc_groups), n_test, len(test_doc_groups),
            100 * n_test / (n_train + n_test),
        )

    train_pool_idx = np.array(train_pool_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indices").mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "sources.npy", sources)
    np.save(output_dir / "source_docs.npy", source_docs)
    np.save(output_dir / "indices" / "train.npy", train_pool_idx)
    np.save(output_dir / "indices" / "test.npy", test_idx)

    lines = ["=" * 70]
    lines.append("SPLIT REPORT — Per-source stratified 85/15 (document-grouped)")
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
