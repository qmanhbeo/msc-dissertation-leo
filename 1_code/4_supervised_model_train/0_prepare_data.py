"""
Prepare training and test data for the single-label SDG classifier.

Loads consolidated reference embeddings and metadata from
3_embedded/{model}/reference.npy + reference_ids.json, groups records
by their source field (osdg, benchmark, sdg_knowledge_hub, sdgi, aurora),
filters for single-label records, builds 17D one-hot vectors, and performs
a per-source stratified 85/15 split with document-level grouping.

Excludes: policy (unlabeled), research_corpus (unlabeled)

Outputs (saved to {output_root}/):
  embeddings.npy       (N, dim) float32
  labels.npy           (N, 17) float32   — one-hot
  sources.npy          (N,) str
  source_docs.npy      (N,) str          — document-level grouping key
  indices/train.npy    int64
  indices/test.npy     int64
  split_report.txt

Run from project root:
    python 1_code/4_supervised_model_train/0_prepare_data.py --model all-mpnet-base-v2
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, RANDOM_SEED, embed_dir_for_model, model_results_dir_for_model, resolve_model_alias

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_ids(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def _group_by_source_doc(indices: np.ndarray, source_docs: np.ndarray, labels: np.ndarray) -> tuple[list[list[int]], list[int], list[int]]:
    doc_to_idxs: dict[str, list[int]] = defaultdict(list)
    doc_to_label: dict[str, int] = {}
    for i in indices:
        doc = source_docs[i]
        doc_to_idxs[doc].append(i)
        # Group label = FIRST row seen for this source_doc. Assumes all rows
        # of one document share a single SDG label; mixed-label documents are
        # silently labelled by whichever row arrives first.
        if doc not in doc_to_label:
            label = int(labels[i].argmax())
            doc_to_label[doc] = label
    doc_groups = list(doc_to_idxs.values())
    doc_labels = [doc_to_label[source_docs[group[0]]] for group in doc_groups]
    flat_idxs = [i for group in doc_groups for i in group]
    return doc_groups, doc_labels, flat_idxs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare single-label training data.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                        help="Embed model (default: %(default)s)")
    parser.add_argument("--embed-root", default=None,
                        help="Override embed root dir (derived from --model if omitted)")
    parser.add_argument("--output-root", default=None,
                        help="Override output root dir (derived from --embed-model if omitted)")
    parser.add_argument("--train-frac", type=float, default=0.85,
                        help="Fraction of document-groups assigned to the train pool; the rest go to test (default: %(default)s)")
    parser.add_argument("--split-seed", type=int, default=RANDOM_SEED,
                        help="Random seed for the stratified document-group train/test split (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()
    embed_root = Path(args.embed_root) if args.embed_root else embed_dir_for_model(args.embed_model)
    output_dir = Path(args.output_root) if args.output_root else model_results_dir_for_model(args.embed_model)

    # ── Existence-skip (cheap stage convention) ──
    required_outputs = [
        output_dir / "embeddings.npy",
        output_dir / "labels.npy",
        output_dir / "sources.npy",
        output_dir / "source_docs.npy",
        output_dir / "indices" / "train.npy",
        output_dir / "indices" / "test.npy",
        output_dir / "split_report.txt",
    ]
    if not args.overwrite and all(p.exists() for p in required_outputs):
        log.info("Skipping — outputs present at %s (use --overwrite to regenerate)", output_dir)
        return

    log.info("Embed root: %s  Output: %s  Model: %s", embed_root, output_dir, args.embed_model)

    emb_path = embed_root / "reference.npy"
    ids_path = embed_root / "metadata" / "reference_ids.json"

    if not emb_path.exists() or not ids_path.exists():
        log.error("Missing: %s or %s", emb_path, ids_path)
        # Fail closed: a bare return would let main.py's run_step treat this
        # stage as successful and proceed on stale downstream artifacts.
        raise SystemExit(1)

    all_embs = np.load(emb_path).astype(np.float32)
    all_rows = load_ids(ids_path)

    if len(all_embs) != len(all_rows):
        log.error(
            "Mismatch: reference.npy (%d) vs reference_ids.json (%d)",
            len(all_embs), len(all_rows),
        )
        # Fail closed: see note above — silent success would corrupt the
        # supervised-data contract downstream.
        raise SystemExit(1)

    # Group by source field
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(all_rows):
        src = entry.get("source", "unknown")
        source_to_indices[src].append(i)

    ordered_sources = ["osdg", "benchmark", "sdg_knowledge_hub", "sdgi", "aurora"]

    # WHITELIST: only these five sources enter the supervised pool. Rows with
    # an unlisted or missing "source" (→ "unknown") are SILENTLY dropped —
    # the warning below fires only for expected-but-absent sources, never for
    # present-but-unexpected ones. A new upstream source must be added here.
    embs_list, labels_list, sources_list, source_docs_list = [], [], [], []
    source_summary: dict[str, int] = {}  # name -> kept text count

    for name in ordered_sources:
        indices = source_to_indices.get(name, [])
        if not indices:
            log.warning("  %s: no records found — skipping", name)
            continue

        src_embs = all_embs[indices]
        src_rows = [all_rows[i] for i in indices]

        kept_sub_indices = []
        dropped_multi = 0

        for sub_i, entry in enumerate(src_rows):
            sdg = entry.get("sdg")
            sdgs = entry.get("sdgs")
            is_single = (
                (sdg is not None and 1 <= sdg <= N_SDG)
                or (isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= N_SDG)
            )
            if not is_single:
                dropped_multi += 1
                continue
            kept_sub_indices.append(sub_i)

        kept_embs = src_embs[kept_sub_indices]
        labels = np.zeros((len(kept_sub_indices), N_SDG), dtype=np.float32)
        source_docs_corpus: list[str] = []
        for j, sub_i in enumerate(kept_sub_indices):
            entry = src_rows[sub_i]
            sdg = entry.get("sdg")
            sdgs = entry.get("sdgs")
            if sdg is not None and 1 <= sdg <= N_SDG:
                labels[j, sdg - 1] = 1.0
            elif isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= N_SDG:
                labels[j, sdgs[0] - 1] = 1.0
            # Rows without source_doc get a UNIQUE per-row key, i.e. they
            # become singleton "documents" — the train/test leakage guard
            # silently degrades to row-level splitting for those rows.
            source_docs_corpus.append(entry.get("source_doc", f"{name}_{sub_i}"))

        embs_list.append(kept_embs)
        labels_list.append(labels)
        sources_list.extend([name] * len(kept_sub_indices))
        source_docs_list.extend(source_docs_corpus)

        source_summary[name] = len(kept_sub_indices)

    if not embs_list:
        log.error("No corpora loaded — nothing to do.")
        # Fail closed: see note above — silent success would corrupt the
        # supervised-data contract downstream.
        raise SystemExit(1)

    embeddings = np.vstack(embs_list)
    labels = np.vstack(labels_list)
    sources = np.array(sources_list)
    source_docs = np.array(source_docs_list)

    log.info("Total: %d texts", len(embeddings))

    all_idx = np.arange(len(embeddings))
    train_pool_idx, test_idx = [], []

    for src in np.unique(sources):
        mask = sources == src
        src_idx = all_idx[mask]

        # Sources with < 5 texts cannot be document-group-split meaningfully;
        # they are kept wholly in train and never appear in the held-out test
        # split. This floor shapes test-set composition by construction.
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

        n_test_docs = max(1, int(len(doc_groups) * (1.0 - args.train_frac)))
        try:
            train_doc_groups, test_doc_groups = train_test_split(
                doc_group_indices, test_size=n_test_docs / len(doc_groups),
                random_state=args.split_seed, stratify=doc_label_arr,
            )
        except ValueError:
            log.warning("  %s: stratification failed — falling back to unstratified split", src)
            train_doc_groups, test_doc_groups = train_test_split(
                doc_group_indices, test_size=n_test_docs / len(doc_groups),
                random_state=args.split_seed,
            )

        for gi in train_doc_groups:
            train_pool_idx.extend(doc_groups[gi])
        for gi in test_doc_groups:
            test_idx.extend(doc_groups[gi])

        n_train = sum(len(doc_groups[gi]) for gi in train_doc_groups)
        n_test = sum(len(doc_groups[gi]) for gi in test_doc_groups)
        log.info(
            "  %s: %d texts → %d train / %d test",
            src, source_summary.get(src, n_train + n_test), n_train, n_test,
        )

    train_pool_idx = np.array(train_pool_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indices").mkdir(parents=True, exist_ok=True)

    for _name, _arr in [
        ("embeddings.npy", embeddings),
        ("labels.npy", labels),
        ("sources.npy", sources),
        ("source_docs.npy", source_docs),
    ]:
        _p = output_dir / _name
        with _p.open("wb") as _f:
            np.save(_f, _arr)
            _f.flush()
    for _name, _arr in [
        ("train.npy", train_pool_idx),
        ("test.npy", test_idx),
    ]:
        _p = output_dir / "indices" / _name
        _p.parent.mkdir(parents=True, exist_ok=True)
        with _p.open("wb") as _f:
            np.save(_f, _arr)
            _f.flush()

    lines = ["=" * 70]
    lines.append("SPLIT REPORT — Per-source stratified document-grouped split")
    lines.append("=" * 70)
    lines.append(f"Total: {len(embeddings)} texts, {len(np.unique(sources))} sources")
    lines.append(f"train_frac={args.train_frac}  test_frac={1.0 - args.train_frac:.2f}  split_seed={args.split_seed}\n")

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
    log.info("Done. Train: %d  Test: %d", len(train_pool_idx), len(test_idx))


if __name__ == "__main__":
    main()
