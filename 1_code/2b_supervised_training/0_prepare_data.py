"""
Prepare training and test data from embedded reference corpora.

Loads all 5 embedded corpora from 2_data/2a_embedded_supervised/, builds binary
17D label vectors, and performs a source-blocked 85/15 split using iterative
multi-label stratification (Sechidis et al. 2011).

Why source-blocked?
  Each source (OSDG, KH, SDGi, Benchmark, Aurora) represents a different domain
  (community, journalism, policy, expert-policy, research). A global random split
  would produce a test set dominated by OSDG (30K texts) while under-representing
  Benchmark (616). Source-blocking ensures every domain is proportionally
  represented in both train and test.

Why iterative multi-label stratification?
  Standard train_test_split(stratify=y) works for single-label but cannot handle
  multi-label y. Random split could strand rare SDGs (e.g. SDG-14 in KH) entirely
  in train or test. The iterative algorithm (Sechidis et al. 2011, ECML PKDD)
  greedily assigns examples to test, prioritizing rare labels, so each SDG's
  prevalence is preserved across the split.

Why no validation set?
  Hyperparameter tuning uses 5-fold cross-validation on the training pool. A
  fixed validation set would waste ~17% of the 85% training pool (~7K texts).
  CV uses all training data for both fitting and scoring, producing a distribution
  of scores rather than a single val point.

Outputs (saved to 2_data/2b_supervised/):
  embeddings.npy    (N, dim) float32   — stacked from all reference corpora
  labels.npy        (N, 17) float32    — binary multi-label vectors
  sources.npy       (N,) str           — source corpus for each row
  indices/train.npy int64 — training indices (85% of each source, stratified)
  indices/test.npy  int64 — test indices (15% of each source, stratified)
  split_report.txt   — per-source and per-SDG breakdown of train/test

Run from project root (MiniLM default):
    python 1_code/2b_supervised_training/0_prepare_data.py
Run with MPNet:
    python 1_code/2b_supervised_training/0_prepare_data.py \\
        --embed-root 2_data/2c_embedded_mpnet_supervised \\
        --output-root 2_data/2c_supervised_mpnet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

N_SDG = 17

REFERENCE_CORPORA = [
    "osdg",
    "benchmark",
    "sdg_knowledge_hub",
    "sdgi",
    "aurora",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def iterative_multilabel_train_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-label stratified split (Sechidis et al. 2011, ECML PKDD).

    The algorithm:
      1. For each label l, compute desired_test_count[l] = round(n_positives[l] * test_size)
      2. While the test set has fewer than n_test examples:
         a. Among labels that still need test examples, find the one with the
            smallest proportion of remaining examples already assigned — this is
            the most "under-pressure" label
         b. Among available examples that have this label, pick the one with the
            fewest total active labels — this maximizes the chance of also covering
            other needed labels
         c. Move it to test, update remaining counts, repeat

    Args:
        indices: array of original indices (e.g., all source indices)
        y: (n, n_labels) binary label matrix
        test_size: fraction for test set
        random_state: reproducibility seed

    Returns:
        train_indices, test_indices — subsets of `indices`
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    n_test = int(round(n * test_size))
    n_labels = y.shape[1]

    n_positives = y.sum(axis=0)  # how many examples have each label
    desired = np.round(n_positives * test_size).astype(int)
    remaining_needed = desired.copy().astype(float)

    available = list(range(n))
    test_selected = []

    while len(test_selected) < n_test:
        labels_still_needed = np.where(remaining_needed > 0)[0]

        if len(labels_still_needed) == 0:
            # All label quotas filled — fill remaining test slots randomly
            n_remaining = n_test - len(test_selected)
            chosen = rng.choice(available, size=n_remaining, replace=False)
            test_selected.extend(chosen.tolist())
            break

        # Find the most under-pressure label: largest proportion of its
        # remaining positive examples that still need to go to test
        best_label = None
        best_ratio = -1.0
        for l in labels_still_needed:
            n_available_with_label = y[available, l].sum()
            if n_available_with_label > 0:
                ratio = remaining_needed[l] / n_available_with_label
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_label = l

        if best_label is None:
            # Safety: no available examples for any needed label — fill randomly
            n_remaining = n_test - len(test_selected)
            chosen = rng.choice(available, size=n_remaining, replace=False)
            test_selected.extend(chosen.tolist())
            break

        # Among available examples with this label, pick the one with the
        # fewest total labels (most "focused" — maximizes coverage efficiency)
        candidates = [i for i in available if y[i, best_label] == 1]
        # Sort by number of labels (ascending), random tiebreaker
        rng.shuffle(candidates)
        candidates.sort(key=lambda i: y[i].sum())
        chosen = candidates[0]

        test_selected.append(chosen)
        available.remove(chosen)

        # Decrement remaining_needed for every label this example has
        for l in range(n_labels):
            if y[chosen, l] == 1:
                remaining_needed[l] = max(0.0, remaining_needed[l] - 1.0)

    test_indices = indices[np.array(test_selected)]
    train_indices = indices[np.array(available)]

    return train_indices, test_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-label training data.")
    parser.add_argument("--embed-root", default="2_data/2a_embedded_supervised",
                        help="Embedding root dir (default: 2_data/2a_embedded_supervised)")
    parser.add_argument("--output-root", default="2_data/2b_supervised",
                        help="Output dir (default: 2_data/2b_supervised)")
    args = parser.parse_args()
    embed_root = Path(args.embed_root)
    output_dir = Path(args.output_root)
    log.info("Embed root: %s  Output: %s", embed_root, output_dir)

    all_embs, all_labels, all_sources = [], [], []

    for name in REFERENCE_CORPORA:
        emb_path = embed_root / f"{name}.npy"
        ids_path = embed_root / "metadata" / f"{name}_ids.json"

        if not emb_path.exists() or not ids_path.exists():
            log.warning("Missing: %s or %s — skipping", emb_path, ids_path)
            continue

        embs = np.load(emb_path).astype(np.float32)
        with ids_path.open() as f:
            ids_meta = json.load(f)

        # Build binary 17D label vectors from the "sdgs" field.
        # Each text may have 1..17 active SDGs; a text about SDG 3 and 5
        # gets labels[2] = 1 and labels[4] = 1.
        labels = np.zeros((len(ids_meta), N_SDG), dtype=np.float32)
        for i, entry in enumerate(ids_meta):
            for sdg in entry.get("sdgs", []):
                if 1 <= sdg <= N_SDG:
                    labels[i, sdg - 1] = 1.0

        all_embs.append(embs)
        all_labels.append(labels)
        all_sources.extend([name] * len(ids_meta))

        log.info(
            "Loaded %s: %d texts, %.0f total positive labels (%.2f per text)",
            name, len(ids_meta), labels.sum(), labels.sum() / len(ids_meta),
        )

    if not all_embs:
        log.error("No corpora loaded — nothing to do.")
        return

    embeddings = np.vstack(all_embs)
    labels = np.vstack(all_labels)
    sources = np.array(all_sources)

    log.info(
        "Total: %d texts, %d SDGs, %d sources",
        len(embeddings), N_SDG, len(np.unique(sources)),
    )

    all_idx = np.arange(len(embeddings))
    train_pool_idx, test_idx = [], []

    # Source-blocked split: for each source independently, do an 85/15
    # multi-label stratified split. This guarantees each domain is
    # proportionally represented in both train and test.
    for src in np.unique(sources):
        mask = sources == src
        src_idx = all_idx[mask]
        src_y = labels[mask]

        if len(src_idx) < 5:
            # Too few examples for a meaningful split — assign all to train
            train_pool_idx.extend(src_idx.tolist())
            log.warning("  %s: only %d texts — kept entirely in train", src, len(src_idx))
            continue

        s_train, s_test = iterative_multilabel_train_test_split(
            src_idx, src_y, test_size=0.15, random_state=42,
        )
        train_pool_idx.extend(s_train.tolist())
        test_idx.extend(s_test.tolist())

        log.info(
            "  %s: %d train + %d test (%.1f%% test, desired 15%%)",
            src, len(s_train), len(s_test), 100 * len(s_test) / len(src_idx),
        )

    train_pool_idx = np.array(train_pool_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    # No validation split — hyperparameter tuning uses 5-fold CV
    # on the full training pool. This preserves maximum training data.

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indices").mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "sources.npy", sources)
    np.save(output_dir / "indices" / "train.npy", train_pool_idx)
    np.save(output_dir / "indices" / "test.npy", test_idx)

    # ---- Split quality report ----
    lines = ["=" * 70]
    lines.append("SPLIT REPORT — Source-blocked 85/15 multi-label stratified")
    lines.append("=" * 70)
    lines.append(f"Total: {len(embeddings)} texts, {len(np.unique(sources))} sources\n")

    for name_split, name_idx in [("Train", train_pool_idx), ("Test", test_idx)]:
        lines.append(f"--- {name_split} ({len(name_idx)} texts) ---")
        # Per-source breakdown
        for src in np.unique(sources):
            n = int((sources[name_idx] == src).sum())
            lines.append(f"  {src:20s}: {n} texts")
        lines.append("")

    # Per-SDG label distribution in each split
    lines.append("--- Per-SDG label counts (train | test) ---")
    for sdg in range(N_SDG):
        train_count = int(labels[train_pool_idx, sdg].sum())
        test_count = int(labels[test_idx, sdg].sum())
        train_pct = train_count / labels[:, sdg].sum() * 100
        lines.append(f"  SDG-{sdg+1:2d}: {train_count:5d} train ({train_pct:.1f}%) | {test_count:5d} test")

    lines.append("")
    train_density = labels[train_pool_idx].sum() / len(train_pool_idx)
    test_density = labels[test_idx].sum() / len(test_idx)
    lines.append(f"Train label density: {train_density:.4f}")
    lines.append(f"Test  label density: {test_density:.4f}")
    lines.append(f"Ratio (test/train): {test_density / train_density:.4f}")

    report_path = output_dir / "split_report.txt"
    report_path.write_text("\n".join(lines))
    log.info("Saved split report → %s", report_path)

    print("\n".join(lines[-10:]))
    print(f"\nDone. Train: {len(train_pool_idx)}  Test: {len(test_idx)}")
    print(f"  All data → {output_dir}")


if __name__ == "__main__":
    main()
