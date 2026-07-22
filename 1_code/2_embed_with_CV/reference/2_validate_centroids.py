"""
Validate SDG centroids via cross-validation on pooled 5-source canon.

Pool all 5 labelled sources (OSDG, KH, SDGi, Aurora, Benchmark), apply quality
filters (MIN_WORDS, agreement >= 0.5), perform a per-source stratified 85/15
train/test split, build centroids from the train partition, and evaluate
nearest-centroid classification on the held-out test partition.

This replaces the older single held-out evaluation on the 616-text SDG
Benchmark with a robust cross-validation estimate across ALL available canon
data. It directly parallels the supervised pipeline's evaluation strategy in
2b_supervised_training_singlelabel/0_prepare_data.py.

Interpretation guide (macro-F1 on SDGs 1-17):
  < 0.25   FAIL  - serious concern
  0.25-0.50 WARN  - usable signal but moderate noise
  > 0.50   PASS  - good instrument

Outputs:
  4_outputs/main/data/4_1_validation_results.json
  4_outputs/main/data/4_1_confusion_matrix.csv
  4_outputs/main/data/4_1_centroid_similarity_matrix.csv
  4_outputs/main/tables/num_validation.tex

Run from project root:
    python 1_code/2_embed_with_CV/reference/2_validate_centroids.py
    python 1_code/2_embed_with_CV/reference/2_validate_centroids.py --model all-mpnet-base-v2
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

N_SDG = 17
PREPROCESS_ROOT = Path("2_data/1_preprocessed")

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "3_appendix_centroid" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from shared_utils import ensure_canonical_outputs
from model_utils import embed_dir_for_model, DEFAULT_EMBED_MODEL

THRESH_FAIL = 0.25
THRESH_PASS = 0.50
RANDOM_BASELINE = 1 / 17

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_centroid_from_embeddings(vecs: np.ndarray) -> np.ndarray:
    raw = vecs.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        raise ValueError("Near-zero centroid norm")
    return (raw / norm).astype(np.float32)


def run_bootstrap(true_sdgs: np.ndarray, pred_sdgs: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(true_sdgs)
    per_sdg_boot = np.zeros((n_boot, N_SDG), dtype=np.float64)
    macro_boot = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = true_sdgs[idx]
        p = pred_sdgs[idx]
        f1 = f1_score(t, p, average=None, labels=list(range(1, N_SDG + 1)), zero_division=0)
        per_sdg_boot[b] = f1
        macro_boot[b] = float(f1.mean())
    lo, hi = 2.5, 97.5
    per_sdg_ci = {
        str(sdg): {
            "point": round(float(per_sdg_boot[:, sdg - 1].mean()), 4),
            "ci_low": round(float(np.percentile(per_sdg_boot[:, sdg - 1], lo)), 4),
            "ci_high": round(float(np.percentile(per_sdg_boot[:, sdg - 1], hi)), 4),
        }
        for sdg in range(1, N_SDG + 1)
    }
    macro_ci = {
        "point": round(float(macro_boot.mean()), 4),
        "ci_low": round(float(np.percentile(macro_boot, lo)), 4),
        "ci_high": round(float(np.percentile(macro_boot, hi)), 4),
    }
    return {"n_boot": n_boot, "seed": seed, "per_sdg_f1_ci": per_sdg_ci, "macro_f1_ci": macro_ci}


def save_csv_matrix(matrix: np.ndarray, labels: list, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"SDG{l}" for l in labels])
        for i, row in enumerate(matrix):
            writer.writerow([f"SDG{labels[i]}"] + [f"{v:.4f}" for v in row])


def build_corpora(embed_root: Path) -> list:
    return [
        {
            "name": "osdg",
            "embed_file": "osdg.npy",
            "jsonl_path": PREPROCESS_ROOT / "osdg" / "osdg_clean.jsonl",
            "min_words": 20,
            "need_agreement": True,
        },
        {
            "name": "benchmark",
            "embed_file": "benchmark.npy",
            "jsonl_path": PREPROCESS_ROOT / "sdg_benchmark" / "benchmark_clean.jsonl",
            "min_words": 10,
            "need_agreement": False,
        },
        {
            "name": "sdg_knowledge_hub",
            "embed_file": "sdg_knowledge_hub.npy",
            "jsonl_path": PREPROCESS_ROOT / "sdg_knowledge_hub" / "sdg_knowledge_hub_clean.jsonl",
            "min_words": 20,
            "need_agreement": False,
        },
        {
            "name": "sdgi",
            "embed_file": "sdgi.npy",
            "jsonl_path": PREPROCESS_ROOT / "sdgi_corpus" / "sdgi_clean.jsonl",
            "min_words": 20,
            "need_agreement": False,
        },
        {
            "name": "aurora",
            "embed_file": "aurora.npy",
            "jsonl_path": PREPROCESS_ROOT / "aurora" / "aurora_texts.jsonl",
            "min_words": 20,
            "need_agreement": False,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SDG centroids via pooled CV.")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    args = parser.parse_args()
    model = args.model

    embed_root = embed_dir_for_model(model)
    log.info("Model: %s  Embed root: %s", model, embed_root)
    CORPORA = build_corpora(embed_root)
    for c in CORPORA:
        c["embed_root"] = embed_root

    layout = ensure_canonical_outputs(Path("4_outputs/main"))
    out_results = layout.data_dir / "4_1_validation_results.json"
    out_confusion = layout.data_dir / "4_1_confusion_matrix.csv"
    out_centroid_sim = layout.data_dir / "4_1_centroid_similarity_matrix.csv"

    all_embs, all_labels, all_sources = [], [], []
    total_dropped_short = 0
    total_dropped_agreement = 0

    for corpus in CORPORA:
        name = corpus["name"]
        emb_path = corpus["embed_root"] / corpus["embed_file"]
        jsonl_path = corpus["jsonl_path"]

        if not emb_path.exists() or not jsonl_path.exists():
            log.warning("Missing: %s or %s — skipping", emb_path, jsonl_path)
            continue

        embs = np.load(emb_path).astype(np.float32)
        rows = load_jsonl(jsonl_path)

        if len(embs) != len(rows):
            log.error("Mismatch: %s embeddings (%d) vs JSONL (%d) — skipping",
                      name, len(embs), len(rows))
            continue

        kept_indices = []
        dropped_short = 0
        dropped_agreement = 0

        for i, entry in enumerate(rows):
            text = entry.get("text", "")
            word_count = len(text.split())
            if word_count < corpus["min_words"]:
                dropped_short += 1
                continue
            if corpus["need_agreement"] and entry.get("agreement", 0) < 0.5:
                dropped_agreement += 1
                continue
            kept_indices.append(i)

        kept_embs = embs[kept_indices]
        sdg_labels = np.zeros(len(kept_indices), dtype=int)
        for j, i in enumerate(kept_indices):
            sdg = rows[i].get("sdg")
            if sdg is not None and 1 <= sdg <= N_SDG:
                sdg_labels[j] = sdg

        all_embs.append(kept_embs)
        all_labels.append(sdg_labels)
        all_sources.extend([name] * len(kept_indices))

        log.info("  %s: %d texts (dropped: %d short, %d agreement)",
                 name, len(kept_indices), dropped_short, dropped_agreement)
        total_dropped_short += dropped_short
        total_dropped_agreement += dropped_agreement

    if not all_embs:
        log.error("No corpora loaded — nothing to do.")
        return

    embeddings = np.vstack(all_embs)
    labels = np.concatenate(all_labels)
    sources = np.array(all_sources)

    log.info("Total: %d texts, %d dropped short + %d dropped agreement",
             len(embeddings), total_dropped_short, total_dropped_agreement)

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

        s_train, s_test = train_test_split(
            src_idx, test_size=0.15, random_state=42, stratify=src_y,
        )
        train_pool_idx.extend(s_train.tolist())
        test_idx.extend(s_test.tolist())

        log.info("  %s: %d train + %d test", src, len(s_train), len(s_test))

    train_pool_idx = np.array(train_pool_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    train_emb = embeddings[train_pool_idx]
    train_labels = labels[train_pool_idx]
    test_emb = embeddings[test_idx]
    test_labels = labels[test_idx]

    log.info("\nTrain: %d  Test: %d", len(train_emb), len(test_emb))
    log.info("Per-SDG n (train / test):")
    for sdg in range(1, N_SDG + 1):
        tn = int((train_labels == sdg).sum())
        ten = int((test_labels == sdg).sum())
        log.info("  SDG %2d: %5d train / %5d test", sdg, tn, ten)

    centroids = np.zeros((N_SDG, embeddings.shape[1]), dtype=np.float32)
    for sdg in range(1, N_SDG + 1):
        mask = train_labels == sdg
        if mask.sum() == 0:
            log.warning("SDG %d has zero training examples — centroid will be zero", sdg)
            continue
        centroids[sdg - 1] = build_centroid_from_embeddings(train_emb[mask])

    log.info("Centroids built: %d of %d SDGs have data",
             int((centroids != 0).any(axis=1).sum()), N_SDG)

    norms = np.linalg.norm(centroids, axis=1)
    valid_mask = norms > 1e-8
    if not valid_mask.all():
        log.warning("Zero-norm centroids (no train data): SDGs %s",
                     [i + 1 for i, v in enumerate(valid_mask) if not v])

    centroid_sim = centroids @ centroids.T

    scores = test_emb @ centroids.T
    pred_sdgs = scores.argmax(axis=1) + 1

    labels_17 = list(range(1, N_SDG + 1))
    acc = float(accuracy_score(test_labels, pred_sdgs))
    mf1 = float(f1_score(test_labels, pred_sdgs, average="macro", labels=labels_17, zero_division=0))
    per_sdg_f1 = f1_score(test_labels, pred_sdgs, average=None, labels=labels_17, zero_division=0)

    if mf1 >= THRESH_PASS:
        flag = "PASS"
    elif mf1 >= THRESH_FAIL:
        flag = "WARN"
    else:
        flag = "FAIL"

    log.info("")
    log.info("=" * 60)
    log.info("CENTROID VALIDATION RESULTS (pooled 5-source, 85/15 per-source split)")
    log.info("=" * 60)
    log.info("")
    log.info("  Accuracy : %.4f  (random baseline: %.4f)", acc, RANDOM_BASELINE)
    log.info("  Macro-F1 : %.4f  → %s", mf1, flag)
    if flag == "FAIL":
        log.warning("  FAIL: Macro-F1 < %.2f — instrument too noisy.", THRESH_FAIL)
    elif flag == "WARN":
        log.warning("  WARN: Macro-F1 %.2f–%.2f — usable but moderate noise.",
                    THRESH_FAIL, THRESH_PASS)
    else:
        log.info("  PASS: Macro-F1 >= %.2f — instrument validated.", THRESH_PASS)

    log.info("")
    log.info("PER-SDG F1:")
    log.info("  %-6s  %-8s  %-6s", "SDG", "F1", "n_test")
    log.info("  " + "-" * 30)
    for i, sdg in enumerate(labels_17):
        n_test = int((test_labels == sdg).sum())
        log.info("  SDG %2d   %.4f   n=%3d", sdg, per_sdg_f1[i], n_test)

    log.info("")
    log.info("CENTROID NEAREST NEIGHBOURS (top-2, excluding self):")
    for i in range(N_SDG):
        sim_row = centroid_sim[i].copy()
        sim_row[i] = -1.0
        top2_idx = np.argsort(sim_row)[::-1][:2]
        top2_str = ", ".join(f"SDG{j+1} ({centroid_sim[i,j]:.3f})" for j in top2_idx)
        log.info("  SDG %2d <- nearest: %s", i + 1, top2_str)

    results = {
        "evaluation": {
            "n_train": len(train_emb),
            "n_test": len(test_emb),
            "accuracy": round(acc, 6),
            "macro_f1": round(mf1, 6),
            "split": "per-source stratified 85/15",
            "note": (
                "Pooled all 5 sources (OSDG, KH, SDGi, Aurora, Benchmark), "
                "per-source stratified 85/15 split, centroids from train, "
                "nearest-centroid eval on test."
            ),
        },
        "per_sdg_f1": {str(sdg): round(float(per_sdg_f1[i]), 6)
                       for i, sdg in enumerate(labels_17)},
        "instrument_flag": flag,
        "random_baseline": round(RANDOM_BASELINE, 6),
        "thresholds": {"fail_below": THRESH_FAIL, "pass_above": THRESH_PASS},
    }

    layout.data_dir.mkdir(parents=True, exist_ok=True)
    with out_results.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("\nSaved: %s", out_results)

    cm = confusion_matrix(test_labels, pred_sdgs, labels=labels_17)
    save_csv_matrix(cm.astype(float), labels_17, out_confusion)
    log.info("Saved: %s", out_confusion)

    save_csv_matrix(centroid_sim, labels_17, out_centroid_sim)
    log.info("Saved: %s", out_centroid_sim)

    n_boot = 10000
    boot = run_bootstrap(test_labels, pred_sdgs, n_boot, seed=42)
    out_boot = layout.data_dir / "4_1_validation_bootstrap_ci.json"
    with out_boot.open("w", encoding="utf-8") as f:
        json.dump(boot, f, indent=2)
    log.info("")
    log.info("BOOTSTRAP CIs (n=%d resamples, seed=%d)", n_boot, 42)
    log.info("  Macro-F1 CI : [%.3f, %.3f]  (point %.3f)",
             boot["macro_f1_ci"]["ci_low"], boot["macro_f1_ci"]["ci_high"], boot["macro_f1_ci"]["point"])
    log.info("Saved: %s", out_boot)

    _sdg_num_words = {
        1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
        15: "Fifteen", 16: "Sixteen", 17: "Seventeen",
    }
    tables_dir = layout.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    num_lines = [
        "% Auto-generated by 1_code/2_embed_with_CV/reference/2_validate_centroids.py",
        rf"\newcommand{{\MacroFOne}}{{{mf1:.3f}}}",
        rf"\newcommand{{\ValidationAccuracy}}{{{acc:.3f}}}",
        rf"\newcommand{{\RandomBaselineSeventeenClass}}{{{RANDOM_BASELINE:.3f}}}",
        rf"\newcommand{{\ValidationVsRandomMultiple}}{{{(mf1 / RANDOM_BASELINE):.1f}}}",
    ]
    for i, sdg in enumerate(labels_17):
        word = _sdg_num_words[sdg]
        num_lines.append(rf"\newcommand{{\FiSdg{word}}}{{{per_sdg_f1[i]:.3f}}}")
    (tables_dir / "num_validation.tex").write_text("\n".join(num_lines) + "\n", encoding="utf-8")
    log.info("Saved: %s", tables_dir / "num_validation.tex")

    log.info("\nDone. Macro-F1 = %.4f (%s)", mf1, flag)


if __name__ == "__main__":
    main()
