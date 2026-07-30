"""
token_dist.py — Token-length distribution analysis across the three dissertation corpora.

Computes per-source token/word statistics and produces an overlapping KDE of token
lengths for the three main corpus categories:

    Reference / Training  (OSDG + Benchmark)
    Policy                (Knowledge Hub + SDGi + Aurora)
    Research              (OpenAlex academic papers)

Outputs:
      4_outputs/main/figures/fig2_token_distribution.pdf
      4_outputs/main/figures/fig2_token_distribution.png

Run:
    python 1_code/token_dist.py [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-dissertation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "1_code" / "7_main_analysis" / "0_shared"))
from model_utils import preprocessed_dir, RANDOM_SEED

RNG = np.random.default_rng(RANDOM_SEED)
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MPNET_LIMIT = 384
MINILM_LIMIT = 256
MAX_SAMPLE_PER_SOURCE = 10_000
OUTPUT_DIR = ROOT / "4_outputs" / "main" / "figures"

SOURCE_DEFS = [
    {"group": "Reference / Training", "label": "OSDG",
     "path": "osdg/osdg_clean.jsonl", "field": "text"},
    {"group": "Reference / Training", "label": "Benchmark",
     "path": "sdg_benchmark/benchmark_clean.jsonl", "field": "text"},
    {"group": "Policy", "label": "Knowledge Hub",
     "path": "sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl", "field": "text"},
    {"group": "Policy", "label": "SDGi",
     "path": "sdgi/sdgi_clean.jsonl", "field": "text"},
    {"group": "Policy", "label": "Aurora",
     "path": "aurora/aurora_clean.jsonl", "field": "text"},
    {"group": "Research", "label": "OpenAlex",
     "path": None, "field": "combined_text", "is_research": True},
]

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token distribution analysis across corpora.")
    p.add_argument("--max-sample-per-source", type=int, default=MAX_SAMPLE_PER_SOURCE,
                   help="Max texts sampled per source corpus for the token-length analysis (default: %(default)s)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def load_jsonl(path: Path, field: str, max_records: int | None = None) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if field in obj and obj[field]:
                texts.append(obj[field])
    if max_records and len(texts) > max_records:
        texts = list(RNG.choice(texts, size=max_records, replace=False))
    return texts


def load_research_shards(n_shards: int = 5, target: int = MAX_SAMPLE_PER_SOURCE) -> list[str]:
    base = preprocessed_dir() / "research"
    per_shard = target // n_shards
    records = []
    for i in range(1, n_shards + 1):
        path = base / f"part-{i:05d}.jsonl"
        shard_texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "combined_text" in obj and obj["combined_text"]:
                    shard_texts.append(obj["combined_text"])
        if len(shard_texts) > per_shard:
            shard_texts = list(RNG.choice(shard_texts, size=per_shard, replace=False))
        records.extend(shard_texts)
    return records


def load_all_texts(max_sample_per_source: int = MAX_SAMPLE_PER_SOURCE) -> dict[str, list[str]]:
    texts_by_label: dict[str, list[str]] = {}
    for src in SOURCE_DEFS:
        label = src["label"]
        if src.get("is_research"):
            texts = load_research_shards(n_shards=5, target=max_sample_per_source)
        else:
            path = preprocessed_dir() / src["path"]
            if not path.exists():
                log.warning("Path not found: %s", path)
                continue
            texts = load_jsonl(path, src["field"], max_records=max_sample_per_source)
        if texts:
            texts_by_label[label] = texts
            log.info("  %s: %d texts", label, len(texts))
    return texts_by_label


def tokenize_all(texts_by_label: dict[str, list[str]],
                 tokenizers) -> dict:
    tokenizer_mpnet, tokenizer_minilm = tokenizers
    results = {}
    for label, texts in texts_by_label.items():
        words = []
        mpnet_lens = []
        minilm_lens = []
        for t in texts:
            wc = len(t.split())
            words.append(wc)
            mpnet_lens.append(len(tokenizer_mpnet.encode(t)))
            minilm_lens.append(len(tokenizer_minilm.encode(t)))
        results[label] = {
            "words": np.array(words, dtype=np.int32),
            "mpnet": np.array(mpnet_lens, dtype=np.int32),
            "minilm": np.array(minilm_lens, dtype=np.int32),
        }
    return results


def compute_stats(arr_tokens: np.ndarray, arr_words: np.ndarray,
                  trunc_threshold: int) -> dict:
    n = len(arr_tokens)
    pcts = [1, 5, 25, 50, 75, 95, 99]
    t_pct = {p: float(np.percentile(arr_tokens, p)) for p in pcts}
    w_pct = {p: float(np.percentile(arr_words, p)) for p in pcts}
    return {
        "n": n,
        "words": {
            "min": int(arr_words.min()), "max": int(arr_words.max()),
            "mean": float(arr_words.mean()), "p": w_pct,
        },
        "tokens": {
            "min": int(arr_tokens.min()), "max": int(arr_tokens.max()),
            "mean": float(arr_tokens.mean()), "p": t_pct,
            "trunc_pct": float(np.mean(arr_tokens > trunc_threshold)) * 100,
        },
    }


def print_stats(name: str, ws: np.ndarray, ts: np.ndarray, model_label: str,
                trunc_limit: int) -> None:
    w = compute_stats(ts, ws, trunc_limit)
    t = w["tokens"]
    wo = w["words"]
    print(f"  {name:<40s} n={w['n']:<8d}")
    print(f"  {'':40s} Words:    min={wo['min']:<6d}  p1={wo['p'][1]:<6.0f}  "
          f"p5={wo['p'][5]:<6.0f}  p25={wo['p'][25]:<6.0f}  "
          f"p50={wo['p'][50]:<6.0f}  p75={wo['p'][75]:<6.0f}  "
          f"p95={wo['p'][95]:<6.0f}  p99={wo['p'][99]:<6.0f}  "
          f"max={wo['max']:<6d}  mean={wo['mean']:<7.1f}")
    print(f"  {'':40s} {model_label:<7s} min={t['min']:<6d}  p1={t['p'][1]:<6.0f}  "
          f"p5={t['p'][5]:<6.0f}  p25={t['p'][25]:<6.0f}  "
          f"p50={t['p'][50]:<6.0f}  p75={t['p'][75]:<6.0f}  "
          f"p95={t['p'][95]:<6.0f}  p99={t['p'][99]:<6.0f}  "
          f"max={t['max']:<6d}  mean={t['mean']:<7.1f}  "
          f"trunc@{trunc_limit}={t['trunc_pct']:.1f}%")


def find_truncation_thresholds(ws: np.ndarray, ts: np.ndarray,
                               target_tok: int) -> int | None:
    total = len(ws)
    for wc_thresh in range(0, 2001, 10):
        pct_ok = np.sum((ws <= wc_thresh) & (ts <= target_tok)) / total * 100
        if pct_ok >= 99:
            return wc_thresh
    return None


def plot_kde(group_arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    group_colors = {
        "Reference / Training": "#228833",
        "Policy": "#EE7733",
        "Research": "#0077BB",
    }

    group_labels = {
        "Reference / Training": "Reference / Training  (OSDG + Benchmark)",
        "Policy": "Policy  (KH + SDGi + Aurora)",
        "Research": "Research  (OpenAlex papers)",
    }

    x_range = np.linspace(0, 700, 1000)

    for gname in ["Reference / Training", "Policy", "Research"]:
        arr = group_arrays.get(gname)
        if arr is None or len(arr) < 2:
            continue
        clipped = np.clip(arr, 1, None)
        try:
            kde = gaussian_kde(clipped)
            density = kde(x_range)
            ax.fill_between(x_range, density, alpha=0.15, color=group_colors[gname])
            ax.plot(x_range, density, color=group_colors[gname], linewidth=2,
                    label=group_labels[gname])
        except np.linalg.LinAlgError:
            ax.hist(clipped, bins=80, density=True, alpha=0.4, color=group_colors[gname],
                    histtype="step", linewidth=1.5, label=group_labels[gname])

    for limit, style, label in [
        (MPNET_LIMIT, (0, (3, 3)), f"MPNet max length ({MPNET_LIMIT})"),
        (MINILM_LIMIT, (0, (1.5, 2)), f"MiniLM max length ({MINILM_LIMIT})"),
    ]:
        ax.axvline(limit, color="red" if limit == MPNET_LIMIT else "grey",
                   linestyle=style, linewidth=1.5, alpha=0.7, label=label)

    ax.set_xlabel("Token length (MPNet tokenizer)")
    ax.set_ylabel("Density")
    ax.set_title("Token-length distribution across dissertation corpora", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig2_token_distribution.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig2_token_distribution.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'fig2_token_distribution.pdf'}")


def main() -> None:
    args = parse_args()

    output_dir = OUTPUT_DIR
    pdf_path = output_dir / "fig2_token_distribution.pdf"
    if pdf_path.exists() and not args.overwrite:
        print(f"{pdf_path} exists. Use --overwrite to regenerate.")
        return

    print("Loading tokenizers...", file=sys.stderr)
    tokenizer_mpnet = AutoTokenizer.from_pretrained(MPNET_MODEL)
    tokenizer_minilm = AutoTokenizer.from_pretrained(MINILM_MODEL)

    MPNET_LIMIT = 384
    MINILM_LIMIT = 256

    print("Loading and tokenizing texts...", file=sys.stderr)
    texts_by_label = load_all_texts(args.max_sample_per_source)
    tokenized = tokenize_all(texts_by_label, (tokenizer_mpnet, tokenizer_minilm))

    print("\n\n" + "=" * 100)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("=" * 100)

    print("\n--- 1. Per-Source Statistics (MPNet tokens) ---\n")
    for label, d in tokenized.items():
        print_stats(label, d["words"], d["mpnet"], "MPNet", MPNET_LIMIT)

    print("\n--- 2. Per-Source Statistics (MiniLM tokens) ---\n")
    for label, d in tokenized.items():
        print_stats(label, d["words"], d["minilm"], "MiniLM", MINILM_LIMIT)

    print("\n--- 3. Group-Level Aggregates (MPNet tokens) ---\n")
    group_lengths: dict[str, np.ndarray] = {}
    for src in SOURCE_DEFS:
        gname = src["group"]
        label = src["label"]
        if label not in tokenized:
            continue
        if gname not in group_lengths:
            group_lengths[gname] = tokenized[label]["mpnet"]
        else:
            group_lengths[gname] = np.concatenate([group_lengths[gname],
                                                   tokenized[label]["mpnet"]])

    for gname, arr in group_lengths.items():
        w_pooled = np.concatenate([tokenized[l]["words"]
                                   for l in tokenized
                                   if any(s["label"] == l and s["group"] == gname
                                          for s in SOURCE_DEFS)])
        print(f"  {gname:<30s} n={len(arr):<8d}  "
              f"min={arr.min():<6d}  p50={np.median(arr):<8.0f}  "
              f"p95={np.percentile(arr, 95):<8.0f}  "
              f"max={arr.max():<6d}  mean={arr.mean():<7.1f}  "
              f"words_mean={w_pooled.mean():<7.1f}")

    print("\n--- 4. Truncation Thresholds (word-count that keeps 99% under model limit) ---\n")
    for label, d in tokenized.items():
        thresholds = {}
        for target, name in [(MPNET_LIMIT, "MPNet"), (MINILM_LIMIT, "MiniLM")]:
            wc = find_truncation_thresholds(d["words"], d["mpnet"], target)
            thresholds[name] = wc
        parts = "  ".join(
            f"{k}: wc <= {v}" if v else f"{k}: not found up to 2000"
            for k, v in thresholds.items()
        )
        print(f"  {label:<30s} {parts}")

    print("\n--- 5. Overlapping KDE Figure ---\n")
    plot_kde(group_lengths, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()
