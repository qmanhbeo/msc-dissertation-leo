"""
fig_truncation_distribution.py — Token-length distribution before and after segmentation.

Two-panel figure (MPNet / MiniLM):
  - KDE of token lengths for each corpus
  - Pre-fix (whole text, dashed) vs post-fix (segmented, solid)
  - Vertical line at model max_seq_length

Output:
  4_outputs/appendix/fig_truncation_distribution.pdf
  4_outputs/appendix/fig_truncation_distribution.png

Run from project root:
    python 1_code/8_visualization/fig_truncation_distribution.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-dissertation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "1_code"))
sys.path.insert(0, str(ROOT / "1_code" / "7_main_analysis" / "0_shared"))

from model_utils import preprocessed_dir
from sentence_transformers import SentenceTransformer

CORPORA = [
    {
        "label": "OSDG",
        "pre_path": str(preprocessed_dir() / "osdg/osdg_clean.jsonl"),
        "pre_field": "text",
        "post_mpnet": None,
        "post_minilm": None,
        "color": "#4477AA",
    },
    {
        "label": "Benchmark",
        "pre_path": str(preprocessed_dir() / "sdg_benchmark/benchmark_clean.jsonl"),
        "pre_field": "text",
        "post_mpnet": None,
        "post_minilm": None,
        "color": "#66CCEE",
    },
    {
        "label": "KH (pre)",
        "pre_path": str(preprocessed_dir() / "sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl"),
        "pre_field": "text",
        "post_mpnet": str(preprocessed_dir() / "sdg_knowledge_hub/sdg_knowledge_hub_segmented_all-mpnet-base-v2.jsonl"),
        "post_minilm": str(preprocessed_dir() / "sdg_knowledge_hub/sdg_knowledge_hub_segmented_all-minilm-l6-v2.jsonl"),
        "color": "#228833",
    },
    {
        "label": "SDGi (pre)",
        "pre_path": str(preprocessed_dir() / "sdgi_corpus/sdgi_clean.jsonl"),
        "pre_field": "text",
        "post_mpnet": str(preprocessed_dir() / "sdgi_corpus/sdgi_unified_all-mpnet-base-v2.jsonl"),
        "post_minilm": str(preprocessed_dir() / "sdgi_corpus/sdgi_unified_all-minilm-l6-v2.jsonl"),
        "color": "#EE7733",
    },
    {
        "label": "Aurora",
        "pre_path": str(preprocessed_dir() / "aurora/aurora_texts.jsonl"),
        "pre_field": "text",
        "post_mpnet": str(preprocessed_dir() / "aurora/aurora_segmented_all-mpnet-base-v2.jsonl"),
        "post_minilm": str(preprocessed_dir() / "aurora/aurora_segmented_all-minilm-l6-v2.jsonl"),
        "color": "#CCBB44",
    },
    {
        "label": "Research\n(shard 10)",
        "pre_path": str(preprocessed_dir() / "research_corpus/part-00010.jsonl"),
        "pre_field": "combined_text",
        "post_mpnet": str(preprocessed_dir() / "research_corpus/segmented_all-mpnet-base-v2/part-00010.jsonl"),
        "post_minilm": None,
        "color": "#AA3377",
    },
]

MODELS = [
    ("all-mpnet-base-v2", 384, "MPNet", "mpnet"),
    ("all-MiniLM-L6-v2", 256, "MiniLM", "minilm"),
]

OUTPUT_DIR = ROOT / "4_outputs" / "appendix"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_texts(path: Path, field: str) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row.get(field, ""))
    return texts


def tokenize_lengths(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    encoded = model.tokenizer(texts, truncation=False, padding=False, return_length=True)
    return np.array(encoded["length"], dtype=np.int32)


def plot_model_panel(ax, model_name: str, max_len: int, model_label: str, corpus_data: list[dict]) -> None:
    st = SentenceTransformer(model_name)
    log.info("Tokenising %s ...", model_label)

    for cp in corpus_data:
        color = cp["color"]
        label_pre = cp["label"]

        # Pre-fix
        pre_path = cp.get("pre_path")
        if pre_path and Path(pre_path).exists():
            texts = load_texts(Path(pre_path), cp.get("pre_field", "text"))
            if len(texts) > 50000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(texts), 50000, replace=False)
                texts = [texts[i] for i in idx]
            if texts:
                lengths = tokenize_lengths(texts, st)
                clipped = np.clip(lengths, 0, max_len * 3)
                ax.hist(clipped, bins=80, density=True, alpha=0.25, color=color,
                        histtype="stepfilled", linewidth=0)
                ax.hist(clipped, bins=80, density=True, alpha=0.8, color=color,
                        histtype="step", linewidth=1.2, linestyle="--",
                        label=f"{label_pre} (n={len(texts):,})")

        # Post-fix
        post_key = "post_mpnet" if "mpnet" in model_name.lower() else "post_minilm"
        post_path = cp.get(post_key)
        if post_path and Path(post_path).exists():
            texts = load_texts(Path(post_path), "text")
            if len(texts) > 50000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(texts), 50000, replace=False)
                texts = [texts[i] for i in idx]
            if texts:
                lengths = tokenize_lengths(texts, st)
                clipped = np.clip(lengths, 0, max_len * 3)
                ax.hist(clipped, bins=80, density=True, alpha=0.25, color=color,
                        histtype="stepfilled", linewidth=0)
                ax.hist(clipped, bins=80, density=True, alpha=0.8, color=color,
                        histtype="step", linewidth=1.8, linestyle="-",
                        label=f"{label_pre} post (n={len(texts):,})")

    ax.axvline(max_len, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"max_len={max_len}")
    ax.set_xlabel("Token length")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, loc="upper right")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for idx, (model_name, max_len, model_label, _) in enumerate(MODELS):
        plot_model_panel(axes[idx], model_name, max_len, model_label, CORPORA)

    fig.suptitle("Token-length distribution before and after segmentation", fontsize=13, y=1.02)
    fig.tight_layout()

    pdf_path = FIGURES_DIR / "fig_truncation_distribution.pdf"
    png_path = FIGURES_DIR / "fig_truncation_distribution.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

if __name__ == "__main__":
    main()
