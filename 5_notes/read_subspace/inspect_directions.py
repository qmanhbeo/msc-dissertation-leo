"""Scratch inspection of the INLP register subspace G (MPNet canonical).

Goal: answer "what words are on the projected (removed) subspace?" -- the dense
analog of Ravfogel et al. (2020) Table 4, which they could only publish directly
in the bag-of-words setting because BOW axes ARE words. In a dense embedding
space you cannot read the subspace; you inspect a removed direction g_k by
finding the words whose frozen encoder embedding is most aligned (cosine) with
it. Each g_k is one row of G (already orthonormalized corpus-discrimination
direction from register_adjust.py).

This is SCRATCH only: writes to 5_notes/read_subspace/. It does NOT touch the
pipeline, 2_data outputs, or 4_outputs.

Outputs:
  - read_subspace_report.md   (human-readable)
  - read_subspace_topwords.json (machine-readable)

Deterministic: encoder is frozen; vocabulary is fixed; TOP_N is a named constant.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

G_PATH = "/home/manh/dissertation/2_data/3b_register/mpnet/canon/G.npy"
MODEL_ID = "all-mpnet-base-v2"
TOP_N = 15          # words reported per direction (documented, no magic number)
SEED = 42

# ---- Diagnostic vocabulary -------------------------------------------------
# Balanced across three bins so the output is diagnostic: if removed directions
# align with REGISTER words, the register reading is supported; if they align
# with TOPIC words, that suggests topic leakage past SDG stratification.
REGISTER_WORDS = [
    # hedges
    "may", "might", "could", "possibly", "suggests", "appears", "potentially",
    "likely", "perhaps", "indicate",
    # deontic / directive modality
    "must", "shall", "should", "will", "need", "require", "expected",
    # first-person / author presence
    "we", "our", "us", "researchers", "authors",
    # nominalization / institutional nouns (style, not topic)
    "implementation", "development", "framework", "strategy", "policy",
    "recommendation", "report", "countries", "nations", "global",
    "institutional", "governance", "partnership", "target", "commitment",
    "conducted", "performed", "established",
]
TOPIC_WORDS = [
    # SDG nouns
    "climate", "energy", "health", "education", "water", "gender",
    "inequality", "innovation", "infrastructure", "cities", "consumption",
    "biodiversity", "oceans", "land", "peace", "poverty", "hunger",
    "sustainability", "sustainable",
    # AI / method nouns
    "neural", "model", "models", "algorithm", "learning", "deep",
    "dataset", "training", "classification", "prediction", "embedding",
]
NEUTRAL_WORDS = [
    "table", "figure", "number", "system", "result", "study", "data",
    "time", "group", "level", "process", "part", "value", "set", "case",
]
# corpus-leaning probes (which side does each direction track?)
RESEARCH_LEANING = ["accuracy", "performance", "proposed", "experiment", "baseline",
                    "ablation", "method", "architecture"]
POLICY_LEANING = ["governments", "states", "agreement", "stakeholders", "voluntary",
                  "national", "legislation", "regulation"]

VOCAB = {}
for w in REGISTER_WORDS:
    VOCAB.setdefault(w, "register")
for w in TOPIC_WORDS:
    VOCAB.setdefault(w, "topic")
for w in NEUTRAL_WORDS:
    VOCAB.setdefault(w, "neutral")
for w in RESEARCH_LEANING:
    VOCAB.setdefault(w, "research-lean")
for w in POLICY_LEANING:
    VOCAB.setdefault(w, "policy-lean")

WORDS = list(VOCAB.keys())


def main():
    G = np.load(G_PATH).astype(np.float32)        # (K, d), rows unit-norm
    K, d = G.shape
    print(f"Loaded G: {K} directions x {d} dims")

    model = SentenceTransformer(MODEL_ID)
    # Encode vocabulary as single-word "sentences"; mean-pooled embedding is the
    # word's representation in the SAME frozen space G lives in.
    WE = model.encode(WORDS, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float32)   # (V, d)
    WE = WE / np.linalg.norm(WE, axis=1, keepdims=True)
    print(f"Encoded {WE.shape[0]} vocabulary words")

    # cosine(g_k, word) = dot since both unit norm
    sims = G @ WE.T                                   # (K, V)
    word_idx = {w: i for i, w in enumerate(WORDS)}

    # ---- per-direction top words ----
    per_dir = []
    for k in range(K):
        order = np.argsort(-sims[k])
        top = [(WORDS[j], float(sims[k, j])) for j in order[:TOP_N]]
        per_dir.append({"k": k + 1, "top_words": top})

    # ---- aggregate: mean alignment of each word across all K directions ----
    mean_sim = sims.mean(axis=0)                      # (V,)
    agg = sorted(
        [{"word": WORDS[i], "bin": VOCAB[WORDS[i]], "mean_cosine": float(mean_sim[i])}
         for i in range(len(WORDS))],
        key=lambda r: -r["mean_cosine"],
    )

    # ---- bin means: do removed directions track register or topic? ----
    from collections import defaultdict
    bin_acc = defaultdict(list)
    for i, w in enumerate(WORDS):
        bin_acc[VOCAB[w]].append(float(mean_sim[i]))
    bin_means = {b: float(np.mean(v)) for b, v in bin_acc.items()}

    # ---- corpus-leaning signal per direction (research vs policy probes) ----
    rl_idx = [word_idx[w] for w in RESEARCH_LEANING]
    pl_idx = [word_idx[w] for w in POLICY_LEANING]
    corpus_track = []
    for k in range(K):
        rl = float(sims[k][rl_idx].mean())
        pl = float(sims[k][pl_idx].mean())
        corpus_track.append({"k": k + 1, "research_lean": rl, "policy_lean": pl,
                             "diff": rl - pl})

    # ---- report ----
    lines = []
    lines.append("# INLP register subspace — word inspection (MPNet canon)\n")
    lines.append(f"G = {K} orthonormalized removed directions (x {d}). "
                 "Each direction is the corpus-discrimination vector from one "
                 "SDG-stratified INLP iteration, orthogonalized against earlier ones.\n")
    lines.append("## Per-direction top words (first 6 removed directions)\n")
    for pd in per_dir[:6]:
        wt = ", ".join(f"{w} ({s:+.2f})" for w, s in pd["top_words"][:TOP_N])
        lines.append(f"- **iter {pd['k']}**: {wt}")
    lines.append("")
    lines.append("## Aggregate: words most aligned with the removed subspace (mean cosine over all iters)\n")
    for r in agg[:30]:
        lines.append(f"- {r['word']} [{r['bin']}] {r['mean_cosine']:+.3f}")
    lines.append("")
    lines.append("## Bin means (does the removed subspace track register or topic?)\n")
    for b, m in sorted(bin_means.items(), key=lambda x: -x[1]):
        lines.append(f"- {b}: {m:+.3f}")
    lines.append("")
    lines.append("## Corpus-leaning per direction (research-probe minus policy-probe mean cosine)\n")
    for ct in corpus_track[:10]:
        lines.append(f"- iter {ct['k']}: research {ct['research_lean']:+.3f}, "
                     f"policy {ct['policy_lean']:+.3f}, diff {ct['diff']:+.3f}")
    lines.append("")

    out_md = "\n".join(lines)
    print(out_md)

    with open("/home/manh/dissertation/5_notes/read_subspace/read_subspace_report.md", "w") as f:
        f.write(out_md)
    with open("/home/manh/dissertation/5_notes/read_subspace/read_subspace_topwords.json", "w") as f:
        json.dump({"G_shape": [K, d], "per_direction": per_dir, "aggregate_top": agg,
                   "bin_means": bin_means, "corpus_track": corpus_track}, f, indent=2)
    print("\nWROTE 5_notes/read_subspace/read_subspace_report.md + .json")


if __name__ == "__main__":
    main()
