"""P0 — Curated word probe with permutation null (MPNet-canonical, scratch only).

Answers: do the removed directions align more with REGISTER words than TOPIC
words? Uses the same curated bins as inspect_directions.py but adds a
permutation null so the register-vs-topic bin-mean gap is tested against chance.

Writes to 5_notes/register_validity/.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

G_PATH = "/home/manh/dissertation/2_data/3b_register/mpnet/canon/G.npy"
MODEL_ID = "all-mpnet-base-v2"
TOP_N = 15
SEED = 42
OUT = Path("/home/manh/dissertation/5_notes/register_validity")
OUT.mkdir(parents=True, exist_ok=True)

REGISTER_WORDS = [
    "may", "might", "could", "possibly", "suggests", "appears", "potentially",
    "likely", "perhaps", "indicate",
    "must", "shall", "should", "will", "need", "require", "expected",
    "we", "our", "us", "researchers", "authors",
    "implementation", "development", "framework", "strategy", "policy",
    "recommendation", "report", "countries", "nations", "global",
    "institutional", "governance", "partnership", "target", "commitment",
    "conducted", "performed", "established",
]
TOPIC_WORDS = [
    "climate", "energy", "health", "education", "water", "gender",
    "inequality", "innovation", "infrastructure", "cities", "consumption",
    "biodiversity", "oceans", "land", "peace", "poverty", "hunger",
    "sustainability", "sustainable",
    "neural", "model", "models", "algorithm", "learning", "deep",
    "dataset", "training", "classification", "prediction", "embedding",
]
NEUTRAL_WORDS = [
    "table", "figure", "number", "system", "result", "study", "data",
    "time", "group", "level", "process", "part", "value", "set", "case",
]
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
    G = np.load(G_PATH).astype(np.float32)
    K, d = G.shape

    model = SentenceTransformer(MODEL_ID)
    WE = model.encode(WORDS, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    WE = WE / np.linalg.norm(WE, axis=1, keepdims=True)

    sims = G @ WE.T  # (K, V) cosine since both unit norm
    mean_sim = sims.mean(axis=0)  # (V,)

    bin_acc = defaultdict(list)
    for i, w in enumerate(WORDS):
        bin_acc[VOCAB[w]].append(float(mean_sim[i]))
    bin_means = {b: float(np.mean(v)) for b, v in bin_acc.items()}

    reg_mean = bin_means["register"]
    top_mean = bin_means["topic"]
    diff = reg_mean - top_mean

    # ---- permutation null on register-vs-topic gap ----
    rng = np.random.default_rng(SEED)
    reg_idx = [i for i, w in enumerate(WORDS) if VOCAB[w] == "register"]
    top_idx = [i for i, w in enumerate(WORDS) if VOCAB[w] == "topic"]
    n_perm = 5000
    null = np.empty(n_perm)
    all_idx = reg_idx + top_idx
    m = len(reg_idx)
    for p in range(n_perm):
        perm = rng.permutation(all_idx)
        null[p] = np.mean(mean_sim[perm[:m]]) - np.mean(mean_sim[perm[m:]])
    p_val = float(np.mean(null >= diff))
    # two-sided
    p_two = float(np.mean(np.abs(null) >= abs(diff)))

    # ---- per-direction top words (aggregate signal) ----
    per_dir = []
    for k in range(K):
        order = np.argsort(-sims[k])
        per_dir.append({"k": k + 1,
                        "top_words": [(WORDS[j], float(sims[k, j])) for j in order[:TOP_N]]})

    lines = []
    lines.append("# P0 — Curated word probe + permutation null (MPNet canon)\n")
    lines.append(f"G = {K} orthonormal removed directions x {d}. Single-word embeddings "
                 "in the frozen MPNet space; cosine with each g_k.\n")
    lines.append("## Bin means (mean cosine of bin words with the removed subspace)\n")
    for b, m_ in sorted(bin_means.items(), key=lambda x: -x[1]):
        lines.append(f"- {b}: {m_:+.4f}")
    lines.append(f"\nRegister - Topic gap = {diff:+.4f}")
    lines.append(f"Permutation null (5000): p(>=obs) = {p_val:.3f}, two-sided p = {p_two:.3f}")
    lines.append("Interpretation: if the gap is not distinguishable from the permutation "
                 "null, the single-word probe does not support a register>topic reading.\n")
    lines.append("## Aggregate top words (mean cosine over all iters)\n")
    agg = sorted([{"word": WORDS[i], "bin": VOCAB[WORDS[i]], "mean_cosine": float(mean_sim[i])}
                  for i in range(len(WORDS))], key=lambda r: -r["mean_cosine"])
    for r in agg[:30]:
        lines.append(f"- {r['word']} [{r['bin']}] {r['mean_cosine']:+.4f}")
    lines.append("\n## First 3 directions' top words\n")
    for pd in per_dir[:3]:
        wt = ", ".join(f"{w} ({s:+.2f})" for w, s in pd["top_words"])
        lines.append(f"- iter {pd['k']}: {wt}")

    out_md = "\n".join(lines)
    print(out_md)
    (OUT / "p0_curated_probe.md").write_text(out_md)
    json.dump({"bin_means": bin_means, "register_minus_topic": diff,
               "perm_p_ge_obs": p_val, "perm_p_two_sided": p_two,
               "aggregate_top": agg, "per_direction": per_dir, "G_shape": [K, d]},
              open(OUT / "p0_curated_probe.json", "w"), indent=2)
    print("\nWROTE p0_curated_probe.md + .json")


if __name__ == "__main__":
    main()
