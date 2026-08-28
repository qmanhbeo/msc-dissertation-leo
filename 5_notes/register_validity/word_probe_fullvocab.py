"""P0-full — Unbiased full-vocabulary scan of the removed subspace (scratch).

Encodes a large English word list with the frozen MPNet encoder and ranks every
word by how much of its embedding lies in span(G): score = ||G w|| (G has
orthonormal rows, so this is the L2 norm of the projection onto the removed
subspace). This is the unbiased "what words are on the subspace" answer,
free of the curated-bias problem.

A random-orthonormal-direction null (same K=62) gives the chance level: a word's
projection onto a RANDOM 62-dim subspace has expected norm sqrt(62/768) ~= 0.284.
If G's top words exceed the random-direction top words, the subspace is
non-randomly aligned with specific vocabulary.

MUST run under tmux (encoding ~150-250k words exceeds the 120s tool timeout).
Writes to 5_notes/register_validity/.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

G_PATH = "/home/manh/dissertation/2_data/3b_register/mpnet/canon/G.npy"
MODEL_ID = "all-mpnet-base-v2"
SEED = 42
OUT = Path("/home/manh/dissertation/5_notes/register_validity")
OUT.mkdir(parents=True, exist_ok=True)


def load_vocab():
    words = set()
    # nltk English words
    try:
        from nltk.corpus import words as nltk_words
        for w in nltk_words.words():
            w = w.lower()
            if w.isalpha() and 2 <= len(w) <= 25:
                words.add(w)
    except Exception as e:
        print("nltk words unavailable:", e)
    # fallback / supplement: system dict
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    w = line.strip().lower()
                    if w.isalpha() and 2 <= len(w) <= 25:
                        words.add(w)
        except FileNotFoundError:
            pass
    return sorted(words)


def main():
    G = np.load(G_PATH).astype(np.float32)
    K, d = G.shape
    rng = np.random.default_rng(SEED)

    vocab = load_vocab()
    print(f"Vocab size: {len(vocab)}")

    model = SentenceTransformer(MODEL_ID)
    # encode in batches; GPU if available else CPU
    batch = 2048
    W = np.zeros((len(vocab), d), dtype=np.float32)
    for i in range(0, len(vocab), batch):
        chunk = vocab[i:i + batch]
        e = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        W[i:i + len(chunk)] = e
    print(f"Encoded {W.shape[0]} words -> {W.shape}")

    # projection norm onto span(G): ||G w|| since rows orthonormal
    proj_G = (W @ G.T) @ G            # (V, d)
    score_G = np.linalg.norm(proj_G, axis=1)   # (V,)

    # random-direction null: row-orthonormal (K,d) basis
    R = np.linalg.qr(rng.standard_normal((d, K)))[0][:, :K].T.astype(np.float32)
    proj_R = (W @ R.T) @ R
    score_R = np.linalg.norm(proj_R, axis=1)

    order_G = np.argsort(-score_G)
    order_R = np.argsort(-score_R)

    top_G = [(vocab[j], float(score_G[j])) for j in order_G[:300]]
    top_R = [(vocab[j], float(score_R[j])) for j in order_R[:300]]

    # calibration stats
    def stats(a):
        return {"max": float(a.max()), "p99": float(np.percentile(a, 99)),
                "mean": float(a.mean()), "median": float(np.median(a)),
                "exp_random": float(np.sqrt(K / d))}
    calib = {"G": stats(score_G), "random": stats(score_R)}

    lines = []
    lines.append("# P0-full — Unbiased full-vocabulary scan (MPNet canon)\n")
    lines.append(f"G = {K} x {d}. Word score = ||projection onto span(G)||. "
                 "Random-direction expected score = sqrt(K/d) = "
                 f"{calib['random']['exp_random']:.4f}.\n")
    lines.append("## Calibration: G vs random-direction subspace\n")
    lines.append(f"- G:    max={calib['G']['max']:.4f}, p99={calib['G']['p99']:.4f}, "
                 f"median={calib['G']['median']:.4f}")
    lines.append(f"- Rand: max={calib['random']['max']:.4f}, p99={calib['random']['p99']:.4f}, "
                 f"median={calib['random']['median']:.4f}")
    lines.append("\nIf G's top score is near the random top score, the subspace is not "
                 "preferentially aligned with any specific vocabulary (register reading "
                 "not supported at word level).\n")
    lines.append("## Top 200 words by alignment with the removed subspace\n")
    for i, (w, s) in enumerate(top_G[:200], 1):
        lines.append(f"{i:3d}. {w} {s:.4f}")
    lines.append("\n## Top 50 words by alignment with a RANDOM 62-dim subspace (null)\n")
    for i, (w, s) in enumerate(top_R[:50], 1):
        lines.append(f"{i:3d}. {w} {s:.4f}")

    out_md = "\n".join(lines)
    print(out_md)
    (OUT / "p0_fullvocab_scan.md").write_text(out_md)
    json.dump({"calib": calib, "top_G": top_G, "top_R": top_R,
               "vocab_size": len(vocab), "G_shape": [K, d]},
              open(OUT / "p0_fullvocab_scan.json", "w"), indent=2)
    print("\nWROTE p0_fullvocab_scan.md + .json")


if __name__ == "__main__":
    main()
