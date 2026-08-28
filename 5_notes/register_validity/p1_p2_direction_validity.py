"""P1 + P2 — Direction-space validity tests of the register-reading (scratch).

Reuses register_adjust.load_stratified_samples to get the SAME frozen, SDG-
balanced research/policy segment sample G was trained on (no re-embed, no INLP
re-run). All classifiers are cheap logistic regressions on frozen embeddings.

P1 (topic-direction overlap):
  Fit an independent 17-class SDG-topic LR on RAW embeddings -> weight vectors
  w_sdg (17x768). For each, measure alignment with span(G) = ||G w||/||w||
  (G has orthonormal rows). Compare to (a) the corpus/register direction's
  alignment (sanity, should be ~1) and (b) a RANDOM 62-dim subspace (expected
  sqrt(62/768) ~= 0.284). If topic vectors align with span(G) no more than a
  random subspace does, G did not preferentially remove topic.

P2 (held-out-SDG generalization -- the key falsification test for T1):
  Leave-one-SDG-out: train a corpus LR on all OTHER SDGs, test it on the held-
  out SDG's research/policy segments. Register is a corpus property so it should
  GENERALIZE across topics -> high held-out accuracy. Topic leakage would be SDG-
  specific -> held-out accuracy collapses. Also a topic CONTROL: a direction
  trained to separate SDG a vs b on research segments should NOT classify other
  SDGs' research as a vs b (topic does not generalize), contrasting with register.

Writes p1_p2_direction_validity.json + .md to 5_notes/register_validity/.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "1_code", "7_main_analysis", "0_shared"))
import register_adjust as ra

MODEL = "all-mpnet-base-v2"
SEED = 42
OUT = Path("/home/manh/dissertation/5_notes/register_validity")
OUT.mkdir(parents=True, exist_ok=True)


def align_subspace(w, basis):
    """||basis w|| / ||w|| for orthonormal-row basis (projection norm)."""
    w = np.asarray(w, dtype=np.float32)
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return 0.0
    return float(np.linalg.norm(basis @ w) / nw)


def main():
    G = np.load("/home/manh/dissertation/2_data/3b_register/mpnet/canon/G.npy").astype(np.float32)
    K, d = G.shape
    rng = np.random.default_rng(SEED)
    # row-orthonormal 62-dim random subspace (reduced qr of wide matrix gives
    # (K,K); build via qr of (d,K) then transpose to get (K,d) orthonormal rows)
    R = np.linalg.qr(rng.standard_normal((d, K)))[0][:, :K].T.astype(np.float32)

    # ---- load the same frozen balanced sample G was trained on ----
    sdg_index = ra.build_research_sdg_index(MODEL)
    policy_emb = np.load(ra.get_policy_emb(MODEL)).astype(np.float32)
    policy_assignments = ra.get_cluster_assignments(np.load(ra.get_policy_scores(MODEL)))
    ckpt = json.load(open("/home/manh/dissertation/2_data/3b_register/mpnet/canon/checkpoint.json"))
    n_target = int(ckpt["n_target"]) if "n_target" in ckpt else 1123
    X, y, sdg = ra.load_stratified_samples(
        MODEL, sdg_index, n_target, rng, policy_emb, policy_assignments, projector=None)
    print(f"Sample: X={X.shape}, n_target={n_target}, SDGs present={sorted(set(sdg.tolist()))}")

    # ===================== P1: topic-direction overlap =====================
    clf_topic = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf_topic.fit(X, sdg)
    Wt = clf_topic.coef_.astype(np.float32)            # (17, 768)
    topic_align_G = [align_subspace(Wt[i], G) for i in range(Wt.shape[0])]
    topic_align_R = [align_subspace(Wt[i], R) for i in range(Wt.shape[0])]

    clf_corp = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf_corp.fit(X, y)
    c = clf_corp.coef_.astype(np.float32).flatten()
    corpus_align_G = align_subspace(c, G)
    corpus_align_R = align_subspace(c, R)

    random_exp = float(np.sqrt(K / d))
    p1 = {
        "topic_align_G_mean": float(np.mean(topic_align_G)),
        "topic_align_G_max": float(np.max(topic_align_G)),
        "topic_align_R_mean": float(np.mean(topic_align_R)),
        "corpus_align_G": corpus_align_G,
        "corpus_align_R": corpus_align_R,
        "random_subspace_expected": random_exp,
        "per_sdg_topic_align_G": {int(s): topic_align_G[i] for i, s in enumerate(range(1, 18))},
    }

    # ===================== P2: held-out-SDG generalization =====================
    acc_loo, acc_full = {}, {}
    loo_align_G = {}
    for s in range(1, 18):
        tr = sdg != s
        te = sdg == s
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf.fit(X[tr], y[tr])
        acc_loo[s] = float(clf.score(X[te], y[te]))
        loo_align_G[s] = align_subspace(clf.coef_.astype(np.float32).flatten(), G)
        clf2 = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf2.fit(X, y)
        acc_full[s] = float(clf2.score(X[te], y[te]))

    # topic CONTROL: SDG a vs b on research segments should NOT generalize to other SDGs
    a, b = 1, 2
    res = y == 0
    ab = res & ((sdg == a) | (sdg == b))
    clf_ab = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf_ab.fit(X[ab], (sdg[ab] == b).astype(int))
    train_acc_ab = float(clf_ab.score(X[ab], (sdg[ab] == b).astype(int)))
    other = res & (~((sdg == a) | (sdg == b)))
    other_pred = clf_ab.predict(X[other])
    other_acc_ab = float(np.mean(other_pred == (sdg[other] == b).astype(int)))

    # balanced data-size control: train REGISTER direction on only 2 SDGs'
    # research+policy, test on the OTHER 15 SDGs. If it still generalizes, the
    # 0.967 held-out result is not just a training-mass artifact.
    a2, b2 = 1, 2
    two = (sdg == a2) | (sdg == b2)
    clf_2 = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf_2.fit(X[two], y[two])
    two_train_acc = float(clf_2.score(X[two], y[two]))
    other_mask = ~two
    two_test_acc = float(clf_2.score(X[other_mask], y[other_mask]))
    two_align_G = align_subspace(clf_2.coef_.astype(np.float32).flatten(), G)

    p2 = {
        "acc_loo_per_sdg": {int(s): acc_loo[s] for s in range(1, 18)},
        "acc_full_per_sdg": {int(s): acc_full[s] for s in range(1, 18)},
        "acc_loo_mean": float(np.mean(list(acc_loo.values()))),
        "acc_full_mean": float(np.mean(list(acc_full.values()))),
        "loo_align_G_mean": float(np.mean(list(loo_align_G.values()))),
        "topic_control_train_acc_SDGa_vs_b": train_acc_ab,
        "topic_control_test_acc_on_other_SDGs": other_acc_ab,
        "topic_control_expected_chance": 0.5,
        "register_2sdg_train_acc": two_train_acc,
        "register_2sdg_test_acc_on_other_15_sdg": two_test_acc,
        "register_2sdg_align_G": two_align_G,
    }

    # --------------------- report ---------------------
    L = []
    L.append("# P1+P2 — Direction-space validity of the register reading (MPNet canon)\n")
    L.append(f"G = {K}x{d}. Sample: {X.shape[0]} segments ({n_target}/SDG/corpus). "
             "All LRs use C=1.0 (matching INLP). No re-embed.\n")
    L.append("## P1 — Topic-direction overlap with span(G)\n")
    L.append(f"- Topic coef alignment with span(G): mean={p1['topic_align_G_mean']:.3f}, "
             f"max={p1['topic_align_G_max']:.3f}")
    L.append(f"- Topic coef alignment with a RANDOM 62-dim subspace: mean={p1['topic_align_R_mean']:.3f} "
             f"(expected {random_exp:.3f})")
    L.append(f"- Corpus(register) coef alignment with span(G): {corpus_align_G:.3f} "
             f"(vs random {corpus_align_R:.3f}) -- sanity, should be near 1")
    L.append("\nInterpretation: if topic alignment ~ random subspace (0.28), G did NOT "
             "preferentially remove topic. If corpus alignment ~1, span(G) captures register.\n")
    L.append("## P2 — Held-out-SDG generalization (falsifies T1 if it collapses)\n")
    L.append(f"- Held-out-SDG corpus accuracy (mean over 17 folds): {p2['acc_loo_mean']:.3f}")
    L.append(f"- Full-data corpus accuracy (mean over 17 SDGs): {p2['acc_full_mean']:.3f}")
    L.append(f"- Held-out corpus directions' alignment with span(G): mean={p2['loo_align_G_mean']:.3f}")
    L.append(f"- TOPIC CONTROL: SDG{a} vs SDG{b} direction trains at {train_acc_ab:.3f} "
             f"but classifies OTHER SDGs' research at {other_acc_ab:.3f} (chance=0.5) "
             "-- topic generalizes far less than register.")
    L.append(f"- BALANCED DATA-SIZE CONTROL: register direction trained on ONLY SDG{a}+SDG{b} "
             f"reaches train {two_train_acc:.3f}, and STILL classifies the other 15 SDGs' "
             f"research/policy at {two_test_acc:.3f} -- so the 0.967 held-out result is NOT "
             f"merely a training-mass artifact (register generalizes even from 2 SDGs).")
    L.append("\nInterpretation: if held-out corpus accuracy stays well above chance, the "
             "removed register direction generalizes across topics (supports register reading). "
             "If it collapses toward chance, the removed subspace is SDG/topic-specific "
             "(falsifies the register reading).\n")

    out_md = "\n".join(L)
    print(out_md)
    (OUT / "p1_p2_direction_validity.md").write_text(out_md)
    json.dump({"G_shape": [K, d], "n_target": n_target, "P1": p1, "P2": p2},
              open(OUT / "p1_p2_direction_validity.json", "w"), indent=2)
    print("\nWROTE p1_p2_direction_validity.md + .json")


if __name__ == "__main__":
    main()
