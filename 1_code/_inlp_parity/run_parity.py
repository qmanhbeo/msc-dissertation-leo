"""INLP algorithmic-parity check (canonical, reproducible).

Purpose
-------
Our register-subspace ``G`` is built by ``1_code/7_main_analysis/0_shared/
register_adjust.py`` — our own re-implementation of Iterative Nullspace
Projection (INLP; Ravfogel et al., 2020). This script proves the
*algorithmic* equivalence of our loop to Ravfogel's reference
``get_debiasing_projection`` on a controlled toy problem, so that "our G is a
legitimate INLP projection" is a verified statement, not an assumption.

What it does
------------
On ONE synthetic dataset (``K`` orthogonal signal dimensions embedded in ``D``
dimensions, strong signal so ``K`` real directions exist), it runs:
  * ``our_inlp`` — a faithful replica of ``register_adjust.py``'s loop
    (project X onto the nullspace of the accumulated directions, fit a binary
    logistic regression, take ``coef_`` as the direction, Gram–Schmidt
    orthonormalise, accumulate),
  * Ravfogel's ``get_debiasing_projection`` (vendored verbatim under
    ``vendor/``, ``is_autoregressive=True``, ``min_accuracy=0.0``), imported
    from that vendored copy (no network, no dependency on any external clone).

It then compares:
  (a) principal angles between ``span(our_G)`` and ``span(their Ws)`` — must be
      ~0 (cosine ~1);
  (b) effect equivalence — a fresh classifier's accuracy after projecting
      through OUR G vs THEIR final projection P — must match.

This is a fidelity check ONLY. It does NOT bear on whether "register" is the
right *label* for what G captures (that is a separate question, addressed by
the direction-space tests in ``5_notes/register_validity/``).

Outputs
-------
  parity_result.json  — all numbers (machine-readable)
and exits non-zero if the subspaces diverge (a usable regression guard).

Run
---
  conda activate dissertation
  python 1_code/_inlp_parity/run_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# ---- named constants (no magic numbers) --------------------------------------
SEED = 42
D = 40            # embedding dimensionality of the toy problem
N = 1200          # number of toy points
K = 5             # number of separating directions to remove

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
sys.path.insert(0, str(VENDOR))          # so `from src.debias import ...` resolves
from src.debias import get_debiasing_projection  # noqa: E402


def make_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    S = rng.standard_normal((K, D))
    S, _ = np.linalg.qr(S.T)             # (D, K) orthonormal columns
    S = S.T                              # (K, D) orthonormal rows
    X = rng.standard_normal((N, D))
    feats = np.sign(X @ S.T * 3.0)       # (N, K) crisp +-1 features
    y = (feats.sum(axis=1) >= 0).astype(int)   # needs all K dims -> K real directions
    return X, y, S


def our_inlp(X: np.ndarray, y: np.ndarray, k: int = K, seed: int = SEED) -> np.ndarray:
    """Faithful replica of register_adjust.py's INLP loop (projector=G_prev + GS)."""
    rng = np.random.default_rng(seed)
    G_list: list[np.ndarray] = []
    for it in range(1, k + 1):
        G_prev = np.vstack(G_list) if G_list else None
        if G_prev is not None and G_prev.shape[0] > 0:
            Xp = X - (X @ G_prev.T) @ G_prev           # nullspace projection
        else:
            Xp = X
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=int(seed + it))
        clf.fit(Xp, y)
        g = clf.coef_.flatten().astype(np.float32)
        g = g / np.linalg.norm(g)
        for prev in G_list:                            # Gram-Schmidt
            g = g - np.dot(g, prev) * prev
        g = g / np.linalg.norm(g)
        G_list.append(g)
    return np.vstack(G_list)                            # (K, D) orthonormal rows


def principal_angle_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """cos of principal angles between row-orthonormal U and V (both (K,D))."""
    return np.linalg.svd(U @ V.T, compute_uv=False)


def fresh_acc(X: np.ndarray, y: np.ndarray) -> float:
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    return float(clf.fit(X, y).score(X, y))


def main() -> int:
    X, y, _ = make_toy()
    our_G = our_inlp(X, y, K)

    cls_params = {"C": 1.0, "max_iter": 2000, "random_state": SEED}
    P, _rowspace_projections, Ws = get_debiasing_projection(
        LogisticRegression, cls_params, K, D,
        True,   # is_autoregressive
        0.0,    # min_accuracy (never skip)
        X, y, X, y,
        by_class=False)

    Wmat = np.vstack([w.reshape(1, -1) for w in Ws]).astype(np.float32)   # (K, D)
    V, _ = np.linalg.qr(Wmat.T)        # (D, K) orthonormal columns
    V = V.T                            # (K, D) orthonormal rows = their rowspace basis

    angles = principal_angle_cosines(our_G, V)
    min_cos = float(np.min(angles))

    acc_orig = fresh_acc(X, y)
    X_our = X - (X @ our_G.T) @ our_G
    acc_our = fresh_acc(X_our, y)
    X_their = (P @ X.T).T
    acc_their = fresh_acc(X_their, y)

    result = {
        "source_commit": "e1edcc19d808108ab71cbb3afb0389db0206a7eb",
        "K": K, "D": D, "N": N, "seed": SEED,
        "principal_angle_cosines": [round(float(a), 6) for a in angles],
        "min_principal_angle_cosine": round(min_cos, 6),
        "fresh_acc_on_original": round(acc_orig, 4),
        "fresh_acc_after_OUR_projection": round(acc_our, 4),
        "fresh_acc_after_THEIR_projection": round(acc_their, 4),
        "Ws_len": len(Ws),
    }
    (HERE / "parity_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    verdict = "MATCH" if min_cos > 0.999 else ("CLOSE" if min_cos > 0.99 else "DIVERGE")
    print(f"\nVERDICT (min principal-angle cosine = {min_cos:.6f}): {verdict}")
    if verdict == "MATCH":
        print("PASS: our INLP loop is algorithmically equivalent to Ravfogel's "
              "get_debiasing_projection on this toy problem.")
        return 0
    print("FAIL: subspaces diverged — investigate register_adjust.py vs vendored src.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
