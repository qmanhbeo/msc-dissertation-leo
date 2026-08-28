# INLP algorithmic-parity check

This directory verifies that **our** INLP implementation
(`1_code/7_main_analysis/0_shared/register_adjust.py`, which builds the
register-subspace `G`) is **algorithmically equivalent** to the reference
implementation from Ravfogel et al. (2020), *Iterative Nullspace Projection*.

## Why this exists

`G` is the removed subspace whose interpretation ("register", not topic) the
dissertation defends. That defence is only meaningful if `G` is a *correct* INLP
projection. This check proves the loop equivalence on a controlled toy problem,
so "our `G` is a legitimate INLP projection" is a **verified** statement rather
than an assumption.

## What it checks (and what it does NOT)

It runs **both** implementations on one synthetic dataset (`K` orthogonal signal
dimensions inside `D` dimensions, strong signal so `K` real directions exist):

1. `our_inlp` — a faithful replica of `register_adjust.py`'s loop (project `X`
   onto the nullspace of the accumulated directions, fit a binary logistic
   regression, take `coef_` as the direction, Gram–Schmidt orthonormalise,
   accumulate).
2. Ravfogel's `get_debiasing_projection` — **vendored verbatim** under
   `vendor/` (see below).

It then compares:

- **principal angles** between `span(our_G)` and `span(their Ws)` — must be ~0
  (cosine ~1);
- **effect equivalence** — a fresh classifier's accuracy after projecting through
  OUR `G` vs THEIR final projection `P` — must match.

**This is a fidelity check only.** It does *not* address whether "register" is
the correct *label* for what `G` captures — that is the separate question
examined by the direction-space tests in `5_notes/register_validity/`.

## Vendored source (reproducibility)

`vendor/src/{debias,classifier}.py` are copied **verbatim** (only an attribution
header added) from:

- Repository: `https://github.com/shauli-ravfogel/nullspace_projection`
- Commit: `e1edcc19d808108ab71cbb3afb0389db0206a7eb` (2022-06-06)
- License: **MIT**, Copyright (c) 2022 Shauli Ravfogel

Vendoring (rather than a git submodule or a runtime fetch) keeps the check
offline, deterministic, and independent of the upstream repo's availability.
Do not edit the vendored files; if the reference implementation changes, re-vendor
at a pinned commit and update the commit hash above and in `run_parity.py`.

## Run

```bash
conda activate dissertation
python 1_code/_inlp_parity/run_parity.py
```

## Expected result

On the fixed toy problem the subspaces are identical and the removal effect
matches:

```
principal_angle_cosines: [1.0, 1.0, 1.0, 1.0, 1.0]
min_principal_angle_cosine: 1.0
fresh_acc_on_original:        0.815
fresh_acc_after_OUR_projection:   0.515
fresh_acc_after_THEIR_projection: 0.515
VERDICT: MATCH
```

The script exits non-zero if the subspaces diverge, so it can serve as a
regression guard.
