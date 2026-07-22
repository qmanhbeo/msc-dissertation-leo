# Supervised Model Comparison (5-fold CV macro-F1)

| Pipeline | Model | MiniLM (384d) | MPNet (768d) | Δ |
|----------|-------|--------------:|-------------:|---:|
| Single-label | LR | 0.779 ± 0.005 | **0.803** ± 0.004 | +0.024 |
| Single-label | MLP | 0.799 ± 0.002 | **0.816** ± 0.005 | +0.017 |
| Multi-label | LR | 0.566 ± 0.004 | **0.598** ± 0.004 | +0.032 |
| Multi-label | MLP | 0.692 ± 0.006 | **0.708** ± 0.006 | +0.016 |
| Zero-shot | centroid CV | 0.723 | 0.738 | +0.015 |

**Consistent pattern:** MPNet adds +0.015–0.032 across the board. Larger gain on LR (+0.024–0.032) than on MLP (+0.016–0.017), likely because MLP learns nonlinear projections that partially compensate for weaker embeddings.

**Single-label MPNet MLP (0.816)** is the overall best model. Trained with champion config (4/384/lr=1e-3/wd=0/dropout=0.3).
