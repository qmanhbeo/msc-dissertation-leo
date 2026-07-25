# Single-Label Classifier Results

## Model comparison

| Pipeline | Model | CV macro-F1 | ±σ | Time |
|----------|-------|-------------|-----|------|
| Multi-label | LR (OvR) | 0.566 | 0.004 | 136s |
| Multi-label | RF | 0.398 | 0.006 | 94s |
| Multi-label | XGB | 0.571 | 0.004 | 147s |
| Multi-label | MLP | 0.692 | 0.006 | 156s |
| **Single-label** | **LR (multinomial)** | **0.779** | **0.005** | **300s** |
| **Single-label** | **MLP** | **0.799** | **0.002** | **36s** |

## Data

- Total: 42,626 texts (5 sources, canon embeddings + multi-label quality filters)
- Train: 36,229 | Test: 6,397 (untouched)
- Aurora dropped 1,597 short texts (MIN_WORDS=20 filter)
- Per-source stratified 85/15 split

## Best single-label params

- Model: MLP (BCEWithLogitsLoss, 17 sigmoid outputs)
- n_layers=4, hidden_size=384, lr=0.001, weight_decay=0, dropout=0.3
- Saved to: `2_data/2b_supervised_singlelabel/model/sdg_classifier.joblib`

## Key takeaway

Single-label is dramatically easier than multi-label (+0.107 F1). Even linear regression
(0.779) beats the best multi-label MLP (0.692). A small MLP grid could push further,
but 0.80 is the practical ceiling for this data.
