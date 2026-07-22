"""Load single-label MPNet MLP, evaluate on held-out test set, print per-SDG F1."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import f1_score

# Import the MLP class without triggering __main__ block
mlp_path = Path("1_code/2b_supervised_training_singlelabel/1_train_models_MLP.py")
spec = importlib.util.spec_from_file_location("mlp_mod", mlp_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiLabelMLP = mod.MultiLabelMLP

DATA_DIR = Path("2_data/2b_supervised_singlelabel_mpnet")
MODEL_PATH = DATA_DIR / "model" / "mlp_classifier.joblib"

embeddings = np.load(DATA_DIR / "embeddings.npy")
labels_onehot = np.load(DATA_DIR / "labels.npy")
test_idx = np.load(DATA_DIR / "indices" / "test.npy")

X_test = embeddings[test_idx]
Y_onehot = labels_onehot[test_idx]
Y_int = Y_onehot.argmax(axis=1) + 1

clf = joblib.load(MODEL_PATH)
preds_onehot = clf.predict(X_test)
preds_int = preds_onehot.argmax(axis=1) + 1

labels_17 = list(range(1, 18))
per_sdg = f1_score(Y_int, preds_int, average=None, labels=labels_17, zero_division=0)
macro = float(f1_score(Y_int, preds_int, average="macro", labels=labels_17, zero_division=0))

print()
print("=" * 60)
print("SINGLE-LABEL MPNet MLP — TEST SET PER-SDG F1")
print("=" * 60)
print(f"  Test size: {len(X_test)}")
print(f"  Macro-F1 : {macro:.4f}")
print()

sorted_idx = np.argsort(per_sdg)
print(f"  {'SDG':<5} {'F1':<8}  {'n_test':<7}")
print("  " + "-" * 25)
for i in sorted_idx:
    sdg = i + 1
    n = int((Y_int == sdg).sum())
    print(f"  SDG {sdg:<2d}  {per_sdg[i]:.4f}  n={n:<4d}")

result = {
    "model": "single-label MPNet MLP",
    "test_size": len(X_test),
    "macro_f1": round(macro, 6),
    "per_sdg_f1": {str(sdg): round(float(per_sdg[i]), 6) for i, sdg in enumerate(labels_17)},
}
Path("tmp_inspect_mlp_per_sdg.json").write_text(json.dumps(result, indent=2))
print(f"\nSaved: tmp_inspect_mlp_per_sdg.json")
