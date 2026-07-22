"""
Train a multi-label SDG classifier using a shared PyTorch MLP.

Architecture:
  One network with shared hidden layers → 17 sigmoid outputs (binary relevance).
  This is the architecturally correct multi-label approach — unlike
  MultiOutputClassifier (17 independent models), hidden representations
  are shared across all SDGs.

   384 → [Linear → BN → ReLU → Dropout] × n_layers → 17

Training:
   - AdamW optimizer, BCEWithLogitsLoss
   - Internal 90/10 validation split per fold for early stopping (patience=7)
   - Restores best weights after early stop

CV strategy:
   - Random 5-fold (shuffled)
   - Scoring: macro F1
   - Full factorial grid: n_layers, hidden_size, lr, weight_decay, dropout

Inputs:
  2_data/2b_supervised/embeddings.npy
  2_data/2b_supervised/labels.npy
  2_data/2b_supervised/indices/train.npy

Outputs:
  2_data/2b_supervised/model/mlp_classifier.joblib  (best estimator)
  2_data/2b_supervised/model/mlp_cv_results.json     (full CV results)

Run from project root:
    python 1_code/2b_supervised_training/1_train_models_MLP.py
"""

import argparse
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = "2_data/2b_supervised"
OUTPUT_DIR = Path(DEFAULT_DATA_DIR) / "model"
MODEL_TAG = "mlp"
N_SDG = 17

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class MultiLabelMLP(BaseEstimator, ClassifierMixin):
    """Shared multi-label MLP with sklearn-compatible interface.

    Single network with shared hidden layers and 17 sigmoid outputs.
    Internal 90/10 validation split for early stopping within each fold.

    Works with GridSearchCV out of the box via BaseEstimator/ClassifierMixin.
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_size: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 7,
        random_state: int = 42,
        input_dim: int | None = None,
    ):
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.input_dim = input_dim

    def _build_network(self) -> nn.Module:
        input_dim = self.input_dim if self.input_dim is not None else 384
        layers = []
        for i in range(self.n_layers):
            in_dim = input_dim if i == 0 else self.hidden_size
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_size, N_SDG))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiLabelMLP":
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))

        # Internal 90/10 split for early stopping
        n_val = max(1, int(len(X) * 0.1))
        perm = torch.randperm(len(X_t), generator=torch.Generator().manual_seed(self.random_state))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_tr, X_val = X_t[train_idx], X_t[val_idx]
        y_tr, y_val = y_t[train_idx], y_t[val_idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("      MLP device: %s", device)
        log.info("      internal split: %d train, %d val", len(X_tr), len(X_val))

        self.net_ = self._build_network().to(device)
        self.criterion_ = nn.BCEWithLogitsLoss()
        self.optimizer_ = optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        train_ds = TensorDataset(X_tr.to(device), y_tr.to(device))
        val_ds = TensorDataset(X_val.to(device), y_val.to(device))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2)

        best_val_f1 = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.net_.train()
            for batch_X, batch_y in train_loader:
                self.optimizer_.zero_grad()
                loss = self.criterion_(self.net_(batch_X), batch_y)
                loss.backward()
                self.optimizer_.step()

            self.net_.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.net_(batch_X)
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_true.append(batch_y.cpu().numpy())

            val_preds = np.vstack(all_preds)
            val_true = np.vstack(all_true)
            val_f1 = f1_score(val_true, val_preds > 0.5, average="macro", zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info("      early stop at epoch %d (best val F1=%.4f)", epoch + 1, best_val_f1)
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_f1_ = float(best_val_f1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = next(self.net_.parameters()).device
        self.net_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(self.net_(X_t))
        return probs.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return f1_score(y, self.predict(X), average="macro", zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-label MLP.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Data dir (default: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "model"

    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)

    param_grid = {
        "n_layers": [4, 8, 16],
        "hidden_size": [256, 384],
        "lr": [0.001],
        "weight_decay": [0],
        "dropout": [0.3],
    }
    keys, vals = list(param_grid.keys()), list(param_grid.values())
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_scores = []
    best_score = -1.0
    best_clf = None
    best_params = None

    n_combos = np.prod([len(v) for v in vals])
    log.info("Starting MLP grid (%d combos × 5 folds)", n_combos)

    for combo in product(*vals):
        params = dict(zip(keys, combo))
        fold_scores = []

        for fold, (tr_i, val_i) in enumerate(cv.split(X)):
            t1 = time.perf_counter()
            log.info("  Fold %d/5  %s", fold + 1, params)
            clf = MultiLabelMLP(random_state=42, **params)
            clf.fit(X[tr_i], Y[tr_i])
            preds = clf.predict(X[val_i])
            f1 = f1_score(Y[val_i], preds, average="macro", zero_division=0)
            fold_scores.append(f1)
            log.info("  Fold %d/5: macro-F1=%.4f  [%.1fs]", fold + 1, f1, time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        all_scores.append({"params": params, "mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores})
        log.info("  %s  →  %.4f ± %.4f", params, mean_f1, std_f1)

        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_clf = clf

    elapsed = time.perf_counter() - t0
    log.info("Best: %s  macro-F1=%.4f ± %.4f  [%.1fs]", best_params, best_score,
             next(s["std_f1"] for s in all_scores if s["params"] == best_params), elapsed)

    # Summary table
    sorted_scores = sorted(all_scores, key=lambda x: x["mean_f1"], reverse=True)
    lines = ["", "=" * 70, "  MLP Grid Results (sorted by mean macro-F1)", "=" * 70]
    header = f"  {'n_layers':<9} {'hidden':<7} {'lr':<8} {'wd':<9} {'dropout':<8} {'mean F1':<8} {'±σ':<6}"
    sep = "  " + "-" * (len(header) - 2)
    lines.extend([header, sep])
    for s in sorted_scores:
        p = s["params"]
        lines.append(
            f"  {p['n_layers']:<9} {p['hidden_size']:<7} "
            f"{p['lr']:<8} {p['weight_decay']:<9} {p['dropout']:<8} "
            f"{s['mean_f1']:<8.4f} {s['std_f1']:<6.4f}"
        )
    lines.append(sep)
    lines.append(f"  Best: {best_params}  macro-F1={best_score:.4f} ± "
                 f"{next(s['std_f1'] for s in all_scores if s['params'] == best_params):.4f}")
    lines.append("=" * 70)
    log.info("\n%s", "\n".join(lines))

    results = {
        "model": MODEL_TAG,
        "best_params": {k: v for k, v in best_params.items()},
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": next(s["std_f1"] for s in all_scores if s["params"] == best_params),
        "per_fold_macro_f1": next(s["per_fold"] for s in all_scores if s["params"] == best_params),
        "elapsed_seconds": elapsed,
        "all_cv_results": all_scores,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    model_path = output_dir / f"{MODEL_TAG}_classifier.joblib"
    results_path = output_dir / f"{MODEL_TAG}_cv_results.json"

    joblib.dump(best_clf, model_path)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved model → %s", model_path)
    log.info("Saved results → %s", results_path)

    canonical_path = output_dir / "sdg_classifier.joblib"
    canonical_results_path = output_dir / "sdg_cv_results.json"
    joblib.dump(best_clf, canonical_path)
    with canonical_results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Canonical model → %s", canonical_path)
    log.info("Canonical results → %s", canonical_results_path)

    print(f"\nMLP done. {elapsed:.0f}s  Best: {best_params}  macro-F1: {best_score:.4f} ± {results['best_cv_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
