"""
Retrain the champion classifier on the FULL train pool (indices/train.npy),
then evaluate on the held-out test set (indices/test.npy).

Champion (final, LR): C=10.0, penalty=l2, class_weight=None, solver=lbfgs
Champion (prior, MLP): n_layers=4, hidden_size=384, lr=1e-3, wd=0, dropout=0.3

Inputs:
  {data_dir}/embeddings.npy
  {data_dir}/labels.npy
  {data_dir}/indices/train.npy
  {data_dir}/indices/test.npy

Outputs:
  {data_dir}/model/sdg_classifier_retrained.joblib   (used by scoring pipeline)
  {data_dir}/model/sdg_classifier.joblib              (used by 2_evaluate.py)
  {data_dir}/model/sdg_retrain_results.json

Run from project root:
    python 1_code/4_supervised_model_train/1_retrain_full_data.py --model all-mpnet-base-v2
    python 1_code/4_supervised_model_train/1_retrain_full_data.py --model all-mpnet-base-v2 --classifier-type lr
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, N_SDG, model_results_dir_for_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim: int, n_layers: int = 4, hidden_size: int = 384,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, N_SDG))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _NetWrapper:
    """Pickle-friendly wrapper for the retrained PyTorch model."""
    def __init__(self, net, input_dim):
        self.net = net
        self.input_dim = input_dim
    def predict_proba(self, X):
        self.net.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(self.net(X_t))
        return probs.cpu().numpy()
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain champion MLP on full train pool.")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                        help=f"Embed model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--data-dir", default=None,
                        help="Override data dir (derived from --model if omitted)")
    parser.add_argument("--classifier-type", default="lr", choices=["mlp", "lr"],
                        help="Classifier family (default: lr)")
    # MLP-specific args
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else model_results_dir_for_model(args.model)
    output_dir = data_dir / "model"

    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")
    test_idx = np.load(data_dir / "indices" / "test.npy")

    X_train = embeddings[train_idx]
    Y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    Y_test = labels[test_idx]

    input_dim = X_train.shape[1]
    log.info("Train: %d  Test: %d  dims: %d", len(X_train), len(X_test), input_dim)

    # ── Train ──────────────────────────────────────────────────────────
    t1 = time.perf_counter()

    if args.classifier_type == "mlp":
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        log.info("Config: n_layers=%d hidden=%d lr=%g wd=%g dropout=%g",
                 args.n_layers, args.hidden_size, args.lr, args.weight_decay, args.dropout)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)

        net = MultiLabelMLP(input_dim, args.n_layers, args.hidden_size, args.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        X_t = torch.from_numpy(X_train.astype(np.float32))
        Y_t = torch.from_numpy(Y_train.astype(np.float32))

        # Stage 1: train with 90/10 validation split to find best epoch
        n_val = max(1, int(len(X_t) * 0.1))
        perm = torch.randperm(len(X_t), generator=torch.Generator().manual_seed(42))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, X_val = X_t[tr_idx], X_t[val_idx]
        Y_tr, Y_val = Y_t[tr_idx], Y_t[val_idx]

        train_ds = TensorDataset(X_tr.to(device), Y_tr.to(device))
        val_ds = TensorDataset(X_val.to(device), Y_val.to(device))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2)

        best_val_f1 = -1.0
        patience_counter = 0
        best_epoch = 0
        best_state = None

        for epoch in range(args.max_epochs):
            net.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            net.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = net(batch_X)
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_true.append(batch_y.cpu().numpy())

            val_preds = np.vstack(all_preds)
            val_true = np.vstack(all_true)
            val_f1 = f1_score(val_true, val_preds > 0.5, average="macro", zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log.info("Early stop at epoch %d (best val F1=%.4f at epoch %d)", epoch + 1, best_val_f1, best_epoch)
                    break

        # Stage 2: retrain from scratch on 100% of training data for best_epoch epochs
        log.info("Retraining on full training pool (%d docs) for %d epochs...", len(X_t), best_epoch)
        net = MultiLabelMLP(input_dim, args.n_layers, args.hidden_size, args.dropout).to(device)
        optimizer2 = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        full_loader = DataLoader(
            TensorDataset(X_t.to(device), Y_t.to(device)),
            batch_size=args.batch_size, shuffle=True,
        )
        for epoch in range(best_epoch):
            net.train()
            for batch_X, batch_y in full_loader:
                optimizer2.zero_grad()
                loss = criterion(net(batch_X), batch_y)
                loss.backward()
                optimizer2.step()
        net.eval()

        train_time = time.perf_counter() - t1
        log.info("Training done: %.1fs  best val macro-F1=%.4f  full-retrain epochs=%d", train_time, best_val_f1, best_epoch)

    else:
        log.info("Config: C=10.0 penalty=l2 class_weight=None solver=lbfgs")

        y_int_train = Y_train.argmax(axis=1)
        clf = LogisticRegression(
            C=10.0, penalty="l2", solver="lbfgs",
            class_weight=None, max_iter=1000, random_state=42,
        )
        clf.fit(X_train, y_int_train)

        train_time = time.perf_counter() - t1
        log.info("Training done: %.1fs", train_time)

    # ── Evaluate on test set ───────────────────────────────────────────
    if args.classifier_type == "mlp":
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        with torch.no_grad():
            test_logits = net(X_test_t)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(np.float32)
    else:
        preds_int = clf.predict(X_test)
        test_preds = np.zeros((len(preds_int), N_SDG), dtype=np.float32)
        test_preds[np.arange(len(preds_int)), preds_int] = 1.0

    test_macro_f1 = f1_score(Y_test, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(Y_test, test_preds, average="micro", zero_division=0)

    per_sdg = {}
    for sdg in range(N_SDG):
        y_true_s = Y_test[:, sdg]
        y_pred_s = test_preds[:, sdg]
        f1_s = f1_score(y_true_s, y_pred_s, zero_division=0)
        per_sdg[f"SDG_{sdg+1}"] = round(float(f1_s), 4)

    total_elapsed = time.perf_counter() - t0

    log.info("Test macro-F1=%.4f  micro-F1=%.4f", test_macro_f1, test_micro_f1)
    for sdg_label, f1_s in per_sdg.items():
        log.info("  %s: %.4f", sdg_label, f1_s)

    # ── Save ───────────────────────────────────────────────────────────
    if args.classifier_type == "mlp":
        config = {
            "classifier_type": "mlp",
            "n_layers": args.n_layers,
            "hidden_size": args.hidden_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
        }
        training_info = {
            "device": str(device),
            "best_val_macro_f1": round(float(best_val_f1), 4),
            "train_seconds": round(train_time, 1),
            "total_seconds": round(total_elapsed, 1),
        }
    else:
        config = {
            "classifier_type": "lr",
            "C": 10.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": None,
        }
        training_info = {
            "train_seconds": round(train_time, 1),
            "total_seconds": round(total_elapsed, 1),
        }

    results = {
        "config": config,
        "data": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "input_dim": input_dim,
        },
        "training": training_info,
        "test_results": {
            "macro_f1": round(float(test_macro_f1), 4),
            "micro_f1": round(float(test_micro_f1), 4),
            "per_sdg_f1": per_sdg,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    if args.classifier_type == "mlp":
        wrapper = _NetWrapper(net.cpu(), input_dim)
        model_path = output_dir / "mlp_retrained.joblib"
        results_path = output_dir / "mlp_retrain_results.json"
    else:
        wrapper = clf
        model_path = output_dir / "sdg_classifier_retrained.joblib"
        results_path = output_dir / "sdg_retrain_results.json"

    joblib.dump(wrapper, model_path)

    if args.classifier_type != "mlp":
        # Also update sdg_classifier.joblib (used by 2_evaluate.py)
        canonical_path = output_dir / "sdg_classifier.joblib"
        joblib.dump(wrapper, canonical_path)
        log.info("Saved canonical model → %s", canonical_path)

    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved retrained model → %s", model_path)
    log.info("Saved results → %s", results_path)

    print(f"\n{'='*70}")
    print(f"  RETRAIN COMPLETE — {args.classifier_type.upper()} on full train pool (n={len(X_train)})")
    if args.classifier_type == "mlp":
        print(f"  Internal val macro-F1: {best_val_f1:.4f}")
    print(f"  Test macro-F1: {test_macro_f1:.4f}")
    print(f"  Test micro-F1: {test_micro_f1:.4f}")
    print(f"{'='*70}")
    print(f"\n  Per-SDG F1 on held-out test (n={len(X_test)}):")
    print(f"  {'SDG':<7} {'F1':<8}")
    for sdg_label, f1_s in per_sdg.items():
        print(f"  {sdg_label:<7} {f1_s:<8.4f}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
