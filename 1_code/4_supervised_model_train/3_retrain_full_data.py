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
  {data_dir}/model/sdg_retrain_results.json           (per-SDG F1, macro-F1)
  4_outputs/main/data/4_1_confusion_matrix.csv        (LR only: ROWS=pred, COLS=true)

Run from project root:
    python 1_code/4_supervised_model_train/3_retrain_full_data.py --model all-mpnet-base-v2
    python 1_code/4_supervised_model_train/3_retrain_full_data.py --model all-mpnet-base-v2 --classifier-type lr
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, RANDOM_SEED, model_results_dir_for_model, output_dir_for_model, resolve_model_alias
from shard_pipeline_utils import atomic_write_joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Champion LR hyperparameters (see module docstring for provenance).
LR_C = 10.0
LR_PENALTY = "l2"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000

# Champion MLP hyperparameters as selected by the manual grid-search CV
# (grid_search_log.json, 2026-07-25): 4 layers / 384 hidden / lr=3e-4 / wd=0 /
# dropout=0.3, CV macro-F1 0.8243. The argparse defaults below derive from this
# champion so the retrained MLP artifact (mlp_retrained.joblib +
# model_config.json) matches the dissertation text, which cites lr=3e-4.
MLP_CHAMPION_CONFIG = {
    "n_layers": 4,
    "hidden_size": 384,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
}

CV_N_SPLITS = 5


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
        return self.predict_proba(X).argmax(axis=1).astype(np.float32)


def _cv_mlp_fold(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: dict,
) -> tuple[float, list[float], float, int]:
    """Train an MLP on the fold-train portion with an internal 90/10 early-stop
    split, then evaluate the best-epoch state on the fold's held-out portion.

    Mirrors the canonical MLP training loop (stage 1 only) so the canonical
    retrain path is not refactored. Returns
    (macro_f1, per_sdg_f1, best_internal_val_f1, best_epoch).
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    input_dim = X_tr.shape[1]
    net = MultiLabelMLP(input_dim, cfg["n_layers"], cfg["hidden_size"], cfg["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    X_t = torch.from_numpy(X_tr.astype(np.float32))
    Y_t = torch.from_numpy(Y_tr.astype(np.float32))

    n_val = max(1, int(len(X_t) * 0.1))
    perm = torch.randperm(len(X_t), generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_ds = TensorDataset(X_t[tr_idx].to(device), Y_t[tr_idx].to(device))
    val_ds = TensorDataset(X_t[val_idx].to(device), Y_t[val_idx].to(device))
    _seed_loader = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=_seed_loader)
    val_loader = DataLoader(val_ds, batch_size=512)

    best_val_f1 = -1.0
    patience_counter = 0
    best_epoch = 0
    best_state = None

    for epoch in range(100):
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
                all_preds.append(torch.sigmoid(net(batch_X)).cpu().numpy())
                all_true.append(batch_y.cpu().numpy())

        val_preds = np.vstack(all_preds)
        val_true = np.vstack(all_true)
        val_pred_int = val_preds.argmax(axis=1)
        val_pred_bin = np.zeros_like(val_preds)
        val_pred_bin[np.arange(len(val_pred_int)), val_pred_int] = 1.0
        val_f1 = f1_score(val_true, val_pred_bin, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 7:
                break

    net.load_state_dict(best_state)
    net.eval()
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    with torch.no_grad():
        val_probs = torch.sigmoid(net(X_val_t)).cpu().numpy()
    val_pred_int = val_probs.argmax(axis=1)
    val_preds = np.zeros_like(val_probs)
    val_preds[np.arange(len(val_pred_int)), val_pred_int] = 1.0

    macro = float(f1_score(Y_val, val_preds, average="macro", zero_division=0))
    per_sdg = f1_score(Y_val, val_preds, average=None, zero_division=0).tolist()
    return macro, per_sdg, float(best_val_f1), best_epoch


def run_cv_full_data(data_dir: Path, classifier_type: str) -> None:
    """EXPLORATORY (manual only): GroupKFold CV on 100% of labelled data with
    the grid-search champion hyperparameters.

    Unlike the canonical path, which trains on indices/train.npy (85%) and
    evaluates once on indices/test.npy (15%), this uses ALL rows (train+test)
    in document-grouped K-fold CV and reports the mean macro-F1. It writes
    cv_full_data_results.json and saves/overwrites NO model artifacts.
    """
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")

    sd_path = data_dir / "source_docs.npy"
    if sd_path.exists():
        groups = np.load(sd_path)
        log.info("Document-grouped CV (%s): %d unique groups, %d rows",
                 sd_path.name, len(np.unique(groups)), len(embeddings))
    else:
        groups = None
        log.warning("source_docs.npy not found — falling back to ungrouped folds")

    if classifier_type == "mlp":
        cfg = dict(MLP_CHAMPION_CONFIG)
    else:
        cfg = {"C": LR_C, "penalty": LR_PENALTY, "solver": LR_SOLVER,
               "class_weight": None, "max_iter": LR_MAX_ITER}

    cv = GroupKFold(n_splits=CV_N_SPLITS)
    fold_scores: list[float] = []
    fold_per_sdg: list[list[float]] = []
    fold_n_train: list[int] = []
    fold_n_val: list[int] = []

    t0 = time.perf_counter()
    for fold, (tr_i, va_i) in enumerate(cv.split(embeddings, groups=groups)):
        if classifier_type == "mlp":
            macro, per_sdg, internal_val_f1, best_epoch = _cv_mlp_fold(
                embeddings[tr_i], labels[tr_i], embeddings[va_i], labels[va_i], cfg,
            )
            log.info("  Fold %d/%d: macro-F1=%.4f (internal val %.4f @ epoch %d)",
                     fold + 1, CV_N_SPLITS, macro, internal_val_f1, best_epoch)
        else:
            y_int_train = labels[tr_i].argmax(axis=1)
            clf = LogisticRegression(
                C=cfg["C"], penalty=cfg["penalty"], solver=cfg["solver"],
                class_weight=cfg["class_weight"], max_iter=cfg["max_iter"],
                random_state=RANDOM_SEED,
            )
            clf.fit(embeddings[tr_i], y_int_train)
            preds_int = clf.predict(embeddings[va_i])
            preds = np.zeros((len(preds_int), N_SDG), dtype=np.float32)
            preds[np.arange(len(preds_int)), preds_int] = 1.0
            macro = float(f1_score(labels[va_i], preds, average="macro", zero_division=0))
            per_sdg = f1_score(labels[va_i], preds, average=None, zero_division=0).tolist()
            log.info("  Fold %d/%d: macro-F1=%.4f", fold + 1, CV_N_SPLITS, macro)

        fold_scores.append(macro)
        fold_per_sdg.append(per_sdg)
        fold_n_train.append(len(tr_i))
        fold_n_val.append(len(va_i))

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    mean_per_sdg = [float(np.mean([f[c] for f in fold_per_sdg])) for c in range(N_SDG)]

    results = {
        "mode": "cv_full_data",
        "classifier_type": classifier_type,
        "config": cfg,
        "cv": {
            "n_splits": CV_N_SPLITS,
            "grouped": groups is not None,
            "groups_key": "source_doc" if groups is not None else None,
            "seed": RANDOM_SEED,
        },
        "data": {"n_total": len(embeddings), "input_dim": embeddings.shape[1]},
        "mean_macro_f1": round(mean_f1, 6),
        "std_macro_f1": round(std_f1, 6),
        "per_fold_macro_f1": [round(f, 6) for f in fold_scores],
        "per_fold_n_train": fold_n_train,
        "per_fold_n_val": fold_n_val,
        "mean_per_sdg_f1": {f"SDG_{i+1}": round(mean_per_sdg[i], 6) for i in range(N_SDG)},
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }

    out_dir = data_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_full_data_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  CV ON 100% OF LABELLED DATA — {classifier_type.upper()} "
          f"(n={len(embeddings)}, GroupKFold({CV_N_SPLITS}), grouped={groups is not None})")
    print(f"  Champion config: {cfg}")
    print(f"  Mean macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Per-fold: {[f'{f:.4f}' for f in fold_scores]}")
    print(f"{'='*70}")
    log.info("Saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain champion MLP on full train pool.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias,
                        help=f"Embed model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--data-dir", default=None,
                        help="Override data dir (derived from --model if omitted)")
    parser.add_argument("--classifier-type", default="lr", choices=["mlp", "lr"],
                        help="Classifier family (default: lr)")
    # MLP-specific args
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=MLP_CHAMPION_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7)
    # LR classifier hyperparameters (explicit — previously hardcoded champions)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of the train pool held out for MLP early-stop validation (default: %(default)s)")
    parser.add_argument("--C", type=float, default=LR_C,
                        help="LogisticRegression inverse regularisation strength (default: %(default)s)")
    parser.add_argument("--penalty", default=LR_PENALTY, choices=["l2", "l1", "none"],
                        help="LogisticRegression penalty (default: %(default)s)")
    parser.add_argument("--solver", default=LR_SOLVER, choices=["lbfgs", "liblinear"],
                        help="LogisticRegression solver (default: %(default)s)")
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"],
                        help="LogisticRegression class_weight (default: %(default)s)")
    parser.add_argument("--max-iter", type=int, default=LR_MAX_ITER,
                        help="LogisticRegression max iterations (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force retrain even if a previously trained model exists.")
    parser.add_argument("--cv-full-data", action="store_true",
                        help="EXPLORATORY (manual only): run GroupKFold CV on 100%% of labelled data "
                             "with the grid-search champion hyperparameters (LR: C=10/l2/lbfgs; "
                             "MLP: 4/384/lr=3e-4). Writes cv_full_data_results.json and exits — does "
                             "NOT retrain, save, or overwrite any model artifact. Not part of the "
                             "main pipeline; not invoked by main.py.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else model_results_dir_for_model(args.embed_model)
    output_dir = data_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cv_full_data:
        run_cv_full_data(data_dir, args.classifier_type)
        return

    if not args.overwrite:
        model_name = "mlp_retrained.joblib" if args.classifier_type == "mlp" else "sdg_classifier_retrained.joblib"
        if (output_dir / model_name).exists():
            log.info("Model already exists at %s — skip. Use --overwrite to retrain.", output_dir / model_name)
            return

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

        torch.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        net = MultiLabelMLP(input_dim, args.n_layers, args.hidden_size, args.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        X_t = torch.from_numpy(X_train.astype(np.float32))
        Y_t = torch.from_numpy(Y_train.astype(np.float32))

        # Stage 1: train with 90/10 validation split to find best epoch
        n_val = max(1, int(len(X_t) * args.val_fraction))
        perm = torch.randperm(len(X_t), generator=torch.Generator().manual_seed(RANDOM_SEED))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, X_val = X_t[tr_idx], X_t[val_idx]
        Y_tr, Y_val = Y_t[tr_idx], Y_t[val_idx]

        train_ds = TensorDataset(X_tr.to(device), Y_tr.to(device))
        val_ds = TensorDataset(X_val.to(device), Y_val.to(device))
        _seed_loader = torch.Generator().manual_seed(RANDOM_SEED)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=_seed_loader)
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
            val_pred_int = val_preds.argmax(axis=1)
            val_pred_bin = np.zeros_like(val_preds)
            val_pred_bin[np.arange(len(val_pred_int)), val_pred_int] = 1.0
            val_f1 = f1_score(val_true, val_pred_bin, average="macro", zero_division=0)

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
            generator=torch.Generator().manual_seed(RANDOM_SEED),
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
            C=args.C, penalty=args.penalty, solver=args.solver,
            class_weight=args.class_weight, max_iter=args.max_iter, random_state=RANDOM_SEED,
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
        test_pred_int = test_probs.argmax(axis=1)
        test_preds = np.zeros_like(test_probs)
        test_preds[np.arange(len(test_pred_int)), test_pred_int] = 1.0
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

    # ── Confusion matrix CSV ──────────────────────────────────────────
    # LR is single-label (softmax argmax) → sklearn confusion matrix (rows=pred, cols=true).
    # MLP is multi-label (independent per-class sigmoid) → a single-label confusion
    # matrix does not apply; instead emit a per-SDG binary (2x2) confusion summary
    # at a fixed probability threshold.
    cm_dir = output_dir_for_model(args.embed_model) / "data"
    cm_dir.mkdir(parents=True, exist_ok=True)
    if args.classifier_type == "lr":
        y_true_int = Y_test.argmax(axis=1)              # (n_test,) ground-truth SDG 0-16
        y_pred_int = test_preds.argmax(axis=1)          # (n_test,) single-label prediction
        cm = confusion_matrix(y_true_int, y_pred_int, labels=range(N_SDG))

        cm_header = "," + ",".join(f"SDG {i+1}" for i in range(N_SDG))
        cm_rows = [cm_header]
        for i in range(N_SDG):
            cm_rows.append(f"SDG {i+1}," + ",".join(str(int(cm[i, j])) for j in range(N_SDG)))

        cm_path = cm_dir / "4_1_confusion_matrix.csv"
        cm_path.write_text("\n".join(cm_rows) + "\n", encoding="utf-8")
        log.info("Saved confusion matrix CSV (ROWS=pred, COLS=true): %s", cm_path)
    elif args.classifier_type == "mlp":
        MLP_CONFUSION_THRESHOLD = 0.5
        y_true_bin = Y_test                                   # (n_test, N_SDG) one-hot ground truth
        y_pred_bin = (test_probs >= MLP_CONFUSION_THRESHOLD).astype(np.int64)
        cm_rows = ["sdg,tn,fp,fn,tp,precision,recall"]
        for k in range(N_SDG):
            tn = int(((y_true_bin[:, k] == 0) & (y_pred_bin[:, k] == 0)).sum())
            fp = int(((y_true_bin[:, k] == 0) & (y_pred_bin[:, k] == 1)).sum())
            fn = int(((y_true_bin[:, k] == 1) & (y_pred_bin[:, k] == 0)).sum())
            tp = int(((y_true_bin[:, k] == 1) & (y_pred_bin[:, k] == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cm_rows.append(f"SDG {k+1},{tn},{fp},{fn},{tp},{prec:.4f},{rec:.4f}")
        cm_path = cm_dir / "4_1_confusion_matrix_mlp.csv"
        cm_path.write_text("\n".join(cm_rows) + "\n", encoding="utf-8")
        log.info("Saved MLP per-SDG confusion matrix (threshold=%.2f): %s",
                 MLP_CONFUSION_THRESHOLD, cm_path)

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

    (output_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

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

    atomic_write_joblib(model_path, wrapper)

    if args.classifier_type != "mlp":
        # Also update sdg_classifier.joblib (used by 2_evaluate.py)
        canonical_path = output_dir / "sdg_classifier.joblib"
        atomic_write_joblib(canonical_path, wrapper)
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
