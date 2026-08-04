"""
Shared training utilities for SDG classifier grid search and retrain.

Provides:
  - MultiLabelMLP (nn.Module) + _NetWrapper (pickle-friendly)
  - load_training_data() — common data loading from .npy files
  - train_mlp_fold() — single MLP fold training with early-stop
  - run_lr_grid_search() — LR grid search, sequential (one config at a time)
  - run_mlp_grid_search() — MLP grid search, sequential (one config at a time)
  - compute_fold_metrics() — f1 scoring helper

Used by:
  1_grid_search.py (orchestrator called by main.py)
  2_retrain_full_data.py (champion retrain)
"""

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, TensorDataset

import warnings
# sklearn 1.8 deprecates LogisticRegression's `penalty` arg; we select L2/elasticnet/L1
# via `l1_ratio`, which still emits a harmless FutureWarning — silence it so the
# grid-search log stays readable.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*penalty.*deprecated.*")

import sys
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from model_utils import (
    model_results_dir_for_model, append_grid_log, N_SDG, RANDOM_SEED,
)

log = logging.getLogger(__name__)


# ── MLP architecture ──────────────────────────────────────────────────

class MultiLabelMLP(nn.Module):
    """384 → [Linear → BN → ReLU → Dropout] × n_layers → 17"""

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


# ── Data loading ──────────────────────────────────────────────────────

def load_training_data(embed_model: str):
    """Load train split from .npy files. Returns (X, Y, y_int, sd_train, cv)."""
    data_dir = model_results_dir_for_model(embed_model)
    t0 = time.perf_counter()
    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")
    train_idx = np.load(data_dir / "indices" / "train.npy")

    source_docs_path = data_dir / "source_docs.npy"
    if source_docs_path.exists():
        source_docs = np.load(source_docs_path)
        sd_train = source_docs[train_idx]
    else:
        sd_train = None
        log.warning("source_docs.npy not found — falling back to row-level splits")

    X = embeddings[train_idx]
    Y = labels[train_idx]
    y_int = Y.argmax(axis=1)
    cv = GroupKFold(n_splits=5)
    log.info("Train: %d texts, %d dims  [%.1fs]", len(X), X.shape[1], time.perf_counter() - t0)
    return X, Y, y_int, sd_train, cv


# ── Scoring ───────────────────────────────────────────────────────────

def compute_fold_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[float, list[float]]:
    """Return (macro_f1, per_class_f1_list)."""
    macro = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    per_class = f1_score(Y_true, Y_pred, average=None, zero_division=0).tolist()
    return macro, per_class


# ── MLP fold training ─────────────────────────────────────────────────

def train_mlp_fold(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: dict,
    device: torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[float, list[float], float, int]:
    """Train an MLP on fold-train with internal 90/10 early-stop split.

    Returns (macro_f1, per_sdg_f1, best_internal_val_f1, best_epoch).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

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


# ── LR grid search ────────────────────────────────────────────────────

LR_PARAM_GRID = {
    # L2-only (l1_ratio=0.0). The L1/elasticnet family is a known-losing
    # regulariser for this task and was dropped to keep model selection fast
    # and focused on the decision-relevant axis (regularisation strength).
    # C is sampled on a log-ish scale with extra density in the [1,10] peak
    # region (found in the first pass: C=1 and C=10 tie, C=100 drops to 0.797).
    "C": [0.1, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0],
    "l1_ratio": [0.0],
    "class_weight": [None, "balanced"],
}

# L1 spot-check intentionally omitted from the live grid: pure-L1 under
# LogisticRegression is too slow to be worth re-running here (it needs
# OneVsRestClassifier(liblinear) or SAGA and adds minutes for a single
# config). The L1-vs-L2 conclusion is already established by a prior
# spot-check (C=10 L1 macro-F1 = 0.8066 ± 0.0069, documented in
# 5_notes/MODEL_SELECTION.md) and reproduced qualitatively below.
LR_L1_SPOT_CHECKS = []


def _load_completed_grid_log(log_path):
    """Return {canonical_cfg_key: cv_metrics} for already-logged configs.

    Used for true resume: a config present here is skipped (not retrained) and
    its stored CV metrics are reused to rebuild the results table, so an
    interrupted run loses nothing (per AGENTS.md resume-safety).
    """
    if not log_path.exists():
        return {}
    with log_path.open() as f:
        data = json.load(f)
    done = {}
    for e in data.get("log", []):
        key = json.dumps(e["config"], sort_keys=True)
        done[key] = e["cv_metrics"]
    return done


def run_lr_grid_search(X, y_int, Y, sd_train, cv, output_dir):
    """Run LR grid search sequentially, one config at a time.

    Simple single-loop over the config grid (pre-refactor style): each config is
    fit across 5 document-grouped CV folds, logged per-config for real-time
    progress, and the best classifier saved incrementally. Already-logged
    configs are skipped via append_grid_log's dedup, so an interrupted run
    resumes cleanly.
    """
    t0 = time.perf_counter()
    keys, vals = list(LR_PARAM_GRID.keys()), list(LR_PARAM_GRID.values())
    combos = [dict(zip(keys, combo)) for combo in product(*vals)]
    # Append the cheap L1 spot-check(s) so they go through the same
    # resume-aware loop (and get skipped on re-run if already logged).
    combos = combos + [dict(c) for c in LR_L1_SPOT_CHECKS]

    # True resume: reuse metrics for configs already in the durable log and skip
    # retraining them (an interrupted run loses nothing).
    log_path = output_dir / "lr_grid_search_log.json"
    done = _load_completed_grid_log(log_path)

    all_scores = []

    for i, params in enumerate(combos, 1):
        key = json.dumps(params, sort_keys=True)
        if key in done:
            m = done[key]
            all_scores.append({
                "params": params, "mean_f1": m["mean_f1"], "std_f1": m["std_f1"],
                "per_fold": m["per_fold"], "per_class_f1": m.get("per_class_f1"),
            })
            log.info("  [LR %d/%d] SKIP (already logged) C=%s l1_ratio=%s cw=%s",
                     i, len(combos), params["C"], params["l1_ratio"], params["class_weight"])
            continue

        cfg_t0 = time.perf_counter()
        log.info("  [LR %d/%d] START C=%s l1_ratio=%s cw=%s",
                 i, len(combos), params["C"], params["l1_ratio"], params["class_weight"])
        fold_scores = []
        fold_per_class = []
        cv_split = cv.split(X, groups=sd_train) if sd_train is not None else cv.split(X)
        for fold, (tr_i, val_i) in enumerate(cv_split):
            t1 = time.perf_counter()
            # L2 (l1_ratio==0) uses lbfgs — fast and matches the production
            # retrain solver. The single L1 spot-check (l1_ratio==1.0) uses
            # liblinear (pure-L1 lasso), which is far faster than SAGA for L1.
            # SAGA was pathologically slow for weak L2 regularisation (e.g.
            # C=100), so it must not be used for the L2 grid.
            if params["l1_ratio"] == 0.0:
                clf = LogisticRegression(
                    C=params["C"], penalty="l2", class_weight=params["class_weight"],
                    solver="lbfgs", max_iter=1000, random_state=42,
                )
            else:
                # L1 spot-check: liblinear is binary-only, so wrap it in
                # OneVsRestClassifier for multiclass. Fast (no SAGA).
                base = LogisticRegression(
                    C=params["C"], penalty="l1", class_weight=params["class_weight"],
                    solver="liblinear", max_iter=1000, random_state=42,
                )
                clf = OneVsRestClassifier(base)
            clf.fit(X[tr_i], y_int[tr_i])
            preds_int = clf.predict(X[val_i])
            preds = np.zeros((len(preds_int), N_SDG), dtype=np.float32)
            preds[np.arange(len(preds_int)), preds_int] = 1.0
            macro, per_class = compute_fold_metrics(Y[val_i], preds)
            fold_scores.append(macro)
            fold_per_class.append(per_class)
            log.info("  [LR %d/%d] fold %d C=%s l1_ratio=%s cw=%s: macro-F1=%.4f  [%.1fs]",
                     i, len(combos), fold + 1, params["C"], params["l1_ratio"],
                     params["class_weight"], macro, time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        mean_per_class = [float(np.mean([f[c] for f in fold_per_class])) for c in range(N_SDG)]
        all_scores.append({
            "params": params, "mean_f1": mean_f1, "std_f1": std_f1,
            "per_fold": fold_scores, "per_class_f1": mean_per_class,
        })
        log.info("  [LR %d/%d] C=%s l1_ratio=%s cw=%s -> macro-F1=%.4f ± %.4f",
                 i, len(combos), params["C"], params["l1_ratio"],
                 params["class_weight"], mean_f1, std_f1)
        log.info("  [LR %d/%d] END C=%s l1_ratio=%s cw=%s  [%.1fs]",
                 i, len(combos), params["C"], params["l1_ratio"],
                 params["class_weight"], time.perf_counter() - cfg_t0)

        # Per-config durable log (incremental + dedup) so a later resume can skip.
        output_dir.mkdir(parents=True, exist_ok=True)
        append_grid_log(
            log_path, "lr", params,
            {"mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores,
             "per_class_f1": mean_per_class},
            n_train=len(X), input_dim=X.shape[1],
            random_seed=RANDOM_SEED, search_name="comprehensive",
        )

    # Best across ALL results (trained + resumed); save a full-data fit as the
    # parked classifier (not consumed downstream, so a clean retrain is fine).
    best_entry = max(all_scores, key=lambda s: s["mean_f1"])
    best_params = best_entry["params"]
    best_score = best_entry["mean_f1"]
    best_std = best_entry["std_f1"]
    import joblib
    if best_params["l1_ratio"] == 0.0:
        best_clf = LogisticRegression(
            C=best_params["C"], penalty="l2", class_weight=best_params["class_weight"],
            solver="lbfgs", max_iter=1000, random_state=42,
        )
    else:
        best_clf = OneVsRestClassifier(
            LogisticRegression(
                C=best_params["C"], penalty="l1", class_weight=best_params["class_weight"],
                solver="liblinear", max_iter=1000, random_state=42,
            )
        )
    best_clf.fit(X, y_int)
    model_path = output_dir / "lr_classifier.joblib"
    joblib.dump(best_clf, model_path)
    log.info("  Saved best LR classifier (full-data fit) -> %s", model_path)

    results_data = {
        "model": "lr",
        "best_params": best_params,
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": best_std,
        "per_fold_macro_f1": best_entry["per_fold"],
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "all_cv_results": all_scores,
    }
    results_path = output_dir / "lr_cv_results.json"
    with results_path.open("w") as f:
        json.dump(results_data, f, indent=2, default=str)

    sorted_scores = sorted(all_scores, key=lambda x: x["mean_f1"], reverse=True)
    lines = ["", "=" * 70, "  LR Results", "=" * 70]
    header = f"  {'C':<6} {'reg':<9} {'l1_ratio':<9} {'cw':<11} {'mean F1':<8} {'±σ':<6}"
    sep = "  " + "-" * (len(header) - 2)
    lines.extend([header, sep])
    for s in sorted_scores:
        p = s["params"]
        reg = "L2" if p["l1_ratio"] == 0.0 else ("L1" if p["l1_ratio"] == 1.0 else "EN")
        cw_str = str(p["class_weight"]) if p["class_weight"] is not None else "None"
        lines.append(
            f"  {p['C']:<6} {reg:<9} {p['l1_ratio']:<9} {cw_str:<11} "
            f"{s['mean_f1']:<8.4f} {s['std_f1']:<6.4f}"
        )
    lines.append(sep)
    best_reg = "L2" if best_params["l1_ratio"] == 0.0 else ("L1" if best_params["l1_ratio"] == 1.0 else "EN")
    lines.append(f"  Best: C={best_params['C']} reg={best_reg} "
                 f"l1_ratio={best_params['l1_ratio']} "
                 f"cw={best_params['class_weight']}  macro-F1={best_score:.4f} ± {best_std:.4f}")
    lines.append("=" * 70)
    log.info("\n%s", "\n".join(lines))
    print(f"\nLR done. Best: C={best_params['C']} l1_ratio={best_params['l1_ratio']} "
          f"cw={best_params['class_weight']}  macro-F1={best_score:.4f} ± {best_std:.4f}")


# ── MLP grid search ───────────────────────────────────────────────────

MLP_PARAM_GRID = {
    # n_layers∈{2,3,4,6}: tests whether depth beyond 4 helps (first pass showed
    # 2L≈4L within noise, so this checks 3L and 6L for a monotonicity signal).
    "n_layers": [2, 3, 4, 6],
    "hidden_size": [256, 384],
    "lr": [3e-4, 1e-3],
    "weight_decay": [0],
    "dropout": [0.3],
}


def run_mlp_grid_search(X, Y, sd_train, cv, output_dir, device=None):
    """Run MLP grid search sequentially, one config at a time.

    device: None -> auto (cuda if available else cpu). Configs are trained
    one-by-one (no joblib / no worker contention) so each config's result is
    logged and appended in real time. On CUDA the single device is the
    bottleneck, so sequential configs avoid thrashing it; determinism is
    preserved via CUBLAS_WORKSPACE_CONFIG plus the flags inside train_mlp_fold.
    """
    from torch.cuda import get_device_name

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    device_info = {"device": str(device)}
    if device.type == "cuda":
        # Deterministic cuBLAS requires this workspace config (CUDA >= 10.2).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            device_info["gpu"] = get_device_name(0)
        except Exception:
            pass
        device_info["cuda_version"] = torch.version.cuda

    t0 = time.perf_counter()
    keys, vals = list(MLP_PARAM_GRID.keys()), list(MLP_PARAM_GRID.values())
    combos = [dict(zip(keys, combo)) for combo in product(*vals)]

    # True resume: reuse metrics for configs already in the durable log and skip
    # retraining them.
    log_path = output_dir / "mlp_grid_search_log.json"
    done = _load_completed_grid_log(log_path)

    log.info("Starting MLP grid search (%d configs × 5 folds, %s)", len(combos), device.type)
    all_scores = []

    for i, params in enumerate(combos, 1):
        key = json.dumps(params, sort_keys=True)
        if key in done:
            m = done[key]
            all_scores.append({"params": params, "mean_f1": m["mean_f1"],
                               "std_f1": m["std_f1"], "per_fold": m["per_fold"]})
            log.info("  [MLP %d/%d] SKIP (already logged) n_layers=%s hidden=%s lr=%s wd=%s do=%s",
                     i, len(combos), params["n_layers"], params["hidden_size"],
                     params["lr"], params["weight_decay"], params["dropout"])
            continue

        cfg_t0 = time.perf_counter()
        log.info("  [MLP %d/%d] START n_layers=%s hidden=%s lr=%s wd=%s do=%s",
                 i, len(combos), params["n_layers"], params["hidden_size"],
                 params["lr"], params["weight_decay"], params["dropout"])
        fold_scores = []
        cv_split = cv.split(X, Y, groups=sd_train) if sd_train is not None else cv.split(X)
        for fold, (tr_i, val_i) in enumerate(cv_split):
            t1 = time.perf_counter()
            macro, _, _, _ = train_mlp_fold(
                X[tr_i], Y[tr_i], X[val_i], Y[val_i], params, device,
            )
            fold_scores.append(macro)
            log.info("  [MLP %d/%d] fold %d n_layers=%s hidden=%s lr=%s wd=%s do=%s: macro-F1=%.4f  [%.1fs]",
                     i, len(combos), fold + 1, params["n_layers"], params["hidden_size"],
                     params["lr"], params["weight_decay"], params["dropout"], macro,
                     time.perf_counter() - t1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        all_scores.append({"params": params, "mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores})
        log.info("  [MLP %d/%d] n_layers=%s hidden=%s lr=%s wd=%s do=%s -> macro-F1=%.4f ± %.4f",
                 i, len(combos), params["n_layers"], params["hidden_size"], params["lr"],
                 params["weight_decay"], params["dropout"], mean_f1, std_f1)
        log.info("  [MLP %d/%d] END n_layers=%s hidden=%s lr=%s wd=%s do=%s  [%.1fs]",
                 i, len(combos), params["n_layers"], params["hidden_size"],
                 params["lr"], params["weight_decay"], params["dropout"],
                 time.perf_counter() - cfg_t0)

        # Per-config durable log (incremental + dedup) so a later resume can skip.
        output_dir.mkdir(parents=True, exist_ok=True)
        append_grid_log(
            log_path, "mlp", params,
            {"mean_f1": mean_f1, "std_f1": std_f1, "per_fold": fold_scores},
            n_train=len(X), input_dim=X.shape[1],
            random_seed=RANDOM_SEED, search_name="comprehensive", device_info=device_info,
        )

    # Best across ALL results (trained + resumed); retrained on full data below.
    best_entry = max(all_scores, key=lambda s: s["mean_f1"])
    best_params = best_entry["params"]
    best_score = best_entry["mean_f1"]
    best_std = best_entry["std_f1"]

    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    input_dim = X.shape[1]
    net = MultiLabelMLP(input_dim, best_params["n_layers"],
                        best_params["hidden_size"], best_params["dropout"]).to(device)
    train_ds = TensorDataset(torch.from_numpy(X.astype(np.float32)).to(device),
                              torch.from_numpy(Y.astype(np.float32)).to(device))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              generator=torch.Generator().manual_seed(RANDOM_SEED))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=best_params["lr"],
                            weight_decay=best_params["weight_decay"])
    for epoch in range(100):
        net.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    net.eval()

    import joblib
    wrapper = _NetWrapper(net.cpu(), input_dim)
    model_path = output_dir / "mlp_classifier.joblib"
    joblib.dump(wrapper, model_path)

    canonical_path = output_dir / "sdg_classifier.joblib"
    joblib.dump(wrapper, canonical_path)

    results_data = {
        "model": "mlp",
        "best_params": {k: v for k, v in best_params.items()},
        "best_cv_macro_f1_mean": best_score,
        "best_cv_macro_f1_std": best_std,
        "per_fold_macro_f1": next(s["per_fold"] for s in all_scores if s["params"] == best_params),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "all_cv_results": all_scores,
    }
    results_path = output_dir / "mlp_cv_results.json"
    with results_path.open("w") as f:
        json.dump(results_data, f, indent=2, default=str)

    canonical_results_path = output_dir / "sdg_cv_results.json"
    with canonical_results_path.open("w") as f:
        json.dump(results_data, f, indent=2, default=str)
    sorted_scores = sorted(all_scores, key=lambda x: x["mean_f1"], reverse=True)

    lines = ["", "=" * 70, "  MLP Results", "=" * 70]
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
    lines.append(f"  Best: {best_params}  macro-F1={best_score:.4f} ± {best_std:.4f}")
    lines.append("=" * 70)
    log.info("\n%s", "\n".join(lines))

    print(f"\nMLP done. Best: {best_params}  macro-F1={best_score:.4f} ± {best_std:.4f}")
