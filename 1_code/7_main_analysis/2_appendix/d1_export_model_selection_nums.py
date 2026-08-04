"""
Export grid-search CV macro-F1 values from training outputs to a num_*.tex
macro file for use in Appendix D prose.

Reads lr_cv_results.json and mlp_grid_search_log.json from the parked one-off
artifact at 4_outputs/not_in_replay/model_selection/{model}/ and writes
num16_model_selection.tex into 4_outputs/{model}/tables/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
SHARED_DIR = Path(__file__).resolve().parents[1] / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import DEFAULT_EMBED_MODEL, model_slug, resolve_model_alias
from shared_utils import ensure_canonical_outputs, fingerprint_of, should_skip, record_fingerprint


def run(model: str, output_dir: Path, overwrite: bool = False) -> None:
    outs = ensure_canonical_outputs(output_dir, model=model)
    model_dir = Path("4_outputs") / "not_in_replay" / "model_selection" / model_slug(model)

    # --- LR ---
    lr_path = model_dir / "lr_cv_results.json"
    if not lr_path.exists():
        print(f"LR CV results not found at {lr_path}. Run 1_grid_search.py first.", file=sys.stderr)
        sys.exit(1)

    gs_path = model_dir / "mlp_grid_search_log.json"
    if not gs_path.exists():
        print(f"Grid search log not found at {gs_path}. Run 1_grid_search.py first.", file=sys.stderr)
        sys.exit(1)

    SCRIPT_VERSION = "2"
    PRIMARY = outs.tables_dir / "num16_model_selection.tex"
    OUTPUTS = [PRIMARY]
    fp = fingerprint_of(lr_path, gs_path) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, overwrite, PRIMARY):
        print(f"Skipping {PRIMARY} \u2014 inputs unchanged")
        return

    with open(lr_path) as f:
        lr = json.load(f)

    # Champion LR is C=10.0, L2, class_weight=None by deliberate policy: the raw
    # CV argmax can tie at C=1.0 within ~0.0004, and C=10.0 is selected for
    # conservatism (see Appendix D.1). Report the champion's own CV score as the
    # headline figure rather than the raw argmax.
    LR_CHAMPION = {"C": 3.0, "l1_ratio": 0.0, "class_weight": None}
    lr_champ = None
    for r in lr["all_cv_results"]:
        p = r["params"]
        if (p.get("C"), p.get("l1_ratio"), p.get("class_weight")) == (
            LR_CHAMPION["C"], LR_CHAMPION["l1_ratio"], LR_CHAMPION["class_weight"]
        ):
            lr_champ = r
            break
    if lr_champ is None:
        lr_champ = max(lr["all_cv_results"], key=lambda r: r["mean_f1"])
    lr_best_mean = lr_champ["mean_f1"]
    lr_best_std = lr_champ["std_f1"]

    lr_c1 = None
    for r in lr["all_cv_results"]:
        if r["params"]["C"] == 1.0:
            lr_c1 = r
            break
    if lr_c1 is None:
        print("LR C=1 config not found in lr_cv_results.json", file=sys.stderr)
        sys.exit(1)
    lr_c1_mean = lr_c1["mean_f1"]
    lr_c1_std = lr_c1["std_f1"]

    # L1 spot-check (pure-L1 at C=10, cw=None) — evidence-backed L1 statement.
    lr_l1_mean = lr_l1_std = "n/a"
    for r in lr["all_cv_results"]:
        p = r["params"]
        if p.get("l1_ratio") == 1.0 and p.get("C") == 10.0 and p.get("class_weight") is None:
            lr_l1_mean = f"{r['mean_f1']:.4f}"
            lr_l1_std = f"{r['std_f1']:.4f}"
            break

    with open(gs_path) as f:
        gs = json.load(f)

    seen = {}
    for entry in gs.get("log", []):
        cfg = entry.get("config", {})
        if "n_layers" not in cfg:
            continue
        # Full config key so weight_decay / dropout variants are distinct
        # (collapsing them would silently report the first-seen trial).
        key = (cfg["n_layers"], cfg["hidden_size"], cfg["lr"],
               cfg.get("weight_decay", 0.0), cfg.get("dropout", 0.0))
        if key not in seen:
            seen[key] = entry["cv_metrics"]

    if not seen:
        print("No MLP configs found in grid search log", file=sys.stderr)
        sys.exit(1)

    sorted_mlp = sorted(seen.items(),
                        key=lambda kv: (-kv[1]["mean_f1"], kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4]))

    # Champion MLP = empirical optimum from the expanded grid:
    # 3 layers / 256 hidden / lr=3e-4 (CV macro-F1 0.8192).
    MLP_CHAMPION_KEY = (3, 256, 0.0003, 0.0, 0.3)
    mlp_champ = seen.get(MLP_CHAMPION_KEY)
    if mlp_champ is None:
        mlp_champ = sorted_mlp[0][1]
    mlp_best_mean = mlp_champ["mean_f1"]
    mlp_best_std = mlp_champ["std_f1"]
    best_nl, best_hs = MLP_CHAMPION_KEY[0], MLP_CHAMPION_KEY[1]

    # Rank-2 reported in the D.1 table: 4 layers / 256 hidden / lr=3e-4.
    MLP_RANK2_KEY = (4, 256, 0.0003, 0.0, 0.3)
    mlp_rank2 = seen.get(MLP_RANK2_KEY)
    if mlp_rank2 is None:
        mlp_rank2 = next((v for k, v in sorted_mlp if k[0] == 4), sorted_mlp[1][1])
    mlp_rank2_mean = mlp_rank2["mean_f1"]
    mlp_rank2_std = mlp_rank2["std_f1"]

    gap = mlp_best_mean - lr_best_mean

    # The new MLP grid spans n_layers in {2,4} only. Report the full-config range
    # and the learning-rate sweep range at the best architecture.
    all_mlp_vals = [v["mean_f1"] for v in seen.values()]
    range_all = f"{min(all_mlp_vals):.3f}--{max(all_mlp_vals):.3f}" if all_mlp_vals else "n/a"
    best_arch_vals = [v["mean_f1"] for k, v in seen.items()
                      if k[0] == best_nl and k[1] == best_hs]
    range_lr = f"{min(best_arch_vals):.3f}--{max(best_arch_vals):.3f}" if best_arch_vals else "n/a"

    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/"
        "d1_export_model_selection_nums.py -- do not edit manually",
        rf"\newcommand{{\LrCvMacroFone}}{{{lr_best_mean:.4f}}}",
        rf"\newcommand{{\LrCvMacroFoneStd}}{{{lr_best_std:.4f}}}",
        rf"\newcommand{{\LrCvMacroFoneCOne}}{{{lr_c1_mean:.4f}}}",
        rf"\newcommand{{\LrCvMacroFoneCOneStd}}{{{lr_c1_std:.4f}}}",
        rf"\newcommand{{\LrCvLOneSpotCheckMean}}{{{lr_l1_mean}}}",
        rf"\newcommand{{\LrCvLOneSpotCheckStd}}{{{lr_l1_std}}}",
        rf"\newcommand{{\MlpCvMacroFone}}{{{mlp_best_mean:.4f}}}",
        rf"\newcommand{{\MlpCvMacroFoneStd}}{{{mlp_best_std:.4f}}}",
        rf"\newcommand{{\MlpCvMacroFoneFourLTwoFiftySixH}}{{{mlp_rank2_mean:.4f}}}",
        rf"\newcommand{{\MlpCvMacroFoneFourLTwoFiftySixHStd}}{{{mlp_rank2_std:.4f}}}",
        rf"\newcommand{{\LrMlpGap}}{{{gap:.4f}}}",
        rf"\newcommand{{\MlpCvRangeTwoLFourL}}{{{range_all}}}",
        rf"\newcommand{{\MlpCvRangeLrSweep}}{{{range_lr}}}",
    ]

    path = outs.tables_dir / "num16_model_selection.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Written {path}")
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run(args.embed_model, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
