"""
Appendix stage: INLP register removal — validation against independent
linguistic register markers (step F2; canonical MPNet track only).

Motivation
----------
The main analysis interprets the INLP-adjusted gap as a *topic* gap on a
structural (design-based) argument: the SDG-stratified training procedure
leaves only corpus-level register signal linearly decodable at each iteration.
That argument does not by itself show that what is removed *behaves* like
register. This stage evaluates the interpretation empirically with an
independent, surface-linguistic register score computed from the segment
texts (Biber-style features: hedge rate, deontic-modality rate, passive-voice
frequency, mean sentence length, first-person rate, nominalisation rate).

Scope (deliberately narrow)
---------------------------
This is the verified promotion of the scratch diagnostic line
(5_notes/scratch/register_validation_{check,followup,followup2}.py). It
reproduces exactly the reconciled final numbers in `5_notes/report_editorial_suggestions_ignore.md` §2.2 (the
acceptance-gate block at the end of the run verifies each one; any mismatch
fails the run). It does NOT scale the sample, change statistics, or resolve
the open operationalization question (PC1 vs a-priori z-sum register score).

The stage is canonical-MPNet-only: the validated facts are MPNet-canonical
facts (gating mirrors the zero-shot appendix precedent).

Environment dependency (nltk data)
----------------------------------
The passive-voice feature requires the nltk POS tagger resource
``averaged_perceptron_tagger_eng`` (nltk 3.10's ``nltk.pos_tag`` default) and
``punkt`` for sentence tokenisation. These are NOT bundled with the conda
package install; a clean environment must fetch them once:

    python -c "import nltk; nltk.download('punkt')"
    python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"

The stage fails closed at runtime with an actionable message if either
resource is missing.

Determinism / seed design (do not refactor)
-------------------------------------------
A single module-level generator ``_rng = np.random.default_rng(SEED)`` is
created once and never reassigned; the three ``build_sample`` calls are three
successive draws of ONE continuous seed-42 stream (draw 1 = original
per-SDG-dedup sample, draw 2 = one-per-parent sample, draw 3 = Item-3
sample). Within each draw, research is sampled before policy (RNG
consumption order matters). Draw-instability is tested with *fresh*
generators at seeds 43/44/45. Per-call re-seeding produced wrong samples once
in the scratch line (see followup2.md) — do not "fix" the sampling.

Outputs (under 4_outputs/appendix/{model}/a1_register_validation/)
-------------------------------------------------------------------
- data/register_validation.json      all computed numbers (nested, machine-readable)
- data/register_validation.csv       corpus-discrimination accuracy rows
- tables/tab_a1_register_validation.tex            corpus-discrimination table
- tables/tab_a1_register_validation_selectivity.tex 17-way SDG selectivity table
- tables/num_a1_register_validation.tex           prose macros (\RegVal* prefix;
                                                   namespace distinct from the
                                                   Register*/RegIter* macros of
                                                   g_register_decomposition.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_utils import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    RANDOM_SEED,
    embed_dir_for_model,
    embed_research_dir_for_model,
    model_slug,
    open_text,
    register_dir_for_model,
    resolve_model_alias,
    resolve_policy_text_path,
    resolve_research_text_path,
    scored_dir_for_model,
)
from register_utils import TRACK_CANON, load_G  # noqa: E402
from semantic_gap_shared import latex_escape, write_csv  # noqa: E402
from shared_utils import fingerprint_of, permutation_p, record_fingerprint, should_skip  # noqa: E402

SEED = RANDOM_SEED  # 42
N_PER_SDG = 12
N_BOOT = 500
SCRIPT_VERSION = "1"

SLUG = "a1_register_validation"
JSON_OUT = "register_validation.json"
CSV_OUT = "register_validation.csv"
TABLE_TEX = "tab_a1_register_validation.tex"
TABLE_TEX_SELECTIVITY = "tab_a1_register_validation_selectivity.tex"
NUM_TEX = "num_a1_register_validation.tex"

FEAT_KEYS = [
    "hedge_rate",
    "deontic_rate",
    "passive_rate",
    "mean_sent_len",
    "first_person_rate",
    "nominal_rate",
]

HEDGE = ["may", "might", "could", "suggests", "appears", "potentially", "likely"]
DEONTIC = ["must", "shall", "should", "will"]
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours", "ourselves"}
NOM_SUFFIX = re.compile(r"\w+(tion|ment|ness)$")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the INLP register-validation appendix (MPNet canonical).")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=SEED,
                   help="Pinned to %d: this appendix is the verified seed-%d diagnostic; "
                        "draw-instability is tested at fixed fresh seeds 43/44/45." % (SEED, SEED))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, type=resolve_model_alias, help=argparse.SUPPRESS)
    p.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _check_nltk_resources() -> None:
    required = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource, name in required:
        try:
            nltk.data.find(resource)
        except LookupError:
            raise RuntimeError(
                f"nltk resource '{resource}' is missing (required for the passive-voice / "
                f"mean-sentence-length register features). Install it with:\n"
                f"    python -c \"import nltk; nltk.download('{name}')\"\n"
                f"then re-run. (A clean conda env rebuild does NOT include nltk data.)"
            )


# --------------------------------------------------------------------------- #
# Data access (module-level caches, populated in run(); model == canonical)   #
# --------------------------------------------------------------------------- #

res_sdg_index: dict[int, list[tuple[int, int]]] = defaultdict(list)
policy_assign: np.ndarray | None = None
policy_emb_full: np.ndarray | None = None
policy_ids: list[dict] | None = None
G: np.ndarray | None = None
_model: str | None = None
score_meta_dir: Path | None = None
emb_research_dir: Path | None = None
paper_ids_by_shard: dict[int, list[str]] = {}


def paper_id(shard_id: int, row_idx: int) -> str:
    lst = paper_ids_by_shard.get(shard_id)
    if lst is None:
        assert score_meta_dir is not None
        with open(score_meta_dir / f"part-{shard_id:05d}_ids.jsonl") as fh:
            lst = [json.loads(l)["openalex_id"] for l in fh]
        paper_ids_by_shard[shard_id] = lst
    return lst[row_idx]


def load_indices(model: str) -> None:
    """Build the SDG index and policy arrays for the canonical model."""
    global policy_assign, policy_emb_full, policy_ids, G, score_meta_dir, emb_research_dir
    score_meta_dir = scored_dir_for_model(model) / "paper_scores_shards" / "metadata"
    emb_research_dir = embed_research_dir_for_model(model)

    res_sdg_index.clear()
    for meta_path in sorted(score_meta_dir.glob("part-*_ids.jsonl")):
        shard_id = int(re.search(r"part-(\d+)_ids", meta_path.name).group(1))
        with open(meta_path) as fh:
            for row_idx, line in enumerate(fh):
                res_sdg_index[int(json.loads(line)["assigned_sdg"])].append((shard_id, row_idx))

    policy_scores = np.load(scored_dir_for_model(model) / "policy_scores.npy")
    policy_assign = policy_scores.argmax(axis=1)
    policy_emb_full = np.load(embed_dir_for_model(model) / "policy.npy", mmap_mode="r")
    policy_ids = json.load(open(embed_dir_for_model(model) / "metadata" / "policy_ids.json"))
    G = load_G(model)  # fail-closed on incomplete checkpoint


# --------------------------------------------------------------------------- #
# Sampling (exact port of the verified scratch logic — do not refactor)       #
# --------------------------------------------------------------------------- #

def sample_research(n: int, global_dedup: bool, rng: np.random.Generator) -> tuple[dict, dict]:
    used = set() if global_dedup else None
    per_sdg_used = {sdg: set() for sdg in range(1, 18)}
    chosen: dict[tuple[int, int], int] = {}
    per_sdg_count: dict[int, int] = {}
    for sdg in range(1, 18):
        entries = list(res_sdg_index[sdg])
        rng.shuffle(entries)
        picks = 0
        for shard_id, row_idx in entries:
            if picks >= n:
                break
            pid = paper_id(shard_id, row_idx)
            if global_dedup:
                if pid in used:
                    continue
                used.add(pid)
            else:
                if pid in per_sdg_used[sdg]:
                    continue
                per_sdg_used[sdg].add(pid)
            chosen[(shard_id, row_idx)] = sdg
            picks += 1
        per_sdg_count[sdg] = picks
    return chosen, per_sdg_count


def sample_policy(n: int, global_dedup: bool, rng: np.random.Generator) -> tuple[dict, dict]:
    used = set() if global_dedup else None
    per_sdg_used = {sdg: set() for sdg in range(1, 18)}
    chosen: dict[int, int] = {}
    per_sdg_count: dict[int, int] = {}
    for sdg_idx in range(17):
        sdg = sdg_idx + 1
        idxs = np.where(policy_assign == sdg_idx)[0].tolist()
        rng.shuffle(idxs)
        picks = 0
        for i in idxs:
            if picks >= n:
                break
            doc = policy_ids[i]["source_doc"]
            if global_dedup:
                if doc in used:
                    continue
                used.add(doc)
            else:
                if doc in per_sdg_used[sdg]:
                    continue
                per_sdg_used[sdg].add(doc)
            chosen[int(i)] = sdg
            picks += 1
        per_sdg_count[sdg] = picks
    return chosen, per_sdg_count


# Module-level RNG: created once, never reassigned. The three default
# build_sample calls draw three successive samples from ONE continuous
# seed-42 stream (matching the verified scratch behaviour).
_rng = np.random.default_rng(SEED)


def build_sample(global_dedup: bool, rng: np.random.Generator | None = None) -> tuple[list[dict], dict, dict]:
    """Build a sample with texts loaded; draws research BEFORE policy."""
    if rng is None:
        rng = _rng
    assert _model is not None
    res_chosen, res_counts = sample_research(N_PER_SDG, global_dedup, rng)
    pol_chosen, pol_counts = sample_policy(N_PER_SDG, global_dedup, rng)

    # Load research embeddings + texts (shard-native, row-aligned)
    by_shard: dict[int, list[int]] = defaultdict(list)
    for sid, ri in res_chosen:
        by_shard[sid].append(ri)

    res_data: dict[tuple[int, int], tuple[np.ndarray, str]] = {}
    for sid, rids in by_shard.items():
        emb = np.load(emb_research_dir / f"part-{sid:05d}.npy", mmap_mode="r")
        rids_set = set(rids)
        text_path = resolve_research_text_path(_model, f"part-{sid:05d}")
        with open_text(text_path) as fh:
            for ri, line in enumerate(fh):
                if ri in rids_set:
                    res_data[(sid, ri)] = (np.asarray(emb[ri]), json.loads(line)["text"])

    # Load policy embeddings + texts
    pol_need = set(pol_chosen)
    pol_data: dict[int, tuple[np.ndarray, str]] = {}
    with open_text(resolve_policy_text_path(_model)) as fh:
        for i, line in enumerate(fh):
            if i in pol_need:
                pol_data[i] = (np.asarray(policy_emb_full[i]), json.loads(line)["text"])

    # Assemble records
    records = []
    for (sid, ri), sdg in res_chosen.items():
        emb, text = res_data[(sid, ri)]
        records.append({"corpus": 0, "sdg": sdg, "emb_raw": emb, "text": text,
                        "parent": paper_id(sid, ri)})

    for i, sdg in pol_chosen.items():
        emb, text = pol_data[i]
        records.append({"corpus": 1, "sdg": sdg, "emb_raw": emb, "text": text,
                        "parent": policy_ids[i]["source_doc"]})

    return records, res_counts, pol_counts


# --------------------------------------------------------------------------- #
# Register features / score (exact port of the verified scratch logic)        #
# --------------------------------------------------------------------------- #

def compute_features(text: str) -> dict:
    words = word_tokenize(text.lower())
    n_words = len(words)
    if n_words == 0:
        return {k: 0.0 for k in FEAT_KEYS}
    sentences = sent_tokenize(text)
    n_sents = max(1, len(sentences))
    hedge = sum(1 for w in words if w in HEDGE)
    deontic = sum(1 for w in words if w in DEONTIC)
    first_person = sum(1 for w in words if w in FIRST_PERSON)
    nominal = sum(1 for w in words if NOM_SUFFIX.match(w))
    tagged = nltk.pos_tag(words)
    passive = 0
    for i in range(1, len(tagged)):
        if tagged[i][1] == "VBN":
            if tagged[i - 1][0] in BE_FORMS or (
                i >= 2 and tagged[i - 1][1] in {"RB", "RBR", "RBS"} and tagged[i - 2][0] in BE_FORMS
            ):
                passive += 1
    return {
        "hedge_rate": 1000.0 * hedge / n_words,
        "deontic_rate": 1000.0 * deontic / n_words,
        "passive_rate": 1000.0 * passive / n_words,
        "mean_sent_len": n_words / n_sents,
        "first_person_rate": 1000.0 * first_person / n_words,
        "nominal_rate": 1000.0 * nominal / n_words,
    }


def reg_score_from_features(F: np.ndarray) -> np.ndarray:
    """PC1 of the standardised feature matrix, oriented so that correlation
    with mean_sent_len (Fz[:, 3]) is positive."""
    Fz = StandardScaler().fit_transform(F)
    Fzc = Fz - Fz.mean(axis=0)
    _, _, Vt = np.linalg.svd(Fzc, full_matrices=False)
    pc1 = Fzc @ Vt[0]
    if np.corrcoef(pc1, Fz[:, 3])[0, 1] < 0:
        pc1 = -pc1
    return pc1


def assemble(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_raw = np.stack([r["emb_raw"].astype(np.float32) for r in records])
    assert G is not None
    X_adj = _project(X_raw, G).astype(np.float32)
    F = np.array([[compute_features(r["text"])[k] for k in FEAT_KEYS] for r in records], dtype=np.float64)
    reg = reg_score_from_features(F)
    corr = np.array([r["corpus"] for r in records])
    sdg = np.array([r["sdg"] for r in records])
    return X_raw, X_adj, F, reg, corr, sdg


def _project(emb: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Project + L2-renormalise per row (identical to register_utils.project)."""
    if g.shape[0] == 0:
        return emb.copy()
    proj = (emb @ g.T) @ g
    residual = emb - proj
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (residual / norms).astype(np.float32)


def spearman(a, b):
    return permutation_p(a, b, kind="spearman")


def partial_spearman(x, y, z):
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    return permutation_p(rx - Z @ bx, ry - Z @ by, kind="spearman")


def sdg_centroid_dists(X: np.ndarray, sdg_arr: np.ndarray) -> np.ndarray:
    dist = np.full(len(X), np.nan)
    for sdg in range(1, 18):
        m = sdg_arr == sdg
        if m.sum() < 2:
            continue
        c = X[m].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        dist[m] = 1.0 - (X[m] @ c)
    return dist


def own_other_dists(X: np.ndarray, sdg_arr: np.ndarray, corr_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d_own = np.full(len(X), np.nan)
    d_other = np.full(len(X), np.nan)
    for sdg in range(1, 18):
        for corp in [0, 1]:
            m = (sdg_arr == sdg) & (corr_arr == corp)
            if m.sum() < 2:
                continue
            c = X[m].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            d_own[m] = 1.0 - (X[m] @ c)
            mo = (sdg_arr == sdg) & (corr_arr == (1 - corp))
            if mo.sum() < 2:
                continue
            c_o = X[mo].mean(axis=0)
            c_o = c_o / (np.linalg.norm(c_o) + 1e-12)
            d_other[m] = 1.0 - (X[m] @ c_o)
    return d_own, d_other


# --------------------------------------------------------------------------- #
# Statistics / classifiers (exact port of the verified scratch logic)         #
# --------------------------------------------------------------------------- #

def cv_acc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    return cross_val_score(clf, X, y, cv=skf, scoring="accuracy").mean()


def pooled_cv_predictions(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """5-fold stratified CV; returns per-unit correctness bool + fold id.

    Each row is classified by the classifier trained on the other 4 folds, so
    pooled correctness = a single honest accuracy estimate (no leakage).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.empty(len(y), dtype=bool)
    folds = np.empty(len(y), dtype=int)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te]) == y[te]
        folds[te] = f
    return preds, folds


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def acc_report(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """CV accuracy three ways: fold-mean (original metric), pooled k/n with
    Wilson 95% CI, and one-sided binomial test vs 0.5."""
    preds, _ = pooled_cv_predictions(X, y, seed)
    k = int(preds.sum())
    n = len(preds)
    lo, hi = wilson_ci(k, n)
    bt = binomtest(k, n, 0.5, alternative="greater")
    fold_mean = cv_acc(X, y, seed=seed)
    return dict(fold_mean=float(fold_mean), pooled=float(k / n), k=k, n=n,
                wilson_lo=float(lo), wilson_hi=float(hi), binom_p=float(bt.pvalue))


def boot_diff_pvals(preds_a: np.ndarray, preds_b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED + 100) -> dict:
    """Bootstrap the accuracy DIFFERENCE on pooled fold predictions (not re-CV)."""
    rng_d = np.random.default_rng(seed)
    na, nb = len(preds_a), len(preds_b)
    diffs = []
    for _ in range(n_boot):
        da = rng_d.choice(na, size=na, replace=True)
        db = rng_d.choice(nb, size=nb, replace=True)
        diffs.append(preds_b[db].mean() - preds_a[da].mean())
    diffs = np.array(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)), p_pos=float((diffs > 0).mean()))


def sdg_selectivity(X_raw: np.ndarray, X_adj: np.ndarray, sdg_arr: np.ndarray) -> dict:
    """17-way SDG classifier CV accuracy (raw vs adjusted), LR + kNN(5)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {}
    for key, X in [("raw", X_raw), ("adj", X_adj)]:
        lr = LogisticRegression(C=10.0, max_iter=2000, random_state=SEED)
        knn = KNeighborsClassifier(n_neighbors=5)
        out[key] = {
            "lr_acc": float(cross_val_score(lr, X, sdg_arr, cv=skf, scoring="accuracy").mean()),
            "knn_acc": float(cross_val_score(knn, X, sdg_arr, cv=skf, scoring="accuracy").mean()),
        }
    out["chance"] = 1 / 17
    return out


# --------------------------------------------------------------------------- #
# Acceptance gate                                                             #
# --------------------------------------------------------------------------- #

def _close(a: float, b: float, tol: float = 0.0005) -> bool:
    return abs(a - b) <= tol


def _fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def check_gates(checks: list[tuple[str, bool, str]]) -> None:
    print("\n" + "=" * 70)
    print("ACCEPTANCE GATE (targets: `5_notes/report_editorial_suggestions_ignore.md` section 2.2)")
    print("=" * 70)
    n_fail = 0
    for label, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"  [{mark}] {label}: {detail}")
    if n_fail:
        raise RuntimeError(
            f"{n_fail} acceptance-gate check(s) FAILED. The promoted diagnostic must "
            "reproduce the verified numbers exactly; do not fudge. See log above."
        )
    print("  All gates passed — numbers match the verified report.")


# --------------------------------------------------------------------------- #
# Output writers                                                              #
# --------------------------------------------------------------------------- #

def _p_fmt(p: float) -> str:
    return f"$<0.001$" if p < 0.001 else f"{p:.3f}"


def write_json(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_csv_out(path: Path, rows: list[dict]) -> None:
    write_csv(path, list(rows[0].keys()), rows)


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a1_register_validation.py — do not edit manually",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Sample & Input & $n$ & Fold-mean acc. & Pooled acc. ($k/n$) & Wilson 95\% CI & Binom.\ $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['sample_label']))} & "
            f"{latex_escape(str(row['input_label']))} & "
            f"{int(row['n'])} & "
            f"{float(row['fold_mean']):.3f} & "
            f"{float(row['pooled']):.3f} ({int(row['k'])}/{int(row['n'])}) & "
            f"[{float(row['wilson_lo']):.3f}, {float(row['wilson_hi']):.3f}] & "
            f"{_p_fmt(float(row['binom_p']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_selectivity(path: Path, sel: dict) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a1_register_validation.py — do not edit manually",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Input & LR CV acc. & kNN(5) CV acc. & Chance \\",
        r"\midrule",
        f"Raw & {sel['raw']['lr_acc']:.3f} & {sel['raw']['knn_acc']:.3f} & {sel['chance']:.3f} \\\\",
        f"Adjusted (INLP) & {sel['adj']['lr_acc']:.3f} & {sel['adj']['knn_acc']:.3f} & {sel['chance']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_num_macros(path: Path, m: dict) -> None:
    lines = [
        "% Auto-generated by 1_code/7_main_analysis/2_appendix/a1_register_validation.py — do not edit manually",
        "% Register-validation appendix prose macros (\\RegVal* namespace; distinct from",
        "% the Register*/RegIter* macros of g_register_decomposition.py).",
        "% p-values: two-sided Monte Carlo permutation (100,000 resamples, seed 42).",
    ]
    for key in sorted(m):
        lines.append(f"\\newcommand{{\\RegVal{key}}}{{{m[key]}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    model = args.embed_model
    if model != DEFAULT_EMBED_MODEL:
        log.info(
            "Register-validation appendix is canonical-MPNet-only (got %s) — skipping.",
            model,
        )
        return
    if args.seed != SEED:
        raise RuntimeError(
            f"--seed is pinned to {SEED} for this appendix: it is the verified seed-{SEED} "
            "diagnostic (draw-instability is tested at fixed fresh seeds 43/44/45). "
            f"Got --seed {args.seed}."
        )
    _check_nltk_resources()

    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / model_slug(model) / SLUG
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    PRIMARY = data_dir / JSON_OUT
    OUTPUTS = [
        PRIMARY,
        data_dir / CSV_OUT,
        tables_dir / TABLE_TEX,
        tables_dir / TABLE_TEX_SELECTIVITY,
        tables_dir / NUM_TEX,
    ]
    g_path = register_dir_for_model(model) / TRACK_CANON / "G.npy"
    fp = fingerprint_of(
        scored_dir_for_model(model) / "paper_scores_shards" / "metadata" / "manifest.json",
        embed_research_dir_for_model(model) / "metadata" / "manifest.json",
        g_path,
    ) + SCRIPT_VERSION
    if should_skip(OUTPUTS, fp, args.overwrite, PRIMARY):
        log.info("Skipping %s — inputs unchanged", PRIMARY)
        return

    global _model
    _model = model
    log.info("Building SDG index and loading G (%s)", g_path)
    load_indices(model)

    results, csv_rows, checks = compute_diagnostics()
    check_gates(checks)

    write_json(PRIMARY, results)
    write_csv_out(data_dir / CSV_OUT, csv_rows)
    write_table(tables_dir / TABLE_TEX, csv_rows)
    write_table_selectivity(tables_dir / TABLE_TEX_SELECTIVITY, results["step3_selectivity"])
    write_num_macros(tables_dir / NUM_TEX, results["macros"])

    log.info("Saved: %s", PRIMARY)
    log.info("Saved: %s", data_dir / CSV_OUT)
    log.info("Saved: %s", tables_dir / TABLE_TEX)
    log.info("Saved: %s", tables_dir / TABLE_TEX_SELECTIVITY)
    log.info("Saved: %s", tables_dir / NUM_TEX)
    record_fingerprint(OUTPUTS, fp, PRIMARY)


def compute_diagnostics() -> tuple[dict, list[dict], list[tuple[str, bool, str]]]:
    """Run the full verified diagnostic line; returns (results, csv_rows, gate_checks)."""
    checks: list[tuple[str, bool, str]] = []

    print("=" * 70)
    print("ITEM 1: One-per-parent sample construction")
    print("=" * 70)

    # ---- Draw 1: original sample (per-SDG dedup) ----
    recs_a, res_cts_a, pol_cts_a = build_sample(global_dedup=False)
    parents_a = Counter(r["parent"] for r in recs_a)
    n_multi_a = sum(c for c in parents_a.values() if c > 1)
    mega_parents = {p for p, c in parents_a.items() if c > 1}
    mega_sdgs = sorted({r["sdg"] for r in recs_a if r["parent"] in mega_parents})

    print(f"\nOriginal sample (per-SDG dedup, seed {SEED}):")
    print(f"  Total: {len(recs_a)}  Research: {sum(1 for r in recs_a if r['corpus']==0)}  "
          f"Policy: {sum(1 for r in recs_a if r['corpus']==1)}")
    print(f"  Distinct parents: {len(parents_a)}  Multi-parent units: {n_multi_a}")
    for mp in sorted(mega_parents):
        sdgs = sorted(set(r["sdg"] for r in recs_a if r["parent"] == mp))
        print(f"    {mp}: {parents_a[mp]} units, SDGs {sdgs}")
    print(f"  Mega-docs: {len(mega_parents)} docs, {n_multi_a} segments, "
          f"{len(mega_sdgs)}/17 SDGs")

    # ---- Draw 2: one-per-parent sample (global dedup) ----
    recs_b, res_cts_b, pol_cts_b = build_sample(global_dedup=True)
    parents_b = Counter(r["parent"] for r in recs_b)
    n_multi_b = sum(c for c in parents_b.values() if c > 1)
    print(f"\nOne-per-parent sample (global dedup, seed {SEED}):")
    print(f"  Total: {len(recs_b)}  Research: {sum(1 for r in recs_b if r['corpus'] == 0)}  "
          f"Policy: {sum(1 for r in recs_b if r['corpus'] == 1)}")
    print(f"  Distinct parents: {len(parents_b)}  Multi-parent units: {n_multi_b}")

    checks.append(("Item-1 2A distinct parents == 390",
                   len(parents_a) == 390, f"{len(parents_a)}"))
    checks.append(("Item-1 2A multi-parent units == 25",
                   n_multi_a == 25, f"{n_multi_a}"))
    checks.append(("Item-1 2A mega-docs == 7",
                   len(mega_parents) == 7, f"{len(mega_parents)}"))
    checks.append(("Item-1 2A mega SDGs == 15",
                   len(mega_sdgs) == 15, f"{len(mega_sdgs)}"))
    checks.append(("Item-1 2B distinct parents == 408",
                   len(parents_b) == 408, f"{len(parents_b)}"))
    checks.append(("Item-1 2B multi-parent units == 0",
                   n_multi_b == 0, f"{n_multi_b}"))

    print("\n" + "=" * 70)
    print("Computing features and projections...")
    print("=" * 70)
    X_a, X_adj_a, F_a, reg_a, corr_a, sdg_a = assemble(recs_a)
    X_b, X_adj_b, F_b, reg_b, corr_b, sdg_b = assemble(recs_b)

    # ---- Corpus-mean register features (orig sample; report1 Step 1) ----
    corpus_means = {}
    for j, k in enumerate(FEAT_KEYS):
        corpus_means[k] = {
            "research": float(F_a[corr_a == 0, j].mean()),
            "policy": float(F_a[corr_a == 1, j].mean()),
        }
    print("\n--- mean register features by corpus (orig sample) ---")
    for k in FEAT_KEYS:
        print(f"  {k}: research={corpus_means[k]['research']:.3f}  "
              f"policy={corpus_means[k]['policy']:.3f}")
    expected_corpus_means = {
        "hedge_rate": (1.427, 0.854), "deontic_rate": (0.780, 3.130),
        "passive_rate": (10.303, 8.453), "mean_sent_len": (37.035, 63.803),
        "first_person_rate": (6.901, 6.463), "nominal_rate": (36.367, 39.003),
    }
    for k, (r_exp, p_exp) in expected_corpus_means.items():
        checks.append((f"corpus-mean {k} research == {r_exp:.3f}",
                       _close(corpus_means[k]["research"], r_exp),
                       f"{corpus_means[k]['research']:.3f}"))
        checks.append((f"corpus-mean {k} policy == {p_exp:.3f}",
                       _close(corpus_means[k]["policy"], p_exp),
                       f"{corpus_means[k]['policy']:.3f}"))

    # ---- Mega vs non-mega feature contrast (orig sample) ----
    mega_recs = [r for r in recs_a if r["parent"] in mega_parents]
    non_mega_recs = [r for r in recs_a if r["parent"] not in mega_parents]
    mega_F = np.array([[compute_features(r["text"])[k] for k in FEAT_KEYS] for r in mega_recs])
    non_mega_F = np.array([[compute_features(r["text"])[k] for k in FEAT_KEYS] for r in non_mega_recs])
    mega_feat = {k: float(mega_F[:, j].mean()) for j, k in enumerate(FEAT_KEYS)}
    non_mega_feat = {k: float(non_mega_F[:, j].mean()) for j, k in enumerate(FEAT_KEYS)}
    print(f"\n--- Mega-doc features (n={len(mega_recs)}) vs non-mega (n={len(non_mega_recs)}) ---")
    for k in FEAT_KEYS:
        print(f"  {k:18s} mega={mega_feat[k]:.3f}  non-mega={non_mega_feat[k]:.3f}")
    expected_mega = {"mean_sent_len": (276.879, 35.637), "passive_rate": (3.349, 9.771),
                     "first_person_rate": (18.996, 5.879), "nominal_rate": (20.016, 38.838)}
    for k, (m_exp, n_exp) in expected_mega.items():
        checks.append((f"mega-feature {k} == {m_exp:.3f}", _close(mega_feat[k], m_exp),
                       f"{mega_feat[k]:.3f}"))
        checks.append((f"non-mega-feature {k} == {n_exp:.3f}", _close(non_mega_feat[k], n_exp),
                       f"{non_mega_feat[k]:.3f}"))

    # ---- 2b / 2c on both samples ----
    def run_2b_2c(X_raw, X_adj, reg_score, corpus_arr, sdg_arr, label):
        removed_norm = np.linalg.norm(X_raw - X_adj, axis=1)
        print(f"\n--- {label} (n={len(corpus_arr)}) ---")
        rho, p = spearman(reg_score, removed_norm)
        print(f"2b) reg_score ~ ||x-x'||  pooled rho={rho:.3f} (p={p:.3g})")
        within = {}
        for corp in [0, 1]:
            m = corpus_arr == corp
            rho_c, p_c = spearman(reg_score[m], removed_norm[m])
            within[corp] = (float(rho_c), float(p_c))
            print(f"    within-{'research' if corp==0 else 'policy'}: rho={rho_c:.3f} (p={p_c:.3g})")
        dist_raw = sdg_centroid_dists(X_raw, sdg_arr)
        dist_adj = sdg_centroid_dists(X_adj, sdg_arr)
        rho_r, p_r = spearman(reg_score, dist_raw)
        rho_a, p_a = spearman(reg_score, dist_adj)
        print(f"2c) reg_score ~ dist-SDG-centroid RAW rho={rho_r:.3f} (p={p_r:.3g})  "
              f"ADJ rho={rho_a:.3f} (p={p_a:.3g})")
        rho_pr, _ = partial_spearman(reg_score, dist_raw, corpus_arr)
        rho_pa, _ = partial_spearman(reg_score, dist_adj, corpus_arr)
        print(f"    partial (controlling corpus) RAW rho={rho_pr:.3f}  ADJ rho={rho_pa:.3f}")
        return dict(
            pooled=(float(rho), float(p)), within_res=(within[0][0], within[0][1]),
            within_pol=(within[1][0], within[1][1]),
            raw=(float(rho_r), float(p_r)), adj=(float(rho_a), float(p_a)),
            partial_raw=float(rho_pr), partial_adj=float(rho_pa),
        )

    res_2b2c_a = run_2b_2c(X_a, X_adj_a, reg_a, corr_a, sdg_a, "ITEM 2A original sample (per-SDG dedup)")
    res_2b2c_b = run_2b_2c(X_b, X_adj_b, reg_b, corr_b, sdg_b, "ITEM 2B one-per-parent sample")

    for label, got, target in [
        ("2b orig pooled rho", res_2b2c_a["pooled"][0], 0.102),
        ("2b orig within-res rho", res_2b2c_a["within_res"][0], 0.212),
        ("2b orig within-pol rho", res_2b2c_a["within_pol"][0], 0.191),
        ("2b opp pooled rho", res_2b2c_b["pooled"][0], 0.092),
        ("2b opp within-res rho", res_2b2c_b["within_res"][0], -0.043),
        ("2b opp within-pol rho", res_2b2c_b["within_pol"][0], -0.036),
        ("2c orig RAW rho", res_2b2c_a["raw"][0], 0.126),
        ("2c orig ADJ rho", res_2b2c_a["adj"][0], 0.247),
        ("2c opp RAW rho", res_2b2c_b["raw"][0], -0.212),
        ("2c opp ADJ rho", res_2b2c_b["adj"][0], -0.197),
    ]:
        checks.append((f"{label} == {target:.3f}", _close(got, target), f"{got:.3f}"))

    print("\n" + "=" * 70)
    print("ITEM 2: Corpus classifier accuracy — valid CIs (pooled fold predictions)")
    print("=" * 70)

    acc_rows = []
    acc_results = {}
    for key, label, Xraw, Xadj, reg_score, corr_arr in [
        ("orig", "Original (per-SDG dedup)", X_a, X_adj_a, reg_a, corr_a),
        ("opp", "One-per-parent (draw 2)", X_b, X_adj_b, reg_b, corr_b),
    ]:
        print(f"\n--- {label} (n={len(corr_arr)}) ---")
        reg_only = acc_report(reg_score.reshape(-1, 1), corr_arr)
        raw = acc_report(Xraw, corr_arr)
        adj = acc_report(Xadj, corr_arr)
        acc_results[key] = dict(reg=reg_only, raw=raw, adj=adj)
        for name, rec in [("register-only", reg_only), ("raw", raw), ("adj", adj)]:
            print(f"  {name}: fold-mean acc={rec['fold_mean']:.4f} | pooled acc={rec['pooled']:.4f} "
                  f"({rec['k']}/{rec['n']})  Wilson 95% CI [{rec['wilson_lo']:.4f}, {rec['wilson_hi']:.4f}]  "
                  f"binom p(vs 0.5)={rec['binom_p']:.4g}")
        sample_label = "Original (per-SDG dedup)" if key == "orig" else "One-per-parent"
        for name, rec in [("register-only", reg_only), ("raw embeddings", raw),
                          ("adjusted (INLP)", adj)]:
            acc_rows.append(dict(sample=key, sample_label=sample_label, input=name,
                                 input_label=name, n=rec["n"], fold_mean=rec["fold_mean"],
                                 pooled=rec["pooled"], k=rec["k"],
                                 wilson_lo=rec["wilson_lo"], wilson_hi=rec["wilson_hi"],
                                 binom_p=rec["binom_p"]))

    # Mega-policy exclusion on the ORIGINAL sample
    mega_pol_idx = {i for i, r in enumerate(recs_a)
                    if r["corpus"] == 1 and r["parent"] in mega_parents}
    keep_a = [i for i in range(len(recs_a)) if i not in mega_pol_idx]
    excl = acc_report(X_adj_a[keep_a], corr_a[keep_a])
    print(f"\n--- Mega-policy exclusion effect (original sample) ---")
    print(f"  Original adj acc, ALL 408 units: {acc_results['orig']['adj']['pooled']:.4f}")
    print(f"  Original adj, mega-policy EXCLUDED (n={len(keep_a)}): pooled acc={excl['pooled']:.4f} "
          f"({excl['k']}/{excl['n']})  Wilson 95% CI [{excl['wilson_lo']:.4f}, {excl['wilson_hi']:.4f}]  "
          f"binom p={excl['binom_p']:.4g}")
    acc_rows.append(dict(sample="orig", sample_label="Original (per-SDG dedup)",
                         input="adj_mega_excluded", input_label="adjusted, mega-policy excluded",
                         n=excl["n"], fold_mean=excl["fold_mean"], pooled=excl["pooled"],
                         k=excl["k"], wilson_lo=excl["wilson_lo"], wilson_hi=excl["wilson_hi"],
                         binom_p=excl["binom_p"]))

    # Bootstrap diff on pooled predictions (NOT resample-then-CV)
    preds_a = pooled_cv_predictions(X_adj_a, corr_a)[0]
    preds_b = pooled_cv_predictions(X_adj_b, corr_b)[0]
    boot = boot_diff_pvals(preds_a, preds_b)
    print(f"\n--- Bootstrap diff: one-per-parent adj acc vs original adj acc ---")
    print(f"  Diff (opp - orig): {boot['mean']:+.4f}  95% CI [{boot['lo']:.4f}, {boot['hi']:.4f}]  "
          f"p(>0): {boot['p_pos']:.4f}")

    # ---- Step 3: 17-way SDG selectivity (draw-1 sample) ----
    print("\n" + "=" * 70)
    print("STEP 3: 17-way SDG classifier, 5-fold CV accuracy (draw-1 sample)")
    print("=" * 70)
    sel = sdg_selectivity(X_a, X_adj_a, sdg_a)
    for name in ["raw", "adj"]:
        print(f"  {name}: LR acc={sel[name]['lr_acc']:.3f}  kNN(5) acc={sel[name]['knn_acc']:.3f}  "
              f"(chance={sel['chance']:.3f})")
    for label, got, target in [
        ("Step-3 raw LR acc", sel["raw"]["lr_acc"], 0.691),
        ("Step-3 raw kNN acc", sel["raw"]["knn_acc"], 0.554),
        ("Step-3 adj LR acc", sel["adj"]["lr_acc"], 0.672),
        ("Step-3 adj kNN acc", sel["adj"]["knn_acc"], 0.578),
    ]:
        checks.append((f"{label} == {target:.3f}", _close(got, target), f"{got:.3f}"))

    print("\n" + "=" * 70)
    print("ITEM 3: Policy other-dist pull — per-SDG + mega-doc exclusion")
    print("=" * 70)

    # 3rd draw of the SAME stream == the original followup's Item-3 sample
    recs_c, res_cts_c, pol_cts_c = build_sample(global_dedup=True)
    parents_c = Counter(r["parent"] for r in recs_c)
    print(f"\nItem-3 sample = 3rd draw (global dedup): n={len(recs_c)}, "
          f"distinct parents={len(parents_c)}")
    X_c, X_adj_c, F_c, reg_c, corr_c, sdg_c = assemble(recs_c)

    # Reproduction gate: pooled reg ~ dist RAW/ADJ must equal -0.088 / -0.074
    dist_raw_c = sdg_centroid_dists(X_c, sdg_c)
    dist_adj_c = sdg_centroid_dists(X_adj_c, sdg_c)
    r_raw, p_raw = spearman(reg_c, dist_raw_c)
    r_adj, p_adj = spearman(reg_c, dist_adj_c)
    print(f"\nREPRO GATE pooled (n={len(recs_c)}): reg ~ dist RAW rho={r_raw:.3f} (p={p_raw:.3g})  "
          f"ADJ rho={r_adj:.3f} (p={p_adj:.3g})   [target -0.088 / -0.074]")
    checks.append(("Item-3 pooled reg~dist RAW rho == -0.088", _close(float(r_raw), -0.088),
                   f"{r_raw:.3f}"))
    checks.append(("Item-3 pooled reg~dist ADJ rho == -0.074", _close(float(r_adj), -0.074),
                   f"{r_adj:.3f}"))

    do_c, dx_c = own_other_dists(X_adj_c, sdg_c, corr_c)
    pol_mask = corr_c == 1
    rho_pool, p_pool = spearman(reg_c[pol_mask], dx_c[pol_mask])
    print(f"REPRO GATE policy reg ~ other-dist ADJ pooled: rho={rho_pool:.3f} (p={p_pool:.3g})"
          f"   [target -0.197]")
    checks.append(("Item-3 policy other-dist ADJ rho == -0.197", _close(float(rho_pool), -0.197),
                   f"{rho_pool:.3f}"))

    per_sdg_rows = []
    n_sig = 0
    n_pos = 0
    n_neg = 0
    sig_sdgs = []
    print(f"\nPer-SDG (policy segments, ADJ space, reg ~ other-dist):")
    print(f"  {'SDG':>4s}  {'n':>3s}  {'rho':>7s}  {'p':>8s}  {'mega':>4s}")
    for sdg in range(1, 18):
        m = (sdg_c == sdg) & pol_mask
        n_sdg = int(m.sum())
        if n_sdg < 4:
            print(f"  {sdg:4d}  {n_sdg:3d}  (too few)")
            continue
        r, p = spearman(reg_c[m], dx_c[m])
        n_mega = sum(1 for rec in recs_c
                     if rec["corpus"] == 1 and rec["sdg"] == sdg and rec["parent"] in mega_parents)
        per_sdg_rows.append(dict(sdg=sdg, n=n_sdg, rho=float(r), p=float(p), n_mega=n_mega))
        print(f"  {sdg:4d}  {n_sdg:3d}  {r:+7.3f}  {p:8.4g}  {n_mega:4d}")
        if p < 0.05:
            n_sig += 1
            sig_sdgs.append(sdg)
        if r > 0:
            n_pos += 1
        else:
            n_neg += 1
    print(f"\n  SDGs with p<0.05: {n_sig}/17  {sorted(sig_sdgs)}  "
          f"(positive rho: {n_pos}, negative: {n_neg})")
    checks.append(("Item-3 per-SDG significant == 0", n_sig == 0, f"{n_sig}/17"))
    checks.append(("Item-3 per-SDG positive rho count == 3 (SDG 3, 7, 10)",
                   n_pos == 3 and sorted(sig_sdgs) == [], f"{n_pos} pos / {n_neg} neg"))

    # Mega-doc exclusion across all three samples (policy reg ~ other-dist ADJ)
    mega_excl_rows = {}
    for key, label, recs_full, Xadj_full, reg_full, sdg_full, corr_full in [
        ("draw1", "Original (draw 1)", recs_a, X_adj_a, reg_a, sdg_a, corr_a),
        ("draw2", "One-per-parent (draw 2)", recs_b, X_adj_b, reg_b, sdg_b, corr_b),
        ("draw3", "One-per-parent (draw 3 / Item-3)", recs_c, X_adj_c, reg_c, sdg_c, corr_c),
    ]:
        mega_idx = {i for i, r in enumerate(recs_full)
                    if r["parent"] in mega_parents and r["corpus"] == 1}
        keep = [i for i in range(len(recs_full)) if i not in mega_idx]
        X_adj_sub = Xadj_full[keep]
        reg_sub = reg_full[keep]
        sdg_sub = sdg_full[keep]
        corr_sub = corr_full[keep]
        dist_sub = sdg_centroid_dists(X_adj_sub, sdg_sub)
        pol_sub = corr_sub == 1
        if pol_sub.sum() > 10:
            r, p = spearman(reg_sub[pol_sub], dist_sub[pol_sub])
            do_sub, dx_sub = own_other_dists(X_adj_sub, sdg_sub, corr_sub)
            r2, p2 = spearman(reg_sub[pol_sub], dx_sub[pol_sub])
            mega_excl_rows[key] = dict(centroid_rho=float(r), centroid_p=float(p),
                                       other_rho=float(r2), other_p=float(p2),
                                       n=int(len(keep)), policy_n=int(pol_sub.sum()))
            print(f"  {label} (n={len(keep)}, policy n={pol_sub.sum()}):")
            print(f"    policy reg ~ centroid-dist ADJ (mega excluded): rho={r:.3f} (p={p:.3g})")
            print(f"    policy reg ~ other-dist ADJ  (mega excluded):   rho={r2:.3f} (p={p2:.3g})")

    for label, key, target in [
        ("mega-excl draw1 other rho", "draw1", 0.127),
        ("mega-excl draw2 other rho", "draw2", -0.138),
        ("mega-excl draw3 other rho", "draw3", -0.213),
    ]:
        checks.append((f"{label} == {target:.3f}", _close(mega_excl_rows[key]["other_rho"], target),
                       f"{mega_excl_rows[key]['other_rho']:.3f}"))

    # Draw-instability check: fresh independent one-per-parent draws seeds 43/44/45
    draw_rows = []
    print(f"\n--- Draw-instability check (fresh one-per-parent draws, seeds 43/44/45) ---")
    for seed in (43, 44, 45):
        rng_fresh = np.random.default_rng(seed)
        recs_x, _, _ = build_sample(global_dedup=True, rng=rng_fresh)
        X_x, X_adj_x, F_x, reg_x, corr_x, sdg_x = assemble(recs_x)
        dist_adj_x = sdg_centroid_dists(X_adj_x, sdg_x)
        rx, px = spearman(reg_x, dist_adj_x)
        do_x, dx_x = own_other_dists(X_adj_x, sdg_x, corr_x)
        polx = corr_x == 1
        rpolx, ppolx = spearman(reg_x[polx], dx_x[polx])
        draw_rows.append(dict(seed=seed, n=len(recs_x), pooled_rho=float(rx), pooled_p=float(px),
                              policy_rho=float(rpolx), policy_p=float(ppolx)))
        print(f"  seed {seed}: n={len(recs_x)}  pooled reg~dist ADJ rho={rx:.3f} (p={px:.3g})  "
              f"policy reg~other-dist ADJ rho={rpolx:.3f} (p={ppolx:.3g})")
    for label, seed, target in [
        ("draw-stability seed 43 policy rho", 43, -0.130),
        ("draw-stability seed 44 policy rho", 44, -0.004),
        ("draw-stability seed 45 policy rho", 45, 0.126),
    ]:
        row = next(r for r in draw_rows if r["seed"] == seed)
        checks.append((f"{label} == {target:.3f}", _close(row["policy_rho"], target),
                       f"{row['policy_rho']:.3f}"))

    # ---- Accuracy gates ----
    for label, rec, targets in [
        ("2A reg-only fold-mean", acc_results["orig"]["reg"], 0.456),
        ("2A raw fold-mean", acc_results["orig"]["raw"], 0.909),
        ("2A adj fold-mean", acc_results["orig"]["adj"], 0.505),
        ("2B reg-only fold-mean", acc_results["opp"]["reg"], 0.544),
        ("2B raw fold-mean", acc_results["opp"]["raw"], 0.944),
        ("2B adj fold-mean", acc_results["opp"]["adj"], 0.603),
    ]:
        checks.append((f"{label} == {targets:.3f}", _close(rec["fold_mean"], targets),
                       f"{rec['fold_mean']:.4f}"))
    checks.append(("2A adj pooled k/n == 206/408",
                   acc_results["orig"]["adj"]["k"] == 206, f"{acc_results['orig']['adj']['k']}/408"))
    checks.append(("2B adj pooled k/n == 246/408",
                   acc_results["opp"]["adj"]["k"] == 246, f"{acc_results['opp']['adj']['k']}/408"))
    checks.append(("mega-policy exclusion pooled k/n == 220/383",
                   excl["k"] == 220 and excl["n"] == 383, f"{excl['k']}/{excl['n']}"))
    checks.append(("bootstrap diff mean == +0.098", _close(boot["mean"], 0.098), f"{boot['mean']:+.4f}"))

    # ---- Assemble results dict ----
    macros = {
        "SampleN": str(len(recs_a)),
        "DistinctParentsOrig": str(len(parents_a)),
        "MultiParentUnits": str(n_multi_a),
        "MegaDocs": str(len(mega_parents)),
        "MegaSegments": str(n_multi_a),
        "MegaSdgs": str(len(mega_sdgs)),
        "OppDistinctParents": str(len(parents_b)),
        "AccRegOrig": f"{acc_results['orig']['reg']['fold_mean']:.3f}",
        "AccRawOrig": f"{acc_results['orig']['raw']['fold_mean']:.3f}",
        "AccAdjOrig": f"{acc_results['orig']['adj']['fold_mean']:.3f}",
        "AccAdjOrigK": str(acc_results["orig"]["adj"]["k"]),
        "AccAdjOrigN": str(acc_results["orig"]["adj"]["n"]),
        "AccAdjOrigCi": f"[{acc_results['orig']['adj']['wilson_lo']:.3f}, {acc_results['orig']['adj']['wilson_hi']:.3f}]",
        "AccAdjOrigP": _fmt_p(acc_results["orig"]["adj"]["binom_p"]),
        "AccRegOpp": f"{acc_results['opp']['reg']['fold_mean']:.3f}",
        "AccRawOpp": f"{acc_results['opp']['raw']['fold_mean']:.3f}",
        "AccAdjOpp": f"{acc_results['opp']['adj']['fold_mean']:.3f}",
        "AccAdjOppK": str(acc_results["opp"]["adj"]["k"]),
        "AccAdjOppN": str(acc_results["opp"]["adj"]["n"]),
        "AccAdjOppCi": f"[{acc_results['opp']['adj']['wilson_lo']:.3f}, {acc_results['opp']['adj']['wilson_hi']:.3f}]",
        "AccAdjOppP": _fmt_p(acc_results["opp"]["adj"]["binom_p"]),
        "AccRise": f"{boot['mean']:+.3f}",
        "AccRiseCi": f"[{boot['lo']:.3f}, {boot['hi']:.3f}]",
        "AccRiseP": f"{boot['p_pos']:.3f}",
        "AccMegaExcl": f"{excl['pooled']:.3f}",
        "AccMegaExclK": str(excl["k"]),
        "AccMegaExclN": str(excl["n"]),
        "AccMegaExclCi": f"[{excl['wilson_lo']:.3f}, {excl['wilson_hi']:.3f}]",
        "AccMegaExclP": f"{excl['binom_p']:.3f}",
        "TwoBOrigPooled": f"{res_2b2c_a['pooled'][0]:+.3f}",
        "TwoBOrigPooledP": f"{res_2b2c_a['pooled'][1]:.3f}",
        "TwoBOrigWithinRes": f"{res_2b2c_a['within_res'][0]:+.3f}",
        "TwoBOrigWithinPol": f"{res_2b2c_a['within_pol'][0]:+.3f}",
        "TwoBOppPooled": f"{res_2b2c_b['pooled'][0]:+.3f}",
        "TwoBOppWithinRes": f"{res_2b2c_b['within_res'][0]:+.3f}",
        "TwoBOppWithinPol": f"{res_2b2c_b['within_pol'][0]:+.3f}",
        "TwoCOrigRaw": f"{res_2b2c_a['raw'][0]:+.3f}",
        "TwoCOrigAdj": f"{res_2b2c_a['adj'][0]:+.3f}",
        "TwoCOrigPartialRaw": f"{res_2b2c_a['partial_raw']:+.3f}",
        "TwoCOrigPartialAdj": f"{res_2b2c_a['partial_adj']:+.3f}",
        "TwoCOppRaw": f"{res_2b2c_b['raw'][0]:+.3f}",
        "TwoCOppAdj": f"{res_2b2c_b['adj'][0]:+.3f}",
        "TwoCOppPartialRaw": f"{res_2b2c_b['partial_raw']:+.3f}",
        "TwoCOppPartialAdj": f"{res_2b2c_b['partial_adj']:+.3f}",
        "ItemThreePooledRaw": f"{float(r_raw):+.3f}",
        "ItemThreePooledAdj": f"{float(r_adj):+.3f}",
        "OtherDistRho": f"{float(rho_pool):+.3f}",
        "OtherDistP": f"{float(p_pool):.3f}",
        "PerSdgSig": str(n_sig),
        "PerSdgPos": str(n_pos),
        "PerSdgNeg": str(n_neg),
        "MegaExclDrawOne": f"{mega_excl_rows['draw1']['other_rho']:+.3f}",
        "MegaExclDrawOneP": f"{mega_excl_rows['draw1']['other_p']:.3f}",
        "MegaExclDrawTwo": f"{mega_excl_rows['draw2']['other_rho']:+.3f}",
        "MegaExclDrawTwoP": f"{mega_excl_rows['draw2']['other_p']:.3f}",
        "MegaExclDrawThree": f"{mega_excl_rows['draw3']['other_rho']:+.3f}",
        "MegaExclDrawThreeP": f"{mega_excl_rows['draw3']['other_p']:.3f}",
        "DrawFortyThree": f"{draw_rows[0]['policy_rho']:+.3f}",
        "DrawFortyThreeP": f"{draw_rows[0]['policy_p']:.3f}",
        "DrawFortyFour": f"{draw_rows[1]['policy_rho']:+.3f}",
        "DrawFortyFourP": f"{draw_rows[1]['policy_p']:.3f}",
        "DrawFortyFive": f"{draw_rows[2]['policy_rho']:+.3f}",
        "DrawFortyFiveP": f"{draw_rows[2]['policy_p']:.3f}",
        "SelRawLr": f"{sel['raw']['lr_acc']:.3f}",
        "SelRawKnn": f"{sel['raw']['knn_acc']:.3f}",
        "SelAdjLr": f"{sel['adj']['lr_acc']:.3f}",
        "SelAdjKnn": f"{sel['adj']['knn_acc']:.3f}",
        "SelChance": f"{sel['chance']:.3f}",
        "MegaMeanSentLen": f"{mega_feat['mean_sent_len']:.1f}",
        "NonMegaMeanSentLen": f"{non_mega_feat['mean_sent_len']:.1f}",
        "MegaPassive": f"{mega_feat['passive_rate']:.1f}",
        "NonMegaPassive": f"{non_mega_feat['passive_rate']:.1f}",
        "MegaFirstPerson": f"{mega_feat['first_person_rate']:.1f}",
        "NonMegaFirstPerson": f"{non_mega_feat['first_person_rate']:.1f}",
        "MegaNominal": f"{mega_feat['nominal_rate']:.1f}",
        "NonMegaNominal": f"{non_mega_feat['nominal_rate']:.1f}",
    }
    macro_feat_map = {
        "hedge_rate": "Hedge", "deontic_rate": "Deontic", "passive_rate": "Passive",
        "mean_sent_len": "SentLen", "first_person_rate": "FirstPerson", "nominal_rate": "Nominal",
    }
    for k in FEAT_KEYS:
        tag = macro_feat_map[k]
        macros[f"Corpus{tag}Res"] = f"{corpus_means[k]['research']:.3f}"
        macros[f"Corpus{tag}Pol"] = f"{corpus_means[k]['policy']:.3f}"

    results = {
        "script": "1_code/7_main_analysis/2_appendix/a1_register_validation.py",
        "script_version": SCRIPT_VERSION,
        "model": DEFAULT_EMBED_MODEL,
        "model_slug": model_slug(DEFAULT_EMBED_MODEL),
        "seed": SEED,
        "n_per_sdg": N_PER_SDG,
        "features": FEAT_KEYS,
        "register_score_operationalization": (
            "PC1 of the six standardised features, oriented so that correlation with "
            "mean_sent_len is positive. (An alternative a-priori z-sum operationalization "
            "was not adopted; see `5_notes/report_editorial_suggestions_ignore.md` section 2.4 item 1.)"
        ),
        "item1_sample_construction": {
            "orig_2A": {"n": len(recs_a), "distinct_parents": len(parents_a),
                        "multi_parent_units": n_multi_a,
                        "mega_docs": sorted(mega_parents),
                        "mega_doc_counts": {mp: parents_a[mp] for mp in sorted(mega_parents)},
                        "mega_sdg_count": len(mega_sdgs)},
            "opp_2B": {"n": len(recs_b), "distinct_parents": len(parents_b),
                       "multi_parent_units": n_multi_b},
            "is_rebuild_not_subset": True,
        },
        "corpus_mean_features": {k: corpus_means[k] for k in FEAT_KEYS},
        "mega_vs_nonmega_features": {
            "mega_n": len(mega_recs), "non_mega_n": len(non_mega_recs),
            "mega": mega_feat, "non_mega": non_mega_feat},
        "step2b_removed_norm": {"orig": res_2b2c_a, "opp": res_2b2c_b},
        "step2c_centroid_dist": {
            "orig": {"raw": list(res_2b2c_a["raw"]), "adj": list(res_2b2c_a["adj"]),
                     "partial_raw": res_2b2c_a["partial_raw"], "partial_adj": res_2b2c_a["partial_adj"]},
            "opp": {"raw": list(res_2b2c_b["raw"]), "adj": list(res_2b2c_b["adj"]),
                    "partial_raw": res_2b2c_b["partial_raw"], "partial_adj": res_2b2c_b["partial_adj"]},
        },
        "step2d_accuracy": {
            "orig": acc_results["orig"], "opp": acc_results["opp"],
            "bootstrap_diff": boot,
            "mega_policy_exclusion": dict(excl, **{"n_kept": len(keep_a)}),
        },
        "step3_selectivity": sel,
        "item3": {
            "pooled_reg_dist": {"raw": [float(r_raw), float(p_raw)],
                                "adj": [float(r_adj), float(p_adj)]},
            "policy_other_dist_pooled": [float(rho_pool), float(p_pool)],
            "per_sdg": per_sdg_rows,
            "per_sdg_significant": n_sig,
            "mega_exclusion": mega_excl_rows,
            "draw_stability": draw_rows,
        },
        "macros": macros,
    }
    return results, acc_rows, checks


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
