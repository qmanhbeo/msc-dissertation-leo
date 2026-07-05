"""
Semantic-gap text interpretability diagnostic.

This appendix-stage diagnostic gives a small, reproducible language-level view
of what the semantic gap looks like for three focal SDGs:

  * SDG 17: high gap, policy-dominant
  * SDG 13: high gap, policy-dominant
  * SDG 9: lower-gap comparison case

It does not relabel texts, change the semantic-gap estimates, or introduce a
qualitative coding scheme. Distinctive terms are computed from deterministic
samples of existing hard-assigned research and policy texts. Representative
examples are saved for audit only and are not used as proof.

Run from project root:
    python 1_code/3_main_analysis/3_appendix/7_semantic_gap_text_interpretability.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "1_code"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = ANALYSIS_ROOT / "0_shared"
for path in (CODE_ROOT, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from semantic_gap_shared import (
    POLICY_EMB,
    POLICY_IDS,
    POLICY_SCORES,
    RESEARCH_CENTROIDS,
    get_cluster_assignments,
    load_json,
)


DEFAULT_OUTPUT_ROOT = Path("4_outputs")
RESEARCH_TEXT_MANIFEST = Path("2_data/1_preprocessed/research_corpus/metadata/manifest.json")
RESEARCH_EMBED_MANIFEST = Path("2_data/2_embedded/research_shards/metadata/manifest.json")
RESEARCH_SCORE_MANIFEST = Path("2_data/3_scored/paper_scores_shards/metadata/manifest.json")
POLICY_TEXT_IDS = Path("2_data/2_embedded/metadata/policy_ids.json")

TARGET_SDGS = (17, 13, 9)
SAMPLE_PER_SIDE = 6000
TERMS_PER_SIDE = 8
EXAMPLES_PER_SIDE = 3
MIN_WORDS = 30
RANDOM_SEED = 42

OUTPUT_SUBDIR = "b3_semantic_gap_interpretability"
TERMS_CSV = "semantic_gap_distinctive_terms.csv"
EXAMPLES_CSV = "semantic_gap_representative_examples.csv"
SUMMARY_JSON = "semantic_gap_interpretability_summary.json"
TABLE_TEX = "tab_b3_semantic_gap_interpret.tex"

SDG_LABELS = {
    9: "SDG 9",
    13: "SDG 13",
    17: "SDG 17",
}

INTERPRETIVE_READINGS = {
    17: (
        "High-gap case: the research-side sample is technical and biomedical/ML-heavy, "
        "while policy language is international, institutional, and agenda-oriented."
    ),
    13: (
        "High-gap case: both sides concern climate, but policy language is more "
        "institutional and commitment-oriented while research language is more "
        "technical and measurement-oriented."
    ),
    9: (
        "Lower-gap comparison: differences remain visible, but the contrast is closer "
        "to a technology/infrastructure register split than to the broader institutional "
        "gap seen in SDGs 13 and 17."
    ),
}

CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "abstract",
    "article",
    "chapter",
    "data",
    "development",
    "figure",
    "goal",
    "goals",
    "paper",
    "research",
    "result",
    "results",
    "sdg",
    "sdgs",
    "study",
    "sustainable",
    "table",
    "text",
    "using",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run semantic-gap text interpretability diagnostic.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--sample-per-side", type=int, default=SAMPLE_PER_SIDE)
    return p.parse_args()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_manifest_path(stored_path: str, required_prefix: str) -> Path:
    raw = Path(stored_path)
    if raw.is_absolute():
        if raw.exists():
            return raw
        raise FileNotFoundError(f"Absolute path from manifest does not exist: {raw}")
    if not raw.as_posix().startswith(required_prefix):
        raise RuntimeError(f"Expected path under {required_prefix}, got: {stored_path}")
    resolved = ROOT / raw
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {stored_path}")
    return resolved


def load_research_shards() -> list[dict[str, Any]]:
    text_manifest = load_json(RESEARCH_TEXT_MANIFEST)
    emb_manifest = load_json(RESEARCH_EMBED_MANIFEST)
    score_manifest = load_json(RESEARCH_SCORE_MANIFEST)

    text_shards = sorted(text_manifest["shards"], key=lambda x: int(x["shard_id"]))
    emb_shards = sorted(emb_manifest["shards"], key=lambda x: int(x["shard_id"]))
    score_shards = sorted(score_manifest["shards"], key=lambda x: int(x["shard_id"]))

    if not (len(text_shards) == len(emb_shards) == len(score_shards)):
        raise RuntimeError("Research text, embedding, and score manifests are not aligned.")

    shards: list[dict[str, Any]] = []
    for text_shard, emb_shard, score_shard in zip(text_shards, emb_shards, score_shards):
        shard_id = int(text_shard["shard_id"])
        if shard_id != int(emb_shard["shard_id"]) or shard_id != int(score_shard["shard_id"]):
            raise RuntimeError("Research manifests do not align on shard_id.")
        if int(text_shard["rows"]) != int(emb_shard["rows"]) or int(text_shard["rows"]) != int(score_shard["rows"]):
            raise RuntimeError(f"Research manifests do not align on rows for shard {shard_id}.")
        shards.append(
            {
                "shard_id": shard_id,
                "name": text_shard["name"],
                "rows": int(text_shard["rows"]),
                "text_path": resolve_manifest_path(text_shard["data_path"], "2_data/1_preprocessed/"),
                "emb_path": resolve_manifest_path(emb_shard["embedding_path"], "2_data/2_embedded/"),
                "score_ids_path": resolve_manifest_path(score_shard["ids_path"], "2_data/3_scored/"),
            }
        )
    return shards


def usable_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z-]+", text)
    return len(words) >= MIN_WORDS


def snippet(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def add_sample(
    samples: dict[int, list[tuple[float, str]]],
    rngs: dict[int, random.Random],
    sdg: int,
    text: str,
    sample_cap: int,
) -> None:
    key = rngs[sdg].random()
    heap = samples[sdg]
    item = (key, text)
    if len(heap) < sample_cap:
        heapq.heappush(heap, item)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, item)


def add_top_example(
    heaps: dict[int, list[tuple[float, int, dict[str, Any]]]],
    sdg: int,
    score: float,
    seq: int,
    row: dict[str, Any],
) -> None:
    heap = heaps[sdg]
    item = (float(score), seq, row)
    if len(heap) < EXAMPLES_PER_SIDE:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def collect_research(
    sample_cap: int,
    seed: int,
    research_centroids: np.ndarray,
) -> tuple[dict[int, list[str]], dict[int, int], dict[int, list[dict[str, Any]]]]:
    samples_heap: dict[int, list[tuple[float, str]]] = {sdg: [] for sdg in TARGET_SDGS}
    rngs = {sdg: random.Random(seed + sdg * 101 + 1) for sdg in TARGET_SDGS}
    counts = {sdg: 0 for sdg in TARGET_SDGS}
    example_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = {sdg: [] for sdg in TARGET_SDGS}
    seq = 0

    shards = load_research_shards()
    for shard_idx, shard in enumerate(shards, start=1):
        emb = np.load(shard["emb_path"], mmap_mode="r")
        score_rows = list(iter_jsonl(shard["score_ids_path"]))
        if emb.shape[0] != len(score_rows):
            raise RuntimeError(f"Embedding/score row mismatch for {shard['name']}")

        with shard["text_path"].open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                score_meta = score_rows[row_idx]
                sdg = int(score_meta["assigned_sdg"])
                if sdg not in TARGET_SDGS:
                    continue
                payload = json.loads(line)
                text = str(payload.get("combined_text") or "")
                if not usable_text(text):
                    continue
                counts[sdg] += 1
                add_sample(samples_heap, rngs, sdg, text, sample_cap)
                sim = float(np.dot(emb[row_idx], research_centroids[sdg - 1]))
                seq += 1
                add_top_example(
                    example_heaps,
                    sdg,
                    sim,
                    seq,
                    {
                        "side": "research",
                        "sdg": sdg,
                        "item_id": str(payload.get("openalex_id") or score_meta.get("openalex_id") or ""),
                        "source": str(payload.get("publication_year") or ""),
                        "centroid_similarity": round(sim, 6),
                        "preview": snippet(text),
                    },
                )
        log.info("Scanned research shard %s/%s (%s)", shard_idx, len(shards), shard["name"])

    samples = {sdg: [text for _, text in sorted(heap, reverse=True)] for sdg, heap in samples_heap.items()}
    examples = {
        sdg: [row for _, _, row in sorted(heap, key=lambda item: item[0], reverse=True)]
        for sdg, heap in example_heaps.items()
    }
    return samples, counts, examples


def collect_policy(
    sample_cap: int,
    seed: int,
    policy_scores: np.ndarray,
    policy_emb: np.ndarray,
) -> tuple[dict[int, list[str]], dict[int, int], dict[int, list[dict[str, Any]]]]:
    policy_text_rows = load_json(POLICY_TEXT_IDS)
    policy_score_rows = load_json(POLICY_IDS)
    if len(policy_text_rows) != policy_scores.shape[0] or len(policy_score_rows) != policy_scores.shape[0]:
        raise RuntimeError("Policy text, score metadata, and score matrix row counts do not align.")

    assignments = get_cluster_assignments(policy_scores) + 1
    samples_heap: dict[int, list[tuple[float, str]]] = {sdg: [] for sdg in TARGET_SDGS}
    rngs = {sdg: random.Random(seed + sdg * 101 + 2) for sdg in TARGET_SDGS}
    counts = {sdg: 0 for sdg in TARGET_SDGS}
    sdg_indices: dict[int, list[int]] = {sdg: [] for sdg in TARGET_SDGS}

    for idx, sdg in enumerate(assignments):
        sdg_int = int(sdg)
        if sdg_int not in TARGET_SDGS:
            continue
        text = str(policy_text_rows[idx].get("text") or "")
        if not usable_text(text):
            continue
        counts[sdg_int] += 1
        sdg_indices[sdg_int].append(idx)
        add_sample(samples_heap, rngs, sdg_int, text, sample_cap)

    samples = {sdg: [text for _, text in sorted(heap, reverse=True)] for sdg, heap in samples_heap.items()}
    examples: dict[int, list[dict[str, Any]]] = {}
    for sdg in TARGET_SDGS:
        idxs = sdg_indices[sdg]
        if not idxs:
            examples[sdg] = []
            continue
        vecs = policy_emb[idxs]
        raw = vecs.mean(axis=0)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            examples[sdg] = []
            continue
        centroid = raw / norm
        sims = vecs @ centroid
        order = np.argsort(sims)[::-1][:EXAMPLES_PER_SIDE]
        rows: list[dict[str, Any]] = []
        for local_idx in order:
            idx = idxs[int(local_idx)]
            rows.append(
                {
                    "side": "policy",
                    "sdg": sdg,
                    "item_id": str(policy_text_rows[idx].get("id") or policy_score_rows[idx].get("id") or ""),
                    "source": str(policy_score_rows[idx].get("source_doc") or ""),
                    "centroid_similarity": round(float(sims[int(local_idx)]), 6),
                    "preview": snippet(str(policy_text_rows[idx].get("text") or "")),
                }
            )
        examples[sdg] = rows
    return samples, counts, examples


def is_valid_term(term: str) -> bool:
    if not re.search(r"[a-zA-Z]", term):
        return False
    if len(term) < 3:
        return False
    tokens = term.split()
    if any(token in CUSTOM_STOP_WORDS for token in tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return False
    return True


def top_terms_for_sdg(research_texts: list[str], policy_texts: list[str]) -> tuple[list[str], list[str]]:
    if not research_texts or not policy_texts:
        return [], []
    docs = research_texts + policy_texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(CUSTOM_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.65,
        max_features=20_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z-]+\b",
    )
    matrix = vectorizer.fit_transform(docs)
    n_research = len(research_texts)
    terms = vectorizer.get_feature_names_out()
    research_mean = np.asarray(matrix[:n_research].mean(axis=0)).ravel()
    policy_mean = np.asarray(matrix[n_research:].mean(axis=0)).ravel()

    def select(scores: np.ndarray) -> list[str]:
        selected: list[str] = []
        for idx in np.argsort(scores)[::-1]:
            term = str(terms[idx])
            if scores[idx] <= 0:
                break
            if not is_valid_term(term):
                continue
            if any(term in existing or existing in term for existing in selected):
                continue
            selected.append(term)
            if len(selected) >= TERMS_PER_SIDE:
                break
        return selected

    return select(research_mean - policy_mean), select(policy_mean - research_mean)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by 1_code/3_main_analysis/3_appendix/7_semantic_gap_text_interpretability.py -- do not edit manually",
        r"\begin{tabular}{lcp{0.26\textwidth}p{0.26\textwidth}p{0.25\textwidth}}",
        r"\toprule",
        r"SDG & Gap & Research-side distinctive terms & Policy-side distinctive terms & Descriptive reading \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['sdg_label']))} & "
            f"{float(row['semantic_gap']):.3f} & "
            f"{latex_escape(str(row['research_terms']))} & "
            f"{latex_escape(str(row['policy_terms']))} & "
            f"{latex_escape(str(row['interpretive_reading']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def semantic_gap_map(canonical_data_dir: Path) -> dict[int, float]:
    payload = load_json(canonical_data_dir / "4_3_semantic_gap_distances.json")
    return {int(row["sdg"]): float(row["semantic_gap"]) for row in payload["per_sdg"]}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_root = output_dir / "appendix" / OUTPUT_SUBDIR
    data_dir = out_root / "data"
    tables_dir = out_root / "tables"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    gaps = semantic_gap_map(Path(args.output_dir) / "main" / "data")
    research_centroids = np.load(RESEARCH_CENTROIDS).astype(np.float32)
    policy_scores = np.load(POLICY_SCORES).astype(np.float32)
    policy_emb = np.load(POLICY_EMB, mmap_mode="r")

    log.info("Collecting policy samples and representative audit examples")
    policy_samples, policy_counts, policy_examples = collect_policy(args.sample_per_side, args.seed, policy_scores, policy_emb)
    log.info("Collecting research samples and representative audit examples")
    research_samples, research_counts, research_examples = collect_research(args.sample_per_side, args.seed, research_centroids)

    term_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for sdg in TARGET_SDGS:
        research_terms, policy_terms = top_terms_for_sdg(research_samples[sdg], policy_samples[sdg])
        row = {
            "sdg": sdg,
            "sdg_label": SDG_LABELS[sdg],
            "semantic_gap": round(gaps[sdg], 6),
            "research_assigned_usable_n": research_counts[sdg],
            "policy_assigned_usable_n": policy_counts[sdg],
            "research_sample_n": len(research_samples[sdg]),
            "policy_sample_n": len(policy_samples[sdg]),
            "research_terms": ", ".join(research_terms),
            "policy_terms": ", ".join(policy_terms),
            "interpretive_reading": INTERPRETIVE_READINGS[sdg],
        }
        term_rows.append(row)
        table_rows.append(row)

    example_rows: list[dict[str, Any]] = []
    for sdg in TARGET_SDGS:
        example_rows.extend(research_examples[sdg])
        example_rows.extend(policy_examples[sdg])

    write_csv(
        data_dir / TERMS_CSV,
        [
            "sdg",
            "sdg_label",
            "semantic_gap",
            "research_assigned_usable_n",
            "policy_assigned_usable_n",
            "research_sample_n",
            "policy_sample_n",
            "research_terms",
            "policy_terms",
            "interpretive_reading",
        ],
        term_rows,
    )
    write_csv(
        data_dir / EXAMPLES_CSV,
        ["side", "sdg", "item_id", "source", "centroid_similarity", "preview"],
        example_rows,
    )
    summary = {
        "generated_from": "1_code/3_main_analysis/3_appendix/7_semantic_gap_text_interpretability.py",
        "target_sdgs": list(TARGET_SDGS),
        "random_seed": args.seed,
        "sample_per_side": args.sample_per_side,
        "terms_per_side": TERMS_PER_SIDE,
        "examples_per_side": EXAMPLES_PER_SIDE,
        "note": (
            "Distinctive terms are a descriptive interpretability aid, not a qualitative coding scheme "
            "and not a replacement for the semantic-gap estimates. Representative examples are saved "
            "for audit only."
        ),
        "rows": term_rows,
    }
    (data_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_table(tables_dir / TABLE_TEX, table_rows)

    log.info("Saved: %s", data_dir / TERMS_CSV)
    log.info("Saved: %s", data_dir / EXAMPLES_CSV)
    log.info("Saved: %s", data_dir / SUMMARY_JSON)
    log.info("Saved: %s", tables_dir / TABLE_TEX)


if __name__ == "__main__":
    main()
