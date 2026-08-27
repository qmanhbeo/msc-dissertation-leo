"""Export a corpus-provenance table (LaTeX) for the methodology section.

Reports record counts at each pipeline stage for all seven corpora used in the
study (five labelled reference corpora + the unlabelled research and policy
corpora):

    Corpus | Raw | Prep | Seg | Lab

"Prep" is the cleaned, de-duplicated record count that survives the 20-word
minimum and proceeds to embedding; equivalently, the number of documents
(abstracts for research, records for reference and policy) that yield at least
one segment. "Lab" (single-label) is not applicable (---) for the unlabelled
research and policy corpora.

The numbers are computed from the frozen snapshot (2_data/...), not hardcoded,
so the table stays reproducible. The generator is deterministic and idempotent.

Outputs (model-independent, written to the default-model tables dir like the
other reference/policy export script):
    4_outputs/mpnet/tables/num18_corpus_provenance.tex   (macros)
    4_outputs/mpnet/tables/tab18_corpus_provenance.tex   (tabular)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_UTILS_DIR = SCRIPT_DIR.parent / "0_shared"
if str(MODEL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_UTILS_DIR))
from model_utils import resolve_corpus_provenance_path  # noqa: E402

SEGMENTED = ROOT / "2_data" / "2_segmented"
PREPROCESSED = ROOT / "2_data" / "1_preprocessed"
RAW = ROOT / "2_data" / "0_raw"
OUT_DIR = ROOT / "4_outputs" / "mpnet" / "tables"

log = logging.getLogger(__name__)


def _tex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def _count_jsonl_dir(d: Path, skip: set[str] = {"metadata"}) -> int:
    total = 0
    for f in sorted(d.iterdir()):
        if f.name in skip or not f.name.endswith(".jsonl"):
            continue
        total += _count_jsonl(f)
    return total


def _group_segments_by_source(path: Path) -> dict[str, int]:
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            c[rec.get("source", "unknown")] += 1
    return c


def _group_single_label_segments_by_source(path: Path) -> dict[str, int]:
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sdgs = rec.get("sdgs")
            if isinstance(sdgs, list) and len(sdgs) == 1 and 1 <= sdgs[0] <= 17:
                c[rec.get("source", "unknown")] += 1
    return c


def _distinct_docs_by_source(path: Path) -> dict[str, int]:
    """Count distinct documents (by source_doc) per source in a segmented file.

    This is the post-20-word-minimum document count (records that yield at
    least one segment), used uniformly for the Prep column across all corpora.
    """
    from collections import defaultdict

    d: dict[str, set] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            d[rec.get("source", "unknown")].add(rec.get("source_doc"))
    return {k: len(v) for k, v in d.items()}


def _count_distinct_docs(path: Path) -> int:
    """Count distinct documents (by source_doc) in a segmented file."""
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s.add(rec.get("source_doc"))
    return len(s)


def _research_segment_stats(seg_dir: Path) -> tuple[int, int, int]:
    """Return (n_abstracts, n_segments, n_multi_segment_abstracts).

    A research record is identified by ``openalex_id``; an abstract is
    "multi-segment" if it yields more than one segment.
    """
    counts: Counter = Counter()
    n_seg = 0
    for f in sorted(seg_dir.iterdir()):
        if not f.name.endswith(".jsonl"):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = rec.get("openalex_id") or rec.get("source_doc")
                if pid is None:
                    continue
                counts[pid] += 1
                n_seg += 1
    n_abstracts = len(counts)
    n_multi = sum(1 for v in counts.values() if v > 1)
    return n_abstracts, n_seg, n_multi


def _policy_subcollection_stats(policy_seg: Path) -> dict:
    fam_seg: Counter = Counter()
    fam_doc: dict[str, set] = {}
    with open(policy_seg, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sf = rec.get("source_family")
            fam_seg[sf] += 1
            fam_doc.setdefault(sf, set()).add(rec.get("source_doc"))

    def _get(sf):
        return int(fam_seg.get(sf, 0)), len(fam_doc.get(sf, set()))

    curated_seg, curated_doc = _get("curated_ai_sdg")
    sdgi_seg, sdgi_doc = _get("sdgi_vnr_vlr")
    ungdc_seg, ungdc_doc = _get("ungdc_speeches")

    sdgi_single = 0
    with open(SEGMENTED / "reference.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("source") == "sdgi":
                L = rec.get("sdgs")
                if isinstance(L, list) and len(L) == 1:
                    sdgi_single += 1

    return {
        "curated_seg": curated_seg, "curated_doc": curated_doc,
        "sdgi_seg": sdgi_seg, "sdgi_doc": sdgi_doc,
        "ungdc_seg": ungdc_seg, "ungdc_doc": ungdc_doc,
        "sdgi_single": sdgi_single,
    }


def compute() -> dict:
    """Return the corpus-provenance rows dict.

    In the dev repo 2_segmented/ is present, so the counts are computed
    locally. On a fresh clone (no 2_segmented/, no 1_preprocessed/) the
    embedded snapshot ships a byte-identical precomputed copy under
    3a_warm_replay_texts/_shared_metadata/corpus_provenance.json; load that
    instead. Fail closed via the resolver if neither is available.
    """
    if SEGMENTED.exists():
        return _compute_from_segmented()
    path = resolve_corpus_provenance_path()
    log.info("2_segmented/ absent; loading shipped provenance JSON: %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_from_segmented() -> dict:
    rows = {}

    # ---- Reference corpora ----
    ref_prep = _distinct_docs_by_source(SEGMENTED / "reference.jsonl")
    ref_seg = _group_segments_by_source(SEGMENTED / "reference.jsonl")
    ref_sl = _group_single_label_segments_by_source(SEGMENTED / "reference.jsonl")

    # Raw input counts per reference source. Each read FAILS OPEN (except →
    # warning → None): a missing/moved raw file renders as an empty ("---")
    # cell in the manuscript provenance table instead of failing the stage.
    # Deliberate for this one-off provenance export — but check the warnings
    # if any cell is unexpectedly empty.
    raw_counts = {}
    # OSDG
    try:
        import csv
        with open(RAW / "osdg" / "osdg_dataset.csv", encoding="utf-8") as f:
            raw_counts["osdg"] = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    except Exception as e:
        log.warning("osdg raw: %s", e)
    # Benchmark
    try:
        import csv
        with open(RAW / "sdg_benchmark" / "benchmark.csv", encoding="utf-8") as f:
            raw_counts["benchmark"] = sum(1 for _ in csv.DictReader(f))
    except Exception as e:
        log.warning("benchmark raw: %s", e)
    # Knowledge Hub
    try:
        df = __import__("pandas").read_csv(RAW / "sdg_knowledge_hub" / "sdg_knowledge_hub.csv")
        raw_counts["sdg_knowledge_hub"] = len(df)
    except Exception as e:
        log.warning("kh raw: %s", e)
    # Aurora
    try:
        raw_counts["aurora"] = _count_jsonl(RAW / "aurora" / "aurora_raw.jsonl")
    except Exception as e:
        log.warning("aurora raw: %s", e)
    # SDGi
    try:
        df = __import__("pandas").read_parquet(RAW / "sdgi_corpus" / "sdgi_corpus.parquet")
        raw_counts["sdgi"] = len(df)
    except Exception as e:
        log.warning("sdgi raw: %s", e)

    ref_sources = {
        "osdg": "OSDG",
        "benchmark": "SDGCB",
        "sdg_knowledge_hub": "SDGKH",
        "sdgi": "SDGi",
        "aurora": "Aurora",
    }
    for key, label in ref_sources.items():
        rows[key] = {
            "label": label,
            "raw": raw_counts.get(key),
            "prep": ref_prep.get(key),
            "seg": ref_seg.get(key),
            "sln": ref_sl.get(key),
        }

    # ---- Research corpus ----
    try:
        meta = json.loads((RAW / "openalex" / "artifact" / "metadata.json").read_text())
        research_raw = int(meta["total_unique_papers"])
    except Exception as e:
        log.warning("research raw: %s", e)
        research_raw = None
    # Prep = distinct abstracts that yield >=1 segment (post 20-word minimum);
    # this is the analysis-ready research count, equal to \NResearchAbstracts{}.
    research_abstracts, research_seg, _ = _research_segment_stats(SEGMENTED / "research")
    rows["research"] = {
        "label": "Research (OpenAlex)",
        "raw": research_raw,
        "prep": research_abstracts,
        "seg": research_seg,
        "sln": None,
    }

    # ---- Policy corpus ----
    try:
        pmeta = json.loads((PREPROCESSED / "metadata" / "build_policy_corpus.json").read_text())
        policy_raw = int(pmeta["total_raw"])
    except Exception as e:
        log.warning("policy raw: %s", e)
        policy_raw = None
    # Prep = distinct documents (post 20-word minimum) that proceed to embedding.
    policy_prep = _count_distinct_docs(SEGMENTED / "policy.jsonl")
    policy_seg = _count_jsonl(SEGMENTED / "policy.jsonl")
    sub = _policy_subcollection_stats(SEGMENTED / "policy.jsonl")
    rows["policy"] = {
        "label": "Policy (3 Collections)",
        "raw": policy_raw,
        "prep": policy_prep,
        "seg": policy_seg,
        "sln": None,
        "sub": sub,
    }

    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = argparse.ArgumentParser(description="Export corpus-provenance table.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = p.parse_args()

    num_out = OUT_DIR / "num18_corpus_provenance.tex"
    tab_out = OUT_DIR / "tab18_corpus_provenance.tex"
    if not args.overwrite and num_out.exists() and tab_out.exists():
        log.info("skip: outputs present (%s, %s)", num_out.name, tab_out.name)
        return

    rows = compute()

    # ---- Macro prefixes per row ----
    prefix = {
        "osdg": "RefOSDG",
        "benchmark": "RefBenchmark",
        "sdg_knowledge_hub": "RefKH",
        "sdgi": "RefSDGi",
        "aurora": "RefAurora",
        "research": "Research",
        "policy": "Policy",
    }
    # Macro names must be letters-only (LaTeX \newcommand forbids digits),
    # so we keep Raw/Prep/Seg/Sln internally; the "Step 0/1/2/3" labels live
    # only in the table header + note.
    stage_keys = [("Raw", "raw"), ("Prep", "prep"), ("Seg", "seg"), ("Sln", "sln")]

    macro_lines = [
        "% Auto-generated by export_corpus_provenance.py — do not edit manually",
        "% Corpus-provenance macros (Raw / Prep / Seg / Sln -> Step 0/1/2/3 in header)",
        "",
    ]
    for key in prefix:
        r = rows[key]
        pre = prefix[key]
        for mstage, fstage in stage_keys:
            val = r[fstage]
            text = _tex_int(val) if isinstance(val, int) else "---"
            macro_lines.append(f"\\newcommand{{\\{pre}{mstage}}}{{{text}}}")
        macro_lines.append("")

    # Policy sub-collection macros (now sourced from Table 3.5; num19 retired).
    sub = rows["policy"].get("sub", {})
    macro_lines += [
        "% Policy sub-collection counts (folded from num19 into Table 3.5)",
        f"\\newcommand{{\\PolicyCuratedSegments}}{{{_tex_int(sub['curated_seg'])}}}",
        f"\\newcommand{{\\PolicyCuratedDocs}}{{{_tex_int(sub['curated_doc'])}}}",
        f"\\newcommand{{\\PolicySDGiSegAtSeg}}{{{_tex_int(sub['sdgi_seg'])}}}",
        f"\\newcommand{{\\PolicySDGiDocsAtSeg}}{{{_tex_int(sub['sdgi_doc'])}}}",
        f"\\newcommand{{\\PolicySDGiSingleLabel}}{{{_tex_int(sub['sdgi_single'])}}}",
        f"\\newcommand{{\\PolicyUNGDCSegments}}{{{_tex_int(sub['ungdc_seg'])}}}",
        f"\\newcommand{{\\PolicyUNGDCDocs}}{{{_tex_int(sub['ungdc_doc'])}}}",
        "",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_out.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    log.info("wrote %s", num_out)

    # ---- Tabular ----
    order = ["osdg", "benchmark", "sdg_knowledge_hub", "sdgi", "aurora",
             "research", "policy"]
    tab_lines = [
        "% Auto-generated by export_corpus_provenance.py — do not edit manually",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Corpus & Raw & Prep & Seg & Lab \\\\",
        "\\midrule",
    ]
    for key in order[:5]:
        r = rows[key]
        pre = prefix[key]
        sln_cell = f"\\{pre}Sln"
        tab_lines.append(
            f"{r['label']} & \\{pre}Raw & \\{pre}Prep & \\{pre}Seg & {sln_cell} \\\\"
        )
    tab_lines.append("\\midrule")
    for key in order[5:]:
        r = rows[key]
        pre = prefix[key]
        tab_lines.append(
            f"{r['label']} & \\{pre}Raw & \\{pre}Prep & \\{pre}Seg & --- \\\\"
        )
    tab_lines.append("\\bottomrule")
    tab_lines.append("\\end{tabular}")

    tab_out.write_text("\n".join(tab_lines) + "\n", encoding="utf-8")
    log.info("wrote %s", tab_out)

    # Echo for verification
    print("Corpus provenance:")
    for key in order:
        r = rows[key]
        sln = r["sln"] if r["sln"] is not None else "---"
        print(f"  {r['label']:32s} raw={r['raw']} prep={r['prep']} seg={r['seg']} sln={sln}")
    print(f"  policy sub: curated={sub['curated_seg']}/{sub['curated_doc']} "
          f"sdgi={sub['sdgi_seg']}/{sub['sdgi_doc']} ungdc={sub['ungdc_seg']}/{sub['ungdc_doc']}")


if __name__ == "__main__":
    main()
