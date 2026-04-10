"""
fix_sdg4_artefact.py — SDG 4 artefact diagnostic

Purpose:
    The research coverage profile assigns 22.1% of papers (n=1,362) to SDG 4
    (Quality Education). This is suspected to be a measurement artefact: ML papers
    routinely use 'learning', 'training', and 'model' as core vocabulary, which
    overlaps with the OSDG Education centroid. This script disambiguates the
    SDG 4 assignments into:
        - genuine education research (education keywords present)
        - ML terminology artefact (ML keywords present, education keywords absent)
        - ambiguous (both or neither keyword set present)

Inputs:
    data/openalex/papers_clean.jsonl        — research paper texts
    data/paper_scores.npy                   — (6172, 17) cosine similarities
    data/paper_scores_ids.json              — list of {id} per row

Outputs:
    data/sdg4_artefact_analysis.json        — counts and revised coverage profile

Run:
    python code/fix_sdg4_artefact.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Terms that indicate genuine education/learning research
EDUCATION_KEYWORDS = [
    r"\bstudent\b",
    r"\bteacher\b",
    r"\bclassroom\b",
    r"\bcurriculum\b",
    r"\bschool\b",
    r"\buniversity\b",
    r"\blearner\b",
    r"\bpedagog",
    r"\bliteracy\b",
    r"\beducation\b",
    r"\binstruction\b",
    r"\btutoring\b",
    r"\be-learning\b",
    r"\belearning\b",
    r"\bteaching\b",
    r"\bcoursew",
    r"\bacademic performance\b",
    r"\bstudent performance\b",
]

# Terms that indicate ML-methodology papers (not education-specific)
ML_ARTEFACT_KEYWORDS = [
    r"\bneural network\b",
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\btraining data\b",
    r"\bmodel training\b",
    r"\bclassification\b",
    r"\bregression\b",
    r"\bgradient\b",
    r"\boptimizer\b",
    r"\bloss function\b",
    r"\bbackpropagation\b",
    r"\bconvolutional\b",
    r"\brecurrent\b",
    r"\btransformer\b",
    r"\battention mechanism\b",
    r"\bfine-tuning\b",
    r"\bfine tuning\b",
    r"\bpre-trained\b",
    r"\bpretrained\b",
    r"\bepoch\b",
    r"\bbatch size\b",
]

_EDU_RE = re.compile("|".join(EDUCATION_KEYWORDS), re.IGNORECASE)
_ML_RE = re.compile("|".join(ML_ARTEFACT_KEYWORDS), re.IGNORECASE)


def classify_paper(text: str) -> str:
    """
    Returns one of: 'genuine_education', 'ml_artefact', 'ambiguous'.
    Rule:
        - If ML keywords present AND education keywords absent → ml_artefact
        - If education keywords present → genuine_education
        - Otherwise → ambiguous
    """
    has_edu = bool(_EDU_RE.search(text))
    has_ml = bool(_ML_RE.search(text))

    if has_edu:
        return "genuine_education"
    elif has_ml:
        return "ml_artefact"
    else:
        return "ambiguous"


def main():
    # Load paper texts from raw per-SDG files (papers_clean.jsonl may not exist yet)
    print("Loading raw per-SDG paper files ...")
    papers = {}
    openalex_dir = DATA / "openalex"
    for sdg_file in sorted(openalex_dir.glob("papers_sdg*.jsonl")):
        with open(sdg_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                oid = record.get("openalex_id") or record.get("id", "")
                title = record.get("title", "") or ""
                abstract = record.get("abstract", "") or ""
                text = f"{title} {abstract}".strip()
                if oid and oid not in papers:
                    papers[oid] = text
    print(f"  Loaded {len(papers)} unique papers from raw files.")

    # Load scores and IDs
    print("Loading paper_scores.npy and paper_scores_ids.json ...")
    scores = np.load(DATA / "embeddings" / "papers.npy" if not (DATA / "paper_scores.npy").exists()
                     else DATA / "paper_scores.npy")
    # Prefer paper_scores.npy (17 SDG scores); fall back to raw embeddings
    score_path = DATA / "paper_scores.npy"
    if not score_path.exists():
        print("ERROR: data/paper_scores.npy not found. Run alignment_score.py first.")
        return

    scores = np.load(score_path)  # (N, 17)
    with open(DATA / "paper_scores_ids.json") as f:
        ids = json.load(f)

    n_papers = len(ids)
    assert scores.shape[0] == n_papers, f"Shape mismatch: {scores.shape[0]} vs {n_papers}"

    # Hard assignment: SDG index = argmax (0-indexed → SDG = i+1)
    assignments = scores.argmax(axis=1)  # 0-indexed
    sdg4_mask = assignments == 3  # SDG 4 is index 3 (0-indexed)
    sdg4_indices = np.where(sdg4_mask)[0]
    n_sdg4 = int(sdg4_mask.sum())

    print(f"Total papers: {n_papers}")
    print(f"Papers assigned to SDG 4: {n_sdg4} ({n_sdg4/n_papers*100:.1f}%)")

    # Classify each SDG 4 paper
    categories = {"genuine_education": 0, "ml_artefact": 0, "ambiguous": 0}
    category_ids = {"genuine_education": [], "ml_artefact": [], "ambiguous": []}

    for idx in sdg4_indices:
        paper_id = ids[idx]["id"]
        text = papers.get(paper_id, "")
        cat = classify_paper(text)
        categories[cat] += 1
        category_ids[cat].append(paper_id)

    print("\nSDG 4 classification breakdown:")
    for cat, count in categories.items():
        pct = count / n_sdg4 * 100 if n_sdg4 > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}% of SDG 4 papers, {count/n_papers*100:.2f}% of total)")

    # Revised coverage profile: reassign ml_artefact papers to their 2nd-best SDG
    print("\nComputing revised coverage profile (ml_artefact → 2nd-best SDG) ...")
    revised_assignments = assignments.copy()
    n_reassigned = 0

    for idx in sdg4_indices:
        paper_id = ids[idx]["id"]
        text = papers.get(paper_id, "")
        cat = classify_paper(text)
        if cat == "ml_artefact":
            # Find 2nd-best SDG
            row = scores[idx].copy()
            row[3] = -999  # mask SDG 4
            revised_assignments[idx] = row.argmax()
            n_reassigned += 1

    # Compute revised proportions
    original_proportions = {}
    revised_proportions = {}
    for sdg_idx in range(17):
        original_proportions[f"SDG{sdg_idx+1}"] = float((assignments == sdg_idx).sum() / n_papers)
        revised_proportions[f"SDG{sdg_idx+1}"] = float((revised_assignments == sdg_idx).sum() / n_papers)

    print(f"\nReassigned {n_reassigned} ml_artefact papers from SDG 4 to their 2nd-best SDG.")
    print("\nRevised SDG proportions (top 5 changed):")
    changes = {k: revised_proportions[k] - original_proportions[k] for k in original_proportions}
    for k, delta in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {k}: {original_proportions[k]*100:.1f}% → {revised_proportions[k]*100:.1f}% (Δ{delta*100:+.1f}%)")

    # Save results
    output = {
        "n_papers_total": n_papers,
        "n_sdg4_assigned": n_sdg4,
        "sdg4_pct_of_total": round(n_sdg4 / n_papers * 100, 1),
        "classification_breakdown": {
            cat: {
                "count": count,
                "pct_of_sdg4": round(count / n_sdg4 * 100, 1) if n_sdg4 > 0 else 0,
                "pct_of_total": round(count / n_papers * 100, 2),
            }
            for cat, count in categories.items()
        },
        "n_reassigned_to_2nd_best": n_reassigned,
        "original_research_proportions": original_proportions,
        "revised_research_proportions": revised_proportions,
        "methodology_note": (
            "Papers hard-assigned to SDG 4 are classified as 'genuine_education' if "
            "education keywords are present, 'ml_artefact' if ML methodology keywords "
            "are present but education keywords are absent, and 'ambiguous' otherwise. "
            "Revised proportions re-assign ml_artefact papers to their second-best SDG "
            "centroid. This is a diagnostic estimate; the keyword rule is heuristic and "
            "should be validated by qualitative inspection."
        ),
    }

    out_path = DATA / "sdg4_artefact_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {out_path}")
    print("\nKey numbers for dissertation text:")
    ml_artefact = categories["ml_artefact"]
    genuine = categories["genuine_education"]
    ambig = categories["ambiguous"]
    print(f"  Of {n_sdg4} SDG 4 papers:")
    print(f"    Genuine education: {genuine} ({genuine/n_sdg4*100:.0f}%)")
    print(f"    ML artefact:       {ml_artefact} ({ml_artefact/n_sdg4*100:.0f}%)")
    print(f"    Ambiguous:         {ambig} ({ambig/n_sdg4*100:.0f}%)")


if __name__ == "__main__":
    main()
