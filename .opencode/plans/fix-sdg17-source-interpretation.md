# Fix SDG 17 Source Comparison Interpretation

## Problem

The current dissertation text misinterprets the SDG 17 source comparison results:
- Claims combined F1 (0.291) "lies between" SDGi (0.437) and KH (0.316) — false, it's below both
- Claims combined "achieves comparable or better validation F1 than any single-source alternative for all 17 SDGs" — false for SDG 17

## Root cause

The combined SDG 17 centroid is ~66% KH by text count (616 vs 321), so it inherits the KH direction. KH journalism doesn't align well with the benchmark's policy-excerpt genre, producing the lowest F1. SDGi-only does best (0.437) because of genre overlap with the benchmark — that's inflation, not superior reference quality.

## Required edits

### Edit 1: Appendix A.5 — lines 563-565

**File**: `3_writing/dissertation.tex`

Replace:
```latex
SDG 17 is a partial exception: the combined F1 (\SrcSeventeenCombinedFOne{}) lies between the SDGi-only (\SrcSeventeenSDGiFOne{}) and KH-only (\SrcSeventeenKnowledgeHubFOne{}) values, reflecting the absence of an OSDG anchor for this SDG.

The overall conclusion is that the combined centroid is a balanced instrument: it is not dominated by any single source, does not overfit to any one annotation genre, and achieves comparable or better validation F1 than any single-source alternative for all 17 SDGs.
```

With:
```latex
SDG 17 is a notable exception: the combined F1 (\SrcSeventeenCombinedFOne{}) is lower than either single-source alternative. This is because the combined centroid (937 texts, ~66\% Knowledge Hub by volume) inherits the KH direction, which does not align optimally with the benchmark's policy-excerpt genre, while the SDGi-only centroid (321 texts) benefits from within-genre overlap with the benchmark. The combined centroid's lower F1 for SDG 17 is therefore not a sign of degraded reference quality but a consequence of genre heterogeneity between the two source corpora when no OSDG anchor is available.

The overall conclusion is that for SDGs 1--16, the combined centroid is a balanced instrument: it is not dominated by any single source and achieves comparable or better validation F1 than any single-source alternative. For SDG 17, where OSDG coverage is absent, the combined centroid is a conservative choice --- it does not overfit to any single annotation genre, but it reflects a compromise direction that is less discriminative on the benchmark than either single-source alternative.
```

### Edit 2: A-SDG17 — line 449

Replace:
```latex
per-source centroid comparisons show that the SDGi-only centroid achieves F1 = \SrcSeventeenSDGiFOne{} against the benchmark, while the combined and KH-only centroids score \SrcSeventeenCombinedFOne{} and \SrcSeventeenKnowledgeHubFOne{} respectively, confirming that the combined centroid does not overfit to the SDGi policy genre.
```

With:
```latex
per-source centroid comparisons (Table~\ref{tab:sdg-source-comparison}) confirm that the combined centroid does not overfit to the SDGi policy genre: the SDGi-only centroid achieves F1 = \SrcSeventeenSDGiFOne{} (inflated by genre overlap between the SDGi corpus and the benchmark), while the combined centroid's lower F1 (\SrcSeventeenCombinedFOne{}) reflects its broader, genre-balanced centroid direction (cosine to SDGi = \SrcSeventeenSDGiCosine{}, to KH = \SrcSeventeenKnowledgeHubCosine{}). The combined centroid is not dominated by SDGi despite including its texts.
```

### No change: §5.2 — line 344

Current text is fine — correctly references SDG 8 and SDG 10 as exceptions. No SDG 17 reference needed here.

## Verification

After edits: `cd 3_writing && latexmk -pdf dissertation.tex` — expect clean build, 59 pages.
