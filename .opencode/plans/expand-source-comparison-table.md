# Expand source comparison: add coverage + semantic gap with cache

## Goal

Add coverage (research paper count per SDG) and semantic gap to the per-source
comparison table, alongside existing F1 and centroid cosine. Two side-by-side
portrait tables, each 9 columns (SDG + 4 sources × 2 metrics).

## Key design decisions

- **Combined re-computed from scratch** using the same code path as alternatives,
  ensuring consistency. Cache makes this a one-time cost.
- **Cache root**: `2_data/3_scored/source_comparison_cache/` — stores per-source
  research counts/sums and policy counts/sums as small `.npy` files. Validated
  by input-file mtime comparison.
- **Two output table files**: `tab_sdg_source_comparison_f1cos.tex` and
  `tab_sdg_source_comparison_covgap.tex`.

## Script changes (`8_sdg_source_comparison.py`)

### New functions

```python
def score_research_full(centroids, research_shards):
    """Vectorised: scores/argmax per shard, accumulate counts + embedding sums per SDG.
    Returns (counts[17], sums[17, 384])."""

def score_policy_full(centroids, policy_emb, policy_ids, chunk_cap=50):
    """Same pattern + chunk capping per doc before accumulating.
    Returns (counts[17], sums[17, 384])."""

def compute_or_load_research(source_name, centroids, research_shards, *, overwrite, cache_dir):
    """Check cache manifest mtimes → load or compute → cache."""

def compute_or_load_policy(source_name, centroids, policy_emb, policy_ids, *, overwrite, cache_dir):
    """Same for policy."""

def normalize_sums(counts, sums, min_cluster=10):
    """Unit-normalise sums → (17, 384) centroids. NaN rows where count < min_cluster."""
```

### Main loop

```python
sources = [
    ("combined",  combined_centroids, combined_counts),
    ("osdg",      osdg_centroids,     osdg_counts),
    ("sdgi",      sdgi_centroids,     sdgi_counts),
    ("knowledgehub", kh_centroids,    kh_counts),
]

for source_name, centroids, n_counts in sources:
    # F1 — cheap
    f1[source_name] = compute_validation_f1(centroids, bench_emb, bench_ids)

    # Cosine to combined — cheap
    cos[source_name] = per_sdg_cosine(centroids, combined_centroids)

    # Research scores — cached
    rc, rs = compute_or_load_research(source_name, centroids, research_shards, ...)
    research_centroids[source_name] = normalize_sums(rc, rs)
    coverage[source_name] = rc

    # Policy scores — cached
    pc, ps = compute_or_load_policy(source_name, centroids, policy_emb, policy_ids, ...)
    policy_centroids[source_name] = normalize_sums(pc, ps)

    # Semantic gap
    sim = (research_centroids[source_name] * policy_centroids[source_name]).sum(axis=1)
    gap[source_name] = np.where(~np.isnan(sim), 1.0 - sim, np.nan)
```

### LaTeX generators

**`write_table_f1cos()`**: 9 columns — F1 and cos per source (same as current).
**`write_table_covgap()`**: 9 columns — coverage (comma-int) and gap (3-dec).
**`write_num_tex()`**: Extended with `\Src{Word}{Source}Cov` and
  `\Src{Word}{Source}Gap` macros.

### Cache implementation

```python
CACHE_DIR = Path("2_data/3_scored/source_comparison_cache")

def _cache_paths(source_name):
    return {
        "research_counts": CACHE_DIR / f"{source_name}_research_counts.npy",
        "research_sums": CACHE_DIR / f"{source_name}_research_sums.npy",
        "policy_counts": CACHE_DIR / f"{source_name}_policy_counts.npy",
        "policy_sums": CACHE_DIR / f"{source_name}_policy_sums.npy",
        "manifest": CACHE_DIR / f"{source_name}_manifest.json",
    }

def _input_mtimes(research_shards, policy_emb, source_emb_path, centroids_path):
    """Dict of {str(path): mtime} for all inputs that affect results."""
    paths = [policy_emb, source_emb_path, centroids_path]
    paths += [s["embedding_path"] for s in research_shards]
    return {str(p): os.path.getmtime(p) for p in paths}

def _cache_valid(paths, input_mtimes):
    m = paths["manifest"]
    if not m.exists():
        return False
    manifest = json.loads(m.read_text())
    return manifest["input_mtimes"] == input_mtimes

def compute_or_load_research(source_name, centroids, research_shards, *, overwrite, cache_dir):
    paths = _cache_paths(source_name)
    inputs = _input_mtimes(research_shards, ...)

    if not overwrite and _cache_valid(paths, inputs):
        return np.load(paths["research_counts"]), np.load(paths["research_sums"])

    counts, sums = score_research_full(centroids, research_shards)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths["research_counts"], counts)
    np.save(paths["research_sums"], sums)
    paths["manifest"].write_text(json.dumps({
        "source": source_name, "type": "research",
        "input_mtimes": inputs, "computed_at": datetime.now().isoformat(),
    }))

    return counts, sums
```

## Cost estimate

| Step | CPU/IO | Time |
|---|---|---|
| Load 8 research shards (once) | 368 MB I/O | ~15s |
| Matmul per source per shard (3×) | ~200M FLOPs × 3 | ~0.3s |
| Policy load + 3 matmuls | 230 MB + 3×1B FLOPs | ~10s |
| F1, cosines, gaps | trivial | ~0.5s |
| **Total first run** | | **~30s** |
| **Cache hit** (load 8 small .npy) | 8×52KB | **<0.1s** |

## Appendix A.5 text changes

The current text references one table. After the change, reference both tables,
e.g.:
```
Table~\ref{tab:sdg-source-comparison-f1cos} reports per-SDG validation F1 and
centroid cosine similarity for each source. Table~\ref{tab:sdg-source-comparison-covgap}
reports the corresponding research coverage and semantic gap.
```

All existing `\Src{Word}{Source}FOne` and `\Src{Word}{Source}Cosine` macros
continue to work. New `\Src{Word}{Source}Cov` and `\Src{Word}{Source}Gap`
macros are added but not required by any existing prose.
