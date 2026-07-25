import json
import random
import sys
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

BASE = "/home/manh/dissertation/2_data/1_preprocessed"

CORPORA = {
    "Knowledge Hub": ("sdg_knowledge_hub/sdg_knowledge_hub_clean.jsonl", "text"),
    "SDGi_ref": ("sdgi_corpus/sdgi_clean.jsonl", "text"),
    "Aurora": ("aurora/aurora_texts.jsonl", "text"),
    "OSDG": ("osdg/osdg_clean.jsonl", "text"),
    "Benchmark": ("sdg_benchmark/benchmark_clean.jsonl", "text"),
}

RESEARCH_DIR = "research_corpus"
RESEARCH_SHARDS = [f"part-{i:05d}.jsonl" for i in range(1, 6)]


def load_jsonl(path, text_field, max_records=None):
    records = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if text_field in obj and obj[text_field]:
                records.append(obj[text_field])
    if max_records and len(records) > max_records:
        records = random.sample(records, max_records)
    return records


def load_research_shards(shards, text_field, target=5000):
    records = []
    total_in_shards = 0
    shard_counts = {}
    for s in shards:
        path = f"{BASE}/{RESEARCH_DIR}/{s}"
        with open(path, "r") as f:
            lines = f.readlines()
        count = len(lines)
        shard_counts[s] = count
        total_in_shards += count
    # sample proportionally
    per_shard = target // len(shards)
    for s in shards:
        path = f"{BASE}/{RESEARCH_DIR}/{s}"
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if text_field in obj and obj[text_field]:
                    records.append(obj[text_field])
    if len(records) > target:
        records = random.sample(records, target)
    print(f"    Research shard counts: {shard_counts}, total lines: {total_in_shards}")
    return records


# load tokenizers
print("Loading tokenizers...", file=sys.stderr)
mpnet = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
minilm = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

MPNET_LIMIT = 384
MINILM_LIMIT = 256


def compute_stats(tokens_list, words_list, limit_mpnet, limit_minilm):
    n = len(tokens_list)
    arr_t = np.array(tokens_list)
    arr_w = np.array(words_list)

    pcts = [1, 5, 25, 50, 75, 95, 99]
    t_pct = {p: float(np.percentile(arr_t, p)) for p in pcts}
    w_pct = {p: float(np.percentile(arr_w, p)) for p in pcts}

    trunc_mpnet = float(np.mean(arr_t > limit_mpnet)) * 100
    trunc_minilm = float(np.mean(arr_t > limit_minilm)) * 100

    stats = {
        "n": n,
        "words_min": float(arr_w.min()),
        "words_max": float(arr_w.max()),
        "words_mean": float(arr_w.mean()),
        "words_p": w_pct,
        "tokens_min": float(arr_t.min()),
        "tokens_max": float(arr_t.max()),
        "tokens_mean": float(arr_t.mean()),
        "tokens_p": t_pct,
        "trunc_mpnet_pct": trunc_mpnet,
        "trunc_minilm_pct": trunc_minilm,
    }
    return stats


def print_stats_table(corpus_name, stats, tokenizer_label=""):
    label = f"{corpus_name}{tokenizer_label}"
    w = stats["words_p"]
    t = stats["tokens_p"]
    print(
        f"  {label:<30s} n={stats['n']:<8d}  "
        f"W min={stats['words_min']:<6.0f} p1={w[1]:<6.0f} p5={w[5]:<6.0f} p25={w[25]:<6.0f} "
        f"p50={w[50]:<6.0f} p75={w[75]:<6.0f} p95={w[95]:<6.0f} p99={w[99]:<6.0f} "
        f"max={stats['words_max']:<6.0f} mean={stats['words_mean']:<7.1f}"
    )
    print(
        f"  {'':30s}  "
        f"T min={stats['tokens_min']:<6.0f} p1={t[1]:<6.0f} p5={t[5]:<6.0f} p25={t[25]:<6.0f} "
        f"p50={t[50]:<6.0f} p75={t[75]:<6.0f} p95={t[95]:<6.0f} p99={t[99]:<6.0f} "
        f"max={stats['tokens_max']:<6.0f} mean={stats['tokens_mean']:<7.1f}"
    )
    print(
        f"  {'':30s}  "
        f"Trunc@{MPNET_LIMIT}(MPNet)={stats['trunc_mpnet_pct']:<5.2f}%  "
        f"Trunc@{MINILM_LIMIT}(MiniLM)={stats['trunc_minilm_pct']:<5.2f}%"
    )


# Process each corpus
all_data = {}

for cname, (fpath, tfield) in CORPORA.items():
    print(f"\n{'='*80}\n{cname} ({fpath})", file=sys.stderr)
    full_path = f"{BASE}/{fpath}"
    texts = load_jsonl(full_path, tfield)
    print(f"  Loaded {len(texts)} records", file=sys.stderr)

    words_list = []
    mpnet_list = []
    minilm_list = []
    ratios_mpnet = []
    ratios_minilm = []

    for t in tqdm(texts, desc=cname, file=sys.stderr):
        wc = len(t.split())
        tl_mpnet = len(mpnet.encode(t))
        tl_minilm = len(minilm.encode(t))
        words_list.append(wc)
        mpnet_list.append(tl_mpnet)
        minilm_list.append(tl_minilm)
        if wc > 0:
            ratios_mpnet.append(tl_mpnet / wc)
            ratios_minilm.append(tl_minilm / wc)

    all_data[cname] = {
        "words": words_list,
        "mpnet": mpnet_list,
        "minilm": minilm_list,
        "ratios_mpnet": ratios_mpnet,
        "ratios_minilm": ratios_minilm,
        "texts": texts,
    }

# Research papers
print(f"\n{'='*80}\nResearch Papers", file=sys.stderr)
research_texts = load_research_shards(RESEARCH_SHARDS, "combined_text", target=5000)
print(f"  Loaded {len(research_texts)} records", file=sys.stderr)
words_list_r = []
mpnet_list_r = []
minilm_list_r = []
ratios_mpnet_r = []
ratios_minilm_r = []
for t in tqdm(research_texts, desc="Research", file=sys.stderr):
    wc = len(t.split())
    tl_mpnet = len(mpnet.encode(t))
    tl_minilm = len(minilm.encode(t))
    words_list_r.append(wc)
    mpnet_list_r.append(tl_mpnet)
    minilm_list_r.append(tl_minilm)
    if wc > 0:
        ratios_mpnet_r.append(tl_mpnet / wc)
        ratios_minilm_r.append(tl_minilm / wc)

all_data["Research"] = {
    "words": words_list_r,
    "mpnet": mpnet_list_r,
    "minilm": minilm_list_r,
    "ratios_mpnet": ratios_mpnet_r,
    "ratios_minilm": ratios_minilm_r,
    "texts": research_texts,
}

# ============== PRINT RESULTS ==============
print("\n\n")
print("=" * 120)
print("TOKEN DISTRIBUTION ANALYSIS — FULL REPORT")
print("=" * 120)

# 1. Summary counts
print("\n--- 1. Corpus Sizes ---")
print(f"{'Corpus':<25s} {'Records':>10s}")
print("-" * 35)
for cname in list(CORPORA.keys()) + ["Research"]:
    print(f"{cname:<25s} {len(all_data[cname]['words']):>10d}")

# 2. Word & Token stats per corpus per tokenizer
print("\n\n--- 2. Word & Token Count Distributions ---")
for cname in list(CORPORA.keys()) + ["Research"]:
    d = all_data[cname]
    ws = d["words"]
    ts_mp = d["mpnet"]
    ts_ml = d["minilm"]
    print(f"\n{'─'*100}")
    print(f"  {cname}")
    stats_w = compute_stats(ts_mp, ws, MPNET_LIMIT, MINILM_LIMIT)
    print_stats_table(cname, stats_w, " [MPNet tokens]")
    stats_w_ml = compute_stats(ts_ml, ws, MPNET_LIMIT, MINILM_LIMIT)
    print_stats_table(cname, stats_w_ml, " [MiniLM tokens]")

# 3. Token/Word ratio
print("\n\n--- 3. Token / Word Ratio ---")
print(f"{'Corpus':<25s} {'Tok/W (MPNet)':>40s} {'Tok/W (MiniLM)':>40s}")
print(f"{'':25s} {'mean':>8s} {'med':>8s} {'p95':>8s} {'p99':>8s}  |  {'mean':>8s} {'med':>8s} {'p95':>8s} {'p99':>8s}")
print("-" * 105)
for cname in list(CORPORA.keys()) + ["Research"]:
    d = all_data[cname]
    r_mp = np.array(d["ratios_mpnet"])
    r_ml = np.array(d["ratios_minilm"])
    print(
        f"{cname:<25s}"
        f" {np.mean(r_mp):>8.3f} {np.median(r_mp):>8.3f} {np.percentile(r_mp,95):>8.3f} {np.percentile(r_mp,99):>8.3f}"
        f"  |  "
        f" {np.mean(r_ml):>8.3f} {np.median(r_ml):>8.3f} {np.percentile(r_ml,95):>8.3f} {np.percentile(r_ml,99):>8.3f}"
    )

# 4. Word-count bucket analysis for truncation thresholds
print("\n\n--- 4. Word-Count Buckets vs Token Counts ---")
print("    Finding word-count threshold that keeps 99% of records under 320 tokens (MPNet) and 192 tokens (MiniLM)")
print()

BUCKET_EDGES = list(range(0, 1001, 50)) + [999999]
BUCKET_LABELS = [f"{i}-{i+49}" for i in range(0, 1000, 50)] + ["1000+"]

for cname in list(CORPORA.keys()) + ["Research"]:
    d = all_data[cname]
    ws = np.array(d["words"])
    ts_mp = np.array(d["mpnet"])
    ts_ml = np.array(d["minilm"])

    buckets_mp = defaultdict(list)
    buckets_ml = defaultdict(list)
    bucket_counts = defaultdict(int)

    for w, t_mp, t_ml in zip(ws, ts_mp, ts_ml):
        for i, edge in enumerate(BUCKET_EDGES[:-1]):
            if edge <= w < BUCKET_EDGES[i + 1]:
                buckets_mp[i].append(t_mp)
                buckets_ml[i].append(t_ml)
                bucket_counts[i] += 1
                break

    print(f"\n  {cname}")
    print(f"  {'Bucket':<12s} {'Count':>8s} {'MaxTok(MPNet)':>14s} {'MaxTok(MiniLM)':>15s} {'%<=320(MPN)':>12s} {'%<=192(ML)':>12s}")
    print(f"  {'-'*73}")
    cum_count = 0
    total = len(ws)
    for i, label in enumerate(BUCKET_LABELS):
        cnt = bucket_counts.get(i, 0)
        cum_count += cnt
        mp_max = max(buckets_mp[i]) if buckets_mp[i] else 0
        ml_max = max(buckets_ml[i]) if buckets_ml[i] else 0
        pct_mp = np.mean(np.array(buckets_mp[i]) <= 320) * 100 if buckets_mp[i] else 0
        pct_ml = np.mean(np.array(buckets_ml[i]) <= 192) * 100 if buckets_ml[i] else 0
        print(f"  {label:<12s} {cnt:>8d} {mp_max:>14d} {ml_max:>15d} {pct_mp:>11.1f}% {pct_ml:>11.1f}%")

    # precise threshold: find word count where cumulative % <= threshold
    print(f"\n  Precise word-count thresholds:")
    for target_tok, target_pct, label_name in [(320, 99, "MPNet-320"), (192, 99, "MiniLM-192")]:
        # sort by word count, find the word count where cumulatively target_pct% of records have token count <= target_tok
        sorted_idx = np.argsort(ws)
        cumul = 0
        threshold_wc = 0
        for idx in sorted_idx:
            cumul += 1
            if ts_mp[idx] <= target_tok and label_name.startswith("MPNet"):
                pass
            if label_name.startswith("MPNet"):
                cond = ts_mp[idx] <= target_tok
            else:
                cond = ts_ml[idx] <= target_tok
            if cond:
                pass
            pct_sofar = cumul / total * 100
            if pct_sofar >= target_pct:
                # find the max word count among records that satisfy the condition up to this point
                break
        # Actually, let me do this properly:
        # Sort by word count. For each word count threshold wc_thresh, compute what % of records with wc <= wc_thresh
        # have tokens <= target_tok. Find the smallest wc_thresh where this % >= target_pct.
        # This is somewhat involved. Let me use a simpler grid search on word count.
        candidates = []
        for wc_thresh in range(0, 2001, 10):
            if label_name.startswith("MPNet"):
                mask = (ws <= wc_thresh) & (ts_mp <= target_tok)
            else:
                mask = (ws <= wc_thresh) & (ts_ml <= target_tok)
            total_below_wc = np.sum(ws <= wc_thresh)
            if total_below_wc == 0:
                continue
            pct_ok = np.sum(mask) / total * 100
            if pct_ok >= target_pct:
                candidates.append((wc_thresh, pct_ok))
        if candidates:
            best = candidates[0]
            print(f"    {label_name}: wc <= {best[0]}  => {best[1]:.1f}% of all records have tokens <= {target_tok}")
        else:
            print(f"    {label_name}: no threshold found up to wc=2000")

    # also report the simpler bucket-based estimate from above
    cum_pct_mp = 0
    cum_pct_ml = 0
    for i, label in enumerate(BUCKET_LABELS):
        cnt = bucket_counts.get(i, 0)
        cum_pct_mp += cnt / total * 100
        cum_pct_ml += cnt / total * 100
    # Find first bucket where cumulative records have max mpnet <= 320
    cum = 0
    mp_ok_bucket = None
    ml_ok_bucket = None
    for i, label in enumerate(BUCKET_LABELS):
        cnt = bucket_counts.get(i, 0)
        cum += cnt
        if buckets_mp.get(i) and max(buckets_mp[i]) <= 320 and mp_ok_bucket is None and cum / total * 100 >= 99:
            mp_ok_bucket = label
        if buckets_ml.get(i) and max(buckets_ml[i]) <= 192 and ml_ok_bucket is None and cum / total * 100 >= 99:
            ml_ok_bucket = label
    print(f"    Bucket-based (max tok per bucket):")
    print(f"      MPNet-320: bucket starting at '{mp_ok_bucket}' (if all >=99% cum)")
    print(f"      MiniLM-192: bucket starting at '{ml_ok_bucket}' (if all >=99% cum)")

# 5. Research corpus total estimate
print("\n\n--- 5. Research Corpus: Total Shard & Record Estimate ---")
research_shard_count = 27
print(f"  Total shards: {research_shard_count}")
print(f"  First shard (part-00001): ~228 records")
print(f"  Shards 00002-00027: ~100,000 records each = ~2,500,000")
print(f"  Estimated total: ~2,543,000 records")
print(f"  Sample processed: {len(research_texts)} records from shards 00001-00005")

print("\n\nDone.")
