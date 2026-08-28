# Register-adjustment validity inspection — synthesis (scratch, MPNet-canonical)

**Branch:** `register-validity-inspection`
**Scope:** scratch-only. No edits to `dissertation.tex`, the a1 script, or `register_adjust.py`.
**Encoder:** `all-mpnet-base-v2` (canonical). `G` loaded from `2_data/3b_register/mpnet/canon/G.npy` (62×768, already frozen). No re-embed, no INLP re-run.
**Determinism:** `SEED = 42` everywhere. Frozen encoder.

---

## 1. The claim under test

The dissertation reads the INLP-removed subspace `G` as **register** (writing style),
so what remains is **topic**. Identifying assumption (`register_adjust.py:521–542`):
within an SDG, research≠policy only by register, so each corpus-discrimination
direction `g_k` cannot absorb topic. Appendix a1 currently states this is
*"consistent with, not confirmed by"* (`dissertation.tex:886`).

### Threats that would falsify the reading
- **T1 — topic leakage past SDG stratification.** The LR direction is fit on the
  *full pooled sample* to predict corpus (lines 539–540); the 34-class
  `stratify_key` governs only the train/test *split*, not the fit objective. If
  research/policy differ in topic even within an SDG, `g_k` absorbs it.
- **T2 — `g_k` is just any corpus-correlated direction**; its top words could be
  topic words, not register words.
- **T3 — `G` also carries topic**, silently damaging topic (appendix's unchanged
  SDG selectivity is necessary, not sufficient).

---

## 2. What each test found

### P0a — Curated word probe + permutation null (`word_probe_curated.py`)
Balanced register / topic / neutral / corpus-leaning bins (≈90 words), encoded as
single-word embeddings, aligned with each `g_k` by cosine.

- Bin means: policy-lean +0.0074, register +0.0008, research-lean −0.0021,
  topic −0.0032, neutral −0.0039.
- Register − Topic gap = **+0.0040**; permutation null (5000) two-sided **p = 0.186**.
- **Conclusion:** at the single-word level the subspace does *not* significantly
  favour register over topic vocabulary. Surface register words (hedges, modals,
  nominalizations) do not stand out. Inconclusive for the register reading.

### P0b — Unbiased full-vocabulary scan (`word_probe_fullvocab.py`)
234,349 English words encoded; each word scored by `‖projection onto span(G)‖`,
compared to a random 62-dim subspace (expected score √(62/768) ≈ 0.284).

- G: max=0.527, p99=0.393, median=0.302. Random: max=0.401, p99=0.348, median=0.292.
- G's **top ~1% of words sit clearly above the random null** (max 0.527 vs 0.401);
  the bulk of the distribution is at chance.
- G's top words form a coherent **socio-economic / governance / development /
  policy-domain** cluster: `economy, economically, liberalization, infrastructure,
  governance, legislation, poverty, prosperity, socioeconomic, nationalize,
  decentralization, industrialization, socialism, capitalism, bureaucracy, welfare,
  sustainable, government, environmentalism, famine, pollution, livelihood …`
  plus a dense tail of `development`-family morphological variants
  (`developmentary, developmentist, superdevelopment, reprivatization …`) and some
  AI/technical terms (`ai, convolutional, cybernetics`).
- **SDG leaf-topics (climate / water / health / biodiversity …) are absent from the
  top 200** — consistent with the appendix's topic-preservation result.

**Conclusion:** the subspace is *not* aligned with arbitrary words and *not* with
SDG leaf-topics; it is specifically aligned with a **policy-subject-matter /
socio-economic-governance vocabulary** — the lexical signature of the policy corpus
genre, not pure surface-linguistic register (hedges/modals).

### P1 — Topic-direction ⊥ span(G) overlap (`p1_p2_direction_validity.py`)
Independent 17-class SDG-topic LR on raw embeddings → weight vectors `w_sdg`.
Alignment = `‖G·w‖/‖w‖` (G orthonormal rows). Compared to the corpus(register)
direction and to a random 62-dim subspace.

- Topic coef alignment with span(G): **mean 0.248, max 0.430**.
- Topic coef alignment with a RANDOM 62-dim subspace: **mean 0.286** (expected 0.284).
- Corpus(register) coef alignment with span(G): **0.979** (vs random 0.242) — sanity ✓.

**Conclusion:** topic directions are, if anything, *less* aligned with span(G) than a
random subspace would be. `G` did **not** preferentially remove SDG topic; it is the
corpus/register subspace (sanity direction sits at 0.979 inside it).

### P2 — Held-out-SDG generalization (key falsification test for T1)
Leave-one-SDG-out: train a corpus LR on the other 16 SDGs, test on the held-out
SDG's research/policy segments. Register is a corpus property ⇒ should generalize
across topics; topic leakage would be SDG-specific ⇒ would collapse.

- Held-out-SDG corpus accuracy: **mean 0.967** (full-data 0.978).
- Held-out corpus directions' alignment with span(G): mean 0.981.
- **Topic control:** a SDG1-vs-SDG2 direction trains at 0.991 but classifies *other*
  SDGs' research at only **0.781** (chance 0.5) — topic generalizes far less.
- **Balanced data-size control:** a register direction trained on ONLY SDG1+SDG2
  reaches train 0.976 and *still* classifies the other 15 SDGs at **0.920** —
  so the 0.967 is not merely a training-mass artifact.

**Conclusion:** the removed register direction generalizes across *unseen topics*;
the T1 falsification test **passes**. The subspace is topic-independent at the
SDG level. (Topic direction also shows weak cross-SDG generalization at 0.781, but
far below register's 0.967 — the contrast is the point.)

---

## 3. Synthesis

| Test | Question | Result | Supports register reading? |
|------|----------|--------|----------------------------|
| P0a curated | Do single words favour register over topic? | gap +0.004, p=0.186 (n.s.) | No (inconclusive) |
| P0b full-vocab | What vocabulary lies on span(G)? | policy/socio-economic/governance cluster; no SDG leaf-topics | Partially — genre/domain lexicon, not surface register |
| P1 topic⊥G | Does G remove SDG topic? | topic align 0.248 < random 0.286; corpus 0.979 | **Yes** |
| P2 held-out | Is the direction topic-independent? | held-out acc 0.967; topic control 0.781 | **Yes** (falsifies T1) |

**Overall:** The removed subspace is a **corpus/genre-distinguishing, topic-independent**
direction (P1/P2 decisive), but its lexical signature is a **socio-economic /
governance / policy-domain vocabulary**, not the surface Biber features (hedges,
deontic modals, nominalizations) that Appendix a1's manual score targets. So:

- The "register" reading is **strengthened** on the structural axis: what is removed
  is corpus-distinguishing and generalizes across SDG topics, and does not encode
  SDG topic structure.
- It is **qualified** on the lexical axis: the removed component is more a
  *policy-subject-matter / genre* axis than a pure grammatical register. The
  manual 6-feature register score in a1 is therefore not expected to track it
  closely (already reported: ρ≈0.10) — and that mismatch is now explained, not
  just noted.

This keeps the manuscript's honest *"consistent with, not confirmed by"* stance,
but the consistency evidence is now substantially stronger than the a1 text
currently reflects. The three new facts worth surfacing in a1 (if ever promoted):
(1) SDG-topic directions are orthogonal to span(G); (2) the register direction
generalizes across held-out SDGs (0.967); (3) the subspace's lexical signature is
a policy/socio-economic governance vocabulary, absent SDG leaf-topics.

---

## 4. Limitations of this inspection (carry forward honestly)
- Single-word embeddings are degenerate in MPNet space; P0a's null result is about
  word types, not how words function in context. P0b and P1/P2 operate on the
  segment/direction level where the signal lives, and they agree.
- `G`'s rows are Gram–Schmidt orthogonalized; per-direction interpretation is
  unreliable, so aggregate statistics (mean alignment, held-out accuracy) are used.
- MPNet-canonical only (matches a1's stated limitation #5). MiniLM/SciBERT not
  tested — a separate encoder-sensitivity question.
- Dense subspace shows *association*, never literal "words on it"; the lexical
  reading of P0b is a proxy, not proof of mechanism.

---

## 5. Reproduce
```
conda activate dissertation
# P0a (fast)
python 5_notes/register_validity/word_probe_curated.py
# P0b (long; ~234k words — run under tmux)
tmux new-session -d -s fv "conda run -n dissertation python 5_notes/register_validity/word_probe_fullvocab.py > 5_notes/register_validity/fullvocab.log 2>&1; touch 5_notes/register_validity/fullvocab.DONE"
# P1+P2 (loads frozen embeddings; run under tmux)
tmux new-session -d -s p1p2 "conda run -n dissertation python 5_notes/register_validity/p1_p2_direction_validity.py > 5_notes/register_validity/p1p2.log 2>&1; touch 5_notes/register_validity/p1p2.DONE"
```
Outputs: `p0_curated_probe.{md,json}`, `p0_fullvocab_scan.{md,json}`,
`p1_p2_direction_validity.{md,json}` — all under `5_notes/register_validity/`.

**P3 (NN-shift illustration) was scoped optional and not run** — P1/P2 already
answer the validity question; P3 would only be a presentational figure.

---

## 6. Algorithmic-fidelity check against Ravfogel's own code (added)

To close the gap flagged earlier ("we did not run Ravfogel's code; G was produced
by our own re-implementation"), we cloned `https://github.com/shauli-ravfogel/nullspace_projection`
to **`/home/manh/nullspace_projection`** (pure scratch, *outside* `dissertation/`,
never imported into the pipeline — so "our G was produced by our own code" stays true
in the manuscript text). We ran BOTH implementations on one synthetic toy set
(5 orthogonal signal dims in 40D, strong, K=5 directions) and compared:

- `our_inlp` — a faithful replica of `register_adjust.py`'s loop (project X onto
  nullspace of accumulated G, fit LR, Gram–Schmidt, accumulate).
- Ravfogel's `get_debiasing_projection(LogisticRegression, is_autoregressive=True,
  min_accuracy=0.0, num_classifiers=5)`.

**Result (`verify_toy_results.json`):** principal-angle cosines between
`span(our_G)` and `span(their Ws)` = **[1.0, 1.0, 1.0, 1.0, 1.0]** (subspaces
identical); fresh-classifier accuracy after our projection = 0.515, after their `P`
= 0.515 (both ≈ chance; original = 0.815). **VERDICT: MATCH.**

**Implication:** our INLP implementation is algorithmically equivalent to Ravfogel
et al.'s. The removed subspace `G` we inspected is a legitimate INLP projection, so
the P1/P2 validity results are trustworthy as INLP results. This resolves the
fidelity caveat; no re-run of P1/P2 against their code is needed.

**Scope note:** this checks *algorithmic fidelity only*. It does NOT bear on whether
"register" is the right *label* for what G captures — that is the separate question
P1/P2 speak to.

**Promoted (2026-08-28):** the parity check is now a permanent, reproducible part of
the repo at `1_code/_inlp_parity/` (self-contained: Ravfogel's `debias.py` +
`classifier.py` are vendored verbatim under `vendor/`, pinned to commit
`e1edcc19d808108ab71cbb3afb0389db0206a7eb`, MIT). Run with
`python 1_code/_inlp_parity/run_parity.py`; it exits non-zero on divergence. The
external scratch clone at `/home/manh/nullspace_projection` is no longer needed.
