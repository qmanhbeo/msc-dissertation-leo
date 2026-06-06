# Dissertation Canon

This repository measures research-policy misalignment in AI-for-SDG discourse along two separate dimensions:
- `coverage gap`: which SDGs each corpus emphasizes
- `semantic gap`: how differently research and policy discuss the same SDG

The submission surface is intentionally narrow:
- one canonical analysis package under `outputs/`
- one canonical PDF at `outputs/dissertation.pdf`
- one entrypoint at `main.py`

## Main Thesis

Topical overlap is not enough to claim alignment. Research and policy can reference the same SDG while still allocating attention differently and framing the goal differently.

## Canonical Commands

Show repo status only:

```bash
python main.py
```

Rebuild the canonical analysis package and PDF from existing `data/`:

```bash
python main.py --warm-replay --overwrite
```

Refresh the active policy corpus snapshot, fully re-embed policy chunks, and re-score policy against the current centroids:

```bash
python main.py --refresh-policy-corpus --overwrite
```

If the model must be loaded from the local Hugging Face cache without any network access:

```bash
python main.py --refresh-policy-corpus --overwrite --local-files-only
```

Rebuild the canonical analysis package and PDF but skip the expensive sample-stability sweep when those artifacts already exist:

```bash
python main.py --warm-replay --overwrite --skip-sample-stability
```

Run only the sample-stability robustness stage from existing canonical analysis outputs:

```bash
python main.py --sample-stability --overwrite
```

Run the appendix-style genre-adjustment robustness suite from existing embedded/scored data:

```bash
python main.py --genre-adjustment --overwrite
```

Run warm replay and then append the full genre-adjustment robustness suite before the PDF build:

```bash
python main.py --warm-replay --genre-adjustment --overwrite
```

Run the genre-adjustment stage but skip the additional genre-confidence checks:

```bash
python main.py --genre-adjustment --overwrite --skip-genre-confidence-checks
```

Run only the new SDG-aware genre robustness methods inside the stage from the script CLI:

```bash
python code/3_main_analysis/robustness/1_genre_adjustment.py --method sdg_balanced
python code/3_main_analysis/robustness/1_genre_adjustment.py --method within_sdg
python code/3_main_analysis/robustness/1_genre_adjustment.py --method both
```

Build only the canonical PDF from existing canonical tables and figures:

```bash
python main.py --build-pdf --overwrite
```

Run the full active pipeline facade from fetch through PDF:

```bash
python main.py --full-pipeline --overwrite --device cuda --batch-size 256 --local-files-only
```

`--full-pipeline` requires network access, OpenAlex credentials, and the upstream fetch prerequisites expected by the active fetch scripts. It is materially heavier than warm replay.

Important behavior:
- `main.py` with no flags is read-only and prints status
- mutation requires an explicit action flag
- if canonical artifacts already exist, reruns fail closed unless `--overwrite` is supplied
- `--output-dir` can redirect the canonical output root, but the default contract is `outputs/`
- `--warm-replay` and `--full-pipeline` run sample stability by default; use `--skip-sample-stability` only when the existing sample-stability artifacts are already present and should be reused
- `--refresh-policy-corpus` is the canonical response to any policy-corpus change; it rebuilds `policy_chunks_all`, fully re-embeds `policy`, and refreshes policy scoring without touching `osdg`, `benchmark`, or the research shard embeddings
- `--genre-adjustment` is additive only: it does not replace the canonical raw semantic-gap outputs, and it exists as an appendix-style robustness suite
- `--skip-genre-confidence-checks` is an escape hatch for the expensive follow-on genre-confidence checks; it only matters when `--genre-adjustment` is present
- the SDG-aware genre controls run by default inside `--genre-adjustment`; use `--sdg-genre-method`, `--sdg-genre-samples-per-cell`, `--sdg-genre-min-samples-per-class`, `--sdg-genre-test-size`, `--sdg-genre-classifier-type`, and `--sdg-genre-random-seed` from `main.py` to override their defaults
- smaller adjusted gaps are not automatically better; the raw gap remains the dissertation's main estimand

## Reproducibility Contract

Warm replay is the primary reproducibility target.

Assumed existing inputs:
- hydrated `data/0_raw/` through `data/3_scored/`
- policy embeddings in `data/2_embedded/`
- research shard embeddings in `data/2_embedded/research_shards/`
- paper score shards in `data/3_scored/paper_scores_shards/`

Policy-specific reproducibility rule:
- `data/1_preprocessed/policy_all/policy_chunks_all.jsonl` is the authoritative active policy corpus snapshot
- when any policy source changes, the canonical workflow is a full policy refresh via `python main.py --refresh-policy-corpus --overwrite`
- append-only policy embedding is intentionally unsupported; `data/2_embedded/policy.npy` should always be a full embedding of the active `policy_chunks_all` snapshot

`data/` and `outputs/` are intentionally not tracked in Git. A stranger can verify pipeline logic from source, but must hydrate `data/` before replaying the canon.

## Canonical Outputs

Root artifacts:
- `outputs/validation_results.json`
- `outputs/confusion_matrix.csv`
- `outputs/centroid_similarity_matrix.csv`
- `outputs/sdg_attention_distribution_document_weighted.json`
- `outputs/diagnostic_sdg_attention_distribution_unweighted_chunks.json`
- `outputs/sdg_conceptual_alignment_cosine_distances.json`
- `outputs/robustness_check_semantic_distances_by_chunk_cap.json`
- `outputs/statistical_tests_hypothesis_25_hypothesis_26_and_bias_calibration.json`
- `outputs/visualization_source_sdg_attention_vs_semantic_distance.csv`
- `outputs/sample_stability_summary.json`
- `outputs/sample_stability_draws.jsonl`
- `outputs/sample_stability_per_sdg.json`
- `outputs/sample_stability_table.csv`
- `outputs/dissertation.pdf`

Table artifacts:
- `outputs/tables/num_validation.tex`
- `outputs/tables/tab_validation.tex`
- `outputs/tables/num_coverage.tex`
- `outputs/tables/tab_coverage.tex`
- `outputs/tables/num_semantic.tex`
- `outputs/tables/tab_semgap.tex`
- `outputs/tables/num_h25.tex`
- `outputs/tables/tab_h25.tex`
- `outputs/tables/num_sample_stability.tex`
- `outputs/tables/tab_sample_stability.tex`

Figure artifacts:
- `outputs/figures/fig1_coverage_profiles.pdf`
- `outputs/figures/fig1_coverage_profiles.png`
- `outputs/figures/fig2_semantic_gap.pdf`
- `outputs/figures/fig2_semantic_gap.png`
- `outputs/figures/fig3_coverage_semantic_scatter.pdf`
- `outputs/figures/fig3_coverage_semantic_scatter.png`

Genre-adjustment robustness artifacts when `--genre-adjustment` is run:
- all genre outputs live under `outputs/robustness/genre_adjustment/`
- `outputs/robustness/genre_adjustment/data/genre_adjusted_semantic_gaps.json`
- `outputs/robustness/genre_adjustment/data/genre_adjusted_semantic_gaps.csv`
- `outputs/robustness/genre_adjustment/data/genre_adjustment_classifier_metrics.json`
- `outputs/robustness/genre_adjustment/data/genre_adjustment_classifier_validation_grid.csv`
- `outputs/robustness/genre_adjustment/data/genre_extreme_examples.csv`
- `outputs/robustness/genre_adjustment/data/genre_tfidf_terms.csv`
- `outputs/robustness/genre_adjustment/data/summary.json`
- `outputs/robustness/genre_adjustment/data/sdg_balanced_gap_comparison.csv`
- `outputs/robustness/genre_adjustment/data/within_sdg_gap_comparison.csv`
- `outputs/robustness/genre_adjustment/data/regression_gap_comparison.csv`
- `outputs/robustness/genre_adjustment/data/genre_confidence_checks/`
- `outputs/robustness/genre_adjustment/tables/`
- `outputs/robustness/genre_adjustment/figures/`
- `outputs/robustness/genre_adjustment/README_genre_adjustment.md`

Interpretation status of the genre suite:
- raw semantic gap remains the primary result because it directly measures the observed within-SDG research-policy difference
- one-direction global subtraction and SDG-balanced global subtraction are sensitivity checks for broad corpus-level register effects
- within-SDG classifier subtraction is an over-subtraction stress test
- within-SDG regression subtraction is an aggressive over-subtraction stress test
- smaller adjusted gaps do not imply higher validity

## Active Pipeline

Policy / benchmark side:
1. fetch policy sources
2. convert manually downloaded policy PDFs into text, when present
3. run the policy-source preprocess stage under `code/1_preprocess/policy/`
4. build `policy_scrape`, `sdgi_corpus`, and `ungdc_sdg` source corpora, with `policy_scrape` and `policy_manual` handled together inside `0_preprocess_policy.py`
5. merge policy sources into `data/1_preprocessed/policy_all/policy_chunks_all.jsonl`
6. embed `policy`, `osdg`, and `benchmark`
7. build SDG centroids
8. validate centroids against the benchmark

Research side:
1. fetch OpenAlex works
2. preprocess into resume-safe shards
3. embed paper shards
4. score paper shards against SDG centroids
5. rebuild research centroids from the full scored shard set

Downstream analysis:
1. score the active policy corpus against SDG and research centroids
2. compute coverage gap
3. compute semantic gap with chunk-cap sensitivity checks
4. compute H25/H26 interaction outputs
5. run the sample-stability robustness sweep
6. optionally run the genre-adjustment robustness suite
7. generate figures
8. build the dissertation PDF from canonical tables and figures

## Robustness and Validation

The active canon includes these safeguards:
- centroid validation on the expert-labelled benchmark (`macro-F1` reported before downstream use)
- document-weighted policy coverage, so long policy reports do not dominate by chunk count alone
- semantic-gap chunk-cap sensitivity at 20, 50, and 100 chunks per document
- explicit SDG reliability flags when clusters are too small
- A15 calibration check comparing policy-vs-OSDG and paper-vs-OSDG top scores
- sample-size stability sweep across repeated random research subsamples, with 100 deterministic draw seeds (`42`-`141`) reused across tiers and cached sampled aggregates under `data/3_scored/paper_sample_seed_42_141/`
- genre-adjustment robustness suite with deterministic train/validation/test logistic-regression selection before projection subtraction
- additional genre-confidence checks covering adjusted re-separability, held-out-SDG generalization, multi-direction subtraction sensitivity, and topic-matched nearest-neighbor comparison
- SDG-balanced genre classifier control, which equalises SDG composition before learning one global research-policy direction
- within-SDG genre classifiers, which allow the genre direction to vary by SDG and are interpreted as over-subtraction stress tests rather than replacement estimates
- genre-vector interpretability outputs that expose the most aligned texts and measure cosine overlap between the learned genre direction and the canonical SDG centroids without fitting any additional classifier
- Rodriguez-style embedding regression robustness, which treats embedding coordinates as outcomes and compares regression-derived genre directions against the classifier-derived direction after controlling for SDG and genre x SDG structure; the within-SDG regression variant is intentionally treated as an aggressive stress test
- SDG 4 caveat carried into the manuscript where learning vocabulary may inflate education assignments
- hard protection against partial shard runs overwriting canonical research centroids unless explicitly allowed

## Repo Layout

Active source:
- `main.py`
- `code/0_fetch/`
- `code/1_preprocess/`
- `code/2_embed/`
  Reference subchain: `reference/0_embed_reference_corpora.py`, `reference/1_build_sdg_centroids.py`, `reference/2_validate_centroids.py`
  Research subchain: `research/0_embed_paper_shards.py`, `research/1_score_paper_shards.py`
  Policy subchain: `policy/0_score_policy_corpus.py`
- `code/3_main_analysis/`
  Canonical chain: `canonical/0_coverage_gap.py`, `canonical/1_semantic_gap.py`, `canonical/2_coverage_semantic_interaction.py`
  Robustness chain: `robustness/0_sample_stability.py`, `robustness/1_genre_adjustment.py`
  Shared helpers: `shared/`
- `code/4_visualization/`
- `code/shared_utils.py`
- `writing/dissertation.tex`
- `writing/references.bib`
- `writing/build_pdf.sh`

Working notes kept for the active thesis:
- `notes/ASSUMPTIONS.md`
- `notes/HYPOTHESES.md`
- `notes/LIT_REVIEW_INSIGHTS.md`

## Environment

Recommended setup:

```bash
conda env create -f conda-env-dissertation.yml
conda activate dissertation
pip install -r requirements.txt
```

Lock snapshots retained for auditability:
- `requirements.lock.txt`
- `conda-explicit-dissertation.txt`

## What Was Removed

This cleanup intentionally removed the old multi-run output layout, versioned dissertation builds, legacy manuscript dependencies, and stale prototype notes that contradicted the active pipeline.



# Walkthrough: This Paper Explained Like You're New Here

## What This Paper Is About

This paper asks a simple but important question:

> If both AI researchers and policymakers say they care about the Sustainable Development Goals (SDGs), are they actually talking about the same things?

The paper says there are **two different ways** to be "aligned":

1. **Attention alignment**: Are they focusing on the same goals?
2. **Meaning alignment**: When they talk about the same goal, do they mean the same kind of thing?

That matters because two groups can use the same label, like "climate" or "health", while still talking past each other.

## The Big Idea In Very Simple Words

Think of the 17 SDGs as **17 places on a stage**.

- The **attention question** is: which parts of the stage get the most spotlight?
- The **meaning question** is: when two people stand in the same place on the stage, are they saying similar things?

This paper argues that those are **not the same question**.

So the paper does **not** just ask:

> "Does AI research mention the SDGs?"

It asks two deeper questions:

1. **Which SDGs get more room and attention in research, and which get more room and attention in policy?**
2. **Inside each SDG, are research and policy actually speaking in the same semantic language?**

## The Whole Pipeline At A Glance

Here is the full pipeline before we slow it down:

1. Fetch research papers and policy texts.
2. Clean them and prepare them into comparable text units.
3. Turn each text into a vector using an embedding model.
4. Build 17 SDG "reference points" from labeled examples.
5. Compare every paper and every policy chunk to those 17 SDG reference points.
6. Measure which SDGs each corpus emphasizes.
7. Measure how similar research and policy are *inside* each SDG.
8. Run extra checks to make sure the results are not a fluke or a measurement artefact.

Now the same story, step by step.

---

## Step 1: Fetching The Data

### 1A. Research data: AI papers

The research corpus comes from **OpenAlex**.

OpenAlex is a very large open database of scholarly works. It stores paper metadata such as titles, abstracts, publication years, and identifiers.

### What was fetched?

The paper collected AI-related research linked to each SDG by using:

- **17 SDGs**
- **4 AI search terms**
  - `artificial intelligence`
  - `machine learning`
  - `deep learning`
  - `neural network`
- **Years 2018-2025**

That creates **68 structured queries** in total:

- `17 SDGs x 4 AI terms = 68 queries`

### How was it fetched?

The pipeline used OpenAlex queries that combine:

- an SDG filter
- an AI term
- the target year range

Then it merged the results and removed duplicates.

### Why this source?

Because OpenAlex is:

- large
- structured
- widely used in research mapping
- practical for building a very big AI-for-SDG corpus

### Why this matters

If you want to know what **research** is paying attention to, you need a very large and reasonably systematic paper source. OpenAlex gives that.

### Final research size

After cleaning and deduplication, the canonical corpus contains:

- **2,543,698 research papers**

That is the main research dataset used in the final analysis.

---

### 1B. Policy data: what institutions say and prioritize

The policy corpus is built from **three policy-oriented sources**.

#### Source 1: Curated AI and SDG policy documents

These are hand-selected high-salience documents such as:

- the UN 2030 Agenda
- the Paris Agreement
- SDG progress reports
- IPCC summaries
- OECD AI Principles
- UNESCO AI ethics documents
- the EU AI Act
- national or regional AI strategy documents

These texts represent how major institutions talk about AI, sustainability, development, governance, and implementation.

#### Source 2: SDGi VNR/VLR corpus

This is a collection of:

- **Voluntary National Reviews (VNRs)**
- **Voluntary Local Reviews (VLRs)**

These are reports that governments and cities submit to the UN to describe SDG progress and priorities.

They matter because they show how public institutions frame real-world SDG work.

#### Source 3: UN General Debate Corpus (UNGDC)

This is a large collection of speeches delivered at the United Nations General Assembly.

The pipeline filtered it down to speeches or passages relevant to the SDGs.

### Why use all three?

Because "policy" is not one thing.

If the corpus only used formal global strategy documents, it would overrepresent one kind of policy voice.

Using all three helps the paper capture:

- high-level global policy framing
- national and local implementation framing
- diplomatic and political framing

### Final policy size

After preprocessing and chunking, the canonical corpus contains:

- **47,005 policy text chunks**
- coming from **2,392 source documents**

---

### 1C. Support datasets: not the main result, but needed to build the measuring tool

The paper also uses two labeled SDG datasets:

1. **OSDG Community Dataset**
2. **SDG Classification Benchmark**

These are **not** the main research corpus and **not** the main policy corpus.

They are used to build and test the SDG measuring instrument.

Think of them as the paper's **map key** or **ruler**.

Without them, the paper would have lots of text, but no trustworthy way to say:

> "This looks most like SDG 3" or "this is semantically closer to SDG 13 than SDG 9."

---

## Step 2: Cleaning And Preparing The Text

Raw text is messy. Before comparing research and policy, the paper has to make the text usable and reasonably fair.

### 2A. Research preprocessing

For research papers, the pipeline:

1. **deduplicates** papers by OpenAlex work ID
2. **drops papers with no usable abstract**
3. **combines the title and abstract into one text unit**

### Why combine title and abstract?

Because a title alone is too short, and the abstract usually contains the actual topic and framing.

Putting them together gives a better summary of what the paper is about.

### Why drop records with no abstract?

Because this project is about comparing meanings in text.

If there is no real text to compare, the paper cannot be placed reliably in semantic space.

---

### 2B. Policy preprocessing

Policy documents are very different from abstracts.

They are usually:

- longer
- broader
- more mixed in topic

So the pipeline does **not** treat each whole policy document as one unit.

Instead, it:

1. cleans and normalizes the text
2. splits long documents into chunks of about **150-300 words**
3. tries to keep sentence boundaries intact
4. removes very short fragments, especially under **20 words**
5. deduplicates repeated text after normalization

### Why chunk policy documents?

Because a whole report can talk about many SDGs at once.

If you score a 200-page report as one giant block, you blur everything together.

Chunking lets the paper say:

> "This part of the report is mostly about climate."
>
> "This part is mostly about governance."
>
> "This part is mostly about partnerships."

That makes the comparison much sharper.

### Important fairness issue

Chunking creates a problem too:

- a long policy report can produce many chunks
- a research paper stays one paper

So later in the analysis, the paper uses **document-weighted coverage** so that one giant report does not dominate just because it was cut into many pieces.

That is an important robustness choice.

---

## Step 3: Turning Text Into Meaning Vectors

This is the "embedding" step.

### What is an embedding?

An embedding is a way to turn text into numbers so that:

- texts with similar meaning end up close together
- texts with different meaning end up farther apart

You can think of it like turning every paragraph into a point in a giant meaning-map.

### Which embedding model was used?

The paper uses:

- `all-MiniLM-L6-v2`

This is a **Sentence-BERT** style embedding model that produces:

- **384-dimensional vectors**

### Why this model?

Because it is:

- good at capturing sentence-level meaning
- efficient enough for a very large corpus
- widely used
- simple to use without extra fine-tuning

That last point matters.

The goal of the paper is not to train a complicated black-box classifier. The goal is to build a transparent measuring instrument.

### What happens here in practice?

Every text becomes a vector:

- every research paper
- every policy chunk
- every labeled SDG example in OSDG
- every benchmark example

### Why normalize the vectors?

The vectors are L2-normalized.

In plain English, that means:

> make all arrows the same length, so comparison focuses on direction of meaning, not raw size

That makes cosine similarity easier and cleaner to interpret.

---

## Step 4: Building The SDG Reference Map

Now the paper needs a way to represent each of the **17 SDGs** in the same embedding space.

This is where the SDG centroids come in.

### What is a centroid?

A centroid is just an average vector.

If you take many texts that belong to SDG 3, embed them, and average them, you get a kind of **center of gravity** for SDG 3.

The paper builds one such reference point for each SDG.

### Where do these SDG examples come from?

#### SDGs 1-16

These come from the **OSDG Community Dataset**.

This dataset contains over 40,000 texts with human-validated SDG labels.

The paper filters them to keep more reliable examples, including:

- minimum text length
- annotator agreement threshold of at least **0.5**

Then it averages the embeddings inside each SDG.

#### SDG 17

OSDG does **not** include SDG 17.

So SDG 17 is built separately from the **SDG Classification Benchmark**, which contains expert-labeled examples.

### Why do this at all?

Because the paper needs 17 anchor points saying:

- "this is what SDG 1 tends to look like in semantic space"
- "this is what SDG 2 tends to look like"
- and so on

Once those anchors exist, the paper can compare any new text to them.

### Why this is transparent

This is a **nearest-centroid** method.

That means the model is not making mysterious hidden decisions. It is doing something conceptually simple:

1. build an average meaning-point for each SDG
2. compare a new text to all 17
3. see which one is closest

That is easier to inspect than a heavy supervised classifier.

---

## Step 5: Checking Whether The SDG Map Actually Works

Before using the SDG centroids for the real analysis, the paper tests them on an independent benchmark.

### How?

The centroids are validated against the **SDG Classification Benchmark**.

This asks:

> If a benchmark text is truly SDG 6, does the centroid system usually put it near SDG 6?

### Why is this necessary?

Because without validation, the whole downstream analysis could look precise while resting on a weak measuring tool.

### Result

The canonical centroid validation reports:

- **macro-F1 = 0.733**
- **accuracy = 0.738**

That is not perfect, but it is much better than random guessing and good enough to justify using the instrument for corpus-level analysis.

### Important SDG 17 caveat

SDG 17 is methodologically special because it had to be built from benchmark texts rather than OSDG.

So the paper treats SDG 17 carefully in interpretation.

---

## Step 6: Scoring The Research Papers

Now the paper takes each of the **2,543,698 research papers** and compares it to the 17 SDG centroids.

### What does each paper get?

Each paper gets a score against all 17 SDGs.

That creates a profile like:

- maybe strongly close to SDG 3
- somewhat close to SDG 9
- weakly close to SDG 10
- and far from the others

### What is the simplest use of that score?

The simplest use is the **top match**:

> Which SDG is this paper closest to?

That gives each paper a strongest-alignment SDG.

### Why do this?

Because now the paper can count, across millions of papers:

- how much research ends up near each SDG
- how concentrated or dispersed the research profile is

### Important detail

The research corpus is processed in **resume-safe shards**.

That matters because the dataset is huge.

Instead of trying to process millions of papers as one giant fragile job, the pipeline works in smaller pieces that can be resumed safely if interrupted.

---

## Step 7: Scoring The Policy Text

The paper does the same SDG-centroid comparison for each of the **47,005 policy chunks**.

### What does each policy chunk get?

Just like research papers, each chunk gets a score against all 17 SDGs.

So each chunk can be placed on the same SDG meaning-map.

### Why chunk-level scoring matters here

A long policy report might include:

- climate promises
- governance language
- international cooperation
- education references
- poverty framing

all in the same document.

Chunk-level scoring avoids flattening all of that into one average blob.

### Important fairness correction

Because long documents create many chunks, the paper later uses **document-weighted policy coverage** for the main attention comparison.

Plain English version:

> A 300-page report should not get 300 times more voice than a shorter policy document just because it was split more times.

---

## Step 8: Comparing Research And Policy

Now we get to the heart of the paper.

The paper compares research and policy in **two different ways**.

### 8A. First comparison: who gives attention to which SDG?

This is the paper's **coverage gap** idea.

Plain English:

> Which SDGs get more spotlight in research, and which get more spotlight in policy?

This is the big-picture, stage-level view.

You can also call this the **macro** view if you want the technical word, but plain English is better:

> it compares the overall distribution of attention across the 17 goals

### How is it measured?

The paper uses the strongest SDG assignment for each item:

- each research paper
- each policy document, with document weighting so long documents do not dominate

Then it asks what share of the corpus falls near each SDG.

### Main finding

The research and policy profiles are very different.

#### Policy is dominated by:

- **SDG 13 (Climate Action): 36.1%**
- **SDG 17 (Partnerships): 34.9%**
- **SDG 16 (Institutions/Governance): 20.3%**

#### Research is dominated by:

- **SDG 3 (Health): 17.9%**
- **SDG 4 (Education/Learning): 13.8%**
- **SDG 9 (Industry/Innovation): 10.2%**

That means the two corpora are not just using SDG language in the same proportions.

They are putting their spotlight in different places.

---

### 8B. Second comparison: inside one SDG, are they talking similarly?

This is the paper's **semantic gap** idea.

Plain English:

> Even when both sides talk about the same goal, are they talking about the same kinds of things?

This is the inside-the-goal view.

You can call it the **micro** view if you want the technical word, but again plain English is cleaner:

> it measures meaning mismatch within each SDG

### How is it measured?

Take one SDG at a time.

For example, SDG 13:

1. gather the research papers whose strongest alignment is SDG 13
2. average their vectors into a **research sub-centroid**
3. gather the policy chunks whose strongest alignment is SDG 13
4. average their vectors into a **policy sub-centroid**
5. compare those two averages

If they are close, research and policy are speaking more similarly inside that SDG.

If they are far apart, the two sides are using the same SDG label but framing it differently.

### Main finding

The largest semantic gaps appear in:

- **SDG 17: 0.669**
- **SDG 13: 0.544**
- **SDG 10: 0.518**

So some of the most policy-heavy goals are also the goals where the two sides are furthest apart in meaning.

That is the core intellectual contribution of the paper:

> topical overlap is not enough to claim alignment

---

## Step 9: The Main Tests, In Plain English

The dissertation uses labels like **H25**, **H26**, and **A15**.

Those labels are just shorthand. Here is what they mean in ordinary language.

### 9A. H25: Do bigger attention differences go together with bigger meaning differences?

Plain English question:

> If research and policy give very different amounts of attention to a goal, does that usually mean they also talk about that goal very differently?

### Why ask this?

Because it would be tempting to assume:

> "If the spotlight is in different places, the meaning mismatch must also be bigger."

But that is a hypothesis, not a fact.

### Result

The paper's primary H25 test is **not supported**.

The reported result is:

- **r = -0.074**
- **p = 0.779**

Plain English:

> simply knowing how much research attention a goal gets does not tell you much about whether research and policy are semantically close inside that goal

However, the paper also finds a more nuanced pattern:

- large overall imbalance between the two corpora does tend to go with larger semantic gaps
- but research attention by itself is not a good predictor

So the final message is:

> attention and meaning are related in some ways, but they are not the same thing and do not move together neatly

---

### 9B. H26: Is the disconnect one-sided?

Plain English question:

> Does research ignore policy framing more than policy ignores research framing?

This is the **directional asymmetry** question.

The paper does not want to assume mismatch is perfectly symmetrical. It checks whether one side seems more "out of tune" with the other.

### Result

The raw directional result points in the direction that:

> research may be further from policy framing than policy is from research framing

But the paper does **not** treat that as a firm conclusion.

Why not?

Because of the next check.

---

### 9C. A15: Is the measuring tool itself slightly biased?

Plain English question:

> Before claiming that one side is more misaligned, are we sure the ruler itself is neutral?

This is a calibration or fairness check.

The SDG centroids are built from OSDG-style material, which is somewhat policy-adjacent in language.

So it is possible that policy-like language naturally scores higher even if no true asymmetry exists.

### What did the paper find?

The A15 bias check found a baseline difference of:

- **0.326**

The observed H26 asymmetry gap was only:

- **0.144**

That means the measured asymmetry is **smaller than the tool's own calibration bias**.

Plain English:

> the ruler already leans enough that we should not make a strong directional claim

So H26 is reported as:

- **suggestive**
- but **inconclusive**

That is a good example of the paper being careful rather than overclaiming.

---

## Step 10: Robustness Checks - How The Paper Tries To Make The Result Trustworthy

The paper does not just produce a headline number and stop.

It runs several checks to ask:

> "Can we trust this result?"

### 10A. Centroid validation

Before using the SDG centroids, the paper checks whether they can classify benchmark SDG texts reasonably well.

That protects against building the whole thesis on a weak SDG map.

### 10B. Document-weighted policy coverage

This prevents giant policy reports from dominating just because they create many chunks.

Without this correction, coverage results could be distorted by document length rather than true attention.

### 10C. Chunk-cap sensitivity

For semantic gap, the paper caps how many chunks one document can contribute to an SDG cluster.

It tests multiple caps:

- 20
- 50
- 100

Why?

Because otherwise one long document with many similar chunks could overpower the semantic center for a goal.

The rankings remain stable across these checks, which is a good sign.

### 10D. Reliability flags for sparse goals

Some SDGs have fewer assigned texts than others.

The paper flags small or less reliable clusters rather than pretending every SDG estimate is equally strong.

### 10E. A15 calibration check

This is the fairness test described above.

It asks whether the measuring instrument itself favors one kind of language.

### 10F. Sample stability check

This is one of the strongest robustness steps.

The paper asks:

> "If we randomly use fewer research papers, do the results jump around a lot, or do they stay basically the same?"

### How was that done?

The pipeline repeatedly sampled the already embedded and already scored research corpus at these sizes:

- 1k
- 2k
- 5k
- 10k
- 20k
- 50k
- 100k
- 200k
- 500k
- 1m
- 2m
- plus the full corpus as the anchor row

For each sampled tier, it ran:

- **100 random draws**

using the same deterministic seed list:

- **42-141**

reused across all tiers.

### Important detail

The pipeline did **not** re-embed the text.

It reused the precomputed vectors and scored data.

That makes the robustness test fast and reproducible.

### What did the stability ladder show?

It showed that the results become very stable as the sample grows.

The paper's interpretation is:

- by around **200k papers**, the main metrics are already very close to the full-corpus values
- by **500k+**, the results are practically flat at the headline level

### Why still use the full corpus then?

Because the expensive part, embedding, had already been done.

So once all vectors already exist, there is no methodological reason to throw away evidence.

That is why the paper keeps:

- **the full 2,543,698-paper corpus as the canonical result**

and uses the subsampling ladder as a robustness check, not as a replacement.

---

## Step 11: What The Paper Found

Here is the short version.

### Finding 1: Research and policy focus on different SDGs

Policy discourse is heavily concentrated on:

- climate
- partnerships
- institutions and governance

Research discourse is more concentrated on:

- health
- education/learning-related material
- innovation

So the two sides are not placing their spotlight on the same goals.

### Finding 2: Even when they discuss the same SDG, they often mean different things

The biggest within-goal meaning mismatches happen in:

- SDG 17
- SDG 13
- SDG 10

That means shared SDG labels do **not** guarantee shared framing.

### Finding 3: Attention mismatch and meaning mismatch are not the same thing

Just because research talks a lot about a goal does **not** mean it is semantically close to policy on that goal.

That is why the paper insists on measuring both:

- overall attention
- within-goal meaning

### Finding 4: The asymmetry claim is cautious

There is some directional signal suggesting research may be further from policy than policy is from research.

But the paper does not overclaim this because the fairness check shows the measuring tool itself has a non-trivial bias.

That restraint is important.

---

## Step 12: What This Paper Does **Not** Claim

This paper is careful about its scope.

It does **not** claim:

### 1. It does not claim to measure real-world impact

It measures **discourse alignment** in text space.

That means:

- what research papers say
- what policy documents say

It does **not** directly measure whether real-world policy changed, or whether research actually solved SDG problems.

### 2. It does not claim embeddings are perfect

Embeddings are powerful, but they are still a modeling tool.

They help measure semantic proximity, not absolute truth.

### 3. It does not claim every SDG estimate is equally strong

Some goals are denser and cleaner than others.

Some, like SDG 17, are methodologically special.

Some, like SDG 4, carry known interpretation caveats.

### 4. It does not claim the policy corpus is every policy voice on Earth

The policy corpus is broad, but it still reflects a particular slice of institutional discourse:

- mostly formal
- mostly English-language
- mostly international or official

### 5. It does not claim topical overlap equals substantive agreement

In fact, that is exactly what the paper argues against.

Two texts can both say "SDG 13" and still mean very different things by it.

---

## Why The Paper's Logic Makes Sense

If you had only measured keyword counts, you could say:

> "Both sides mention climate, so they must be aligned."

But that is too shallow.

This paper improves on that by asking:

1. **How much attention does each goal get?**
2. **Inside that goal, what kind of meaning is actually there?**

That is why the paper's central thesis is stronger than a normal bibliometric map.

It does not stop at "who mentioned what."

It asks:

> "When they use the same goal label, are they actually meeting each other there?"

---

## The Whole Pipeline In One Line Each

- **Fetch**: collect AI research papers from OpenAlex and policy texts from several institutional sources.
- **Clean**: remove duplicates, remove unusable text, and standardize the corpora.
- **Chunk**: split long policy documents into smaller meaningful pieces.
- **Embed**: turn every text into a 384-dimensional meaning vector using `all-MiniLM-L6-v2`.
- **Build SDG anchors**: create 17 SDG reference centroids from labeled datasets.
- **Validate the anchors**: test whether the SDG centroids behave sensibly on benchmark data.
- **Score research**: compare each research paper to all 17 SDG centroids.
- **Score policy**: compare each policy chunk to the same 17 centroids.
- **Compare attention**: measure which SDGs dominate research and which dominate policy.
- **Compare meaning**: measure whether research and policy talk similarly inside each SDG.
- **Stress-test the result**: run calibration checks, chunk-cap checks, reliability checks, and repeated subsampling.
- **Conclude**: research-policy alignment cannot be inferred from topical overlap alone.

---

## If You Want To Match This To The Repository

If you want to connect the explanation above to the codebase, the project is laid out in the same order as the pipeline:

- `main.py`: the top-level entrypoint
- `code/0_fetch/`: data collection scripts
- `code/1_preprocess/`: cleaning, filtering, merging, and chunking
- `code/2_embed/`: embeddings, SDG centroids, scoring, and validation
- `code/3_main_analysis/`: canonical main-text analysis in `canonical/`, appendix robustness in `robustness/`, and shared helper logic in `shared/`
- `code/4_visualization/`: final plots
- `writing/dissertation.tex`: the paper itself
- `outputs/`: the canonical results, tables, figures, and final PDF

So if you read the repository from `code/0_fetch/` to `code/3_main_analysis/`, you are basically walking through the same story this document just explained.

---

## If You Only Remember Five Things

1. The paper compares **AI research** with **SDG policy discourse**.
2. It measures both **what gets attention** and **what that attention means**.
3. It uses embeddings and SDG centroids to compare millions of texts in one semantic space.
4. It finds that research and policy often spotlight different goals and also frame the same goals differently.
5. Its main message is: **sharing an SDG label does not automatically mean genuine alignment.**
