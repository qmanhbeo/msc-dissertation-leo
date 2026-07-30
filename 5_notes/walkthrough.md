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
5. Compare every paper and every policy segment to those 17 SDG reference points.
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

After preprocessing and segmenting, the canonical corpus contains:

- **47,005 policy text segments**
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

### What does the cleaned research corpus look like?

One important detail: the research corpus is **not segmented** the way the policy corpus is.

Each research unit stays as **one paper-level text**, built from:

- the paper title
- plus the paper abstract

In the canonical corpus of **2,543,698 papers**:

- the average title length is **12.6 words**
- the average abstract length is **195.4 words**
- the average combined title+abstract length is **208.0 words**
- the median combined title+abstract length is **197 words**
- the middle of the corpus is fairly compact: the 10th to 90th percentile range is **97 to 303 words**

That means the research side is made of short, focused paper summaries rather than long mixed-topic documents. That is exactly why the policy side has to be segmented later, while the research side does not.

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
2. splits long documents into segments of about **150-300 words**
3. tries to keep sentence boundaries intact
4. removes very short fragments, especially under **20 words**
5. deduplicates repeated text after normalization

### Why segment policy documents?

Because a whole report can talk about many SDGs at once.

If you score a 200-page report as one giant block, you blur everything together.

Segmenting lets the paper say:

> "This part of the report is mostly about climate."
>
> "This part is mostly about governance."
>
> "This part is mostly about partnerships."

That makes the comparison much sharper.

### Important fairness issue

Segmenting creates a problem too:

- a long policy report can produce many segments
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

- `all-mpnet-base-v2`

This is a **Sentence-BERT** style embedding model that produces:

- **768-dimensional vectors**

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
- every policy segment
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

The paper does the same SDG-centroid comparison for each of the **47,005 policy segments**.

### What does each policy segment get?

Just like research papers, each segment gets a score against all 17 SDGs.

So each segment can be placed on the same SDG meaning-map.

### Why segment-level scoring matters here

A long policy report might include:

- climate promises
- governance language
- international cooperation
- education references
- poverty framing

all in the same document.

Segment-level scoring avoids flattening all of that into one average blob.

### Important fairness correction

Because long documents create many segments, the paper later uses **document-weighted policy coverage** for the main attention comparison.

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
3. gather the policy segments whose strongest alignment is SDG 13
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

This prevents giant policy reports from dominating just because they create many segments.

Without this correction, coverage results could be distorted by document length rather than true attention.

### 10C. Segment-cap sensitivity

For semantic gap, the paper caps how many segments one document can contribute to an SDG cluster.

It tests multiple caps:

- 20
- 50
- 100

Why?

Because otherwise one long document with many similar segments could overpower the semantic center for a goal.

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
- **Segment**: split long policy documents into smaller meaningful pieces.
- **Embed**: turn every text into a 768-dimensional meaning vector using `all-mpnet-base-v2`.
- **Build SDG anchors**: create 17 SDG reference centroids from labeled datasets.
- **Validate the anchors**: test whether the SDG centroids behave sensibly on benchmark data.
- **Score research**: compare each research paper to all 17 SDG centroids.
- **Score policy**: compare each policy segment to the same 17 centroids.
- **Compare attention**: measure which SDGs dominate research and which dominate policy.
- **Compare meaning**: measure whether research and policy talk similarly inside each SDG.
- **Stress-test the result**: run calibration checks, segment-cap checks, reliability checks, and repeated subsampling.
- **Conclude**: research-policy alignment cannot be inferred from topical overlap alone.

---

## If You Want To Match This To The Repository

If you want to connect the explanation above to the codebase, the project is laid out in the same order as the pipeline:

- `main.py`: the top-level entrypoint
- `1_code/0_fetch/`: data collection scripts
- `1_code/1_preprocess/`: cleaning, filtering, merging, and segmenting
- `1_code/2_segment/`: corpus segmentation
- `1_code/3_embed/`: embeddings, SDG centroids, scoring, and validation
- `1_code/4_supervised_model_train/`: classifier training
- `1_code/5_supervised_model_infer/`: classifier inference
- `1_code/6_calculate_centroids/`: compute research centroids
- `1_code/7_main_analysis/`: coverage gap, semantic gap, robustness checks, and interaction tests
- `1_code/8_visualization/`: final plots
- `3_writing/dissertation.tex`: the paper itself
- `4_outputs/`: artifact root (`{model}/`, `appendix/{model}/`)

So if you read the repository from `1_code/0_fetch/` to `1_code/7_main_analysis/`, you are basically walking through the same story this document just explained.

---

## If You Only Remember Five Things

1. The paper compares **AI research** with **SDG policy discourse**.
2. It measures both **what gets attention** and **what that attention means**.
3. It uses embeddings and SDG centroids to compare millions of texts in one semantic space.
4. It finds that research and policy often spotlight different goals and also frame the same goals differently.
5. Its main message is: **sharing an SDG label does not automatically mean genuine alignment.**
