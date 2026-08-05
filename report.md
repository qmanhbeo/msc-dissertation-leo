# Independent Editorial Architecture Audit — Dissertation Structure

**Title examined:** *Measuring the Gap: Semantic Alignment Between AI-for-Sustainability Research and SDG Policy Frameworks*
**Role:** independent senior-researcher / journal-editor reviewer. Not a coauthor.
**Materials read in full:** `3_writing/dissertation.tex` (978 lines, all chapters, subsections, appendices), abstract, table/figure inventory, references to generated outputs.
**Date:** 2026-08-05
**Repo:** `/home/manh/dissertation`

Scope: numerical correctness and scientific validity are only considered where they affect structure. This document is not a proof-edit; it is a verdict on whether the dissertation reads as one designed argument or as accumulated development history. It replaces the previous contents of `report.md`.

---

# Phase 1 — First-impression reader test

Simulated reader: an external examiner with only the title, abstract, ToC, and headings.

## 1. What appears to be the central research contribution?

Two contributions of unequal weight are claimed, and the reader cannot tell from the ToC which is the load-bearing one:

1. **A measurement framework** that separates research–policy divergence into two axes (coverage vs. within-SDG semantic framing) and decomposes the semantic gap into topic vs. register via INLP (Iterative Nullspace Projection).
2. **An empirical finding** about the AI–SDG interface: an apparent "cancellation" (raw coverage–semantic correlation ≈ 0 because topic and register components offset) and SDG-specific divergence rankings (SDG 17 most divergent under the adjusted gap, SDG 13 under the raw).

The abstract announces both. But the ToC's chapter titles ("Literature Review", "Methodology", "Results", "Discussion") never say what the paper is *about*. The contribution is a measurement protocol whose headline is a statistical decomposition — a mismatch that only becomes clear on a second read.

## 2. What is the logical sequence of the argument?

From the headings alone:

1. Introduction (RQ, two-dimension claim, *results previewed with p-values*)
2. Literature Review (bibliometrics → three-level framework → theory of divergence → semantic-method cautions → research gap)
3. Methodology (corpora → embedding → segmentation → classifier → coverage gap → semantic gap → **INLP register adjustment** → **coverage–semantic interaction/hypotheses** → pipeline summary)
4. Results (classifier validation + coverage → semantic gap *after register removal* → cancellation → robustness)
5. Discussion → Conclusion → 11 appendices

The skeleton is standard. But the argument's spine is *coverage → semantic gap → register decomposition → cancellation*, and the decomposition is not visible in the ToC as the central apparatus. The reader only discovers that the whole paper turns on INLP in Methodology §3.8.

## 3. Planned research design or evolving project?

**Evolving project.** The fingerprints are unambiguous:

- The paper's stated innovation — separating topic from register so the "adjusted (topic) gap" can be the primary estimate — is introduced mid-Methodology (§3.8), validated only in Appendix G, and its headline numbers are repeated **six times** (abstract, intro, results, discussion, conclusion, Appendix F). Repetition at that density is the signature of sections written at different times, each restating the full summary.
- The original metric is retroactively demoted by *naming*: the raw gap is repeatedly called the **"raw gap (naive baseline)"** (lines 366, 804). A "naive baseline" is a label applied after a better method exists, not at design time.
- Hypotheses H1a–H1d appear for the first time at the *end* of Methodology (§3.9), never derived in the literature review or intro. A designed study states its hypotheses where the reader is deciding what to believe, not after the instruments are built.
- Appendix H begins: **"Advisor review flagged the four-term keyword query"** (line 798). That sentence is a time-stamp of a review cycle. Appendix A.3 is literally titled **"Truncation Fix"** and opens "During pipeline development, a truncation issue was discovered and corrected" (line 527). Appendix C.2 carries the heading-prose **"Why the raw robustness result does not carry over"** (line 604). Appendix G §5 says it "reports the robustness investigation as it unfolded" (line 756). These are process diaries, not supporting material.
- The empirical "headline" — the cancellation — is explicitly framed in the abstract as "tentative rather than established" (line 92). A headline the author must so aggressively hedge is usually a *discovered* result that was later promoted to organizing claim.

## 4. Sections that feel added later

| Section | Why it feels bolted on |
|---|---|
| §3.8 Register Adjustment (INLP) + Appendices F, G | The analytic crux of the paper, yet squeezed in as one methods subsection with its validation exiled to an appendix that ends "consistent with... rather than confirmed" (G, line 769). |
| §4.3 "A Possible Cancellation" | The word "Possible" in a results heading; the phenomenon is the *retroactive* headline, not a pre-registered expectation. |
| §4.4 Robustness of the Gap Rankings | A catalog whose own text admits stability is conditional ("not equally stable across all measurement choices", line 419). Reads as a set of reviewer-driven checks assembled into one section. |
| Appendices C, D, F, G, H, I, J | Eleven appendices for a ~60-page argument. Several narrate debugging history (C.2, C.3, F.1, G.5) or review responses (H). |
| §5.2 "Robust Patterns of Divergence" | A Discussion subsection that re-reads the robustness grid — i.e., a robustness digest written after the fact. |

**First-impression verdict:** a coherent, well-written main text that has had a second study (the register decomposition and its consequences) surgically inserted into a first study's skeleton. A careful reader will notice the seams.

---

# Phase 2 — Architectural dependency map

## The intellectual dependency graph

```
RQ: coverage & framing divergence, and their relation
   │
   ├──► Framework: coverage vs. framing + 3-level construct (§2.2)
   │         │
   │         ├──► Data: research corpus, policy corpus, reference corpora (§3.1–3.2, §5.1)
   │         │
   │         ├──► Measurement: embeddings + segmentation + supervised LR (§3.3–3.5)
   │         │         │
   │         │         ├──► Coverage gap (§3.6)
   │         │         └──► Semantic gap (§3.7)
   │         │                   │
   │         │                   └──► Register decomposition (INLP) (§3.8) ──► adjusted gap (PRIMARY)
   │         │                             │
   │         │                             └──► Register validation (Appendix G) ← needed to trust primary
   │         │
   │         ├──► Hypotheses H1a–H1d (§3.9) ──► Interaction / cancellation (§4.3)
   │         │
   │         └──► Robustness grid (§4.4 + App. C/D/F/H/I/J)
   │
   └──► Discussion claims: SDG17 divergent, SDG15 least, cancellation as
        knowledge-system difference, coverage≠framing for monitoring (§5, §6)
```

## What depends on what

1. **The research question** depends on the two-dimension (coverage/framing) concept.
2. **The framework** (§2.2) depends on the bibliometric review (§2.1) and the theory literature (§2.3). Order is fine.
3. **The gap metrics** depend on the classifier and embeddings. Fine.
4. **The adjusted (primary) gap** depends on the INLP decomposition (§3.8), which in turn depends on the *register concept* (§2.4 — Biber) **and** on the reader trusting the decomposition. The trust evidence is in Appendix G, placed after Results.
5. **The cancellation claim** (the paper's headline) depends on: (a) the H1a–H1d hypotheses, (b) the decomposition, (c) n=17 power analysis. Two of these three (hypotheses and decomposition) are introduced *after* the measurement machinery, and the third (power) is disclosed only when the result is borderline.
6. **Discussion claims** depend on the adjusted gap being accepted as valid and on the robustness grid. The robustness grid depends on Appendix I, whose conclusions are pre-stated in §4.4.
7. **The conclusion** depends on the framework (§2.2), the adjusted gap, and the cancellation — i.e., the conclusion is the only place that restates the *whole* argument, which is more than a summary should need to do if the argument had been cumulative.

## Ordering violations

| # | Violation | Evidence |
|---|---|---|
| V1 | **Methods explained before their motivation.** H1a–H1d (the study's only formal hypotheses) are introduced in Methodology §3.9, *after* all instruments are built. The literature review's "Research Gap" (§2.5) ends without deriving them; the intro's RQ is never decomposed into testable hypotheses. The reader is asked to accept the gap metrics before being told the questions they answer. | §3.9 (lines 285–287); §2.5 (lines 166–168) |
| V2 | **Validity evidence provided after the claim it supports.** The adjusted gap is declared "the primary estimate" in Results §4.2–4.3 and is the basis of the Discussion/Conclusion, but the independent evidence that the removed subspace is really *register* (Appendix G) appears only after the reader has accepted the adjusted gap. G itself concludes the support is partial, not confirmatory. | §4.2 (line 344), G conclusion (line 767–769) |
| V3 | **Hypotheses inserted where an analysis plan belongs.** §3.9 is titled "Coverage–Semantic Interaction" — it is really the hypotheses + estimation plan, and its correct home is the end of the literature review or a dedicated "Conceptual framework and hypotheses" chapter. | §3.9 (lines 285–287) |
| V4 | **Concepts introduced after they are already load-bearing.** "Register" is defined in §2.4 and operationalised in §3.8, which is chronologically correct; but the *register-vs-topic decomposition* as the paper's central analytical move is never foreshadowed in the intro's roadmap, and the reader meets it as a method, not as the contribution. | Intro roadmap (line 115) vs. §3.8 |
| V5 | **Robustness defenses embedded before the quantities they defend exist.** §3.1 (research corpus) and §3.5 (classifier) contain multi-sentence "this asymmetry does not threaten the findings" defenses that forward-reference appendices the reader has not been motivated to care about. These are answers to objections the reader has not yet had. | §3.1 (lines 180–182), §3.5 (line 226) |
| V6 | **Pipeline summary placed at the end of the methods.** §3.10 summarises a pipeline the reader has just finished reading. A summary of the design belongs at the *start* of Methodology as an orientation. | §3.10 (lines 289–291) |
| V7 | **Distributional-robustness motivation appears after its contradiction.** Appendix C.2 explains "why the raw robustness result does not carry over" — i.e., the metric battery was added only after the register decomposition broke an earlier-looking raw robustness result. The reader encounters the fix before the problem is stated. | C.2 (lines 599–604) |

---

# Phase 3 — "Patchwork detection"

## A. Methods introduced too late

### A1. INLP register adjustment is the paper's central method but arrives as one subsection among nine, with its validation in the appendix — HIGH severity
The paper's entire distinctiveness — the topic/register decomposition that produces the adjusted primary gap and the "cancellation" — is the INLP procedure. Yet structurally it is presented as a routine sub-step of the semantic-gap measurement (§3.8), the hypotheses it serves come after it (§3.9), and the evidence that it does what it claims is exiled to Appendix G, whose own conclusion is "consistent with the validation evidence rather than confirmed by it" (line 769). A method that the conclusion, abstract, and title-adjacent framing all rest on cannot have its warrant parked in an appendix that ends in "not confirmed."
**Why it feels wrong:** the abstract leads with the framework; the *mechanism* is back-loaded; the *trust* is further back-loaded.
**Fix:** either (a) foreground the decomposition: make "measuring topic and register separately" the stated objective from the intro, present the adjusted gap as the primary outcome in Results with a one-paragraph main-text summary of the G validation, and keep G as full detail; or (b) demote the cancellation to a secondary result and let the framework be the headline (see Phase 4).

### A2. Robustness grid reads as reviewer-driven additions — MEDIUM
§4.4 plus Appendices C, D, F, H, I, J are a wall of robustness: encoder sensitivity, cross-sensitivity, sample stability, balanced subsets, concept retrieval, distributional metrics, pooled regressions, assignment-method comparisons. Each is defensible; *together* they say "this finding needed a lot of defending." Appendix H's "Advisor review flagged..." (line 798) is the smoking gun.
**Why it feels wrong:** a designed study chooses its sensitivity axes up front and reports them once, with a single synthesis. Here the axes accumulated, and §4.4 must supply the synthesis (§5.2 re-supplies it, and limitations §5.4 re-supplies it again).
**Fix:** consolidate the robustness story into one Results section with one synthesis paragraph; move raw tables to a single "Robustness and sensitivity" appendix; delete the meta-commentary (below).

### A3. Zero-shot nearest-centroid is a vestigial method — LOW severity but a visible seam
The zero-shot method is scoped out of the paper's own claims (MPNet-only, one comparison, Appendix I.4) and its AGENTS.md status is explicitly "do not re-add." In the dissertation it appears in §4.4, Appendix I.4, and limitations §5.4. A method kept only to prove it was tried is developmental residue.
**Fix:** either cut it entirely, or state in §4.4 that a single supervised-vs-unsupervised assignment comparison was run as a boundary check and leave one appendix row.

## B. Concepts introduced in the wrong chapter

### B1. The register concept is introduced as a caution (§2.4) and a method (§3.8), but its interpretative payoff is only established in Discussion — MEDIUM
§2.4 correctly motivates register (Biber, "embedding distance mixes topic, register, and contextual convention"). But the *decomposition into topic vs. register* — the move that makes the cancellation possible — is first presented as machinery. Its full interpretative weight ("a structural difference between knowledge systems", §5.1) only lands in Discussion, three chapters after the reader first needed to know that "register" was going to be the load-bearing analytical object.
**Fix:** state in §2.4 (or the framework §2.2) that the study will not merely flag register but *measure and remove it*, so the method in §3.8 is an anticipated plan, not a surprise.

### B2. Limitations arrive in Discussion but are already being defended in Methodology — LOW
§3.1 and §3.5 contain inline "the asymmetry does not threaten the findings" defenses, while the formal limitations are §5.4. The reader gets the answer twice, in the wrong order: the defense precedes the objection.
**Fix:** strip the defensive paragraphs from §3.1/§3.5 to their factual content, and let §5.4 carry the adjudication. (This also shortens the methods chapter.)

### B3. Appendix C.3, "Implications for Interpreting the Main Estimates," is a Discussion chapter inside an appendix — MEDIUM
Appendix C.3 (lines 608–612) re-litigates the classifier choice ("these diagnostics are methodological stress tests... not failed repairs"; "the hard-threshold classifier assignment is therefore retained"). This is the voice of an author arguing with their own robustness work, and it is material that belongs either in Methodology's design rationale or in Discussion — not in an appendix.
**Fix:** move the substance to §3.5 (design rationale) or §5.4 (limitations); delete the appendix subsection.

## C. Results that feel detached

### C1. The cancellation is reported before the reader is equipped to evaluate it — HIGH
The near-zero raw correlation, the decomposition, the SDG 17/13 inversion, and the borderline p-values (0.054 / 0.045) are previewed in the abstract (line 92) and the intro (line 113) *before* the reader has seen either the raw gap or the INLP procedure. Then §4.3 presents the cancellation as a finding. The structure reverses the epistemic order: the claim precedes the machinery the reader needs to judge it.
**Fix:** remove the p-value preview from the intro (the abstract may keep one sentence); let §4.2–4.3 deliver raw gap → decomposition → cancellation in order.

### C2. The SDG 4 lexical artefact — the single most consequential data caveat — is relegated to an appendix — HIGH
The SDG 4 audit (Appendix B.1) qualifies one of the paper's own headline coverage results (SDG 4's inflated third-place rank, which contradicts prior bibliometrics). The main text flags it repeatedly (lines 330, 336, 377, 404), but the actual analysis is in an appendix. A caveat that must be cited four times is a main-text element.
**Fix:** promote a compact SDG-4 paragraph to Results §4.1 (with the audit table retained in the appendix), so the reader sees the caveat where the claim is made.

### C3. Multiple appendix sections are process diaries, not supporting material — MEDIUM
- A.3 "Truncation Fix": documents a mid-development bug ("a truncation issue was discovered and corrected"). This is a git log entry, not a scientific appendix. One sentence ("segmentation enforces the encoder's token budget; truncation was verified at ≤x%") suffices.
- C.2 "Why the raw robustness result does not carry over": narrates the contradiction discovered when robustness metrics were added after register removal.
- F.1 duplicates §4.2's SDG 17 register-component story verbatim (lines 366 vs. 697).
- G.5 "residual register diagnostics and robustness investigation": explicitly "reports the robustness investigation as it unfolded" and "This history is reported deliberately" (lines 756, 762). The *history* of an analysis is not evidence.
**Fix:** compress A.3 to a sentence; cut F.1's narrative duplication to a pointer to §4.2; convert G.5 from investigation-narrative to results ("draw-stability and clustering checks show the apparent residual signals are not stable"); delete C.3.

### C4. The "possible cancellation" is an inductive result carrying the paper's weight — HIGH
Read honestly, the empirical arc is: H1a on the raw gap is null → the author decomposes the gap → a borderline (n=17) two-signal cancellation emerges → this tentative finding becomes the paper's headline, the Discussion's centerpiece, and the Conclusion's first paragraph. Structurally, the paper has *built a cathedral on a p=0.054 finding whose mechanism is only partially validated*. That is the single largest architectural risk: the framing makes the reader believe the whole apparatus exists to establish the cancellation, and then the numbers are honest enough to admit it is tentative.
**Fix:** re-weight the paper so the *framework + measurement protocol* is the thesis (robust, replicable, portable), and the cancellation is "a first application suggests..." — the Discussion already says this ("The framework is the main contribution; the alignment findings are a first application," line 439); the *structure* must be made to agree with that sentence. The intro and abstract should not preview the cancellation as the headline.

---

# Phase 4 — Publication-level restructuring

No content is rewritten; the proposal is order, composition, and naming.

## 4.1 Ideal chapter structure

1. **Introduction** — RQ; the two-dimension claim; the register problem stated as part of the research problem (not a discovery); roadmap. No p-value preview.
2. **Literature Review** — §2.1 bibliometrics; §2.2 three-level framework; §2.3 theory of divergence; §2.4 semantic methods **including a forward statement that the study will decompose topic vs. register**; §2.5 research gap **ending in the explicit hypotheses H1a–H1d**.
3. **Methodology**
   - *3.0 Pipeline overview* (move §3.10's figure here, at the start)
   - Research corpus / Policy corpus (facts only; defenses moved to Limitations)
   - Reference data and supervised classifier
   - Embedding, normalisation, segmentation
   - Measurement: coverage gap; semantic gap
   - Register decomposition (INLP) — same content, now presented as *the* measurement instrument, with a one-paragraph validity summary (moved from G)
   - Analysis plan: the H1a–H1d operationalisation (now that hypotheses already exist in §2.5)
4. **Results**
   - 4.1 Classifier validation → coverage profiles (incl. the promoted SDG-4 paragraph)
   - 4.2 Semantic gaps: raw and adjusted; register decomposition explains the inversion
   - 4.3 Interaction: the cancellation, stated as *the* test of H1a–H1d
   - 4.4 Robustness (single consolidated section with one synthesis)
5. **Discussion** — two interpretive subsections (dimensions; robust patterns), implications, limitations. Trim result-repetition.
6. **Conclusion**
7. **Appendices** — regrouped (below).

## 4.2 Moves

- **Merge:** Appendix G's *conclusion* into §3.8 (one paragraph); Appendix C.3 into §5.4; Appendix F.1's narrative into §4.2; the three robustness appendices (C, I) into one "Robustness and sensitivity" appendix; D's balanced-subset into D's main flow.
- **Move:** the hypotheses from §3.9 to the end of the literature review; the pipeline figure from §3.10 to the opening of Methodology; the SDG-4 analysis into Results.
- **Cut/compress:** Appendix A.3 to one sentence; zero-shot (A3) to a single boundary-check sentence; §5.1/§5.2 result-repetition.
- **Rename:**
  - §3.9 "Coverage–Semantic Interaction" → "Hypotheses and Analysis Plan"
  - §4.1 "Supervised Reference Classifier and Coverage Gap" → "Classifier Validation and Coverage Profiles"
  - Appendix I "Supplementary Cross-Method Data" → "Cross-Method Robustness Data"
  - Appendix G title is fine; its §5 should be renamed from "residual register diagnostics and robustness investigation" to something result-shaped.

## 4.3 Principle applied

Every section exists because the reader needs it at that point: hypotheses before instruments (so the instruments answer questions), trust evidence before the primary estimate (so the estimate is credible), caveats where the claims are made (so the reader isn't re-reading after being surprised), robustness once (so it reads as design, not defense).

---

# Phase 5 — Specific ToC audit

Rating scale: **E** essential · **UM** useful but misplaced · **R** redundant · **U** unclear

## Abstract / front matter
| Item | Rating | Note |
|---|---|---|
| Abstract | E | Dense to the point of being a results summary; fine for a dissertation, but the cancellation is over-weighted relative to the framework that is the actual contribution. |
| Intro roadmap | E | "Sections X and Y report findings and interpretation" is boilerplate; it does not telegraph the register decomposition, the paper's actual center of gravity. **U**-adjacent. |

## Literature Review
| Item | Rating | Note |
|---|---|---|
| 2.1 Bibliometrics | E | Anchors the field; correctly shows SDG 3 lead. |
| 2.2 Three-level framework | E | Best-executed section; the paper's spine. |
| 2.3 Theory of divergence | E | Motivated. |
| 2.4 Semantic methods | E | Correctly introduces register — but should commit to decomposing it (B1). |
| 2.5 Research gap | UM | Stops before its logical endpoint: the hypotheses. **UM** — should end with H1a–H1d. |

## Methodology
| Item | Rating | Note |
|---|---|---|
| 3.1 Research corpus | E | Facts fine; the asymmetry *defense* is misplaced (B2). |
| 3.2 Policy corpus | E | |
| 3.3 Embedding model | E | |
| 3.4 Segmentation | E | |
| 3.5 Supervised classifier | E | Contains an over-long "probability gap" defense; trim. |
| 3.6 Coverage gap | E | Order is: problem → measurement → adjustment → estimation? Not quite. The gap metrics (3.6/3.7) come before the hypotheses (3.9) they serve, and the decomposition (3.8) precedes the question that needs it. The methodology follows *implementation order*, not argument order. |
| 3.7 Semantic gap | E | |
| 3.8 Register adjustment (INLP) | E but **UM** in presentation | The paper's central instrument appears as a routine sub-step; its validation summary is missing here (V2). |
| 3.9 Coverage–semantic interaction | **UM** | A hypotheses + analysis-plan section living at the end of Methods (V3). |
| 3.10 Methodology summary | **UM** | A pipeline overview placed after the pipeline is described (V6). |

## Results
| Item | Rating | Note |
|---|---|---|
| 4.1 Classifier + coverage | E | Correct: instrument → coverage. But heading merges two distinct results, and the SDG-4 caveat lives in the appendix (C2). |
| 4.2 Semantic gap after register removal | E | Correct sequence within the section (PCA → gap → decomposition). The PCA figure leads; the gap figure follows — defensible. |
| 4.3 Cancellation | E but **UM**-adjacent | The section is where the H1 hypotheses are *actually* tested — they should already exist (V1/V3). "A Possible Cancellation" as a heading signals inductive discovery. |
| 4.4 Robustness | E but **UM** | Correct placement (after findings), but it is a catalog; the synthesis paragraph does heavy lifting that the appendices should have shared. |

Results order verdict: main finding → interpretation → robustness is *mostly* honored, but the paper's true main finding (cancellation) is not framed as the pre-planned test of H1; it reads as "whatever was computed first" because the hypotheses arrived last.

## Discussion
| Item | Rating | Note |
|---|---|---|
| 5.1 Two distinct dimensions | E | Interprets; but re-states the cancellation numbers from §4.3. Trim to interpretation. |
| 5.2 Robust patterns | **UM** | A second robustness digest (repeats §4.4). Should be folded into 5.1 or trimmed. |
| 5.3 Implications | E | Genuinely new content. |
| 5.4 Limitations | E | Strong; but duplicates defenses already embedded in §3.1/§3.5 (B2). |

## Conclusion
| Item | Rating | Note |
|---|---|---|
| Conclusion | E | Correctly identifies the framework as the contribution; the opening paragraph re-states the cancellation at length — fine, but it exposes that the body never gave the framework the same prominence. |

## Appendices
| Item | Rating | Note |
|---|---|---|
| A.1 Retrieval/query | E | Supports methods. |
| A.2 Segmentation mechanics | E | |
| A.3 Truncation fix | **R** | Debugging diary. One sentence belongs in A.2. |
| A.4 Reference provenance | E | |
| B.1 SDG 4 audit | E but **UM** | Load-bearing caveat in an appendix (C2). |
| B.2 Centroid similarity | E | Supports classifier section. |
| C.1 Lexical illustration | E | Concrete, interpretable. |
| C.2 Distance-functional robustness | E but **UM** | The "why raw robustness doesn't carry over" narrative is process-history (C3). |
| C.3 Implications for interpreting main estimates | **R** | A discussion inside the appendix (B3). |
| D Sample stability (+ balanced subset) | E | Legitimate design check. The balanced-subset block is a floating paragraph with a stray `\label` mid-appendage — tidiness issue. |
| E Model selection | E | Supports §3.5/§4.1; a bit long (MLP grid detail is beyond need). |
| F Register convergence | E but **UM** | F.1 duplicates §4.2's SDG 17 narrative (C3); convergence table is the real content. |
| G Register validation | E content, **U** framing | *The* trust evidence for the primary estimate, after the fact; its own conclusion is "consistent with rather than confirmed" and its §5 is a research narrative. Must either be integrated into the argument's flow or explicitly framed as secondary support. |
| H Concept-retrieval sensitivity | E | Legitimate; the "advisor review flagged" opening is a time-stamp (A2). |
| I Supplementary cross-method data | E | The values/rank tables are needed; the *prose* subsections (I.4, I.5) narrate troubleshooting. |
| J Pooled regression | E | Supports the cancellation. |
| K Declaration of AI use | E | Compliance. |

---

# Phase 6 — Final verdict

## 1. Overall architectural grade

**Strong dissertation but needs restructuring** — one step below publication-level, and closer to it than to "visibly developmental" in the *main text*, but clearly developmental in the *appendix architecture* and in the *rhetorical weight given to the discovered cancellation*.

The main text's skeleton is defensible and well-written. What holds it back from publication-level is not prose: it is that (a) the paper's actual contribution (the topic/register measurement protocol) and its actual headline (the tentative cancellation) are out of proportion with each other; (b) the trust evidence for the primary estimate lives in an appendix whose own conclusion is "not confirmed"; (c) the hypotheses arrive after the instruments; and (d) eleven appendices containing multiple process diaries make the development history visible.

## 2. The three highest-impact structural changes

1. **Re-weight the thesis around the measurement framework, and demote the cancellation to a first-application result.** Make the intro, abstract, and conclusion agree with the Discussion's own sentence: "The framework is the main contribution; the alignment findings are a first application" (line 439). The cancellation stays — but as *suggestive evidence that the decomposition is useful*, not as the headline it is currently previewed as in the abstract and intro. This single move removes the paper's largest credibility risk (a tentative p≈0.05 finding carrying the whole load) and makes the structure match the science.
2. **Move the hypotheses (H1a–H1d) to the end of the literature review and the pipeline overview to the start of Methodology.** This converts the ordering violations V1, V3, V6 into a designed arc: question → instruments built to answer it → results as the test of the question.
3. **Bring the register decomposition's warrant into the argument's flow.** Put a one-paragraph validity summary (from Appendix G) into §3.8 or §4.2, promote the SDG-4 caveat into Results, and compress the process-history appendices (A.3, C.2, C.3, F.1, G.5's narrative) so the appendices read as supporting *evidence* rather than a debugging diary. This removes the visible "patched things together" fingerprints.

## 3. The three things that should NOT be changed

1. **The three-level framework (§2.2) and the coverage/framing decomposition as the organizing concepts.** This is the single best-designed element: it motivates the data, the metrics, and the interpretation in one clean stroke. Do not touch it.
2. **The results arc — instrument validation → coverage → semantic gap → interaction → robustness.** Within each section the internal order (establish the measure, report the gap, decompose, then stress-test) is correct and cumulative. Keep it.
3. **The raw/adjusted panel convention and the "primary estimate vs. baseline" labeling discipline in the rank tables.** The consistent Panel (a) adjusted / Panel (b) raw structure across every robustness table, and the cross-sensitivity grid as a template for bounding uncertainty, are genuinely publication-grade. Keep this machinery exactly as it is; it is the strongest evidence that the author can do top-tier empirical work.

---

**Bottom line.** The science is strong and honestly reported. The architecture is a good first study with a second study inserted, and the current structure lets the reader see the insertion. Restructure to let the measurement framework — which is the real contribution and is already well-built — be the spine, and the dissertation will read as one designed argument rather than an honest but visible development history. The single most important sentence already exists in the manuscript at line 439; the structure should be made to say it first.
