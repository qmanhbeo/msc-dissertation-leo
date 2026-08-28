# P1+P2 — Direction-space validity of the register reading (MPNet canon)

G = 62x768. Sample: 38182 segments (1123/SDG/corpus). All LRs use C=1.0 (matching INLP). No re-embed.

## P1 — Topic-direction overlap with span(G)

- Topic coef alignment with span(G): mean=0.248, max=0.430
- Topic coef alignment with a RANDOM 62-dim subspace: mean=0.286 (expected 0.284)
- Corpus(register) coef alignment with span(G): 0.979 (vs random 0.242) -- sanity, should be near 1

Interpretation: if topic alignment ~ random subspace (0.28), G did NOT preferentially remove topic. If corpus alignment ~1, span(G) captures register.

## P2 — Held-out-SDG generalization (falsifies T1 if it collapses)

- Held-out-SDG corpus accuracy (mean over 17 folds): 0.967
- Full-data corpus accuracy (mean over 17 SDGs): 0.978
- Held-out corpus directions' alignment with span(G): mean=0.981
- TOPIC CONTROL: SDG1 vs SDG2 direction trains at 0.991 but classifies OTHER SDGs' research at 0.781 (chance=0.5) -- topic generalizes far less than register.
- BALANCED DATA-SIZE CONTROL: register direction trained on ONLY SDG1+SDG2 reaches train 0.976, and STILL classifies the other 15 SDGs' research/policy at 0.920 -- so the 0.967 held-out result is NOT merely a training-mass artifact (register generalizes even from 2 SDGs).

Interpretation: if held-out corpus accuracy stays well above chance, the removed register direction generalizes across topics (supports register reading). If it collapses toward chance, the removed subspace is SDG/topic-specific (falsifies the register reading).
