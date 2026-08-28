# P0 — Curated word probe + permutation null (MPNet canon)

G = 62 orthonormal removed directions x 768. Single-word embeddings in the frozen MPNet space; cosine with each g_k.

## Bin means (mean cosine of bin words with the removed subspace)

- policy-lean: +0.0074
- register: +0.0008
- research-lean: -0.0021
- topic: -0.0032
- neutral: -0.0039

Register - Topic gap = +0.0040
Permutation null (5000): p(>=obs) = 0.096, two-sided p = 0.186
Interpretation: if the gap is not distinguishable from the permutation null, the single-word probe does not support a register>topic reading.

## Aggregate top words (mean cosine over all iters)

- sustainability [topic] +0.0233
- governance [register] +0.0229
- biodiversity [topic] +0.0207
- poverty [topic] +0.0201
- stakeholders [policy-lean] +0.0180
- legislation [policy-lean] +0.0168
- sustainable [topic] +0.0148
- nations [register] +0.0142
- commitment [register] +0.0141
- inequality [topic] +0.0134
- countries [register] +0.0124
- strategy [register] +0.0110
- established [register] +0.0104
- governments [policy-lean] +0.0103
- education [topic] +0.0100
- development [register] +0.0099
- energy [topic] +0.0096
- consumption [topic] +0.0090
- partnership [register] +0.0089
- climate [topic] +0.0088
- performed [register] +0.0085
- implementation [register] +0.0081
- peace [topic] +0.0075
- land [topic] +0.0073
- policy [register] +0.0069
- national [policy-lean] +0.0064
- gender [topic] +0.0063
- performance [research-lean] +0.0062
- infrastructure [topic] +0.0051
- authors [register] +0.0049

## First 3 directions' top words

- iter 1: regulation (+0.25), infrastructure (+0.24), governments (+0.22), legislation (+0.21), sustainability (+0.21), governance (+0.21), biodiversity (+0.20), health (+0.20), countries (+0.20), sustainable (+0.19), education (+0.17), level (+0.17), indicate (+0.17), cities (+0.17), development (+0.17)
- iter 2: legislation (+0.27), governance (+0.25), sustainability (+0.23), infrastructure (+0.22), governments (+0.20), policy (+0.20), countries (+0.20), sustainable (+0.19), health (+0.19), regulation (+0.19), nations (+0.19), climate (+0.18), biodiversity (+0.18), oceans (+0.17), development (+0.16)
- iter 3: legislation (+0.17), governance (+0.13), policy (+0.13), sustainability (+0.13), should (+0.12), could (+0.11), implementation (+0.11), proposed (+0.11), sustainable (+0.11), national (+0.10), experiment (+0.10), report (+0.09), set (+0.09), number (+0.09), voluntary (+0.09)