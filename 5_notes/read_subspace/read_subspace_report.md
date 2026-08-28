# INLP register subspace — word inspection (MPNet canon)

G = 62 orthonormalized removed directions (x 768). Each direction is the corpus-discrimination vector from one SDG-stratified INLP iteration, orthogonalized against earlier ones.

## Per-direction top words (first 6 removed directions)

- **iter 1**: regulation (+0.25), infrastructure (+0.24), governments (+0.22), legislation (+0.21), sustainability (+0.21), governance (+0.21), biodiversity (+0.20), health (+0.20), countries (+0.20), sustainable (+0.19), education (+0.17), level (+0.17), indicate (+0.17), cities (+0.17), development (+0.17)
- **iter 2**: legislation (+0.27), governance (+0.25), sustainability (+0.23), infrastructure (+0.22), governments (+0.20), policy (+0.20), countries (+0.20), sustainable (+0.19), health (+0.19), regulation (+0.19), nations (+0.19), climate (+0.18), biodiversity (+0.18), oceans (+0.17), development (+0.16)
- **iter 3**: legislation (+0.17), governance (+0.13), policy (+0.13), sustainability (+0.13), should (+0.12), could (+0.11), implementation (+0.11), proposed (+0.11), sustainable (+0.11), national (+0.10), experiment (+0.10), report (+0.09), set (+0.09), number (+0.09), voluntary (+0.09)
- **iter 4**: implementation (+0.14), policy (+0.14), poverty (+0.12), governance (+0.12), innovation (+0.11), sustainability (+0.11), should (+0.11), legislation (+0.11), consumption (+0.10), stakeholders (+0.10), partnership (+0.10), inequality (+0.09), development (+0.09), commitment (+0.09), infrastructure (+0.09)
- **iter 5**: governments (+0.15), poverty (+0.15), policy (+0.14), sustainability (+0.14), infrastructure (+0.14), sustainable (+0.13), commitment (+0.12), implementation (+0.12), governance (+0.12), consumption (+0.12), legislation (+0.12), partnership (+0.12), researchers (+0.12), stakeholders (+0.11), architecture (+0.11)
- **iter 6**: countries (+0.13), sustainability (+0.12), governments (+0.12), nations (+0.11), stakeholders (+0.11), partnership (+0.11), poverty (+0.10), sustainable (+0.10), strategy (+0.10), legislation (+0.09), infrastructure (+0.09), oceans (+0.09), innovation (+0.09), researchers (+0.09), cities (+0.09)

## Aggregate: words most aligned with the removed subspace (mean cosine over all iters)

- sustainability [topic] +0.023
- governance [register] +0.023
- biodiversity [topic] +0.021
- poverty [topic] +0.020
- stakeholders [policy-lean] +0.018
- legislation [policy-lean] +0.017
- sustainable [topic] +0.015
- nations [register] +0.014
- commitment [register] +0.014
- inequality [topic] +0.013
- countries [register] +0.012
- strategy [register] +0.011
- established [register] +0.010
- governments [policy-lean] +0.010
- education [topic] +0.010
- development [register] +0.010
- energy [topic] +0.010
- consumption [topic] +0.009
- partnership [register] +0.009
- climate [topic] +0.009
- performed [register] +0.008
- implementation [register] +0.008
- peace [topic] +0.008
- land [topic] +0.007
- policy [register] +0.007
- national [policy-lean] +0.006
- gender [topic] +0.006
- performance [research-lean] +0.006
- infrastructure [topic] +0.005
- authors [register] +0.005

## Bin means (does the removed subspace track register or topic?)

- policy-lean: +0.007
- register: +0.001
- research-lean: -0.002
- topic: -0.003
- neutral: -0.004

## Corpus-leaning per direction (research-probe minus policy-probe mean cosine)

- iter 1: research +0.072, policy +0.164, diff -0.092
- iter 2: research +0.107, policy +0.170, diff -0.063
- iter 3: research +0.043, policy +0.078, diff -0.035
- iter 4: research +0.042, policy +0.071, diff -0.028
- iter 5: research +0.044, policy +0.083, diff -0.039
- iter 6: research +0.018, policy +0.055, diff -0.037
- iter 7: research +0.014, policy +0.057, diff -0.042
- iter 8: research +0.017, policy +0.051, diff -0.033
- iter 9: research -0.026, policy +0.026, diff -0.052
- iter 10: research -0.018, policy +0.029, diff -0.047
