# Register-adjustment robustness suite

This folder contains appendix-style robustness and diagnostic checks.
The raw within-SDG semantic gap remains the main estimand.

1. `sdg_balanced` trains one global research-vs-policy classifier on an SDG-balanced sample.
2. `within_sdg` trains separate research-vs-policy classifiers within each SDG.
3. `regression` treats each embedding coordinate as an outcome and estimates register, SDG, and register x SDG effects by exact OLS cell contrasts.

How to run:
- `python code/3_main_analysis/3_appendix/3_register_adjustment.py --method both`

Active configuration:
- `random_seed = 42`
- `classifier_type = logistic_regression_liblinear`
- `samples_per_cell = None`
- `min_samples_per_class = 50`
- `test_size = 0.2`

How to interpret:
- Smaller adjusted gaps do not automatically mean a better estimate of the dissertation's target quantity.
- The one-direction global subtraction and the SDG-balanced global subtraction are sensitivity checks for broad corpus-level register effects.
- The within-SDG classifier and within-SDG regression procedures are stress tests: they can over-subtract by learning the very within-goal research-policy contrast the raw gap is meant to measure.
- The regression method estimates register-associated embedding variation after controlling for SDG and compares that direction directly with the classifier-derived direction.
- The register_vector_cosine_similarity.csv file shows whether the within-SDG directions are broadly stable or highly heterogeneous.
