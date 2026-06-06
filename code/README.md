# Code Surface

`code/` contains only the active pipeline used by `main.py`.

Structure:
- `0_fetch/` source acquisition
- `1_preprocess/` corpus cleaning, filtering, and shard building
- `2_embed/` embeddings, centroids, and scoring producers
- `3_main_analysis/` coverage, semantic, interaction, and sample-stability analysis
- `4_visualization/` dissertation figures
- `data_backup_and_fetch/` operator backup and marker-facing frozen data bootstrap utilities
- `shared_utils.py` canonical output helpers and shared path utilities

Operator documentation lives in `../README.md`.
