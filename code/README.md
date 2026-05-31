# Code Surface

`code/` contains only the active pipeline used by `main.py`.

Structure:
- `0_fetch/` source acquisition
- `1_preprocess/` corpus cleaning, filtering, and shard building
- `2_embed/` embeddings, centroids, and scoring producers
- `3_main_analysis/` coverage, semantic, and interaction analysis
- `4_visualization/` dissertation figures
- `shared_utils.py` canonical output helpers and shared path utilities
- `backup_data_snapshot.py` data backup utility

Operator documentation lives in `../README.md`.
