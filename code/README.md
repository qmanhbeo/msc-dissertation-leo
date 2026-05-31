# Code Directory (Active Surface)

`code/` contains only active, supported pipeline scripts.

Structure:
- `0_fetch/` data ingestion scripts
- `1_preprocess/` cleaning, filtering, and corpus assembly scripts
- `2_embed/` embedding, centroid, and shard scoring scripts
- `3_main_analysis/` coverage/semantic analysis scripts and shared analysis utilities
- `4_visualization/` figure generation scripts
- `shared_utils.py` shared run-output utilities used across stages
- root: `backup_data_snapshot.py` (operations utility)

Canonical workflow and dissertation-facing methodology notes are documented in:
- `../README.md`

Legacy/experimental scripts have been moved to:
- `../_legacy/code/`
