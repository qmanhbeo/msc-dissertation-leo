# Code Directory (Active Surface)

`code/` contains only active, supported pipeline scripts.

Structure:
- `fetch/` data ingestion scripts
- `preprocess/` cleaning, filtering, and corpus assembly scripts
- `embed/` embedding, centroid, and shard scoring scripts
- `main_analysis/` coverage/semantic analysis scripts and shared analysis utilities
- `visualization/` figure generation scripts
- root: `backup_data_snapshot.py` (operations utility)

Canonical workflow and dissertation-facing methodology notes are documented in:
- `../README.md`

Legacy/experimental scripts have been moved to:
- `../archive/legacy/code/`
