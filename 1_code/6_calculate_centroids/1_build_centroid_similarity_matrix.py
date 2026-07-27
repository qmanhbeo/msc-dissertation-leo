"""
Build centroid similarity matrix CSV from reference centroids.

Computes pairwise cosine similarity between all 17 SDG reference centroids
and exports the resulting 17x17 matrix as a CSV for manuscript consumption.

Output:
  {output_dir}/main/data/4_1_centroid_similarity_matrix.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
ANALYSIS_DIR = CODE_ROOT / "7_main_analysis" / "0_shared"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from model_utils import DEFAULT_EMBED_MODEL, DEFAULT_OUTPUT_ROOT, N_SDG, scored_dir_for_model

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build centroid similarity matrix CSV.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"Embed model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Manuscript output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    out_path = output_dir / "main" / args.embed_model / "data" / "4_1_centroid_similarity_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        log.info("Centroid similarity matrix already exists: %s (use --overwrite to rebuild)", out_path)
        return

    scored_root = scored_dir_for_model(args.embed_model)
    centroids_path = scored_root / "sdg_centroids.npy"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Reference centroids not found: {centroids_path}")

    centroids = np.load(centroids_path).astype(np.float32)
    log.info("Loaded centroids: %s", centroids.shape)

    sim = centroids @ centroids.T
    assert sim.shape == (N_SDG, N_SDG), f"Unexpected similarity shape: {sim.shape}"

    labels = [f"SDG {i+1}" for i in range(N_SDG)]
    header = "," + ",".join(labels)
    rows = [header]
    for i in range(N_SDG):
        row = labels[i] + "," + ",".join(f"{sim[i, j]:.6f}" for j in range(N_SDG))
        rows.append(row)

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log.info("Saved centroid similarity matrix: %s", out_path)


if __name__ == "__main__":
    main()
