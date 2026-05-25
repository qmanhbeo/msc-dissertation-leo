"""
Orchestrate the full corpus shard pipeline with resume-safe stages.

Stages:
  1) preprocess_papers_streaming.py
  2) embed_paper_shards.py
  3) score_paper_shards.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument("--limit", type=int, default=0, help="Debug limit for clean stage kept rows.")
    p.add_argument("--limit-shards", type=int, default=0, help="Debug limit for embed/score stages.")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    py = sys.executable

    clean_cmd = [
        py,
        "code/preprocess_papers_streaming.py",
        "--shard-size",
        str(args.shard_size),
    ]
    if args.limit > 0:
        clean_cmd.extend(["--limit", str(args.limit)])
    run(clean_cmd)

    embed_cmd = [
        py,
        "code/embed_paper_shards.py",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.local_files_only:
        embed_cmd.append("--local-files-only")
    if args.limit_shards > 0:
        embed_cmd.extend(["--limit-shards", str(args.limit_shards)])
    run(embed_cmd)

    score_cmd = [py, "code/score_paper_shards.py"]
    if args.limit_shards > 0:
        score_cmd.extend(["--limit-shards", str(args.limit_shards)])
    run(score_cmd)

    print("\nFull pipeline finished.")


if __name__ == "__main__":
    main()
