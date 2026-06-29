from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SNAPSHOT_METADATA_DIR = Path("_snapshot_metadata")
SNAPSHOT_METADATA_FILE = SNAPSHOT_METADATA_DIR / "snapshot_manifest.json"

CURATED_EXCLUDED_PATHS = (
    Path("0_raw/openalex"),
    Path("3_scored/paper_sample_seed_42_141"),
    Path("3_scored/register_adjustment_cache"),
    Path("3_scored/paper_scores_shards/metadata/subset_index.sqlite"),
)

BASE_WARM_REPLAY_PATHS = (
    Path("2_data/2_embedded/policy.npy"),
    Path("2_data/2_embedded/metadata/policy_ids.json"),
    Path("2_data/2_embedded/osdg.npy"),
    Path("2_data/2_embedded/benchmark.npy"),
    Path("2_data/2_embedded/research_shards/metadata/manifest.json"),
    Path("2_data/3_scored/sdg_centroids.npy"),
    Path("2_data/3_scored/paper_scores_shards/metadata/manifest.json"),
    Path("2_data/1_preprocessed/policy_all/policy_chunks_all.jsonl"),
    Path("2_data/0_raw/policy_manual/artifact/convert_policy_manual_summary.json"),
)

REGISTER_REPLAY_EXTRA_PATHS = (
    Path("2_data/1_preprocessed/research_corpus/metadata/manifest.json"),
    Path("2_data/1_preprocessed/research_corpus/part-00001.jsonl"),
)

FULL_PIPELINE_WARNING_LINES = (
    "A live-source full-pipeline rerun is not expected to be byte-identical to the frozen snapshot.",
    "OpenAlex content changes over time.",
    "Some fetched policy links may drift or change their payloads.",
    "The manual policy supplement includes documents that are not fully automated from stable source URLs.",
)

REPRODUCIBILITY_WARNING_LINES = (
    "Warm replay from the frozen snapshot is the primary reproducibility target.",
    "A live-source full-pipeline rerun is not guaranteed to match the frozen snapshot exactly.",
    "OpenAlex changes over time, scraper links may drift, and the manual policy supplement is not fully automatable from stable URLs.",
)


@dataclass(frozen=True)
class SnapshotProfile:
    name: str
    description: str
    excluded_data_paths: tuple[Path, ...]
    expected_repo_paths: tuple[Path, ...]


SNAPSHOT_PROFILES: dict[str, SnapshotProfile] = {
    "full": SnapshotProfile(
        name="full",
        description="Literal data/ snapshot, including rebuildable caches and raw OpenAlex fetch artifacts.",
        excluded_data_paths=(),
        expected_repo_paths=BASE_WARM_REPLAY_PATHS + REGISTER_REPLAY_EXTRA_PATHS,
    ),
    "curated": SnapshotProfile(
        name="curated",
        description="Marker-facing replay snapshot with warm-replay inputs preserved and large rebuildable caches removed.",
        excluded_data_paths=CURATED_EXCLUDED_PATHS,
        expected_repo_paths=BASE_WARM_REPLAY_PATHS + REGISTER_REPLAY_EXTRA_PATHS,
    ),
}


def get_snapshot_profile(name: str) -> SnapshotProfile:
    try:
        return SNAPSHOT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown snapshot profile: {name}") from exc


def snapshot_archive_prefix(profile_name: str) -> str:
    if profile_name == "full":
        return "dissertation-data-snapshot"
    return f"dissertation-data-snapshot-{profile_name}"


def should_exclude_data_path(rel_data_path: Path, profile: SnapshotProfile) -> bool:
    rel = Path(rel_data_path)
    for excluded in profile.excluded_data_paths:
        if rel == excluded or excluded in rel.parents:
            return True
    return False


def manual_policy_inventory(source_dir: Path) -> list[str]:
    manual_root = source_dir / "0_raw" / "policy_manual"
    texts_dir = manual_root / "texts"
    pdf_dir = manual_root / "pdf"
    text_stems = {path.stem for path in texts_dir.glob("*.txt")} if texts_dir.exists() else set()
    pdf_stems = {path.stem for path in pdf_dir.glob("*.pdf")} if pdf_dir.exists() else set()
    paired = sorted(text_stems & pdf_stems)
    return paired or sorted(text_stems | pdf_stems)


def current_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_snapshot_metadata(
    *,
    workspace_root: Path,
    source_dir: Path,
    profile: SnapshotProfile,
    snapshot_version: str,
) -> dict:
    return {
        "snapshot_version": snapshot_version,
        "profile": profile.name,
        "description": profile.description,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "workspace_root": str(workspace_root),
        "source_dir": str(source_dir),
        "source_commit": current_git_commit(workspace_root),
        "included_root": "2_data/",
        "excluded_data_paths": [str(path) for path in profile.excluded_data_paths],
        "expected_repo_paths": [str(path) for path in profile.expected_repo_paths],
        "reproducibility_warnings": list(REPRODUCIBILITY_WARNING_LINES),
        "manual_policy_documents": manual_policy_inventory(source_dir),
    }


def write_snapshot_metadata_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
