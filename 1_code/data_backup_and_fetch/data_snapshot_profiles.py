from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SNAPSHOT_METADATA_DIR = Path("_snapshot_metadata")
SNAPSHOT_METADATA_FILE = SNAPSHOT_METADATA_DIR / "snapshot_manifest.json"

RAW_ONLY_PATHS = (
    Path("0_raw"),
)

EMBEDDED_ONLY_PATHS = (
    Path("3_embedded"),
    Path("3a_warm_replay_texts"),
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
    included_data_paths: tuple[Path, ...]
    expected_repo_paths: tuple[Path, ...]


SNAPSHOT_PROFILES: dict[str, SnapshotProfile] = {
    "raw": SnapshotProfile(
        name="raw",
        description="Raw fetched data only (0_raw/). For cold-replay rebuilds.",
        excluded_data_paths=(),
        included_data_paths=(Path("0_raw"),),
        expected_repo_paths=RAW_ONLY_PATHS,
    ),
    "embedded": SnapshotProfile(
        name="embedded",
        description=(
            "Embedded checkpoint (3_embedded/) plus gzipped warm-replay appendix text "
            "(3a_warm_replay_texts/: research shards + policy.jsonl for the default model). "
            "For warm-replay analysis."
        ),
        excluded_data_paths=(),
        included_data_paths=EMBEDDED_ONLY_PATHS,
        expected_repo_paths=EMBEDDED_ONLY_PATHS,
    ),
}


def get_snapshot_profile(name: str) -> SnapshotProfile:
    try:
        return SNAPSHOT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown snapshot profile: {name}") from exc


def snapshot_archive_prefix(profile_name: str) -> str:
    return f"dissertation-data-snapshot-{profile_name}"


def should_exclude_data_path(rel_data_path: Path, profile: SnapshotProfile) -> bool:
    rel = Path(rel_data_path)
    if profile.included_data_paths:
        for included in profile.included_data_paths:
            if rel == included or included in rel.parents:
                return False
        return True
    for excluded in profile.excluded_data_paths:
        if rel == excluded or excluded in rel.parents:
            return True
    return False


def manual_policy_inventory(source_dir: Path) -> list[str]:
    manual_root = source_dir / "2_data" / "0_raw" / "policy_manual"
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
