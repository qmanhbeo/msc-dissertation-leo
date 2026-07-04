"""
Fetch and extract the frozen dissertation data snapshot into ./data/.

This script underlies the main entrypoint snapshot fetch path:
- `python main.py --fetch-data-snapshot curated`
- `python main.py --fetch-data-snapshot full`

It is intentionally separate from the operator backup script:
- `backup_data_snapshot.py` creates and uploads archives.
- `fetch_data_snapshot.py` downloads one frozen release snapshot and extracts it.

Usage:
    python main.py --fetch-data-snapshot curated

Debugging / direct script usage:
    python 1_code/data_backup_and_fetch/fetch_data_snapshot.py --profile curated

For local pre-release testing with a one-off archive:
    python 1_code/data_backup_and_fetch/fetch_data_snapshot.py \\
        --profile curated \\
        --url file:///tmp/dissertation-data-snapshot-curated.tar.zst \\
        --sha256 <digest>

Reproducibility notes:
- Warm replay from the frozen snapshot is the primary reproducibility target.
- A live-source full-pipeline rerun is not guaranteed to match the frozen snapshot.
- The manual policy supplement includes documents that were curated manually; the
  snapshot metadata preserves the document inventory but not stable source URLs.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import hashlib
import json
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urljoin

import zstandard as zstd

from data_snapshot_profiles import SNAPSHOT_METADATA_FILE, get_snapshot_profile


ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = Path(__file__).resolve().with_name("data_snapshot_manifest.json")


def log(msg: str) -> None:
    print(f"[fetch-data-snapshot] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and extract the frozen dissertation data snapshot.")
    p.add_argument("--profile", choices=["curated", "full"], default="curated")
    p.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH, help=argparse.SUPPRESS)
    p.add_argument("--url", default="", help="Optional one-off URL override for pre-release testing.")
    p.add_argument("--sha256", default="", help="Optional checksum override, usually paired with --url.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing ./data/ tree before extraction.")
    return p.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"snapshot manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_profile_entry(manifest: dict, profile_name: str) -> dict:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise RuntimeError(f"snapshot manifest does not define profile '{profile_name}'")
    entry = profiles[profile_name]
    if not isinstance(entry, dict):
        raise RuntimeError(f"snapshot manifest entry for profile '{profile_name}' is malformed")
    return entry


def ensure_snapshot_available(url: str, sha256_hex: str, profile_name: str) -> None:
    if not url or not sha256_hex:
        raise RuntimeError(
            "The committed data snapshot manifest is still unpublished for profile "
            f"'{profile_name}'. Populate 1_code/data_backup_and_fetch/data_snapshot_manifest.json with the final HTTPS "
            "URL and SHA256, or pass --url and --sha256 for one-off local testing."
        )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_to_file(response: urllib.request.addinfourl, out_path: Path, *, total: int | None = None) -> None:
    """Stream an HTTP response to disk with a tqdm progress bar (1 MB chunks)."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) if tqdm else None
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            if pbar is not None:
                pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()
    log(f"downloaded {out_path.stat().st_size / (1024**3):.2f} GB → {out_path.name}")


def build_url_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def is_google_drive_warning_page(content_type: str, body: bytes) -> bool:
    if "text/html" not in content_type.lower():
        return False
    snippet = body[:4096].decode("utf-8", errors="replace")
    return "Google Drive - Virus scan warning" in snippet and "download-form" in snippet


def extract_google_drive_confirm_url(base_url: str, html: str) -> str:
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not form_match:
        raise RuntimeError("Google Drive warning page did not expose a download form.")
    action_url = urljoin(base_url, form_match.group(1))
    hidden_inputs = dict(re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
    required_fields = {"id", "export", "confirm"}
    if not required_fields.issubset(hidden_inputs):
        raise RuntimeError("Google Drive warning page is missing required confirm fields.")
    return f"{action_url}?{urlencode(hidden_inputs)}"


def describe_html_download_failure(body: bytes) -> str:
    text = body[:4096].decode("utf-8", errors="replace")
    title_match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()
    return text[:160].strip() or "unknown HTML response"


def download_snapshot(url: str, out_path: Path) -> None:
    opener = build_url_opener()
    log(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "dissertation-snapshot-fetch/1.0"})
    with opener.open(request) as response:
        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length else None
        if total:
            log(f"archive size: {total / (1024**3):.2f} GB")

        # Google Drive virus-scan warning — read a small head to extract confirm URL
        if "text/html" in content_type.lower():
            head = response.read(8192)
            body = head + response.read()
            if is_google_drive_warning_page(content_type, body):
                warning_html = body.decode("utf-8", errors="replace")
                confirm_url = extract_google_drive_confirm_url(url, warning_html)
                log("Google Drive returned a virus-scan warning page; retrying via confirmed download URL")
                confirm_request = urllib.request.Request(
                    confirm_url, headers={"User-Agent": "dissertation-snapshot-fetch/1.0"}
                )
                with opener.open(confirm_request) as confirm_resp:
                    stream_to_file(confirm_resp, out_path)
                return
            failure = describe_html_download_failure(body)
            raise RuntimeError(f"snapshot URL returned HTML instead of an archive: {failure}")

        # Actual archive download — stream with progress
        stream_to_file(response, out_path, total=total)


def safe_extract_tar_zst(archive_path: Path, dest_root: Path) -> None:
    log(f"extracting {archive_path.name} into {dest_root}")
    count = 0
    with archive_path.open("rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    count += 1
                    if count == 1 or count % 2000 == 0:
                        log(f"  extracting entry {count}: {member.name}")
                    if member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise RuntimeError(f"unsafe archive member path: {member.name}")
                    target = (dest_root / member.name).resolve()
                    if ROOT not in target.parents and target != ROOT:
                        raise RuntimeError(f"archive member escapes repo root: {member.name}")
                    tf.extract(member, path=dest_root, set_attrs=True)
    log(f"extraction complete ({count} entries)")


def validate_extraction(profile_name: str, expected_repo_paths: list[str]) -> None:
    missing = [Path(path) for path in expected_repo_paths if not (ROOT / path).exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing[:8])
        raise RuntimeError(f"snapshot extraction is incomplete for profile '{profile_name}'. Missing: {missing_str}")

    snapshot_meta_path = ROOT / "2_data" / SNAPSHOT_METADATA_FILE
    if not snapshot_meta_path.exists():
        raise RuntimeError(f"snapshot extraction is missing internal metadata: {snapshot_meta_path}")
    payload = json.loads(snapshot_meta_path.read_text(encoding="utf-8"))
    if payload.get("profile") != profile_name:
        raise RuntimeError(
            f"snapshot profile mismatch after extraction: expected '{profile_name}', found '{payload.get('profile')}'"
        )


def main() -> None:
    args = parse_args()
    profile = get_snapshot_profile(args.profile)
    manifest = load_manifest(args.manifest_path)
    entry = resolve_profile_entry(manifest, profile.name)

    url = args.url or str(entry.get("url") or "").strip()
    sha256_hex = args.sha256 or str(entry.get("sha256") or "").strip()
    expected_repo_paths = entry.get("expected_repo_paths") or [str(path) for path in profile.expected_repo_paths]
    if not isinstance(expected_repo_paths, list) or not all(isinstance(item, str) for item in expected_repo_paths):
        raise RuntimeError(f"snapshot manifest entry for profile '{profile.name}' has malformed expected_repo_paths")

    ensure_snapshot_available(url, sha256_hex, profile.name)

    archive_dir = ROOT / "data_archive_fetch"
    archive_path = archive_dir / f"{profile.name}.tar.zst"

    # Cache hit: skip download if archive exists and SHA matches
    if archive_path.exists():
        cached_sha = compute_sha256(archive_path)
        if cached_sha.lower() == sha256_hex.lower():
            log(f"cached archive {archive_path.name} matches sha256 — skipping download")
        else:
            log(f"cached archive {archive_path.name} has wrong sha256 ({cached_sha[:16]}...) — re-downloading")
            archive_path.unlink()
            archive_dir.mkdir(parents=True, exist_ok=True)
            download_snapshot(url, archive_path)
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)
        download_snapshot(url, archive_path)

    observed_sha = compute_sha256(archive_path)
    if observed_sha.lower() != sha256_hex.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {archive_path.name}: expected {sha256_hex}, observed {observed_sha}"
        )
    log(f"verified sha256 {observed_sha}")

    data_root = ROOT / "2_data"
    if data_root.exists():
        if not args.overwrite:
            raise RuntimeError("2_data/ already exists. Re-run with --overwrite to replace it.")
        log(f"removing existing {data_root}")
        shutil.rmtree(data_root)

    safe_extract_tar_zst(archive_path, ROOT)
    validate_extraction(profile.name, expected_repo_paths)
    log(f"snapshot ready for profile '{profile.name}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
