"""Hash-based dataset exclusions for quarantined memorization examples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exclusion_manifest(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text())
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("exclusion manifest must contain an entries list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError("every exclusion entry must contain a sha256 string")
    return entries


def filter_excluded_paths(paths: list[Path], manifest: Path | None) -> list[Path]:
    """Filter exact paths and content hashes, hashing only matching basenames."""
    entries = load_exclusion_manifest(manifest)
    if not entries:
        return paths
    excluded_paths = {
        str(Path(entry["path"]).expanduser().resolve())
        for entry in entries
        if isinstance(entry.get("path"), str)
    }
    hashes_by_name: dict[str, set[str]] = {}
    for entry in entries:
        name = entry.get("name")
        if isinstance(name, str):
            hashes_by_name.setdefault(name, set()).add(entry["sha256"])
    result: list[Path] = []
    for path in paths:
        if str(path.resolve()) in excluded_paths:
            continue
        candidate_hashes = hashes_by_name.get(path.name)
        if candidate_hashes and sha256_file(path) in candidate_hashes:
            continue
        result.append(path)
    return result
