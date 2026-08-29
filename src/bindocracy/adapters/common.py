"""Collection helpers every tool's adapter needs identically.

Only the parts that are genuinely the same are here. Reading native rows,
mapping them to metrics, interpreting a tool's own status, and deciding what
kind of candidate was produced all stay with the tool, because that is where
tools actually differ.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from bindocracy.adapters.base import CollectionError
from bindocracy.runs.manifest import RunManifest
from bindocracy.store.records import ArtifactRecord, RunStatus, stable_id

MEDIA_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".json": "application/json",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".py": "text/x-python",
    ".yaml": "application/yaml",
    ".cif": "chemical/x-cif",
}

# Artifacts small enough to be worth a checksum. A design complex is kilobytes;
# a log can be hundreds of megabytes and is not the thing anyone re-verifies.
MAX_CHECKSUM_BYTES = 32 * 1024 * 1024


def read_manifest(run_dir: Path) -> RunManifest:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise CollectionError(f"missing run manifest: {manifest_path}")
    return RunManifest.read(manifest_path)


def artifact(
    run_dir: Path, run_id: str, relative: str, kind: str, *, design_id: str | None = None
) -> ArtifactRecord | None:
    """One artifact row, or None when the file was never written."""
    path = run_dir / relative
    if not path.is_file():
        return None
    stat = path.stat()
    return ArtifactRecord(
        artifact_id=stable_id("artifact", run_id, relative),
        run_id=run_id,
        design_id=design_id,
        kind=kind,
        uri=relative,
        media_type=MEDIA_TYPES.get(path.suffix),
        size_bytes=stat.st_size,
        sha256=_checksum(path, stat.st_size),
        created_at=datetime.fromtimestamp(stat.st_mtime, UTC),
    )


def provenance_artifacts(
    run_dir: Path, run_id: str, manifest: RunManifest, kinds: dict[str, str]
) -> list[ArtifactRecord]:
    """The archived inputs a run executed, as artifact rows."""
    return [
        record
        for label, archived in manifest.provenance.items()
        if (record := artifact(run_dir, run_id, archived.path, kinds.get(label, label)))
        is not None
    ]


def unique_by_uri(artifacts: list[ArtifactRecord]) -> list[ArtifactRecord]:
    by_uri: dict[str, ArtifactRecord] = {}
    for record in artifacts:
        by_uri.setdefault(record.uri, record)
    return list(by_uri.values())


def run_window(
    statuses: list,
) -> tuple[datetime | None, datetime | None]:
    """When the first task started and the last one finished."""
    starts = [s.started_at for s in statuses if s is not None and s.started_at]
    finishes = [s.finished_at for s in statuses if s is not None and s.finished_at]
    return (min(starts) if starts else None, max(finishes) if finishes else None)


def run_status(produced: int, statuses: list, complete: bool) -> RunStatus:
    """Nothing produced is a failure; anything incomplete is partial."""
    if produced == 0:
        return RunStatus.FAILED
    if complete and all(s is not None and s.status == "succeeded" for s in statuses):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _checksum(path: Path, size: int) -> str | None:
    if size > MAX_CHECKSUM_BYTES:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
