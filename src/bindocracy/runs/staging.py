"""The staging bundle between collection and ingestion.

Collection writes a validated `CollectedRun` to `collected.json`; ingestion
reads it back, validates it again, and only then opens DuckDB.

A bundle is immutable and long-lived: the database is meant to be rebuildable
from run directories and their bundles, so a bundle written months ago has to
stay readable after the record classes change. Two rules make that work.

*Reading migrates.* Fields the records no longer carry are dropped on the way
in, so a bundle written before `designs.sequence_hash` was removed still loads.

*The identity of a bundle is its file, not its parsed form.* Ingestion is
idempotent on a content hash, and that hash is computed from the JSON as
written. Otherwise removing a field would silently change every old bundle's
identity, and re-ingesting one would look like a conflicting rewrite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from bindocracy.store.records import CollectedRun, canonical_json, sha256_text

# Fields dropped from a record class, by the table they belonged to. A bundle
# written before the removal still carries them; they are ignored on read.
REMOVED_FIELDS: dict[str, set[str]] = {
    "designs": {"sequence_hash"},  # dropped in schema v3
}


def write_collected(collected: CollectedRun, path: str | Path) -> Path:
    """Serialize one bundle atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(collected.model_dump_json(indent=2) + "\n")
    os.replace(tmp, destination)
    return destination


def read_collected(path: str | Path) -> CollectedRun:
    """Load a bundle, migrating anything the records no longer carry."""
    return CollectedRun.model_validate(migrate(_raw(path)))


def staged_digest(path: str | Path) -> str:
    """The bundle's identity, from the file as written.

    Ingestion compares this against what it stored, so a record class losing a
    field does not make every existing bundle look like a different one.
    """
    return sha256_text(canonical_json(_raw(path)))


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop fields the record classes have since removed."""
    migrated = dict(raw)
    for table, removed in REMOVED_FIELDS.items():
        rows = migrated.get(table)
        if not isinstance(rows, list):
            continue
        migrated[table] = [
            {key: value for key, value in row.items() if key not in removed}
            if isinstance(row, dict) else row
            for row in rows
        ]
    return migrated


def _raw(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
