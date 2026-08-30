"""The staging bundle between collection and ingestion.

Collection writes a validated `CollectedRun` to `collected.json`; ingestion
reads it back, validates it again, and only then opens DuckDB.

A bundle is immutable and long-lived: the database is meant to be rebuildable
from run directories and their bundles, so a bundle written months ago has to
stay readable after the record classes change. Three rules make that work.

*A bundle says what it is.* It carries a `bundle_version` beside its payload,
and reading it runs the migrations between that version and this one. The first
format had no envelope at all, which is version 0 and still loads.

*Reading migrates; it never guesses.* A bundle from a **newer** version than
this code knows is refused rather than half-read. Dropping fields we do not
recognise is only safe going backwards -- forwards, an unknown field is
something this code was supposed to understand and does not.

*The identity of a bundle is its payload, not its envelope.* Ingestion is
idempotent on a content hash, computed from the payload exactly as written.
Otherwise removing a field -- or merely bumping the version -- would silently
change every old bundle's identity, and re-ingesting one would look like a
conflicting rewrite of a run that had not changed at all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bindocracy.store.records import CollectedRun, canonical_json, sha256_text

# The format this code writes. Bump it in the same commit as the migration
# that reads the version below it.
BUNDLE_VERSION = 1
# The original format: a bare CollectedRun, with no envelope to version it.
LEGACY_VERSION = 0

VERSION_KEY = "bundle_version"
PAYLOAD_KEY = "collected"

# Fields dropped from a record class, by the table they belonged to. A version
# 0 bundle predates the removal and still carries them.
REMOVED_FIELDS: dict[str, set[str]] = {
    "designs": {"sequence_hash"},  # dropped in schema v3
}


class StagingFormatError(ValueError):
    """A bundle cannot be read as the staging format this code knows."""


def write_collected(collected: CollectedRun, path: str | Path) -> Path:
    """Serialize one bundle atomically, stamped with the format it is in."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        VERSION_KEY: BUNDLE_VERSION,
        PAYLOAD_KEY: collected.model_dump(mode="json"),
    }
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(envelope, indent=2) + "\n")
    os.replace(tmp, destination)
    return destination


def read_collected(path: str | Path) -> CollectedRun:
    """Load a bundle, migrating it from whatever version wrote it."""
    return CollectedRun.model_validate(migrate(_raw(path)))


def staged_digest(path: str | Path) -> str:
    """The bundle's identity: its payload, as written, before any migration.

    Ingestion compares this against what it stored, so neither a record class
    losing a field nor a new envelope around the same records makes an existing
    bundle look like a different one.
    """
    payload, _ = _unwrap(_raw(path))
    return sha256_text(canonical_json(payload))


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring one bundle's payload up to `BUNDLE_VERSION`."""
    payload, version = _unwrap(raw)
    if version > BUNDLE_VERSION:
        raise StagingFormatError(
            f"this bundle is version {version}, and this code writes version "
            f"{BUNDLE_VERSION}. It was written by a newer bindocracy; reading it "
            "here would silently discard whatever that version added."
        )
    while version < BUNDLE_VERSION:
        try:
            step = _MIGRATIONS[version]
        except KeyError as error:
            raise StagingFormatError(
                f"no migration from staging version {version} to {version + 1}"
            ) from error
        payload = step(payload)
        version += 1
    return payload


def _unwrap(raw: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """The payload and the version that wrote it.

    A bundle with no envelope is version 0 -- the format that existed before
    bundles said which format they were in.
    """
    if not isinstance(raw, dict):
        raise StagingFormatError("a staging bundle must be a JSON object")
    if VERSION_KEY not in raw:
        return raw, LEGACY_VERSION

    version, payload = raw.get(VERSION_KEY), raw.get(PAYLOAD_KEY)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise StagingFormatError(f"{VERSION_KEY} must be a non-negative integer")
    if not isinstance(payload, dict):
        raise StagingFormatError(f"a versioned bundle must carry {PAYLOAD_KEY!r}")
    return payload, version


def _drop_removed_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Version 0 to 1: drop fields the record classes no longer carry."""
    migrated = dict(payload)
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


# One entry per version, taking a payload at that version to the next one.
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    LEGACY_VERSION: _drop_removed_fields,
}


def _raw(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
