"""A bundle written months ago must still load.

The database is meant to be rebuildable from run directories and their staging
bundles. That promise quietly broke when schema v3 removed
`designs.sequence_hash`: every existing bundle failed validation, because the
loader validated the file directly against the current record classes.

The fixture here is a real bundle from `runs/first_run`, written under v2,
trimmed to two designs. It is committed permanently -- its whole purpose is to
be old.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bindocracy.runs.staging import migrate, read_collected, staged_digest, write_collected

V2_BUNDLE = Path(__file__).resolve().parent / "fixtures" / "staging" / "v2_bundle.collected.json"


def test_the_fixture_really_is_old() -> None:
    """Guards the guard: if this stops being a v2 bundle it proves nothing."""
    raw = json.loads(V2_BUNDLE.read_text())

    assert "sequence_hash" in raw["designs"][0]


def test_a_v2_bundle_still_loads() -> None:
    collected = read_collected(V2_BUNDLE)

    assert collected.run.tool == "mosaic"
    assert len(collected.designs) == 2
    assert collected.designs[0].sequence
    assert collected.designs[0].length == len(collected.designs[0].sequence)


def test_a_v2_bundle_keeps_the_identity_it_was_ingested_under() -> None:
    """Ingestion is idempotent on a content hash.

    If removing a field changed every old bundle's hash, re-ingesting one would
    look like a conflicting rewrite of a run that had not changed at all.
    """
    raw = json.loads(V2_BUNDLE.read_text())
    from bindocracy.store.records import canonical_json, sha256_text

    assert staged_digest(V2_BUNDLE) == sha256_text(canonical_json(raw))


def test_migration_only_drops_what_the_records_no_longer_carry() -> None:
    raw = json.loads(V2_BUNDLE.read_text())

    migrated = migrate(raw)

    assert "sequence_hash" not in migrated["designs"][0]
    assert migrated["designs"][0]["sequence"] == raw["designs"][0]["sequence"]
    assert migrated["run"] == raw["run"]
    assert len(migrated["metrics"]) == len(raw["metrics"])


def test_a_bundle_written_now_round_trips(tmp_path: Path) -> None:
    collected = read_collected(V2_BUNDLE)
    path = write_collected(collected, tmp_path / "collected.json")

    assert read_collected(path) == collected
    # A freshly written bundle's file identity is its content hash.
    assert staged_digest(path) == collected.content_hash()


def test_an_unreadable_bundle_still_fails(tmp_path: Path) -> None:
    """Migration is not permission to accept nonsense."""
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"run": {"name": "x"}}))

    with pytest.raises(ValueError):
        read_collected(path)
