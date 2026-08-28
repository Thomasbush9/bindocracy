"""The staging bundle between collection and ingestion.

Collection writes a validated `CollectedRun` to `collected.json`; ingestion
reads it back, validates it again, and only then opens DuckDB.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.store.records import CollectedRun


def write_collected(collected: CollectedRun, path: str | Path) -> Path:
    """Serialize one bundle atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(collected.model_dump_json(indent=2) + "\n")
    os.replace(tmp, destination)
    return destination


def read_collected(path: str | Path) -> CollectedRun:
    return CollectedRun.model_validate_json(Path(path).read_text())
