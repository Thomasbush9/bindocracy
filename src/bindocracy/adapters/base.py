"""Contract implemented by every tool-specific output adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bindocracy.store.records import CollectedRun, RunRecord


class CollectionError(RuntimeError):
    """Raised when a run directory is incomplete, malformed, or inconsistent."""


class OutputAdapter(ABC):
    """Normalize one tool's run directory without writing to the database.

    Adapters are pure collectors: Snakemake runs them after the container job,
    writes their normalized result to a staging bundle, and a serialized ingest
    rule later writes that bundle to DuckDB.
    """

    tool: str

    @abstractmethod
    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        """Parse one completed run directory into normalized records."""

    @abstractmethod
    def succeeded(self, run_dir: Path) -> bool:
        """Return whether expected artifacts exist and parse successfully."""
