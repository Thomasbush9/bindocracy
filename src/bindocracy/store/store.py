"""Single-writer interface to a campaign DuckDB database."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb
from pydantic import BaseModel

from bindocracy.store import schema
from bindocracy.store.records import CollectedRun, ConfigRecord

_COLUMNS: dict[str, tuple[str, ...]] = {
    "configs": (
        "model_config_id",
        "general_config_id",
        "general_name",
        "general_schema_version",
        "general_config_json",
        "general_config_hash",
        "general_source_uri",
        "general_source_sha256",
        "model_name",
        "tool",
        "model_schema_version",
        "model_config_json",
        "model_config_hash",
        "model_source_uri",
        "model_source_sha256",
        "author",
        "rationale",
        "created_at",
    ),
    "runs": (
        "run_id",
        "parent_run_id",
        "name",
        "tool",
        "kind",
        "model_config_id",
        "status",
        "n_requested",
        "n_attempted",
        "n_produced",
        "n_passed",
        "count_details",
        "container_digest",
        "code_revision",
        "workflow_metadata",
        "resources",
        "output_uri",
        "error",
        "created_at",
        "started_at",
        "finished_at",
    ),
    "designs": (
        "design_id",
        "run_id",
        "parent_design_id",
        "native_id",
        "candidate_type",
        "sequence",
        "length",
        "seed",
        "status",
        "metadata",
        "created_at",
    ),
    "artifacts": (
        "artifact_id",
        "run_id",
        "design_id",
        "kind",
        "uri",
        "media_type",
        "sha256",
        "size_bytes",
        "metadata",
        "created_at",
    ),
    "metrics": (
        "metric_id",
        "run_id",
        "design_id",
        "name",
        "value",
        "unit",
        "direction",
        "replicate",
        "status",
        "details",
        "measured_at",
    ),
    "decisions": (
        "decision_id",
        "run_id",
        "design_id",
        "kind",
        "name",
        "value",
        "passed",
        "rank",
        "scope_id",
        "reason",
        "created_at",
    ),
}

_JSON_COLUMNS = {
    "general_config_json",
    "model_config_json",
    "count_details",
    "workflow_metadata",
    "resources",
    "error",
    "metadata",
    "details",
    "reason",
}


class IngestConflictError(RuntimeError):
    """A different bundle has already been ingested under this run ID."""


class TargetMismatchError(RuntimeError):
    """This database follows a different biological target."""


class CampaignStore:
    """Own one DuckDB connection; only the serialized ingest job should write."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.connection = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            schema.apply(self.connection)

    @classmethod
    def create(cls, path: str | Path, *, exist_ok: bool = False) -> Self:
        """Create an initialized empty database without overwriting an existing file."""
        database_path = Path(path)
        if database_path.exists() and not exist_ok:
            raise FileExistsError(f"database already exists: {database_path}")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(database_path)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def assert_target(self, name: str, sequence_sha256: str) -> bool:
        """Stamp this database's target, or refuse a run against another one.

        One DuckDB file follows one target -- that assumption is everywhere,
        from `design_history` to any cross-tool comparison. Nothing enforced
        it, so a general config naming a different protein would have loaded
        happily and mixed two campaigns into one table. The identity is the
        resolved sequence, not the path it came from, so replacing a FASTA
        in place is caught too.

        Returns True when this call stamped the database.
        """
        stored = self.connection.execute(
            "SELECT value FROM _meta WHERE key = 'target_sha256'"
        ).fetchone()
        if stored is None:
            schema._set_meta(self.connection, "target_sha256", sequence_sha256)
            schema._set_meta(self.connection, "target_name", name)
            return True
        if stored[0] != sequence_sha256:
            existing = self.connection.execute(
                "SELECT value FROM _meta WHERE key = 'target_name'"
            ).fetchone()
            raise TargetMismatchError(
                f"this database follows target {existing[0] if existing else '?'} "
                f"({stored[0]}), not {name} ({sequence_sha256}). "
                "One campaign database is one target; use a different database."
            )
        return False

    def add_configs(
        self,
        configs: Iterable[ConfigRecord],
        *,
        target: tuple[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Insert config pairs atomically, skipping IDs already present.

        This method intentionally touches only ``configs``. Re-loading identical
        YAML is safe because IDs are derived from canonical JSON content.

        `target` is checked in the same transaction. Configs used to be able to
        enter a campaign before any run did, so a pair naming a different
        protein could sit in the table indefinitely without anything objecting.
        """
        records = list(configs)
        if not records:
            return ()

        inserted: list[str] = []
        con = self.connection
        con.execute("BEGIN TRANSACTION")
        try:
            if target is not None:
                self.assert_target(*target)
            for record in records:
                if record.model_config_id is None:  # guarded by ConfigRecord validation
                    raise ValueError("ConfigRecord has no model_config_id")
                exists = con.execute(
                    "SELECT 1 FROM configs WHERE model_config_id = ?",
                    [record.model_config_id],
                ).fetchone()
                if exists is not None:
                    continue
                self._insert("configs", [record])
                inserted.append(record.model_config_id)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return tuple(inserted)

    def ingest(
        self,
        collected: CollectedRun,
        *,
        configs: Iterable[ConfigRecord] = (),
        digest: str | None = None,
        target: tuple[str, str] | None = None,
    ) -> bool:
        """Atomically ingest one normalized adapter result and its configs.

        Restart-safe, because Snakemake may rerun the ingestion rule: an
        already-ingested identical bundle is a no-op returning ``False``, and a
        changed bundle for a known run raises before writing anything. Neither
        path can ever create a second copy of the run under new IDs.
        """
        digest = digest if digest is not None else collected.content_hash()
        stored = self.connection.execute(
            "SELECT workflow_metadata->>'bundle_sha256' FROM runs WHERE run_id = ?",
            [collected.run.run_id],
        ).fetchone()
        if stored is not None:
            if stored[0] == digest:
                return False
            raise IngestConflictError(
                f"run {collected.run.run_id} is already ingested with different "
                "content; collect into a new run directory instead"
            )

        run = collected.run.model_copy(update={
            "workflow_metadata": {
                **(collected.run.workflow_metadata or {}),
                "bundle_sha256": digest,
            }
        })
        new_configs = [config for config in configs if not self._config_exists(config)]

        con = self.connection
        con.execute("BEGIN TRANSACTION")
        try:
            # Inside the transaction: a failed ingestion must not leave the
            # database stamped with a target whose run never landed.
            if target is not None:
                self.assert_target(*target)
            self._insert("configs", new_configs)
            self._insert("runs", [run])
            self._insert("designs", collected.designs)
            self._insert("artifacts", collected.artifacts)
            self._insert("metrics", collected.metrics)
            self._insert("decisions", collected.decisions)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return True

    def _config_exists(self, record: ConfigRecord) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM configs WHERE model_config_id = ?", [record.model_config_id]
        ).fetchone() is not None

    def _insert(self, table: str, records: Iterable[BaseModel]) -> None:
        rows = list(records)
        if not rows:
            return
        columns = _COLUMNS[table]
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        values = [_record_values(record, columns) for record in rows]
        self.connection.executemany(
            f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})", values
        )


def create_database(path: str | Path, *, exist_ok: bool = False) -> Path:
    """Create and close an empty campaign database, returning its path."""
    database_path = Path(path)
    with CampaignStore.create(database_path, exist_ok=exist_ok):
        pass
    return database_path


def _record_values(record: BaseModel, columns: tuple[str, ...]) -> list[Any]:
    data = record.model_dump(mode="python")
    values: list[Any] = []
    for column in columns:
        value = data[column]
        if column in _JSON_COLUMNS and value is not None:
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        values.append(value)
    return values
