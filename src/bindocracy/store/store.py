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
        "sequence_hash",
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

    def add_configs(self, configs: Iterable[ConfigRecord]) -> tuple[str, ...]:
        """Insert config pairs atomically, skipping IDs already present.

        This method intentionally touches only ``configs``. Re-loading identical
        YAML is safe because IDs are derived from canonical JSON content.
        """
        records = list(configs)
        if not records:
            return ()

        inserted: list[str] = []
        con = self.connection
        con.execute("BEGIN TRANSACTION")
        try:
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
    ) -> None:
        """Atomically ingest one normalized adapter result and its configs."""
        con = self.connection
        con.execute("BEGIN TRANSACTION")
        try:
            self._insert("configs", configs)
            self._insert("runs", [collected.run])
            self._insert("designs", collected.designs)
            self._insert("artifacts", collected.artifacts)
            self._insert("metrics", collected.metrics)
            self._insert("decisions", collected.decisions)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

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
