"""Validated records emitted by configs, launchers, and output adapters."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def new_id() -> str:
    """Return an opaque identifier suitable for any record primary key."""
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically for content hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def config_hash(kind: str, schema_version: int, payload: dict[str, Any]) -> str:
    """Hash behavior-bearing config content, excluding authorship metadata."""
    return sha256_text(
        canonical_json(
            {"kind": kind, "schema_version": schema_version, "payload": payload}
        )
    )


def sequence_hash(sequence: str) -> str:
    """Hash an uppercase, whitespace-free single-chain sequence."""
    return sha256_text(sequence)


class ConfigKind(StrEnum):
    GENERAL = "general"
    FILTERING = "filtering"
    MODEL = "model"
    OPTIMIZATION = "optimization"
    RESOLVED = "resolved"


class RunKind(StrEnum):
    GENERATE = "generate"
    EVALUATE = "evaluate"
    FILTER = "filter"
    CLUSTER = "cluster"
    OPTIMIZE = "optimize"
    RANK = "rank"
    IMPORT = "import"


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CandidateType(StrEnum):
    SEQUENCE = "sequence"
    BACKBONE = "backbone"
    COMPLEX = "complex"


class DesignStatus(StrEnum):
    PRODUCED = "produced"
    PARTIAL = "partial"
    INVALID = "invalid"


class MetricDirection(StrEnum):
    MIN = "min"
    MAX = "max"
    NONE = "none"


class MetricStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    MISSING = "missing"


class DecisionKind(StrEnum):
    FILTER = "filter"
    CLUSTER = "cluster"
    SELECTION = "selection"
    RANK = "rank"
    LABEL = "label"


class Record(BaseModel):
    """Strict base class shared by records written to DuckDB."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class ConfigRecord(Record):
    config_hash: str | None = None
    kind: ConfigKind
    name: str
    schema_version: int = Field(default=1, gt=0)
    payload: dict[str, Any]
    author: str | None = None
    rationale: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def fill_or_check_hash(self) -> Self:
        expected = config_hash(self.kind, self.schema_version, self.payload)
        if self.config_hash is None:
            object.__setattr__(self, "config_hash", expected)
        elif self.config_hash != expected:
            raise ValueError(f"config_hash does not match canonical payload: {expected}")
        return self


class RunRecord(Record):
    run_id: str = Field(default_factory=new_id)
    parent_run_id: str | None = None
    name: str
    tool: str
    kind: RunKind
    config_hash: str
    status: RunStatus = RunStatus.PLANNED
    n_requested: int | None = Field(default=None, ge=0)
    n_attempted: int | None = Field(default=None, ge=0)
    n_produced: int | None = Field(default=None, ge=0)
    n_passed: int | None = Field(default=None, ge=0)
    count_details: dict[str, Any] | None = None
    container_digest: str | None = None
    code_revision: str | None = None
    workflow_metadata: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    output_uri: str | None = None
    error: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def check_times(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot be earlier than started_at")
        return self


class DesignRecord(Record):
    design_id: str = Field(default_factory=new_id)
    run_id: str
    parent_design_id: str | None = None
    native_id: str
    candidate_type: CandidateType
    sequence: str | None = None
    sequence_hash: str | None = None
    length: int | None = Field(default=None, gt=0)
    seed: int | None = None
    status: DesignStatus = DesignStatus.PRODUCED
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r"\s+", "", value).upper()
        if not normalized or not re.fullmatch(r"[A-Z]+", normalized):
            raise ValueError("sequence must contain amino-acid letter codes only")
        return normalized

    @model_validator(mode="after")
    def fill_or_check_sequence_fields(self) -> Self:
        if self.sequence is None:
            if self.sequence_hash is not None or self.length is not None:
                raise ValueError("sequence_hash and length require sequence")
            return self

        expected_hash = sequence_hash(self.sequence)
        expected_length = len(self.sequence)
        if self.sequence_hash is not None and self.sequence_hash != expected_hash:
            raise ValueError(f"sequence_hash does not match sequence: {expected_hash}")
        if self.length is not None and self.length != expected_length:
            raise ValueError(f"length does not match sequence: {expected_length}")
        object.__setattr__(self, "sequence_hash", expected_hash)
        object.__setattr__(self, "length", expected_length)
        return self


class ArtifactRecord(Record):
    artifact_id: str = Field(default_factory=new_id)
    run_id: str
    design_id: str | None = None
    kind: str
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MetricRecord(Record):
    metric_id: str = Field(default_factory=new_id)
    run_id: str
    design_id: str
    name: str
    value: float | None = None
    unit: str | None = None
    direction: MetricDirection = MetricDirection.NONE
    replicate: int = Field(default=0, ge=0)
    status: MetricStatus = MetricStatus.OK
    details: dict[str, Any] | None = None
    measured_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_success_value(self) -> Self:
        if self.status == MetricStatus.OK and self.value is None:
            raise ValueError("a metric with status='ok' requires a numeric value")
        return self


class DecisionRecord(Record):
    decision_id: str = Field(default_factory=new_id)
    run_id: str
    design_id: str
    kind: DecisionKind
    name: str
    value: str | None = None
    passed: bool | None = None
    rank: int | None = Field(default=None, gt=0)
    scope_id: str | None = None
    reason: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_rank_scope(self) -> Self:
        if self.kind == DecisionKind.RANK and (self.rank is None or self.scope_id is None):
            raise ValueError("a rank decision requires both rank and scope_id")
        return self


class CollectedRun(Record):
    """Normalized output of one adapter before the serialized DB ingest job."""

    run: RunRecord
    designs: tuple[DesignRecord, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    metrics: tuple[MetricRecord, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()

    @model_validator(mode="after")
    def require_one_run_identity(self) -> Self:
        records = (*self.designs, *self.artifacts, *self.metrics, *self.decisions)
        mismatches = [record.run_id for record in records if record.run_id != self.run.run_id]
        if mismatches:
            raise ValueError(
                f"collected records must use run_id={self.run.run_id}; got {mismatches[0]}"
            )
        return self
