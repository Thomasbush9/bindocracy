"""Versioned public benchmark inputs and organizer-owned attempt records.

These declarations describe policy; filesystem permissions and scheduler identities
must be provisioned by the organizer, not inferred from an agent-owned manifest.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from bindocracy.campaign.models import Site
from bindocracy.config.models import ConfigModel, TargetConfig


class AssetIdentity(ConfigModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["file", "directory"] = "file"


class EnvironmentManifest(ConfigModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    harness_revision: str = Field(min_length=1)
    harness_path: Path
    lock_path: Path
    harness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assets: dict[str, AssetIdentity] = Field(default_factory=dict)
    site: Site
    worker_partition: str | None = Field(default=None, min_length=1)
    bindings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_configs: dict[str, Path] = Field(default_factory=dict)
    qualification: dict[str, Any] = Field(default_factory=dict)


class BudgetPolicy(ConfigModel):
    mode: Literal["allocation", "compute"] = "allocation"
    walltime_seconds: int = Field(gt=0)
    max_gpus: int = Field(ge=0)
    max_cpus: int = Field(gt=0)
    max_memory_gb: int = Field(gt=0)
    gpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cpu_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def cumulative_limits(self) -> Self:
        if self.mode == "compute" and self.gpu_seconds is None and self.cpu_seconds is None:
            raise ValueError("compute policy requires gpu_seconds or cpu_seconds")
        return self


class SubmissionPolicy(ConfigModel):
    min_candidates: int = Field(default=1, ge=1)
    max_candidates: int = Field(default=64, ge=1)
    require_structures: bool = False
    distinct_sequences: bool = True

    @model_validator(mode="after")
    def valid_count(self) -> Self:
        if self.min_candidates > self.max_candidates:
            raise ValueError("min_candidates exceeds max_candidates")
        return self


class TaskSpec(ConfigModel):
    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    target: TargetConfig
    budget: BudgetPolicy
    submission: SubmissionPolicy = Field(default_factory=SubmissionPolicy)
    allowed_tools: tuple[str, ...] = Field(min_length=1)
    allowed_extensions: tuple[Literal["configuration", "orchestration", "scoring_function", "optimizer", "selection"], ...] = (
        "configuration", "orchestration", "scoring_function", "optimizer", "selection",
    )
    internet: Literal["none", "allowed"] = "none"
    failure_policy: Literal["no_compensation", "organizer_adjudicated"] = "no_compensation"


class WorkspaceManifest(ConfigModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    root: Path
    database: Path
    general_config: Path
    site_config: Path
    bindings: Path
    task: TaskSpec
    environment: EnvironmentManifest
    task_digest: str
    environment_digest: str
    input_assets: dict[str, AssetIdentity]
    created_at: datetime


class AttemptRecord(ConfigModel):
    schema_version: Literal[1] = 1
    attempt_id: str
    workspace: Path
    task_id: str
    task_digest: str
    environment_digest: str
    agent: dict[str, Any] = Field(default_factory=dict)
    command: tuple[str, ...] = ()
    state: Literal["prepared", "running", "completed", "timed_out", "budget_exhausted", "agent_failed", "infrastructure_failed", "cancelled"] = "prepared"
    created_at: datetime
    started_at: datetime | None = None
    deadline: datetime | None = None
    finished_at: datetime | None = None
    pid: int | None = None
    returncode: int | None = None
    latest_submission: dict[str, Any] | None = None
    jobs: tuple[dict[str, Any], ...] = ()
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    cleanup: dict[str, Any] = Field(default_factory=dict)
