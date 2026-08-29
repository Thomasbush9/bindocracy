"""Pydantic models for author-facing campaign configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLURM_WALLTIME = re.compile(r"(?:(?P<days>\d+)-)?(?P<hours>\d{1,3}):(?P<minutes>\d{2}):(?P<seconds>\d{2})")


def slurm_walltime_seconds(value: str) -> int:
    """Convert ``[days-]HH:MM:SS`` to seconds, raising on invalid input."""
    match = _SLURM_WALLTIME.fullmatch(value)
    if match is None:
        raise ValueError("walltime must use [days-]HH:MM:SS")

    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("walltime contains invalid minutes or seconds")
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


class ConfigModel(BaseModel):
    """Strict, immutable base for authored configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class CampaignConfig(ConfigModel):
    name: str = Field(min_length=1)


class TargetConfig(ConfigModel):
    name: str = Field(min_length=1)
    sequence_fasta: Path
    msa: Path
    chain_id: str = Field(pattern=r"^[A-Za-z0-9]$")
    hotspots: tuple[str, ...] = ()
    # Optional here because the target is shared and not every tool needs it:
    # Mosaic folds the target from sequence, BoltzGen requires geometry. Each
    # tool's preflight asserts the representation it actually consumes.
    structure_cif: Path | None = None


class ClusterConfig(ConfigModel):
    executor: Literal["slurm"]
    account: str = Field(min_length=1)
    default_partition: str = Field(min_length=1)
    max_concurrent_jobs: int = Field(gt=0)


class GeneralConfig(ConfigModel):
    schema_version: Literal[1]
    campaign: CampaignConfig
    target: TargetConfig
    cluster: ClusterConfig


class ResourceConfig(ConfigModel):
    gpus: int = Field(gt=0)
    cpus: int = Field(gt=0)
    memory_gb: int = Field(gt=0)
    walltime: str

    @field_validator("walltime")
    @classmethod
    def validate_walltime(cls, value: str) -> str:
        slurm_walltime_seconds(value)
        return value

    @property
    def walltime_seconds(self) -> int:
        return slurm_walltime_seconds(self.walltime)


class ToolConfig(ConfigModel):
    """What every tool's model config must carry, whatever else it adds.

    Generic code (planning, the workflow, the database) only ever touches
    these. Everything tool-specific is reached through that tool's plugin.
    """

    schema_version: int = Field(gt=0)
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    resources: ResourceConfig
