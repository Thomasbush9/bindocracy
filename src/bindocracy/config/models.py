"""Pydantic models for author-facing campaign configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class MosaicDriverConfig(ConfigModel):
    script: Path
    archive: bool = True


class MosaicOptimizerConfig(ConfigModel):
    """APGM schedule. The defaults are the lab's benchmarked settings.

    Three stages: a soft optimization over the PSSM, then two sharpening
    passes towards a discrete sequence. Shortening them is how a test run
    stays cheap; it is also the only real lever on Mosaic's per-design cost,
    which is ~7 min at these defaults.
    """

    soft_steps: int = Field(default=100, gt=0)
    sharpen_steps: int = Field(default=50, gt=0)
    final_steps: int = Field(default=15, gt=0)


class MosaicSamplingConfig(ConfigModel):
    binder_length: int = Field(gt=0)
    jobs: int = Field(gt=0)
    designs_per_job: int = Field(gt=0)
    max_runtime_hours: float = Field(gt=0)
    seed_base: int = Field(ge=0)
    optimizer: MosaicOptimizerConfig = MosaicOptimizerConfig()


class MosaicRuntimeConfig(ConfigModel):
    container: Path
    weights: Path
    # mosaic-exec.sh is the only supported entry point: it builds the private
    # container HOME and binds the external weight caches onto the paths mosaic
    # looks for. See docs/containers/mosaic.md.
    exec_wrapper: Path
    scratch: Path


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


class MosaicConfig(ConfigModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    tool: Literal["mosaic"]
    driver: MosaicDriverConfig
    sampling: MosaicSamplingConfig
    runtime: MosaicRuntimeConfig
    resources: ResourceConfig

    @model_validator(mode="after")
    def runtime_must_fit_walltime(self) -> Self:
        requested_seconds = self.sampling.max_runtime_hours * 3_600
        if requested_seconds >= self.resources.walltime_seconds:
            raise ValueError("max_runtime_hours must be shorter than resources.walltime")
        return self
