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


class ToolConfig(ConfigModel):
    """What every tool's model config must carry, whatever else it adds.

    Generic code (planning, the workflow, the database) only ever touches
    these. Everything tool-specific is reached through that tool's plugin.
    """

    schema_version: int = Field(gt=0)
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    resources: ResourceConfig


class MosaicConfig(ToolConfig):
    # v2 added runtime.exec_wrapper, runtime.scratch, and sampling.optimizer.
    # The first two are required, so a v1 document cannot be loaded by this
    # code. Bumping makes that say "schema_version: input should be 2" instead
    # of an unexplained missing field — which matters now that stored configs,
    # not files, are the source of truth.
    schema_version: Literal[2]
    tool: Literal["mosaic"]
    driver: MosaicDriverConfig
    sampling: MosaicSamplingConfig
    runtime: MosaicRuntimeConfig

    @model_validator(mode="after")
    def runtime_must_fit_walltime(self) -> Self:
        requested_seconds = self.sampling.max_runtime_hours * 3_600
        if requested_seconds >= self.resources.walltime_seconds:
            raise ValueError("max_runtime_hours must be shorter than resources.walltime")
        return self


class BoltzGenSpecConfig(ConfigModel):
    """BoltzGen is driven by a design spec, not by a script.

    The spec is authored (entities, length range, optional binding sites) and
    bound into the container. It is archived per run for the same reason
    Mosaic's driver is: it is consumed by the run, and BoltzGen's spec grammar
    is richer than anything worth re-deriving from this config.
    """

    template: Path


class BoltzGenSamplingConfig(ConfigModel):
    """Three counts, not one -- see docs/harness-design.md section 2.

    `num_designs` backbones are generated; `budget` is how many survive into
    final_ranked_designs; the tool's own filters then mark a subset as passing.
    Setting them equal with filtering off makes a run comparable to other
    tools' "N designs"; a production run should oversample instead.
    """

    jobs: int = Field(default=1, gt=0)
    num_designs: int = Field(gt=0)
    budget: int = Field(gt=0)
    protocol: str = Field(default="protein-anything", min_length=1)
    filter_biased: bool = False
    # Defaults to 1 below 100 designs and 10 above, which silently halves GPU
    # utilisation on a small run. Set it explicitly rather than inherit that.
    diffusion_batch_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def budget_cannot_exceed_generated(self) -> Self:
        if self.budget > self.num_designs:
            raise ValueError("budget cannot exceed num_designs")
        return self


class BoltzGenRuntimeConfig(ConfigModel):
    container: Path
    # Node-local scratch root. TMPDIR must NOT be on Lustre: Triton compiles
    # kernels in a temp dir whose cleanup fails with Errno 39 there, killing
    # the job on its first kernel. See docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class BoltzGenConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["boltzgen"]
    spec: BoltzGenSpecConfig
    sampling: BoltzGenSamplingConfig
    runtime: BoltzGenRuntimeConfig
