"""Mosaic's authored configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig


class MosaicDriverConfig(ConfigModel):
    # Always archived and executed from the archive; there is no mode in which
    # a run executes the live working tree, so there is no flag for one.
    script: Path


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
