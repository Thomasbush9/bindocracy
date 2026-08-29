"""BoltzGen's authored configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig


class BoltzGenSpecConfig(ConfigModel):
    """BoltzGen is driven by a design spec, not by a script.

    The spec is authored (entities, length range, optional binding sites) and
    bound into the container. It is archived per run for the same reason
    Mosaic's driver is: it is consumed by the run, and BoltzGen's spec grammar
    is richer than anything worth re-deriving from this config.
    """

    template: Path
    # Filled in from `template` when the config is loaded, so the stored JSON
    # holds the specification itself rather than a path to it. Authors leave
    # it out; a recovered config carries it and still validates.
    contents: dict | None = None


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
