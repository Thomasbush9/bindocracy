"""Genie 3's authored configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, ToolConfig


class Genie3ExperimentConfig(ConfigModel):
    """Genie 3 is driven by its own experiment YAML, not by a script.

    The template holds the science: which problem set, how the interface is
    conditioned, how many sequences per backbone, and how the folding stage is
    configured. The harness owns exactly three keys in it -- the output root,
    the seed, and the sample count -- because those are per task, and preflight
    refuses a template that also sets them.
    """

    template: Path
    # Filled in from `template` when the config is loaded, so the stored JSON
    # holds the experiment itself rather than a path to it. Authors leave it
    # out; a recovered config carries it and still validates.
    contents: dict | None = None


class Genie3DriverConfig(ConfigModel):
    # Archived and executed from the archive, like Mosaic's. It renders one
    # task's experiment config from the archived template and runs `genie3 run`.
    script: Path


class Genie3SamplingConfig(ConfigModel):
    """Backbones per task, not designs per task -- they are different numbers.

    Genie 3 diffuses `backbones_per_job` backbones, and ProteinMPNN then writes
    `evaluation.inverse_folding.num_seq` sequences for each one. The designs a
    task produces is the product, which is what the run asks for; see
    docs/known-issues.md section 3.
    """

    jobs: int = Field(default=1, gt=0)
    backbones_per_job: int = Field(gt=0)
    # Each task samples with seed_base + task_id. Two tasks sharing a seed
    # would diffuse the same backbones and the run would silently return
    # duplicates of half its designs.
    seed_base: int = Field(ge=0)


class Genie3RuntimeConfig(ConfigModel):
    container: Path
    # genie3.sif ships jax/jaxlib 0.6.2 with no CUDA plugin, so without these
    # three the AF2 evaluation stage runs on the CPU and never finishes, with
    # no error at all. Preflight asserts each one, and they are recorded per
    # run because the image alone no longer determines the result.
    # See docs/known-issues.md section 2.3.
    jax_plugin_overlay: Path
    cudnn_overlay: Path
    cuda_nvcc_overlay: Path
    # Node-local scratch root. TMPDIR must NOT be on Lustre; see
    # docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class Genie3Config(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["genie3"]
    experiment: Genie3ExperimentConfig
    driver: Genie3DriverConfig
    sampling: Genie3SamplingConfig
    runtime: Genie3RuntimeConfig
