"""PXDesign's authored configuration.

Two of PXDesign's CLI defaults are actively wrong for this campaign, and both
are represented here as required or explicitly-defaulted fields rather than
left to the CLI. See docs/known-issues.md sections 1.5 and 1.6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, ToolConfig


class PXDesignSpecConfig(ConfigModel):
    """PXDesign is driven by an input YAML naming the target, MSA and length.

    The spec is authored (target structure, per-chain MSA directory, optional
    crop and hotspots, binder length) and passed to the container's `pipeline`
    with `-i`. It is archived per run and run from the archive, exactly as
    BoltzGen's spec is.
    """

    template: Path
    # Filled in from `template` when the config is loaded, so the stored JSON
    # holds the specification itself rather than a path to it. Authors leave it
    # out; a recovered config carries it and still validates.
    contents: dict | None = None


class PXDesignSamplingConfig(ConfigModel):
    """What one task samples, and the schedule it samples on.

    `designs_per_job` is `--N_sample`, and it is exactly how many rows
    `summary.csv` will have: the CLI appends `--min_total_return` and
    `--max_success_return` equal to it, so the table is **padded with failed
    designs** when fewer pass. Produced and passed are therefore always
    different questions here; see docs/known-issues.md section 3.
    """

    jobs: int = Field(default=1, gt=0)
    designs_per_job: int = Field(gt=0)
    diffusion_steps: int = Field(default=400, gt=0)
    # Passed as `--seeds`. Without it PXDesign seeds from time.time_ns(), which
    # makes a run unreproducible; with it, each task also provably samples
    # something different from its siblings.
    seed_base: int = Field(ge=0)
    # No default, and `custom` is not offered. The CLI's own default IS
    # `custom`, despite its docstring, and `custom` configures no confidence
    # filters at all -- the run completes and writes a summary.csv whose
    # success columns mean nothing. Choosing is cheaper than discovering that.
    preset: Literal["preview", "extended"]
    # The container's config intends piecewise_65 / 1.0 / 2.5, but the CLI
    # emits every shared option unconditionally, so its own defaults
    # (const / 2.5 / 2.5) overwrite the config on every run. Passing the
    # intended values back explicitly is the only way to get them.
    eta_type: str = Field(default="piecewise_65", min_length=1)
    eta_min: float = 1.0
    eta_max: float = 2.5


class PXDesignRuntimeConfig(ConfigModel):
    container: Path
    # bf16 on A100/H100/H200. V100 needs fp32 with the fused kernels off.
    dtype: Literal["fp32", "bf16"] = "bf16"
    # Forwarded to the inner argparse rather than consumed by click, because
    # `pipeline` sets ignore_unknown_options. That also means a typo in one of
    # these reaches argparse instead of being rejected by the CLI, which is why
    # they are typed here and not a free-form list.
    use_fast_ln: bool = True
    use_deepspeed_evo_attention: bool = True
    # Node-local scratch root. TMPDIR must NOT be on Lustre: this image JITs
    # custom kernels through Triton, whose temp-dir cleanup fails there with
    # Errno 39. See docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class PXDesignConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["pxdesign"]
    spec: PXDesignSpecConfig
    sampling: PXDesignSamplingConfig
    runtime: PXDesignRuntimeConfig
