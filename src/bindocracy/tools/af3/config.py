"""What an AlphaFold 3 scoring run's YAML is allowed to say.

A third plugin, for the same reason Chai-1 is a second: AF3 has its own image,
its own entry point (`run_alphafold.py`, driven by a JSON job file rather than a
Python API), and its own knob vocabulary. What it shares is the design set, the
metric registry and the `metrics.jsonl` contract, which is the seam worth
sharing.

Two AF3-specific facts are encoded here rather than left to a runbook, because
both are the kind of thing that silently produces a different experiment.

**The weights live outside the image.** `WEIGHTS_TERMS_OF_USE.md` restricts
distribution, so `af3.def` binds them read-only instead of embedding them, and
`runtime.model_dir` is a required field. An image that carried them would make
every copy a redistribution.

**The data pipeline is never run.** AF3's own MSA search wants ~630 GB of
genetic databases. The campaign already has an alignment, and AF3 takes it
directly as `unpairedMsa`, so the driver always passes
`--norun_data_pipeline`. That is not a config option here: making it one would
let a run quietly search a database that is not installed, and fail late.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig
from bindocracy.store.records import canonical_json, sha256_text


class AF3Protocol(ConfigModel):
    """The knobs that change the number, all required.

    `num_trunk_recycles` is AF3's trunk pass count and is the analogue of
    `recycling_steps` elsewhere, with no offset: AF3 counts its own passes and
    is not one of the two backends that add one internally.
    """

    num_trunk_recycles: int = Field(ge=1, le=20)
    # AF3's own default is 200. Stated rather than inherited, like everywhere
    # else here -- the sampler budgets across this panel span 2 to 200.
    num_diffn_timesteps: int = Field(ge=1, le=1000)
    # Structures per design; each becomes one metric replicate.
    num_diffn_samples: int = Field(ge=1, le=20)
    seed: int = 0
    # Whether the campaign alignment is supplied. False means AF3 folds the
    # target single-sequence, which is a real protocol but a different one, and
    # must be chosen rather than fallen into.
    use_target_msa: bool = True

    @model_validator(mode="after")
    def msa_is_explicit(self) -> Self:
        return self


class AF3Readers(ConfigModel):
    """Only `complex`. AF3 co-folds; there is no monomer path that would be the
    same model measuring the binder alone, and a flag that validates but does
    nothing is the failure mode docs/scoring-stage.md section 10 exists to
    stop."""

    complex: bool = True

    @model_validator(mode="after")
    def require_one(self) -> Self:
        if not self.complex:
            raise ValueError("an AF3 run with no readers enabled would measure nothing")
        return self


class AF3Sharding(ConfigModel):
    jobs: int = Field(ge=1, le=512)


class AF3Runtime(ConfigModel):
    container: Path
    # Directory holding af3.bin.zst, bound read-only. Required: the image does
    # not carry the weights and never should.
    model_dir: Path
    # Job-private writable scratch for the per-design job JSON and AF3's own
    # output tree, which is large and disposable.
    work_root: Path


class AF3Config(ToolConfig):
    tool: str = "af3"
    design_set: Path
    protocol: AF3Protocol
    readers: AF3Readers = AF3Readers()
    sharding: AF3Sharding
    runtime: AF3Runtime
    driver_script: Path

    @property
    def metric_prefix(self) -> str:
        return "af3"

    @property
    def protocol_fields(self) -> dict[str, Any]:
        """What makes two runs comparable. Excludes the design set, so scoring
        more designs later extends a measurement rather than starting one."""
        return {
            "model": "af3",
            "readers": self.readers.model_dump(mode="json"),
            "data_pipeline": "disabled",
            **self.protocol.model_dump(mode="json"),
        }

    @property
    def protocol_hash(self) -> str:
        return sha256_text(canonical_json(self.protocol_fields))
