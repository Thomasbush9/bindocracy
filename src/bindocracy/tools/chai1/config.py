"""What a Chai-1 scoring run's YAML is allowed to say.

Chai-1 is a second scorer, not a seventh entry in `ScorerConfig.model`. The
existing scorer is one plugin over six models because those six share an API:
mosaic's `StructurePredictionModel`, one container, one driver, one set of
knobs. Chai-1 shares none of that -- its own image, its own CLI, its own
alignment format (`.aligned.pqt`, not `.a3m`) and its own sampler vocabulary.
Adding it as a `ScoringModelName` literal would put a second container and a
second featurization path behind a field that every other value reads from
mosaic, and `ACCEPTS_TARGET_MSA` would start meaning two different things.

What it *does* share is the part worth sharing: the design set, the metric
registry, and the `metrics.jsonl` contract the scorer adapter already reads.
So this is a separate plugin over the same seam, which is what
`docs/adding-a-tool.md` is for.

The protocol knobs have no defaults, for the reason `scorer/config.py` gives at
length: an unset default is an experiment nobody chose. Chai's own defaults are
a case in point -- `num_diffn_timesteps=200` against Boltz's 25 and Protenix
mini's 2, and `num_diffn_samples=5` where the scorer's smoke runs used one
sample. A config that may omit these is a config that silently ran something
else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig
from bindocracy.store.records import canonical_json, sha256_text


class Chai1Protocol(ConfigModel):
    """The knobs that change the number, all of them required.

    `num_trunk_recycles` is Chai's trunk pass count and is the analogue of
    `recycling_steps` elsewhere. It is NOT normalised by `RECYCLING_OFFSET`:
    that table exists because OpenFold3 and ESMFold2 add a pass internally
    while Boltz does not, and Chai's count is its own. Comparing a Chai
    recycle budget to a Boltz one is a judgement call for whoever reads the
    numbers, not something this config can launder into equivalence.
    """

    num_trunk_recycles: int = Field(ge=1, le=20)
    num_diffn_timesteps: int = Field(ge=1, le=1000)
    num_diffn_samples: int = Field(ge=1, le=20)
    num_trunk_samples: int = Field(ge=1, le=10)
    # 0 means no subsampling. Chai's default; stated rather than assumed.
    recycle_msa_subsample: int = Field(ge=0, le=16384)
    seed: int = 0
    # ESM embeddings are part of Chai's featurization and the traced ESM2-3B is
    # embedded in the image. Turning them off is a different model, not a
    # cheaper one, so it is recorded in the protocol hash.
    use_esm_embeddings: bool = True
    # Whether the target alignment is supplied. False is a real protocol, not a
    # misconfiguration -- but it must be chosen, because Chai only *warns* when
    # an MSA is absent and otherwise runs single-sequence, which looks like a
    # quality result rather than a plumbing failure.
    use_target_msa: bool = True
    # Chai's memory/speed tradeoff. It does not change the prediction, so it
    # sits outside `protocol` below.
    low_memory: bool = True

    @property
    def total_samples(self) -> int:
        """Structures produced per design. Each becomes one metric replicate."""
        return self.num_trunk_samples * self.num_diffn_samples


class Chai1Readers(ConfigModel):
    """Which measurements are taken.

    Only `complex` today. Chai-1 co-folds; there is no monomer path that would
    be the same model measuring the binder alone, and a flag that validates but
    does nothing is the failure mode `docs/scoring-stage.md` §10 item 2 exists
    to stop. When a monomer or epitope reader lands it gets a flag then.
    """

    complex: bool = True

    @model_validator(mode="after")
    def require_one(self) -> Self:
        if not self.complex:
            raise ValueError("a Chai-1 run with no readers enabled would measure nothing")
        return self


class Chai1Sharding(ConfigModel):
    jobs: int = Field(ge=1, le=512)


class Chai1Runtime(ConfigModel):
    """Where the image and its inputs live.

    No `weights` and no `dev_source`. Chai's eight inference assets are
    embedded in the image and mounted read-only, so the container digest alone
    determines what ran -- which is the property the mosaic scorer is currently
    missing and has to record a `dev_source_sha256` to compensate for.
    """

    container: Path
    # Directory of `<sha256>.aligned.pqt` files, one per sequence. Produced by
    # `chai-lab a3m-to-pqt`; see `docs/chai1.md`.
    msa_directory: Path | None = None
    # Job-private writable scratch. The image routes every incidental cache
    # (torch, numba, triton, matplotlib) here via CHAI_RUNTIME_DIR; without it
    # the container makes a fresh /tmp/chai1-* per invocation and leaves it.
    scratch: Path


class Chai1Config(ToolConfig):
    tool: str = "chai1"
    design_set: Path
    protocol: Chai1Protocol
    readers: Chai1Readers = Chai1Readers()
    sharding: Chai1Sharding
    runtime: Chai1Runtime
    driver_script: Path

    @property
    def metric_prefix(self) -> str:
        return "chai1"

    @property
    def protocol_fields(self) -> dict[str, Any]:
        """What makes two runs comparable.

        Excludes the design set, so scoring more designs later extends a
        measurement rather than starting a new one. Excludes `low_memory`,
        which trades speed for memory without changing the prediction.
        """
        knobs = self.protocol.model_dump(mode="json")
        knobs.pop("low_memory", None)
        return {
            "model": "chai1",
            "readers": self.readers.model_dump(mode="json"),
            **knobs,
        }

    @property
    def protocol_hash(self) -> str:
        return sha256_text(canonical_json(self.protocol_fields))
