"""What an upstream OpenFold3 scoring run's YAML is allowed to say.

The official `openfoldconsortium/openfold3` image, not mosaic's `jopenfold3`
port. This exists because the two disagree, and measurably: on GFP, mosaic's
OF3 returns pLDDT 38.5 and a structure 24 A from the six-model consensus, while
this image returns 88.7 and 3.9 A -- and the two differ from each other by
24.6 A. On the labelled Nipah-G set mosaic's OF3 scored 0.597, below the
sequence-only control bar of 0.642.

**`metric_prefix` is `of3_upstream`, and that is not cosmetic.** They are
different models. One `of3_iptm` column holding both would silently average two
implementations, which is the mistake `protenix_mini` and `protenix_base` are
kept apart to avoid -- and here the two differ far more than those do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig
from bindocracy.store.records import canonical_json, sha256_text

# The thirteen basenames `parse_msas_direct` accepts. Anything else is dropped
# with a bare `continue` (core/data/io/sequence/msa.py:274), leaving an empty
# MSA dict and an IndexError several frames later that names nothing relevant.
# The driver stages the campaign alignment as `colabfold_main.a3m`; this list
# is here so preflight can say why if that ever changes.
ACCEPTED_MSA_BASENAMES = frozenset({
    "uniref90_hits", "uniprot_hits", "bfd_uniclust_hits", "bfd_uniref_hits",
    "cfdb_uniref30", "mgnify_hits", "rfam_hits", "rnacentral_hits", "nt_hits",
    "concat_cfdb_uniref100_filtered", "mmseqs_colabfold", "colabfold_main",
    "colabfold_paired",
})


class OF3Protocol(ConfigModel):
    """The knobs that change the number, all required.

    OpenFold3 exposes fewer than the mosaic backends do: there is no trunk-pass
    argument on the CLI, so recycling is whatever the runner yaml specifies.
    That is recorded rather than pretended away -- a protocol hash that claimed
    to pin recycling it cannot set would be worse than one that does not.
    """

    num_diffusion_samples: int = Field(ge=1, le=20)
    seed: int = 0
    # Whether the campaign alignment is supplied. False means folding the
    # target single-sequence, which is a real protocol but a different one.
    use_target_msa: bool = True
    # Never true in this harness: letting OpenFold3 search its own alignment
    # would fold the target against a different MSA from every other scorer,
    # which is the confound the MSA work removed. Kept as a field so the run
    # records that it was off rather than leaving it unstated.
    use_msa_server: bool = False

    @model_validator(mode="after")
    def refuse_the_server(self) -> Self:
        if self.use_msa_server:
            raise ValueError(
                "use_msa_server: true would fold the target against an alignment "
                "OpenFold3 searched for itself, not the campaign's, making every "
                "cross-model comparison measure a change of MSA as well as a "
                "change of model"
            )
        return self


class OF3Readers(ConfigModel):
    complex: bool = True

    @model_validator(mode="after")
    def require_one(self) -> Self:
        if not self.complex:
            raise ValueError("a run with no readers enabled would measure nothing")
        return self


class OF3Sharding(ConfigModel):
    jobs: int = Field(ge=1, le=512)


class OF3Runtime(ConfigModel):
    container: Path
    # The original torch checkpoint, e.g. of3-p2-155k.pt. Not in the image.
    checkpoint: Path
    work_root: Path


class OF3UpstreamConfig(ToolConfig):
    tool: str = "of3_upstream"
    design_set: Path
    protocol: OF3Protocol
    readers: OF3Readers = OF3Readers()
    sharding: OF3Sharding
    runtime: OF3Runtime
    driver_script: Path

    @property
    def metric_prefix(self) -> str:
        return "of3_upstream"

    @property
    def protocol_fields(self) -> dict[str, Any]:
        return {
            "model": "of3_upstream",
            "readers": self.readers.model_dump(mode="json"),
            "checkpoint": Path(self.runtime.checkpoint).name,
            **self.protocol.model_dump(mode="json"),
        }

    @property
    def protocol_hash(self) -> str:
        return sha256_text(canonical_json(self.protocol_fields))
