"""Protein-Hunter's authored configuration (Boltz pipeline).

Two things here are unlike the other four tools. The target arrives as a
*sequence on the command line* rather than a file the tool opens, and there is
no seed anywhere in the pipeline -- so a run cannot be reproduced or split
deterministically, and the config says so rather than implying otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig


class ProteinHunterDriverConfig(ConfigModel):
    # Archived and executed from the archive. It seeds the ColabFold MSA cache
    # this run needs and then runs the Boltz design pipeline.
    script: Path


class ProteinHunterSamplingConfig(ConfigModel):
    """Trajectories and cycles, which are different numbers.

    Each trajectory starts from a mostly-X binder and loops `cycles` times of
    {Boltz-2 co-fold, LigandMPNN redesign}. Every cycle emits exactly one
    sequence -- MPNN's batch size is hardcoded to 1 -- so the designs a task
    produces is trajectories times cycles. See docs/known-issues.md section 3.
    """

    jobs: int = Field(default=1, gt=0)
    trajectories_per_job: int = Field(gt=0)
    cycles: int = Field(default=5, gt=0)
    min_binder_length: int = Field(gt=0)
    max_binder_length: int = Field(gt=0)
    # How much of the starting binder is X. The search begins from noise; 90 is
    # the benchmark's value.
    percent_x: int = Field(default=90, ge=0, le=100)
    # Residues MPNN may not use. Cysteine is excluded by default here because
    # an unpaired cysteine is a liability in a secreted binder.
    omit_aa: str = Field(default="C", pattern=r"^[A-Z]*$")
    temperature: float = Field(default=0.1, gt=0)
    diffuse_steps: int = Field(default=200, gt=0)
    recycling_steps: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def lengths_must_be_a_range(self) -> Self:
        if self.max_binder_length < self.min_binder_length:
            raise ValueError("max_binder_length cannot be below min_binder_length")
        return self


class ProteinHunterFilterConfig(ConfigModel):
    """The tool's own thresholds, which decide what lands in high_iptm_*.

    Recorded because they are the definition of `n_passed` for this tool, and
    a run collected under one threshold is not comparable to a run collected
    under another.
    """

    high_iptm_threshold: float = Field(default=0.7, ge=0, le=1)
    high_plddt_threshold: float = Field(default=0.7, ge=0, le=1)


class ProteinHunterMSAConfig(ConfigModel):
    """`single` folds the target with no alignment; `mmseqs` needs a cache.

    No default: `mmseqs` reaches api.colabfold.com unless the ColabFold cache
    is already on disk, and `single` quietly folds a 201-residue target with no
    MSA at all. Both are legitimate and they are not the same experiment, so
    the author picks.
    """

    mode: Literal["single", "mmseqs"]
    # Sequences kept when seeding the cache. Downstream hardcodes 4096 and
    # overrides the caller, so the full alignment would be pushed through the
    # MSA module on every one of a few hundred predictions.
    max_seqs: int = Field(default=512, gt=0)


class ProteinHunterContactConfig(ConfigModel):
    """How the campaign's epitope is enforced, when there is one.

    `--contact_residues` is load-bearing in three separate places upstream: it
    adds a Boltz pocket constraint so generation is conditioned on the epitope,
    it drives a retry loop that rejects binders which miss it, and it is a
    third condition on landing in `high_iptm_*`. Passing the residues without
    saying how they are enforced would leave two of those three implicit.
    """

    # Angstroms, CA-to-CA. The CLI's default is 15.0; the docstring of the
    # function that consumes it says 10.0. The CLI wins, and it is written
    # here so neither has to be guessed.
    cutoff: float = Field(default=15.0, gt=0)
    # When true, a binder that misses the epitope is resampled rather than
    # kept. Upstream calls this `--no_contact_filter` and defaults it off,
    # i.e. filtering on.
    filter: bool = True
    max_retries: int = Field(default=6, gt=0)


class ProteinHunterRuntimeConfig(ConfigModel):
    container: Path
    # Node-local scratch root. LigandMPNN opens a TemporaryDirectory per call,
    # and the container's runscript mkdirs its cache tree under TMPDIR before
    # anything else runs. See docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class ProteinHunterConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["protein_hunter"]
    driver: ProteinHunterDriverConfig
    sampling: ProteinHunterSamplingConfig
    filters: ProteinHunterFilterConfig = ProteinHunterFilterConfig()
    # Only consulted when the campaign names an epitope; a hotspot-free
    # campaign passes no contact residues and none of this applies.
    contacts: ProteinHunterContactConfig = ProteinHunterContactConfig()
    msa: ProteinHunterMSAConfig
    runtime: ProteinHunterRuntimeConfig
