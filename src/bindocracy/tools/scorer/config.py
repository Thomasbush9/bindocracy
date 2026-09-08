"""What a scoring run's YAML is allowed to say.

DRAFT -- not registered yet. See docs/scoring-stage.md.

**A scorer is an ordinary tool, and this is an ordinary model config.** That is
the whole design decision. `bindocracy config load`, `config export`, the
workflow index, provenance archiving and the config-hash identity all work
unchanged, and `docs/adding-a-tool.md`'s "one package and one line" still
holds. Nothing about scoring needed a third kind of config document.

**One plugin, not six.** The structural model is a *field*, not a tool name.
Six plugins named `boltz2_scorer`, `esmfold2_scorer` and so on would be six
copies of one adapter and one launch path differing by a string, and a bug
fixed in the adapter would need fixing six times. What genuinely differs
between models is which knobs are valid, and that is validation, not
architecture. `MetricSpec.stored_name(model)` already makes the stored metric
name follow the field.

**The protocol knobs have no defaults on purpose.** `mosaic_setup/benchmark`
documents six knobs that move model-to-model correlations more than the model
differences being measured, and two of them are traps that only bite through
defaults: Protenix's diffusion sampler defaults to *2 steps* against Boltz's
25, and OpenFold3 and ESMFold2 each add one trunk pass internally, so
`recycling 3` gave three models three passes and two of them four. A config
that may omit these is a config that silently ran a different experiment. So
they are required fields, and the resolved values are echoed into the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from bindocracy.config.models import ConfigModel, ToolConfig
from bindocracy.store.records import canonical_json, sha256_text

# Models reachable through mosaic.sif today, confirmed by its own model audit.
# Adding one is a literal here plus whatever validation its knobs need; it is
# not a new plugin.
ScoringModelName = Literal["boltz2", "boltz1", "af2", "esmfold2", "of3", "protenix"]

# Which backends read a supplied target alignment. OpenFold3, Protenix and
# ESMFold2 historically accepted `msa_path` and silently queried a public
# server instead, so every cross-family comparison measured a change of
# alignment as well as a change of model. mosaic now routes all of them
# through `require_msa`, which raises rather than falling back; this table is
# the harness-side assertion that the campaign's alignment is the one used.
#
# af2 is False, and the reasoning that says otherwise is a trap worth
# recording. `models/af2.py:343-350` does read `chain.use_msa` and load the
# a3m, which reads as support; `design_config.py` marks af2 `accepts_msa:
# False`, which reads as a stale note contradicting the code. The code path
# that settles it is neither: a *complex* run raises
# `AssertionError: AF2 interface does not support MSA yet`, observed on all 20
# designs of the first scoring run (2026-09-07). AF2 takes an MSA for a single
# chain and refuses one at an interface, so for binder scoring it is False.
#
# Trusting the source read over the observation cost a GPU job. This is the
# case harness-design §3 makes about writing adapters against observed output:
# where the docs and the code disagreed, both were misleading and only running
# it was decisive.
ACCEPTS_TARGET_MSA: dict[str, bool] = {
    "boltz2": True,
    "boltz1": True,
    "af2": False,
    "esmfold2": True,  # Full only; Fast has no MSA encoder and raises
    "of3": True,
    "protenix": True,
}

# Backends with a diffusion sampler. AF2 has none and asserts that it is not
# handed one, so `sampling_steps` must be absent for it and present for
# everything else.
HAS_SAMPLER: dict[str, bool] = {
    "boltz2": True,
    "boltz1": True,
    "af2": False,
    "esmfold2": True,
    "of3": True,
    "protenix": True,
}


class ScoringModel(ConfigModel):
    """The structural model and the exact protocol it runs.

    Every field that changes the numbers is required. There is no `= 25` here
    and there should never be one: a default is how two runs come to disagree
    while appearing to say the same thing.
    """

    name: ScoringModelName

    # Trunk passes, as the *user* means them. Backends that add one internally
    # are normalised by the driver, so this number means the same thing for
    # every model -- which is the whole point of stating it here.
    recycling_steps: int = Field(ge=1, le=20)

    # Diffusion steps. Required for every sampler-bearing backend, refused for
    # AF2. Protenix's own default is 2 and scoring it there put it last in the
    # benchmark for reasons that were the harness's fault, not the model's.
    sampling_steps: int | None = Field(default=None, ge=1, le=200)

    # Independent samples per design. Each becomes one `replicate` in the
    # metrics table; nothing is averaged at write time.
    num_samples: int = Field(ge=1, le=32)

    seed: int = 42

    # Protenix ships mini, tiny, base, 2025 and v2; only mini's weights are on
    # disk and the others would download. Refused for every other model.
    variant: str | None = None

    # Whether the campaign's target alignment is passed in. Required to be
    # explicit rather than inferred, because "the model ignored the MSA I gave
    # it" is invisible in the output and changes every cross-family number.
    use_target_msa: bool

    @model_validator(mode="after")
    def check_model_specific_knobs(self) -> Self:
        has_sampler = HAS_SAMPLER[self.name]
        if has_sampler and self.sampling_steps is None:
            raise ValueError(
                f"{self.name} has a diffusion sampler, so sampling_steps is required; "
                "leaving it unset means taking whichever default that backend ships, "
                "and those differ by more than an order of magnitude"
            )
        if not has_sampler and self.sampling_steps is not None:
            raise ValueError(f"{self.name} has no diffusion sampler; drop sampling_steps")

        if self.variant is not None and self.name != "protenix":
            raise ValueError(f"variant applies to protenix only, not {self.name}")

        if self.use_target_msa and not ACCEPTS_TARGET_MSA[self.name]:
            raise ValueError(
                f"{self.name} cannot consume a supplied target MSA; set "
                "use_target_msa: false, or score with a model that can"
            )
        if self.name == "af2" and self.num_samples > 1:
            raise ValueError(
                "af2 is deterministic, so num_samples > 1 folds the same structure "
                "repeatedly; use one sample, or vary the MSA seed instead"
            )
        return self


class Readers(ConfigModel):
    """Which measurements are taken from the fold.

    This is the coarse-versus-fine switch, and it is a set of flags rather than
    a mode because the groups are independent and three of the four cost no
    extra inference. `complex` reads the confidence tensors the fold already
    returned; `epitope` reads its coordinates. Only `monomer` costs a second
    pass, and only `inverse_folding` loads another model.
    """

    complex: bool = True
    monomer: bool = False
    epitope: bool = False
    inverse_folding: bool = False

    @model_validator(mode="after")
    def require_one(self) -> Self:
        if not any((self.complex, self.monomer, self.epitope, self.inverse_folding)):
            raise ValueError("a scoring run with no readers enabled would measure nothing")
        return self


class ScorerDriver(ConfigModel):
    """The script executed inside the container.

    Archived per run and run from the archive, so the code that produced a
    number stays beside the number even after the working tree moves on.
    """

    script: Path


class ScorerRuntime(ConfigModel):
    """Where the container and its weights are, exactly as Mosaic's config does."""

    container: Path
    weights: Path
    exec_wrapper: Path
    scratch: Path

    # A source tree bound over the image's `/opt/mosaic/src`.
    #
    # Needed because `mosaic.sif` embeds a build that predates the MSA-routing
    # fix: its OpenFold3, ESMFold2 and Protenix wrappers read `use_msa` to
    # decide a chain needs an alignment and then fetch one from ColabFold,
    # never consulting `msa_path`. The image ships `mosaic/msa.py` and imports
    # `require_msa` in exactly zero files. Boltz-1 and Boltz-2 read the local
    # a3m; AF2 asserts rather than substituting. So three of six backends
    # silently scored against a different alignment (2026-09-07).
    #
    # This is a stopgap and is typed as one. `mosaic-exec.sh` announces the
    # override loudly because a run using it is not reproducible from the
    # container digest alone -- so preflight digests the tree and the run
    # records that digest beside the container, the same rule
    # `known-issues.md` §2.3 sets for the Genie 3 JAX overlays. The durable
    # fix is rebuilding the image; then this field goes away.
    dev_source: Path | None = None


class ScorerSharding(ConfigModel):
    """How the design set splits across tasks.

    `jobs` is a task count, not a batch size. Entries are ordered by binder
    length and cut into contiguous blocks, so raising this lowers wall clock
    without widening the JIT recompilation each task pays.
    """

    jobs: int = Field(default=1, ge=1, le=256)


class ScorerConfig(ToolConfig):
    """One scoring run: one model, one protocol, one design set."""

    schema_version: Literal[1]
    tool: Literal["scorer"]

    # The frozen candidate set. A path to a design-set manifest written by
    # `bindocracy designset build`; its digest is what a rank decision is
    # scoped to, and `ToolPlan.inputs` digests the file so the run records the
    # bytes it scored rather than the path it read them from.
    design_set: Path

    model: ScoringModel
    readers: Readers = Readers()
    sharding: ScorerSharding = ScorerSharding()
    driver: ScorerDriver
    runtime: ScorerRuntime

    @property
    def driver_script(self) -> Path:
        return self.driver.script

    @model_validator(mode="after")
    def check_runtime_fits(self) -> Self:
        if self.resources.gpus < 1:
            raise ValueError("a scoring run folds structures and needs a GPU")
        return self

    @property
    def protocol(self) -> dict[str, Any]:
        """Everything that decides what the numbers mean, and nothing else.

        Deliberately excludes the design set, the run name and the resources. Two
        runs sharing this dictionary measured the same quantity the same way, so
        their metrics are comparable even though they scored different designs
        -- which is what makes incremental scoring safe. Two runs that differ
        here are not comparable however similar the YAML looks.
        """
        return {
            "model": self.model.model_dump(mode="json"),
            "readers": self.readers.model_dump(mode="json"),
        }

    @property
    def protocol_hash(self) -> str:
        """The comparability key, stamped into `runs.workflow_metadata`.

        `model_config_id` cannot serve this purpose because it also covers the
        design set, so scoring two different sets under one protocol yields two
        config IDs and no way to tell they agree. This hash is that way.
        """
        return sha256_text(canonical_json(self.protocol))

    @property
    def metric_prefix(self) -> str:
        """What every metric this run writes is named with.

        Protenix's variants are genuinely different models with different
        weights, so the variant is part of the prefix; folding them into one
        `protenix_iptm` column would silently mix mini and base.
        """
        if self.model.name == "protenix" and self.model.variant:
            return f"protenix_{self.model.variant}"
        return self.model.name
