"""What a custom optimization run's YAML is allowed to say.

An optimization run is the first thing in this harness that is **both** shapes
at once: it reads a frozen design set like a scorer, and it emits designs like a
generator. Everything unusual in this module follows from that.

The script is the user's, not this repository's -- the same bargain
`functions/models.py` strikes for scoring, one step up in consequence. A bad
custom scoring function adds a column a query can ignore. A bad custom
optimizer writes sequences into `designs` that look exactly like designs a
model produced, permanently, and nothing downstream can tell them apart. So the
contract here is stricter in three specific places: the sequence alphabet
(`contract.py`), the child accounting (`contract.py`), and `loss_models` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from bindocracy.adapters.scoring import MetricSpec, registered_keys
from bindocracy.config.models import ConfigModel, ToolConfig
from bindocracy.functions.models import MetricDeclaration
from bindocracy.store.records import canonical_json, sha256_text
from bindocracy.tools.optimize.contract import OptimizerInputKind


class OptimizerRuntime(ConfigModel):
    """Where the script runs.

    One container for both the wrapper and the script, because the wrapper's
    only job is to marshal files and it is written against the standard library
    alone so that it can run inside whatever image the script needs. Two
    containers would mean two images to keep in step for no gain.
    """

    # Absent means the harness's own interpreter on the host. Right for a
    # numpy-and-stdlib optimizer; wrong for anything that folds.
    container: Path | None = None
    # Bound over the image's own copy of a library, the way the scorer's
    # `dev_source` does. Needed today because `mosaic.sif` predates the MSA
    # routing; see docs/mosaic-rebuild.md. Recorded by digest in the run.
    dev_source: Path | None = None
    # Job-private writable scratch, per the same reasoning as Chai-1's.
    scratch: Path | None = None
    # The interpreter inside the container. Images here disagree: mosaic has
    # `python`, Chai-1 needs `/opt/chai-venv/bin/python`. Stated rather than
    # guessed, because `singularity exec ... python` resolves against whatever
    # PATH survives `--cleanenv`.
    container_python: str = "python"


class OptimizerSharding(ConfigModel):
    jobs: int = Field(ge=1, le=512)


class OptimizeConfig(ToolConfig):
    """One optimization pass over one frozen design set.

    `name` is also the metric prefix, so every number this run stores is
    `<name>_<metric>`. Two optimizers with different losses must therefore have
    different names, which is the same rule that keeps `protenix_mini` and
    `protenix_base` in separate columns.
    """

    tool: str = "optimize"

    # The candidate set to improve. Build it with `bindocracy designset build`,
    # usually from what a filter chose:
    #   designset build DB --passed-filter worth_optimizing --filter-run <id>
    design_set: Path

    # The user's script. Invoked as:
    #   <python> script.py --inputs IN.jsonl --outputs OUT.jsonl --context CTX.json
    script: Path
    args: tuple[str, ...] = ()
    inputs: tuple[OptimizerInputKind, ...] = ("sequence",)
    # Which stored metrics to hand the script per parent, by full stored name
    # (`boltz2_iptm`, not `iptm`). Required when `inputs` includes `metrics`:
    # handing over every column would make the script's behaviour depend on
    # which scoring runs happen to have landed.
    metric_inputs: tuple[str, ...] = ()

    # WHICH MODELS THE LOSS SAW. Required, and the most important field here.
    #
    # If an optimizer drives a design against Boltz-2 ipTM and the children are
    # later ranked by Boltz-2 ipTM, the ranking measures the optimizer rather
    # than the binder -- and after the fact nothing in the database can detect
    # that, because no run has ever recorded which models fed a loss. This is
    # the field that makes the check possible: it is stored on the run and in
    # every child's metadata, so a selection can exclude the models that
    # already had their say.
    #
    # An empty tuple is a claim, not a default: it says the loss consulted no
    # structure predictor at all (a sequence-only optimizer -- charge, motif
    # removal, a language model). Say so explicitly by writing `[]`.
    loss_models: tuple[str, ...]

    # What the script returns, per child. Directions are required for the same
    # reason `functions/models.py` requires them.
    metrics: dict[str, MetricDeclaration] = Field(default_factory=dict)

    # Which model's poses to hand over, when `inputs` includes 'structure'.
    # Required in that case: two models disagree about where the binder sits by
    # a median 22.7 A, so "the structure" of a design is not one thing.
    structures_from: str | None = None
    # Optimize the parents that have a pose, rather than refusing the run, when
    # some do not. Off by default -- a run that silently optimizes a subset
    # reports a smaller number with nothing saying why.
    allow_unfolded: bool = False

    # Ceiling on children per parent. Enforced by the driver, which refuses
    # rather than truncates.
    max_children: int = Field(default=1, ge=1, le=64)
    seed: int = 0
    timeout_seconds: int = Field(default=86400, ge=1)

    # Allowed length change, as a guard rather than a preference. An optimizer
    # that may change length is fine and sometimes the point, but a length
    # equal to the target's would make the epitope function's
    # binder-found-by-length heuristic ambiguous, and every JAX scorer
    # recompiles per length -- so a run that will produce 40 new lengths should
    # say so rather than discover it at scoring time.
    length_delta: int = Field(default=0, ge=0, le=512)

    sharding: OptimizerSharding
    runtime: OptimizerRuntime = OptimizerRuntime()
    driver_script: Path

    @model_validator(mode="after")
    def metrics_do_not_shadow_registered(self) -> Self:
        """A declared metric may not reuse a registered name.

        `iptm` means one thing. An optimizer reporting its own `iptm` -- its
        internal estimate rather than a scorer's measurement -- under that name
        would put two definitions in one column, which is the mistake the whole
        prefix convention exists to prevent. The stored column is prefixed with
        this run's `name`, but the registry key must still be unique.
        """
        clashes = sorted(set(self.metrics) & set(registered_keys()))
        if clashes:
            raise ValueError(
                f"optimizer {self.name!r} declares metric(s) {clashes} that are "
                "already registered with a fixed meaning; choose another name "
                "(`opt_iptm`, `loss_iptm`) so a stored number cannot be mistaken "
                "for a scorer's measurement of the same quantity"
            )
        return self

    @model_validator(mode="after")
    def metric_inputs_are_named(self) -> Self:
        """Asking for `metrics` means naming which ones.

        Handing over every metric column a design happens to have would make a
        script's behaviour depend on which scoring runs had landed by then --
        the same run silently doing something different next month. Naming them
        also lets preflight check they exist before a GPU is allocated.
        """
        if "metrics" in self.inputs and not self.metric_inputs:
            raise ValueError(
                "inputs includes 'metrics' but metric_inputs names none; list the "
                "stored metric names the script should be handed (`boltz2_iptm`, "
                "not `iptm`), or drop 'metrics' from inputs"
            )
        if self.metric_inputs and "metrics" not in self.inputs:
            raise ValueError(
                "metric_inputs names metrics but inputs does not include 'metrics', "
                "so the script would never be handed them"
            )
        return self

    @property
    def metric_prefix(self) -> str:
        return self.name

    @property
    def specs(self) -> dict[str, MetricSpec]:
        return {key: value.as_spec(key) for key, value in self.metrics.items()}

    @property
    def protocol_fields(self) -> dict[str, Any]:
        """What makes two optimization runs comparable.

        Excludes the design set, so optimizing more parents later extends a
        measurement rather than starting a new one -- the same choice
        `scorer/config.py` makes. Excludes runtime entirely: a container path
        and a scratch directory do not change what the optimizer does. Includes
        the script's *path* but not its bytes, because the bytes are hashed
        separately at preflight and recorded on the run; a config hash that
        moved every time a script was edited would make the protocol
        unstateable.
        """
        return {
            "optimizer": self.name,
            "script": str(self.script),
            "args": list(self.args),
            "inputs": list(self.inputs),
            "metric_inputs": list(self.metric_inputs),
            "loss_models": sorted(self.loss_models),
            "metrics": {
                key: value.model_dump(mode="json") for key, value in sorted(self.metrics.items())
            },
            "structures_from": self.structures_from,
            "max_children": self.max_children,
            "length_delta": self.length_delta,
            "seed": self.seed,
        }

    @property
    def protocol_hash(self) -> str:
        return sha256_text(canonical_json(self.protocol_fields))

    @property
    def protocol(self) -> dict[str, Any]:
        return self.protocol_fields
