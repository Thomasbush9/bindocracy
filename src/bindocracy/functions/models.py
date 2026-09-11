"""What a scoring function is, and what its config is allowed to say.

A **condition** is something that gets folded — it costs a GPU pass per design
per model, and lives in the scorer's `readers` block. A **function** is
something computed from the result. The two were one flag until 2026-09-11,
which put three different cost classes behind one name:

    complex, monomer    a GPU fold, per design, per model
    epitope             geometry over a pose already on disk -- free
    inverse_folding     loads a second model

Functions read what the scorer already produced, so they can be added later and
backfilled across every structure ever saved, on a CPU, in minutes. That is a
different enough proposition from adding a folding model to deserve its own
path.

**A custom function is the extension point.** A script that reads a JSONL of
designs and writes a JSONL of metrics -- any language, any container, or none.
It needs no plugin, no `register()` line and no change under `src/`. What it
does need is to declare what its metrics mean, which is the one thing the
harness will not infer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from bindocracy.adapters.scoring import MetricSpec, registered_keys, spec_for
from bindocracy.config.models import ConfigModel
from bindocracy.store.records import MetricDirection

# What a custom script can be handed per design. `structure` requires a saved
# pose, so a function asking for it is skipped for designs that have none
# rather than being fed a null path.
FunctionInputKind = Literal["sequence", "structure", "target_sequence"]


class MetricDeclaration(ConfigModel):
    """What one custom metric means.

    `direction` has no default, deliberately. The metric registry exists
    because a number stored with the wrong direction sorts backwards and
    nothing in the row says so, and the only person who knows which way a new
    metric points is whoever wrote the script. Letting this default to `none`
    would quietly give away the guarantee the registry is for.
    """

    direction: MetricDirection
    unit: str | None = None
    description: str = ""

    def as_spec(self, key: str) -> MetricSpec:
        return MetricSpec(
            key=key, direction=self.direction, unit=self.unit,
            description=self.description,
        )


# How a function's metrics are named in the database.
#
#   function      `<function name>_<metric>` -- for anything whose value does
#                 not depend on which model produced the input. A binder's net
#                 charge is its net charge.
#   source_model  `<model>_<metric>` -- for geometry read off a particular
#                 model's structure. `adapters/scoring.py::epitope_metric_name`
#                 makes the argument: two models disagree about where the
#                 binder sits, so one unprefixed `epitope_coverage` column
#                 would collapse that disagreement and hide it.
MetricPrefix = Literal["function", "source_model"]


class ScoringFunction(ConfigModel):
    """What every scoring function has, built-in or user-supplied.

    Built-ins go through exactly this shape and exactly the runner below, so
    the extension point is exercised by the harness rather than merely offered
    to others -- the contract is proven by use.
    """

    name: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    script: Path
    # Run inside this image. Absent means the harness's own interpreter, which
    # is right for a function whose dependencies are already installed.
    container: Path | None = None
    # Extra arguments appended after the contract's own, for a script that
    # takes options.
    args: tuple[str, ...] = ()
    inputs: tuple[FunctionInputKind, ...] = ("sequence",)
    timeout_seconds: int = Field(default=3600, ge=1)
    prefix: MetricPrefix = "function"

    @property
    def specs(self) -> dict[str, MetricSpec]:  # pragma: no cover - overridden
        raise NotImplementedError


class BuiltinFunction(ScoringFunction):
    """A function this repository ships.

    Its metrics are already in the registry with a fixed meaning, so it
    declares nothing: `specs` resolves them by name. That is the difference
    from a custom function, which must declare because nobody else knows what
    its numbers mean.
    """

    metric_keys: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def metrics_are_registered(self) -> Self:
        missing = sorted(set(self.metric_keys) - set(registered_keys()))
        if missing:
            raise ValueError(
                f"built-in function {self.name!r} names unregistered metric(s) "
                f"{missing}; a built-in's metrics belong in adapters/scoring.py"
            )
        return self

    @property
    def specs(self) -> dict[str, MetricSpec]:
        return {key: spec_for(key) for key in self.metric_keys}


class CustomFunction(ScoringFunction):
    """A user-supplied scoring script.

    The contract is the one every driver here already uses: JSONL in, JSONL
    out. `contract.py` documents the exact row shapes.
    """

    metrics: dict[str, MetricDeclaration] = Field(min_length=1)

    @model_validator(mode="after")
    def metrics_do_not_shadow_builtins(self) -> Self:
        """A custom metric may not reuse a registered name.

        Two different definitions under one name is the failure the prefix
        convention exists to prevent -- it is why `protenix_mini` and
        `protenix_base` are separate columns, and why mosaic's OF3 and the
        upstream one are. A custom function silently redefining `iptm` would be
        the same mistake with none of the visibility.
        """
        clashes = sorted(set(self.metrics) & set(registered_keys()))
        if clashes:
            raise ValueError(
                f"custom function {self.name!r} declares metric(s) {clashes} that "
                "are already registered with a fixed meaning; choose another name "
                "-- the stored column is prefixed with the function name, but the "
                "registry key must still be unique"
            )
        return self

    @property
    def specs(self) -> dict[str, MetricSpec]:
        return {key: value.as_spec(key) for key, value in self.metrics.items()}


class FunctionsConfig(ConfigModel):
    """Which functions run over a scored design set.

    Built-ins are flags; user scripts are a list. Both write metric rows the
    same way, and the built-ins go through the same runner, so the extension
    point is exercised by the harness rather than merely offered to others.
    """

    # Geometry over a saved pose. No GPU, no model.
    epitope: bool = False
    # Sequence-only. No GPU, no structure. These are also the negative
    # controls: on Nipah-G, length alone reaches AUC 0.642 and five of nine
    # scorers sit within 0.06 of it.
    sequence: bool = False
    # Loads a second model, so it is opt-in on cost grounds rather than
    # correctness.
    inverse_folding: bool = False
    custom: tuple[CustomFunction, ...] = ()

    @model_validator(mode="after")
    def require_one(self) -> Self:
        if not (self.epitope or self.sequence or self.inverse_folding or self.custom):
            raise ValueError(
                "a functions block with nothing enabled would compute nothing"
            )
        names = [f.name for f in self.custom]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate custom function name(s): {duplicates}")
        return self

    @property
    def enabled_builtins(self) -> tuple[str, ...]:
        return tuple(
            name for name, on in (
                ("epitope", self.epitope),
                ("sequence", self.sequence),
                ("inverse_folding", self.inverse_folding),
            ) if on
        )

    def resolve(self, **builtin_overrides) -> tuple[ScoringFunction, ...]:
        """Every enabled function, built-in and custom, in one list.

        Built-ins come back as the same type the runner takes for a user
        script, because they are run the same way. `builtin_overrides` is how a
        caller supplies what only it knows -- the epitope function needs the
        campaign's hotspots as `args`, for instance -- keyed by function name.
        """
        resolved: list[ScoringFunction] = []
        for name in self.enabled_builtins:
            if name not in BUILTIN_FUNCTIONS:
                raise ValueError(
                    f"function {name!r} is enabled but not implemented yet; it "
                    "would validate, launch and measure nothing"
                )
            resolved.append(builtin(name, **builtin_overrides.get(name, {})))
        resolved.extend(self.custom)
        return tuple(resolved)


# The functions this repository ships. `script` is resolved against
# `drivers/functions/` at use, so these stay declarative.
BUILTIN_FUNCTIONS: dict[str, dict] = {
    "sequence": {
        "script": "sequence_metrics.py",
        "inputs": ("sequence",),
        "prefix": "function",
        "metric_keys": (
            "length", "net_charge", "molecular_weight", "hydrophobic_fraction",
            "n_cysteines", "n_glycosylation_motifs", "max_low_complexity_run",
        ),
    },
    "epitope": {
        "script": "epitope_metrics.py",
        "inputs": ("structure", "sequence"),
        # Geometry off one model's pose: prefix by that model, not by this
        # function, or two models' disagreement collapses into one column.
        "prefix": "source_model",
        "metric_keys": (
            "epitope_coverage", "n_epitope_contacts", "n_interface_residues",
            "epitope_offset",
        ),
    },
}

BUILTIN_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "drivers" / "functions"


def builtin(name: str, **overrides) -> BuiltinFunction:
    """The shipped function called `name`."""
    spec = dict(BUILTIN_FUNCTIONS[name])
    spec["name"] = name
    spec["script"] = BUILTIN_SCRIPT_DIR / spec["script"]
    spec.update(overrides)
    return BuiltinFunction.model_validate(spec)


__all__ = [
    "BUILTIN_FUNCTIONS",
    "BuiltinFunction",
    "CustomFunction",
    "FunctionInputKind",
    "FunctionsConfig",
    "MetricDeclaration",
    "MetricDirection",
    "MetricPrefix",
    "ScoringFunction",
    "builtin",
]
