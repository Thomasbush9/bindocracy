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

from bindocracy.adapters.scoring import MetricSpec, registered_keys
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


class CustomFunction(ConfigModel):
    """A user-supplied scoring script.

    The contract is the one every driver here already uses: JSONL in, JSONL
    out. `contract.py` documents the exact row shapes.
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
    metrics: dict[str, MetricDeclaration] = Field(min_length=1)
    timeout_seconds: int = Field(default=3600, ge=1)

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


__all__ = [
    "CustomFunction",
    "FunctionInputKind",
    "FunctionsConfig",
    "MetricDeclaration",
    "MetricDirection",
]
