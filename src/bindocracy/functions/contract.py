"""The file contract between the harness and a scoring function.

JSONL in, JSONL out -- the same shape every driver here already uses, so a
custom function is not a new thing to learn. It is also why a function can be
written in any language and run in any container, or none: the interface is two
files, not a Python class.

**Input**, one line per design::

    {"index": 0,
     "design_id": "…",            # present, but a script should key on index
     "sequence": "MKT…",          # the binder
     "target_sequence": "DDN…",   # omitted unless requested
     "structure": "/abs/path.pdb" # omitted unless requested
    }

**Output**, one line per design (the caller selects the replicate)::

    {"index": 0, "metrics": {"buried_sasa": 812.4}}
    {"index": 1, "failed": "no interface found"}

A row may report fewer metrics than declared; a missing one is stored as an
absent row rather than a zero, because "not measured" and "measured as zero"
are different facts. A row with `failed` records that something ran and could
not produce a number, which is different again.

`index` is the design-set index, and it is the join key precisely because a
`design_id` has no business crossing into a user script -- the same reason the
design-set FASTA carries an index rather than an ID.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FunctionInput:
    """One design as a scoring function sees it."""

    index: int
    design_id: str
    sequence: str
    target_sequence: str | None = None
    structure: Path | None = None
    # Which model produced `structure`. Carried because a geometric metric read
    # off a pose belongs to the model that predicted that pose -- two models
    # disagree about where the binder sits by a median 22.7 A, and one
    # unprefixed column would hide it.
    source_model: str | None = None

    def as_row(self, wants: Iterable[str]) -> dict:
        """Only what the function asked for.

        A function declaring `inputs: [sequence]` is not handed a structure
        path, so it cannot come to depend on one without saying so -- which is
        what lets the runner know it can be applied to designs that were never
        folded.
        """
        wanted = set(wants)
        # No `design_id`. A driver runs inside a container, knows nothing about
        # the database, and joins back through `index` -- the design-set
        # position -- exactly as `DesignSetEntry` describes. Handing over an id
        # a script has no use for invites it to be echoed back as the join key,
        # and `docs/scoring-functions.md` already documents its absence.
        row: dict = {"index": self.index}
        if "sequence" in wanted:
            row["sequence"] = self.sequence
        if "target_sequence" in wanted and self.target_sequence is not None:
            row["target_sequence"] = self.target_sequence
        if "structure" in wanted and self.structure is not None:
            row["structure"] = str(self.structure)
        return row


@dataclass(frozen=True, slots=True)
class FunctionOutput:
    """One design's worth of results from a scoring function."""

    index: int
    metrics: dict[str, float]
    failed: str | None = None


def write_inputs(path: Path, rows: Iterable[FunctionInput], wants: Iterable[str]) -> int:
    """Write the input JSONL. Returns how many rows were written."""
    wanted = tuple(wants)
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row.as_row(wanted)) + "\n")
            written += 1
    return written


def read_outputs(path: Path) -> Iterator[FunctionOutput]:
    """Parse strict output rows, tolerating only a torn, unterminated last line.

    The caller counts that lost final row as missing output. Malformed complete
    rows are errors, not absent measurements that could imply a successful run.
    """
    if not path.is_file():
        return
    lines = path.read_text().splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines) and not line.endswith(("\n", "\r")):
                return
            raise ValueError(f"invalid JSON in output line {number}") from error
        if not isinstance(row, dict):
            raise TypeError(f"output line {number} must be an object")
        index = row.get("index")
        if type(index) is not int or index < 0:
            raise ValueError(f"output line {number} needs a nonnegative integer index")
        failed = row.get("failed")
        if failed is not None and (not isinstance(failed, str) or not failed.strip()):
            raise ValueError(f"output line {number} has an invalid failure reason")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError(f"output line {number} metrics must be an object")
        if failed is not None and metrics:
            raise ValueError(f"output line {number} cannot report metrics and failure")
        values: dict[str, float] = {}
        for key, value in metrics.items():
            if (
                not isinstance(key, str) or not key
                or isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"output line {number} has an invalid metric {key!r}")
            try:
                number_value = float(value)
            except OverflowError as error:
                raise ValueError(f"output line {number} has a nonfinite metric {key!r}") from error
            if not math.isfinite(number_value):
                raise ValueError(f"output line {number} has a nonfinite metric {key!r}")
            values[key] = number_value
        yield FunctionOutput(index=index, metrics=values, failed=failed)


def declared_but_absent(
    declared: Iterable[str], produced: Mapping[str, float]
) -> tuple[str, ...]:
    """Metrics a function promised and did not deliver for one design.

    Reported rather than filled in: a function that declares four metrics and
    returns three has either failed partially or drifted from its own config,
    and both are worth seeing.
    """
    return tuple(sorted(set(declared) - set(produced)))
