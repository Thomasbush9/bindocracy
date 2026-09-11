"""Run scoring functions over designs the scorer already folded.

No GPU, no model load for the built-ins: a function reads a pose that is
already on disk, or just a sequence. That is what lets a new metric be
backfilled across every structure ever saved rather than requiring everything
to be folded again -- which was the state of things until structures started
being kept on 2026-09-09.

A custom function is an external program, so it gets the same treatment the
harness gives any other external program: its bytes are hashed and recorded, it
is refused before it runs if it cannot work, and what it returns is checked
against what it promised.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from bindocracy.adapters.scoring import MetricSpec, metric_records
from bindocracy.config.load import sha256_file
from bindocracy.functions.contract import (
    FunctionInput,
    FunctionOutput,
    declared_but_absent,
    read_outputs,
    write_inputs,
)
from bindocracy.functions.models import ScoringFunction
from bindocracy.store.records import MetricRecord

INPUT_FILE = "inputs.jsonl"
OUTPUT_FILE = "metrics.jsonl"


class FunctionError(RuntimeError):
    """A scoring function could not be run, or could not be trusted."""


@dataclass
class FunctionResult:
    """What one function produced, and what it failed to produce."""

    name: str
    records: tuple[MetricRecord, ...]
    n_inputs: int
    n_scored: int
    script_sha256: str | None = None
    seconds: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)
    incomplete: dict[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def summary(self) -> dict:
        return {
            "function": self.name,
            "n_inputs": self.n_inputs,
            "n_scored": self.n_scored,
            "n_metric_rows": len(self.records),
            "script_sha256": self.script_sha256,
            "seconds": round(self.seconds, 2),
            "failures": dict(self.failures),
            "n_incomplete": len(self.incomplete),
        }


def command_for(function: ScoringFunction, inputs: Path, outputs: Path) -> tuple[str, ...]:
    """The argv for one custom function.

    The script is always passed `--inputs` and `--outputs`, in that order,
    before any of its own `args`. A function that wants options gets them; a
    function that wants none does not have to parse anything else.
    """
    if function.container is not None:
        # The image's own interpreter, by the name it has inside.
        prefix = ("singularity", "exec", "--cleanenv", str(function.container), "python")
    else:
        # `sys.executable`, not "python": there is no bare `python` on this
        # cluster's PATH, and a host function should run under the harness's
        # own interpreter so it sees the dependencies the harness installed.
        prefix = (sys.executable,)
    return (
        *prefix, str(function.script),
        "--inputs", str(inputs), "--outputs", str(outputs),
        *function.args,
    )


def preflight_custom(function: ScoringFunction) -> str:
    """Refuse a function that cannot work, and hash the bytes that will run.

    The digest is the point: a run has to record which *bytes* scored it, not
    which path they were read from. `runs.workflow_metadata` keeps it, so a
    script edited between two runs cannot make them look comparable.
    """
    script = Path(function.script)
    if not script.is_file():
        raise FunctionError(
            f"custom function {function.name!r}: script not found at {script}"
        )
    if function.container is not None and not Path(function.container).exists():
        raise FunctionError(
            f"custom function {function.name!r}: container not found at "
            f"{function.container}"
        )
    return sha256_file(script)


def run_custom(
    function: ScoringFunction,
    inputs: list[FunctionInput],
    *,
    run_id: str,
    work_dir: Path,
    measured_at: datetime,
    replicate: int = 0,
) -> FunctionResult:
    """Run one custom function over one set of designs.

    Metrics are stored as `<function name>_<metric>`, so two functions may
    compute a `score` without colliding and the stored column always says which
    function produced it -- the same reason a model's metrics carry its name.
    """
    digest = preflight_custom(function)

    work_dir.mkdir(parents=True, exist_ok=True)
    # Archive the script beside its own output. The config records a path; this
    # records the file, so the run stays readable after the original moves.
    shutil.copyfile(function.script, work_dir / Path(function.script).name)

    input_path = work_dir / INPUT_FILE
    output_path = work_dir / OUTPUT_FILE
    wanted = [row for row in inputs if _has_required_inputs(row, function)]
    skipped = len(inputs) - len(wanted)
    n_written = write_inputs(input_path, wanted, function.inputs)

    started = time.time()
    failures: dict[str, int] = {}
    if skipped:
        # A function asking for a structure cannot score a design that has
        # none. Counted rather than silently dropped.
        failures["missing_required_input"] = skipped

    if n_written:
        try:
            completed = subprocess.run(
                command_for(function, input_path, output_path),
                capture_output=True, text=True, check=False,
                timeout=function.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise FunctionError(
                f"custom function {function.name!r} exceeded "
                f"{function.timeout_seconds}s"
            ) from exc
        if completed.returncode != 0:
            raise FunctionError(
                f"custom function {function.name!r} exited {completed.returncode}: "
                f"{completed.stderr.strip()[-400:]}"
            )
        (work_dir / "stdout.log").write_text(completed.stdout)
        (work_dir / "stderr.log").write_text(completed.stderr)

    by_index = {row.index: row for row in wanted}
    specs = function.specs
    records: list[MetricRecord] = []
    incomplete: dict[int, tuple[str, ...]] = {}
    scored: set[int] = set()

    for output in read_outputs(output_path):
        source = by_index.get(output.index)
        if source is None:
            failures["unknown_index"] = failures.get("unknown_index", 0) + 1
            continue
        if output.failed:
            reason = output.failed.split(":")[0][:60]
            failures[reason] = failures.get(reason, 0) + 1
            continue
        missing = declared_but_absent(specs, output.metrics)
        if missing:
            incomplete[output.index] = missing
        unknown = sorted(set(output.metrics) - set(specs))
        if unknown:
            raise FunctionError(
                f"custom function {function.name!r} returned undeclared metric(s) "
                f"{unknown} for design index {output.index}. Declare them in the "
                "config with a direction, or stop emitting them -- a metric whose "
                "direction nothing records sorts backwards silently."
            )
        if not output.metrics:
            continue
        records.extend(
            metric_records(
                run_id=run_id,
                design_id=source.design_id,
                model=_prefix_for(function, source),
                values=dict(output.metrics),
                replicate=replicate,
                measured_at=measured_at,
                details={"function": function.name, "script_sha256": digest},
                specs=specs,
            )
        )
        scored.add(output.index)

    return FunctionResult(
        name=function.name,
        records=tuple(records),
        n_inputs=len(inputs),
        n_scored=len(scored),
        script_sha256=digest,
        seconds=time.time() - started,
        failures=failures,
        incomplete=incomplete,
    )


def _prefix_for(function: ScoringFunction, row: FunctionInput) -> str:
    """What the stored metric name is prefixed with.

    `function` for anything whose value does not depend on which model produced
    the input -- a binder's net charge is its net charge. `source_model` for
    geometry read off a particular pose, so `boltz2_epitope_coverage` and
    `chai1_epitope_coverage` stay separate columns. A pose whose model is
    unrecorded falls back to the function name rather than inventing one.
    """
    if function.prefix == "source_model" and row.source_model:
        return row.source_model
    return function.name


def _has_required_inputs(row: FunctionInput, function: ScoringFunction) -> bool:
    if "structure" in function.inputs and row.structure is None:
        return False
    return not ("target_sequence" in function.inputs and row.target_sequence is None)


def write_summary(path: Path, results: list[FunctionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.summary for r in results], indent=2) + "\n")


__all__ = [
    "FunctionError",
    "FunctionOutput",
    "FunctionResult",
    "MetricSpec",
    "command_for",
    "preflight_custom",
    "run_custom",
    "write_summary",
]
