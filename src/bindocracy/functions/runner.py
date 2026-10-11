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
import os
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
IO_HELPER = Path(__file__).resolve().parents[3] / "drivers" / "bindocracy_io.py"


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
    helper_sha256: str | None = None
    seconds: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)
    incomplete: dict[int, tuple[str, ...]] = field(default_factory=dict)
    process_error: str | None = None
    output_present: bool = False

    @property
    def summary(self) -> dict:
        return {
            "function": self.name,
            "n_inputs": self.n_inputs,
            "n_scored": self.n_scored,
            "n_metric_rows": len(self.records),
            "script_sha256": self.script_sha256,
            "helper_sha256": self.helper_sha256,
            "seconds": round(self.seconds, 2),
            "failures": dict(self.failures),
            "n_incomplete": len(self.incomplete),
            "process_error": self.process_error,
            "output_present": self.output_present,
        }


def command_for(
    function: ScoringFunction, inputs: Path, outputs: Path,
    *, bind_paths: tuple[Path, ...] = (), gpus: int = 0,
    dependency_bindings: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The argv for one custom function.

    The script is always passed `--inputs` and `--outputs`, in that order,
    before any of its own `args`. A function that wants options gets them; a
    function that wants none does not have to parse anything else.
    """
    if function.container is not None:
        # The image's own interpreter, by the name it has inside.
        bindings = tuple(
            part for path in dict.fromkeys((inputs.parent.resolve(), function.script.parent.resolve(), *bind_paths))
            for part in ("--bind", str(path))
        )
        dependency_args = tuple(part for binding in dependency_bindings for part in ("--bind", binding))
        dev_args = ("--bind", f"{function.dev_source}:/opt/mosaic/src/mosaic:ro") if function.dev_source else ()
        prefix = (
            "singularity", "exec", "--cleanenv", *(("--nv",) if gpus else ()),
            *bindings, *dependency_args, *dev_args,
            str(function.container), function.container_python,
        )
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
    if getattr(function, "kit_lock", None) is not None:
        from bindocracy.kits.scoring import validate_function_lock

        validate_function_lock(function)
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
    prepared: bool = False,
    retain_partial: bool = False,
    gpus: int = 0,
    dependency_bindings: tuple[str, ...] = (),
) -> FunctionResult:
    """Run one custom function over one set of designs.

    Metrics are stored as `<function name>_<metric>`, so two functions may
    compute a `score` without colliding and the stored column always says which
    function produced it -- the same reason a model's metrics carry its name.
    """
    preflight_custom(function)
    if len({row.index for row in inputs}) != len(inputs):
        raise FunctionError("scoring inputs contain duplicate indices")
    if Path(function.script).name == IO_HELPER.name:
        raise FunctionError(f"scoring script cannot be named {IO_HELPER.name}")

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    # Prepared workers consume the plan's immutable archive, never live sources.
    archived_function, helper_path = (
        (function, function.script.parent / IO_HELPER.name)
        if prepared else archive_function(function, work_dir)
    )
    digest = sha256_file(archived_function.script)
    helper_digest = sha256_file(helper_path)
    if not prepared and getattr(function, "kit_lock", None) is not None:
        from bindocracy.kits.scoring import validate_function_lock

        dependency_bindings = (*dependency_bindings, *kit_bindings(
            validate_function_lock(function), work_dir / "kit",
        ))

    input_path = work_dir / INPUT_FILE
    output_path = work_dir / OUTPUT_FILE
    output_path.unlink(missing_ok=True)
    wanted = [row for row in inputs if _has_required_inputs(row, function)]
    skipped = len(inputs) - len(wanted)
    n_written = write_inputs(input_path, wanted, function.inputs)

    started = time.time()
    failures: dict[str, int] = {}
    process_error = None
    if skipped:
        # A function asking for a structure cannot score a design that has
        # none. Counted rather than silently dropped.
        failures["missing_required_input"] = skipped

    if n_written:
        try:
            completed = subprocess.run(
                command_for(
                    archived_function, input_path, output_path,
                    bind_paths=tuple(
                        row.structure.resolve().parent for row in wanted
                        if row.structure is not None
                    ),
                    gpus=gpus, dependency_bindings=dependency_bindings,
                ),
                capture_output=True, text=True, check=False,
                timeout=function.timeout_seconds,
                cwd=archived_function.script.parent,
                env={
                    **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                    "SINGULARITYENV_PYTHONDONTWRITEBYTECODE": "1",
                    **({"SINGULARITYENV_CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}
                       if gpus and "CUDA_VISIBLE_DEVICES" in os.environ else {}),
                    **({"CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
                        "SINGULARITYENV_CUDA_VISIBLE_DEVICES": "",
                        "SINGULARITYENV_ROCR_VISIBLE_DEVICES": ""} if not gpus else {}),
                },
            )
        except subprocess.TimeoutExpired as exc:
            (work_dir / "stdout.log").write_text(_output_text(exc.stdout))
            (work_dir / "stderr.log").write_text(_output_text(exc.stderr))
            if not retain_partial:
                raise FunctionError(
                    f"custom function {function.name!r} exceeded {function.timeout_seconds}s"
                ) from exc
            failures["timeout"] = 1
            process_error = f"custom function {function.name!r} exceeded {function.timeout_seconds}s"
        else:
            (work_dir / "stdout.log").write_text(completed.stdout)
            (work_dir / "stderr.log").write_text(completed.stderr)
            if completed.returncode != 0:
                if not retain_partial:
                    raise FunctionError(
                        f"custom function {function.name!r} exited {completed.returncode}: "
                        f"{completed.stderr.strip()[-400:]}"
                    )
                failures[f"exit_{completed.returncode}"] = 1
                process_error = (
                    f"custom function {function.name!r} exited {completed.returncode}: "
                    f"{completed.stderr.strip()[-400:]}"
                )

    by_index = {row.index: row for row in wanted}
    specs = function.specs
    records: list[MetricRecord] = []
    incomplete: dict[int, tuple[str, ...]] = {}
    scored: set[int] = set()

    seen: set[int] = set()
    try:
        for output in read_outputs(output_path):
            source = by_index.get(output.index)
            if source is None:
                raise FunctionError(f"unknown design index {output.index} in scoring output")
            if output.index in seen:
                raise FunctionError(f"duplicate design index {output.index} in scoring output")
            seen.add(output.index)
            details = {
                "function": function.name, "script_sha256": digest,
                "helper_sha256": helper_digest,
            }
            if output.failed is not None:
                reason = output.failed.split(":")[0][:60]
                failures[reason] = failures.get(reason, 0) + 1
                details["failure_reason"] = output.failed
                values = dict.fromkeys(specs)
            else:
                missing = declared_but_absent(specs, output.metrics)
                if missing:
                    incomplete[output.index] = missing
                unknown = sorted(set(output.metrics) - set(specs))
                if unknown:
                    raise FunctionError(
                        f"custom function {function.name!r} returned undeclared metric(s) "
                        f"{unknown} for design index {output.index}. Declare them in the "
                        "config with a direction, or stop emitting them."
                    )
                if not output.metrics:
                    continue
                values = output.metrics
                scored.add(output.index)
            records.extend(
                metric_records(
                    run_id=run_id,
                    design_id=source.design_id,
                    model=_prefix_for(function, source),
                    values=values,
                    replicate=replicate,
                    measured_at=measured_at,
                    details=details,
                    specs=specs,
                )
            )
    except (TypeError, ValueError) as error:
        raise FunctionError(f"custom function {function.name!r}: {error}") from error
    if missing_outputs := len(wanted) - len(seen):
        failures["missing_output"] = failures.get("missing_output", 0) + missing_outputs

    return FunctionResult(
        name=function.name,
        records=tuple(records),
        n_inputs=len(inputs),
        n_scored=len(scored),
        script_sha256=digest,
        helper_sha256=helper_digest,
        seconds=time.time() - started,
        failures=failures,
        incomplete=incomplete,
        process_error=process_error,
        output_present=output_path.is_file(),
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


def _output_text(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def archive_function(function: ScoringFunction, work_dir: Path):
    """Copy the complete verified kit, preserving sibling imports and assets."""
    lock = None
    if getattr(function, "kit_lock", None) is not None:
        from bindocracy.kits.scoring import validate_function_lock

        lock = validate_function_lock(function)
    if lock is None:
        script = work_dir / function.script.name
        if script != function.script.resolve():
            shutil.copyfile(function.script, script)
    else:
        root = work_dir / "kit"
        root.mkdir(parents=True, exist_ok=False)
        for relative, expected in lock.files.items():
            source = lock.kit_root / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256_file(destination) != expected:
                raise FunctionError(f"kit file changed while staging: {relative}")
        script = root / lock.manifest.entrypoint
    helper = script.parent / IO_HELPER.name
    if helper.is_file() and sha256_file(helper) != sha256_file(IO_HELPER):
        raise FunctionError("kit supplies a conflicting bindocracy_io.py")
    if helper.resolve() != IO_HELPER.resolve():
        shutil.copyfile(IO_HELPER, helper)
    return function.model_copy(update={"script": script, "kit_lock": None}), helper


def kit_bindings(lock, root: Path) -> tuple[str, ...]:
    """Read-only container mounts for the bundle and its verified dependencies."""
    pairs = {(str(root), str(root))}
    for dependency in lock.dependencies.values():
        source = dependency["path"]
        if dependency["kind"] == "harness_module":
            paths = [str(Path(source) / name) for name in dependency["identity"]["files"]]
        else:
            paths = [source]
        pairs.update((path, path) for path in paths)
        if dependency.get("target"):
            pairs.add((source, dependency["target"]))
    for source, target in pairs:
        if any(character in source + target for character in (":", ",", "\n")):
            raise FunctionError(f"dependency path cannot be bound read-only: {source!r}")
    return tuple(f"{source}:{target}:ro" for source, target in sorted(pairs))


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
