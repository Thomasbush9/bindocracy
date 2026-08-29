"""How one task ended, written by the tool or by the harness around it.

Two rules decide who writes `status.json`:

* A tool that knows more than its exit code writes its own. Mosaic's driver
  does, because only it can tell a walltime kill that kept nine designs
  (`partial`) from a crash that kept none (`failed`).
* Everything else gets one written for it. A tool that merely exits, like
  BoltzGen, would otherwise leave the workflow with no output to declare.

A status the tool wrote is never overwritten.

The second rule here is that **an ordinary tool failure is not a workflow
failure**. `docs/harness-design.md` §6 says success is "the expected artifacts
exist and parse", and the database is supposed to keep failed and partial runs
as campaign history. If a non-zero exit aborted the rule, Snakemake would never
run collection and the failure would exist only in a log. So a tool that runs
and fails is recorded and collection proceeds; only a harness or infrastructure
failure — one where we could not run the tool or could not record what happened
— is raised.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bindocracy.store.records import utc_now


class HarnessError(RuntimeError):
    """The harness could not run the tool, or could not record what happened.

    Distinct from the tool itself failing, which is recorded rather than raised.
    """


class TaskStatus(BaseModel):
    """One task's outcome, validated rather than passed around as a dict.

    Extra keys are kept: the tool owns this file and may record more than the
    harness knows to ask for. The fields below are the ones collection reads.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    task_id: int
    status: Literal["succeeded", "partial", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    n_attempted: int | None = Field(default=None, ge=0)
    n_produced: int | None = Field(default=None, ge=0)
    exit_code: int | None = None
    error: str | None = None
    output_file: str | None = None
    written_by: Literal["tool", "harness"] = "tool"


def read_task_status(path: str | Path) -> TaskStatus | None:
    """Read one status file. None means the task never wrote one."""
    status_path = Path(path)
    if not status_path.is_file():
        return None
    try:
        return TaskStatus.model_validate_json(status_path.read_text())
    except ValueError as error:
        raise HarnessError(f"invalid status file {status_path}: {error}") from error


def write_task_status(
    status_path: str | Path, status: TaskStatus, *, overwrite: bool = False
) -> bool:
    """Write `status.json` atomically. False means the tool already wrote one."""
    path = Path(status_path)
    if path.is_file() and not overwrite:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(status.model_dump_json(indent=2, exclude_none=False) + "\n")
        os.replace(tmp, path)
    except OSError as error:
        raise HarnessError(f"cannot record task status at {path}: {error}") from error
    return True


def run_task(
    argv: tuple[str, ...],
    env: dict[str, str],
    log_path: str | Path,
    status_path: str | Path,
    task_id: int,
    mkdirs: tuple[Path, ...] = (),
) -> int:
    """Run one task, tee its output to `log_path`, and record how it ended.

    Returns the tool's exit code. A non-zero code is recorded, not raised, so
    that collection still runs and the failed run reaches the database. Being
    unable to start the tool, or to write its log or status, raises
    `HarnessError` instead — that is the harness failing, not the tool.
    """
    log = Path(log_path)
    started_at = utc_now()
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        for directory in mkdirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            completed = subprocess.run(
                argv, env={**os.environ, **env},
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
    except OSError as error:
        write_task_status(status_path, TaskStatus(
            task_id=task_id, status="failed", started_at=started_at,
            finished_at=utc_now(), error=f"{type(error).__name__}: {error}",
            written_by="harness",
        ))
        raise HarnessError(f"could not run task {task_id}: {error}") from error

    write_task_status(status_path, TaskStatus(
        task_id=task_id,
        status="succeeded" if completed.returncode == 0 else "failed",
        started_at=started_at,
        finished_at=utc_now(),
        exit_code=completed.returncode,
        error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
        written_by="harness",
    ))
    return completed.returncode


def harness_status(task_id: int, status: str, **fields: Any) -> TaskStatus:
    """Build a harness-authored status; convenience for callers and tests."""
    return TaskStatus(task_id=task_id, status=status, written_by="harness", **fields)
