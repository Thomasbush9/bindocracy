"""Record how one task ended, for tools that do not report it themselves.

Mosaic's driver writes its own `status.json`, because only it can tell a
walltime kill that kept nine designs (partial) from a crash that produced none
(failed). Most tools just exit, so the harness has to record the outcome around
the process instead.

A status file the tool wrote itself is never overwritten: the tool knows more
than the exit code does.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from bindocracy.store.records import utc_now


def write_task_status(
    status_path: str | Path,
    task_id: int,
    *,
    started_at: str,
    status: str,
    exit_code: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Write `status.json` atomically. False means the tool already wrote one."""
    path = Path(status_path)
    if path.is_file():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now().isoformat(),
        "exit_code": exit_code,
        "error": error,
        "written_by": "harness",
        **(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)
    return True


def run_task(
    argv: tuple[str, ...],
    env: dict[str, str],
    log_path: str | Path,
    status_path: str | Path,
    task_id: int,
) -> int:
    """Run one task, tee its output to `log_path`, and record its status.

    Raises on a non-zero exit so the failure is loud in Snakemake, after the
    status file has been written. Native output is never a declared workflow
    output, so whatever the task produced before dying survives either way.
    """
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    # Node-local TMPDIR and per-tool caches have to exist before the container
    # starts; the tool's runscript may mkdir under them but not create them.
    for key in ("TMPDIR", "SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE"):
        if key in env:
            Path(env[key]).mkdir(parents=True, exist_ok=True)

    started_at = utc_now().isoformat()
    with log.open("w") as stream:
        completed = subprocess.run(
            argv, env={**os.environ, **env},
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )

    write_task_status(
        status_path,
        task_id,
        started_at=started_at,
        status="succeeded" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    return completed.returncode
