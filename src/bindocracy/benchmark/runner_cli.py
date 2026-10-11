"""Organizer-owned attempt runner CLI; local smoke is explicitly unisolated."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from bindocracy.benchmark import runner
from bindocracy.cli_output import emit, fail

app = typer.Typer(help="Run, recover and inspect externally managed benchmark attempts.")


def _invoke(operation, *args, **kwargs):
    try:
        result = operation(*args, **kwargs)
    except (RuntimeError, OSError, ValueError) as error:
        fail(error, code="benchmark_error")
    emit(result)


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    workspace: Annotated[Path, typer.Argument(help="Initialized workspace root or workspace.json.")],
    attempt_dir: Annotated[Path, typer.Argument(help="New organizer-owned directory outside workspace.")],
    command: Annotated[list[str], typer.Argument(help="Agent argv after --; no shell interpolation.")],
    development: Annotated[bool, typer.Option("--development", help="Explicit unisolated local CPU mode; Slurm disabled.")] = False,
    agent_name: Annotated[str, typer.Option("--agent-name")] = "external-agent",
    repeat: Annotated[int | None, typer.Option("--repeat", min=0)] = None,
    agent_uid: Annotated[int | None, typer.Option("--agent-uid", min=0)] = None,
    isolation: Annotated[Path | None, typer.Option("--isolation", help="Organizer JSON prerequisite attestation.")] = None,
    launcher: Annotated[Path | None, typer.Option("--launcher", help="JSON argv prefix for preprovisioned UID launcher.")] = None,
    broker_socket: Annotated[Path | None, typer.Option("--broker-socket")] = None,
) -> None:
    """Launch and supervise an external autonomous command until exit or cutoff."""
    try:
        prerequisites = json.loads(isolation.read_text()) if isolation else None
        prefix = json.loads(launcher.read_text()) if launcher else []
        if not isinstance(prefix, list) or any(not isinstance(v, str) or not v for v in prefix):
            raise ValueError("launcher must be a JSON array of nonempty argv strings")
    except (OSError, ValueError) as error:
        fail(error, code="benchmark_error")
    metadata = {"name": agent_name}
    if repeat is not None:
        metadata["repeat"] = repeat
    _invoke(runner.run_attempt, workspace, attempt_dir, command,
            development=development, agent_uid=agent_uid, isolation=prerequisites,
            agent=metadata, launcher=prefix, broker_socket=broker_socket)


@app.command("status")
def status(
    attempt_dir: Path,
    reconcile: Annotated[bool, typer.Option("--reconcile", help="Query terminal allocation accounting; may contact Slurm.")] = False,
) -> None:
    _invoke(runner.status, attempt_dir, reconcile=reconcile)


@app.command("cancel")
def cancel(attempt_dir: Path) -> None:
    """Close gates, kill the process group and cancel exact owned scheduler tags."""
    _invoke(runner.cancel, attempt_dir)


@app.command("recover")
def recover(attempt_dir: Path) -> None:
    """Close and clean up after a dead runner; never relaunch an uncertain attempt."""
    _invoke(runner.recover, attempt_dir)


@app.command("usage")
def usage(attempt_dir: Path, record: Path) -> None:
    """Ingest a provider-reported usage JSON record (missing quantities stay unknown)."""
    try:
        value = json.loads(record.read_text())
    except (OSError, ValueError) as error:
        fail(error, code="benchmark_error")
    socket_path = os.environ.get("BINDOCRACY_BROKER")
    if socket_path:
        _invoke(runner._broker_request, socket_path, "usage", record=value)
    else:
        _invoke(runner.record_usage, attempt_dir, value)


@app.command("smoke")
def smoke(workspace: Path, attempt_dir: Path) -> None:
    """Launch a genuine short CPU subprocess; NOT isolation or Slurm qualification."""
    command = [sys.executable, "-c", "import os,sys; print('benchmark local CPU smoke'); print(os.environ['BINDOCRACY_ATTEMPT_ID']); print('stderr captured', file=sys.stderr)"]
    _invoke(runner.run_attempt, workspace, attempt_dir, command,
            development=True, agent={"name": "local-cpu-smoke"})
