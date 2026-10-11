"""Candidate bundle commands; acceptance is an organizer operation or broker RPC."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from bindocracy.benchmark.submission import (
    accept_submission,
    archive_evidence,
    build_submission,
    check_submission,
)
from bindocracy.cli_output import emit, fail

app = typer.Typer(help="Build and validate portable score-free candidate submissions.")


@app.command("build")
def build(database: Annotated[Path, typer.Argument()], output: Annotated[Path, typer.Argument()],
          design_set: Annotated[Path | None, typer.Option()] = None,
          workspace: Annotated[Path | None, typer.Option()] = None,
          structures_from: Annotated[str | None, typer.Option()] = None) -> None:
    """Freeze real DB candidates, lineage, target and complete kit source bytes."""
    try:
        emit(build_submission(database, output, design_set=design_set, workspace=workspace,
                              structures_from=structures_from))
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="submission_error")


@app.command("check")
def check(path: Annotated[Path, typer.Argument()]) -> None:
    """Check integrity and membership; this never validates binding or scores."""
    try:
        manifest = check_submission(path)
        emit({"valid": True, "scientific_validation": False, "manifest": manifest})
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="submission_error")


@app.command("accept")
def accept(source: Annotated[Path, typer.Argument()],
           attempt_dir: Annotated[Path | None, typer.Option(help="Organizer control directory; agents use BINDOCRACY_BROKER.")] = None) -> None:
    """Request deadline-bound organizer acceptance, directly or through the broker."""
    try:
        broker = os.environ.get("BINDOCRACY_BROKER")
        if broker:
            if attempt_dir is not None:
                raise ValueError("agent broker acceptance cannot choose an attempt directory")
            from bindocracy.benchmark.runner import _broker_request
            result = _broker_request(broker, "accept", source=str(source.absolute()))
        else:
            if attempt_dir is None:
                raise ValueError("organizer acceptance requires --attempt-dir (or an authenticated agent broker)")
            result = accept_submission(source, attempt_dir)
        emit(result)
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="submission_error")


@app.command("archive-evidence")
def archive(attempt_dir: Annotated[Path, typer.Argument()], output: Annotated[Path, typer.Argument()],
            workspace: Annotated[Path | None, typer.Option()] = None,
            evaluations: Annotated[list[Path] | None, typer.Option("--evaluation")] = None) -> None:
    """Archive the stopped experiment separately from its fixed candidate set."""
    try:
        emit(archive_evidence(attempt_dir, output, workspace=workspace, evaluations=tuple(evaluations or ())))
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="submission_error")
