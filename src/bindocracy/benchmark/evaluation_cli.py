"""Organizer-only held-out grading and roster-complete aggregation."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from bindocracy.benchmark.evaluation import aggregate_reports, evaluate_submission
from bindocracy.benchmark.submission import _atomic_json, _json, _read
from bindocracy.cli_output import emit, fail

app = typer.Typer(help="Organizer held-out evaluation and task-balanced aggregation.")


@app.command("evaluate")
def evaluate(attempt_dir: Annotated[Path, typer.Argument()],
             protocol: Annotated[Path, typer.Option(help="Trusted organizer protocol JSON.")],
             output: Annotated[Path, typer.Option()]) -> None:
    """Grade the latest accepted immutable bundle with an organizer-owned command."""
    try:
        emit(evaluate_submission(attempt_dir, protocol, output))
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="evaluation_error")


@app.command("aggregate")
def aggregate(roster: Annotated[Path, typer.Argument(help="JSON list of ALL attempt.json paths, relative to this file.")],
              protocol: Annotated[Path, typer.Option()],
              reports: Annotated[list[Path] | None, typer.Option("--report")] = None,
              output: Annotated[Path | None, typer.Option()] = None,
              baseline: Annotated[str | None, typer.Option()] = None,
              bootstrap_samples: Annotated[int, typer.Option(min=1)] = 2000,
              seed: Annotated[int, typer.Option()] = 0) -> None:
    """Include every roster slot; absent/failed results retain declared failure scores."""
    try:
        payload = _json(_read(roster))
        if not isinstance(payload, list) or not all(isinstance(path, str) for path in payload):
            raise ValueError("roster must be a JSON list of attempt record paths")
        attempts = [Path(path) if Path(path).is_absolute() else roster.parent / path for path in payload]
        result = aggregate_reports(reports or [], attempts=attempts, protocol=protocol,
                                   bootstrap_samples=bootstrap_samples, seed=seed, baseline=baseline)
        if output is not None:
            if output.exists():
                raise ValueError("aggregate output already exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(output, result)
        emit(result)
    except (ValueError, RuntimeError, OSError) as error:
        fail(error, code="evaluation_error")
