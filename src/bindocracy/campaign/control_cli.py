"""CLI adapters for approval-gated campaign lifecycle operations."""

from pathlib import Path
from typing import Annotated

import typer

from bindocracy.campaign import control
from bindocracy.cli_output import emit, fail

app = typer.Typer(help="Manage approved frozen campaign executions.")


def _invoke(operation, plan, **kwargs):
    try:
        result = operation(plan, **kwargs)
    except (RuntimeError, ValueError, OSError) as error:
        fail(error, code="campaign_error")
    emit(result)


@app.command("submit")
def submit(
    plan: Annotated[Path, typer.Argument(help="Frozen plan.json to execute.")],
    approve: Annotated[str, typer.Option("--approve", help="Exact sha256-prefixed plan digest.")],
) -> None:
    """Submit one controller; repeated submission never creates a duplicate."""
    _invoke(control.submit, plan, approve=approve)


@app.command("status")
def status(plan: Annotated[Path, typer.Argument(help="Frozen plan.json to inspect.")]) -> None:
    """Reconcile scheduler identities and report task artifacts without changing metrics."""
    _invoke(control.status, plan)


@app.command("cancel")
def cancel(plan: Annotated[Path, typer.Argument(help="Frozen plan.json to stop.")]) -> None:
    """Stop further submissions, then cancel only this plan's tagged allocations."""
    _invoke(control.cancel, plan)


@app.command("resume")
def resume(
    plan: Annotated[Path, typer.Argument(help="Frozen plan.json to resume.")],
    approve: Annotated[str, typer.Option("--approve", help="Exact sha256-prefixed plan digest.")],
) -> None:
    """Resume an inactive attempt, retaining completed outputs and frozen manifests."""
    _invoke(control.resume, plan, approve=approve)
