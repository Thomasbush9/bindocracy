"""CLI for explicitly approved, bounded deployment qualification."""

from pathlib import Path
from typing import Annotated

import typer

from bindocracy.campaign.qualify import qualify
from bindocracy.cli_output import emit, fail


def register(app: typer.Typer) -> None:
    app.command("qualify")(qualify_command)


def qualify_command(
    plan: Annotated[Path, typer.Argument(help="Fresh, frozen canary plan.json.")],
    approve: Annotated[
        str | None, typer.Option(help="Exact reviewed plan digest; omission is read-only.")
    ] = None,
    mode: Annotated[str, typer.Option(help="completion or cancel-resume.")] = "completion",
    allow_gpu: Annotated[
        bool, typer.Option(help="Explicitly permit the bounded GPU canary policy.")
    ] = False,
    timeout: Annotated[
        float, typer.Option(help="Overall polling deadline, at most 1800 seconds.")
    ] = 900,
    poll_interval: Annotated[float, typer.Option(help="Polling interval, 1–30 seconds.")] = 5,
) -> None:
    """Qualify a small canary; never submit without its exact approval digest."""
    try:
        report = qualify(
            plan,
            approve,
            mode=mode,
            allow_gpu=allow_gpu,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    except (RuntimeError, ValueError, OSError) as error:
        fail(error)
    emit(report)
    # A machine-readable report remains available even for incomplete qualification.
    if report["outcome"] != "passed":
        raise typer.Exit(code=1)
