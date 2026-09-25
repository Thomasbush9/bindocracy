"""CLI adapter for the read-only campaign readiness report."""

from pathlib import Path
from typing import Annotated

import typer

from bindocracy.campaign.check import check_campaign
from bindocracy.cli_output import emit, fail


def register(app: typer.Typer) -> None:
    @app.command("check")
    def check(
        index: Annotated[
            Path, typer.Argument(help="Workflow index YAML; paths use the current directory.")
        ],
        site: Annotated[
            Path | None, typer.Option("--site", help="Controller and campaign budget policy.")
        ] = None,
        probe_site: Annotated[
            bool, typer.Option("--probe-site", help="Bounded, read-only Slurm queries.")
        ] = False,
    ) -> None:
        """Check every run without planning, reserving directories or submitting jobs."""
        try:
            report = check_campaign(index, site, probe_site=probe_site)
        except (ValueError, RuntimeError, OSError) as error:
            fail(error)
        emit(report)
        if report["status"] != "ready":
            raise typer.Exit(1)
