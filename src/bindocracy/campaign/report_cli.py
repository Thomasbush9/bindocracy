"""Command adapters for offline campaign reporting, exports and integrity audit."""

from pathlib import Path
from typing import Annotated

import duckdb
import typer

from bindocracy.campaign.report import (
    audit_campaign,
    campaign_report,
    export_campaign,
    report_text,
    write_report,
)
from bindocracy.cli_output import emit, fail


def register(app: typer.Typer) -> None:
    app.command("report")(report)
    app.command("export")(export)
    app.command("audit")(audit)


def report(
    database: Annotated[Path, typer.Argument(help="Existing campaign database (read-only).")],
    run: Annotated[
        list[str] | None,
        typer.Option("--run", help="Exact run ID or unambiguous name; repeatable."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", help="New .json report or human-readable text file; never overwritten."
        ),
    ] = None,
) -> None:
    """Report scientific outcomes, scoped metrics/replicas and decision cohorts."""
    try:
        result = campaign_report(database, runs=run or ())
        if output is not None:
            write_report(result, output)
        emit(result, text=report_text(result))
    except (OSError, ValueError, RuntimeError, duckdb.Error) as error:
        fail(error)


def export(
    database: Annotated[Path, typer.Argument(help="Existing campaign database (read-only).")],
    output: Annotated[
        Path, typer.Option("--output", help="New .csv or .parquet file; never overwritten.")
    ],
    run: Annotated[
        list[str] | None,
        typer.Option("--run", help="Owning run ID or unambiguous name; repeatable."),
    ] = None,
    table: Annotated[
        str,
        typer.Option(
            "--table",
            help="metrics (default), designs, decisions, artifacts, or runs. No replica reduction.",
        ),
    ] = "metrics",
) -> None:
    """Export a single long-form scientific table with run/config provenance."""
    try:
        result = export_campaign(database, output, runs=run or (), table=table)
        emit(result, text=f"Exported {result['rows']} {result['table']} rows to {result['output']}")
    except (OSError, ValueError, RuntimeError, duckdb.Error, ImportError) as error:
        fail(error)


def audit(
    plan: Annotated[
        Path, typer.Argument(help="Frozen plan.json; no scheduler queries or repairs.")
    ],
) -> None:
    """Compare immutable inputs, task artifacts, bundle hashes and actual DB rows."""
    try:
        result = audit_campaign(plan)
        lines = [f"Integrity: {result['integrity']} (scheduler not queried)"]
        lines.extend(
            f"{item['severity']}: {item['code']}: {item['message']} {item['path'] or item['run_id'] or ''}"
            for item in result["findings"]
        )
        emit(result, text="\n".join(lines))
    except (OSError, ValueError, RuntimeError, duckdb.Error) as error:
        fail(error)
    if result["integrity"] == "corrupt":
        raise typer.Exit(code=1)
