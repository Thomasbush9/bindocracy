"""Command-line entry point.

A placeholder so the `bindocracy` console script declared in pyproject.toml
resolves against a real callable. Subcommands get added as the harness lands.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bindocracy import __version__
from bindocracy.config import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigPreflightError,
    load_mosaic_configs,
    recover_config_yaml,
)
from bindocracy.store import CampaignStore, create_database

app = typer.Typer(
    add_completion=False,
    help="Run many binder-design models from one place.",
)
config_app = typer.Typer(help="Validate and load campaign configuration.")
app.add_typer(config_app, name="config")


@app.callback()
def main() -> None:
    """Keep Typer in subcommand mode.

    Without a callback, Typer collapses a single-command app into the top-level
    command, so `bindocracy version` would be parsed as a stray argument.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command("init-db")
def init_db(
    path: Path,
    if_not_exists: bool = typer.Option(
        False, "--if-not-exists", help="Accept an already existing database."
    ),
) -> None:
    """Create an initialized empty database for one target campaign."""
    try:
        database_path = create_database(path, exist_ok=if_not_exists)
    except FileExistsError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error
    typer.echo(database_path)


@config_app.command("load-mosaic")
def load_mosaic_config(
    database: Path,
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    model: Annotated[Path, typer.Option("--model", help="Mosaic model YAML.")],
) -> None:
    """Validate general and Mosaic configs and add only their config rows."""
    try:
        loaded = load_mosaic_configs(general, model)
    except (ConfigLoadError, ConfigPreflightError, ValidationError) as error:
        typer.echo(f"Configuration error:\n{error}", err=True)
        raise typer.Exit(code=2) from error

    record = loaded.to_record()
    with CampaignStore.create(database, exist_ok=True) as store:
        inserted = set(store.add_configs([record]))

    typer.echo(f"database: {database}")
    typer.echo(f"target: {loaded.general.target.name} ({loaded.preflight.target_length} aa)")
    status = "inserted" if record.model_config_id in inserted else "already present"
    typer.echo(f"general: {record.general_name} {record.general_config_id}")
    typer.echo(f"model: {record.model_name} {record.model_config_id} [{status}]")


@config_app.command("export")
def export_config(
    database: Path,
    config_id: str,
    output: Path,
    force: Annotated[bool, typer.Option("--force", help="Overwrite OUTPUT if it exists.")] = False,
) -> None:
    """Recover a general or model YAML file from its database JSON."""
    try:
        output_path = recover_config_yaml(
            database, config_id, output, overwrite=force
        )
    except ConfigNotFoundError as error:
        typer.echo(f"Configuration ID not found: {config_id}", err=True)
        raise typer.Exit(code=2) from error
    except FileExistsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(output_path)


if __name__ == "__main__":
    app()
