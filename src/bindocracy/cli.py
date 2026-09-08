"""Command-line entry point.

A placeholder so the `bindocracy` console script declared in pyproject.toml
resolves against a real callable. Subcommands get added as the harness lands.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bindocracy import __version__
from bindocracy.config import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigPreflightError,
    recover_config_yaml,
)
from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig, ResourceConfig
from bindocracy.runs import ingest_bundle, write_collected
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.msa import prepare_target_msa
from bindocracy.store import (
    CampaignStore,
    IngestConflictError,
    TargetMismatchError,
    create_database,
)
from bindocracy.tools import UnknownToolError, collect_run, load_configs
from bindocracy.tools.boltzgen.migrate import backfill_boltzgen_decisions

app = typer.Typer(
    add_completion=False,
    help="Run many binder-design models from one place.",
)
config_app = typer.Typer(help="Validate and load campaign configuration.")
app.add_typer(config_app, name="config")
target_app = typer.Typer(help="Prepare shared target inputs before planning runs.")
app.add_typer(target_app, name="target")


@target_app.command("prepare-msa")
def prepare_msa(
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    script: Annotated[Path, typer.Option(
        "--script", help="Imported Mosaic singularity/msa-search.sbatch from ProtForge.",
    )],
    image: Annotated[Path, typer.Option("--image", help="ProtForge msa.sif image.")],
    database: Annotated[Path, typer.Option("--database", help="Local ColabFold/MMseqs2 database.")],
    cpus: Annotated[int, typer.Option("--cpus", min=1)] = 8,
    memory_gb: Annotated[int, typer.Option("--memory-gb", min=1)] = 64,
    walltime: Annotated[str, typer.Option("--walltime")] = "08:00:00",
) -> None:
    """Generate target.msa on SLURM and wait, or reuse a matching existing A3M.

    Run before config load or Snakemake; no GPU scoring job is submitted here.
    Account and partition come from the campaign's cluster configuration.
    """
    try:
        config = load_yaml(general, GeneralConfig)
        output = prepare_target_msa(
            config, script=script, image=image, database=database,
            resources=ResourceConfig(gpus=1, cpus=cpus, memory_gb=memory_gb, walltime=walltime),
        )
    except (ConfigLoadError, ConfigPreflightError, ValidationError, OSError) as error:
        typer.echo(f"Target preparation error:\n{error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(output)


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


@config_app.command("load")
def load_config(
    database: Path,
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    model: Annotated[Path, typer.Option("--model", help="Model YAML; its `tool:` picks the plugin.")],
) -> None:
    """Validate a general + model config pair and add only their config rows."""
    try:
        loaded = load_configs(general, model)
    except (ConfigLoadError, ConfigPreflightError, ValidationError) as error:
        typer.echo(f"Configuration error:\n{error}", err=True)
        raise typer.Exit(code=2) from error

    record = loaded.to_record()
    target = TargetDigest.of(loaded.general.target.name, loaded.preflight.target_sequence)
    try:
        with CampaignStore.create(database, exist_ok=True) as store:
            inserted = set(store.add_configs(
                [record], target=(target.name, target.sequence_sha256)
            ))
    except TargetMismatchError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

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


@app.command()
def collect(
    manifest: Annotated[Path, typer.Argument(help="runs/<run>/run.json")],
    output: Annotated[Path, typer.Option("--output", help="Staging bundle to write.")],
) -> None:
    """Parse one run directory into a validated staging bundle.

    The tool comes from the manifest, so a run can only be parsed by the
    adapter for the tool that produced it.
    """
    try:
        collected = collect_run(manifest)
    except UnknownToolError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    write_collected(collected, output)
    run = collected.run
    typer.echo(f"run: {run.run_id} [{run.status}]")
    typer.echo(f"designs: {run.n_produced}/{run.n_requested} produced")
    typer.echo(output)


@app.command()
def ingest(
    database: Path,
    collected: Annotated[Path, typer.Argument(help="A collect-produced bundle.")],
) -> None:
    """Write one staging bundle to the campaign database, once."""
    try:
        inserted = ingest_bundle(database, collected)
    except IngestConflictError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo("ingested" if inserted else "already ingested; nothing to do")


@app.command("migrate-boltzgen")
def migrate_boltzgen(
    database: Path,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report without writing.")] = False,
    backup: Annotated[bool, typer.Option("--backup/--no-backup")] = True,
) -> None:
    """Rewrite historical BoltzGen rows into filter and rank decisions.

    Runs collected before the semantics changed stored a filtered-out design as
    `partial`, kept the verdict in `designs.metadata`, and recorded `final_rank`
    as a metric. Without this the table holds two meanings at once.
    """
    if backup and not dry_run:
        copy = database.with_suffix(database.suffix + ".pre-boltzgen-backfill")
        shutil.copyfile(database, copy)
        typer.echo(f"backup: {copy}")

    result = backfill_boltzgen_decisions(database, dry_run=dry_run)
    if result.empty:
        typer.echo("nothing to migrate")
        return
    verb = "would convert" if dry_run else "converted"
    typer.echo(f"{verb} {result.designs} designs")
    typer.echo(f"  filter decisions : {result.filters}")
    typer.echo(f"  rank decisions   : {result.ranks}")
    typer.echo(f"  metrics removed  : {result.metrics_removed}")
    typer.echo(f"  statuses fixed   : {result.statuses_corrected}")


if __name__ == "__main__":
    app()
