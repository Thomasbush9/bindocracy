"""Command-line entry point.

A placeholder so the `bindocracy` console script declared in pyproject.toml
resolves against a real callable. Subcommands get added as the harness lands.
"""

import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from bindocracy import __version__
from bindocracy.campaign.cli import app as campaign_app
from bindocracy.cli_output import OutputGroup, emit, fail
from bindocracy.config import (
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigPreflightError,
    recover_config_yaml,
)
from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig, ResourceConfig
from bindocracy.discovery import register as register_discovery
from bindocracy.filters.config import FilterConfig
from bindocracy.filters.config import summarise as summarise_filter
from bindocracy.filters.run import FilterRunError, run_filter
from bindocracy.functions.run import FunctionRunConfig
from bindocracy.kits.cli import app as kit_app
from bindocracy.ranking.cli import app as rank_app
from bindocracy.runs import ingest_bundle, ingest_collected, write_collected
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.runs.designset import summarise as summarise_set
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.materialize_cli import register as register_materialize
from bindocracy.runs.msa import prepare_target_msa
from bindocracy.runs.selection import SelectionError, read_only, resolve_run_ids, select
from bindocracy.runs.selection import build as build_selection
from bindocracy.store import (
    CampaignStore,
    IngestConflictError,
    TargetMismatchError,
    create_database,
)
from bindocracy.store.query import DesignQuery, metric_catalog
from bindocracy.tools import UnknownToolError, collect_run, load_configs
from bindocracy.tools.boltzgen.migrate import backfill_boltzgen_decisions

app = typer.Typer(
    cls=OutputGroup,
    add_completion=False,
    help="Run many binder-design models from one place.",
)
config_app = typer.Typer(help="Validate and load campaign configuration.")
app.add_typer(config_app, name="config")
target_app = typer.Typer(help="Prepare shared target inputs before planning runs.")
app.add_typer(target_app, name="target")
register_materialize(target_app)
designset_app = typer.Typer(help="Freeze a database query into a reusable candidate set.")
app.add_typer(designset_app, name="designset")
filter_app = typer.Typer(help="Apply a stored filter policy to a frozen design set.")
app.add_typer(filter_app, name="filter")
function_app = typer.Typer(help="Score a frozen database selection with a custom function.")
app.add_typer(function_app, name="function")
app.add_typer(campaign_app, name="campaign")
app.add_typer(rank_app, name="rank")
app.add_typer(kit_app, name="kit")
register_discovery(app, config_app)


@target_app.command("prepare-msa")
def prepare_msa(
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    script: Annotated[
        Path,
        typer.Option(
            "--script",
            help="Imported Mosaic singularity/msa-search.sbatch from ProtForge.",
        ),
    ],
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
            config,
            script=script,
            image=image,
            database=database,
            resources=ResourceConfig(gpus=1, cpus=cpus, memory_gb=memory_gb, walltime=walltime),
        )
    except (ConfigLoadError, ConfigPreflightError, ValidationError, OSError) as error:
        fail(error)
    emit({"output": output}, text=str(output))


@designset_app.command("preview")
@designset_app.command("build")
def designset_build(
    ctx: typer.Context,
    database: Annotated[Path, typer.Argument(help="Campaign database to select from.")],
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="Required for build; not used by preview.")
    ] = None,
    query_file: Annotated[
        Path | None,
        typer.Option(
            "--query",
            help="A DesignQuery YAML, instead of the flags below.",
        ),
    ] = None,
    tool: Annotated[
        list[str] | None, typer.Option("--tool", help="Producing tool; repeatable.")
    ] = None,
    run_name: Annotated[
        list[str] | None, typer.Option("--run-name", help="Producing run; repeatable.")
    ] = None,
    passed_filter: Annotated[
        list[str] | None,
        typer.Option(
            "--passed-filter",
            help="Decision name a design must have passed; repeatable.",
        ),
    ] = None,
    filter_run: Annotated[
        list[str] | None,
        typer.Option(
            "--filter-run",
            help="Filter run whose verdicts to trust; required with --passed-filter.",
        ),
    ] = None,
    exclude_scored_by: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-scored-by",
            help="Skip designs this evaluator run already scored; repeatable.",
        ),
    ] = None,
    min_length: Annotated[int | None, typer.Option("--min-length", min=1)] = None,
    max_length: Annotated[int | None, typer.Option("--max-length", min=1)] = None,
    created_after: Annotated[
        str | None,
        typer.Option(
            "--created-after",
            help="Include designs created at/after this timezone-aware ISO timestamp.",
        ),
    ] = None,
    created_before: Annotated[
        str | None,
        typer.Option(
            "--created-before",
            help="Include designs created before this timezone-aware ISO timestamp.",
        ),
    ] = None,
    distinct_sequences: Annotated[
        bool,
        typer.Option(
            "--distinct-sequences",
            help="Keep one design per identical sequence.",
        ),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
) -> None:
    """Freeze the designs a query selects into a content-addressed set.

    The digest covers the members and their order and nothing else, so the same
    query against an unchanged database is recognisably the same set rather
    than a new one. Use `--passed-filter` with `--filter-run` to select what a
    stored policy chose; see `bindocracy filter apply`.
    """
    preview = ctx.info_name == "preview"
    if preview and out_dir is not None:
        raise typer.BadParameter("preview does not write files; omit --out-dir")
    if not preview and out_dir is None:
        raise typer.BadParameter("build requires --out-dir", param_hint="--out-dir")
    if query_file is not None:
        flags = (
            tool,
            run_name,
            passed_filter,
            filter_run,
            exclude_scored_by,
            min_length,
            max_length,
            created_after,
            created_before,
            limit,
        )
        if any(value for value in flags) or distinct_sequences:
            raise typer.BadParameter(
                "--query replaces the narrowing flags; pass one or the other",
                param_hint="--query",
            )
        try:
            query = load_yaml(query_file, DesignQuery)
        except (ConfigLoadError, ValidationError) as error:
            fail(error)
    else:
        try:
            query = DesignQuery(
                tools=tuple(tool or ()),
                run_names=tuple(run_name or ()),
                passed_filter=tuple(passed_filter or ()),
                filter_runs=tuple(filter_run or ()),
                exclude_scored_by_run=tuple(exclude_scored_by or ()),
                min_length=min_length,
                max_length=max_length,
                created_after=created_after,
                created_before=created_before,
                distinct_sequences=distinct_sequences,
                limit=limit,
            )
        except ValidationError as error:
            fail(error)

    try:
        if preview:
            rows, resolved = select(database, query, allow_empty=True)
            result = {
                "n_designs": len(rows),
                "query": resolved.model_dump(mode="json"),
                "by_tool": dict(sorted(Counter(row.tool for row in rows).items())),
                "by_source_run": dict(sorted(Counter(row.run_id for row in rows).items())),
            }
            emit(result, text=f"{len(rows)} designs; by tool: {result['by_tool']}")
            return
        design_set, fasta, manifest = build_selection(database, query, out_dir)
    except (SelectionError, DesignSetError) as error:
        fail(error, code="selection_error")

    emit(
        {"design_set": design_set, "fasta": fasta, "manifest": manifest},
        text=f"{summarise_set(design_set)}\nfasta:    {fasta}\nmanifest: {manifest}",
    )


@designset_app.command("show")
def designset_show(
    manifest: Annotated[Path, typer.Argument(help="A <digest>.json design-set manifest.")],
) -> None:
    """Summarise a frozen design set, including the query that produced it."""
    try:
        design_set = DesignSet.read(manifest)
    except DesignSetError as error:
        fail(error, code="selection_error")
    emit(design_set, text=summarise_set(design_set))


@filter_app.command("metrics")
def filter_metrics(
    database: Annotated[Path, typer.Argument(help="Campaign database.")],
    run: Annotated[
        list[str] | None, typer.Option("--run", help="Exact measuring run name/ID; repeatable.")
    ] = None,
) -> None:
    """List the metric names present, which is what a filter set may test.

    Stored names carry their scorer prefix (`boltz2_iptm`, not `iptm`). Run
    this before writing thresholds: a filter set naming a metric that does not
    exist is refused by `filter apply`, but reading the list is faster than
    being told.
    """
    try:
        with read_only(database) as connection:
            runs = resolve_run_ids(connection, tuple(run)) if run is not None else None
            metrics = metric_catalog(connection, run_ids=runs)
    except SelectionError as error:
        fail(error, code="selection_error")
    lines = [
        f"{row['name']} [{row['direction']}] run={row['run_id']} "
        f"replicas={row['replicates']}; {row['n_valid']}/{row['n_rows']} valid"
        for row in metrics
    ]
    emit({"metrics": metrics}, text="\n".join(lines) or "no metrics in this database yet")


@filter_app.command("apply")
def filter_apply(
    database: Annotated[Path, typer.Argument(help="Campaign database to read and write.")],
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    filter_config: Annotated[Path, typer.Option("--filter", help="Filter YAML; tool: filter.")],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Where the staging bundle is written.",
        ),
    ],
    ingest: Annotated[
        bool,
        typer.Option(
            "--ingest/--no-ingest",
            help="Write the verdicts to the database.",
        ),
    ] = True,
) -> None:
    """Apply a stored filter policy to a frozen design set, once.

    A filter is not a tool: no container, no GPU, no task fan-out. It reads the
    database and writes verdicts, so it runs here in process and goes to the
    database through the same staging bundle every plugin uses. No design is
    deleted and no metric is rewritten -- a design that fails is a row saying
    so, with the numbers that failed it.
    """
    try:
        general_config = load_yaml(general, GeneralConfig)
        config = load_yaml(filter_config, FilterConfig)
    except (ConfigLoadError, ValidationError) as error:
        fail(error)

    try:
        collected, config_record = run_filter(
            database=database,
            general=general_config,
            config=config,
            output_dir=output_dir,
            general_source=general,
        )
    except FilterRunError as error:
        fail(error, code="selection_error")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = write_collected(collected, output_dir / "collected.json")

    inserted = False
    if ingest:
        try:
            inserted = ingest_collected(database, collected, config=config_record)
        except (IngestConflictError, TargetMismatchError) as error:
            fail(error, code="conflict")
    disposition = (
        "not ingested (--no-ingest)"
        if not ingest
        else "ingested"
        if inserted
        else "already ingested; nothing to do"
    )
    emit(
        {"run": collected.run, "bundle": bundle, "ingested": ingest, "inserted": inserted},
        text=f"{summarise_filter(collected.run)}\nbundle: {bundle}\n{disposition}\nrun: {collected.run.run_id}",
    )


@function_app.command("plan")
def function_plan(
    database: Annotated[Path, typer.Argument(help="Campaign database supplying frozen inputs.")],
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    config: Annotated[Path, typer.Option("--config", help="Function run YAML; tool: function.")],
    site: Annotated[Path, typer.Option("--site", help="Controller and worker resource policy.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="New directory for request and frozen campaign plan."),
    ],
) -> None:
    """Freeze a scoring campaign; execute it with campaign submit after approval."""
    from bindocracy.campaign.plan import _absolute_values, build_plan

    try:
        function_config = load_yaml(config, FunctionRunConfig)
        if function_config.database is not None and function_config.database.resolve() != database.resolve():
            raise ValueError("function database differs from requested campaign database")
        function_config = function_config.model_copy(update={"database": database.resolve()})
        output = output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        resolved = output / "function.yaml"
        resolved.write_text(yaml.safe_dump(
            _absolute_values(function_config.model_dump(mode="python")), sort_keys=False,
        ))
        index = output / "workflow.yaml"
        index.write_text(yaml.safe_dump({
            "database": str(database.resolve()),
            "run_root": str(output / "runs"),
            "general_config": str(general.resolve()),
            "runs": [{"name": function_config.name, "config": str(resolved)}],
        }, sort_keys=False))
        plan = build_plan(index, site, output / "plan")
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
        fail(error)
    emit(plan, text=f"Plan: {plan['plan_dir']}/plan.json\nApprove digest: {plan['digest']}")


@app.callback()
def main(
    ctx: typer.Context,
    json_output: Annotated[
        bool, typer.Option("--json", help="Versioned JSON results/errors; put before the command.")
    ] = False,
) -> None:
    """Keep Typer in subcommand mode.

    Without a callback, Typer collapses a single-command app into the top-level
    command, so `bindocracy version` would be parsed as a stray argument.
    """
    if ctx.invoked_subcommand is None:
        emit(
            {"version": __version__, "commands": sorted(ctx.command.commands)},
            text=ctx.get_help(),
        )


@app.command()
def version() -> None:
    """Print the installed version."""
    emit({"version": __version__}, text=__version__)


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
    emit({"database": database_path}, text=str(database_path))


@config_app.command("load")
def load_config(
    database: Path,
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    model: Annotated[
        Path, typer.Option("--model", help="Model YAML; its `tool:` picks the plugin.")
    ],
) -> None:
    """Validate a general + model config pair and add only their config rows."""
    try:
        loaded = load_configs(general, model)
    except (ConfigLoadError, ConfigPreflightError, ValidationError) as error:
        fail(error)

    record = loaded.to_record()
    target = TargetDigest.of(loaded.general.target.name, loaded.preflight.target_sequence)
    try:
        with CampaignStore.create(database, exist_ok=True) as store:
            inserted = set(
                store.add_configs([record], target=(target.name, target.sequence_sha256))
            )
    except TargetMismatchError as error:
        fail(error, code="conflict")

    status = "inserted" if record.model_config_id in inserted else "already present"
    emit(
        {
            "database": database,
            "tool": loaded.tool,
            "target": loaded.general.target.name,
            "general_config_id": record.general_config_id,
            "model_config_id": record.model_config_id,
            "inserted": record.model_config_id in inserted,
        },
        text=f"database: {database}\ntarget: {loaded.general.target.name} ({loaded.preflight.target_length} aa)\n"
        f"general: {record.general_name} {record.general_config_id}\n"
        f"model: {record.model_name} {record.model_config_id} [{status}]",
    )


@config_app.command("export")
def export_config(
    database: Path,
    config_id: str,
    output: Path,
    force: Annotated[bool, typer.Option("--force", help="Overwrite OUTPUT if it exists.")] = False,
) -> None:
    """Recover a general or model YAML file from its database JSON."""
    try:
        output_path = recover_config_yaml(database, config_id, output, overwrite=force)
    except ConfigNotFoundError as error:
        fail(error, code="not_found")
    except FileExistsError as error:
        fail(error, code="conflict")
    emit({"output": output_path}, text=str(output_path))


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
        fail(error, code="unknown_tool")
    write_collected(collected, output)
    run = collected.run
    emit(
        {"run": run, "output": output},
        text=f"run: {run.run_id} [{run.status}]\ndesigns: {run.n_produced}/{run.n_requested} produced\n{output}",
    )


@app.command()
def ingest(
    database: Path,
    collected: Annotated[Path, typer.Argument(help="A collect-produced bundle.")],
) -> None:
    """Write one staging bundle to the campaign database, once."""
    try:
        inserted = ingest_bundle(database, collected)
    except IngestConflictError as error:
        fail(error, code="conflict")
    emit(
        {"inserted": inserted, "database": database, "bundle": collected},
        text="ingested" if inserted else "already ingested; nothing to do",
    )


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
    copy = None
    if backup and not dry_run:
        copy = database.with_suffix(database.suffix + ".pre-boltzgen-backfill")
        shutil.copyfile(database, copy)

    result = backfill_boltzgen_decisions(database, dry_run=dry_run)
    verb = "would convert" if dry_run else "converted"
    text = (
        "nothing to migrate"
        if result.empty
        else (
            f"{verb} {result.designs} designs\n"
            f"  filter decisions : {result.filters}\n"
            f"  rank decisions   : {result.ranks}\n"
            f"  metrics removed  : {result.metrics_removed}\n"
            f"  statuses fixed   : {result.statuses_corrected}"
        )
    )
    emit(
        {"migration": result, "dry_run": dry_run, "backup": copy},
        text=(f"backup: {copy}\n" if copy else "") + text,
    )


if __name__ == "__main__":
    app()
