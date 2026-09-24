"""CPU-only stored cohort policy command."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from bindocracy.config.load import ConfigLoadError, load_yaml
from bindocracy.config.models import GeneralConfig
from bindocracy.ranking.models import RankingPolicy
from bindocracy.ranking.run import RankingError, apply_ranking, summarise
from bindocracy.runs.designset import DesignSetError
from bindocracy.runs.selection import SelectionError
from bindocracy.store.store import IngestConflictError, TargetMismatchError

app = typer.Typer(help="Apply reproducible ranking and head/tail cohort policies.")


@app.command("apply")
def rank_apply(
    database: Annotated[Path, typer.Argument(help="Campaign database to read and write.")],
    general: Annotated[Path, typer.Option("--general", help="General campaign YAML.")],
    policy: Annotated[Path, typer.Option("--policy", help="Stored ranking policy YAML.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Content-addressed rank outputs.")
    ],
) -> None:
    """Rank an explicit frozen source, ingest decisions, and freeze each cohort."""
    try:
        general_config = load_yaml(general, GeneralConfig)
        ranking_policy = load_yaml(policy, RankingPolicy)
        # Unlike shell working directories, a policy-relative source is portable.
        if not ranking_policy.design_set.is_absolute():
            ranking_policy = ranking_policy.model_copy(
                update={
                    "design_set": (policy.parent / ranking_policy.design_set).resolve(),
                }
            )
        collected, inserted, manifests = apply_ranking(
            database=database,
            general=general_config,
            policy=ranking_policy,
            output_dir=output_dir,
            general_source=general,
        )
    except (
        ConfigLoadError,
        ValidationError,
        RankingError,
        DesignSetError,
        SelectionError,
        IngestConflictError,
        TargetMismatchError,
        OSError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(summarise(collected))
    typer.echo("ingested" if inserted else "already ingested; nothing to do")
    typer.echo(f"outputs: {collected.run.output_uri}")
    for name, manifest in manifests.items():
        typer.echo(f"{name}: {manifest}")
