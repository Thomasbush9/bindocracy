"""CLI adapter for freezing and reviewing execution plans."""

from pathlib import Path
from typing import Annotated

import typer
import yaml

from bindocracy.campaign.plan import build_plan, load_plan

app = typer.Typer(help="Freeze and inspect execution plans without submitting jobs.")


def _summary(plan: dict) -> str:
    site = plan["site"]
    lines = [
        f"Plan: {Path(plan['plan_dir']) / 'plan.json'}",
        f"Digest: {plan['digest']}",
        f"Database: {plan['database']}",
        (
            f"Controller: {site['controller']['gpus']} GPU(s), "
            f"{site['controller']['cpus']} CPUs, {site['controller']['memory_gb']} GiB, "
            f"{site['controller']['walltime']}; "
            f"{site['controller']['account']}/{site['controller']['partition']}"
        ),
        (
            f"Concurrency: at most {site['max_workers']} workers; "
            f"total GPU ceiling: {site['max_total_gpus']}"
        ),
    ]
    for run in plan["runs"]:
        scope = run["scope"]
        detail = f"{run['name']} [{run['tool']}/{run['kind']}]: {run['tasks']} tasks"
        if scope["input_candidates"] is not None:
            detail += f", {scope['input_candidates']} exact input candidates"
        if run["n_predictions"] is not None:
            detail += f", {run['n_predictions']} requested predictions"
        if scope["generation_request_slots"] is not None:
            detail += f", {scope['generation_request_slots']} generation request slots (not guaranteed outputs)"
        detail += f", {run['resources']['gpu']} GPU(s)/task"
        lines.append(detail)
        resources = run["resources"]
        lines.append(
            f"  Worker: {resources['cpus_per_task']} CPUs, {resources['mem_mb']} MiB, "
            f"{resources['runtime']} minutes; "
            f"{resources['slurm_account']}/{resources['slurm_partition']}"
        )
        if scope.get("design_set_digest"):
            lines.append(f"  Design set: {scope['design_set_digest']}")
            lines.append(
                f"  By generator: {scope['by_generator']}; by source run: {scope['by_source_run']}"
            )
    lines.append(
        "Planning does not submit jobs. Submission requires --approve with this exact digest."
    )
    return "\n".join(lines)


@app.command("plan")
def plan_command(
    index: Annotated[Path, typer.Argument(help="Ordinary workflow index YAML.")],
    site: Annotated[Path, typer.Option("--site", help="Controller and campaign resource limits.")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Empty directory for the frozen plan.")
    ],
) -> None:
    """Preflight tools, freeze workload and write normal run manifests; never submit."""
    try:
        plan = build_plan(index, site, output_dir)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
        typer.echo(f"Campaign planning error:\n{error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(_summary(plan))


@app.command("show")
def show_command(
    plan: Annotated[Path, typer.Argument(help="Frozen plan.json to verify and review.")],
) -> None:
    """Verify the frozen execution inputs and print the reviewed workload."""
    try:
        frozen = load_plan(plan)
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Campaign plan error:\n{error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(_summary(frozen))
