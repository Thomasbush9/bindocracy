"""Target input materialization command, registered by the root CLI."""

from pathlib import Path
from typing import Annotated

import typer

from bindocracy.cli_output import emit, fail
from bindocracy.config.load import load_yaml
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.materialize import TargetFormat, materialize_target


def register(app: typer.Typer) -> None:
    app.command("materialize")(materialize)


def materialize(
    general: Annotated[Path, typer.Option("--general", help="General target configuration.")],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", help="New immutable directory, or an identical prior materialization."
        ),
    ],
    formats: Annotated[
        list[TargetFormat],
        typer.Option("--format", help="Repeat for each required target representation."),
    ],
    msa_source: Annotated[
        str | None,
        typer.Option(
            "--msa-source",
            help="Actual Chai alignment database: uniref90, uniprot, bfd_uniclust or mgnify. Otherwise use a recognized A3M filename.",
        ),
    ] = None,
) -> None:
    """Transform existing inputs offline; never fold, search, crop or renumber."""
    try:
        result = materialize_target(
            load_yaml(general, GeneralConfig),
            output_dir=output_dir,
            formats=formats,
            msa_source=msa_source,
        )
    except (OSError, ValueError) as exc:
        fail(exc, code="materialization_error")
    emit(result)
