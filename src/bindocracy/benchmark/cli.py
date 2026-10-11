"""Organizer attempt lifecycle and independent evaluation commands."""
import typer

from bindocracy.benchmark.evaluation_cli import app as evaluation_app
from bindocracy.benchmark.runner_cli import app as runner_app

app = typer.Typer(help="Supervise benchmark attempts and independently evaluate submissions.")
app.add_typer(runner_app)
app.add_typer(evaluation_app)
