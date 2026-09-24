"""One campaign interface for frozen planning and controller lifecycle."""

import typer

from bindocracy.campaign.control_cli import app as control_app
from bindocracy.campaign.plan_cli import app as plan_app

app = typer.Typer(help="Review frozen execution plans and manage their Slurm jobs.")
app.add_typer(plan_app)
app.add_typer(control_app)
