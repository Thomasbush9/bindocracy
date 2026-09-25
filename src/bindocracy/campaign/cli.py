"""One campaign interface for frozen planning and controller lifecycle."""

import typer

from bindocracy.campaign.check_cli import register as register_check
from bindocracy.campaign.control_cli import app as control_app
from bindocracy.campaign.plan_cli import app as plan_app
from bindocracy.campaign.qualify_cli import register as register_qualify
from bindocracy.campaign.report_cli import register as register_report

app = typer.Typer(help="Review frozen execution plans and manage their Slurm jobs.")
app.add_typer(plan_app)
app.add_typer(control_app)
register_check(app)
register_report(app)
register_qualify(app)
