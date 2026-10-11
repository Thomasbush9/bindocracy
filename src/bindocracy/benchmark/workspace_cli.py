"""Human and machine entry points for organizer inputs and agent workspaces."""
from pathlib import Path
from typing import Annotated

import typer
import yaml

from bindocracy.benchmark.workspace import (
    capture_environment,
    initialize_workspace,
    load_environment,
    load_workspace,
)
from bindocracy.cli_output import emit, fail

app = typer.Typer(help="Initialize and verify content-addressed benchmark workspaces.")
environment_app = typer.Typer(help="Capture/check organizer-installed assets; not GPU qualification.")
app.add_typer(environment_app, name="environment")


@app.command("init")
def workspace_init(
    root: Annotated[Path, typer.Argument(help="New workspace directory; must not already exist.")],
    environment: Annotated[Path, typer.Option("--environment", help="Captured environment JSON/YAML.")],
    task: Annotated[Path, typer.Option("--task", help="Organizer task JSON/YAML.")],
    attempt_id: Annotated[str | None, typer.Option("--attempt-id")] = None,
) -> None:
    """Create a database, prompt, policy snapshots, configs and run directories."""
    try:
        workspace = initialize_workspace(environment, task, root, attempt_id=attempt_id)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="workspace_error")
    emit(workspace, text=f"Created workspace {workspace.root}\nAttempt: {workspace.attempt_id}\nNo jobs submitted; run qualification is separate.")


@app.command("check")
def workspace_check(
    path: Annotated[Path, typer.Argument(help="Workspace root or workspace.json.")],
) -> None:
    """Verify source content, task identity and the identical agent prompt."""
    try:
        workspace = load_workspace(path)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="workspace_error")
    emit({"workspace": workspace.root, "attempt_id": workspace.attempt_id,
          "task_digest": workspace.task_digest, "environment_digest": workspace.environment_digest,
          "integrity": "verified", "runtime_qualified": False},
         text=f"Workspace integrity verified: {workspace.root}\nThis is not runtime or scheduler qualification.")


@environment_app.command("capture")
def environment_capture(
    spec: Annotated[Path, typer.Argument(help="Organizer environment spec; paths relative to this document.")],
    output: Annotated[Path, typer.Option("--output", help="New captured manifest; never overwritten.")],
) -> None:
    """Hash source, uv.lock, templates and all declared input assets without executing tools."""
    try:
        environment = capture_environment(spec, output)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        fail(error, code="environment_error")
    emit({"path": output.absolute(), "environment": environment, "runtime_qualified": False},
         text=f"Captured environment: {output.absolute()}\nStatic integrity only; no runtime qualification performed.")


@environment_app.command("check")
def environment_check(
    path: Annotated[Path, typer.Argument(help="Captured environment JSON/YAML.")],
) -> None:
    """Rehash real bytes and reject missing, changed or uninventoried inputs."""
    try:
        environment = load_environment(path)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        fail(error, code="environment_error")
    emit({"name": environment.name, "integrity": "verified",
          "harness_sha256": environment.harness_sha256, "lock_sha256": environment.lock_sha256,
          "assets": len(environment.assets), "qualification": environment.qualification,
          "runtime_qualified": False},
         text=f"Environment integrity verified: {environment.name}\nThis is not runtime or scheduler qualification.")
