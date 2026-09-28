"""Inspect and distribute kit bytes without executing publisher code."""

from pathlib import Path
from typing import Annotated

import typer
import yaml

from bindocracy.cli_output import emit, fail
from bindocracy.kits.bundle import KitRelease, cache_kit, inspect_kit
from bindocracy.kits.models import KitError
from bindocracy.kits.optimizer import optimizer_config
from bindocracy.kits.resolve import resolve_kit, write_lock
from bindocracy.kits.store import fetch_kit, list_releases, publish_kit

app = typer.Typer(help="Inspect, distribute and resolve pinned kits; never execute their code.")


def _mapping(path: Path | None) -> dict:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise KitError(f"{path} must contain a mapping with string keys")
    return value


def _release(release: KitRelease) -> dict:
    return {
        "root": str(release.root),
        "digest": release.digest,
        "manifest": release.manifest.model_dump(mode="json", by_alias=True),
        "files": release.files,
        "publisher_authenticated": False,
        "execution_authorized": False,
    }


@app.command("inspect")
def inspect_command(
    source: Annotated[Path, typer.Argument(help="Directory containing kit.yaml.")],
) -> None:
    """Validate a complete file inventory without imports, hooks or writes."""
    try:
        release = inspect_kit(source)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="kit_error")
    emit(_release(release))


@app.command("check")
def check_command(
    source: Annotated[Path, typer.Argument(help="Directory containing kit.yaml.")],
    bindings: Annotated[Path | None, typer.Option("--bindings", help="Site-owned bindings YAML.")] = None,
    parameters: Annotated[
        Path | None, typer.Option("--parameters", help="Declared parameter overrides as YAML.")
    ] = None,
) -> None:
    """Inspect declared local dependencies; never run container or import probes."""
    try:
        lock = resolve_kit(source, bindings=bindings, parameters=_mapping(parameters))
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="kit_error")
    statuses = {check["status"] for check in lock.checks}
    status = "blocked" if "blocked" in statuses else "unknown" if "unknown" in statuses else "ready"
    emit({
        "status": status,
        "bundle_digest": lock.bundle_digest,
        "checks": lock.checks,
        "parameters": lock.parameters,
        "runtime": lock.runtime,
        "execution_authorized": False,
        "runtime_qualified": False,
    })
    if status != "ready":
        raise typer.Exit(1)


@app.command("list")
def list_command(
    store: Annotated[str, typer.Option("--store", help="Filesystem store or HTTPS base URL.")],
) -> None:
    """List releases; namespaces are claims, not authenticated publisher identities."""
    try:
        releases = list_releases(store)
    except (ValueError, OSError) as error:
        fail(error, code="kit_error")
    emit({"releases": releases, "publisher_authenticated": False})


@app.command("publish")
def publish_command(
    source: Annotated[Path, typer.Argument(help="Directory containing the complete kit.")],
    store: Annotated[Path, typer.Option("--store", help="Writable filesystem store root.")],
    publisher: Annotated[str, typer.Option("--publisher", help="Claimed namespace; not a signature.")],
) -> None:
    """Publish immutable bytes locally; no network uploads or execution hooks."""
    try:
        release = publish_kit(source, store, publisher)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="kit_error")
    emit({"release": release, "publisher_authenticated": False})


@app.command("fetch")
def fetch_command(
    reference: Annotated[str, typer.Argument(help="Exact publisher/name@version reference.")],
    store: Annotated[str, typer.Option("--store", help="Filesystem store or HTTPS base URL.")],
    cache: Annotated[Path, typer.Option("--cache", help="Local verified bundle cache.")],
    digest: Annotated[str, typer.Option("--digest", help="Required expected sha256-prefixed digest.")],
) -> None:
    """Fetch pinned bytes into the cache; fetching never approves execution."""
    try:
        release = fetch_kit(store, reference, cache, digest)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="kit_error")
    emit(_release(release))


@app.command("resolve")
def resolve_command(
    source: Annotated[Path, typer.Argument(help="Reviewed kit directory or fetched cache directory.")],
    cache: Annotated[Path, typer.Option("--cache", help="Cache the complete verified bundle here.")],
    config: Annotated[Path, typer.Option("--config", help="Campaign request, without kit-owned fields.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="New directory for lock and config.")],
    approve: Annotated[
        str, typer.Option("--approve", help="Exact reviewed bundle digest; not a sandbox guarantee.")
    ],
    bindings: Annotated[Path | None, typer.Option("--bindings", help="Site-owned bindings YAML.")] = None,
    parameters: Annotated[
        Path | None, typer.Option("--parameters", help="Declared parameter overrides as YAML.")
    ] = None,
) -> None:
    """Freeze an approved kit into kit-lock.json and an ordinary optimize.yaml.

    Parameters and dependency evidence are frozen but runtime probes remain
    unknown. Job submission still requires the separate campaign-plan approval.
    """
    try:
        destination = output_dir.resolve()
        if destination.exists():
            raise KitError(f"output directory already exists: {destination}")
        if destination.is_relative_to(source.resolve()):
            raise KitError("resolved site configuration must be outside the source kit")
        request = _mapping(config)
        overrides = _mapping(parameters)
        release = cache_kit(source, cache, expected_digest=approve)
        if destination.is_relative_to(release.root):
            raise KitError("resolved site configuration must be outside the cached kit")
        lock = resolve_kit(
            release.root, bindings=bindings, parameters=overrides, approved_digest=approve,
        )
        lock_path = destination / "kit-lock.json"
        model = optimizer_config(lock, request, lock_path)
        destination.mkdir(parents=True, exist_ok=False)
        write_lock(lock, lock_path)
        model_path = destination / "optimize.yaml"
        with model_path.open("x") as handle:
            yaml.safe_dump(model.model_dump(mode="json"), handle, sort_keys=False)
    except (ValueError, OSError, yaml.YAMLError) as error:
        fail(error, code="kit_error")
    emit({
        "bundle_digest": lock.bundle_digest,
        "lock_digest": lock.digest,
        "lock": str(lock_path),
        "config": str(model_path),
        "checks": lock.checks,
        "runtime_qualified": False,
        "campaign_approval_required": True,
    })
