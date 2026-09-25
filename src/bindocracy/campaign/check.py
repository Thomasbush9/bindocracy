"""Read-only readiness using the same loaders and plugin plans as execution.

A ready report is evidence about declared inputs and queried capabilities, not a
reservation, quota check, admission decision, or guarantee of model execution.
Paths retain the workflow loaders' current-working-directory semantics.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import duckdb
import yaml

from bindocracy.campaign.models import Site
from bindocracy.campaign.plan import _check_reuse, load_plan
from bindocracy.config.load import load_yaml
from bindocracy.runs.index import load_workflow_index
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools import load_configs, plugin_for

_PROBE_TIMEOUT = 5


def _status(checks: list[dict]) -> str:
    states = {check["status"] for check in checks}
    return "blocked" if "blocked" in states else "unknown" if "unknown" in states else "ready"


def _check(code: str, status: str, message: str, **details: Any) -> dict:
    return {"code": code, "status": status, "message": message, "details": details}


def _failure(code: str, error: Exception) -> dict:
    return _check(
        code, "blocked", str(error) or type(error).__name__, error_type=type(error).__name__
    )


def _readable_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required file does not exist or is not a regular file: {path}")
    # Opening, rather than trusting permission bits, also detects inaccessible parents.
    with path.open("rb") as handle:
        handle.read(1)


def _destination(path: Path, *, directory: bool) -> dict:
    """Inspect permissions without creating a directory or opening a database."""
    if path.exists():
        if directory != path.is_dir() or (not directory and not path.is_file()):
            raise ValueError(f"destination has the wrong filesystem type: {path}")
        if not os.access(path, os.W_OK | (os.X_OK if directory else 0)):
            raise ValueError(f"destination is not writable by this user: {path}")
    parent = path if directory and path.exists() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"destination ancestor is not a writable, searchable directory: {parent}")
    return _check(
        "destination",
        "ready",
        f"Destination permissions checked without writing: {path}",
        path=str(path),
        existing_parent=str(parent),
        quota_checked=False,
    )


def _probe(executable: str, arguments: list[str]) -> tuple[str | None, dict]:
    program = shutil.which(executable)
    if program is None:
        return None, _check(
            executable,
            "unknown",
            f"{executable} is not on PATH; load the site's Slurm client "
            "environment and retry --probe-site.",
        )
    command = [program, *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, _check(
            executable,
            "unknown",
            f"Read-only scheduler query unavailable: {error}. "
            "Check Slurm connectivity and permissions, then retry.",
            command=command,
        )
    if result.returncode:
        return None, _check(
            executable,
            "unknown",
            "Read-only scheduler query failed; check client configuration, "
            "accounting availability and permissions with the site administrator.",
            command=command,
            returncode=result.returncode,
            diagnostic=(result.stderr or result.stdout).strip()[:2000],
        )
    return result.stdout, _check(
        executable, "ready", "Read-only scheduler query succeeded.", command=command
    )


def _scheduler(requests: set[tuple[str, str]], *, probe_site: bool) -> list[dict]:
    if not probe_site:
        return [
            _check(
                "site_probe",
                "unknown",
                "Scheduler capabilities were not probed. Supply --site SITE "
                "and --probe-site on a host with Slurm clients; static success is not launch readiness.",
            )
        ]
    if not requests:
        return [_check("site_probe", "unknown", "No valid scheduler resource requests to probe.")]
    checks = []
    partitions, query = _probe("sinfo", ["--noheader", "--format=%P|%a"])
    checks.append(query)
    if partitions is not None:
        availability = {}
        malformed = False
        for line in partitions.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split("|")
            if len(fields) != 2 or fields[1].lower() not in {"up", "down", "drain", "inactive"}:
                malformed = True
                continue
            availability[fields[0].rstrip("*")] = fields[1].lower()
        if malformed or not availability:
            checks.append(
                _check(
                    "partitions",
                    "unknown",
                    "sinfo returned no usable partition inventory; "
                    "ask the site administrator to verify partition visibility.",
                )
            )
        else:
            for partition in sorted({partition for _, partition in requests}):
                state = availability.get(partition)
                checks.append(
                    _check(
                        "partition",
                        "ready" if state == "up" else "blocked",
                        f"Partition {partition!r} is {state or 'not visible'}; "
                        "this does not establish node availability or admission.",
                        partition=partition,
                    )
                )
    configuration, query = _probe("scontrol", ["show", "config"])
    checks.append(query)
    if configuration is None:
        return checks
    cluster = next(
        (
            value.strip()
            for line in configuration.splitlines()
            for key, separator, value in [line.partition("=")]
            if separator and key.strip() == "ClusterName"
        ),
        None,
    )
    if not cluster or not all(character.isalnum() or character in "_.-" for character in cluster):
        checks.append(
            _check(
                "cluster",
                "unknown",
                "Cannot determine the current Slurm cluster; "
                "account associations from other clusters cannot establish access.",
            )
        )
        return checks
    try:
        user = getpass.getuser()
    except OSError as error:
        checks.append(_check("associations", "unknown", f"Cannot identify scheduler user: {error}"))
        return checks
    associations, query = _probe(
        "sacctmgr",
        [
            "--noheader",
            "--parsable2",
            "show",
            "assoc",
            "where",
            f"user={user}",
            f"cluster={cluster}",
            "format=Account,Partition",
        ],
    )
    checks.append(query)
    if associations is not None:
        pairs = set()
        malformed = False
        for line in associations.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split("|")
            if len(fields) != 2 or not fields[0]:
                malformed = True
                continue
            pairs.add(tuple(fields))
        if malformed or not pairs:
            checks.append(
                _check(
                    "associations",
                    "unknown",
                    "Accounting returned no usable user "
                    "associations. Ask the site administrator to verify account access.",
                )
            )
        else:
            for account, partition in sorted(requests):
                allowed = (account, partition) in pairs or (account, "") in pairs
                checks.append(
                    _check(
                        "association",
                        "ready" if allowed else "unknown",
                        f"Account {account!r}, partition {partition!r}: "
                        + (
                            "a matching user association is visible; QOS and admission are not verified."
                            if allowed
                            else "no matching user association is visible; ask the site "
                            "administrator to verify account/partition access."
                        ),
                        account=account,
                        partition=partition,
                    )
                )
    return checks


def check_campaign(
    index_path: str | Path,
    site_path: str | Path | None = None,
    *,
    probe_site: bool = False,
) -> dict:
    """Collect readiness failures for every run without materializing an execution.

    Built-in preflights only read inputs (including read-only design-set database
    queries). No planner, container import probe, submission or cancellation is
    invoked. A malformed index cannot provide typed run requests to check.
    """
    checks: list[dict] = []
    runs: list[dict] = []
    requests: set[tuple[str, str]] = set()
    site = None
    if site_path is not None:
        try:
            site = load_yaml(site_path, Site)
            if site.controller.gpus >= site.max_total_gpus:
                raise ValueError("controller GPU allocation leaves no worker GPU budget")
            checks.append(
                _check("site", "ready", "Site configuration and controller budget are valid.")
            )
        except (ValueError, OSError) as error:
            checks.append(_failure("site", error))
    else:
        checks.append(
            _check(
                "site",
                "unknown",
                "No site policy supplied; controller and worker "
                "budgets cannot be checked. Supply --site SITE.",
            )
        )
    if site is not None:
        requests.add((site.controller.account, site.controller.partition))
    try:
        index = load_workflow_index(index_path)
    except (ValueError, OSError, yaml.YAMLError) as error:
        checks.append(_failure("index", error))
        index = None
    if index is not None:
        checks.append(_check("index", "ready", "Workflow index is valid."))
        for path, directory in (
            (index.run_root.absolute(), True),
            (index.database.absolute(), False),
        ):
            try:
                checks.append(_destination(path, directory=directory))
            except (ValueError, OSError) as error:
                checks.append(_failure("destination", error))
        if index.campaign_plan is not None:
            try:
                frozen = load_plan(index.campaign_plan)
                if index != load_workflow_index(frozen["workflow_index"]):
                    raise ValueError("workflow index differs from the approved campaign plan")
                if site is not None and site != Site.model_validate(frozen["site"]):
                    raise ValueError("site policy differs from the approved campaign plan")
                checks.append(_check("frozen_plan", "ready", "Frozen execution closure verified."))
            except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
                checks.append(_failure("frozen_plan", error))
        for request in index.runs:
            findings: list[dict] = []
            run: dict = {"name": request.name, "config": str(request.config), "checks": findings}
            runs.append(run)
            if request.name in {".", ".."}:
                findings.append(_check("run_name", "blocked", "Run names cannot be '.' or '..'."))
            try:
                loaded = load_configs(index.general_config, request.config)
                plugin = plugin_for(loaded.tool)
                run["tool"] = loaded.tool
                findings.append(
                    _check("preflight", "ready", "Typed configs and tool preflight passed.")
                )
            except (
                ValueError,
                RuntimeError,
                OSError,
                KeyError,
                yaml.YAMLError,
                duckdb.Error,
            ) as error:
                findings.append(_failure("preflight", error))
                run["status"] = _status(findings)
                continue
            try:
                resolved = plugin.resources(loaded)
                gpu = loaded.model.resources.gpus
                run["resources"] = {**resolved, "gpu": gpu}
                if not isinstance(resolved.get("gres"), str) or resolved["gres"].split(":")[
                    -1
                ] != str(gpu):
                    raise ValueError("resolved GPU request disagrees with resource configuration")
                for key in ("cpus_per_task", "mem_mb", "runtime"):
                    if not isinstance(resolved.get(key), int) or resolved[key] <= 0:
                        raise ValueError(f"resolved {key} must be a positive integer")
                for key in ("slurm_account", "slurm_partition"):
                    if not isinstance(resolved.get(key), str) or not resolved[key].strip():
                        raise ValueError(f"resolved {key} must be a nonempty string")
                requests.add((resolved["slurm_account"], resolved["slurm_partition"]))
                if site is not None and gpu + site.controller.gpus > site.max_total_gpus:
                    raise ValueError("one worker plus controller exceeds max_total_gpus")
                findings.append(
                    _check(
                        "resources",
                        "ready",
                        "Worker resources are valid and fit "
                        "the supplied budget (if any); max_workers is a cap, not a reservation.",
                    )
                )
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                findings.append(_failure("resources", error))
            try:
                plan = plugin.tool_plan(loaded)
                run["tasks"] = plan.jobs
                if plan.jobs <= 0 or plan.designs_per_task <= 0:
                    raise ValueError("plugin must plan positive task and request counts")
                findings.append(_check("tool_plan", "ready", "In-memory tool plan is valid."))
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                findings.append(_failure("tool_plan", error))
                run["status"] = _status(findings)
                continue
            files = {
                **plan.inputs,
                **{f"archive:{key}": value for key, value in plan.archives.items()},
            }
            files["container"] = plan.container
            files.update(
                {
                    f"parent_structure:{key}": Path(value)
                    for key, value in plan.workflow.get("parent_structures", {}).items()
                }
            )
            for label, path in files.items():
                try:
                    _readable_file(Path(path))
                    findings.append(
                        _check("input", "ready", f"Declared {label} is readable.", path=str(path))
                    )
                except (ValueError, OSError, TypeError) as error:
                    findings.append(_failure("input", error))
            if loaded.tool == "optimize":
                findings.append(
                    _check(
                        "kit_dependencies",
                        "unknown",
                        "Optimizer script dependencies are not declared "
                        "by the workflow index. For a kit, run scripts/check_kit.py with its explicit "
                        "--kit, --bindings, --config and --no-probe; no kit bindings or container imports "
                        "are inferred by this check.",
                    )
                )
            if request.name not in {".", ".."}:
                try:
                    run_dir = index.run_dir(request.name).resolve()
                    manifest_path = run_dir / "run.json"
                    if manifest_path.exists():
                        manifest = RunManifest.read(manifest_path)
                        if manifest.directory != run_dir:
                            raise ValueError(
                                f"manifest points outside its requested directory: {manifest_path}"
                            )
                        _check_reuse(manifest, loaded, plan, request.name)
                    elif run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
                        raise ValueError(f"run directory has data but no manifest: {run_dir}")
                    if index.campaign_plan is None and (run_dir / "campaign-binding.json").exists():
                        raise ValueError(f"run is already bound to a campaign plan: {run_dir}")
                    findings.append(
                        _check(
                            "run_directory",
                            "ready",
                            "Run destination checked without reserving it.",
                        )
                    )
                except (ValueError, RuntimeError, OSError) as error:
                    findings.append(_failure("run_directory", error))
            run["status"] = _status(findings)
    static = checks + [finding for run in runs for finding in run["checks"]]
    scheduler = _scheduler(requests, probe_site=probe_site)
    return {
        "schema_version": 1,
        "index": str(index_path),
        "site": str(site_path) if site_path is not None else None,
        "status": _status(static + scheduler),
        "static_status": _status(static),
        "scheduler_status": _status(scheduler),
        "checks": checks,
        "runs": runs,
        "scheduler_checks": scheduler,
        "scope": "Declared inputs, static policy and optional read-only Slurm visibility only. "
        "No quota, QOS/admission, free-node, container execution, GPU or output-quality guarantee.",
    }
