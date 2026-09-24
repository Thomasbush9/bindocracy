"""Freeze the ordinary plugin/Snakemake path without submitting or opening a writer.

The digest covers the plan's canonical JSON, including manifest and snapshot
hashes. Authored YAML is provenance, never an execution input after planning.
Small consumed inputs and source are content checked; large runtime assets use
an explicitly labelled stat inventory rather than rehashing container images.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from bindocracy.campaign.models import Site
from bindocracy.config.load import load_yaml, read_tool_name
from bindocracy.config.models import GeneralConfig
from bindocracy.runs.designset import DesignSet, read_fasta_entries
from bindocracy.runs.index import load_workflow_index
from bindocracy.runs.inputs import digest_of
from bindocracy.runs.manifest import RunManifest, ToolPlan, plan_run, write_json_atomic
from bindocracy.store.records import RunKind, canonical_json, sha256_text, stable_id
from bindocracy.tools import load_configs, plugin_for


class PlanError(ValueError):
    """An execution cannot be frozen or no longer matches its reviewed plan."""


def _absolute_values(value: Any) -> Any:
    # Match the existing loaders' cwd-relative path semantics, then make them
    # independent of the controller's working directory. Do not guess at strings.
    if isinstance(value, Path):
        return str(value.absolute())
    if isinstance(value, Enum):
        return _absolute_values(value.value)
    if isinstance(value, dict):
        return {key: _absolute_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_absolute_values(item) for item in value]
    return value


def _hash(path: Path) -> str:
    return digest_of(path).sha256


def _identity(plan: dict) -> str:
    return sha256_text(
        canonical_json({key: value for key, value in plan.items() if key != "digest"})
    )


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _hash(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _stat(path: Path) -> dict[str, int | str]:
    info = path.stat()
    return {
        "resolved": str(path.resolve()),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def _asset(path: Path) -> dict:
    if path.is_dir():
        return {
            "kind": "directory-stat",
            "resolved": str(path.resolve()),
            "files": {
                str(item.relative_to(path)): _stat(item)
                for item in sorted(path.rglob("*"))
                if item.is_file()
            },
        }
    return {"kind": "file-stat", "stat": _stat(path)}


def _executables() -> tuple[str, str]:
    # Resolving the Python symlink would discard the virtualenv identity.
    python = Path(sys.executable).absolute()
    snakemake = python.parent / "snakemake"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PlanError(f"Python interpreter is not executable: {python}")
    if not snakemake.is_file() or not os.access(snakemake, os.X_OK):
        raise PlanError(f"Snakemake must be installed beside this interpreter: {snakemake}")
    return str(python), str(snakemake)


def _scope(tool_plan: ToolPlan) -> dict:
    workflow = tool_plan.workflow
    requested = tool_plan.jobs * tool_plan.designs_per_task
    scope: dict[str, Any] = {
        "n_designs": None,
        "n_predictions": None,
        "input_candidates": None,
        "generation_request_slots": requested if tool_plan.kind == RunKind.GENERATE else None,
        "guaranteed_outputs": False,
        "attempt_slots": tool_plan.jobs
        * (tool_plan.generated_per_task or tool_plan.designs_per_task)
        if tool_plan.kind == RunKind.GENERATE
        else None,
    }
    design_set_path = workflow.get("design_set_manifest")
    if design_set_path:
        design_set = DesignSet.read(design_set_path)
        entries = design_set.entries
        digest = sha256_text(
            canonical_json([[entry.design_id, entry.sequence] for entry in entries])
        )
        fasta = read_fasta_entries(design_set.fasta_path(design_set_path))
        try:
            fasta_indices = [int(header.split()[0]) for header, _ in fasta]
        except (ValueError, IndexError) as error:
            raise PlanError(f"invalid design-set FASTA indices: {design_set_path}") from error
        by_generator = dict(sorted(Counter(entry.tool for entry in entries).items()))
        lengths = [len(entry.sequence) for entry in entries]
        if (
            digest != design_set.digest
            or design_set.scope_id != stable_id("design-set", digest)
            or workflow.get("scope_id") != design_set.scope_id
            or design_set.n_designs != len(entries)
            or not entries
            or design_set.by_tool != by_generator
            or design_set.distinct_lengths != len(set(lengths))
            or design_set.length_range != (min(lengths), max(lengths))
            or [entry.index for entry in entries] != list(range(len(entries)))
            or len({entry.design_id for entry in entries}) != len(entries)
            or [sequence for _, sequence in fasta] != [entry.sequence for entry in entries]
            or fasta_indices != [entry.index for entry in entries]
            or any(entry.length != len(entry.sequence) for entry in entries)
            or workflow.get("n_designs") != len(entries)
            or workflow.get("design_set_digest") != digest
        ):
            raise PlanError(
                f"design-set membership, digest, FASTA or scope mismatch: {design_set_path}"
            )
        scope.update(
            {
                "n_designs": len(entries),
                "input_candidates": len(entries),
                "design_set_digest": digest,
                "scope_id": design_set.scope_id,
                "by_generator": by_generator,
                "by_source_run": dict(sorted(Counter(entry.run_name for entry in entries).items())),
                "selection": design_set.query,
                "shard_candidates": [
                    len(design_set.shard(index, tool_plan.jobs)) for index in range(tool_plan.jobs)
                ],
            }
        )
        if tool_plan.kind == RunKind.EVALUATE:
            samples = workflow.get("samples_per_design")
            if samples is None:
                samples = workflow.get("protocol", {}).get("model", {}).get("num_samples")
            readers = workflow.get("readers")
            if isinstance(samples, int) and samples > 0 and isinstance(readers, list) and readers:
                scope.update(
                    {
                        "samples_per_design_per_reader": samples,
                        "readers": readers,
                        "predictions_per_design": samples * len(readers),
                        "n_predictions": len(entries) * samples * len(readers),
                    }
                )
        if tool_plan.kind == RunKind.OPTIMIZE:
            scope["max_children"] = workflow.get("max_children")
    return scope


def _check_reuse(manifest: RunManifest, loaded: Any, tool_plan: ToolPlan, name: str) -> None:
    expected = loaded.to_record()
    if (
        manifest.name != name
        or manifest.tool != loaded.tool
        or manifest.kind != tool_plan.kind
        or manifest.general_config_id != expected.general_config_id
        or manifest.model_config_id != expected.model_config_id
        or manifest.config.general_config_json != expected.general_config_json
        or manifest.config.model_config_json != expected.model_config_json
        or manifest.resources != loaded.model.resources.model_dump(mode="json")
        or manifest.workflow != {"engine": "snakemake", **tool_plan.workflow}
        or manifest.container != str(tool_plan.container)
        or len(manifest.tasks) != tool_plan.jobs
        or manifest.designs_per_task != tool_plan.designs_per_task
        or any(
            task.task_id != index
            or task.n_requested != tool_plan.designs_per_task
            or task.n_generated != tool_plan.generated_per_task
            or task.designs != f"tasks/{index:04d}/{tool_plan.designs_file}"
            for index, task in enumerate(manifest.tasks)
        )
        or set(manifest.inputs) != set(tool_plan.inputs)
        or set(manifest.provenance) != set(tool_plan.archives)
        or any(manifest.inputs[key] != digest_of(path) for key, path in tool_plan.inputs.items())
        or any(
            manifest.provenance[key].sha256 != _hash(path)
            for key, path in tool_plan.archives.items()
        )
    ):
        raise PlanError(
            f"existing run {manifest.directory} belongs to a different execution; use a new run name"
        )
    manifest.verify_inputs()


def build_plan(index_path: str | Path, site_path: str | Path, output_dir: str | Path) -> dict:
    """Create an immutable plan, or return the same verified plan on exact replay.

    Config paths follow the existing workflow's cwd-relative semantics. The
    output directory must be empty or contain this exact completed request.
    No scheduler is invoked and no database writer is opened.
    """
    index_path, site_path = Path(index_path).resolve(), Path(site_path).resolve()
    directory = Path(output_dir).resolve()
    plan_path = directory / "plan.json"
    if plan_path.exists():
        existing = load_plan(plan_path)
        if existing["request"]["index"] != str(index_path) or existing["request"]["site"] != str(
            site_path
        ):
            raise PlanError(
                "output directory already contains a different request; use a new output directory"
            )
        for source, expected in existing["request"]["authored"].items():
            if _hash(Path(source)) != expected:
                raise PlanError(
                    "authored configuration changed; use a new output directory and run names"
                )
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise PlanError(f"plan directory must be empty: {directory}")
    index = load_workflow_index(index_path)
    if index.campaign_plan is not None:
        raise PlanError("index is already frozen; use its existing campaign plan")
    site = load_yaml(site_path, Site)
    python, snakemake = _executables()
    run_root, database = index.run_root.resolve(), index.database.resolve()
    if site.controller.gpus >= site.max_total_gpus:
        raise PlanError("controller GPU allocation leaves no worker GPU budget")
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership prevents concurrent planners from rewriting snapshots.
    with (directory / ".planning").open("x"):
        pass
    snapshots = directory / "configs"
    authored = directory / "authored"
    snapshots.mkdir()
    authored.mkdir()
    originals: dict[str, str] = {}
    artifacts: dict[str, str] = {}

    def capture(source: Path, destination: Path) -> None:
        source = source.resolve()
        shutil.copyfile(source, destination)
        originals[str(source)] = _hash(destination)
        artifacts[str(destination)] = originals[str(source)]

    capture(index_path, authored / "index.yaml")
    capture(site_path, authored / "site.yaml")
    general_source = index.general_config.resolve()
    capture(general_source, authored / "general.yaml")
    general = load_yaml(authored / "general.yaml", GeneralConfig)
    general_snapshot = snapshots / "general.yaml"
    general_snapshot.write_text(yaml.safe_dump(_absolute_values(general.model_dump(mode="python"))))
    artifacts[str(general_snapshot)] = _hash(general_snapshot)
    prepared = []
    frozen_runs = []
    trees = {str(Path(__file__).resolve().parents[1]): _tree(Path(__file__).resolve().parents[1])}
    driver_root = Path(__file__).resolve().parents[3] / "drivers"
    if driver_root.is_dir():
        trees[str(driver_root)] = _tree(driver_root)
    assets: dict[str, dict] = {}
    references: dict[str, str] = {}
    for request in index.runs:
        if request.name in {".", ".."}:
            raise PlanError("run names may not be '.' or '..'")
        source = request.config.resolve()
        original = authored / f"model-{request.name}.yaml"
        capture(source, original)
        plugin = plugin_for(read_tool_name(original))
        model = load_yaml(original, plugin.config_type)
        snapshot = snapshots / f"model-{request.name}.yaml"
        snapshot.write_text(yaml.safe_dump(_absolute_values(model.model_dump(mode="python"))))
        artifacts[str(snapshot)] = _hash(snapshot)
        loaded = load_configs(general_snapshot, snapshot)
        tool_plan = plugin.tool_plan(loaded)
        if tool_plan.jobs <= 0 or tool_plan.designs_per_task <= 0:
            raise PlanError(f"{request.name}: plugin must plan positive task and request counts")
        resolved = plugin.resources(loaded)
        gpu = loaded.model.resources.gpus
        if not isinstance(resolved.get("gres"), str) or resolved["gres"].split(":")[-1] != str(gpu):
            raise PlanError(
                f"{request.name}: resolved GPU request disagrees with resource configuration"
            )
        if gpu + site.controller.gpus > site.max_total_gpus:
            raise PlanError(f"{request.name}: one worker plus controller exceeds max_total_gpus")
        resolved = {**resolved, "gpu": gpu}
        scope = _scope(tool_plan)
        run_dir = run_root / request.name
        manifest_path = run_dir / "run.json"
        if manifest_path.exists():
            manifest = RunManifest.read(manifest_path)
            if manifest.directory != run_dir:
                raise PlanError(
                    f"manifest points outside its requested run directory: {manifest_path}"
                )
            _check_reuse(manifest, loaded, tool_plan, request.name)
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise PlanError(f"run directory has data but no manifest: {run_dir}")
        if (run_dir / "campaign-binding.json").exists():
            raise PlanError(f"existing run {run_dir} is bound to a different campaign plan")
        prepared.append((request.name, loaded, tool_plan, run_dir, resolved, scope))
        frozen_runs.append({"name": request.name, "config": str(snapshot)})
        plugin_file = Path(inspect.getfile(type(plugin))).resolve()
        references[str(plugin_file)] = _hash(plugin_file)
        if (
            not plugin_file.is_relative_to(Path(__file__).resolve().parents[1])
            and (plugin_file.parent / "__init__.py").is_file()
        ):
            trees[str(plugin_file.parent)] = _tree(plugin_file.parent)
        for path in tool_plan.inputs.values():
            references[str(path.absolute())] = _hash(path)
        # Inputs consumed from archives are frozen too; changes to their live
        # originals also require a new plan rather than silently rebinding one.
        for path in tool_plan.archives.values():
            references[str(path.absolute())] = _hash(path)
        for path in tool_plan.workflow.get("parent_structures", {}).values():
            references[str(Path(path).absolute())] = _hash(Path(path))
        if str(tool_plan.container.absolute()) not in references:
            assets[str(tool_plan.container.absolute())] = _asset(tool_plan.container.absolute())
        runtime = getattr(loaded.model, "runtime", None)
        if runtime is not None:
            for key, value in runtime.model_dump(mode="python").items():
                if not isinstance(value, Path) or key in {
                    "scratch",
                    "work_root",
                    "node_tmp_root",
                    "msa_directory",
                }:
                    continue
                path = value.absolute()
                if key == "dev_source":
                    trees[str(path)] = _tree(path)
                elif str(path) not in references:
                    assets[str(path)] = _asset(path)
    root = Path(__file__).resolve().parents[3]
    workflow = root / "workflow" / "Snakefile"
    frozen_workflow = directory / "Snakefile"
    workflow_text = workflow.read_text()
    shutil.copyfile(workflow, frozen_workflow)
    artifacts[str(frozen_workflow)] = _hash(frozen_workflow)
    for path in (root / "pyproject.toml", root / "uv.lock", Path(snakemake)):
        if path.is_file():
            references[str(path)] = _hash(path)
    assets[python] = _asset(Path(python))
    frozen_index = directory / "workflow.yaml"
    frozen_index.write_text(
        yaml.safe_dump(
            {
                "database": str(database),
                "run_root": str(run_root),
                "general_config": str(general_snapshot),
                "runs": frozen_runs,
                "campaign_plan": str(plan_path),
            },
            sort_keys=False,
        )
    )
    artifacts[str(frozen_index)] = _hash(frozen_index)
    runs = []
    for name, loaded, tool_plan, run_dir, resolved, scope in prepared:
        # Reserve ownership before plan_run can create or rewrite any manifest.
        # Two planners with different output directories must not race here.
        run_dir.mkdir(parents=True, exist_ok=True)
        binding_path = run_dir / "campaign-binding.json"
        try:
            with binding_path.open("x") as handle:
                json.dump({"plan_dir": str(directory)}, handle)
        except FileExistsError as error:
            raise PlanError(
                f"existing run {run_dir} is bound to a different campaign plan"
            ) from error
        manifest = plan_run(loaded, tool_plan, run_dir, name=name)
        _check_reuse(manifest, loaded, tool_plan, name)
        manifest_path = run_dir / "run.json"
        manifest_hash = _hash(manifest_path)
        # Unlike config-only reuse, this also binds an existing execution to
        # the harness source and workflow that will interpret its manifest.
        binding = sha256_text(
            canonical_json(
                {
                    "manifest": manifest_hash,
                    "source": trees,
                    "workflow": sha256_text(workflow_text),
                    "resources": resolved,
                }
            )
        )
        ownership = {"execution_digest": binding, "plan_dir": str(directory)}
        write_json_atomic(binding_path, ownership)
        artifacts[str(binding_path)] = _hash(binding_path)
        artifacts[str(manifest_path)] = manifest_hash
        for archive in manifest.provenance.values():
            artifacts[str(manifest.path(archive.path))] = archive.sha256
        runs.append(
            {
                "name": name,
                "manifest": str(manifest_path),
                "tool": manifest.tool,
                "kind": manifest.kind.value,
                "tasks": len(manifest.tasks),
                "n_designs": scope["n_designs"],
                "n_predictions": scope["n_predictions"],
                "resources": resolved,
                "scope": scope,
            }
        )
    total_tasks = sum(run["tasks"] for run in runs)
    worker_budget = site.max_total_gpus - site.controller.gpus
    plan = {
        "schema_version": 1,
        "plan_dir": str(directory),
        "workflow_index": str(frozen_index),
        "snakefile": str(frozen_workflow),
        "python": python,
        "snakemake": snakemake,
        "database": str(database),
        "run_root": str(run_root),
        "site": site.model_dump(mode="json", exclude_none=True),
        "runs": runs,
        "request": {"index": str(index_path), "site": str(site_path), "authored": originals},
        "artifacts": artifacts,
        "referenced_inputs": references,
        "source_trees": trees,
        "runtime_assets": assets,
        "plugin_modules": os.environ.get("BINDOCRACY_PLUGINS", ""),
        "source_root": str(Path(__file__).resolve().parents[1]),
        "budget": {
            "total_tasks": total_tasks,
            "max_workers": min(site.max_workers, total_tasks),
            "controller_gpus": site.controller.gpus,
            "worker_gpu_budget": worker_budget,
            "max_total_gpus": site.max_total_gpus,
            "largest_worker_gpus": max(run["resources"]["gpu"] for run in runs),
            "largest_worker_cpus": max(run["resources"]["cpus_per_task"] for run in runs),
            "largest_worker_mem_mb": max(run["resources"]["mem_mb"] for run in runs),
        },
    }
    plan["digest"] = _identity(plan)
    # Verify the complete frozen closure before publishing its approval token.
    _verify(plan)
    write_json_atomic(plan_path, plan)
    (directory / ".planning").unlink()
    return plan


def _verify(plan: dict) -> None:
    if plan["source_root"] != str(Path(__file__).resolve().parents[1]):
        raise PlanError("plan requires a different bindocracy installation")
    if plan["plugin_modules"] != os.environ.get("BINDOCRACY_PLUGINS", ""):
        raise PlanError("BINDOCRACY_PLUGINS differs from the frozen plan")
    for group in ("artifacts", "referenced_inputs"):
        for path, expected in plan[group].items():
            if _hash(Path(path)) != expected:
                raise PlanError(f"frozen {group} changed: {path}")
    for path, expected in plan["source_trees"].items():
        if _tree(Path(path)) != expected:
            raise PlanError(f"execution source changed: {path}")
    for path, expected in plan["runtime_assets"].items():
        if _asset(Path(path)) != expected:
            raise PlanError(f"runtime asset identity changed: {path}")
    for run in plan["runs"]:
        RunManifest.read(run["manifest"]).verify_inputs()


def load_plan(path: str | Path, verify: bool = True) -> dict:
    """Read a plan and check its identity; by default check its execution closure.

    ``verify=False`` skips filesystem checks, not canonical identity validation.
    It is suitable for historical inspection, never for launching work.
    """
    try:
        plan = json.loads(Path(path).read_text())
        if not isinstance(plan, dict) or plan.get("schema_version") != 1:
            raise PlanError("unsupported campaign plan schema")
        if plan.get("digest") != _identity(plan):
            raise PlanError("campaign plan digest mismatch")
        if Path(path).resolve() != Path(plan["plan_dir"]) / "plan.json":
            raise PlanError("a frozen execution plan cannot be relocated")
        if verify:
            _verify(plan)
        return plan
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load campaign plan {path}: {error}") from error
