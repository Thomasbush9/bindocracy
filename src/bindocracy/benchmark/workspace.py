"""Organizer-captured content identities and non-destructive agent workspaces.

These are integrity checks, not an isolation boundary or scheduler qualification.
Source paths remain organizer-owned; this module never changes their permissions.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, TypeAdapter

from bindocracy.benchmark.models import (
    AssetIdentity,
    EnvironmentManifest,
    TaskSpec,
    WorkspaceManifest,
)
from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import parse_hotspots, read_single_fasta, require_alignment_of
from bindocracy.publication import publish_directory
from bindocracy.runs.index import WorkflowIndex
from bindocracy.store import create_database
from bindocracy.tools import plugin_for

# These paths are writable runtime destinations, not scientific inputs.
_WRITABLE_PATHS = {"scratch", "work_root", "node_tmp_root", "database"}
# Only interpreter caches are excluded from a code tree's content identity.
_CODE_CACHE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_DOCUMENT = TypeAdapter(dict[str, Any])


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read document {path}: {error}") from error
    return _DOCUMENT.validate_python(value)


def _path(value: str | Path, base: Path) -> Path:
    path = Path(os.path.abspath(base / value))
    # Do not resolve away a symlink before checking it, including parent links.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink is not an inventoried input: {part}")
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(path: Path, *, code: bool = False) -> str:
    def fail_walk(error: OSError) -> None:
        raise error

    records: list[list[str]] = []
    for parent, directories, files in os.walk(path, followlinks=False, onerror=fail_walk):
        directory = Path(parent)
        # Check links before any cache filtering, so caches cannot hide escapes.
        for name in directories + files:
            child = directory / name
            if child.is_symlink():
                raise ValueError(f"Symlink is not an inventoried input: {child}")
        if code:
            directories[:] = [name for name in directories if name not in _CODE_CACHE_NAMES]
        for name in sorted(directories):
            records.append([str((directory / name).relative_to(path)), "directory"])
        for name in sorted(files):
            child = directory / name
            if not stat.S_ISREG(child.stat().st_mode):
                raise ValueError(f"Input is not a regular file: {child}")
            records.append([str(child.relative_to(path)), _file_digest(child)])
    return _digest(sorted(records))


def identify_asset(path: str | Path) -> AssetIdentity:
    """Hash every regular byte and relative name; reject links and special files."""
    path = _path(path, Path.cwd())
    if path.is_dir():
        return AssetIdentity(path=path, kind="directory", sha256=_tree_digest(path))
    if not path.is_file():
        raise ValueError(f"Input is missing or not a regular file: {path}")
    return AssetIdentity(path=path, kind="file", sha256=_file_digest(path))


def _harness_digest(root: Path) -> str:
    """Identity of the executable source, drivers, workflow and packaging metadata."""
    root = _path(root, Path.cwd())
    records = {}
    for name in ("src", "drivers", "workflow", "pyproject.toml"):
        path = _path(root / name, root)
        if path.is_dir():
            records[name] = _tree_digest(path, code=True)
        elif path.is_file():
            records[name] = _file_digest(path)
        else:
            raise ValueError(f"Harness checkout lacks required source: {path}")
    return _digest(records)


def _resolve_model(value: Any, base: Path) -> Any:
    if isinstance(value, Path):
        return _path(value, base)
    if isinstance(value, BaseModel):
        return type(value).model_validate({
            name: _resolve_model(getattr(value, name), base)
            for name in type(value).model_fields
        })
    if isinstance(value, dict):
        return {key: _resolve_model(item, base) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_resolve_model(item, base) for item in value)
    return value


def _input_paths(value: Any, field: str = "") -> set[Path]:
    if isinstance(value, Path):
        return set() if field in _WRITABLE_PATHS else {value}
    if isinstance(value, BaseModel):
        return set().union(*(
            _input_paths(getattr(value, name), name) for name in type(value).model_fields
        ))
    if isinstance(value, dict):
        return set().union(*(_input_paths(item, str(key)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return set().union(*(_input_paths(item, field) for item in value))
    return set()


def _tool_config(tool: str, path: Path) -> BaseModel:
    try:
        plugin = plugin_for(tool)
    except KeyError as error:
        raise ValueError(f"Unavailable allowed tool: {tool}") from error
    model = _resolve_model(plugin.config_type.model_validate(_mapping(path)), path.parent)
    if model.tool != tool:
        raise ValueError(f"Template {path} names {model.tool!r}, not {tool!r}")
    return model


def _referenced_paths(path: Path) -> set[Path]:
    """Inventory path-bearing external specs as well as their parent configs.

    Tool specs are upstream YAML/JSON, not harness models. Existing local file
    references are discovered regardless of key; explicit path/file fields must
    resolve. String scalars such as sequences and scientific labels are not paths.
    """
    if path.suffix.lower() not in {".yaml", ".yml", ".json"} or not path.is_file():
        return set()
    value = yaml.safe_load(path.read_text())
    found: set[Path] = set()
    if isinstance(value, dict) and {"entries", "n_designs", "digest"} <= value.keys():
        sibling = path.with_suffix(".fasta")
        if not sibling.is_file():
            raise ValueError(f"Frozen design set lacks sibling FASTA: {sibling}")
        found.add(sibling)

    def walk(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for name, child in item.items():
                walk(child, str(name))
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif isinstance(item, str) and item and "\n" not in item and "://" not in item:
            if key in _WRITABLE_PATHS:
                return
            candidate = _path(item, path.parent)
            explicit = key in {"path", "file", "filename", "pdb", "cif", "msa"} or key.endswith(("_path", "_file", "_fasta", "_pdb", "_cif"))
            try:
                exists = candidate.exists()
            except OSError:
                exists = False
            if exists:
                found.add(candidate)
            elif explicit:
                raise ValueError(f"Missing referenced input in {path}: {key}={item}")

    walk(value)
    return found


def _dependencies(paths: set[Path]) -> set[Path]:
    pending = list(paths)
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        # Fail on absent and special inputs before attempting YAML discovery.
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Missing input: {path}")
        pending.extend(_referenced_paths(path) - seen)
    return seen


def _required_environment_paths(environment: EnvironmentManifest) -> set[Path]:
    paths: set[Path] = set(environment.tool_configs.values())
    for tool, path in environment.tool_configs.items():
        implementation = _path(inspect.getfile(type(plugin_for(tool))), Path.cwd())
        if not implementation.is_relative_to(environment.harness_path / "src" / "bindocracy"):
            raise ValueError(f"Benchmark tools must use the captured harness implementation: {tool}")
        paths.update(_input_paths(_tool_config(tool, path)))
    for name, binding in environment.bindings.items():
        if binding.get("kind") not in {"container", "weights", "source_overlay", "harness_module"}:
            raise ValueError(f"Invalid binding kind: {name}")
        if set(binding) - {"kind", "path", "interpreter"} or not binding.get("path"):
            raise ValueError(f"Binding requires kind and explicit local path: {name}")
        paths.add(Path(binding["path"]))
    return _dependencies(paths)


def _check_covered(paths: set[Path], assets: dict[str, AssetIdentity]) -> None:
    for path in paths:
        if not any(path == asset.path or (asset.kind == "directory" and path.is_relative_to(asset.path)) for asset in assets.values()):
            raise ValueError(f"Mutable input omitted from content inventory: {path}")


def _verify_environment(environment: EnvironmentManifest) -> None:
    if _harness_digest(environment.harness_path) != environment.harness_sha256:
        raise ValueError("Harness content identity changed")
    installed = Path(__file__).resolve().parents[3]
    if installed != environment.harness_path and _harness_digest(installed) != environment.harness_sha256:
        raise ValueError("Captured harness differs from the executing harness")
    if identify_asset(environment.lock_path).sha256 != environment.lock_sha256:
        raise ValueError("uv.lock content identity changed")
    for name, expected in environment.assets.items():
        if identify_asset(expected.path) != expected:
            raise ValueError(f"Environment asset content changed: {name} ({expected.path})")
    _check_covered(_required_environment_paths(environment), environment.assets)


def _environment_paths(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    raw = dict(raw)
    for key in ("harness_path", "lock_path"):
        if key in raw:
            raw[key] = _path(raw[key], base)
    raw["assets"] = {
        name: {**asset, "path": _path(asset["path"], base)}
        for name, asset in raw.get("assets", {}).items()
    }
    raw["tool_configs"] = {name: _path(path, base) for name, path in raw.get("tool_configs", {}).items()}
    raw["bindings"] = {
        name: {**binding, "path": str(_path(binding["path"], base))}
        for name, binding in raw.get("bindings", {}).items()
    }
    return raw


def load_environment(path: str | Path, verify: bool = True) -> EnvironmentManifest:
    path = _path(path, Path.cwd())
    environment = EnvironmentManifest.model_validate(_environment_paths(_mapping(path), path.parent))
    if verify:
        _verify_environment(environment)
    return environment


def _publish_file(path: Path, content: str) -> None:
    """Publish a complete file without overwriting even an existing symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_environment(spec_path: str | Path, output: str | Path) -> EnvironmentManifest:
    """Capture an organizer-authored spec. This performs no runtime qualification."""
    source = _path(spec_path, Path.cwd())
    raw = _mapping(source)
    asset_specs = raw.pop("assets", {})
    raw["assets"] = {
        name: identify_asset(_path(asset["path"] if isinstance(asset, dict) else asset, source.parent)).model_dump(mode="json")
        for name, asset in asset_specs.items()
    }
    raw = _environment_paths(raw, source.parent)
    harness = Path(raw["harness_path"])
    lock = Path(raw["lock_path"])
    if lock.name != "uv.lock":
        raise ValueError("lock_path must identify the harness uv.lock")
    raw["harness_sha256"] = _harness_digest(harness)
    raw["lock_sha256"] = identify_asset(lock).sha256
    raw["qualification"] = {"status": "not_qualified_by_capture", "evidence": raw.get("qualification", {})}
    environment = EnvironmentManifest.model_validate(raw)
    assets = dict(environment.assets)
    # Named user assets plus all template dependencies; no omitted mutable paths.
    for index, path in enumerate(sorted(_required_environment_paths(environment))):
        key = f"captured:{index}"
        if key in assets:
            raise ValueError(f"Reserved asset name: {key}")
        assets[key] = identify_asset(path)
    environment = environment.model_copy(update={"assets": assets})
    _verify_environment(environment)
    destination = _path(output, Path.cwd())
    protected = [environment.harness_path / part for part in ("src", "drivers", "workflow")]
    protected.extend(asset.path for asset in environment.assets.values() if asset.kind == "directory")
    if any(destination.is_relative_to(path) for path in protected):
        raise ValueError("Environment output conflicts with an inventoried source directory")
    _publish_file(destination, environment.model_dump_json(indent=2) + "\n")
    return environment


def load_task(path: str | Path) -> TaskSpec:
    path = _path(path, Path.cwd())
    task = _resolve_model(TaskSpec.model_validate(_mapping(path)), path.parent)
    for target_path in _input_paths(task.target):
        if identify_asset(target_path).kind != "file":
            raise ValueError(f"Target representation must be a file: {target_path}")
    sequence = read_single_fasta(task.target.sequence_fasta)
    parse_hotspots(task.target.hotspots)
    if task.target.msa is not None:
        require_alignment_of(task.target.msa, sequence, described_as="task target MSA")
    return task


def _task_digest(task: TaskSpec, assets: dict[str, AssetIdentity]) -> str:
    """Task equivalence depends on input bytes, not per-agent filesystem paths."""
    payload = task.model_dump(mode="json")
    for name, value in task.target:
        if isinstance(value, Path):
            identity = next((asset for asset in assets.values() if asset.path == value), None)
            if identity is None:
                raise ValueError(f"Target input is absent from content inventory: {name}")
            payload["target"][name] = {"kind": identity.kind, "sha256": identity.sha256}
    return _digest(payload)


def _check_tools(task: TaskSpec, environment: EnvironmentManifest) -> None:
    for tool in task.allowed_tools:
        if tool not in environment.tool_configs:
            raise ValueError(f"Allowed tool has no captured installed template: {tool}")
        _tool_config(tool, environment.tool_configs[tool])


def load_workspace(path: str | Path) -> WorkspaceManifest:
    path = _path(path, Path.cwd())
    if path.is_dir():
        path /= "workspace.json"
    raw = _mapping(path)
    base = path.parent
    for key in ("root", "database", "general_config", "site_config", "bindings"):
        raw[key] = _path(raw[key], base)
    raw["environment"] = _environment_paths(raw["environment"], base)
    raw["input_assets"] = {name: {**asset, "path": _path(asset["path"], base)} for name, asset in raw["input_assets"].items()}
    raw["task"] = _resolve_model(TaskSpec.model_validate(raw["task"]), base)
    workspace = WorkspaceManifest.model_validate(raw)
    if workspace.root != base:
        raise ValueError("Workspace root does not match manifest location")
    for field in ("database", "general_config", "site_config", "bindings"):
        value = getattr(workspace, field)
        if not value.is_relative_to(base) or not value.is_file():
            raise ValueError(f"Workspace {field} must be an existing local file")
    for name, asset in workspace.input_assets.items():
        if identify_asset(asset.path) != asset:
            raise ValueError(f"Task input content changed: {name} ({asset.path})")
    _check_covered(_input_paths(workspace.task.target), workspace.input_assets)
    if _task_digest(workspace.task, workspace.input_assets) != workspace.task_digest:
        raise ValueError("Task digest differs from captured inputs")
    if _digest(workspace.environment.model_dump(mode="json")) != workspace.environment_digest:
        raise ValueError("Environment manifest digest changed")
    _verify_environment(workspace.environment)
    _check_tools(workspace.task, workspace.environment)
    prompt = _path(base / "prompt.txt", base)
    if prompt.read_text() != workspace.task.prompt:
        raise ValueError("Workspace prompt differs from assigned task")
    return workspace




def initialize_workspace(
    environment_path: str | Path,
    task_path: str | Path,
    root: str | Path,
    *,
    attempt_id: str | None = None,
) -> WorkspaceManifest:
    """Create a complete workspace or leave the destination absent/unchanged."""
    root = _path(root, Path.cwd())
    if root.exists():
        raise FileExistsError(f"Workspace root already exists: {root}")
    environment_path = _path(environment_path, Path.cwd())
    task_path = _path(task_path, Path.cwd())
    environment = load_environment(environment_path)
    task = load_task(task_path)
    _check_tools(task, environment)
    assets = {"task_document": identify_asset(task_path)}
    for name, value in task.target:
        if isinstance(value, Path):
            assets[f"target:{name}"] = identify_asset(value)
    # A workspace inside inventoried source would mutate that source on creation.
    source_assets = [*assets.values(), *environment.assets.values()]
    if root.is_relative_to(environment.harness_path):
        # Repository examples/users may keep workspaces outside executable trees.
        protected = [environment.harness_path / part for part in ("src", "drivers", "workflow")]
    else:
        protected = []
    protected.extend(asset.path for asset in source_assets if asset.kind == "directory")
    if any(root.is_relative_to(path) for path in protected):
        raise ValueError("Workspace destination conflicts with inventoried source directory")
    general = GeneralConfig.model_validate({
        "schema_version": 1,
        "campaign": {"name": task.task_id},
        "target": task.target.model_dump(mode="json"),
        "cluster": {"executor": "slurm", "account": environment.site.controller.account,
                    "default_partition": environment.worker_partition or environment.site.controller.partition},
    })
    workspace = WorkspaceManifest(
        attempt_id=attempt_id or uuid.uuid4().hex,
        root=root, database=root / "campaign.duckdb", general_config=root / "general.yaml",
        site_config=root / "site.yaml", bindings=root / "bindings.yaml", task=task,
        environment=environment, input_assets=assets, task_digest=_task_digest(task, assets),
        environment_digest=_digest(environment.model_dump(mode="json")), created_at=datetime.now(UTC),
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        for name in ("configs", "kits", "plans", "runs", "logs", "submissions"):
            (staging / name).mkdir()
        create_database(staging / workspace.database.name)
        (staging / "general.yaml").write_text(yaml.safe_dump(general.model_dump(mode="json"), sort_keys=False))
        (staging / "site.yaml").write_text(yaml.safe_dump(environment.site.model_dump(mode="json"), sort_keys=False))
        (staging / "bindings.yaml").write_text(yaml.safe_dump({"bindings_schema": 1, **environment.bindings}, sort_keys=False))
        for tool in task.allowed_tools:
            model = _tool_config(tool, environment.tool_configs[tool])
            if "database" in type(model).model_fields:
                model = model.model_copy(update={"database": workspace.database})
            if not tool or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in tool):
                raise ValueError(f"Unsafe tool name: {tool}")
            (staging / "configs" / f"{tool}.yaml").write_text(yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))
        index = WorkflowIndex.model_validate({
            "database": str(workspace.database), "run_root": str(root / "runs"),
            "general_config": str(workspace.general_config), "workspace": str(root),
            "attempt_id": workspace.attempt_id,
            "runs": [{"name": f"{tool}-example", "config": str(root / "configs" / f"{tool}.yaml")}
                     for tool in task.allowed_tools],
        })
        (staging / "workflow.yaml").write_text(yaml.safe_dump(index.model_dump(mode="json"), sort_keys=False))
        (staging / "prompt.txt").write_text(task.prompt)
        (staging / "task.json").write_text(task.model_dump_json(indent=2) + "\n")
        (staging / "environment.json").write_text(environment.model_dump_json(indent=2) + "\n")
        (staging / "workspace.json").write_text(workspace.model_dump_json(indent=2) + "\n")
        # Recheck source bytes after copying templates, before publishing anything.
        _verify_environment(environment)
        for asset in assets.values():
            if identify_asset(asset.path) != asset:
                raise ValueError(f"Task input changed during initialization: {asset.path}")
        publish_directory(staging, root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return workspace
