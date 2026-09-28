"""Read-only kit resolution and local, mutation-detecting execution locks.

These locks record approval, not publisher authentication or isolation. Source
and harness files are content hashed; potentially huge assets have explicitly
weaker filesystem-stat identities. Symlinks and special files are refused.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bindocracy.kits.bundle import KitRelease, inspect_kit
from bindocracy.kits.models import KitError, KitManifest, safe_relative
from bindocracy.store.records import canonical_json

_HARNESS_ROOT = Path(__file__).resolve().parents[3]
_HARNESS_FILES = ("drivers/bindocracy_io.py", "drivers/optimize/run_optimizer.py")
_OVERLAY_TARGET = "/opt/mosaic/src/mosaic"
_MAX_ENTRIES = 100_000
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_FLAG = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")
_DIGEST = r"^sha256:[0-9a-f]{64}$"


class KitLock(BaseModel):
    """Canonical local execution contract; digest covers every other field."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lock_schema: Literal[1] = 1
    digest: str = Field(pattern=_DIGEST)
    kit_root: Path
    bundle_digest: str = Field(pattern=_DIGEST)
    manifest: KitManifest
    files: dict[str, str]
    dependencies: dict[str, dict[str, Any]]
    parameters: dict[str, Any]
    args: tuple[str, ...]
    runtime: dict[str, Any]
    approved_digest: str | None = Field(default=None, pattern=_DIGEST)
    checks: list[dict[str, Any]]


class _Binding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["container", "weights", "source_overlay", "harness_module"]
    path: str
    interpreter: str | None = None


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _check(code: str, message: str, status: str = "ready") -> dict[str, Any]:
    return {"code": code, "status": status, "message": message, "details": {}}




def _local_path(path: Path) -> Path:
    """Refuse symlinks in every existing component before canonicalizing."""
    absolute = Path(os.path.abspath(path))
    if any(char in str(absolute) for char in (":", ",", "\n", "\r", "\x00")):
        raise KitError(f"Dependency path cannot safely be bound read-only: {absolute}")
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise KitError(f"Symlink dependency paths are not supported: {component}")
    if not absolute.exists():
        raise KitError(f"Dependency path is missing: {absolute}; update the site bindings")
    return absolute


def _file_digest(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise KitError(f"Expected a regular dependency file: {path}")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stat_record(before) != _stat_record(os.fstat(stream.fileno())):
            raise KitError(f"Dependency changed while being opened: {path}")
        size = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > before.st_size or size > _MAX_SOURCE_BYTES:
                raise KitError(f"Dependency changed or exceeded content-hash limit: {path}")
            digest.update(chunk)
        if size != before.st_size or _stat_record(before) != _stat_record(os.fstat(stream.fileno())):
            raise KitError(f"Dependency changed while being hashed: {path}")
    if _stat_record(before) != _stat_record(path.stat(follow_symlinks=False)):
        raise KitError(f"Dependency changed while being hashed: {path}")
    return "sha256:" + digest.hexdigest()


def _stat_record(info: os.stat_result) -> dict[str, int]:
    return {
        "mode": info.st_mode,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _identity(path: Path, kind: str, files: tuple[str, ...] = ()) -> dict[str, Any]:
    path = _local_path(path)
    if kind == "harness_module":
        inventory = {}
        total_bytes = 0
        for relative in sorted({*_HARNESS_FILES, *files}):
            safe_relative(relative)
            file_path = _local_path(path / relative)
            total_bytes += file_path.stat().st_size
            if total_bytes > _MAX_SOURCE_BYTES or len(inventory) >= _MAX_ENTRIES:
                raise KitError("Declared harness files exceed content-hash inventory limits")
            inventory[relative] = _file_digest(file_path)
        return {"method": "sha256-declared-files", "files": inventory, "digest": _digest(inventory)}
    content = kind == "source_overlay"
    if content and not path.is_dir():
        raise KitError(f"Source overlay must be a directory: {path}")
    inventory: dict[str, Any] = {}
    pending = [(".", path)]
    total_bytes = 0
    while pending:
        relative, current = pending.pop()
        info = current.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise KitError(f"Symlink dependency entries are not supported: {current}")
        if stat.S_ISDIR(info.st_mode):
            inventory[relative] = {"kind": "directory"} if content else _stat_record(info)
            with os.scandir(current) as entries:
                children = sorted(entries, key=lambda item: item.name, reverse=True)
            for child in children:
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                safe_relative(child_relative)
                pending.append((child_relative, current / child.name))
        elif stat.S_ISREG(info.st_mode):
            if content:
                total_bytes += info.st_size
                if total_bytes > _MAX_SOURCE_BYTES:
                    raise KitError("Source overlay exceeds 512 MiB content-hash limit")
                inventory[relative] = {"kind": "file", "digest": _file_digest(current)}
            else:
                inventory[relative] = _stat_record(info)
        else:
            raise KitError(f"Nonregular dependency entry is not supported: {current}")
        if len(inventory) + len(pending) > _MAX_ENTRIES:
            raise KitError(f"Dependency inventory exceeds {_MAX_ENTRIES} entries: {path}")
    return {
        "method": "sha256-tree" if content else "filesystem-stat",
        "entries": inventory,
        "digest": _digest(inventory),
    }


def _bindings(path: Path | None) -> dict[str, _Binding]:
    if path is None:
        return {}
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise KitError(f"Cannot read kit bindings {path}: {error}") from error
    if not isinstance(raw, dict) or type(raw.get("bindings_schema")) is not int or raw["bindings_schema"] != 1:
        raise KitError("Bindings require bindings_schema: 1 and named binding mappings")
    result = {}
    for name, value in raw.items():
        if name == "bindings_schema":
            continue
        if not isinstance(name, str):
            raise KitError("Binding names must be strings")
        try:
            binding = _Binding.model_validate(value)
        except ValidationError as error:
            raise KitError(f"Invalid binding {name!r}: {error}") from error
        # The harness dependency is always controlled by the harness itself.
        if name == "bindocracy-io":
            if binding.kind != "harness_module":
                raise KitError("bindocracy-io is a harness_module, not a user dependency")
            continue
        if not binding.path or "://" in binding.path:
            raise KitError(f"Binding {name!r} requires an explicit local filesystem path")
        candidate = Path(binding.path)
        if not candidate.is_absolute():
            candidate = path.absolute().parent / candidate
        result[name] = binding.model_copy(update={"path": str(candidate)})
    return result


def _parameters(manifest: KitManifest, supplied: dict | None) -> tuple[dict[str, Any], list[str]]:
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict) or any(not isinstance(key, str) for key in supplied):
        raise KitError("Parameters must be a mapping with string names")
    known = set(manifest.args) | set(manifest.parameters)
    reserved = {"inputs", "outputs", "context"}
    collisions = reserved & (known | {flag.removeprefix("--") for flag in manifest.dependency_args.values()})
    if collisions:
        raise KitError(f"Kit arguments collide with harness-owned flags: {', '.join(sorted(collisions))}")
    unknown = set(supplied) - known
    if unknown:
        raise KitError(f"Unknown kit parameters: {', '.join(sorted(unknown))}")
    merged = {**manifest.args, **supplied}
    arguments: list[str] = []
    for key, spec in manifest.parameters.items():
        if spec.required and key not in merged:
            raise KitError(f"Required kit parameter {key!r} is missing")
    dependency_flags = set(manifest.dependency_args.values())
    for key, value in sorted(merged.items()):
        if not _FLAG.fullmatch(key):
            raise KitError(f"Unsafe parameter flag name: {key!r}")
        if f"--{key}" in dependency_flags:
            raise KitError(f"Parameter {key!r} collides with a dependency argument")
        spec = manifest.parameters.get(key)
        expected = spec.type if spec is not None else {
            str: "string", int: "integer", float: "number", bool: "boolean"
        }.get(type(manifest.args[key]))
        valid = {
            "string": type(value) is str,
            "integer": type(value) is int,
            "number": type(value) in (int, float),
            "boolean": type(value) is bool,
        }.get(expected, False)
        if spec is None:
            valid = valid and type(value) is type(manifest.args[key])
        if not valid or (type(value) is float and not math.isfinite(value)):
            raise KitError(f"Parameter {key!r} must be {expected}, got {value!r}")
        if isinstance(value, str) and "\x00" in value:
            raise KitError(f"Parameter {key!r} contains a NUL byte")
        if spec is not None:
            try:
                spec.validate_value(value)
            except ValueError as error:
                raise KitError(f"Invalid parameter {key!r}: {error}") from error
        if type(value) is bool:
            if value:
                arguments.append(f"--{key}")
        else:
            arguments.extend((f"--{key}", str(value)))
    return dict(sorted(merged.items())), arguments


def _resolve(
    release: KitRelease,
    bindings: dict[str, _Binding],
    parameters: dict | None,
    approved_digest: str | None,
) -> KitLock:
    if approved_digest is not None and approved_digest != release.digest:
        raise KitError("Source approval must equal the inspected bundle digest exactly")
    manifest = release.manifest
    merged, arguments = _parameters(manifest, parameters)
    dependencies: dict[str, dict[str, Any]] = {}
    runtime: dict[str, Any] = {}
    checks = [_check("kit.bundle", "Complete kit file inventory is content hashed")]
    harness_files = set(_HARNESS_FILES)
    for requirement in manifest.requires:
        name = requirement.name
        if requirement.kind == "harness_module":
            if name != "bindocracy-io" or requirement.provided_by != "harness":
                raise KitError(f"Unsupported harness module {name!r}; only harness-provided bindocracy-io is available")
            harness_files.update(requirement.verify.files)
            binding = _Binding(kind="harness_module", path=str(_HARNESS_ROOT))
        else:
            binding = bindings.get(name)
        if binding is None:
            if requirement.optional:
                checks.append(_check(f"kit.dependency.{name}", f"Optional dependency {name!r} is unbound and omitted"))
                continue
            raise KitError(f"Missing required dependency {name!r} ({requirement.kind}); add kind and path to a bindings_schema: 1 site bindings file")
        if binding.kind != requirement.kind:
            raise KitError(f"Dependency {name!r} needs kind {requirement.kind!r}, not {binding.kind!r}")
        path = _local_path(Path(binding.path))
        if requirement.target and (requirement.kind != "source_overlay" or requirement.target != _OVERLAY_TARGET):
            raise KitError(f"Unsupported runtime dependency target {requirement.target!r}; only {_OVERLAY_TARGET} source overlays are supported")
        if binding.interpreter is not None and requirement.kind != "container":
            raise KitError(f"Only container bindings may specify an interpreter: {name}")
        interpreter = binding.interpreter
        if interpreter is not None and (not interpreter or any(char.isspace() for char in interpreter) or "\x00" in interpreter or interpreter.startswith("-")):
            raise KitError(f"Container interpreter must be one explicit executable token: {name}")
        for relative in requirement.verify.files:
            safe_relative(relative)
            _local_path(path / relative)
        identity = _identity(path, requirement.kind, tuple(requirement.verify.files))
        dependency: dict[str, Any] = {"path": str(path), "kind": requirement.kind, "identity": identity}
        if interpreter is not None:
            dependency["interpreter"] = interpreter
        if requirement.target:
            if "dev_source" in runtime:
                raise KitError("At most one source overlay may target the primary runtime")
            runtime["dev_source"] = str(path)
            dependency["target"] = requirement.target
        if requirement.provides == "interpreter":
            if requirement.kind != "container":
                raise KitError(f"Only a container may provide the runtime interpreter: {name}")
            if "container" in runtime:
                raise KitError("At most one container may provide the primary runtime interpreter")
            if interpreter is None:
                raise KitError(f"Primary container {name!r} needs an explicit interpreter in site bindings")
            runtime.update(container=str(path), container_python=interpreter)
        dependencies[name] = dependency
        method = identity["method"]
        checks.append(_check(f"kit.dependency.{name}", f"Dependency {name!r} present; identity method {method}"))
        if method == "filesystem-stat":
            checks.append(_check(f"kit.content.{name}", "Filesystem stat inventory detects local drift but is not a content hash or authenticity proof", "unknown"))
        if requirement.verify.import_:
            checks.append(_check(f"kit.import.{name}", f"Import {requirement.verify.import_!r} is unverified; no code was executed", "unknown"))
        if requirement.version:
            checks.append(_check(f"kit.version.{name}", f"Author version hint {requirement.version!r} is unverified compatibility", "unknown"))
        if requirement.kind == "container":
            checks.append(_check(f"kit.container.{name}", "Container executability and runtime compatibility are unverified; no runtime probe was executed", "unknown"))
    dependencies["bindocracy-io"] = {
        "path": str(_HARNESS_ROOT),
        "kind": "harness_module",
        "identity": _identity(_HARNESS_ROOT, "harness_module", tuple(harness_files)),
    }
    if "dev_source" in runtime and "container" not in runtime:
        raise KitError("A runtime source overlay requires a primary interpreter container")
    for name, flag in sorted(manifest.dependency_args.items()):
        if not flag.startswith("--") or not _FLAG.fullmatch(flag[2:]):
            raise KitError(f"Unsafe dependency argument flag: {flag!r}")
        if name in dependencies:
            arguments.extend((flag, dependencies[name]["path"]))
    if "container" in runtime:
        checks.append(_check("kit.python", f"Container Python constraint {manifest.python!r} is unverified", "unknown"))
    else:
        host = ".".join(map(str, sys.version_info[:3]))
        if host not in SpecifierSet(manifest.python):
            raise KitError(f"Host Python {host} does not satisfy kit constraint {manifest.python}")
        checks.append(_check("kit.python", f"Host Python {host} satisfies {manifest.python}"))
    checks.append(_check("kit.approval", "Explicit source digest approved; this is not isolation or publisher authentication" if approved_digest else "Source execution is unapproved; explicit matching bundle digest approval is required", "ready" if approved_digest else "unknown"))
    payload = {
        "lock_schema": 1,
        "kit_root": str(release.root),
        "bundle_digest": release.digest,
        "manifest": manifest.model_dump(mode="json"),
        "files": release.files,
        "dependencies": dependencies,
        "parameters": merged,
        "args": arguments,
        "runtime": runtime,
        "approved_digest": approved_digest,
        "checks": checks,
    }
    payload["digest"] = _digest(payload)
    return KitLock.model_validate_json(canonical_json(payload))


def resolve_kit(
    source: Path,
    bindings: Path | None = None,
    parameters: dict | None = None,
    approved_digest: str | None = None,
) -> KitLock:
    """Resolve local declarations without writing, executing, importing, or fetching."""
    try:
        return _resolve(inspect_kit(Path(source)), _bindings(bindings), parameters, approved_digest)
    except (OSError, ValidationError) as error:
        raise KitError(f"Cannot resolve kit: {error}") from error


def verify_lock(lock: KitLock, *, require_approval: bool = True) -> None:
    """Recheck canonical identity and the complete source/dependency closure."""
    try:
        payload = lock.model_dump(mode="json", exclude={"digest"})
        if _digest(payload) != lock.digest:
            raise KitError("Kit lock digest does not match its canonical contents")
        if not lock.kit_root.is_absolute():
            raise KitError("Kit lock source root must be an absolute path")
        if require_approval and lock.approved_digest != lock.bundle_digest:
            raise KitError("Kit execution requires explicit matching source approval")
        release = inspect_kit(lock.kit_root)
        if release.root != lock.kit_root or release.digest != lock.bundle_digest or release.files != lock.files:
            raise KitError("Kit source inventory or bundle digest changed since resolution")
        if release.manifest.model_dump(mode="json") != lock.manifest.model_dump(mode="json"):
            raise KitError("Kit lock manifest differs from the source bundle")
        bindings = {
            name: _Binding.model_validate({key: value for key, value in dependency.items() if key in {"kind", "path", "interpreter"}})
            for name, dependency in lock.dependencies.items()
            if name != "bindocracy-io"
        }
        current = _resolve(release, bindings, lock.parameters, lock.approved_digest)
        if current.digest != lock.digest:
            raise KitError("Kit dependency identity or resolved execution contract changed since resolution")
    except (OSError, ValidationError, TypeError, KeyError) as error:
        raise KitError(f"Cannot verify kit lock: {error}") from error


def load_lock(path: Path, *, verify: bool = True, require_approval: bool = True) -> KitLock:
    """Load a strict JSON lock; execution callers must retain default verification."""
    try:
        lock = KitLock.model_validate_json(Path(path).read_text())
    except (OSError, UnicodeError, ValidationError) as error:
        raise KitError(f"Cannot read kit lock {path}: {error}") from error
    if verify:
        verify_lock(lock, require_approval=require_approval)
    elif require_approval and lock.approved_digest != lock.bundle_digest:
        raise KitError("Kit execution requires explicit matching source approval")
    return lock


def write_lock(lock: KitLock, path: Path) -> Path:
    """Atomically create an immutable lock, or accept an identical existing one."""
    path = Path(path)
    verify_lock(lock, require_approval=False)
    destination = path.resolve()
    protected_roots = [lock.kit_root] + [
        Path(dependency["path"])
        for dependency in lock.dependencies.values()
        if dependency["kind"] != "harness_module"
    ]
    if any(destination == root or destination.is_relative_to(root) for root in protected_roots):
        raise KitError("Kit locks must be written outside the immutable bundle and dependency trees")
    payload = canonical_json(lock.model_dump(mode="json")) + "\n"
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise KitError(f"Refusing symlink kit lock destination: {path}")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".kit-lock-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        verify_lock(lock, require_approval=False)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text() != payload:
                raise KitError(f"Refusing to replace a different existing kit lock: {path}") from None
        return path
    except OSError as error:
        raise KitError(f"Cannot write kit lock {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
