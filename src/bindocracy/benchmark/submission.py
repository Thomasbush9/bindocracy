"""Portable, score-free candidate bundles and organizer-controlled acceptance.

Integrity is not authenticity, isolation, or evidence that a molecule binds.
The control directory must be writable only by the organizer.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from bindocracy.benchmark.models import AttemptRecord, SubmissionPolicy, TaskSpec
from bindocracy.publication import publish_directory
from bindocracy.runs.designset import DesignSet, build_design_set
from bindocracy.runs.inputs import TargetDigest
from bindocracy.runs.selection import read_only, resolve_run_ids
from bindocracy.store.query import DesignQuery, select_designs
from bindocracy.store.records import canonical_json, sha256_text, stable_id
from bindocracy.store.reporting import rows, target_digest


class SubmissionError(ValueError):
    """A bundle or organizer acceptance was refused."""


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_AMINO = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+\Z")


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + "\n"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False, default=_json_default)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SubmissionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: str | bytes) -> Any:
    def invalid(value):
        raise SubmissionError(f"nonfinite JSON constant: {value}")
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as error:
        raise SubmissionError(f"invalid JSON: {error}") from error


def _relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise SubmissionError(f"invalid relative path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise SubmissionError(f"unsafe relative path: {name!r}")
    return name


@contextmanager
def _open_regular(path: Path):
    """Open each path component with O_NOFOLLOW, including raced ancestors."""
    absolute = Path(os.path.abspath(path))
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise SubmissionError(f"not a regular file: {path}")
            yield handle
    except OSError as error:
        raise SubmissionError(f"cannot safely read {path}: {error}") from error
    finally:
        os.close(fd)


def _read(path: Path) -> bytes:
    with _open_regular(path) as handle:
        return handle.read()


def _copy_file(source: Path, destination: Path, expected: str | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with _open_regular(source) as src, destination.open("xb") as dst:
        before = os.fstat(src.fileno())
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
        after = os.fstat(src.fileno())
        dst.flush()
        os.fsync(dst.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise SubmissionError(f"source changed during copy: {source}")
    result = digest.hexdigest()
    if expected is not None and result != expected.removeprefix("sha256:"):
        raise SubmissionError(f"content hash mismatch: {source}")
    return result


def _inventory(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise SubmissionError(f"not a plain bundle directory: {root}")
    found = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise SubmissionError(f"symlink or special file in bundle: {path}")
            if stat.S_ISREG(mode):
                found.add(_relative(path.relative_to(root).as_posix()))
    return found


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(value))


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_dump(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _collect_provenance(connection, members: list[str], extra_runs: list[str]) -> dict:
    designs, runs, configs = {}, {}, {}
    pending = list(members)
    while pending:
        identity = pending.pop()
        if identity in designs:
            continue
        matches = rows(connection, "SELECT * FROM designs WHERE design_id = ?", [identity])
        if not matches:
            raise SubmissionError(f"missing lineage design {identity}")
        design = designs[identity] = matches[0]
        extra_runs.append(design["run_id"])
        if design["parent_design_id"]:
            pending.append(design["parent_design_id"])
    pending = list(extra_runs)
    while pending:
        identity = pending.pop()
        if identity in runs:
            continue
        matches = rows(connection, "SELECT * FROM runs WHERE run_id = ?", [identity])
        if not matches:
            raise SubmissionError(f"missing lineage run {identity}")
        run = runs[identity] = matches[0]
        config_id = run["model_config_id"]
        matches = rows(connection, "SELECT * FROM configs WHERE model_config_id = ?", [config_id])
        if not matches:
            raise SubmissionError(f"missing source config {config_id}")
        configs[config_id] = matches[0]
        model = configs[config_id]["model_config_json"]
        referenced = [*model.get("evaluator_runs", ()), *model.get("filter_runs", ())]
        if model.get("structures_from"):
            referenced.append(model["structures_from"])
        referenced.extend(item["run_id"] for item in model.get("metric_inputs", {}).values())
        if referenced:
            pending.extend(resolve_run_ids(connection, tuple(referenced)))
        if run["parent_run_id"]:
            pending.append(run["parent_run_id"])
    return {"designs": designs, "runs": runs, "configs": configs}


def _resolve_source(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _kit_provenance(stage: Path, provenance: dict, database: Path) -> list[dict]:
    from bindocracy.kits.resolve import load_lock

    kits = []
    seen = set()
    for config in provenance["configs"].values():
        base = database.parent
        if config.get("model_source_uri"):
            base = _resolve_source(config["model_source_uri"], base).parent
        model = config["model_config_json"]
        custom = model.get("custom") or {}
        lock_value = model.get("kit_lock") or custom.get("kit_lock")
        if not lock_value:
            # Legacy custom functions still need their executable source provenance.
            script = custom.get("script") or model.get("script")
            if script:
                source = _resolve_source(script, base)
                relative = f"provenance/scripts/{_digest(str(source))}/{source.name}"
                if not (stage / relative).exists():
                    _copy_file(source, stage / relative)
                kits.append({"kind": "standalone_script", "config_id": config["model_config_id"], "path": relative})
            continue
        lock_path = _resolve_source(lock_value, base)
        lock = load_lock(lock_path, require_approval=False)
        if lock.digest in seen:
            next(kit for kit in kits if kit.get("digest") == lock.digest)["config_ids"].append(config["model_config_id"])
            continue
        seen.add(lock.digest)
        prefix = f"provenance/kits/{lock.digest.removeprefix('sha256:')}"
        _copy_file(lock_path, stage / prefix / "lock.json")
        copied = {}
        for relative, expected in lock.files.items():
            destination = f"{prefix}/kit/{_relative(relative)}"
            _copy_file(lock.kit_root / relative, stage / destination, expected)
            copied[destination] = expected
        for name, dependency in lock.dependencies.items():
            identity = dependency["identity"]
            inventory = identity.get("files", {})
            if identity.get("method") == "sha256-tree":
                inventory = {key: value["digest"] for key, value in identity["entries"].items() if value.get("kind") == "file"}
            for relative, expected in inventory.items():
                destination = f"{prefix}/dependencies/{_relative(name)}/{_relative(relative)}"
                _copy_file(Path(dependency["path"]) / relative, stage / destination, expected)
                copied[destination] = expected
        kits.append({"kind": "kit", "digest": lock.digest, "bundle_digest": lock.bundle_digest,
                     "lock": f"{prefix}/lock.json", "code_files": copied, "config_ids": [config["model_config_id"]],
                     "external_dependencies": {name: item for name, item in lock.dependencies.items() if item["identity"]["method"] == "filesystem-stat"}})
    return kits


def build_submission(database: str | Path, output_dir: str | Path, *,
                     design_set: DesignSet | str | Path | None = None,
                     workspace: str | Path | None = None,
                     structures_from: str | None = None,
                     policy: SubmissionPolicy | None = None) -> dict:
    """Freeze exact design-set order (or the ordinary default DB selection).

    Structures, when requested, are the unique complex replicate-zero artifact
    from the explicitly selected evaluator run; scores never enter this bundle.
    """
    from bindocracy.config.preflight import read_single_fasta

    database, output_dir = Path(database).absolute(), Path(output_dir).absolute()
    if output_dir.exists():
        raise SubmissionError(f"output already exists: {output_dir}")
    ws = None
    if workspace is not None:
        from bindocracy.benchmark.workspace import load_workspace
        ws = load_workspace(workspace)
        if database.resolve() != ws.database.resolve():
            raise SubmissionError("database differs from workspace database")
        policy = ws.task.submission
    policy = policy or SubmissionPolicy()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".submission-", dir=output_dir.parent))
    try:
        with read_only(database) as connection:
            connection.execute("BEGIN TRANSACTION")
            if design_set is None:
                frozen = build_design_set(select_designs(connection, DesignQuery()), database=database, query=DesignQuery())
            else:
                frozen = design_set if isinstance(design_set, DesignSet) else DesignSet.read(design_set)
            entries = frozen.entries
            digest = sha256_text(canonical_json([[entry.design_id, entry.sequence] for entry in entries]))
            if (not entries or frozen.digest != digest or frozen.scope_id != stable_id("design-set", digest)
                    or frozen.n_designs != len(entries) or [e.index for e in entries] != list(range(len(entries)))):
                raise SubmissionError("inconsistent frozen design-set membership or order")
            run_id = resolve_run_ids(connection, (structures_from,), kind="evaluate")[0] if structures_from else None
            selection_runs = [*frozen.query.get("filter_runs", ()), *frozen.query.get("exclude_scored_by_run", ())]
            extra_runs = list(resolve_run_ids(connection, tuple(selection_runs))) if selection_runs else []
            if run_id:
                extra_runs.append(run_id)
            provenance = _collect_provenance(connection, [e.design_id for e in entries], extra_runs)
            first_config = next(iter(provenance["configs"].values()))
            target = ws.task.target.model_dump(mode="json") if ws else first_config["general_config_json"]["target"]
            base = database.parent
            if not ws and first_config.get("general_source_uri"):
                base = _resolve_source(first_config["general_source_uri"], base).parent
            target_file = _resolve_source(target["sequence_fasta"], base)
            sequence = read_single_fasta(target_file)
            target_identity = TargetDigest.of(target["name"], sequence).model_dump(mode="json")
            if target_digest(connection) != target_identity["sequence_sha256"]:
                raise SubmissionError("database target differs from submitted target")
            target_files = {}
            for field in ("sequence_fasta", "structure_cif", "structure_pdb"):
                if target.get(field):
                    source = _resolve_source(target[field], base)
                    relative = f"target/{field}{source.suffix}"
                    _copy_file(source, stage / relative)
                    target_files[field] = relative
            candidates = []
            for entry in entries:
                design = provenance["designs"][entry.design_id]
                run = provenance["runs"][design["run_id"]]
                if (design["sequence"] != entry.sequence or design["length"] != entry.length
                        or design["native_id"] != entry.native_id or run["tool"] != entry.tool or run["name"] != entry.run_name):
                    raise SubmissionError(f"frozen identity differs from database: {entry.design_id}")
                candidate = entry.model_dump(mode="json")
                candidate["run_id"] = design["run_id"]
                candidate["structures"] = []
                if run_id:
                    artifacts = rows(connection, "SELECT * FROM artifacts WHERE run_id = ? AND design_id = ? AND kind = 'predicted_structure' AND json_extract_string(metadata, '$.condition') = 'complex' AND json_extract_string(metadata, '$.replicate') = '0'", [run_id, entry.design_id])
                    if len(artifacts) != 1:
                        raise SubmissionError(f"expected exactly one complex replicate-zero structure: {entry.design_id}")
                    artifact = artifacts[0]
                    uri = Path(artifact["uri"])
                    output = provenance["runs"][run_id]["output_uri"]
                    if not uri.is_absolute() and not output:
                        raise SubmissionError("relative structure has no run output directory")
                    source = uri if uri.is_absolute() else _resolve_source(output, database.parent) / uri
                    relative = f"structures/{entry.index}{source.suffix}"
                    _copy_file(source, stage / relative, artifact["sha256"])
                    if artifact["size_bytes"] is not None and (stage / relative).stat().st_size != artifact["size_bytes"]:
                        raise SubmissionError("structure size differs from database")
                    candidate["structures"].append({"path": relative, "artifact": artifact})
                candidates.append(candidate)
        # Embedded JSON is the authoritative source configuration; original URIs
        # are provenance, never dependencies required by a portable check.
        _write_json(stage / "provenance/lineage.json", provenance)
        _write_json(stage / "provenance/design_set.json", frozen.model_dump(mode="json"))
        kits = _kit_provenance(stage, provenance, database)
        (stage / "candidates.fasta").write_text("".join(f">{c['index']} {c['design_id']}\n{c['sequence']}\n" for c in candidates))
        if ws:
            _write_json(stage / "provenance/workspace.json", ws.model_dump(mode="json"))
        try:
            version = importlib.metadata.version("bindocracy")
        except importlib.metadata.PackageNotFoundError:
            version = "uninstalled-source"
        manifest = {"schema_version": 1, "kind": "candidate_submission", "software_version": version,
                    "created_at": datetime.now(UTC).isoformat(),
                    "attempt_id": ws.attempt_id if ws else None, "task_digest": ws.task_digest if ws else None,
                    "task_id": ws.task.task_id if ws else None, "task_version": ws.task.version if ws else None,
                    "environment_digest": ws.environment_digest if ws else None,
                    "policy": policy.model_dump(mode="json"), "target": {**target_identity, "sequence": sequence,
                        "chain_id": target["chain_id"], "hotspots": target.get("hotspots", []), "files": target_files},
                    "candidate_count": len(candidates), "candidates": candidates, "kits": kits,
                    "files": {relative: hashlib.sha256(_read(stage / relative)).hexdigest() for relative in sorted(_inventory(stage))}}
        manifest["digest"] = _digest(manifest)
        _write_json(stage / "manifest.json", manifest)
        check_submission(stage)
        publish_directory(stage, output_dir)
        return {"path": str(output_dir), **manifest}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _check_submission(path: str | Path, *, policy: SubmissionPolicy | None = None) -> dict:
    """Check portable bytes and membership only; never run a scorer or kit."""
    manifest_path = _manifest_path(path)
    root = manifest_path.parent
    manifest = _json(_read(manifest_path))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") != "candidate_submission":
        raise SubmissionError("unsupported submission manifest")
    expected = manifest.get("digest")
    if not isinstance(expected, str) or not _HASH.fullmatch(expected) or _digest({key: value for key, value in manifest.items() if key != "digest"}) != expected:
        raise SubmissionError("submission manifest digest mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or "manifest.json" in files:
        raise SubmissionError("invalid file inventory")
    for relative, digest in files.items():
        _relative(relative)
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise SubmissionError(f"invalid file hash: {relative}")
    if _inventory(root) != {*files, "manifest.json"}:
        raise SubmissionError("undeclared or missing files in bundle")
    for relative, digest in files.items():
        if hashlib.sha256(_read(root / relative)).hexdigest() != digest:
            raise SubmissionError(f"content hash mismatch: {relative}")
    policy = policy or SubmissionPolicy.model_validate(manifest.get("policy"))
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or type(manifest.get("candidate_count")) is not int or len(candidates) != manifest["candidate_count"]:
        raise SubmissionError("candidate count mismatch")
    if not policy.min_candidates <= len(candidates) <= policy.max_candidates:
        raise SubmissionError("candidate count outside submission policy")
    identities, sequences = set(), set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or type(candidate.get("index")) is not int or candidate["index"] != index:
            raise SubmissionError("candidate order is not contiguous")
        identity, sequence = candidate.get("design_id"), candidate.get("sequence")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise SubmissionError("missing or duplicate candidate identity")
        if not isinstance(sequence, str) or not _AMINO.fullmatch(sequence) or candidate.get("length") != len(sequence):
            raise SubmissionError("invalid candidate protein sequence or length")
        if policy.distinct_sequences and sequence in sequences:
            raise SubmissionError("duplicate candidate sequence")
        identities.add(identity)
        sequences.add(sequence)
        structures = candidate.get("structures")
        if not isinstance(structures, list) or (policy.require_structures and not structures):
            raise SubmissionError("required candidate structure missing")
        for structure in structures:
            if not isinstance(structure, dict) or structure.get("path") not in files:
                raise SubmissionError("structure not included in file inventory")
            from bindocracy.store.records import ArtifactRecord
            artifact = ArtifactRecord.model_validate(structure.get("artifact"))
            if (artifact.design_id != candidate["design_id"] or artifact.kind != "predicted_structure"
                    or (artifact.sha256 is not None and artifact.sha256.removeprefix("sha256:") != files[structure["path"]])
                    or (artifact.size_bytes is not None and artifact.size_bytes != (root / structure["path"]).stat().st_size)):
                raise SubmissionError("candidate structure provenance mismatch")
    required = {"candidates.fasta", "provenance/design_set.json", "provenance/lineage.json"}
    if not required <= files.keys():
        raise SubmissionError("missing required provenance")
    fasta = "".join(f">{c['index']} {c['design_id']}\n{c['sequence']}\n" for c in candidates)
    if _read(root / "candidates.fasta").decode() != fasta:
        raise SubmissionError("FASTA differs from ordered candidate membership")
    frozen = DesignSet.model_validate(_json(_read(root / "provenance/design_set.json")))
    members = [[c["design_id"], c["sequence"]] for c in candidates]
    if ([[e.design_id, e.sequence] for e in frozen.entries] != members or frozen.n_designs != len(members)
            or frozen.digest != sha256_text(canonical_json(members)) or frozen.scope_id != stable_id("design-set", frozen.digest)):
        raise SubmissionError("design-set provenance differs from candidates")
    lengths = [candidate["length"] for candidate in candidates]
    if (frozen.by_tool != dict(Counter(candidate["tool"] for candidate in candidates))
            or frozen.length_range != (min(lengths), max(lengths)) or frozen.distinct_lengths != len(set(lengths))
            or any(any(candidate.get(key) != value for key, value in entry.model_dump(mode="json").items())
                   for entry, candidate in zip(frozen.entries, candidates, strict=True))):
        raise SubmissionError("design-set metadata/order/count summary mismatch")
    lineage = _json(_read(root / "provenance/lineage.json"))
    for candidate in candidates:
        design = lineage.get("designs", {}).get(candidate["design_id"], {})
        if design.get("sequence") != candidate["sequence"] or design.get("run_id") != candidate.get("run_id"):
            raise SubmissionError("candidate lineage mismatch")
        run = lineage.get("runs", {}).get(candidate["run_id"], {})
        if (design.get("native_id") != candidate["native_id"] or run.get("name") != candidate["run_name"]
                or run.get("tool") != candidate["tool"]):
            raise SubmissionError("candidate source identity mismatch")
    for kind, parent in (("designs", "parent_design_id"), ("runs", "parent_run_id")):
        records = lineage.get(kind, {})
        for key, record in records.items():
            visited = {key}
            ancestor = record.get(parent)
            while ancestor:
                if ancestor in visited or ancestor not in records:
                    raise SubmissionError("cyclic or incomplete lineage")
                visited.add(ancestor)
                ancestor = records[ancestor].get(parent)
            if kind == "designs" and record.get("run_id") not in lineage.get("runs", {}):
                raise SubmissionError("missing source run")
            if kind == "runs" and record.get("model_config_id") not in lineage.get("configs", {}):
                raise SubmissionError("missing source configuration")
    from bindocracy.store.records import ConfigRecord
    for key, config in lineage.get("configs", {}).items():
        parsed = ConfigRecord.model_validate(config)
        if parsed.model_config_id != key:
            raise SubmissionError("source configuration identity mismatch")
    target = manifest.get("target", {})
    sequence = target.get("sequence")
    if not isinstance(sequence, str) or not _AMINO.fullmatch(sequence):
        raise SubmissionError("invalid target sequence")
    identity = TargetDigest.of(target.get("name", ""), sequence).model_dump(mode="json")
    if any(target.get(key) != value for key, value in identity.items()):
        raise SubmissionError("target content identity mismatch")
    if "sequence_fasta" not in target.get("files", {}) or any(value not in files for value in target["files"].values()):
        raise SubmissionError("missing target files")
    from bindocracy.config.preflight import read_single_fasta
    if read_single_fasta(root / target["files"]["sequence_fasta"]) != sequence:
        raise SubmissionError("target FASTA differs from manifest")
    code_configs = set()
    for kit in manifest.get("kits", []):
        if kit.get("kind") == "kit":
            code_configs.update(kit.get("config_ids", []))
            if kit.get("lock") not in files:
                raise SubmissionError("missing kit lock")
            lock = _json(_read(root / kit["lock"]))
            lock_digest = "sha256:" + _digest({key: value for key, value in lock.items() if key != "digest"})
            if lock.get("digest") != kit.get("digest") or lock_digest != kit["digest"]:
                raise SubmissionError("kit lock digest mismatch")
            prefix = str(PurePosixPath(kit["lock"]).parent)
            expected_code = {f"{prefix}/kit/{_relative(name)}": digest for name, digest in lock["files"].items()}
            for name, dependency in lock["dependencies"].items():
                identity = dependency["identity"]
                inventory = identity.get("files", {})
                if identity.get("method") == "sha256-tree":
                    inventory = {name: entry["digest"] for name, entry in identity["entries"].items() if entry.get("kind") == "file"}
                expected_code.update({f"{prefix}/dependencies/{_relative(name)}/{_relative(relative)}": digest for relative, digest in inventory.items()})
            if kit.get("code_files") != expected_code or kit.get("bundle_digest") != lock["bundle_digest"]:
                raise SubmissionError("incomplete kit source/dependency closure")
            for relative, digest in kit.get("code_files", {}).items():
                if files.get(relative) != digest.removeprefix("sha256:"):
                    raise SubmissionError("kit code inventory mismatch")
        elif kit.get("kind") == "standalone_script":
            code_configs.add(kit.get("config_id"))
            if kit.get("path") not in files:
                raise SubmissionError("missing custom script")
        else:
            raise SubmissionError("unknown code provenance kind")
    for config_id, config in lineage["configs"].items():
        model = config["model_config_json"]
        custom = model.get("custom") or {}
        if any((model.get("kit_lock"), custom.get("kit_lock"), model.get("script"), custom.get("script"))) and config_id not in code_configs:
            raise SubmissionError("source configuration is missing executable provenance")
    return manifest


def check_submission(path: str | Path, *, policy: SubmissionPolicy | None = None) -> dict:
    """Validate only self-contained bytes, with malformed data reported as a refusal."""
    try:
        return _check_submission(path, policy=policy)
    except (KeyError, TypeError, AttributeError) as error:
        raise SubmissionError(f"malformed submission: {error}") from error


def accept_submission(source: str | Path, attempt_dir: str | Path, *,
                      now: Callable[[], datetime] | datetime | None = None) -> dict:
    """Organizer-only acceptance. The deadline is checked AFTER copied bytes verify.

    ``now`` is a trusted clock seam for tests, never a CLI/agent parameter.
    A crash can leave an unreferenced digest directory, never a partial acceptance.
    """
    directory = Path(attempt_dir).absolute()
    clock = now if callable(now) else (lambda: now if now is not None else datetime.now(UTC))
    source = _manifest_path(source).parent
    with (directory / "attempt.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = AttemptRecord.model_validate_json(_read(directory / "attempt.json"))
        task = TaskSpec.model_validate_json(_read(directory / "task.json"))
        def authorize(moment):
            if moment.tzinfo is None or record.deadline is None or record.deadline.tzinfo is None:
                raise SubmissionError("acceptance requires timezone-aware clock and deadline")
            if record.state != "running" or moment >= record.deadline:
                raise SubmissionError("attempt is not running before its deadline")
            if (directory / "budget.json").exists():
                from bindocracy.benchmark.budget import BudgetError, BudgetLedger
                try:
                    BudgetLedger(directory).check()
                except BudgetError as error:
                    raise SubmissionError(f"attempt budget admission is closed: {error}") from error
        authorize(clock())
        manifest = check_submission(source, policy=task.submission)
        if (manifest.get("attempt_id") != record.attempt_id or manifest.get("task_digest") != record.task_digest
                or manifest.get("environment_digest") != record.environment_digest or manifest.get("task_id") != record.task_id
                or manifest.get("task_version") != task.version or task.task_id != record.task_id):
            raise SubmissionError("submission attempt/task/environment identity is unauthorized")
        from bindocracy.benchmark.models import WorkspaceManifest
        from bindocracy.benchmark.workspace import _task_digest
        from bindocracy.config.preflight import read_single_fasta
        trusted_workspace = WorkspaceManifest.model_validate_json(_read(directory / "workspace.json"))
        if _task_digest(task, trusted_workspace.input_assets) != record.task_digest:
            raise SubmissionError("organizer task snapshot identity differs from attempt")
        trusted_target = TargetDigest.of(task.target.name, read_single_fasta(directory / "target.fasta"))
        if any(manifest["target"].get(key) != value for key, value in trusted_target.model_dump(mode="json").items()):
            raise SubmissionError("submitted target differs from organizer target")
        if manifest["target"].get("chain_id") != task.target.chain_id or manifest["target"].get("hotspots") != list(task.target.hotspots):
            raise SubmissionError("submitted target chain/hotspots differ from organizer task")
        for field in ("sequence_fasta", "structure_cif", "structure_pdb"):
            target_path = getattr(task.target, field)
            if target_path is not None:
                identity = next((asset for asset in trusted_workspace.input_assets.values() if asset.path == target_path), None)
                relative = manifest["target"]["files"].get(field)
                if identity is None or manifest["files"].get(relative) != identity.sha256:
                    raise SubmissionError(f"submitted target {field} differs from captured organizer input")
        accepted_root = directory / "accepted"
        accepted_root.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".accept-", dir=accepted_root))
        try:
            for relative, digest in manifest["files"].items():
                _copy_file(source / relative, stage / relative, digest)
            _copy_file(source / "manifest.json", stage / "manifest.json")
            copied = check_submission(stage, policy=task.submission)
            if copied["digest"] != manifest["digest"]:
                raise SubmissionError("submission changed during acceptance copy")
            # Detect additions/replacements to the source as well as copy tampering.
            if check_submission(source, policy=task.submission)["digest"] != copied["digest"]:
                raise SubmissionError("submission changed during acceptance")
            accepted_at = clock()
            authorize(accepted_at)
            destination = accepted_root / copied["digest"]
            if destination.exists():
                if check_submission(destination)["digest"] != copied["digest"]:
                    raise SubmissionError("existing accepted archive is corrupt")
            else:
                for relative in _inventory(stage):
                    (stage / relative).chmod(0o444)
                for root, _, _ in os.walk(stage, topdown=False):
                    Path(root).chmod(0o555)
                    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                os.rename(stage, destination)
                descriptor = os.open(accepted_root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            accepted_at = clock()
            authorize(accepted_at)
            acceptance = {"digest": copied["digest"], "path": str(destination),
                          "accepted_at": accepted_at.isoformat(), "attempt_id": record.attempt_id,
                          "task_digest": record.task_digest}
            history = directory / "acceptance_history"
            history.mkdir(exist_ok=True)
            _atomic_json(history / f"{accepted_at.isoformat()}-{copied['digest']}.json", acceptance)
            # The record is the durable journal/commit point, serialized with runner transitions.
            updated = record.model_copy(update={"latest_submission": acceptance})
            _atomic_json(directory / "attempt.json", updated.model_dump(mode="json"))
            return acceptance
        finally:
            if stage.exists():
                for root, _, _ in os.walk(stage):
                    Path(root).chmod(0o700)
                shutil.rmtree(stage)


def archive_evidence(attempt_dir: str | Path, output_dir: str | Path, *,
                     workspace: str | Path | None = None,
                     evaluations: tuple[str | Path, ...] = ()) -> dict:
    """Archive complete experiment trees separately from fixed candidate bundles.

    Call after the attempt has stopped. Any concurrently modified source file or
    symlink is refused, rather than silently omitted from experimental evidence.
    """
    attempt, output = Path(attempt_dir).absolute(), Path(output_dir).absolute()
    record = AttemptRecord.model_validate_json(_read(attempt / "attempt.json"))
    if record.state in {"prepared", "running"}:
        raise SubmissionError("stop the attempt before archiving its evidence")
    roots = {"control": attempt, "workspace": Path(workspace or record.workspace).absolute()}
    roots.update({f"evaluations/{index}": Path(path).absolute() for index, path in enumerate(evaluations)})
    if output.exists() or any(output.is_relative_to(root) for root in roots.values()):
        raise SubmissionError("evidence destination must be new and outside source trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".evidence-", dir=output.parent))
    try:
        files = {}
        for label, root in roots.items():
            before = _inventory(root)
            for relative in sorted(before):
                name = f"{label}/{relative}"
                files[name] = _copy_file(root / relative, stage / name)
            if _inventory(root) != before:
                raise SubmissionError("evidence tree changed during archival")
        manifest = {"schema_version": 1, "kind": "experiment_evidence", "attempt_id": record.attempt_id,
                    "files": files, "created_at": datetime.now(UTC).isoformat()}
        manifest["digest"] = _digest(manifest)
        _write_json(stage / "manifest.json", manifest)
        publish_directory(stage, output)
        return {"path": str(output), **manifest}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
