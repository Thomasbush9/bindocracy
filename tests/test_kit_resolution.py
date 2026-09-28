"""Synthetic distribution/provenance checks; no external programs or model data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from bindocracy.kits.bundle import inspect_kit
from bindocracy.kits.models import KitError
from bindocracy.kits.resolve import load_lock, resolve_kit, verify_lock, write_lock
from bindocracy.store.records import canonical_json


@pytest.fixture
def kit(tmp_path: Path) -> Path:
    root = tmp_path / "kit"
    root.mkdir()
    (root / "main.py").write_text("raise RuntimeError('checking must never execute me')\n")
    (root / "helper.py").write_text("MESSAGE = 'hello'\n")
    (root / "data.txt").write_text("local payload\n")
    manifest = {
        "kit_schema": 1,
        "name": "synthetic",
        "version": "1.0.0",
        "kind": "optimizer",
        "summary": "Synthetic transport test",
        "entrypoint": "main.py",
        "python": ">=3.10",
        "requires": [],
        "inputs": ["sequence"],
        "loss_models": [],
        "capabilities": {"changes_length": False, "max_children": 2},
        "metrics": {"score": {"direction": "min", "description": "Synthetic score"}},
        "cost": {"gpu": "none", "compile_seconds": 0, "per_parent_seconds": 0, "measured_on": {}},
        "args": {"count": 2, "enabled": False, "label": "hello world"},
    }
    (root / "kit.yaml").write_text(yaml.safe_dump(manifest))
    return root


def _manifest(kit: Path, **changes) -> None:
    path = kit / "kit.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))


def _bindings(tmp_path: Path, **entries) -> Path:
    path = tmp_path / "bindings.yaml"
    path.write_text(yaml.safe_dump({"bindings_schema": 1, **entries}))
    return path


def _approved(kit: Path, **kwargs):
    return resolve_kit(kit, approved_digest=inspect_kit(kit).digest, **kwargs)


def _snapshot(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in root.rglob("*")
    }


def test_resolution_is_deterministic_read_only_and_never_executes(kit, tmp_path):
    before = _snapshot(tmp_path)
    first = resolve_kit(kit, parameters={"enabled": True})
    second = resolve_kit(kit, parameters={"enabled": True})
    assert first.digest == second.digest
    assert first.args == ("--count", "2", "--enabled", "--label", "hello world")
    assert _snapshot(tmp_path) == before
    assert any(check["code"] == "kit.approval" and check["status"] == "unknown" for check in first.checks)
    verify_lock(first, require_approval=False)
    with pytest.raises(KitError, match="approval"):
        verify_lock(first)


def test_explicit_approval_is_required_and_must_match_bundle(kit, tmp_path):
    with pytest.raises(KitError, match="approval"):
        resolve_kit(kit, approved_digest="sha256:" + "0" * 64)
    unapproved = resolve_kit(kit)
    path = write_lock(unapproved, tmp_path / "unapproved.json")
    with pytest.raises(KitError, match="approval"):
        load_lock(path)
    assert load_lock(path, require_approval=False).digest == unapproved.digest
    approved = _approved(kit)
    assert load_lock(write_lock(approved, tmp_path / "approved.json")).digest == approved.digest


@pytest.mark.parametrize("parameters", [{"typo": 1}, {"count": True}, {"enabled": "false"}])
def test_unknown_parameters_and_scalar_type_confusion_are_refused(kit, parameters):
    with pytest.raises(KitError, match="parameter|Parameter"):
        resolve_kit(kit, parameters=parameters)


def test_false_boolean_omits_flag_and_explicit_parameter_constraints_apply(kit):
    _manifest(kit, parameters={"ratio": {"type": "number", "required": True, "minimum": 0, "maximum": 1}})
    with pytest.raises(KitError, match="Required kit parameter"):
        resolve_kit(kit)
    with pytest.raises(KitError, match="parameter"):
        resolve_kit(kit, parameters={"ratio": 2})
    lock = resolve_kit(kit, parameters={"ratio": 0.5})
    assert "--enabled" not in lock.args
    assert lock.args[-2:] == ("--ratio", "0.5")


def test_inferred_types_are_exact_and_declared_enum_is_enforced(kit):
    _manifest(kit, args={"ratio": 0.5}, parameters={"choice": {"type": "string", "required": True, "enum": ["a", "b"]}})
    with pytest.raises(KitError, match="Parameter 'ratio'"):
        resolve_kit(kit, parameters={"ratio": 1, "choice": "a"})
    with pytest.raises(KitError, match="parameter 'choice'"):
        resolve_kit(kit, parameters={"choice": "c"})
    assert resolve_kit(kit, parameters={"choice": "b"}).parameters == {"choice": "b", "ratio": 0.5}


def test_missing_required_and_explicitly_omitted_optional_dependencies(kit):
    _manifest(kit, requires=[{"name": "assets", "kind": "weights"}])
    with pytest.raises(KitError, match="Missing required dependency 'assets'.*bindings_schema"):
        resolve_kit(kit)
    _manifest(kit, requires=[{"name": "assets", "kind": "weights", "optional": True}], dependency_args={"assets": "--assets"})
    lock = _approved(kit)
    assert "assets" not in lock.dependencies
    assert "--assets" not in lock.args
    assert any("unbound and omitted" in check["message"] for check in lock.checks)
    verify_lock(lock)


def test_relative_bindings_and_source_overlay_closure(kit, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "helper.py").write_text("ANSWER = 1\n")
    _manifest(kit, requires=[{"name": "library", "kind": "source_overlay", "verify": {"files": ["helper.py"]}}], dependency_args={"library": "--library"})
    bindings = _bindings(tmp_path, library={"kind": "source_overlay", "path": "source"})
    lock = _approved(kit, bindings=bindings)
    assert lock.args[-2:] == ("--library", str(source))
    assert lock.dependencies["library"]["identity"]["method"] == "sha256-tree"
    verify_lock(lock)
    (source / "additional.txt").write_text("new unlisted dependency\n")
    with pytest.raises(KitError, match="dependency identity"):
        verify_lock(lock)


@pytest.mark.parametrize("mutation", ["change", "remove", "add"])
def test_entire_bundle_inventory_is_verified(kit, mutation):
    lock = _approved(kit)
    if mutation == "change":
        (kit / "data.txt").write_text("changed\n")
    elif mutation == "remove":
        (kit / "helper.py").unlink()
    else:
        (kit / "new.txt").write_text("not in the lock\n")
    with pytest.raises(KitError, match="inventory|digest"):
        verify_lock(lock)


def test_overlay_content_drift_is_refused_even_with_unchanged_size(kit, tmp_path):
    source = tmp_path / "overlay"
    source.mkdir()
    helper = source / "helper.py"
    helper.write_text("ANSWER = 1\n")
    _manifest(kit, requires=[{"name": "library", "kind": "source_overlay"}])
    lock = _approved(kit, bindings=_bindings(tmp_path, library={"kind": "source_overlay", "path": str(source)}))
    helper.write_text("ANSWER = 2\n")
    with pytest.raises(KitError, match="dependency identity"):
        verify_lock(lock)


def test_large_asset_identity_is_honestly_stat_based_and_detects_drift(kit, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "asset.bin").write_bytes(b"old")
    _manifest(kit, requires=[{"name": "assets", "kind": "weights", "version": "author hint", "verify": {"files": ["asset.bin"]}}])
    lock = _approved(kit, bindings=_bindings(tmp_path, assets={"kind": "weights", "path": str(weights)}))
    assert lock.dependencies["assets"]["identity"]["method"] == "filesystem-stat"
    assert {check["code"] for check in lock.checks if check["status"] == "unknown"} >= {"kit.content.assets", "kit.version.assets"}
    (weights / "asset.bin").write_bytes(b"changed")
    with pytest.raises(KitError, match="dependency identity"):
        verify_lock(lock)


def test_multiple_nonprimary_containers_are_translated_without_runtime_guessing(kit, tmp_path):
    for name in ("first", "second"):
        (tmp_path / f"{name}.sif").write_bytes(b"synthetic image")
    _manifest(kit, requires=[{"name": name, "kind": "container", "verify": {"import": "untrusted.module"}} for name in ("first", "second")], dependency_args={"first": "--first-image", "second": "--second-image"})
    lock = _approved(kit, bindings=_bindings(tmp_path, **{name: {"kind": "container", "path": f"{name}.sif"} for name in ("first", "second")}))
    assert lock.runtime == {}
    assert lock.args[-4:] == ("--first-image", str(tmp_path / "first.sif"), "--second-image", str(tmp_path / "second.sif"))
    assert all(check["status"] == "unknown" for check in lock.checks if check["code"].startswith("kit.import."))
    verify_lock(lock)


def test_primary_runtime_requires_explicit_interpreter_and_is_unique(kit, tmp_path):
    image = tmp_path / "image.sif"
    image.write_bytes(b"synthetic image")
    _manifest(kit, requires=[{"name": "runtime", "kind": "container", "provides": "interpreter"}])
    bindings = _bindings(tmp_path, runtime={"kind": "container", "path": str(image)})
    with pytest.raises(KitError, match="explicit interpreter"):
        resolve_kit(kit, bindings=bindings)
    bindings = _bindings(tmp_path, runtime={"kind": "container", "path": str(image), "interpreter": "/opt/python"})
    lock = _approved(kit, bindings=bindings)
    assert lock.runtime == {"container": str(image), "container_python": "/opt/python"}
    assert next(check for check in lock.checks if check["code"] == "kit.python")["status"] == "unknown"
    _manifest(kit, requires=[{"name": name, "kind": "container", "provides": "interpreter"} for name in ("first", "second")])
    bindings = _bindings(tmp_path, **{name: {"kind": "container", "path": str(image), "interpreter": "python"} for name in ("first", "second")})
    with pytest.raises(KitError, match="At most one container"):
        resolve_kit(kit, bindings=bindings)


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "folder/../../outside", "folder\\outside"])
def test_unsafe_declared_verification_paths_are_refused(kit, relative):
    _manifest(kit, requires=[{"name": "assets", "kind": "weights", "verify": {"files": [relative]}}])
    with pytest.raises(KitError):
        resolve_kit(kit)


def test_missing_declared_dependency_files_and_unsupported_overlay_targets(kit, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bindings = _bindings(tmp_path, library={"kind": "source_overlay", "path": str(source)})
    _manifest(kit, requires=[{"name": "library", "kind": "source_overlay", "verify": {"files": ["missing.py"]}}])
    with pytest.raises(KitError, match="missing.py"):
        resolve_kit(kit, bindings=bindings)
    _manifest(kit, requires=[{"name": "library", "kind": "source_overlay", "target": "/unsupported"}])
    with pytest.raises(KitError, match="Unsupported runtime"):
        resolve_kit(kit, bindings=bindings)


def test_dependency_symlinks_and_remote_bindings_are_explicitly_refused(kit, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(kit / "data.txt")
    _manifest(kit, requires=[{"name": "library", "kind": "source_overlay"}])
    with pytest.raises(KitError, match="Symlink"):
        resolve_kit(kit, bindings=_bindings(tmp_path, library={"kind": "source_overlay", "path": str(source)}))
    with pytest.raises(KitError, match="local filesystem"):
        resolve_kit(kit, bindings=_bindings(tmp_path, library={"kind": "source_overlay", "path": "https://example.invalid/archive"}))


def test_harness_helpers_cannot_be_replaced_by_site_bindings(kit, tmp_path):
    _manifest(kit, requires=[{"name": "bindocracy-io", "kind": "harness_module", "provided_by": "harness", "verify": {"files": ["drivers/bindocracy_io.py"]}}])
    fake = tmp_path / "untrusted-harness"
    lock = _approved(kit, bindings=_bindings(tmp_path, **{"bindocracy-io": {"kind": "harness_module", "path": str(fake)}}))
    dependency = lock.dependencies["bindocracy-io"]
    assert Path(dependency["path"]) == Path(__file__).resolve().parents[1]
    assert set(dependency["identity"]["files"]) == {"drivers/bindocracy_io.py", "drivers/optimize/run_optimizer.py"}
    verify_lock(lock)


def test_lock_mutation_and_rehashed_argument_forgery_are_refused(kit, tmp_path):
    lock = _approved(kit)
    path = write_lock(lock, tmp_path / "kit-lock.json")
    raw = json.loads(path.read_text())
    raw["args"] = ["--unapproved-flag"]
    path.write_text(json.dumps(raw))
    with pytest.raises(KitError, match="canonical contents"):
        load_lock(path)
    raw["digest"] = "sha256:" + hashlib.sha256(canonical_json({key: value for key, value in raw.items() if key != "digest"}).encode()).hexdigest()
    path.write_text(json.dumps(raw))
    with pytest.raises(KitError, match="execution contract"):
        load_lock(path)


def test_lock_writes_are_idempotent_and_never_replace_different_locks(kit, tmp_path):
    path = tmp_path / "kit-lock.json"
    first = _approved(kit)
    write_lock(first, path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    write_lock(first, path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    second = _approved(kit, parameters={"count": 3})
    with pytest.raises(KitError, match="different existing kit lock"):
        write_lock(second, path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with pytest.raises(KitError, match="outside the immutable"):
        write_lock(first, kit / "lock.json")
    assert not (kit / "lock.json").exists()


def test_lock_publication_rechecks_source_mutation(kit, tmp_path):
    lock = _approved(kit)
    (kit / "data.txt").write_text("changed after resolving")
    destination = tmp_path / "lock.json"
    with pytest.raises(KitError, match="inventory|digest"):
        write_lock(lock, destination)
    assert not destination.exists()


def test_host_python_constraint_is_checked_without_an_import_probe(kit):
    _manifest(kit, python="<2")
    with pytest.raises(KitError, match="Host Python"):
        resolve_kit(kit)


@pytest.mark.parametrize("declaration", ["default", "parameter", "dependency"])
def test_harness_owned_argument_flags_cannot_be_overridden(kit, declaration):
    if declaration == "default":
        _manifest(kit, args={"inputs": "replacement.jsonl"})
    elif declaration == "parameter":
        _manifest(kit, parameters={"outputs": {"type": "string"}})
    else:
        _manifest(
            kit,
            requires=[{"name": "assets", "kind": "weights", "optional": True}],
            dependency_args={"assets": "--context"},
        )
    with pytest.raises(KitError, match="harness-owned"):
        resolve_kit(kit)
