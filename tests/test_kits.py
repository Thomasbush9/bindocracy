"""Distribution is inert and content-addressed, even for hostile input bytes."""

import json
import os
import shutil
import stat
import struct
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from bindocracy.kits.bundle import cache_kit, inspect_kit
from bindocracy.kits.models import KitError, KitManifest
from bindocracy.kits.store import _HTTPSRedirects, fetch_kit, list_releases, publish_kit


@pytest.fixture
def kit(tmp_path):
    root = tmp_path / "author"
    root.mkdir()
    (root / "kit.yaml").write_text(yaml.safe_dump({
        "kit_schema": 1,
        "name": "fixture",
        "version": "1.0.0",
        "kind": "optimizer",
        "summary": "Inert package transport fixture",
        "entrypoint": "entry.py",
        "python": ">=3.10",
        "requires": [],
        "inputs": ["sequence"],
        "loss_models": [],
        "capabilities": {"changes_length": False, "max_children": 2},
        "metrics": {"fixture": {"direction": "none"}},
        "cost": {"gpu": "none", "compile_seconds": 0, "per_parent_seconds": 0},
        "args": {"count": 2, "label": "default", "enabled": False},
        "dependency_args": {},
    }))
    (root / "entry.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('EXECUTED').write_text('unexpected')\n"
        "raise RuntimeError('kit code must not run during transport')\n"
    )
    (root / "helpers").mkdir()
    (root / "helpers" / "constants.py").write_text("VALUE = 42\n")
    (root / "data.bin").write_bytes(b"\x00\x01\x02")
    (root / ".support").write_text("hidden files are part of the bundle\n")
    return root


def test_offline_pinned_roundtrip_preserves_all_bytes_without_execution(kit, tmp_path):
    before = {path.relative_to(kit).as_posix(): path.read_bytes() for path in kit.rglob("*") if path.is_file()}
    inspected = inspect_kit(kit)
    store = tmp_path / "store"
    published = publish_kit(kit, store, "claimed-lab")
    assert list_releases(str(store)) == [published]
    fetched = fetch_kit(str(store), published["reference"], tmp_path / "cache", inspected.digest)
    assert fetched.files == inspected.files
    assert fetched.manifest == inspected.manifest
    assert {path: (fetched.root / path).read_bytes() for path in fetched.files} == before
    assert {path.relative_to(kit).as_posix(): path.read_bytes() for path in kit.rglob("*") if path.is_file()} == before
    shutil.rmtree(store)
    shutil.rmtree(kit)
    offline = inspect_kit(fetched.root)
    assert offline.digest == inspected.digest
    assert offline.files == inspected.files
    assert not (fetched.root / "EXECUTED").exists()


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_entire_inventory_changes_identity(kit, change):
    original = inspect_kit(kit).digest
    if change == "modify":
        (kit / "data.bin").write_bytes(b"changed")
    elif change == "add":
        (kit / ".new-support").write_text("new")
    else:
        (kit / ".support").unlink()
    assert inspect_kit(kit).digest != original


def test_wrong_pin_never_populates_cache(kit, tmp_path):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    cache = tmp_path / "cache"
    with pytest.raises(KitError, match="digest"):
        fetch_kit(str(store), record["reference"], cache, "sha256:" + "0" * 64)
    with pytest.raises(KitError, match="digest"):
        cache_kit(kit, cache, expected_digest="sha256:" + "0" * 64)
    assert not cache.exists()


def test_cache_poison_is_refused_not_repaired(kit, tmp_path):
    cached = cache_kit(kit, tmp_path / "cache")
    (cached.root / "data.bin").write_bytes(b"poison")
    with pytest.raises(KitError, match="tampered"):
        cache_kit(kit, tmp_path / "cache")
    assert (cached.root / "data.bin").read_bytes() == b"poison"


def test_symlink_cache_entry_is_refused(kit, tmp_path):
    release = inspect_kit(kit)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / release.digest[7:]).symlink_to(kit, target_is_directory=True)
    with pytest.raises(KitError, match="directory"):
        cache_kit(kit, cache)


def test_release_republication_is_idempotent_but_replacement_is_refused(kit, tmp_path):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    catalog = (store / "catalog.json").read_bytes()
    archive = (store / record["archive"]).read_bytes()
    assert publish_kit(kit, store, "lab") == record
    (kit / "data.bin").write_bytes(b"replacement")
    with pytest.raises(KitError, match="already exists"):
        publish_kit(kit, store, "lab")
    assert (store / "catalog.json").read_bytes() == catalog
    assert (store / record["archive"]).read_bytes() == archive


def test_concurrent_publishers_preserve_catalog_entries(kit, tmp_path):
    store = tmp_path / "store"
    publishers = ["lab-a", "lab-b", "lab-c", "lab-d"]
    with ThreadPoolExecutor(max_workers=4) as workers:
        published = list(workers.map(lambda publisher: publish_kit(kit, store, publisher), publishers))
    assert {row["reference"] for row in list_releases(str(store))} == {row["reference"] for row in published}
    assert len({row["digest"] for row in published}) == 1


def test_concurrent_conflicting_versions_have_exactly_one_winner(kit, tmp_path):
    alternate = tmp_path / "alternate"
    shutil.copytree(kit, alternate)
    (alternate / "data.bin").write_bytes(b"other content")
    store = tmp_path / "store"

    def attempt(source):
        try:
            return publish_kit(source, store, "lab")
        except KitError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, [kit, alternate]))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert list_releases(str(store)) == winners


def test_tampered_archive_fails_before_cache_publication(kit, tmp_path):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    with zipfile.ZipFile(store / record["archive"], "w") as archive:
        for relative in inspect_kit(kit).files:
            archive.writestr(relative, b"poison" if relative == "data.bin" else (kit / relative).read_bytes())
    with pytest.raises(KitError, match="digest"):
        fetch_kit(str(store), record["reference"], tmp_path / "cache", record["digest"])
    with pytest.raises(KitError, match="tampered"):
        publish_kit(kit, store, "lab")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("malice", ["traversal", "absolute", "backslash", "duplicate", "symlink", "fifo", "encrypted", "oversize"])
def test_hostile_archive_is_refused_without_publication(kit, tmp_path, monkeypatch, malice):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    blob = store / record["archive"]
    name = {"traversal": "../escaped", "absolute": "/escaped", "backslash": "..\\escaped"}.get(malice, "payload")
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    if malice in {"symlink", "fifo"}:
        kind = stat.S_IFLNK if malice == "symlink" else stat.S_IFIFO
        info.external_attr = (kind | 0o644) << 16
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr(info, b"unexpected")
        if malice == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(info, b"again")
    if malice == "encrypted":
        payload = bytearray(blob.read_bytes())
        struct.pack_into("<H", payload, 6, 1)
        central = payload.index(b"PK\x01\x02")
        struct.pack_into("<H", payload, central + 8, 1)
        blob.write_bytes(payload)
    if malice == "oversize":
        monkeypatch.setattr("bindocracy.kits.store.MAX_BYTES", 4)
    with pytest.raises(KitError):
        fetch_kit(str(store), record["reference"], tmp_path / "cache", record["digest"])
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("kind", ["file-symlink", "directory-symlink", "fifo"])
def test_nonregular_source_refused_before_store_creation(kit, tmp_path, kind):
    if kind == "file-symlink":
        (kit / "bad").symlink_to(kit / "data.bin")
    elif kind == "directory-symlink":
        (kit / "bad").symlink_to(kit / "helpers", target_is_directory=True)
    else:
        os.mkfifo(kit / "bad")
    with pytest.raises(KitError):
        publish_kit(kit, tmp_path / "store", "lab")
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("field,value", [
    ("entrypoint", "../entry.py"), ("unexpected", True), ("kit_schema", True),
    ("capabilities", {"changes_length": "false", "max_children": 2}),
    ("args", {"count": float("nan")}),
    ("cost", {"gpu": "none", "compile_seconds": float("inf"), "per_parent_seconds": 0}),
    ("dependency_args", {"undeclared": "--flag; command"}),
])
def test_manifest_rejects_ambiguous_or_unsafe_metadata(kit, field, value):
    path = kit / "kit.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest[field] = value
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(KitError):
        inspect_kit(kit)


def test_manifest_duplicate_keys_are_not_silently_overwritten(kit):
    path = kit / "kit.yaml"
    path.write_text(path.read_text() + "name: replacement\n")
    with pytest.raises(KitError, match="unique"):
        inspect_kit(kit)


def test_legacy_manifests_are_validated_without_importing_scientific_code():
    examples = Path(__file__).resolve().parents[1] / "examples" / "kits"
    for name in ("surface-tuner", "rfd-refine", "mosaic-af2-refine", "tomasz-optimization"):
        manifest = KitManifest.model_validate(yaml.safe_load((examples / name / "kit.yaml").read_text()))
        assert KitManifest.model_validate_json(manifest.model_dump_json()) == manifest
        assert KitManifest.model_validate(manifest.model_dump(mode="json", by_alias=True)) == manifest


def test_catalog_duplicate_release_reference_is_refused(kit, tmp_path):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    (store / "catalog.json").write_text(json.dumps({"schema_version": 1, "releases": [record, record]}))
    with pytest.raises(KitError, match="duplicate"):
        list_releases(str(store))


@pytest.mark.parametrize("url", ["http://example.invalid/catalog.json", "https://user:password@example.invalid/catalog.json"])
def test_https_redirects_cannot_downgrade_or_introduce_credentials(url):
    request = urllib.request.Request("https://example.invalid/catalog.json")
    with pytest.raises(KitError, match="HTTPS"):
        _HTTPSRedirects().redirect_request(request, None, 302, "Found", {}, url)


def test_implicit_archive_directories_count_toward_limit(kit, tmp_path, monkeypatch):
    store = tmp_path / "store"
    record = publish_kit(kit, store, "lab")
    with zipfile.ZipFile(store / record["archive"], "w") as archive:
        archive.writestr("a/b/c/payload", b"content")
    monkeypatch.setattr("bindocracy.kits.store.MAX_DIRECTORIES", 2)
    with pytest.raises(KitError, match="directory limit"):
        fetch_kit(str(store), record["reference"], tmp_path / "cache", record["digest"])
    assert not (tmp_path / "cache").exists()


def test_source_mutation_during_copy_never_publishes_cache(kit, tmp_path, monkeypatch):
    from bindocracy.kits import bundle

    original = bundle._snapshot
    digest = inspect_kit(kit).digest

    def mutate_after_copy(source, destination=None):
        result = original(source, destination)
        if destination is not None:
            (source / "data.bin").write_bytes(b"changed during copy")
        return result

    monkeypatch.setattr(bundle, "_snapshot", mutate_after_copy)
    with pytest.raises(KitError, match="changed"):
        cache_kit(kit, tmp_path / "cache", expected_digest=digest)
    assert not (tmp_path / "cache" / digest[7:]).exists()
