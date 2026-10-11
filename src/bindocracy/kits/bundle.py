"""Complete-content kit inspection and a verified, atomic local cache.

Limits are 100 MiB of payload, 4096 files and 4096 directories per bundle.
Nothing is excluded from the identity. No kit code or install hooks are run.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from bindocracy.kits.models import KIT_MANIFEST_ADAPTER, KitError, KitManifest, safe_relative
from bindocracy.store.records import canonical_json

MAX_BYTES = 100 * 1024 * 1024
MAX_FILES = 4096
MAX_DIRECTORIES = 4096
MAX_MANIFEST_BYTES = 1024 * 1024
_CHUNK = 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class KitRelease:
    root: Path
    manifest: KitManifest
    digest: str
    files: dict[str, str]


class _ManifestLoader(yaml.SafeLoader):
    """Refuse duplicate keys rather than silently replacing declarations."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise KitError("Manifest YAML aliases are not supported; write values explicitly")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise KitError("Manifest mapping keys must be unique strings")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def validate_digest(value: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise KitError("Expected a lowercase sha256:<64 hexadecimal characters> digest")
    return value


def _parse_manifest(payload: bytes) -> KitManifest:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise KitError("kit.yaml exceeds the 1 MiB manifest limit")
    try:
        value = yaml.load(payload, Loader=_ManifestLoader)
        return KIT_MANIFEST_ADAPTER.validate_python(value)
    except (yaml.YAMLError, ValidationError, UnicodeError, RecursionError) as exc:
        raise KitError(f"Invalid kit.yaml: {exc}") from exc


def _signature(value: os.stat_result) -> tuple:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _snapshot(source: Path, destination: Path | None = None) -> KitRelease:
    """Read via directory descriptors, never following a member symlink."""
    source = Path(source).absolute()
    files: dict[str, str] = {}
    manifest_bytes: bytes | None = None
    total = 0
    directories = 0

    def visit(directory_fd: int, prefix: str = "") -> None:
        nonlocal total, directories, manifest_bytes
        before = os.fstat(directory_fd)
        with os.scandir(directory_fd) as entries:
            names = []
            for entry in entries:
                if len(names) >= MAX_FILES + MAX_DIRECTORIES:
                    raise KitError("Kit directory exceeds the bounded member count")
                names.append(entry.name)
            names.sort()
        for name in names:
            relative = safe_relative(prefix + name)
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                directories += 1
                if directories > MAX_DIRECTORIES:
                    raise KitError("Kit exceeds the 4096-directory limit")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    if destination is not None:
                        (destination / relative).mkdir()
                    visit(child, relative + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if len(files) >= MAX_FILES or total + info.st_size > MAX_BYTES:
                    raise KitError("Kit exceeds the 4096-file or 100 MiB payload limit")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                with os.fdopen(fd, "rb") as handle:
                    opened = os.fstat(handle.fileno())
                    if not stat.S_ISREG(opened.st_mode) or _signature(info) != _signature(opened):
                        raise KitError(f"Kit file changed during inspection: {relative}")
                    digest = hashlib.sha256()
                    size = 0
                    payload = bytearray() if relative == "kit.yaml" else None
                    output = (destination / relative).open("xb") if destination is not None else None
                    try:
                        while chunk := handle.read(_CHUNK):
                            size += len(chunk)
                            total += len(chunk)
                            if total > MAX_BYTES:
                                raise KitError("Kit exceeds the 100 MiB payload limit")
                            digest.update(chunk)
                            if payload is not None:
                                if size > MAX_MANIFEST_BYTES:
                                    raise KitError("kit.yaml exceeds the 1 MiB manifest limit")
                                payload.extend(chunk)
                            if output is not None:
                                output.write(chunk)
                        if size != opened.st_size or _signature(opened) != _signature(os.fstat(handle.fileno())):
                            raise KitError(f"Kit file changed during inspection: {relative}")
                    finally:
                        if output is not None:
                            output.close()
                    files[relative] = "sha256:" + digest.hexdigest()
                    if payload is not None:
                        manifest_bytes = bytes(payload)
            else:
                raise KitError(f"Kit member must be a regular file or directory, not a symlink/device: {relative}")
        if _signature(before) != _signature(os.fstat(directory_fd)):
            raise KitError("Kit directory changed during inspection")

    try:
        fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            visit(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise KitError(f"Cannot inspect kit {source}: {exc}") from exc
    if manifest_bytes is None:
        raise KitError("Kit directory must contain kit.yaml at its root")
    manifest = _parse_manifest(manifest_bytes)
    if manifest.entrypoint not in files:
        raise KitError(f"Kit entrypoint is not a bundled regular file: {manifest.entrypoint}")
    ordered = dict(sorted(files.items()))
    digest = "sha256:" + hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()
    return KitRelease(source.resolve(), manifest, digest, ordered)


def inspect_kit(source: Path) -> KitRelease:
    """Inspect and hash a local directory, without executing code or writing files."""
    return _snapshot(Path(source))


def read_manifest(root: Path) -> KitManifest:
    """Read a manifest only after validating its complete bundle."""
    return inspect_kit(root).manifest


@contextmanager
def _filesystem_lock(root: Path) -> Iterator[None]:
    """Serialize cooperating writers; the lock file is never replaced."""
    try:
        fd = os.open(root / ".kit.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a+b") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise KitError("Kit store lock is not a regular file")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise KitError(f"Cannot lock kit store/cache: {exc}") from exc


def cache_kit(source: Path, cache_root: Path, expected_digest: str | None = None) -> KitRelease:
    """Publish a full verified copy at ``cache_root/<digesthex>/kit``.

    Existing entries are always rehashed and poison is refused, never repaired
    silently. The cache is immutable by verification, not an execution sandbox.
    """
    if expected_digest is not None:
        validate_digest(expected_digest)
    release = inspect_kit(source)
    if expected_digest is not None and release.digest != expected_digest:
        raise KitError("Kit digest does not match the required digest")
    cache_root = Path(cache_root).absolute()
    if cache_root == release.root or release.root in cache_root.resolve().parents:
        raise KitError("Cache must not be inside the source kit")
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / release.digest.removeprefix("sha256:")
    with _filesystem_lock(cache_root):
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise KitError("Cache entry is not a real directory")
            cached = inspect_kit(target / "kit")
            if cached.digest != release.digest:
                raise KitError("Cached kit has been tampered with")
            return cached
        staging = Path(tempfile.mkdtemp(prefix=".kit-stage-", dir=cache_root))
        try:
            copied_root = staging / "kit"
            copied_root.mkdir()
            copied = _snapshot(release.root, copied_root)
            verified = inspect_kit(copied_root)
            if copied.digest != release.digest or verified.digest != release.digest or inspect_kit(release.root).digest != release.digest:
                raise KitError("Kit changed while it was being cached")
            os.rename(staging, target)
            return KitRelease((target / "kit").resolve(), verified.manifest, verified.digest, verified.files)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
