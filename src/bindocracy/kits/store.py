"""Static filesystem/HTTPS transport for digest-pinned kit releases.

Publisher namespaces are unauthenticated claims, not signatures or identities.
Publication supports local filesystems only; any HTTPS hosting is external.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from bindocracy.config.models import ConfigModel
from bindocracy.kits.bundle import (
    MAX_BYTES,
    MAX_DIRECTORIES,
    MAX_FILES,
    KitRelease,
    _filesystem_lock,
    _snapshot,
    cache_kit,
    inspect_kit,
    validate_digest,
)
from bindocracy.kits.models import KitError, safe_relative, safe_segment
from bindocracy.store.records import canonical_json

MAX_ARCHIVE_BYTES = MAX_BYTES + 10 * 1024 * 1024
MAX_CATALOG_BYTES = 4 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30


def _reference(value: str) -> tuple[str, str, str]:
    if not isinstance(value, str) or value.count("/") != 1 or value.count("@") != 1:
        raise KitError("Kit reference must be publisher/name@version")
    publisher, rest = value.split("/")
    name, version = rest.split("@")
    for part in (publisher, name, version):
        safe_segment(part)
    if version.lower() == "latest":
        raise KitError("Kit reference must pin an exact version, not latest")
    return publisher, name, version


class _ReleaseRecord(ConfigModel):
    reference: StrictStr
    publisher: StrictStr
    name: StrictStr
    version: StrictStr
    digest: StrictStr
    archive: StrictStr

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if _reference(self.reference) != (self.publisher, self.name, self.version):
            raise ValueError("Release reference disagrees with its metadata")
        validate_digest(self.digest)
        if self.archive != f"blobs/{self.digest[7:]}.zip":
            raise ValueError("Release archive must be its content-addressed blob path")
        return self


class _Catalog(ConfigModel):
    schema_version: Literal[1]
    releases: list[_ReleaseRecord] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema(cls, value: StrictInt) -> int:
        if type(value) is not int:
            raise ValueError("Catalog schema_version must be integer 1")
        return value

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        names = [release.reference for release in self.releases]
        if len(names) != len(set(names)):
            raise ValueError("Catalog contains duplicate release references")
        return self


def _json_pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise KitError(f"Duplicate catalog JSON key: {key}")
        result[key] = value
    return result


def _parse_catalog(payload: bytes) -> _Catalog:
    if len(payload) > MAX_CATALOG_BYTES:
        raise KitError("Kit catalog exceeds the 4 MiB limit")
    try:
        return _Catalog.model_validate(json.loads(payload, object_pairs_hook=_json_pairs))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise KitError(f"Invalid kit catalog: {exc}") from exc


def _https_url(value: str) -> str:
    try:
        if any(ord(char) <= 32 or ord(char) == 127 for char in value):
            raise ValueError("URL whitespace and control characters are not supported")
        parts = urllib.parse.urlsplit(value)
        if parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError("HTTPS without URL credentials is required")
        if parts.fragment:
            raise ValueError("URL fragments are not supported")
        _ = parts.port  # Validate the port before attempting network access.
    except ValueError as exc:
        raise KitError(f"Unsafe kit store URL: {exc}") from exc
    return value


class _HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _bounded_read(handle, limit: int) -> bytes:
    payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise KitError(f"Kit store resource exceeds the {limit}-byte download limit")
    return payload


def _local_read(root: Path, relative: str, limit: int) -> bytes:
    safe_relative(relative)
    parent = root
    for part in relative.split("/")[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise KitError("Kit store resource directories cannot be symlinks")
    fd = os.open(root / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise KitError("Kit store resource must be a regular file")
        if info.st_size > limit:
            raise KitError("Kit store resource exceeds its download limit")
        return _bounded_read(handle, limit)


def _resource(store: str, relative: str, limit: int) -> bytes:
    safe_relative(relative)
    try:
        parts = urllib.parse.urlsplit(store)
    except ValueError as exc:
        raise KitError(f"Unsafe kit store URL: {exc}") from exc
    try:
        if parts.scheme or parts.netloc:
            _https_url(store)
            if parts.query:
                raise KitError("Kit store base URLs cannot contain a query")
            url = store.rstrip("/") + "/" + relative
            opener = urllib.request.build_opener(_HTTPSRedirects())
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            with opener.open(request, timeout=DOWNLOAD_TIMEOUT) as response:
                _https_url(response.geturl())
                return _bounded_read(response, limit)
        return _local_read(Path(store), relative, limit)
    except (OSError, urllib.error.URLError) as exc:
        raise KitError(f"Cannot read kit store resource {relative}: {exc}") from exc


def list_releases(store: str) -> list[dict]:
    """List catalog claims; listing does not authenticate publishers or run code."""
    catalog = _parse_catalog(_resource(str(store), "catalog.json", MAX_CATALOG_BYTES))
    return [item.model_dump() for item in sorted(catalog.releases, key=lambda item: item.reference)]


def _extract_archive(payload: bytes, destination: Path) -> None:
    """Manually extract bounded regular files; never delegate to extractall."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise KitError("Kit archive exceeds its download limit")
    seen: set[str] = set()
    total = 0
    file_count = 0
    directories: set[str] = set()

    def make_directories(parts: list[str]) -> None:
        for index in range(1, len(parts) + 1):
            relative = "/".join(parts[:index])
            if relative not in directories:
                if len(directories) >= MAX_DIRECTORIES:
                    raise KitError("Kit archive exceeds the 4096-directory limit")
                (destination / relative).mkdir(exist_ok=True)
                directories.add(relative)

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_FILES + MAX_DIRECTORIES:
                raise KitError("Kit archive contains too many members")
            for member in members:
                if member.orig_filename != member.filename or "\x00" in member.orig_filename:
                    raise KitError("Kit archive contains a NUL-truncated filename")
                is_directory = member.is_dir()
                relative = safe_relative(member.filename[:-1] if is_directory else member.filename)
                if relative in seen:
                    raise KitError(f"Duplicate kit archive member: {relative}")
                seen.add(relative)
                mode = member.external_attr >> 16
                kind = stat.S_IFMT(mode)
                allowed_kind = stat.S_IFDIR if is_directory else stat.S_IFREG
                if kind not in (0, allowed_kind) or member.flag_bits & 1:
                    raise KitError(f"Encrypted, symlink or nonregular kit archive member: {relative}")
                if member.file_size < 0 or member.file_size > MAX_BYTES - total:
                    raise KitError("Kit archive exceeds the 100 MiB payload limit")
                path = destination / relative
                if is_directory:
                    if member.file_size:
                        raise KitError("Kit archive directory contains a payload")
                    make_directories(relative.split("/"))
                    continue
                file_count += 1
                if file_count > MAX_FILES:
                    raise KitError("Kit archive exceeds the 4096-file limit")
                make_directories(relative.split("/")[:-1])
                written = 0
                with archive.open(member) as source, path.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        total += len(chunk)
                        if total > MAX_BYTES or written > member.file_size:
                            raise KitError("Kit archive exceeds its declared payload limits")
                        target.write(chunk)
                if written != member.file_size:
                    raise KitError("Kit archive member has incorrect size")
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
        raise KitError(f"Invalid kit ZIP archive: {exc}") from exc


def fetch_kit(store: str, reference: str, cache_root: Path, expected_digest: str) -> KitRelease:
    """Fetch an exact reference, requiring an independently supplied digest pin."""
    _reference(reference)
    validate_digest(expected_digest)
    releases = list_releases(store)
    item = next((release for release in releases if release["reference"] == reference), None)
    if item is None:
        raise KitError(f"Kit release is not in the catalog: {reference}")
    if item["digest"] != expected_digest:
        raise KitError("Catalog release does not match the required digest")
    payload = _resource(str(store), item["archive"], MAX_ARCHIVE_BYTES)
    with tempfile.TemporaryDirectory(prefix="bindocracy-kit-fetch-") as temporary:
        root = Path(temporary)
        _extract_archive(payload, root)
        release = inspect_kit(root)
        if release.digest != expected_digest:
            raise KitError("Downloaded kit contents do not match the required digest")
        if (release.manifest.name, release.manifest.version) != (item["name"], item["version"]):
            raise KitError("Downloaded kit manifest disagrees with the catalog")
        if release.manifest.publisher is not None and release.manifest.publisher != item["publisher"]:
            raise KitError("Downloaded kit publisher claim disagrees with the catalog")
        return cache_kit(root, cache_root, expected_digest=expected_digest)


def _atomic_catalog(root: Path, catalog: _Catalog) -> None:
    payload = canonical_json(catalog.model_dump()).encode("utf-8") + b"\n"
    if len(payload) > MAX_CATALOG_BYTES:
        raise KitError("Kit catalog exceeds the 4 MiB limit")
    fd, temporary = tempfile.mkstemp(prefix=".catalog-", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / "catalog.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_kit(source: Path, store: Path, publisher: str) -> dict:
    """Publish an immutable reference under an unauthenticated namespace claim."""
    safe_segment(publisher)
    release = inspect_kit(source)
    if release.manifest.publisher is not None and release.manifest.publisher != publisher:
        raise KitError("Publisher namespace disagrees with the manifest's publisher claim")
    root = Path(store).absolute()
    if root.is_symlink() or root.resolve() == release.root or release.root in root.resolve().parents:
        raise KitError("Store must be a real directory outside the source kit")
    record = _ReleaseRecord(
        reference=f"{publisher}/{release.manifest.name}@{release.manifest.version}",
        publisher=publisher, name=release.manifest.name, version=release.manifest.version,
        digest=release.digest, archive=f"blobs/{release.digest[7:]}.zip",
    )
    root.mkdir(parents=True, exist_ok=True)
    with _filesystem_lock(root):
        catalog_path = root / "catalog.json"
        if catalog_path.exists() or catalog_path.is_symlink():
            try:
                catalog = _parse_catalog(_local_read(root, "catalog.json", MAX_CATALOG_BYTES))
            except OSError as exc:
                raise KitError(f"Cannot read kit catalog: {exc}") from exc
        else:
            catalog = _Catalog(schema_version=1)
        existing = next((item for item in catalog.releases if item.reference == record.reference), None)
        if existing is not None and existing.digest != record.digest:
            raise KitError("Release version already exists with a different digest; publish a new version")
        blobs = root / "blobs"
        if blobs.is_symlink():
            raise KitError("Store blobs directory cannot be a symlink")
        blobs.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=root))
        try:
            kit = staging / "kit"
            kit.mkdir()
            copied = _snapshot(release.root, kit)
            verified = inspect_kit(kit)
            if copied.digest != release.digest or verified.digest != release.digest or inspect_kit(release.root).digest != release.digest:
                raise KitError("Kit changed while it was being published")
            blob = root / record.archive
            if blob.exists() or blob.is_symlink():
                check = staging / "existing"
                check.mkdir()
                try:
                    _extract_archive(_local_read(root, record.archive, MAX_ARCHIVE_BYTES), check)
                except OSError as exc:
                    raise KitError(f"Cannot verify existing kit blob: {exc}") from exc
                if inspect_kit(check).digest != record.digest:
                    raise KitError("Existing store blob has been tampered with")
            else:
                archive_path = staging / "release.zip"
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for relative in verified.files:
                        archive.write(kit / relative, arcname=relative)
                if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
                    raise KitError("Kit archive exceeds its download limit")
                # link is exclusive even if an uncooperative writer races us.
                os.link(archive_path, blob)
            if existing is None:
                updated = sorted([*catalog.releases, record], key=lambda item: item.reference)
                _atomic_catalog(root, _Catalog(schema_version=1, releases=updated))
            return record.model_dump()
        finally:
            shutil.rmtree(staging)
