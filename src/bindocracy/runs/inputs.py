"""Identify the files a run actually consumed, not just where they were.

Config IDs hash the configuration document, which contains *paths*. Replacing
a target FASTA or a design spec at the same path therefore leaves every ID
unchanged while the science changes underneath. Recording a digest per
referenced input closes that at the run level: the manifest says which bytes
this run consumed.

Large files get a fingerprint rather than a digest. Hashing the 17 GB BoltzGen
image costs about half a minute on a login node, every time a run is planned,
and a size-and-mtime pair identifies a rebuilt image in practice. The record
says which kind it holds, so nobody mistakes one for the other.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

# Above this, fingerprint instead of hashing. Containers are gigabytes; every
# other input here — a FASTA, an MSA, a driver, a spec — is far below it.
MAX_HASH_BYTES = 256 * 1024 * 1024


class InputDigest(BaseModel):
    """What a referenced input was, when the run was planned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str
    kind: Literal["sha256", "fingerprint"]
    size_bytes: int
    sha256: str | None = None
    mtime_ns: int | None = None

    @model_validator(mode="after")
    def kind_matches_content(self) -> Self:
        if self.kind == "sha256" and self.sha256 is None:
            raise ValueError("a sha256 digest needs a hash")
        if self.kind == "fingerprint" and self.mtime_ns is None:
            raise ValueError("a fingerprint needs an mtime")
        return self

    def matches(self, path: str | Path) -> bool:
        """Whether the file at `path` is still what this digest recorded."""
        try:
            return self == digest_of(path, uri=self.uri)
        except OSError:
            return False


def digest_of(path: str | Path, *, uri: str | None = None) -> InputDigest:
    """Hash a file, or fingerprint it if hashing would be disproportionate."""
    file_path = Path(path)
    stat = file_path.stat()
    if stat.st_size > MAX_HASH_BYTES:
        return InputDigest(
            uri=uri if uri is not None else str(file_path.resolve()),
            kind="fingerprint",
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return InputDigest(
        uri=uri if uri is not None else str(file_path.resolve()),
        kind="sha256",
        size_bytes=stat.st_size,
        sha256=f"sha256:{digest.hexdigest()}",
    )


class TargetDigest(BaseModel):
    """The biological target a run designed against, by content.

    One database follows one target. This is what makes that checkable: the
    resolved sequence, not the path it was read from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    length: int
    sequence_sha256: str

    @classmethod
    def of(cls, name: str, sequence: str) -> Self:
        return cls(
            name=name,
            length=len(sequence),
            sequence_sha256=f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}",
        )
