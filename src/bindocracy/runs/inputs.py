"""Identify the small files a run actually consumed, not just where they were.

Config IDs hash the configuration document, which contains *paths*. Replacing a
target FASTA or a design spec at the same path therefore leaves every ID
unchanged while the science changes underneath. A digest per referenced input
closes that: the manifest says which bytes this run consumed, and the launch
refuses to start if they have changed since.

Containers are deliberately not digested. They are gigabytes, the image path
and its checksum belong to the image catalogue rather than to every run, and
hashing one on every plan would buy a number nobody reads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class InputDigest(BaseModel):
    """What a referenced input was, when the run was planned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str
    sha256: str
    size_bytes: int

    def matches(self, path: str | Path) -> bool:
        """Whether the file at `path` is still what this digest recorded."""
        try:
            return self == digest_of(path, uri=self.uri)
        except OSError:
            return False


def digest_of(path: str | Path, *, uri: str | None = None) -> InputDigest:
    """Hash one referenced input."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return InputDigest(
        uri=uri if uri is not None else str(file_path.resolve()),
        sha256=f"sha256:{digest.hexdigest()}",
        size_bytes=file_path.stat().st_size,
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
    def of(cls, name: str, sequence: str) -> TargetDigest:
        return cls(
            name=name,
            length=len(sequence),
            sequence_sha256=f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}",
        )
