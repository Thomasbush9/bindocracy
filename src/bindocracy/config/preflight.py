"""Filesystem and content checks performed after structural validation."""

from __future__ import annotations

import re
from pathlib import Path


class ConfigPreflightError(ValueError):
    """A structurally valid configuration cannot run in this environment."""


def read_single_fasta(path: Path) -> str:
    """Read one FASTA sequence and return an uppercase, whitespace-free string."""
    headers = 0
    sequence_lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            headers += 1
            continue
        sequence_lines.append(line)

    if headers > 1:
        raise ConfigPreflightError(f"target FASTA must contain one sequence: {path}")

    sequence = re.sub(r"\s+", "", "".join(sequence_lines)).upper()
    if not sequence or re.fullmatch(r"[A-Z]+", sequence) is None:
        raise ConfigPreflightError(f"target FASTA has no valid amino-acid sequence: {path}")
    return sequence
