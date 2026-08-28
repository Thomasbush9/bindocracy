"""Filesystem and content checks performed after structural validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig, MosaicConfig


class ConfigPreflightError(ValueError):
    """A structurally valid configuration cannot run in this environment."""


@dataclass(frozen=True)
class MosaicPreflight:
    target_sequence: str

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


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


def preflight_mosaic(general: GeneralConfig, mosaic: MosaicConfig) -> MosaicPreflight:
    """Check paths needed to run one Mosaic configuration."""
    required_files = {
        "target FASTA": general.target.sequence_fasta,
        "target MSA": general.target.msa,
        "Mosaic driver": mosaic.driver.script,
        "Mosaic container": mosaic.runtime.container,
        "Mosaic exec wrapper": mosaic.runtime.exec_wrapper,
    }
    errors = [
        f"{description} does not exist: {path}"
        for description, path in required_files.items()
        if not path.is_file()
    ]

    # The wrapper creates the scratch tree itself, but only one level down, so
    # its parent has to exist already.
    if not mosaic.runtime.scratch.parent.is_dir():
        errors.append(f"Mosaic scratch parent does not exist: {mosaic.runtime.scratch.parent}")

    if not mosaic.runtime.weights.is_dir():
        errors.append(f"Mosaic weights directory does not exist: {mosaic.runtime.weights}")
    elif not (mosaic.runtime.weights / "boltz").is_dir():
        errors.append(f"Boltz weights are missing under: {mosaic.runtime.weights}")

    if errors:
        raise ConfigPreflightError("\n".join(errors))

    return MosaicPreflight(target_sequence=read_single_fasta(general.target.sequence_fasta))
