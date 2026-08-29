"""Filesystem checks Mosaic needs before any GPU work starts."""

from __future__ import annotations

from dataclasses import dataclass

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.tools.mosaic.config import MosaicConfig


@dataclass(frozen=True)
class MosaicPreflight:
    target_sequence: str

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_mosaic(general: GeneralConfig, mosaic: MosaicConfig) -> MosaicPreflight:
    """Check paths needed to run one Mosaic configuration."""
    if general.target.msa is None:
        raise ConfigPreflightError(
            "Mosaic folds the target from sequence and needs target.msa, which is not set"
        )
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


