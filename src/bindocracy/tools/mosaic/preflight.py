"""Filesystem checks Mosaic needs before any GPU work starts.

Plus the one content check: Mosaic folds the target from its sequence and
conditions on the epitope by *index into that sequence*, so the mapping from a
campaign residue number to a loss argument happens here, where it can be
refused, rather than inside a GPU job where it would be an array slice.
"""

from __future__ import annotations

from dataclasses import dataclass

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
)
from bindocracy.tools.mosaic.config import MosaicConfig


@dataclass(frozen=True)
class MosaicPreflight:
    target_sequence: str
    # The campaign epitope as Mosaic expresses it: 0-based indices into the
    # target sequence. Empty when the campaign names none, which is a
    # different loss rather than a missing argument.
    epitope_idx: tuple[int, ...] = ()

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def _epitope_indices(general: GeneralConfig, target_sequence: str) -> tuple[int, ...]:
    """Map the campaign epitope onto `BinderTargetContact(epitope_idx=...)`.

    The term slices its binder-by-target contact matrix down to these columns,
    so they are 0-based positions in the target sequence and the mapping is
    `seqid - 1`. Indexing by enumeration position instead is the bug in
    docs/known-issues.md section 1.4, which silently conditioned 23 of 26
    epitope entries on the wrong residues; the campaign FASTA is the whole
    chain, numbered from 1, so `seqid - 1` is the mapping and the bounds check
    below is what makes that assumption fail loudly if it ever stops holding.

    Everything that cannot be mapped is refused. Designing unconstrained while
    another tool in the same campaign uses the epitope is the comparison
    quietly becoming meaningless.
    """
    if not general.target.hotspots:
        return ()

    hotspots = parse_hotspots(general.target.hotspots)
    chain = general.target.chain_id.upper()
    elsewhere = sorted(
        {spot.chain for spot in hotspots if spot.chain and spot.chain != chain}
    )
    if elsewhere:
        raise ConfigPreflightError(
            f"campaign hotspots name chain(s) {', '.join(elsewhere)}, but the "
            f"campaign target is chain {chain}. Mosaic folds one target chain "
            "from its sequence and cannot condition on another."
        )

    length = len(target_sequence)
    outside = sorted(spot.number for spot in hotspots if not 1 <= spot.number <= length)
    if outside:
        raise ConfigPreflightError(
            f"campaign hotspots {outside} fall outside the target sequence, "
            f"which is {length} residues. Mosaic conditions by position in that "
            "sequence, so a residue it does not contain cannot be expressed."
        )
    return tuple(sorted({spot.number - 1 for spot in hotspots}))


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

    target_sequence = read_single_fasta(general.target.sequence_fasta)
    return MosaicPreflight(
        target_sequence=target_sequence,
        epitope_idx=_epitope_indices(general, target_sequence),
    )


