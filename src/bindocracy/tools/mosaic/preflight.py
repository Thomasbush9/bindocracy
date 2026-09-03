"""Filesystem checks Mosaic needs before any GPU work starts.

Plus the content checks: Mosaic folds the target from its sequence and
conditions on the epitope by *index into that sequence*, so the mapping from a
campaign structure residue to a loss argument happens here, where it can be
refused, rather than inside a GPU job where it would be an array slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
)
from bindocracy.tools.mosaic.config import MosaicConfig

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common PDB residue names that have an unambiguous FASTA representation.
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "UNK": "X",
}


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
    so they are 0-based positions in the target sequence. Campaign hotspots,
    however, use the author residue numbers of the target structure. Map them
    through the ordered PDB chain rather than assuming that author residue N
    is always FASTA position N: chains may start above 1 or contain numbering
    gaps, and both cases otherwise condition on the wrong residues silently.

    Everything that cannot be mapped is refused. Designing unconstrained while
    another tool in the same campaign uses the epitope is the comparison
    quietly becoming meaningless.
    """
    if not general.target.hotspots:
        return ()

    hotspots = parse_hotspots(general.target.hotspots)
    chain = general.target.chain_id.upper()
    elsewhere = sorted({spot.chain for spot in hotspots if spot.chain and spot.chain != chain})
    if elsewhere:
        raise ConfigPreflightError(
            f"campaign hotspots name chain(s) {', '.join(elsewhere)}, but the "
            f"campaign target is chain {chain}. Mosaic folds one target chain "
            "from its sequence and cannot condition on another."
        )

    structure = general.target.structure_pdb
    if structure is None:
        raise ConfigPreflightError(
            "Mosaic needs target.structure_pdb to map campaign hotspot residue "
            "numbers onto the target FASTA; residue numbers are not necessarily "
            "1-based sequence positions."
        )
    if not structure.is_file():
        raise ConfigPreflightError(f"Mosaic target PDB does not exist: {structure}")

    residues = _chain_residues(structure, chain)
    if len(residues) != len(target_sequence):
        raise ConfigPreflightError(
            f"Mosaic cannot map the target PDB chain {chain} onto the FASTA: "
            f"the PDB has {len(residues)} CA residues and the FASTA has "
            f"{len(target_sequence)} residues. Provide a structure containing "
            "exactly the chain represented by the FASTA."
        )
    structure_sequence = "".join(amino_acid for _, amino_acid in residues)
    if structure_sequence != target_sequence:
        mismatch = next(
            index for index, pair in enumerate(zip(structure_sequence, target_sequence))
            if pair[0] != pair[1]
        )
        raise ConfigPreflightError(
            f"Mosaic cannot map target PDB chain {chain} onto the FASTA: their "
            f"sequences first differ at position {mismatch + 1} "
            f"({structure_sequence[mismatch]} in the PDB, "
            f"{target_sequence[mismatch]} in the FASTA)."
        )
    positions = {number: index for index, (number, _) in enumerate(residues)}
    absent = sorted({spot.number for spot in hotspots} - positions.keys())
    if absent:
        raise ConfigPreflightError(
            f"campaign hotspots {absent} are absent from target PDB chain {chain}; "
            "Mosaic cannot map them onto the target sequence."
        )
    return tuple(sorted({positions[spot.number] for spot in hotspots}))


def _chain_residues(pdb: Path, chain: str) -> tuple[tuple[int, str], ...]:
    """Author residue numbers and amino acids in sequence order for one chain.

    One CA atom represents one sequence position. Insertion codes make an
    integer-only campaign hotspot ambiguous, so refuse them instead of choosing
    one residue under the same number.
    """
    residues: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    numbers: set[int] = set()
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        if line[21:22].strip().upper() != chain:
            continue
        raw_number = line[22:26].strip()
        insertion = line[26:27].strip()
        if not raw_number.lstrip("-").isdigit():
            continue
        residue_name = line[17:20].strip().upper()
        amino_acid = _THREE_TO_ONE.get(residue_name)
        if amino_acid is None:
            raise ConfigPreflightError(
                f"Mosaic cannot map PDB residue {chain}{raw_number}{insertion}: "
                f"unknown residue name {residue_name!r}."
            )
        number = int(raw_number)
        identity = (number, insertion)
        if identity in seen:
            continue
        seen.add(identity)
        if insertion or number in numbers:
            raise ConfigPreflightError(
                f"Mosaic cannot map PDB residue {chain}{number}{insertion}: "
                "campaign hotspots do not express insertion codes."
            )
        numbers.add(number)
        residues.append((number, amino_acid))
    if not residues:
        raise ConfigPreflightError(
            f"Mosaic found no CA residues for chain {chain} in target PDB {pdb}"
        )
    return tuple(residues)


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
