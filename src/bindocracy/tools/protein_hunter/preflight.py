"""What must be true before Protein-Hunter touches a GPU.

This tool needs no structure at all -- the target is a sequence read from the
campaign FASTA and passed on the command line -- so most of the path checking
the other tools do has nothing to check here. What is left is the one decision
that can quietly change the science and the one that can quietly reach the
network, and they are the same decision:

`--msa_mode` has only `single` and `mmseqs`. `single` folds the target with no
alignment. `mmseqs` calls api.colabfold.com, which a compute node cannot reach,
unless the ColabFold cache is already on disk -- which is what the driver seeds
from the campaign's own alignment. So asking for `mmseqs` without an alignment
to seed from is refused here rather than discovered mid-job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.tools.protein_hunter.config import ProteinHunterConfig

# Written into <save_dir>/0_protein_hunter_design/<chain>_env by the driver.
# `run_mmseqs2` is cache-first: it skips the HTTP call when out.tar.gz exists
# and skips untarring when the a3m files are already there.
CACHE_FILES = ("out.tar.gz", "uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
# The binder is chain A, so the single target chain becomes chain B, and B is
# the prefix the cache directory is named for.
TARGET_CHAIN = "B"


@dataclass(frozen=True)
class ProteinHunterPreflight:
    target_sequence: str
    msa: Path | None
    # The campaign epitope as this tool expresses it: comma-separated residue
    # numbers for the one target chain. Empty when the campaign names none.
    contact_residues: str
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_protein_hunter(
    general: GeneralConfig, protein_hunter: ProteinHunterConfig
) -> ProteinHunterPreflight:
    """Check the files, and that the MSA mode can actually be honoured."""
    required = {
        "Protein-Hunter driver": protein_hunter.driver.script,
        "Protein-Hunter container": protein_hunter.runtime.container,
        "target FASTA": general.target.sequence_fasta,
    }
    errors = [
        f"{description} does not exist: {path}"
        for description, path in required.items()
        if not path.is_file()
    ]
    if errors:
        raise ConfigPreflightError("\n".join(errors))

    msa = _msa_for(general, protein_hunter)
    # The target is passed as an argv value, so this is the only reading of it
    # that happens anywhere -- there is no second copy to drift from.
    target_sequence = read_single_fasta(general.target.sequence_fasta)
    if msa is not None:
        require_alignment_of(msa, target_sequence, described_as="Protein-Hunter's MSA")

    inputs = {"target_fasta": general.target.sequence_fasta}
    if msa is not None:
        inputs["target_msa"] = msa

    return ProteinHunterPreflight(
        target_sequence=target_sequence,
        msa=msa,
        contact_residues=contact_residues(general, target_sequence),
        input_files=inputs,
    )


def contact_residues(general: GeneralConfig, target_sequence: str) -> str:
    """The campaign epitope as `--contact_residues`, or empty if there is none.

    Protein-Hunter folds the target from its sequence, so its residues are
    numbered 1..N over exactly the FASTA the harness passes on the command
    line, and a campaign hotspot maps to its own number. Chains are joined by
    `|`, one group per target chain; a campaign models one target chain, so
    there is one group.

    Everything that cannot be mapped that way is refused. Silently designing
    unconstrained while another tool in the same campaign uses the epitope is
    the comparison quietly becoming meaningless.
    """
    if not general.target.hotspots:
        return ""

    hotspots = parse_hotspots(general.target.hotspots)
    chain = general.target.chain_id.upper()
    elsewhere = sorted({spot.chain for spot in hotspots if spot.chain and spot.chain != chain})
    if elsewhere:
        raise ConfigPreflightError(
            f"campaign hotspots name chain(s) {', '.join(elsewhere)}, but the "
            f"campaign target is chain {chain}. Protein-Hunter folds one target "
            "chain from its sequence and cannot condition on another."
        )

    length = len(target_sequence)
    outside = sorted(spot.number for spot in hotspots if not 1 <= spot.number <= length)
    if outside:
        raise ConfigPreflightError(
            f"campaign hotspots {outside} fall outside the target sequence, "
            f"which is {length} residues. Protein-Hunter numbers the target "
            "1..N over the FASTA it is given, so a residue past the end cannot "
            "be contacted."
        )
    return ",".join(str(number) for number in sorted({s.number for s in hotspots}))


def _msa_for(
    general: GeneralConfig, protein_hunter: ProteinHunterConfig
) -> Path | None:
    """The alignment the cache is seeded from, or None in `single` mode."""
    if protein_hunter.msa.mode == "single":
        return None

    msa = general.target.msa
    if msa is None:
        raise ConfigPreflightError(
            "Protein-Hunter is configured for msa.mode 'mmseqs', but the "
            "campaign target has no `msa`. There is no flag for a precomputed "
            "alignment: the driver seeds the ColabFold cache from this file, "
            "and without it the pipeline calls api.colabfold.com, which a "
            "compute node cannot reach.\n"
            "Set target.msa, or choose msa.mode 'single' and accept folding "
            "the target with no alignment."
        )
    if not msa.is_file():
        raise ConfigPreflightError(f"target MSA does not exist: {msa}")
    if not msa.is_absolute():
        raise ConfigPreflightError(
            f"target.msa is a relative path ({msa}); preflight and the "
            "container resolve it against different working directories."
        )
    if not _has_sequences(msa):
        raise ConfigPreflightError(
            f"target MSA has no sequences to seed the cache with: {msa}"
        )
    return msa


def _has_sequences(msa: Path) -> bool:
    with msa.open() as handle:
        return any(line.startswith(">") for line in handle)
