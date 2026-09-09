"""What must be true before an AlphaFold 3 run reaches a GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.tools.af3.config import AF3Config

TOKENS = "ARNDCQEGHILKMFPSTWYV"
# The parameter file `af3.def` expects to find under runtime.model_dir.
WEIGHTS_FILE = "af3.bin.zst"


@dataclass(frozen=True)
class AF3Preflight:
    target_sequence: str
    design_set: DesignSet
    fasta_path: Path
    n_designs: int
    distinct_lengths: int
    msa_path: Path | None
    weights_path: Path

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_af3(general: GeneralConfig, model: AF3Config) -> AF3Preflight:
    target_sequence = read_single_fasta(general.target.sequence_fasta)

    manifest_path = Path(model.design_set)
    try:
        design_set = DesignSet.read(manifest_path)
    except DesignSetError as exc:
        raise ConfigPreflightError(str(exc)) from exc

    fasta_path = design_set.fasta_path(manifest_path)
    if not fasta_path.is_file():
        raise ConfigPreflightError(
            f"design-set FASTA missing beside its manifest: {fasta_path}"
        )

    offenders = {
        entry.index: sorted({aa for aa in entry.sequence if aa not in TOKENS})
        for entry in design_set.entries
        if any(aa not in TOKENS for aa in entry.sequence)
    }
    if offenders:
        index, residues = next(iter(sorted(offenders.items())))
        raise ConfigPreflightError(
            f"{len(offenders)} design(s) contain residues AlphaFold 3 cannot encode "
            f"as protein; first is index {index} with {residues}"
        )

    # The alignment. AF3 is handed it inline in the job JSON, so unlike Chai
    # there is no hash-named file to resolve -- but the same class of mistake is
    # possible, so the a3m's query row is checked against the campaign target.
    # An a3m for a different protein loads fine and changes every number.
    msa_path: Path | None = None
    if model.protocol.use_target_msa:
        if general.target.msa is None:
            raise ConfigPreflightError(
                "protocol.use_target_msa is true but the campaign names no MSA; "
                "AF3 runs with --norun_data_pipeline and has no database to "
                "search, so it would fold the target single-sequence"
            )
        require_alignment_of(
            general.target.msa, target_sequence, described_as="the campaign target"
        )
        msa_path = Path(general.target.msa)

    if not Path(model.runtime.container).exists():
        raise ConfigPreflightError(f"af3 container not found: {model.runtime.container}")

    # The weights are deliberately outside the image, so their absence is a
    # configuration error rather than a broken build -- and it surfaces here
    # instead of as a confusing failure inside run_alphafold.
    weights = Path(model.runtime.model_dir) / WEIGHTS_FILE
    if not weights.is_file():
        raise ConfigPreflightError(
            f"AlphaFold 3 parameters not found at {weights}.\n"
            "  They are not in the image by design: WEIGHTS_TERMS_OF_USE.md\n"
            "  restricts distribution, so af3.def binds them read-only instead."
        )

    if not Path(model.runtime.work_root).parent.is_dir():
        raise ConfigPreflightError(
            f"work_root parent does not exist: {Path(model.runtime.work_root).parent}"
        )

    if model.sharding.jobs > design_set.n_designs:
        raise ConfigPreflightError(
            f"sharding.jobs={model.sharding.jobs} exceeds the {design_set.n_designs} "
            "designs in the set; some tasks would have nothing to score"
        )

    return AF3Preflight(
        target_sequence=target_sequence,
        design_set=design_set,
        fasta_path=fasta_path,
        n_designs=design_set.n_designs,
        distinct_lengths=design_set.distinct_lengths,
        msa_path=msa_path,
        weights_path=weights,
    )
