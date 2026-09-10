"""What must be true before an upstream OpenFold3 run reaches a GPU."""

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
from bindocracy.tools.of3_upstream.config import OF3UpstreamConfig

TOKENS = "ARNDCQEGHILKMFPSTWYV"


@dataclass(frozen=True)
class OF3UpstreamPreflight:
    target_sequence: str
    design_set: DesignSet
    fasta_path: Path
    n_designs: int
    distinct_lengths: int
    msa_path: Path | None
    checkpoint: Path

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_of3_upstream(
    general: GeneralConfig, model: OF3UpstreamConfig
) -> OF3UpstreamPreflight:
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
            f"{len(offenders)} design(s) contain residues OpenFold3 cannot encode "
            f"as protein; first is index {index} with {residues}"
        )

    msa_path: Path | None = None
    if model.protocol.use_target_msa:
        if general.target.msa is None:
            raise ConfigPreflightError(
                "protocol.use_target_msa is true but the campaign names no MSA. "
                "This run passes --use-msa-server false, so OpenFold3 has nothing "
                "to search and would fold the target single-sequence."
            )
        require_alignment_of(
            general.target.msa, target_sequence, described_as="the campaign target"
        )
        msa_path = Path(general.target.msa)

    if not Path(model.runtime.container).exists():
        raise ConfigPreflightError(
            f"openfold3 container not found: {model.runtime.container}"
        )
    checkpoint = Path(model.runtime.checkpoint)
    if not checkpoint.is_file():
        raise ConfigPreflightError(
            f"OpenFold3 checkpoint not found: {checkpoint}.\n"
            "  The image ships no weights; point runtime.checkpoint at the "
            "original .pt (e.g. of3-p2-155k.pt)."
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

    return OF3UpstreamPreflight(
        target_sequence=target_sequence,
        design_set=design_set,
        fasta_path=fasta_path,
        n_designs=design_set.n_designs,
        distinct_lengths=design_set.distinct_lengths,
        msa_path=msa_path,
        checkpoint=checkpoint,
    )
