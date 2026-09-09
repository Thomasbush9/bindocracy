"""What must be true before a Chai-1 run reaches a GPU.

The one check worth the trouble is the MSA. Chai resolves an alignment by
hashing the chain's sequence and looking for `<sha256>.aligned.pqt` in
`msa_directory`; when the file is absent it logs a warning and folds the chain
single-sequence (`chai_lab/data/dataset/msas/load.py:52`). A warning in a
GPU job's stdout is not a failure anybody sees, and the run produces confident
numbers computed against no alignment. That is exactly the failure that made
OpenFold3 and Protenix incomparable for a month. So the alignment is resolved
here, by the same hash the container will use, before anything is allocated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.tools.chai1.config import Chai1Config

# Chai's own token set for proteins. A design carrying anything else is not
# rejected by the container until featurization, one design into the job.
TOKENS = "ARNDCQEGHILKMFPSTWYV"


def expected_pqt_basename(sequence: str) -> str:
    """The filename Chai will look for, by Chai's own rule.

    Transcribed from `chai_lab/data/parsing/msas/aligned_pqt.py:57`:
    sha256 of the uppercased query sequence, then `.aligned.pqt`. Duplicated
    here rather than imported because this runs on the host, outside the
    image -- and verified against the container in `tests/test_chai1.py`, so
    the duplication cannot drift silently.
    """
    digest = hashlib.sha256(sequence.upper().encode()).hexdigest()
    return f"{digest}.aligned.pqt"


@dataclass(frozen=True)
class Chai1Preflight:
    target_sequence: str
    design_set: DesignSet
    fasta_path: Path
    n_designs: int
    distinct_lengths: int
    msa_path: Path | None

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_chai1(general: GeneralConfig, model: Chai1Config) -> Chai1Preflight:
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
    if design_set.n_designs != len(design_set.entries):
        raise ConfigPreflightError(
            f"design set {design_set.digest} claims {design_set.n_designs} designs "
            f"but lists {len(design_set.entries)}"
        )

    offenders = {
        entry.index: sorted({aa for aa in entry.sequence if aa not in TOKENS})
        for entry in design_set.entries
        if any(aa not in TOKENS for aa in entry.sequence)
    }
    if offenders:
        index, residues = next(iter(sorted(offenders.items())))
        raise ConfigPreflightError(
            f"{len(offenders)} design(s) contain residues Chai-1 cannot encode as "
            f"protein; first is index {index} with {residues}"
        )

    # The alignment, resolved the way the container will resolve it.
    msa_path: Path | None = None
    if model.protocol.use_target_msa:
        if model.runtime.msa_directory is None:
            raise ConfigPreflightError(
                "protocol.use_target_msa is true but runtime.msa_directory is unset; "
                "Chai-1 would warn and fold the target single-sequence"
            )
        directory = Path(model.runtime.msa_directory)
        if not directory.is_dir():
            raise ConfigPreflightError(f"msa_directory is not a directory: {directory}")
        candidate = directory / expected_pqt_basename(target_sequence)
        if not candidate.is_file():
            available = sorted(p.name for p in directory.glob("*.aligned.pqt"))
            raise ConfigPreflightError(
                f"no alignment for the campaign target in {directory}.\n"
                f"  Chai-1 hashes the sequence and looks for: {candidate.name}\n"
                f"  present instead: {available or 'nothing'}\n"
                "  Build it with `chai-lab a3m-to-pqt <dir-of-a3m> "
                "--output-directory <msa_directory>`.\n"
                "  Without it Chai-1 logs a warning and folds single-sequence, "
                "which is not visible in the output."
            )
        msa_path = candidate

    if not Path(model.runtime.container).exists():
        raise ConfigPreflightError(f"chai1 container not found: {model.runtime.container}")
    if not Path(model.runtime.scratch).parent.is_dir():
        raise ConfigPreflightError(
            f"scratch parent does not exist: {Path(model.runtime.scratch).parent}"
        )

    if model.sharding.jobs > design_set.n_designs:
        raise ConfigPreflightError(
            f"sharding.jobs={model.sharding.jobs} exceeds the {design_set.n_designs} "
            "designs in the set; some tasks would have nothing to score"
        )

    return Chai1Preflight(
        target_sequence=target_sequence,
        design_set=design_set,
        fasta_path=fasta_path,
        n_designs=design_set.n_designs,
        distinct_lengths=design_set.distinct_lengths,
        msa_path=msa_path,
    )
