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


def _refuse_unhonoured_hotspots(general: GeneralConfig) -> None:
    """Mosaic can take an epitope; this driver does not yet pass one.

    The loss already contains the term that would use it -- the driver builds
    `sp.BinderTargetContact()`, and that term takes an `epitope_idx`. It is
    simply not given one, so the contact term rewards contact anywhere on the
    target. A campaign that names an epitope and runs this unchanged would
    compare an epitope-conditioned tool against an unconstrained one and call
    the difference a result.

    Refusing is the placeholder, not the answer. Passing the epitope is a
    driver argument threaded into `BinderTargetContact(epitope_idx=...)` as
    0-based indices into the target sequence -- and note that indexing by
    position rather than by `seqid - 1` is the bug in docs/known-issues.md
    section 1.4, which is worth not repeating.
    """
    if general.target.hotspots:
        raise ConfigPreflightError(
            f"the campaign names an epitope ({', '.join(general.target.hotspots)}), "
            "and Mosaic's driver does not pass one on. Its loss already has the "
            "term that would use it -- sp.BinderTargetContact() takes an "
            "epitope_idx and is built without one -- so the run would reward "
            "contact anywhere on the target while the rest of the campaign "
            "designs against the epitope.\n"
            "Either clear target.hotspots, or give "
            "drivers/mosaic/hallucinate_binders.py an --epitope argument and "
            "pass it to BinderTargetContact(epitope_idx=...) as 0-based "
            "indices."
        )


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

    _refuse_unhonoured_hotspots(general)

    return MosaicPreflight(target_sequence=read_single_fasta(general.target.sequence_fasta))


