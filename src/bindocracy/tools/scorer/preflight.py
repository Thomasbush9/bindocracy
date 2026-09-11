"""What must be true before a scoring run touches a GPU.

DRAFT. The checks here are the ones that would otherwise fail silently or
expensively, which is the standard `harness-design.md` §5 sets: refuse, do not
warn, and do it before the allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.load import sha256_file
from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.runs.designset import DesignSet, DesignSetError
from bindocracy.store.records import sha256_text
from bindocracy.tools.scorer.config import (
    DEPRECATED_MODELS,
    MOVED_TO_FUNCTIONS,
    ScorerConfig,
)

# Amino acids mosaic's TOKENS covers. A design containing anything else cannot
# be one-hot encoded and would fail inside the container, one design at a time,
# after the GPU was already allocated.
TOKENS = "ARNDCQEGHILKMFPSTWYV"


@dataclass(frozen=True)
class ScorerPreflight:
    target_sequence: str
    design_set: DesignSet
    fasta_path: Path
    n_designs: int
    distinct_lengths: int
    # None when the run uses the image's own source. Otherwise a content hash
    # of the tree bound over it, so the run records what actually executed.
    dev_source_sha256: str | None = None

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def source_tree_digest(root: Path) -> str:
    """A content hash of every Python file under `root`.

    Deterministic across machines and clocks: sorted relative paths, each with
    its own file digest, hashed together. Byte content only, no mtimes, so
    re-cloning the same commit gives the same digest.

    This is what makes a `dev_source` run auditable rather than merely
    overridden. `known-issues.md` §2.3 sets the rule for the Genie 3 overlays:
    where something outside the image changes the result, the image alone no
    longer determines it, so record the something.
    """
    parts: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts.append(f"{path.relative_to(root)}:{sha256_file(path)}")
    if not parts:
        raise ConfigPreflightError(f"dev_source has no Python files: {root}")
    return sha256_text("\n".join(parts))


def preflight_scorer(general: GeneralConfig, model: ScorerConfig) -> ScorerPreflight:
    """Refuse a scoring run that cannot mean what it says."""
    # A reader that validates, launches, succeeds and measures nothing is
    # worse than one that fails: nothing in the output says the measurement is
    # missing. Refuse before the allocation and name where it moved.
    asked_for = [
        name for name in MOVED_TO_FUNCTIONS if getattr(model.readers, name, False)
    ]
    if asked_for:
        moved = "\n".join(
            f"    readers.{name}  ->  functions.{name}   ({MOVED_TO_FUNCTIONS[name]})"
            for name in asked_for
        )
        raise ConfigPreflightError(
            "these are not fold conditions and were never honoured here -- the "
            "launcher only ever emitted complex and monomer, so a run asking for "
            f"them measured nothing:\n{moved}\n"
            "  They are computed from a fold that already happened, by the "
            "functions stage, which can also backfill them over every structure "
            "already saved. See docs/scoring-functions.md."
        )

    reason = DEPRECATED_MODELS.get(model.model.name)
    if reason is not None and not model.allow_deprecated:
        raise ConfigPreflightError(
            f"{model.model.name} is deprecated as a scorer.\n  {reason}\n"
            "  Set allow_deprecated: true only to reproduce a historical "
            "comparison on purpose."
        )

    target_sequence = read_single_fasta(general.target.sequence_fasta)

    # The design set, and the FASTA beside it.
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

    # An unencodable residue fails inside the container after the allocation,
    # so find it here instead.
    offenders = {
        entry.index: sorted({aa for aa in entry.sequence if aa not in TOKENS})
        for entry in design_set.entries
        if any(aa not in TOKENS for aa in entry.sequence)
    }
    if offenders:
        index, residues = next(iter(sorted(offenders.items())))
        raise ConfigPreflightError(
            f"{len(offenders)} design(s) contain residues mosaic cannot encode; "
            f"first is index {index} with {residues}"
        )

    # The MSA, when the protocol says one is used. `require_alignment_of`
    # checks the alignment's query row actually is this target -- an a3m for a
    # different protein loads fine and silently changes every number.
    if model.model.use_target_msa:
        if general.target.msa is None:
            raise ConfigPreflightError(
                f"model {model.model.name} is configured with use_target_msa: true "
                "but the campaign names no MSA"
            )
        require_alignment_of(
            general.target.msa, target_sequence, described_as="the campaign target"
        )

    # Runtime paths. A missing weights tree turns into a download attempt under
    # offline mode, which surfaces as an unrelated network error deep in a
    # model constructor.
    for label, path in (
        ("container", model.runtime.container),
        ("weights", model.runtime.weights),
        ("exec_wrapper", model.runtime.exec_wrapper),
    ):
        if not Path(path).exists():
            raise ConfigPreflightError(f"scorer {label} not found: {path}")
    dev_source_sha256 = None
    if model.runtime.dev_source is not None:
        dev_root = Path(model.runtime.dev_source)
        if not (dev_root / "mosaic").is_dir():
            raise ConfigPreflightError(
                f"dev_source has no mosaic/ subdirectory: {dev_root}; "
                "mosaic-exec.sh binds this over /opt/mosaic/src and refuses otherwise"
            )
        dev_source_sha256 = source_tree_digest(dev_root)

    if not Path(model.runtime.scratch).parent.is_dir():
        raise ConfigPreflightError(
            f"scratch parent does not exist: {Path(model.runtime.scratch).parent}"
        )

    # Sharding wider than the set would plan empty tasks, and an empty task's
    # status is indistinguishable from a task that died before writing one.
    if model.sharding.jobs > design_set.n_designs:
        raise ConfigPreflightError(
            f"sharding.jobs={model.sharding.jobs} exceeds the {design_set.n_designs} "
            "designs in the set; some tasks would have nothing to score"
        )

    return ScorerPreflight(
        target_sequence=target_sequence,
        design_set=design_set,
        fasta_path=fasta_path,
        n_designs=design_set.n_designs,
        distinct_lengths=design_set.distinct_lengths,
        dev_source_sha256=dev_source_sha256,
    )
