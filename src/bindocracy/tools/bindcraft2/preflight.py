"""What must be true before BindCraft 2 touches a GPU.

BC2 reads one campaign document and one structure, so every check here is about
one of them disagreeing with the campaign while the run still looks normal.

Four earn their place:

* **Six settings are harness-owned.** The output directory, the design count,
  the trajectory budget, the seed, the resume switch and the whole `targets`
  block are written per task from the campaign and this config. An authored
  value for any of them is a second answer, and it is not the one the run
  would use, because `--set` wins over the document.

* **The epitope is resolved against the real structure.** BC2 maps each hotspot
  onto the target's author numbering and refuses one it cannot place -- but it
  refuses inside the container, with a GPU already allocated and a queue
  already waited out. Checked here instead.

* **A PDB is required, though BC2 also reads mmCIF.** The epitope is verified
  by residue number against author numbering, and this module parses PDB
  columns to do it. Accepting a CIF would mean either writing a second parser
  or shipping an unverified epitope, and an unverified epitope is the failure
  this whole check exists to prevent. Both representations of this project's
  targets are produced together, so requiring the PDB costs nothing.

* **`binder_lengths` must be authored.** It is the one science parameter with no
  safe default: it decides the padded complex size, and therefore how much card
  memory a worker needs and how many workers fit. A campaign that leaves it to
  a preset is a campaign whose cost cannot be predicted at planning time.

No MSA is required, and none is read. BC2 is single-sequence AlphaFold
throughout; `target.msa` is simply not a field this tool can consume, which is
why it is neither required nor refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
)
from bindocracy.tools.bindcraft2.config import BindCraft2Config

# Written per task with `--set`, from the harness's own values. An authored
# value for any of these is refused rather than overwritten.
HARNESS_OWNED_SETTINGS = (
    "campaign_seed",
    "max_trajectories",
    "number_of_final_designs",
    "project_folder",
    "resume",
    "targets",
)

# `target` names a target preset shipped inside the image, which would be a
# second target beside the one the harness passes. `workers_per_gpu` is set from
# the runtime config through the environment, so an authored copy is a second
# answer for the same reason as the six above.
CONFLICTING_SETTINGS = ("target", "workers_per_gpu")


@dataclass(frozen=True)
class BindCraft2Preflight:
    """Everything the plugin needs that came out of a file rather than the YAML."""

    target_sequence: str
    # The authored campaign document, folded into the stored config so the
    # database holds the science rather than a path to it.
    settings: dict
    target_pdb: Path
    chain_id: str
    binder_lengths: tuple[int, ...]
    # The epitope as BC2 expresses it: chain-prefixed residues, in the order the
    # campaign wrote them. Empty when the campaign names none, which BC2 reads
    # as designing against the whole surface.
    hotspots: tuple[str, ...]
    hotspot_string: str
    # The acceptance thresholds this campaign authored, which are the part of
    # `n_passed` that lives in the document rather than in the image.
    authored_filters: tuple[str, ...]
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)

    @property
    def binder_length_range(self) -> tuple[int, int]:
        return (min(self.binder_lengths), max(self.binder_lengths))


def preflight_bindcraft2(
    general: GeneralConfig, model: BindCraft2Config
) -> BindCraft2Preflight:
    """Check the campaign document, and above all what it says about the target."""
    target_pdb = _require_files(general, model)
    settings = _load_json(model.settings.template, "campaign settings")

    _require_harness_owned_absent(settings, model.settings.template)
    binder_lengths = _binder_lengths(settings, model.settings.template)

    chain_id = general.target.chain_id
    residues = _chain_residues(target_pdb, chain_id)
    hotspots, hotspot_string = _resolve_epitope(
        general, residues, target_pdb, model.settings.template
    )

    return BindCraft2Preflight(
        target_sequence=read_single_fasta(general.target.sequence_fasta),
        settings=settings,
        target_pdb=target_pdb,
        chain_id=chain_id,
        binder_lengths=binder_lengths,
        hotspots=hotspots,
        hotspot_string=hotspot_string,
        authored_filters=_authored_filters(settings),
        # The structure is the only file this tool opens that the harness owns.
        # AlphaFold parameters and all three ProteinMPNN variants are baked into
        # the image and covered by the container digest; the campaign document
        # is archived rather than digested, because the run executes the
        # archived copy.
        input_files={"bindcraft2_target_pdb": target_pdb},
    )


def _require_files(general: GeneralConfig, model: BindCraft2Config) -> Path:
    if general.target.structure_pdb is None:
        raise ConfigPreflightError(
            "BindCraft 2 designs against a structure, but the campaign sets no "
            "target.structure_pdb. BC2 itself reads mmCIF too; this plugin "
            "requires the PDB because it verifies the epitope against author "
            "residue numbering before a GPU is allocated, and an unverified "
            "epitope is the failure that check exists to prevent."
        )
    required = {
        "BindCraft 2 campaign settings": model.settings.template,
        "BindCraft 2 container": model.runtime.container,
        "target FASTA": general.target.sequence_fasta,
        "target PDB": general.target.structure_pdb,
    }
    missing = [
        f"{description} does not exist: {path}"
        for description, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise ConfigPreflightError("\n".join(missing))
    return general.target.structure_pdb


def _load_json(path: Path, described_as: str) -> dict:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigPreflightError(
            f"cannot read BindCraft 2 {described_as} {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ConfigPreflightError(
            f"BindCraft 2 {described_as} must be a JSON object: {path}"
        )
    return document


def _require_harness_owned_absent(settings: dict, path: Path) -> None:
    """The settings the harness passes with `--set` must not also be authored.

    `--set` wins over the document, so an authored value is silently the loser:
    the config records one number and the run uses another. `project_folder`
    decides where a task writes and two tasks sharing one would resume each
    other, since a campaign resumes by default.
    """
    owned = [key for key in HARNESS_OWNED_SETTINGS if key in settings]
    if owned:
        raise ConfigPreflightError(
            f"BindCraft 2 campaign settings {path} set {', '.join(owned)}, which "
            "the harness writes per task from the campaign and this config "
            f"({', '.join(HARNESS_OWNED_SETTINGS)}). Remove them: `--set` wins "
            "over the document, so an authored value here is recorded and not "
            "used."
        )
    conflicting = [key for key in CONFLICTING_SETTINGS if key in settings]
    if conflicting:
        raise ConfigPreflightError(
            f"BindCraft 2 campaign settings {path} set {', '.join(conflicting)}. "
            "`target` names a preset target inside the image, which would be a "
            "second target beside the campaign's; `workers_per_gpu` is set from "
            "runtime.workers_per_gpu through the environment. Remove them."
        )


def _binder_lengths(settings: dict, path: Path) -> tuple[int, ...]:
    """`[80]` one length, `[60,100]` a range, `[60,80,100]` a choice of three."""
    value = settings.get("binder_lengths")
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 1 for item in value)
    ):
        raise ConfigPreflightError(
            f"BindCraft 2 campaign settings {path} must set `binder_lengths` to a "
            "list of positive integers: [80] for one length, [60,100] for an "
            "inclusive range, [60,80,100] for a choice. It decides the padded "
            "complex size, and with it how much card memory each worker needs, "
            f"so it is not left to a preset. Got {value!r}."
        )
    return tuple(int(item) for item in value)


def _authored_filters(settings: dict) -> tuple[str, ...]:
    """The acceptance thresholds this document sets, by name.

    Recorded rather than checked. BC2 ships filter defaults in its presets, so
    this is the part of `n_passed` the campaign chose, not the whole of it --
    the complete set is inside the image and is covered by the container digest.
    """
    thresholds = [
        name
        for name in settings
        if name.endswith("_final") and (name.startswith(("min_", "max_")))
    ]
    configured = settings.get("filters")
    if isinstance(configured, dict):
        thresholds += [f"filters.{name}" for name in configured]
    return tuple(sorted(thresholds))


def _resolve_epitope(
    general: GeneralConfig, residues: set[int], target_pdb: Path, path: Path
) -> tuple[tuple[str, ...], str]:
    """The campaign epitope, checked against the structure and rendered.

    BC2's hotspots are chain-then-number against the target's author numbering,
    which is exactly how a campaign writes them, so this is a rendering rather
    than a mapping. An unprefixed number addresses the first selected chain,
    and this run selects one, so the chain is written in explicitly: it costs
    nothing and removes the only ambiguity.
    """
    if not general.target.hotspots:
        return (), ""

    chain = general.target.chain_id
    parsed = parse_hotspots(general.target.hotspots)
    foreign = sorted({spot.chain for spot in parsed if spot.chain not in ("", chain)})
    if foreign:
        raise ConfigPreflightError(
            f"the campaign epitope names chain(s) {', '.join(foreign)}, but this "
            f"run designs against chain {chain} alone. BindCraft 2 resolves a "
            "hotspot against the chains named in `chains`, and one on another "
            "chain cannot be expressed."
        )

    absent = sorted({spot.number for spot in parsed} - residues)
    if absent:
        raise ConfigPreflightError(
            f"BindCraft 2 cannot resolve residue(s) "
            f"{', '.join(str(number) for number in absent)} of chain {chain} "
            f"against {target_pdb}.\n"
            "BC2 refuses an unplaceable hotspot itself, but it does so inside "
            f"the container once a GPU has been allocated. Checked here instead. "
            f"({path} carries no epitope of its own; the harness writes it.)"
        )

    spots = tuple(f"{chain}{spot.number}" for spot in parsed)
    return spots, ",".join(spots)


def _chain_residues(pdb: Path, chain_id: str) -> set[int]:
    """Every CA-bearing residue number of one chain, in author numbering.

    Parsed by column rather than by splitting: PDB is fixed-width and the
    coordinate fields run together on large structures.
    """
    residues: set[int] = set()
    try:
        lines = pdb.read_text().splitlines()
    except OSError as error:
        raise ConfigPreflightError(f"cannot read target PDB {pdb}: {error}") from error
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        if line[12:16].strip() != "CA" or line[21].strip() != chain_id:
            continue
        number = line[22:26].strip()
        if number.lstrip("-").isdigit():
            residues.add(int(number))
    if not residues:
        raise ConfigPreflightError(
            f"target PDB {pdb} has no CA atoms on chain {chain_id}, so BindCraft 2 "
            "has nothing to design against. Check target.chain_id."
        )
    return residues
