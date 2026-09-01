"""What must be true before Proteina-Complexa touches a GPU.

Proteina-Complexa reads its target through one small file: a registry entry
naming the structure, the crop, the epitope and the binder length. Nothing on
the command line can override a field inside it, so that file is the only place
the campaign's target reaches the tool -- and every check here exists because
the registry can disagree with the campaign while every other check passes.

The one that matters most is the last. The hotspot mask is built by testing
`f"{chain_id}{res_id}"` against each CA atom of the cropped target, and a
hotspot that matches nothing is **silently ignored**: the mask stays False, the
run generates, scores and ranks normally, and the designs are simply not
answers to the question the campaign asked. So the epitope is resolved against
the real PDB here rather than trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
)
from bindocracy.tools.proteina_complexa.config import ProteinaComplexaConfig

# Where the launcher binds the campaign structure inside the container. The
# registry has to name this exact path: it is the one field that ties the entry
# to the file the run actually reads.
CONTAINER_TARGET = "/mnt/bindocracy_target.pdb"

# `A1-201`, `A1-100,B1-50`, or a bare `A` for a whole chain.
_RANGE = re.compile(r"(?P<chain>[A-Za-z0-9])(?:(?P<start>-?\d+)-(?P<end>-?\d+))?$")


@dataclass(frozen=True)
class ProteinaComplexaPreflight:
    """Everything the plugin needs that came out of a file rather than the YAML."""

    target_sequence: str
    # The whole registry, folded into the stored config so the database holds
    # the target definition rather than a path to it.
    registry: dict
    entry: dict
    target_pdb: Path
    hotspots: tuple[str, ...]
    binder_length: tuple[int, int]
    target_input: str
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_proteina_complexa(
    general: GeneralConfig, model: ProteinaComplexaConfig
) -> ProteinaComplexaPreflight:
    """Check the environment, the registry, and above all the epitope."""
    target_pdb = _require_files(general, model)

    registry = _load_registry(model.registry.template)
    entry = _require_entry(registry, model.registry.task_name, model.registry.template)

    _require_bind_point(entry, model.registry.template)
    binder_length = _binder_length(entry, model.registry.template)
    target_input = _target_input(entry, model.registry.template)
    hotspots = _require_hotspots(general, entry, model.registry.template)
    _require_hotspots_resolve(hotspots, target_input, target_pdb)

    return ProteinaComplexaPreflight(
        target_sequence=read_single_fasta(general.target.sequence_fasta),
        registry=registry,
        entry=entry,
        target_pdb=target_pdb,
        hotspots=hotspots,
        binder_length=binder_length,
        target_input=target_input,
        # The structure is the only file this tool opens that the harness owns.
        # The pipeline config, the weights and the Hydra tree are all inside the
        # image, and are covered by the container digest.
        input_files={"proteina_target_pdb": target_pdb},
    )


def _require_files(general: GeneralConfig, model: ProteinaComplexaConfig) -> Path:
    if general.target.structure_pdb is None:
        raise ConfigPreflightError(
            "Proteina-Complexa reads its target from a PDB, but the campaign "
            "sets no target.structure_pdb. Its crop and its epitope are written "
            "as residue numbers against that file, so the CIF is not a "
            "substitute."
        )
    required = {
        "Proteina-Complexa target registry": model.registry.template,
        "Proteina-Complexa container": model.runtime.container,
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


def _load_registry(path: Path) -> dict:
    try:
        registry = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigPreflightError(
            f"invalid YAML in Proteina-Complexa registry: {error}"
        ) from error
    if not isinstance(registry, dict) or not isinstance(
        registry.get("target_dict_cfg"), dict
    ):
        raise ConfigPreflightError(
            f"Proteina-Complexa registry must be a mapping with a "
            f"`target_dict_cfg` block: {path}"
        )
    return registry


def _require_entry(registry: dict, task_name: str, path: Path) -> dict:
    """The entry `++generation.task_name` selects, or a refusal naming the keys.

    A task_name with no entry falls through to whichever target the image
    shipped, which designs against another protein entirely.
    """
    targets = registry["target_dict_cfg"]
    entry = targets.get(task_name)
    if not isinstance(entry, dict):
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} has no target {task_name!r}. "
            f"It defines: {', '.join(sorted(map(str, targets))) or 'nothing'}."
        )
    return entry


def _require_bind_point(entry: dict, path: Path) -> None:
    """The entry must name the path the launcher binds the campaign target to.

    Any other value is a registry pointing at a structure inside the image, or
    at a host path the container cannot see -- both of which fail as a target
    that is not the campaign's.
    """
    target_path = str(entry.get("target_path", ""))
    if target_path != CONTAINER_TARGET:
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} sets target_path to "
            f"{target_path!r}, but the launcher binds the campaign structure at "
            f"{CONTAINER_TARGET}. A registry naming anything else designs "
            "against a different structure, or fails to find one at all."
        )


def _binder_length(entry: dict, path: Path) -> tuple[int, int]:
    """`[min, max]`, which becomes `UniformInt(low, high)` in the dataloader."""
    value = entry.get("binder_length")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} must set binder_length to two "
            f"integers, [min, max]; got {value!r}"
        )
    low, high = int(value[0]), int(value[1])
    if low < 1 or high < low:
        raise ConfigPreflightError(
            f"Proteina-Complexa binder_length {value!r} in {path} is not an "
            "ascending range of positive lengths"
        )
    return (low, high)


def _target_input(entry: dict, path: Path) -> str:
    value = entry.get("target_input")
    if not isinstance(value, str) or not value.strip():
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} must set target_input to the "
            "contig it crops the target to, such as 'A1-201'"
        )
    return value.strip()


def _require_hotspots(general: GeneralConfig, entry: dict, path: Path) -> tuple[str, ...]:
    """The registry's epitope must be the campaign's, in the same notation.

    Unlike Genie 3's problem set, Proteina-Complexa does not renumber: it keys
    hotspots as chain-then-number, which is exactly how a campaign writes them.
    So this is a straight comparison rather than a mapping.

    An empty list is not the same as no epitope. `target_hotspots=[]` is
    `not None` in TargetFeatures, so an all-False hotspot mask still enters the
    batch, which is a different conditioning signal from omitting it.
    """
    raw = entry.get("hotspot_residues", [])
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} must set hotspot_residues to a "
            f"list of residues such as ['A45', 'A67']; got {raw!r}"
        )
    theirs = tuple(item.strip() for item in raw)
    campaign = tuple(str(spot) for spot in parse_hotspots(general.target.hotspots))

    if set(theirs) != set(campaign):
        raise ConfigPreflightError(
            f"Proteina-Complexa registry {path} conditions on "
            f"{sorted(theirs) or 'no residues'}, but the campaign target names "
            f"{sorted(campaign) or 'no residues'}.\n"
            "Proteina-Complexa can express an epitope, so it either designs "
            "against the campaign's or refuses to start."
        )
    return theirs


def _require_hotspots_resolve(
    hotspots: tuple[str, ...], target_input: str, target_pdb: Path
) -> None:
    """Every hotspot must land on a CA atom inside the crop.

    This is the check the tool itself does not do. `load_target_from_pdb` marks
    the mask True for each CA whose `f"{chain_id}{res_id}"` appears in the list
    and does nothing at all with the rest, so a typo, or an epitope numbered
    against the uncropped structure, produces an all-False mask and a run that
    looks entirely normal.
    """
    if not hotspots:
        return

    residues = _ca_residues(target_pdb)
    if not residues:
        raise ConfigPreflightError(
            f"no CA atoms found in the target PDB {target_pdb}; "
            "Proteina-Complexa resolves its epitope against them"
        )
    cropped = _select(residues, target_input, target_pdb)

    unresolved = [spot for spot in hotspots if spot not in cropped]
    if unresolved:
        outside = [spot for spot in unresolved if spot in residues]
        detail = (
            f"\n{', '.join(sorted(outside))} exist in the structure but fall "
            f"outside the crop target_input={target_input!r}."
            if outside
            else ""
        )
        raise ConfigPreflightError(
            f"Proteina-Complexa cannot resolve {', '.join(sorted(unresolved))} "
            f"against {target_pdb}.{detail}\n"
            "A hotspot matching no CA atom is silently ignored: the run "
            "generates, scores and ranks normally against no epitope at all."
        )


def _ca_residues(pdb: Path) -> set[str]:
    """Every CA atom as `{chain}{number}`, which is how the tool keys hotspots.

    Parsed by column rather than by splitting: PDB is fixed-width, and the
    coordinate fields run together on large structures.
    """
    residues: set[str] = set()
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21].strip()
        number = line[22:26].strip()
        if number:
            residues.add(f"{chain}{number}")
    return residues


def _select(residues: set[str], target_input: str, target_pdb: Path) -> set[str]:
    """The residues a contig keeps, in the same `{chain}{number}` notation."""
    kept: set[str] = set()
    for part in target_input.split(","):
        match = _RANGE.fullmatch(part.strip())
        if match is None:
            raise ConfigPreflightError(
                f"cannot read target_input {part.strip()!r}; expected a chain "
                "with an optional residue range, such as 'A1-201' or 'A'"
            )
        chain = match.group("chain")
        if match.group("start") is None:
            kept |= {spot for spot in residues if spot[:1] == chain}
            continue
        span = range(int(match.group("start")), int(match.group("end")) + 1)
        kept |= {f"{chain}{number}" for number in span} & residues
    if not kept:
        raise ConfigPreflightError(
            f"target_input {target_input!r} selects no residues of {target_pdb}"
        )
    return kept
