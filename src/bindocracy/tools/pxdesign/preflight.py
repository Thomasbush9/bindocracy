"""What must be true before PXDesign touches a GPU.

The spec and target identifiers must be safe before allocation:

* The spec carries its own **target path**, so it can drift away from the
  campaign target while everything else checks out, and the run then designs
  against the wrong protein and looks entirely normal. Same class as BoltzGen's
  spec check.
* The spec's **MSA directory** is what stops PXDesign's Protenix stage from
  reaching for the network. The stage skips its MSA lookup only when the entity
  already carries a precomputed directory; without one it shells out to a
  service a compute node cannot reach.
* A spec without **task_name** falls back to the YAML's filename stem, which
  means the results directory is named after whatever the file happened to be
  called -- and the harness archives that file under its source basename.
* Selected polymer atoms need canonical mmCIF **label_seq_id** identifiers.
  PXDesign parses these before cropping; an author residue number is not a
  substitute, and missing polymer labels can fail later in Protenix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    read_single_fasta,
    require_alignment_of,
)
from bindocracy.tools.pxdesign.config import PXDesignConfig

# Both files, or PXDesign's parser raises FileNotFoundError. The directory is
# what the spec names; these are what has to be inside it.
MSA_FILES = ("non_pairing.a3m", "pairing.a3m")

_TASK_NAME = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class PXDesignPreflight:
    target_sequence: str
    spec: dict
    task_name: str
    binder_length: int
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)

    @property
    def hotspots(self) -> tuple[int, ...]:
        """Every hotspot the spec names, across its chains."""
        residues: list[int] = []
        for chain in _chains(self.spec).values():
            residues.extend(int(residue) for residue in (chain.get("hotspots") or []))
        return tuple(sorted(residues))


def preflight_pxdesign(general: GeneralConfig, pxdesign: PXDesignConfig) -> PXDesignPreflight:
    """Check paths and, above all, that the spec aims at the campaign target."""
    required = {
        "PXDesign spec": pxdesign.spec.template,
        "PXDesign container": pxdesign.runtime.container,
        "target FASTA": general.target.sequence_fasta,
    }
    errors = [
        f"{description} does not exist: {path}"
        for description, path in required.items()
        if not path.is_file()
    ]
    if general.target.structure_cif is None:
        errors.append("PXDesign needs target.structure_cif, which is not set")
    elif not general.target.structure_cif.is_file():
        errors.append(f"target structure does not exist: {general.target.structure_cif}")
    if errors:
        raise ConfigPreflightError("\n".join(errors))

    spec = _load_spec(pxdesign.spec.template)
    task_name = _task_name(spec, pxdesign.spec.template)
    binder_length = _binder_length(spec)
    target_file = _target_file(spec, pxdesign.spec.template)

    assert general.target.structure_cif is not None  # checked above
    campaign_target = general.target.structure_cif.resolve()
    if target_file.resolve() != campaign_target:
        raise ConfigPreflightError(
            f"PXDesign spec designs against {target_file}, not the campaign "
            f"target {campaign_target}.\n"
            "A spec aimed at another structure designs against the wrong "
            "target silently."
        )

    msa_dirs = _msa_directories(spec, pxdesign.spec.template)
    _require_matching_hotspots(general, spec, pxdesign.spec.template)

    _require_cif_identifiers(target_file, _chains(spec))
    target_sequence = read_single_fasta(general.target.sequence_fasta)
    input_files = {"spec_target": target_file}
    for chain, directory in msa_dirs.items():
        for name in MSA_FILES:
            input_files[f"msa_{chain}_{name.split('.')[0]}"] = directory / name
        # An alignment of another protein is a well-formed file that folds the
        # wrong target and looks entirely normal.
        require_alignment_of(
            directory / "non_pairing.a3m", target_sequence,
            described_as=f"PXDesign's MSA for chain {chain}",
        )

    return PXDesignPreflight(
        target_sequence=target_sequence,
        spec=spec,
        task_name=task_name,
        binder_length=binder_length,
        input_files=input_files,
    )


def _load_spec(path: Path) -> dict:
    try:
        spec = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigPreflightError(f"invalid YAML in PXDesign spec: {error}") from error
    if not isinstance(spec, dict):
        raise ConfigPreflightError(f"PXDesign spec must be a mapping: {path}")
    return spec


def _task_name(spec: dict, path: Path) -> str:
    """The name of the results directory, and so of the file the adapter reads.

    Undocumented upstream but real. Left out, PXDesign uses the spec filename's
    stem, which would tie the output path to what the file was called -- and
    the run executes an archived copy, so that is one more thing that can move.
    """
    name = spec.get("task_name")
    if not isinstance(name, str) or not _TASK_NAME.fullmatch(name.strip()):
        raise ConfigPreflightError(
            f"PXDesign spec {path} must set `task_name`. It names "
            "design_outputs/<task_name>/, which is where the harness reads this "
            "run's designs from; without it PXDesign uses the spec's filename."
        )
    return name.strip()


def _binder_length(spec: dict) -> int:
    length = spec.get("binder_length")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ConfigPreflightError(
            "PXDesign spec must set `binder_length` to a positive integer"
        )
    return length


def _target_file(spec: dict, path: Path) -> Path:
    target = spec.get("target")
    file = target.get("file") if isinstance(target, dict) else None
    if not isinstance(file, str) or not file.strip():
        raise ConfigPreflightError(f"PXDesign spec {path} has no `target.file`")
    structure = Path(file.strip())
    _require_absolute(structure, "target.file", path)
    if not structure.is_file():
        raise ConfigPreflightError(
            f"PXDesign spec references a target that does not exist: {structure}"
        )
    return structure


def _require_absolute(candidate: Path, key: str, path: Path) -> None:
    """A relative path in a spec means two different files.

    Preflight resolves it against the submitting host's working directory and
    the container resolves it against `--pwd`, which is the task directory. The
    two agree only by accident.
    """
    if not candidate.is_absolute():
        raise ConfigPreflightError(
            f"PXDesign spec {path} sets `{key}` to a relative path ({candidate}). "
            "Preflight and the container resolve it against different working "
            "directories, so it must be absolute."
        )


def _chains(spec: dict) -> dict[str, dict]:
    target = spec.get("target")
    chains = target.get("chains") if isinstance(target, dict) else None
    if not isinstance(chains, dict) or not chains:
        raise ConfigPreflightError("PXDesign spec has no `target.chains` mapping")
    return {
        str(name): chain if isinstance(chain, dict) else {}
        for name, chain in chains.items()
    }


def _require_cif_identifiers(path: Path, chains: dict[str, dict]) -> None:
    """Check raw labels without Biotite's author-field or missing-ID fallback.

    PXDesign selects mmCIF chains by label_asym_id and parses identifiers before
    applying residue crops. Check the entire selected chain, not just hotspots
    or the crop. Entity metadata also identifies modified polymer HETATM rows;
    ATOM is the fallback for CIFs without entity metadata. Nonpolymer missing
    sequence labels are legal and must not be repaired or blanket-rejected.
    """
    from biotite import DeserializationError, InvalidFileError
    from biotite.structure.io import pdbx

    try:
        document = pdbx.CIFFile.read(path)
        block = document.block
        site = block["atom_site"]
        columns = {name: column.as_array(str) for name, column in site.items()}
        required = ("label_asym_id", "label_entity_id", "label_seq_id", "group_PDB")
        for name in required:
            if name not in columns:
                raise ConfigPreflightError(
                    f"PXDesign target mmCIF {path} lacks _atom_site.{name} "
                    f"for selected chains {', '.join(chains)}"
                )
        entity_types: dict[str, str] = {}
        if "entity" in block:
            entity = block["entity"]
            entity_types = dict(zip(
                entity["id"].as_array(str), entity["type"].as_array(str), strict=True,
            ))
        polymer_entities: set[str] = set()
        for category in ("entity_poly", "entity_poly_seq"):
            if category in block:
                polymer_entities.update(block[category]["entity_id"].as_array(str))
    except ConfigPreflightError:
        raise
    except (
        OSError, UnicodeError, DeserializationError, InvalidFileError,
        ValueError, KeyError, IndexError,
    ) as error:
        raise ConfigPreflightError(
            f"PXDesign cannot read target mmCIF identifiers in {path}: {error}"
        ) from error

    seen: set[str] = set()
    models = columns.get("pdbx_PDB_model_num")
    for index, chain in enumerate(columns["label_asym_id"]):
        # The upstream parser requests model=1, i.e. the first model.
        if models is not None and models[index] != models[0]:
            continue
        if chain not in chains:
            continue
        seen.add(chain)
        context = ", ".join(
            f"{name}={columns[name][index]}"
            for name in ("id", "label_comp_id", "auth_seq_id", "auth_asym_id")
            if name in columns
        )
        entity_id = columns["label_entity_id"][index]
        if not entity_id or entity_id in {".", "?"}:
            raise ConfigPreflightError(
                f"PXDesign target mmCIF {path}: chain {chain}, atom row {index + 1} "
                f"({context}) has invalid _atom_site.label_entity_id={entity_id!r}"
            )
        entity_type = entity_types.get(entity_id)
        polymer = (
            entity_type == "polymer"
            or entity_id in polymer_entities
            or (entity_type is None and columns["group_PDB"][index] == "ATOM")
        )
        if not polymer:
            continue
        identifier = columns["label_seq_id"][index]
        if not re.fullmatch(r"[0-9]+", identifier) or int(identifier) <= 0:
            raise ConfigPreflightError(
                f"PXDesign target mmCIF {path}: chain {chain}, polymer atom row "
                f"{index + 1} ({context}) has invalid "
                f"_atom_site.label_seq_id={identifier!r}; expected a positive integer. "
                "PXDesign needs canonical label identifiers before cropping; "
                "author numbering cannot substitute for missing labels."
            )
    for chain in chains:
        if chain not in seen:
            raise ConfigPreflightError(
                f"PXDesign target mmCIF {path}: selected chain {chain} has no "
                "_atom_site.label_asym_id rows in the first model"
            )



def _msa_directories(spec: dict, path: Path) -> dict[str, Path]:
    """Every chain's precomputed MSA directory, checked for both alignments.

    This is what keeps the run offline. PXDesign's Protenix filter looks an
    entity's MSA up in a cache and, failing that, calls out to an MSA service;
    it only skips that entirely when the entity already carries a precomputed
    directory, which is what this key becomes.
    """
    directories: dict[str, Path] = {}
    problems: list[str] = []
    for chain, values in _chains(spec).items():
        msa = values.get("msa")
        if not isinstance(msa, str) or not msa.strip():
            problems.append(f"chain {chain} names no `msa` directory")
            continue
        directory = Path(msa.strip())
        if not directory.is_absolute():
            problems.append(f"chain {chain}: {directory} is not an absolute path")
            continue
        if not directory.is_dir():
            problems.append(f"chain {chain}: {directory} is not a directory")
            continue
        missing = [name for name in MSA_FILES if not (directory / name).is_file()]
        if missing:
            problems.append(f"chain {chain}: {directory} lacks {', '.join(missing)}")
            continue
        directories[chain] = directory

    if problems:
        raise ConfigPreflightError(
            f"PXDesign spec {path} does not give every target chain a usable "
            "precomputed MSA:\n"
            + "\n".join(f"  {problem}" for problem in problems)
            + "\nWithout one, PXDesign's Protenix stage calls an MSA service "
            "that a compute node cannot reach."
        )
    return directories


def _require_matching_hotspots(general: GeneralConfig, spec: dict, path: Path) -> None:
    """A campaign epitope and a spec epitope must not quietly disagree.

    PXDesign hotspots are `label_seq_id` integers on the full-length chain, so
    only the numbers can be compared -- which is enough to catch a spec that
    still carries the residues from somebody else's target.
    """
    if not general.target.hotspots:
        return
    campaign = _residue_numbers(general.target.hotspots)
    theirs = {residue for chain in _chains(spec).values()
              for residue in (int(value) for value in (chain.get("hotspots") or []))}
    if campaign != theirs:
        raise ConfigPreflightError(
            f"PXDesign spec {path} conditions on residues {sorted(theirs) or 'none'}, "
            f"but the campaign target names {sorted(campaign)}."
        )


def _residue_numbers(residues: tuple[str, ...]) -> set[int]:
    numbers = set()
    for residue in residues:
        match = re.fullmatch(r"[A-Za-z]*(\d+)", str(residue).strip())
        if match is None:
            raise ConfigPreflightError(f"cannot read a residue number from {residue!r}")
        numbers.add(int(match.group(1)))
    return numbers
