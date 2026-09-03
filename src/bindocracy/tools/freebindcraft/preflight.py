"""What must be true before FreeBindCraft touches a GPU.

BindCraft reads three JSON documents and almost nothing else, so every check
here is about one of them disagreeing with the campaign while the run still
looks entirely normal.

Four earn their place:

* The target document carries its own **starting_pdb**, so it can drift away
  from the campaign structure while everything else checks out, and the run
  then designs against the wrong protein. Same class as BoltzGen's spec check.
* Three of its keys are **written by the driver**, not authored: the output
  directory, the design count, and the epitope. An authored value for any of
  them is a second answer, and the one the run uses is not the one the config
  records. `max_trajectories` in the advanced document is the same.
* The **epitope is resolved against the real PDB**. ColabDesign's `prep_pos`
  asserts `len(idx) == 1` for each hotspot, so an unresolvable one is loud --
  but it is loud *inside the container*, four hours into a queue and after a
  GPU has been allocated. The quiet failure is the other direction: an empty
  `target_hotspot_residues` becomes `hotspot=None` and designs against the
  whole surface, which is why the harness writes that field rather than
  trusting it.
* **`enable_mpnn` must be true.** With it false BindCraft hallucinates
  trajectories and writes nothing to `mpnn_design_stats.csv`, never accepts a
  design, and so never stops for any reason but the trajectory budget. The
  whole output contract here is the MPNN table.

The filter set is not refused, only described. Every filter file the image
ships puts a threshold on at least one metric a PyRosetta-free run does not
compute -- `dG`, `Binder_Energy_Score`, the hydrogen-bond counts -- and
`pr_alternative_utils` fills those with constants "chosen to pass active
filters". Refusing them would leave no usable filter set, so the run records
which of its own thresholds were inert instead.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import (
    ConfigPreflightError,
    parse_hotspots,
    read_single_fasta,
)
from bindocracy.tools.freebindcraft.config import FreeBindCraftConfig

# Written per task by the driver, from the harness's own values. An authored
# value for any of these is refused rather than overwritten.
DRIVER_OWNED_TARGET_KEYS = (
    "design_path",
    "number_of_final_designs",
    "target_hotspot_residues",
)

# Every key is present in every advanced profile shipped by FreeBindCraft, and
# the upstream implementation reads each of them by subscript on at least one
# normal execution path. Keeping the complete document contract here turns a
# late container KeyError into an immediate, actionable config error.
ADVANCED_REQUIRED_KEYS = frozenset(
    {
        "acceptance_rate",
        "af_params_dir",
        "backbone_noise",
        "dalphaball_path",
        "design_algorithm",
        "dssp_path",
        "enable_mpnn",
        "enable_rejection_check",
        "force_reject_AA",
        "greedy_iterations",
        "greedy_percentage",
        "hard_iterations",
        "inter_contact_distance",
        "inter_contact_number",
        "intra_contact_distance",
        "intra_contact_number",
        "max_mpnn_sequences",
        "max_trajectories",
        "model_path",
        "mpnn_fix_interface",
        "mpnn_weights",
        "num_recycles_design",
        "num_recycles_validation",
        "num_seqs",
        "omit_AAs",
        "optimise_beta",
        "optimise_beta_extra_soft",
        "optimise_beta_extra_temp",
        "optimise_beta_recycles_design",
        "optimise_beta_recycles_valid",
        "predict_bigbang",
        "predict_initial_guess",
        "random_helicity",
        "remove_binder_monomer",
        "remove_unrelaxed_complex",
        "remove_unrelaxed_trajectory",
        "rm_template_sc_design",
        "rm_template_sc_predict",
        "rm_template_seq_design",
        "rm_template_seq_predict",
        "sample_models",
        "sampling_temp",
        "save_design_animations",
        "save_design_trajectory_plots",
        "save_mpnn_fasta",
        "save_trajectory_pickle",
        "soft_iterations",
        "start_monitoring",
        "temporary_iterations",
        "use_i_ptm_loss",
        "use_multimer_design",
        "use_rg_loss",
        "use_termini_distance_loss",
        "weights_con_inter",
        "weights_con_intra",
        "weights_helicity",
        "weights_iptm",
        "weights_pae_inter",
        "weights_pae_intra",
        "weights_plddt",
        "weights_rg",
        "weights_termini_loss",
        "zip_animations",
        "zip_plots",
    }
)

# Metrics `functions/pr_alternative_utils.py` fills with fixed constants when
# PyRosetta is absent, with the comment "chosen to pass active filters". A
# threshold on any of them is evaluated against a constant: it neither rejects
# nor measures anything, and a design that "passed" it was never scored on it.
PLACEHOLDER_METRICS = (
    "Binder_Energy_Score",
    "PackStat",
    "dG",
    "dG/dSASA",
    "n_InterfaceHbonds",
    "InterfaceHbondsPercentage",
    "n_InterfaceUnsatHbonds",
    "InterfaceUnsatHbondsPercentage",
)

# `check_filters` reads `Average_X` and `1_X`..`5_X` as one filter on X.
_MODEL_PREFIX = re.compile(r"^(?:Average|[1-5])_")

# `A56`, `56`, `A56-60`, `56-60`, or a bare chain letter meaning the whole
# chain. This is what ColabDesign's `prep_pos` accepts, comma separated.
_SEGMENT = re.compile(r"(?P<chain>[A-Za-z])?(?P<start>\d+)(?:-(?:[A-Za-z])?(?P<end>\d+))?$")


@dataclass(frozen=True)
class FreeBindCraftPreflight:
    """Everything the plugin needs that came out of a file rather than the YAML."""

    target_sequence: str
    # The three authored documents, folded into the stored config so the
    # database holds the science rather than three paths to it.
    target: dict
    filters: dict
    advanced: dict
    target_pdb: Path
    binder_name: str
    binder_length: tuple[int, int]
    # The epitope as BindCraft expresses it: the string the driver writes into
    # `target_hotspot_residues`. Empty when the campaign names none, which is
    # what makes `hotspot=None` a decision rather than an omission.
    hotspot_string: str
    hotspots: tuple[str, ...]
    active_filters: tuple[str, ...]
    inert_filters: tuple[str, ...]
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)


def preflight_freebindcraft(
    general: GeneralConfig, model: FreeBindCraftConfig
) -> FreeBindCraftPreflight:
    """Check the three documents, and above all what they say about the target."""
    target_pdb = _require_files(general, model)

    target = _load_json(model.target.template, "target settings")
    filters = _load_json(model.filters.template, "filter set")
    advanced = _load_json(model.advanced.template, "advanced settings")

    _require_driver_owned_absent(target, model.target.template)
    binder_name = _binder_name(target, model.target.template)
    binder_length = _binder_length(target, model.target.template)
    _require_campaign_structure(target, model.target.template, target_pdb)
    _require_campaign_chain(target, model.target.template, general.target.chain_id)

    _require_advanced(advanced, model.advanced.template)

    residues = _chain_residues(target_pdb, general.target.chain_id)
    hotspots, hotspot_string = _resolve_epitope(
        general, residues, target_pdb, model.target.template
    )
    active, inert = _filter_names(filters, model.filters.template)

    return FreeBindCraftPreflight(
        target_sequence=read_single_fasta(general.target.sequence_fasta),
        target=target,
        filters=filters,
        advanced=advanced,
        target_pdb=target_pdb,
        binder_name=binder_name,
        binder_length=binder_length,
        hotspot_string=hotspot_string,
        hotspots=hotspots,
        active_filters=active,
        inert_filters=inert,
        # The structure is the only file this tool opens that the harness owns.
        # AF2 parameters, ProteinMPNN weights, DSSP and FASPR all ship in the
        # image and are covered by the container digest; the three JSON
        # documents are archived rather than digested, because the run executes
        # the archived copies.
        input_files={"freebindcraft_target_pdb": target_pdb},
    )


def _require_files(general: GeneralConfig, model: FreeBindCraftConfig) -> Path:
    if general.target.structure_pdb is None:
        raise ConfigPreflightError(
            "FreeBindCraft hallucinates against a PDB, but the campaign sets no "
            "target.structure_pdb. ColabDesign's prep_pdb reads PDB only, and "
            "the epitope is written as residue numbers against that file, so "
            "the CIF is not a substitute."
        )
    required = {
        "FreeBindCraft target settings": model.target.template,
        "FreeBindCraft filter set": model.filters.template,
        "FreeBindCraft advanced settings": model.advanced.template,
        "FreeBindCraft driver": model.driver.script,
        "FreeBindCraft container": model.runtime.container,
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
            f"cannot read FreeBindCraft {described_as} {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ConfigPreflightError(f"FreeBindCraft {described_as} must be a JSON object: {path}")
    return document


def _require_driver_owned_absent(target: dict, path: Path) -> None:
    """The three keys the driver writes must not also be authored.

    `design_path` decides where a task writes, and two tasks sharing one resume
    each other: the loop skips any trajectory whose PDB already exists.
    `number_of_final_designs` is `sampling.designs_per_job`, and the run row is
    built from that. `target_hotspot_residues` is the campaign's epitope, and
    an authored copy is one more thing that can drift away from it.
    """
    present = [key for key in DRIVER_OWNED_TARGET_KEYS if key in target]
    if present:
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} set {', '.join(present)}, "
            "which the harness writes per task from the campaign and this "
            f"config ({', '.join(DRIVER_OWNED_TARGET_KEYS)}). Remove them: an "
            "authored value here is a second answer, and it is not the one the "
            "run would use."
        )


def _binder_name(target: dict, path: Path) -> str:
    """What every trajectory, design, and structure file is named after."""
    name = target.get("binder_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name.strip()):
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} must set `binder_name` to a "
            "filename-safe string; it prefixes every design name and every "
            "structure this run writes."
        )
    return name.strip()


def _binder_length(target: dict, path: Path) -> tuple[int, int]:
    """`[min, max]`; each trajectory draws a length uniformly from the range."""
    value = target.get("lengths")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} must set `lengths` to two "
            f"integers, [min, max]; got {value!r}"
        )
    low, high = int(value[0]), int(value[1])
    if low < 1 or high < low:
        raise ConfigPreflightError(
            f"FreeBindCraft lengths {value!r} in {path} is not an ascending "
            "range of positive lengths"
        )
    return (low, high)


def _require_campaign_structure(target: dict, path: Path, campaign: Path) -> None:
    """The document's structure must be the campaign's, by resolved path.

    A target document aimed at another structure designs against the wrong
    protein and produces output that looks entirely normal.
    """
    starting = target.get("starting_pdb")
    if not isinstance(starting, str) or not starting.strip():
        raise ConfigPreflightError(f"FreeBindCraft target settings {path} has no `starting_pdb`")
    structure = Path(starting.strip())
    if not structure.is_absolute():
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} set `starting_pdb` to a "
            f"relative path ({structure}). BindCraft resolves it against its own "
            "working directory, which is the task directory, not the directory "
            "this file was authored in."
        )
    if structure.resolve() != campaign.resolve():
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} design against {structure}, "
            f"not the campaign target {campaign.resolve()}."
        )


def _require_campaign_chain(target: dict, path: Path, chain_id: str) -> None:
    chains = target.get("chains")
    if not isinstance(chains, str) or chains.strip() != chain_id:
        raise ConfigPreflightError(
            f"FreeBindCraft target settings {path} set `chains` to {chains!r}, "
            f"but the campaign target is chain {chain_id!r}. BindCraft designs "
            "against the chains named here and nothing re-checks them."
        )


def _require_advanced(advanced: dict, path: Path) -> None:
    """The advanced profile: no trajectory budget of its own, and MPNN on."""
    missing = sorted(ADVANCED_REQUIRED_KEYS - advanced.keys())
    if missing:
        raise ConfigPreflightError(
            f"FreeBindCraft advanced settings {path} are missing "
            f"{', '.join(missing)}. BindCraft reads these keys by subscript, "
            "so an absent one is a KeyError inside the container."
        )
    budget = advanced.get("max_trajectories")
    if "max_trajectories" not in advanced or budget is not False:
        raise ConfigPreflightError(
            f"FreeBindCraft advanced settings {path} set max_trajectories to "
            f"{budget!r}. The harness writes it per task from "
            "`sampling.max_trajectories`; the key must be present and `false`, "
            "which is what the profiles the image ships already say."
        )
    if advanced.get("enable_mpnn") is not True:
        raise ConfigPreflightError(
            f"FreeBindCraft advanced settings {path} disable MPNN. With "
            "enable_mpnn false BindCraft hallucinates trajectories and writes "
            "nothing to mpnn_design_stats.csv, never accepts a design, and "
            "produces no output this harness can collect."
        )


def _filter_names(filters: dict, path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The thresholds this run enforces, and which of them cannot bite.

    `check_filters` treats `Average_X`, `1_X` .. `5_X` as one filter on X, so
    the names here are the metrics rather than the columns. An entry whose
    threshold is null is not a filter at all and is not counted.
    """
    active: set[str] = set()
    for key, condition in filters.items():
        if not isinstance(condition, dict):
            raise ConfigPreflightError(
                f"FreeBindCraft filter set {path} has a non-object entry for "
                f"{key!r}; BindCraft reads every entry as "
                '{"threshold": ..., "higher": ...}'
            )
        if key.endswith("InterfaceAAs"):
            # Per-amino-acid caps, one nested object per residue type.
            nested_active = False
            for amino_acid, value in condition.items():
                nested_active = (
                    _active_filter_condition(value, f"{key}.{amino_acid}", path) or nested_active
                )
            if nested_active:
                active.add(_metric_of(key))
            continue
        if _active_filter_condition(condition, key, path):
            active.add(_metric_of(key))

    inert = tuple(sorted(active & set(PLACEHOLDER_METRICS)))
    return tuple(sorted(active)), inert


def _active_filter_condition(condition: object, label: str, path: Path) -> bool:
    """Validate one threshold object and say whether it is active."""
    if not isinstance(condition, dict):
        raise ConfigPreflightError(
            f"FreeBindCraft filter set {path} has a non-object condition for "
            f'{label!r}; expected {{"threshold": ..., "higher": ...}}'
        )
    if "threshold" not in condition:
        raise ConfigPreflightError(
            f"FreeBindCraft filter set {path} condition {label!r} is missing "
            "`threshold`; BindCraft reads it by subscript."
        )
    threshold = condition["threshold"]
    if threshold is None:
        if "higher" in condition and not isinstance(condition["higher"], bool):
            raise ConfigPreflightError(
                f"FreeBindCraft filter set {path} condition {label!r} must set "
                "`higher` to true or false."
            )
        return False
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ConfigPreflightError(
            f"FreeBindCraft filter set {path} condition {label!r} has a "
            f"non-numeric threshold {threshold!r}."
        )
    if not isinstance(condition.get("higher"), bool):
        raise ConfigPreflightError(
            f"FreeBindCraft filter set {path} condition {label!r} must set "
            "`higher` to true or false when its threshold is active."
        )
    return True


def _metric_of(column: str) -> str:
    """`Average_i_pTM` and `1_i_pTM` are the same filter, on `i_pTM`."""
    return _MODEL_PREFIX.sub("", column, count=1)


def _resolve_epitope(
    general: GeneralConfig, residues: set[int], target_pdb: Path, path: Path
) -> tuple[tuple[str, ...], str]:
    """The campaign epitope, checked against the structure and rendered.

    BindCraft's hotspots are chain-then-number against the author numbering of
    the starting PDB, which is exactly how a campaign writes them, so this is a
    rendering rather than a mapping. What it is not is optional: ColabDesign
    resolves each hotspot with `assert len(idx) == 1`, and that assertion fires
    inside the container with a GPU already allocated.
    """
    if not general.target.hotspots:
        return (), ""

    chain = general.target.chain_id
    parsed = parse_hotspots(general.target.hotspots)
    foreign = sorted({spot.chain for spot in parsed if spot.chain not in ("", chain)})
    if foreign:
        raise ConfigPreflightError(
            f"the campaign epitope names chain(s) {', '.join(foreign)}, but this "
            f"run designs against chain {chain} alone. BindCraft resolves a "
            "hotspot against the chains in `chains`, and one on another chain "
            "cannot be expressed."
        )

    absent = sorted({spot.number for spot in parsed} - residues)
    if absent:
        raise ConfigPreflightError(
            f"FreeBindCraft cannot resolve residue(s) "
            f"{', '.join(str(number) for number in absent)} of chain {chain} "
            f"against {target_pdb}.\n"
            "ColabDesign asserts that every hotspot matches exactly one CA "
            "atom, and it does so inside the container once the GPU has been "
            f"allocated. Checked here instead. ({path} carries no epitope of "
            "its own; the harness writes it.)"
        )

    spots = tuple(f"{chain}{spot.number}" for spot in parsed)
    return spots, ",".join(spots)


def _chain_residues(pdb: Path, chain_id: str) -> set[int]:
    """Every CA-bearing residue number of one chain, as ColabDesign sees them.

    `prep_pdb` is called with `ignore_missing=True`, which keeps exactly the
    residues whose CA coordinate is present, and indexes them by the PDB's own
    residue numbers. Parsed by column rather than by splitting: PDB is
    fixed-width and the coordinate fields run together on large structures.
    """
    residues: set[int] = set()
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        if line[12:16].strip() != "CA" or line[21].strip() != chain_id:
            continue
        number = line[22:26].strip()
        if number.lstrip("-").isdigit():
            residues.add(int(number))
    if not residues:
        raise ConfigPreflightError(
            f"no CA atoms of chain {chain_id} found in {pdb}; BindCraft builds its target from them"
        )
    return residues


def parse_hotspot_string(value: str, default_chain: str) -> tuple[str, ...]:
    """Read a `target_hotspot_residues` string the way `prep_pos` does.

    Used by the adapter to check that the epitope the output reports is the one
    the manifest recorded, which is the only place a driver that wrote the
    wrong string would show.
    """
    spots: list[str] = []
    for part in value.split(","):
        segment = part.strip()
        if not segment:
            continue
        if segment.isalpha():
            # A bare chain letter means the whole chain, which is not an
            # epitope; kept verbatim so the caller sees it rather than a
            # silently empty result.
            spots.append(segment.upper())
            continue
        match = _SEGMENT.fullmatch(segment)
        if match is None:
            raise ConfigPreflightError(
                f"cannot read {segment!r} as a BindCraft hotspot; expected a "
                "number, optionally prefixed by a chain, optionally a range"
            )
        chain = (match.group("chain") or default_chain).upper()
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        spots.extend(f"{chain}{number}" for number in range(start, end + 1))
    return tuple(spots)
