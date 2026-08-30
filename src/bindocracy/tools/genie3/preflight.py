"""What must be true before Genie 3 touches a GPU.

Three of these checks exist because the failure they prevent is silent:

* The **problem set** carries its own copy of the target, renumbered and with
  its chains renamed by whatever built it. It can drift away from the campaign
  target while every other check passes, and the run then designs against the
  wrong protein and looks entirely normal. Same class as BoltzGen's spec check.
* The **JAX overlays** are the difference between an AF2 evaluation stage that
  runs on the GPU and one that runs on the CPU forever without erroring. The
  benchmark lost a whole job to this. See docs/known-issues.md section 2.3.
* **`folding.mode: msa`** calls the ColabFold server, which a compute node
  cannot reach.

The fourth, refusing a template that sets the three keys the harness owns, is
about not having two answers to "how many designs did this run ask for".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bindocracy.config.models import GeneralConfig
from bindocracy.config.preflight import ConfigPreflightError, read_single_fasta
from bindocracy.tools.genie3.config import Genie3Config

# Keys the harness writes into every rendered per-task config. A template that
# also sets one has an answer that is silently discarded, so it is refused.
HARNESS_OWNED_KEYS: tuple[tuple[str, ...], ...] = (
    ("paths", "rootdir"),
    # An alias for paths.rootdir in Genie 3's own loader.
    ("paths", "outdir"),
    # Preferred over paths.rootdir by ExperimentRunConfig.rootdir().
    ("generation", "io", "outdir"),
    ("experiment", "seed"),
    ("generation", "dataset", "n_sample"),
)

# 'msa' fetches alignments from api.colabfold.com, which a compute node cannot
# reach. The other two fold offline from the bundled weights.
OFFLINE_FOLDING_MODES = ("template", "singleseq")

# One file each, proving an overlay tree is the thing it claims to be rather
# than an empty directory left by a failed build.
OVERLAY_MARKERS = {
    "jax_plugin_overlay": "jax_plugins",
    "cudnn_overlay": "libcudnn.so.9",
    "cuda_nvcc_overlay": "bin/ptxas",
}

_RESIDUE_NUMBER = re.compile(r"[A-Za-z]*(\d+)$")
# `name-version.dist-info`, which is how each overlay records what it holds.
_DIST_INFO = re.compile(r"(?P<name>.+?)-(?P<version>[0-9][^-]*)\.dist-info$")


@dataclass(frozen=True)
class Genie3Preflight:
    target_sequence: str
    experiment: dict
    selection: str
    dataset_root: Path
    problem_path: Path
    problem: dict
    # Sequences ProteinMPNN writes per backbone. The designs a task produces is
    # this times the backbones it diffuses.
    sequences_per_backbone: int
    input_files: dict[str, Path]

    @property
    def target_length(self) -> int:
        return len(self.target_sequence)

    @property
    def hotspots(self) -> tuple[str, ...]:
        residues = self.problem["target_interface_residues"]["hotspot"]
        return tuple(str(residue) for residue in residues)

    @property
    def binder_length_range(self) -> tuple[int, int]:
        return (int(self.problem["binder_min_length"]), int(self.problem["binder_max_length"]))

    @property
    def cond_strategy(self) -> str:
        return str(self.experiment["generation"]["dataset"].get("cond_strategy", "hotspot"))


def overlay_packages(path: Path) -> dict[str, str]:
    """The packages an overlay tree holds, by version.

    The overlay paths point at where things are *loaded from* -- a `lib`
    directory, a `bin` directory -- while the wheel metadata sits at the root
    of the tree. Walking up finds it. Recording the versions is the point:
    §2.3 says the image alone no longer determines the result, and a path
    alone does not say which build of the plugin was on it.
    """
    packages: dict[str, str] = {}
    for root in (path, *path.parents[:3]):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            match = _DIST_INFO.fullmatch(entry.name)
            if entry.is_dir() and match:
                packages[match.group("name")] = match.group("version")
        if packages:
            break
    return packages


def preflight_genie3(general: GeneralConfig, genie3: Genie3Config) -> Genie3Preflight:
    """Check the environment, the template, and above all the problem set."""
    _require_files(general, genie3)
    _require_overlays(genie3)

    experiment = _load_template(genie3.experiment.template)
    _refuse_harness_owned_keys(experiment)
    _require_evaluation(experiment)

    dataset_root = _dataset_root(experiment)
    selection = _selection(experiment)
    problem_path = dataset_root / "problems" / f"{selection}.json"
    if not problem_path.is_file():
        raise ConfigPreflightError(
            f"Genie 3 problem set has no problem {selection!r}: {problem_path} does not exist.\n"
            "Build it with legacy/genie3/prepare_problemset.py."
        )
    problem = _load_problem(problem_path)

    target_sequence = read_single_fasta(general.target.sequence_fasta)
    _require_same_target(problem, target_sequence, problem_path)
    _require_hotspots(general, problem, problem_path)

    return Genie3Preflight(
        target_sequence=target_sequence,
        experiment=experiment,
        selection=selection,
        dataset_root=dataset_root,
        problem_path=problem_path,
        problem=problem,
        sequences_per_backbone=_sequences_per_backbone(experiment),
        input_files=_input_files(problem, problem_path),
    )


def _require_files(general: GeneralConfig, genie3: Genie3Config) -> None:
    required = {
        "Genie 3 experiment template": genie3.experiment.template,
        "Genie 3 driver": genie3.driver.script,
        "Genie 3 container": genie3.runtime.container,
        "target FASTA": general.target.sequence_fasta,
    }
    missing = [
        f"{description} does not exist: {path}"
        for description, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise ConfigPreflightError("\n".join(missing))


def _require_overlays(genie3: Genie3Config) -> None:
    """Refuse rather than fall back to a CPU JAX.

    Falling back costs a whole job before anyone notices, because nothing
    errors: `nvidia-smi` shows an idle GPU and a live, busy process.
    """
    missing = [
        f"{field}: {getattr(genie3.runtime, field) / marker} does not exist"
        for field, marker in OVERLAY_MARKERS.items()
        if not (getattr(genie3.runtime, field) / marker).exists()
    ]
    if missing:
        raise ConfigPreflightError(
            "Genie 3's JAX CUDA overlays are incomplete, so its AF2 evaluation "
            "stage would run on the CPU and never finish:\n"
            + "\n".join(f"  {item}" for item in missing)
            + "\nBuild them with legacy/genie3/build_jax_overlay.sh."
        )


def _load_template(path: Path) -> dict:
    try:
        experiment = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigPreflightError(f"invalid YAML in Genie 3 experiment: {error}") from error
    if not isinstance(experiment, dict):
        raise ConfigPreflightError(f"Genie 3 experiment must be a mapping: {path}")
    return experiment


def _load_problem(path: Path) -> dict:
    try:
        problem = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConfigPreflightError(f"invalid JSON in Genie 3 problem {path}: {error}") from error
    if not isinstance(problem, dict):
        raise ConfigPreflightError(f"Genie 3 problem must be a JSON object: {path}")
    required = ("target_fasta_filepath", "target_pdb_filepath", "target_interface_residues",
                "binder_min_length", "binder_max_length")
    absent = [key for key in required if key not in problem]
    if absent:
        raise ConfigPreflightError(
            f"Genie 3 problem {path} is missing {', '.join(absent)}"
        )
    return problem


def _refuse_harness_owned_keys(experiment: dict) -> None:
    present = [".".join(key) for key in HARNESS_OWNED_KEYS if _lookup(experiment, key) is not None]
    if present:
        raise ConfigPreflightError(
            "Genie 3's experiment template sets keys the harness writes per task: "
            + ", ".join(present)
            + ".\nEach task gets its own output root, its own seed, and the sample "
            "count from sampling.backbones_per_job, so a value here would be "
            "discarded without saying so. Remove them from the template."
        )


def _require_evaluation(experiment: dict) -> None:
    """No evaluation block means backbones and no sequences.

    Genie 3 diffuses UNK backbones; ProteinMPNN is what turns them into binder
    sequences and ColabFold is what scores them. A generation run without the
    evaluation stage produces nothing this campaign can compare.
    """
    evaluation = experiment.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("version"):
        raise ConfigPreflightError(
            "Genie 3's experiment template has no `evaluation.version`. Genie 3 "
            "generates UNK backbones, so without the evaluation stage the run "
            "produces no sequences at all."
        )

    mode = str(_lookup(experiment, ("evaluation", "folding", "mode")) or "template")
    if mode not in OFFLINE_FOLDING_MODES:
        raise ConfigPreflightError(
            f"Genie 3 evaluation.folding.mode is {mode!r}, which queries "
            "api.colabfold.com. A compute node has no route to it; use "
            + " or ".join(repr(value) for value in OFFLINE_FOLDING_MODES)
            + "."
        )


def _sequences_per_backbone(experiment: dict) -> int:
    """`evaluation.inverse_folding.num_seq`, defaulted as Genie 3 defaults it."""
    value = _lookup(experiment, ("evaluation", "inverse_folding", "num_seq"))
    if value is None:
        return 8  # EvaluationInverseFoldingConfig.num_seq
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigPreflightError(
            f"evaluation.inverse_folding.num_seq must be an integer, not {value!r}"
        ) from error
    if count < 1:
        raise ConfigPreflightError("evaluation.inverse_folding.num_seq must be at least 1")
    return count


def _dataset_root(experiment: dict) -> Path:
    dataset = _lookup(experiment, ("paths", "dataset"))
    if not isinstance(dataset, str) or not dataset.strip():
        raise ConfigPreflightError(
            "Genie 3's experiment template must set `paths.dataset` to the "
            "problem-set directory it designs against."
        )
    root = Path(dataset)
    if not root.is_absolute():
        raise ConfigPreflightError(
            f"Genie 3's `paths.dataset` is a relative path ({root}). Preflight "
            "resolves it against the submitting host's working directory and "
            "the container resolves it against its own, so it must be absolute."
        )
    if not root.is_dir():
        raise ConfigPreflightError(f"Genie 3 problem set does not exist: {root}")
    return root


def _selection(experiment: dict) -> str:
    """The one problem this run designs against, which also names its output.

    Genie 3 accepts a comma-separated list and writes one output directory per
    problem. A campaign follows one target, and the harness reads exactly one
    results table per task, so more than one here is refused rather than
    half-collected.
    """
    selections = _lookup(experiment, ("generation", "dataset", "selections"))
    if selections is None:
        raise ConfigPreflightError(
            "Genie 3's experiment template must set `generation.dataset.selections` "
            "to the problem this run designs against."
        )
    names = [str(name).strip() for name in
             (selections if isinstance(selections, list) else str(selections).split(","))]
    names = [name for name in names if name]
    if len(names) != 1:
        raise ConfigPreflightError(
            f"generation.dataset.selections names {len(names)} problems "
            f"({', '.join(names) or 'none'}); this harness collects one results "
            "table per task, and a campaign follows one target."
        )
    return names[0]


def _require_same_target(problem: dict, target_sequence: str, problem_path: Path) -> None:
    """The problem set's target must be the campaign's target.

    The problem set carries a renumbered copy of the structure and its own
    FASTA, so it can point at another protein entirely while the template, the
    container and the overlays all check out.
    """
    fasta = Path(problem["target_fasta_filepath"])
    if not fasta.is_file():
        raise ConfigPreflightError(
            f"Genie 3 problem {problem_path} names a target FASTA that does not "
            f"exist: {fasta}"
        )
    # A problem-set FASTA is one record whose chains are joined by ':'.
    chains = [chain for chain in _fasta_sequence(fasta).split(":") if chain]
    if target_sequence not in chains:
        raise ConfigPreflightError(
            f"Genie 3 problem {problem_path} is not the campaign target.\n"
            f"  campaign: {len(target_sequence)} aa\n"
            f"  problem:  {', '.join(str(len(chain)) + ' aa' for chain in chains) or 'no chains'}"
            f" in {fasta}\n"
            "A problem set built from another structure designs against the "
            "wrong target silently."
        )


def _require_hotspots(general: GeneralConfig, problem: dict, problem_path: Path) -> None:
    """Genie 3 has no hotspot-free binder mode, and its hotspots are its own.

    The reducer reads `target_interface_residues['hotspot']` unconditionally,
    so an empty list is a run that generates and then fails to score. And
    because the problem set renumbers residues, a campaign that names an
    epitope and a problem set that names a different one would otherwise agree
    with each other only by coincidence.
    """
    residues = problem.get("target_interface_residues", {})
    hotspots = residues.get("hotspot") if isinstance(residues, dict) else None
    if not isinstance(hotspots, list) or not hotspots:
        raise ConfigPreflightError(
            f"Genie 3 problem {problem_path} declares no hotspots. Genie 3's "
            "success reducer reads them unconditionally, so the run would "
            "generate backbones and then fail to score them."
        )

    if not general.target.hotspots:
        return
    campaign = _residue_numbers(general.target.hotspots)
    theirs = _residue_numbers(hotspots)
    if campaign != theirs:
        raise ConfigPreflightError(
            f"Genie 3 problem {problem_path} conditions on residues "
            f"{sorted(theirs)}, but the campaign target names {sorted(campaign)}.\n"
            "The problem set renumbers residues, so rebuild it from the "
            "campaign's epitope rather than leaving the two to disagree."
        )


def _residue_numbers(residues: object) -> set[int]:
    """Residue numbers without their chain, which the problem set renames."""
    numbers = set()
    for residue in residues:  # type: ignore[union-attr]
        match = _RESIDUE_NUMBER.fullmatch(str(residue).strip())
        if match is None:
            raise ConfigPreflightError(f"cannot read a residue number from {residue!r}")
        numbers.add(int(match.group(1)))
    return numbers


def _input_files(problem: dict, problem_path: Path) -> dict[str, Path]:
    """Everything Genie 3 opens to define the target, digested per run."""
    files = {"genie3_problem": problem_path,
             "genie3_target_pdb": Path(problem["target_pdb_filepath"]),
             "genie3_target_fasta": Path(problem["target_fasta_filepath"])}
    for index, path in enumerate(problem.get("target_pdb_filepath_by_chain") or []):
        files[f"genie3_target_pdb_chain_{index}"] = Path(path)

    missing = [f"{label}: {path}" for label, path in files.items() if not path.is_file()]
    if missing:
        raise ConfigPreflightError(
            f"Genie 3 problem {problem_path} references files that do not exist:\n"
            + "\n".join(f"  {item}" for item in missing)
        )
    return files


def _fasta_sequence(path: Path) -> str:
    body = [line.strip() for line in path.read_text().splitlines() if not line.startswith(">")]
    return re.sub(r"\s+", "", "".join(body)).upper()


def _lookup(mapping: dict, keys: tuple[str, ...]) -> Any:
    """Follow a key path, returning None as soon as it stops being a mapping."""
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
