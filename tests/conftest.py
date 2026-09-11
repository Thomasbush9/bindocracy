"""Shared fixtures: a valid config pair on disk, and a fake Mosaic run.

The fake driver and exec wrapper let the whole workflow run locally without a
GPU or a container, which is the point of the local end-to-end test.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

# Stand-in for singularity/mosaic-exec.sh: drop the "python" argument and run
# the driver with this interpreter, which needs no container.
FAKE_EXEC_WRAPPER = f'#!/usr/bin/env bash\nshift\nexec {sys.executable} "$@"\n'

# Writes the real output contract with no models involved.
FAKE_DRIVER = '''\
import argparse, json, os
from datetime import UTC, datetime

ap = argparse.ArgumentParser()
for flag in ("--target-fasta", "--target-msa", "--save-dir"):
    ap.add_argument(flag, required=True)
for flag in ("--binder-length", "--task-id", "--seed-base", "--n-designs",
             "--soft-steps", "--sharpen-steps", "--final-steps"):
    ap.add_argument(flag, type=int, required=True)
ap.add_argument("--max-runtime", type=float, required=True)
ap.add_argument("--epitope", default="")
a = ap.parse_args()

save_dir = os.path.abspath(a.save_dir)
os.makedirs(save_dir, exist_ok=True)
now = datetime.now(UTC).isoformat()
with open(os.path.join(save_dir, "designs.jsonl"), "a") as out:
    for index in range(a.n_designs):
        out.write(json.dumps({
            "native_id": f"task-{a.task_id:04d}-design-{index:06d}",
            "sequence": "ACDEFGHIKL"[: max(2, a.binder_length % 10 + 2)],
            "seed": a.seed_base + a.task_id * 100_000 + index,
            "ranking_loss": -0.1 * (index + 1),
            "completed_at": now,
            "seconds": 0.1,
        }) + "\\n")
with open(os.path.join(save_dir, "status.json"), "w") as out:
    json.dump({
        "task_id": a.task_id, "status": "succeeded",
        "started_at": now, "finished_at": now,
        "n_attempted": a.n_designs, "n_produced": a.n_designs,
        "output_file": "designs.jsonl", "error": None,
    }, out)
'''


# A driver that starts, writes nothing, and exits non-zero -- an ordinary tool
# failure, which must still reach the database rather than aborting the workflow.
FAILING_DRIVER = "import sys\nsys.stderr.write('boom\\n')\nraise SystemExit(3)\n"


def write_configs(root: Path, *, driver_source: str | None = None, **mosaic_overrides
                  ) -> tuple[Path, Path]:
    """Write a valid general + Mosaic config pair and everything they reference."""
    fasta = root / "target.fasta"
    pdb = root / "target.pdb"
    msa = root / "target.a3m"
    driver = root / "hallucinate_binders.py"
    container = root / "mosaic.sif"
    wrapper = root / "mosaic-exec.sh"
    weights = root / "weights"
    scratch = root / "scratch" / "mosaic"

    (weights / "boltz").mkdir(parents=True, exist_ok=True)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    fasta.write_text(">target\nACDEFG\n")
    pdb.write_text("\n".join(
        f"ATOM  {index:>5d}  CA  {residue} A{index:>4d}    "
        f"{index:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C"
        for index, residue in enumerate(
            ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY"), start=1
        )
    ) + "\nEND\n")
    msa.write_text(">target\nACDEFG\n")
    driver.write_text(driver_source or FAKE_DRIVER)
    container.write_bytes(b"fixture")
    wrapper.write_text(FAKE_EXEC_WRAPPER)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "structure_pdb": str(pdb),
            "msa": str(msa),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    mosaic = {
        "schema_version": 2,
        "name": "mosaic-test",
        "tool": "mosaic",
        "driver": {"script": str(driver)},
        "sampling": {
            "binder_length": 70,
            "jobs": 2,
            "designs_per_job": 4,
            "max_runtime_hours": 1,
            "seed_base": 0,
        },
        "runtime": {
            "container": str(container),
            "weights": str(weights),
            "exec_wrapper": str(wrapper),
            "scratch": str(scratch),
        },
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "walltime": "02:00:00"},
    }
    for section, values in mosaic_overrides.items():
        mosaic[section].update(values)

    model_path = root / "mosaic.yaml"
    model_path.write_text(yaml.safe_dump(mosaic, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_configs(tmp_path)


DRIVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "drivers" / "mosaic" / "hallucinate_binders.py"
)

# The driver runs inside mosaic.sif and imports jax and mosaic at module scope;
# neither exists here. Stubbing them exposes the driver's real CLI surface and
# its file-writing helpers to tests, which is the only way to check that what
# the connector launches is what the driver actually accepts.
_CONTAINER_MODULES = [
    "jax", "jax.numpy", "mosaic", "mosaic.common", "mosaic.losses",
    "mosaic.losses.structure_prediction", "mosaic.losses.protein_mpnn",
    "mosaic.losses.transformations", "mosaic.models", "mosaic.models.boltz2",
    "mosaic.optimizers", "mosaic.proteinmpnn", "mosaic.proteinmpnn.mpnn",
    "mosaic.structure_prediction",
]


@pytest.fixture
def driver():
    """Import hallucinate_binders.py with its container-only imports stubbed."""
    import importlib.util
    import sys
    import types

    saved = {name: sys.modules.get(name) for name in _CONTAINER_MODULES}
    for name in _CONTAINER_MODULES:
        module = types.ModuleType(name)
        module.__getattr__ = lambda attribute: object()
        sys.modules[name] = module
    for name in _CONTAINER_MODULES:  # `import a.b as x` needs b on a
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules[parent], child, sys.modules[name])
    try:
        spec = importlib.util.spec_from_file_location("hallucinate_binders", DRIVER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def design_line(task_id: int, index: int, sequence: str = "ACDEFG", **overrides) -> str:
    """One valid designs.jsonl record, with fields overridable per test."""
    record = {
        "native_id": f"task-{task_id:04d}-design-{index:06d}",
        "sequence": sequence,
        "seed": task_id * 100_000 + index,
        "ranking_loss": -0.5 - index,
        "completed_at": f"2026-08-28T12:00:{index:02d}+00:00",
        "seconds": 420.0,
    }
    record.update(overrides)
    return json.dumps(record)


def write_task(
    run_dir: Path,
    task_id: int,
    lines: list[str],
    *,
    text: str | None = None,
    status: dict | None = None,
) -> None:
    """Populate one task directory. `status=None` leaves the status file out."""
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = text if text is not None else "".join(line + "\n" for line in lines)
    if body:
        (task_dir / "designs.jsonl").write_text(body)
    if status is not None:
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-28T11:00:00+00:00",
            "finished_at": "2026-08-28T12:00:00+00:00",
            "n_attempted": len(lines),
            "n_produced": len(lines),
            "output_file": "designs.jsonl",
            "error": None,
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("fake log\n")


BOLTZGEN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "boltzgen"


def write_boltzgen_configs(root: Path, **overrides) -> tuple[Path, Path]:
    """A general + BoltzGen pair. BoltzGen needs geometry, not a sequence."""
    fasta = root / "target.fasta"
    msa = root / "target.a3m"
    cif = root / "target.cif"
    container = root / "boltzgen.sif"
    spec = root / "binder_spec.yaml"

    fasta.write_text(">target\nACDEFG\n")
    msa.write_text(">target\nACDEFG\n")
    cif.write_text("data_target\n#\n")
    container.write_bytes(b"fixture")
    spec.write_text(yaml.safe_dump({
        "entities": [
            {"protein": {"id": "B", "sequence": "70..90"}},
            {"file": {"path": str(cif), "include": [{"chain": {"id": "A"}}]}},
        ]
    }, sort_keys=False))

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "msa": str(msa),
            "chain_id": "A",
            "hotspots": [],
            "structure_cif": str(cif),
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    boltzgen = {
        "schema_version": 1,
        "name": "boltzgen-test",
        "tool": "boltzgen",
        "spec": {"template": str(spec)},
        "sampling": {"jobs": 1, "num_designs": 8, "budget": 4,
                     "protocol": "protein-anything", "filter_biased": False},
        "runtime": {"container": str(container), "node_tmp_root": str(root / "nodetmp")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 96, "walltime": "04:00:00"},
    }
    for section, values in overrides.items():
        boltzgen[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "boltzgen.yaml"
    model_path.write_text(yaml.safe_dump(boltzgen, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def boltzgen_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_boltzgen_configs(tmp_path)


def write_boltzgen_task(
    run_dir: Path, task_id: int, *, rows: int | None = None, status: dict | None = None
) -> None:
    """Lay out one BoltzGen task from the committed real-output fixture."""
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    ranked = task_dir / "final_ranked_designs"
    ranked.mkdir(parents=True, exist_ok=True)

    lines = (BOLTZGEN_FIXTURE / "all_designs_metrics.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    if rows is not None:
        body = body[:rows]
    (ranked / "all_designs_metrics.csv").write_text("\n".join([header, *body]) + "\n")

    structures = ranked / "final_40_designs"
    structures.mkdir(exist_ok=True)
    for line in body:
        fields = line.split(",")
        (structures / f"rank{int(fields[1]):02d}_{fields[4]}").write_text("data_design\n#\n")

    if status is not None:
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-27T10:00:00+00:00",
            "finished_at": "2026-08-27T10:23:00+00:00",
            "n_attempted": 8,
            "exit_code": 0,
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("boltzgen log\n")


GENIE3_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "genie3"
# The fixture rows are unmodified output of the 2026-08-26 benchmark run, whose
# problem was keyed `dio3_cut`. The key names the output tree, so the test
# configs use it too and the fixture never has to be edited.
GENIE3_SELECTION = "dio3_cut"
GENIE3_TARGET = "ACDEFGHIKLMNPQRSTVWY"


def write_genie3_problemset(root: Path, *, key: str = GENIE3_SELECTION,
                            sequence: str = GENIE3_TARGET,
                            hotspots: list[str] | None = None) -> Path:
    """A Genie 3 problem set: one problem JSON and the target files it names."""
    dataset = root / "genie3_dataset"
    (dataset / "problems").mkdir(parents=True, exist_ok=True)
    (dataset / "targets" / "pdb").mkdir(parents=True, exist_ok=True)
    (dataset / "targets" / "fasta").mkdir(parents=True, exist_ok=True)

    fasta = dataset / "targets" / "fasta" / f"{key}.fasta"
    fasta.write_text(f">{key}\n{sequence}\n")
    pdb = dataset / "targets" / "pdb" / f"{key}.pdb"
    pdb.write_text("ATOM      1  CA  ALA B   1       0.000   0.000   0.000\n")
    chain_pdb = dataset / "targets" / "pdb" / f"{key}-chain_B.pdb"
    chain_pdb.write_text(pdb.read_text())

    (dataset / "problems" / f"{key}.json").write_text(json.dumps({
        "key": key,
        "name": key,
        "target_pdb_filepath": str(pdb),
        "target_fasta_filepath": str(fasta),
        "target_pdb_filepath_by_chain": [str(chain_pdb)],
        "target_chain_and_residues": [f"B1-{len(sequence)}"],
        "target_interface_residues": {
            "hotspot": hotspots if hotspots is not None else ["B10", "B12", "B13"],
            "extended": ["B9", "B10", "B11", "B12", "B13"],
        },
        "binder_min_length": 60,
        "binder_max_length": 120,
    }, indent=4))
    return dataset


def write_genie3_overlays(root: Path) -> dict[str, Path]:
    """The three JAX overlays, as preflight looks for them."""
    overlays = root / "overlays"
    (overlays / "jax" / "jax_plugins").mkdir(parents=True, exist_ok=True)
    (overlays / "cudnn").mkdir(parents=True, exist_ok=True)
    (overlays / "cudnn" / "libcudnn.so.9").write_bytes(b"fixture")
    (overlays / "nvcc" / "bin").mkdir(parents=True, exist_ok=True)
    (overlays / "nvcc" / "bin" / "ptxas").write_bytes(b"fixture")
    return {
        "jax_plugin_overlay": overlays / "jax",
        "cudnn_overlay": overlays / "cudnn",
        "cuda_nvcc_overlay": overlays / "nvcc",
    }


def genie3_experiment(dataset: Path, *, key: str = GENIE3_SELECTION, **sections) -> dict:
    """A template with none of the three keys the harness writes per task."""
    experiment = {
        "experiment": {"name": "test-genie3"},
        "paths": {"dataset": str(dataset)},
        "generation": {
            "dataset": {"source": "target", "selections": key, "cond_strategy": "extended"},
            "sampler": {"sampler": {"direction_scale": 0.0}},
        },
        "evaluation": {
            "version": "binder",
            "inverse_folding": {"model_name": "proteinmpnn", "num_seq": 1},
            "folding": {"model_name": "colabfold", "mode": "template",
                        "backend": "subprocess", "num_models": 5, "num_recycles": 20},
        },
        "runtime": {"num_devices": 1},
    }
    for section, values in sections.items():
        experiment[section] = values
    return experiment


GENIE3_DRIVER_PATH = (
    Path(__file__).resolve().parents[1] / "drivers" / "genie3" / "run_genie3.py"
)


def write_genie3_configs(root: Path, *, experiment: dict | None = None,
                         **overrides) -> tuple[Path, Path]:
    """A general + Genie 3 pair. Genie 3 needs a problem set, not a FASTA."""
    fasta = root / "target.fasta"
    fasta.write_text(f">target\n{GENIE3_TARGET}\n")
    container = root / "genie3.sif"
    container.write_bytes(b"fixture")
    driver = root / "run_genie3.py"
    driver.write_text(GENIE3_DRIVER_PATH.read_text())

    # A caller passing its own experiment has already built the problem set it
    # points at, and rebuilding the default one here would overwrite it.
    if experiment is None:
        experiment = genie3_experiment(write_genie3_problemset(root))
    overlays = write_genie3_overlays(root)
    template = root / "experiment.yaml"
    template.write_text(yaml.safe_dump(experiment, sort_keys=False))

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    genie3 = {
        "schema_version": 1,
        "name": "genie3-test",
        "tool": "genie3",
        "experiment": {"template": str(template)},
        "driver": {"script": str(driver)},
        "sampling": {"jobs": 1, "backbones_per_job": 4, "seed_base": 100},
        "runtime": {
            "container": str(container),
            "node_tmp_root": str(root / "nodetmp"),
            **{name: str(path) for name, path in overlays.items()},
        },
        "resources": {"gpus": 1, "cpus": 16, "memory_gb": 96, "walltime": "08:00:00"},
    }
    for section, values in overrides.items():
        genie3[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "genie3.yaml"
    model_path.write_text(yaml.safe_dump(genie3, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def genie3_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_genie3_configs(tmp_path)


def write_genie3_task(
    run_dir: Path,
    task_id: int,
    *,
    designs: int | None = None,
    successes: tuple[str, ...] = (),
    backbones: int | None = None,
    results: str | None = None,
    status: dict | None = None,
    reducer: bool = True,
    selection: str = GENIE3_SELECTION,
) -> None:
    """Lay out one Genie 3 task from the committed real-output fixture.

    `designs` keeps that many design groups (five rows each); `successes` names
    the designs the v0 reducer called successes, using real rows for those too.

    `reducer=True` writes `success_info.csv` whether or not anything passed,
    which is what Genie 3 does -- the benchmark's zero-hit run left a header
    and no rows. `reducer=False` leaves the file out, which is what an
    evaluation that never reached the reduce step leaves behind.
    """
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    output = task_dir / selection
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "pdbs").mkdir(exist_ok=True)
    (output / "sequences").mkdir(exist_ok=True)

    lines = (GENIE3_FIXTURE / "info.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    if designs is not None:
        names = list(dict.fromkeys(line.split(",")[0] for line in body))[:designs]
        body = [line for line in body if line.split(",")[0] in names]
    text = results if results is not None else "\n".join([header, *body]) + "\n"
    (output / "results" / "info.csv").write_text(text)

    if reducer:
        (output / "results" / "v0_success").mkdir(exist_ok=True)
        winners = [line for line in body if line.split(",")[0] in successes]
        (output / "results" / "v0_success" / "success_info.csv").write_text(
            "\n".join([header, *winners]) + "\n"
        )

    for line in body:
        fields = line.split(",")
        name, domain = fields[0], fields[1]
        (output / "pdbs" / f"{domain}.pdb").write_text("ATOM\n")
        (output / "sequences" / f"{domain}.fasta").write_text(f">{domain}\nACDEF\n")
        structures = output / "structures" / name
        structures.mkdir(parents=True, exist_ok=True)
        (structures / Path(fields[3]).name).write_text("ATOM\n")
    for index in range(backbones or 0):
        (output / "pdbs" / f"{selection}_extra_{index}.pdb").write_text("ATOM\n")

    (task_dir / "experiment.yaml").write_text("experiment:\n  name: test-genie3\n")
    if status is not None:
        # Genie 3 writes no status of its own, so this is the shape the harness
        # writes around it -- notably with no count of its own.
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-26T19:00:00+00:00",
            "finished_at": "2026-08-26T22:31:00+00:00",
            "exit_code": 0,
            "error": None,
            "written_by": "harness",
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("genie3 log\n")


@pytest.fixture
def genie3_driver():
    """Import the Genie 3 driver, which needs only stdlib and PyYAML."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_genie3", GENIE3_DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PXDESIGN_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pxdesign"
# The fixture rows are unmodified output of the 2026-08-26 benchmark run, whose
# spec was keyed `dio3_cut`. task_name names the output tree, so the test
# configs use it too and the fixture never has to be edited.
PXDESIGN_TASK_NAME = "dio3_cut"


def write_pxdesign_msa(root: Path, chain: str = "A") -> Path:
    """A precomputed MSA directory, with both alignments PXDesign requires."""
    directory = root / "msa" / chain
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("non_pairing.a3m", "pairing.a3m"):
        (directory / name).write_text(">target\nACDEFG\n")
    return directory


def pxdesign_spec(cif: Path, msa: Path, *, task_name: str = PXDESIGN_TASK_NAME,
                  binder_length: int = 80, hotspots: list[int] | None = None) -> dict:
    chain: dict = {"msa": str(msa)}
    if hotspots is not None:
        chain["hotspots"] = hotspots
    return {
        "task_name": task_name,
        "target": {"file": str(cif), "chains": {"A": chain}},
        "binder_length": binder_length,
    }


def write_pxdesign_configs(root: Path, *, spec: dict | None = None,
                           **overrides) -> tuple[Path, Path]:
    """A general + PXDesign pair. PXDesign needs geometry and a precomputed MSA."""
    fasta = root / "target.fasta"
    cif = root / "target.cif"
    container = root / "pxdesign.sif"
    fasta.write_text(">target\nACDEFG\n")
    cif.write_text("data_target\n#\n")
    container.write_bytes(b"fixture")

    if spec is None:
        spec = pxdesign_spec(cif, write_pxdesign_msa(root))
    template = root / "pxdesign_input.yaml"
    template.write_text(yaml.safe_dump(spec, sort_keys=False))

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "chain_id": "A",
            "hotspots": [],
            "structure_cif": str(cif),
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    pxdesign = {
        "schema_version": 1,
        "name": "pxdesign-test",
        "tool": "pxdesign",
        "spec": {"template": str(template)},
        "sampling": {"jobs": 1, "designs_per_job": 4, "diffusion_steps": 400,
                     "seed_base": 100, "preset": "extended"},
        "runtime": {"container": str(container), "node_tmp_root": str(root / "nodetmp")},
        "resources": {"gpus": 1, "cpus": 16, "memory_gb": 96, "walltime": "10:00:00"},
    }
    for section, values in overrides.items():
        pxdesign[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "pxdesign.yaml"
    model_path.write_text(yaml.safe_dump(pxdesign, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def pxdesign_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_pxdesign_configs(tmp_path)


def write_pxdesign_task(
    run_dir: Path,
    task_id: int,
    *,
    designs: int | None = None,
    summary: str | None = None,
    status: dict | None = None,
    task_name: str = PXDESIGN_TASK_NAME,
) -> None:
    """Lay out one PXDesign task from the committed real-output fixture."""
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    outputs = task_dir / "design_outputs" / task_name
    outputs.mkdir(parents=True, exist_ok=True)

    lines = (PXDESIGN_FIXTURE / "summary.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    if designs is not None:
        body = body[:designs]
    text = summary if summary is not None else "\n".join([header, *body]) + "\n"
    (outputs / "summary.csv").write_text(text)
    (outputs / "task_info.json").write_text(
        json.dumps({"mode": "Extended", "protenix": "Protenix"})
    )
    (task_dir / "config.yaml").write_text("dump_dir: fixture\n")

    columns = header.split(",")
    chosen = columns.index("chosen_struct_path")
    for line in body:
        structure = outputs / line.split(",")[chosen]
        structure.parent.mkdir(parents=True, exist_ok=True)
        structure.write_text("data_design\n#\n")

    if status is not None:
        # PXDesign writes no status of its own, so this is the shape the
        # harness writes around it -- notably with no count of its own.
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-26T19:43:00+00:00",
            "finished_at": "2026-08-26T19:58:00+00:00",
            "exit_code": 0,
            "error": None,
            "written_by": "harness",
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("pxdesign log\n")


PROTEIN_HUNTER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "protein_hunter"


def write_protein_hunter_configs(root: Path, **overrides) -> tuple[Path, Path]:
    """A general + Protein-Hunter pair. This tool needs a sequence, not geometry."""
    fasta = root / "target.fasta"
    msa = root / "target.a3m"
    container = root / "protein_hunter.sif"
    driver = root / "run_boltz_design.py"

    fasta.write_text(">target\nACDEFG\n")
    msa.write_text(">target\nACDEFG\n>hit_1\nACDEFG\n")
    container.write_bytes(b"fixture")
    driver.write_text(
        (Path(__file__).resolve().parents[1]
         / "drivers" / "protein_hunter" / "run_boltz_design.py").read_text()
    )

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "msa": str(msa),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    protein_hunter = {
        "schema_version": 1,
        "name": "protein-hunter-test",
        "tool": "protein_hunter",
        "driver": {"script": str(driver)},
        "sampling": {
            "jobs": 1,
            "trajectories_per_job": 3,
            "cycles": 5,
            "min_binder_length": 65,
            "max_binder_length": 120,
            "percent_x": 90,
            "omit_aa": "C",
            "temperature": 0.1,
            "diffuse_steps": 200,
            "recycling_steps": 3,
        },
        "filters": {"high_iptm_threshold": 0.7, "high_plddt_threshold": 0.7},
        "contacts": {"cutoff": 15.0, "filter": True, "max_retries": 6},
        "msa": {"mode": "mmseqs", "max_seqs": 512},
        "runtime": {"container": str(container), "node_tmp_root": str(root / "nodetmp")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 64, "walltime": "08:00:00"},
    }
    for section, values in overrides.items():
        protein_hunter[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "protein_hunter.yaml"
    model_path.write_text(yaml.safe_dump(protein_hunter, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def protein_hunter_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_protein_hunter_configs(tmp_path)


def write_protein_hunter_task(
    run_dir: Path,
    task_id: int,
    *,
    trajectories: int | None = None,
    summary: str | None = None,
    thresholds: bool = True,
    status: dict | None = None,
) -> None:
    """Lay out one Protein-Hunter task from the committed real-output fixture.

    `thresholds=True` writes `summary_high_iptm.csv` whether or not anything
    cleared them; `thresholds=False` leaves it out, which is what a run that
    stopped before the threshold pass leaves behind.
    """
    task_dir = run_dir / "tasks" / f"{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    lines = (PROTEIN_HUNTER_FIXTURE / "summary_all_runs.csv").read_text().splitlines()
    header, body = lines[0], lines[1:]
    if trajectories is not None:
        body = body[:trajectories]
    kept = {line.split(",")[0] for line in body}
    text = summary if summary is not None else "\n".join([header, *body]) + "\n"
    (task_dir / "summary_all_runs.csv").write_text(text)

    hi_lines = (PROTEIN_HUNTER_FIXTURE / "summary_high_iptm.csv").read_text().splitlines()
    hi_header, hi_body = hi_lines[0], [
        line for line in hi_lines[1:] if line.split(",")[0] in kept
    ]
    if thresholds:
        (task_dir / "summary_high_iptm.csv").write_text(
            "\n".join([hi_header, *hi_body]) + "\n"
        )
        columns = hi_header.split(",")
        pdb, spec = columns.index("pdb_filename"), columns.index("yaml_filename")
        (task_dir / "high_iptm_pdb").mkdir(exist_ok=True)
        (task_dir / "high_iptm_yaml").mkdir(exist_ok=True)
        for line in hi_body:
            fields = line.split(",")
            (task_dir / "high_iptm_pdb" / fields[pdb]).write_text("ATOM\n")
            (task_dir / "high_iptm_yaml" / fields[spec]).write_text("sequences: []\n")

    if status is not None:
        # Protein-Hunter writes no status of its own, so this is the shape the
        # harness writes around it.
        (task_dir / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-26T19:00:00+00:00",
            "finished_at": "2026-08-26T22:00:00+00:00",
            "exit_code": 0,
            "error": None,
            "written_by": "harness",
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("protein-hunter log\n")


@pytest.fixture
def protein_hunter_driver():
    """Import the Protein-Hunter driver, which needs only the stdlib."""
    import importlib.util

    path = (Path(__file__).resolve().parents[1]
            / "drivers" / "protein_hunter" / "run_boltz_design.py")
    spec = importlib.util.spec_from_file_location("run_boltz_design", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Proteina-Complexa -------------------------------------------------------

PROTEINA_TARGET = "".join(["ACDEFGHIKLMNPQRSTVWY"] * 10) + "A"  # 201 aa, chain A


def write_proteina_target_pdb(path: Path, sequence: str = PROTEINA_TARGET) -> Path:
    """A CA-only PDB of the target, numbered 1..len from residue 1 of chain A.

    Proteina-Complexa resolves its epitope against CA atoms keyed
    `f"{chain_id}{res_id}"`, so a fixture that omits them cannot exercise the
    check that matters.
    """
    lines = []
    for index, _ in enumerate(sequence, start=1):
        lines.append(
            f"ATOM  {index:>5d}  CA  ALA A{index:>4d}    "
            f"{index:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def proteina_registry(task_name: str = "test_target", *, hotspots: list[str] | None = None,
                      **entry_overrides) -> dict:
    entry = {
        "source": "bindocracy",
        "target_filename": task_name,
        "target_path": "/mnt/bindocracy_target.pdb",
        "target_input": "A1-201",
        "hotspot_residues": hotspots if hotspots is not None else [],
        "binder_length": [70, 110],
        "pdb_id": None,
    }
    entry.update(entry_overrides)
    return {"target_dict_cfg": {task_name: entry}}


def write_proteina_complexa_configs(
    root: Path, *, registry: dict | None = None, hotspots: list[str] | None = None,
    **overrides,
) -> tuple[Path, Path]:
    """A general + Proteina-Complexa pair, with the target PDB it reads."""
    fasta = root / "target.fasta"
    fasta.write_text(f">target\n{PROTEINA_TARGET}\n")
    pdb = write_proteina_target_pdb(root / "target.pdb")
    container = root / "proteina_complexa.sif"
    container.write_bytes(b"fixture")

    document = registry if registry is not None else proteina_registry(hotspots=hotspots)
    template = root / "target_registry.yaml"
    template.write_text(yaml.safe_dump(document, sort_keys=False))

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "structure_pdb": str(pdb),
            "chain_id": "A",
            "hotspots": hotspots or [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    model = {
        "schema_version": 1,
        "name": "proteina-complexa-test",
        "tool": "proteina_complexa",
        "registry": {"template": str(template), "task_name": "test_target"},
        "sampling": {
            "jobs": 1,
            "samples_per_job": 4,
            "keep_per_job": 4,
            "seed_base": 5,
        },
        "runtime": {
            "container": str(container),
            "node_tmp_root": str(root / "nodetmp"),
        },
        "resources": {"gpus": 1, "cpus": 16, "memory_gb": 160, "walltime": "10:00:00"},
    }
    for section, values in overrides.items():
        model[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "proteina_complexa.yaml"
    model_path.write_text(yaml.safe_dump(model, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def proteina_complexa_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_proteina_complexa_configs(tmp_path)


# --- FreeBindCraft -----------------------------------------------------------

FREEBINDCRAFT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "freebindcraft"
# The fixture rows are output of the 2026-08-26 benchmark run, sliced to two
# trajectories. BindCraft stamps the stem of each settings document onto every
# row it writes, so the test documents carry the run's own names and the
# fixture never has to be edited. The one edit is `Rank`: a slice of a ranked
# table is not a ranked table, and BindCraft ranks its whole accepted pool
# 1..N, so the two surviving rows were renumbered 1 and 2.
FREEBINDCRAFT_TARGET_NAME = "dio3_cut_target"
FREEBINDCRAFT_FILTERS_NAME = "relaxed_filters"
FREEBINDCRAFT_ADVANCED_NAME = "dio3_cut_advanced_max40"
FREEBINDCRAFT_BINDER = "dio3_cut"
# 201 aa, chain A, numbered 1..201 -- the campaign target the benchmark ran on.
FREEBINDCRAFT_TARGET = "".join(["ACDEFGHIKLMNPQRSTVWY"] * 10) + "A"

FREEBINDCRAFT_DRIVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "drivers" / "freebindcraft" / "run_freebindcraft.py"
)


def freebindcraft_target(pdb: Path, **overrides) -> dict:
    """A target document with none of the three keys the driver writes."""
    document = {
        "binder_name": FREEBINDCRAFT_BINDER,
        "starting_pdb": str(pdb),
        "chains": "A",
        "lengths": [65, 150],
    }
    document.update(overrides)
    return document


def freebindcraft_advanced(**overrides) -> dict:
    """The keys preflight insists on, with `max_trajectories` unset as shipped."""
    document = {
        "acceptance_rate": 0.01,
        "af_params_dir": "",
        "backbone_noise": 0.0,
        "dalphaball_path": "",
        "design_algorithm": "4stage",
        "dssp_path": "",
        "enable_mpnn": True,
        "enable_rejection_check": True,
        "force_reject_AA": False,
        "greedy_iterations": 15,
        "greedy_percentage": 1,
        "hard_iterations": 5,
        "inter_contact_distance": 20.0,
        "inter_contact_number": 2,
        "intra_contact_distance": 14.0,
        "intra_contact_number": 2,
        "max_mpnn_sequences": 2,
        "max_trajectories": False,
        "model_path": "v_48_020",
        "mpnn_fix_interface": True,
        "mpnn_weights": "soluble",
        "num_recycles_design": 1,
        "num_recycles_validation": 3,
        "num_seqs": 20,
        "omit_AAs": "C",
        "optimise_beta": True,
        "optimise_beta_extra_soft": 0,
        "optimise_beta_extra_temp": 0,
        "optimise_beta_recycles_design": 3,
        "optimise_beta_recycles_valid": 3,
        "predict_bigbang": False,
        "predict_initial_guess": False,
        "random_helicity": False,
        "remove_binder_monomer": True,
        "remove_unrelaxed_complex": True,
        "remove_unrelaxed_trajectory": True,
        "rm_template_sc_design": False,
        "rm_template_sc_predict": False,
        "rm_template_seq_design": False,
        "rm_template_seq_predict": False,
        "sample_models": True,
        "sampling_temp": 0.1,
        "save_design_animations": True,
        "save_design_trajectory_plots": True,
        "save_mpnn_fasta": False,
        "save_trajectory_pickle": False,
        "soft_iterations": 75,
        "start_monitoring": 600,
        "temporary_iterations": 45,
        "use_i_ptm_loss": True,
        "use_multimer_design": True,
        "use_rg_loss": True,
        "use_termini_distance_loss": False,
        "weights_con_inter": 1.0,
        "weights_con_intra": 1.0,
        "weights_helicity": -0.3,
        "weights_iptm": 0.05,
        "weights_pae_inter": 0.1,
        "weights_pae_intra": 0.4,
        "weights_plddt": 0.1,
        "weights_rg": 0.3,
        "weights_termini_loss": 0.1,
        "zip_animations": True,
        "zip_plots": True,
    }
    document.update(overrides)
    return document


def freebindcraft_filters(**overrides) -> dict:
    """A filter set thresholding one metric a PyRosetta-free run cannot compute.

    `Average_dG` is one of the eight constants, so a run with this filter set
    has a non-empty `inert_filters` -- which is the point: every filter file
    the image ships is like this.
    """
    document = {
        "MPNN_score": {"threshold": None, "higher": False},
        "Average_pLDDT": {"threshold": 0.8, "higher": True},
        "1_pLDDT": {"threshold": 0.8, "higher": True},
        "Average_i_pTM": {"threshold": 0.5, "higher": True},
        "Average_dG": {"threshold": 0.0, "higher": False},
        "Average_InterfaceAAs": {"C": {"threshold": 0, "higher": False}},
    }
    document.update(overrides)
    return document


def write_freebindcraft_configs(
    root: Path,
    *,
    hotspots: list[str] | None = None,
    target: dict | None = None,
    advanced: dict | None = None,
    filters: dict | None = None,
    **overrides,
) -> tuple[Path, Path]:
    """A general + FreeBindCraft pair, with the PDB it hallucinates against."""
    fasta = root / "target.fasta"
    fasta.write_text(f">target\n{FREEBINDCRAFT_TARGET}\n")
    pdb = write_proteina_target_pdb(root / "target.pdb", FREEBINDCRAFT_TARGET)
    container = root / "freebindcraft.sif"
    container.write_bytes(b"fixture")
    driver = root / "run_freebindcraft.py"
    driver.write_text(FREEBINDCRAFT_DRIVER_PATH.read_text())

    documents = {
        f"{FREEBINDCRAFT_TARGET_NAME}.json": (
            target if target is not None else freebindcraft_target(pdb)
        ),
        f"{FREEBINDCRAFT_ADVANCED_NAME}.json": (
            advanced if advanced is not None else freebindcraft_advanced()
        ),
        f"{FREEBINDCRAFT_FILTERS_NAME}.json": (
            filters if filters is not None else freebindcraft_filters()
        ),
    }
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document, indent=2))

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "test-target",
            "sequence_fasta": str(fasta),
            "structure_pdb": str(pdb),
            "chain_id": "A",
            "hotspots": hotspots or [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    model = {
        "schema_version": 1,
        "name": "freebindcraft-test",
        "tool": "freebindcraft",
        "target": {"template": str(root / f"{FREEBINDCRAFT_TARGET_NAME}.json")},
        "filters": {"template": str(root / f"{FREEBINDCRAFT_FILTERS_NAME}.json")},
        "advanced": {"template": str(root / f"{FREEBINDCRAFT_ADVANCED_NAME}.json")},
        "driver": {"script": str(driver)},
        "sampling": {"jobs": 1, "designs_per_job": 2, "max_trajectories": 4},
        "runtime": {"container": str(container), "node_tmp_root": str(root / "nodetmp")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 64, "walltime": "04:00:00"},
    }
    for section, values in overrides.items():
        model[section].update(values)
    (root / "nodetmp").mkdir(exist_ok=True)

    model_path = root / "freebindcraft.yaml"
    model_path.write_text(yaml.safe_dump(model, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def freebindcraft_configs(tmp_path: Path) -> tuple[Path, Path]:
    return write_freebindcraft_configs(tmp_path)


def write_freebindcraft_task(
    run_dir: Path,
    task_id: int,
    *,
    designs: int | None = None,
    ranked: bool = True,
    clashing: int = 1,
    low_confidence: int = 1,
    scored: str | None = None,
    structures: bool = True,
    status: dict | None = None,
) -> None:
    """Lay out one FreeBindCraft task from the committed real-output fixture.

    `ranked=False` blanks the Rank column rather than dropping the rows, which
    is what a task that stopped on its trajectory budget actually writes: the
    table names every accepted design and ranks none of them, and
    `Accepted/Ranked/` stays empty.
    """
    design_path = run_dir / "tasks" / f"{task_id:04d}" / "bindcraft"
    design_path.mkdir(parents=True, exist_ok=True)

    def table(name: str, rows: int | None = None) -> list[str]:
        lines = (FREEBINDCRAFT_FIXTURE / name).read_text().splitlines()
        header, body = lines[0], lines[1:]
        if rows is not None:
            body = body[:rows]
        (design_path / name).write_text("\n".join([header, *body]) + "\n")
        return body

    scored_rows = table("mpnn_design_stats.csv", designs)
    if scored is not None:
        (design_path / "mpnn_design_stats.csv").write_text(scored)
        scored_rows = scored.splitlines()[1:]
    kept = {line.split(",")[0] for line in scored_rows}
    rejected_rows = table("rejected_mpnn_full_stats.csv")
    table("trajectory_stats.csv")
    table("failure_csv.csv")

    final_lines = (FREEBINDCRAFT_FIXTURE / "final_design_stats.csv").read_text().splitlines()
    final_body = [line for line in final_lines[1:] if line.split(",")[1] in kept]
    if not ranked:
        # What a task that stopped on its trajectory budget actually leaves:
        # one row per accepted design, appended as it was accepted, with the
        # Rank column still empty. BindCraft fills the ranks in only inside the
        # check that ends the loop on having enough designs, so a header-only
        # file is a shape it never writes.
        final_body = ["," + line.split(",", 1)[1] for line in final_body]
    (design_path / "final_design_stats.csv").write_text(
        "\n".join([final_lines[0], *final_body]) + "\n"
    )

    accepted = kept - {line.split(",")[0] for line in rejected_rows}
    if structures:
        for directory, names in (
            ("Accepted", accepted),
            ("Rejected", kept & {line.split(",")[0] for line in rejected_rows}),
        ):
            (design_path / directory).mkdir(exist_ok=True)
            for name in sorted(names):
                (design_path / directory / f"{name}_model1.pdb").write_text("ATOM\n")
    (design_path / "Accepted" / "Ranked").mkdir(parents=True, exist_ok=True)

    # `max_trajectories` counts Relaxed alone; the other two are attempts that
    # ended before they could be scored.
    for directory, count in (
        ("Relaxed", len((FREEBINDCRAFT_FIXTURE / "trajectory_stats.csv")
                        .read_text().splitlines()) - 1),
        ("Clashing", clashing),
        ("LowConfidence", low_confidence),
    ):
        path = design_path / "Trajectory" / directory
        path.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (path / f"{FREEBINDCRAFT_BINDER}_traj_{directory}_{index}.pdb").write_text("ATOM\n")

    if status is not None:
        # FreeBindCraft writes no status of its own, so this is the shape the
        # harness writes around it.
        (run_dir / "tasks" / f"{task_id:04d}" / "status.json").write_text(json.dumps({
            "task_id": task_id,
            "status": "succeeded",
            "started_at": "2026-08-26T23:38:00+00:00",
            "finished_at": "2026-08-27T07:02:00+00:00",
            "exit_code": 0,
            "error": None,
            "written_by": "harness",
        } | status))
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"task-{task_id:04d}.log").write_text("freebindcraft log\n")


@pytest.fixture
def freebindcraft_driver():
    """Import the FreeBindCraft driver, which needs only the stdlib."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_freebindcraft", FREEBINDCRAFT_DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Chai-1 -----------------------------------------------------------------

CHAI1_DRIVER_PATH = (
    Path(__file__).resolve().parents[1] / "drivers" / "chai1" / "score_chai1.py"
)
# The campaign's real DIO3 target, so the fixture's alignment filename is the
# one the container actually computes. See tests/test_chai1.py.
CHAI1_TARGET = (
    "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQR"
    "LVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERL"
    "YVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
)


def _write_chai1_design_set(root: Path, sequences: list[str]) -> Path:
    """Freeze a design set the way the CLI would, returning its manifest path."""
    from datetime import UTC, datetime

    from bindocracy.runs.designset import build_design_set, write_design_set
    from bindocracy.store.query import DesignQuery, DesignRow

    rows = [
        DesignRow(
            design_id=f"design-{index:03d}",
            run_id="run-fixture",
            tool="mosaic",
            run_name="fixture",
            native_id=f"n{index}",
            candidate_type="binder",
            sequence=sequence,
            length=len(sequence),
            metadata={},
        )
        for index, sequence in enumerate(sequences)
    ]
    design_set = build_design_set(
        rows,
        database="fixture.duckdb",
        query=DesignQuery(),
        created_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    _, manifest = write_design_set(design_set, root / "design_sets")
    return manifest


@pytest.fixture
def write_design_set(tmp_path: Path):
    """Freeze an arbitrary set of sequences; returns the manifest path."""

    def build(sequences: list[str]) -> Path:
        return _write_chai1_design_set(tmp_path / "extra", sequences)

    return build


@pytest.fixture
def chai1_configs(tmp_path: Path):
    """A validated (GeneralConfig, Chai1Config) pair with every path present."""
    from bindocracy.config.models import GeneralConfig
    from bindocracy.tools.chai1.config import Chai1Config
    from bindocracy.tools.chai1.preflight import expected_pqt_basename

    target_fasta = tmp_path / "target.fasta"
    target_fasta.write_text(f">dio3\n{CHAI1_TARGET}\n")

    msa_dir = tmp_path / "chai_msas"
    msa_dir.mkdir()
    (msa_dir / expected_pqt_basename(CHAI1_TARGET)).write_bytes(b"parquet-fixture")

    container = tmp_path / "chai1.sif"
    container.write_bytes(b"fixture")
    driver = tmp_path / "score_chai1.py"
    driver.write_text("# fixture\n")
    scratch = tmp_path / "scratch" / "chai1"
    scratch.parent.mkdir(parents=True, exist_ok=True)

    manifest = _write_chai1_design_set(tmp_path, ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKM"])

    general = GeneralConfig.model_validate({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "dio3-cut",
            "sequence_fasta": str(target_fasta),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    })
    model = Chai1Config.model_validate({
        "schema_version": 1,
        "name": "chai1-test",
        "tool": "chai1",
        "design_set": str(manifest),
        "driver_script": str(driver),
        "protocol": {
            "num_trunk_recycles": 3,
            "num_diffn_timesteps": 200,
            "num_diffn_samples": 5,
            "num_trunk_samples": 1,
            "recycle_msa_subsample": 0,
            "seed": 0,
        },
        "sharding": {"jobs": 1},
        "runtime": {
            "container": str(container),
            "msa_directory": str(msa_dir),
            "scratch": str(scratch),
        },
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 96, "walltime": "12:00:00"},
    })
    return general, model


@pytest.fixture
def chai1_loaded(chai1_configs):
    """What `plugin.load()` would hand `tool_plan`, without the YAML round trip."""
    from bindocracy.config.load import LoadedConfigs
    from bindocracy.tools.chai1.preflight import preflight_chai1

    general, model = chai1_configs
    return LoadedConfigs(
        general=general,
        model=model,
        preflight=preflight_chai1(general, model),
        general_path=Path("general.yaml"),
        model_path=Path("chai1.yaml"),
    )


def write_chai1_configs(root: Path, **overrides) -> tuple[Path, Path]:
    """Write a valid general + chai1 YAML pair and everything they reference."""
    from bindocracy.tools.chai1.preflight import expected_pqt_basename

    target_fasta = root / "target.fasta"
    target_fasta.write_text(f">dio3\n{CHAI1_TARGET}\n")
    msa_dir = root / "chai_msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    (msa_dir / expected_pqt_basename(CHAI1_TARGET)).write_bytes(b"parquet-fixture")
    container = root / "chai1.sif"
    container.write_bytes(b"fixture")
    # The real driver, so the argv the connector builds is parsed by the parser
    # that will actually receive it.
    driver = root / "score_chai1.py"
    driver.write_text(CHAI1_DRIVER_PATH.read_text())
    (root / "scratch").mkdir(parents=True, exist_ok=True)

    manifest = _write_chai1_design_set(
        root, ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKM", "WYVTSRQPNM"]
    )

    general_path = root / "general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {
            "name": "dio3-cut",
            "sequence_fasta": str(target_fasta),
            "chain_id": "A",
            "hotspots": [],
        },
        "cluster": {
            "executor": "slurm",
            "account": "test-account",
            "default_partition": "test-gpu",
        },
    }, sort_keys=False))

    chai1 = {
        "schema_version": 1,
        "name": "chai1-test",
        "tool": "chai1",
        "design_set": str(manifest),
        "driver_script": str(driver),
        "protocol": {
            "num_trunk_recycles": 3,
            "num_diffn_timesteps": 200,
            "num_diffn_samples": 5,
            "num_trunk_samples": 1,
            "recycle_msa_subsample": 0,
            "seed": 0,
        },
        "readers": {"complex": True},
        "sharding": {"jobs": 2},
        "runtime": {
            "container": str(container),
            "msa_directory": str(msa_dir),
            "scratch": str(root / "scratch" / "chai1"),
        },
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 96, "walltime": "12:00:00"},
    }
    for section, values in overrides.items():
        chai1[section].update(values)

    model_path = root / "chai1.yaml"
    model_path.write_text(yaml.safe_dump(chai1, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def chai1_config_files(tmp_path: Path) -> tuple[Path, Path]:
    return write_chai1_configs(tmp_path)


# --- AlphaFold 3 -------------------------------------------------------------

AF3_DRIVER_PATH = (
    Path(__file__).resolve().parents[1] / "drivers" / "af3" / "score_af3.py"
)


def write_af3_configs(root: Path, **overrides) -> tuple[Path, Path]:
    """A general + af3 YAML pair with every referenced path present."""
    target_fasta = root / "af3_target.fasta"
    target_fasta.write_text(f">dio3\n{CHAI1_TARGET}\n")
    msa = root / "af3_target.a3m"
    msa.write_text(f">dio3\n{CHAI1_TARGET}\n>hom\n{CHAI1_TARGET}\n")
    container = root / "af3.sif"
    container.write_bytes(b"fixture")
    models = root / "af3_models"
    models.mkdir(parents=True, exist_ok=True)
    # Not the real 1 GB file: preflight checks presence, not contents.
    (models / "af3.bin.zst").write_bytes(b"fixture-weights")
    driver = root / "score_af3.py"
    driver.write_text(AF3_DRIVER_PATH.read_text())
    (root / "af3_work").mkdir(parents=True, exist_ok=True)

    manifest = _write_chai1_design_set(
        root / "af3_set", ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKM"]
    )

    general_path = root / "af3_general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {"name": "dio3-cut", "sequence_fasta": str(target_fasta),
                   "msa": str(msa), "chain_id": "A", "hotspots": []},
        "cluster": {"executor": "slurm", "account": "a", "default_partition": "p"},
    }, sort_keys=False))

    af3 = {
        "schema_version": 1, "name": "af3-test", "tool": "af3",
        "design_set": str(manifest), "driver_script": str(driver),
        "protocol": {"num_trunk_recycles": 3, "num_diffn_timesteps": 200,
                     "num_diffn_samples": 5, "seed": 0, "use_target_msa": True},
        "readers": {"complex": True},
        "sharding": {"jobs": 1},
        "runtime": {"container": str(container), "model_dir": str(models),
                    "work_root": str(root / "af3_work" / "run")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 128, "walltime": "12:00:00"},
    }
    for section, values in overrides.items():
        af3[section].update(values)
    model_path = root / "af3.yaml"
    model_path.write_text(yaml.safe_dump(af3, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def af3_config_files(tmp_path: Path) -> tuple[Path, Path]:
    return write_af3_configs(tmp_path)


@pytest.fixture
def af3_configs(af3_config_files):
    from bindocracy.config.load import load_yaml
    from bindocracy.config.models import GeneralConfig
    from bindocracy.tools.af3.config import AF3Config

    general_path, model_path = af3_config_files
    return load_yaml(general_path, GeneralConfig), load_yaml(model_path, AF3Config)


@pytest.fixture
def af3_loaded(af3_configs):
    from bindocracy.config.load import LoadedConfigs
    from bindocracy.tools.af3.preflight import preflight_af3

    general, model = af3_configs
    return LoadedConfigs(
        general=general, model=model, preflight=preflight_af3(general, model),
        general_path=Path("general.yaml"), model_path=Path("af3.yaml"),
    )


# --- upstream OpenFold3 -------------------------------------------------------

OF3_DRIVER_PATH = (
    Path(__file__).resolve().parents[1] / "drivers" / "of3_upstream" / "score_of3.py"
)


def write_of3_configs(root: Path, **overrides) -> tuple[Path, Path]:
    target_fasta = root / "of3_target.fasta"
    target_fasta.write_text(f">dio3\n{CHAI1_TARGET}\n")
    msa = root / "of3_target.a3m"
    msa.write_text(f">dio3\n{CHAI1_TARGET}\n>hom\n{CHAI1_TARGET}\n")
    container = root / "openfold3.sif"
    container.write_bytes(b"fixture")
    checkpoint = root / "of3-p2-155k.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    driver = root / "score_of3.py"
    driver.write_text(OF3_DRIVER_PATH.read_text())
    (root / "of3_work").mkdir(parents=True, exist_ok=True)
    manifest = _write_chai1_design_set(
        root / "of3_set", ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKM"]
    )

    general_path = root / "of3_general.yaml"
    general_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "campaign": {"name": "test-campaign"},
        "target": {"name": "dio3-cut", "sequence_fasta": str(target_fasta),
                   "msa": str(msa), "chain_id": "A", "hotspots": []},
        "cluster": {"executor": "slurm", "account": "a", "default_partition": "p"},
    }, sort_keys=False))

    of3 = {
        "schema_version": 1, "name": "of3-upstream-test", "tool": "of3_upstream",
        "design_set": str(manifest), "driver_script": str(driver),
        "protocol": {"num_diffusion_samples": 1, "seed": 0,
                     "use_target_msa": True, "use_msa_server": False},
        "readers": {"complex": True},
        "sharding": {"jobs": 1},
        "runtime": {"container": str(container), "checkpoint": str(checkpoint),
                    "work_root": str(root / "of3_work" / "run")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 128, "walltime": "12:00:00"},
    }
    for section, values in overrides.items():
        of3[section].update(values)
    model_path = root / "of3_upstream.yaml"
    model_path.write_text(yaml.safe_dump(of3, sort_keys=False))
    return general_path, model_path


@pytest.fixture
def of3_config_files(tmp_path: Path) -> tuple[Path, Path]:
    return write_of3_configs(tmp_path)


@pytest.fixture
def of3_configs(of3_config_files):
    from bindocracy.config.load import load_yaml
    from bindocracy.config.models import GeneralConfig
    from bindocracy.tools.of3_upstream.config import OF3UpstreamConfig

    general_path, model_path = of3_config_files
    return load_yaml(general_path, GeneralConfig), load_yaml(model_path, OF3UpstreamConfig)


@pytest.fixture
def of3_loaded(of3_configs):
    from bindocracy.config.load import LoadedConfigs
    from bindocracy.tools.of3_upstream.preflight import preflight_of3_upstream

    general, model = of3_configs
    return LoadedConfigs(
        general=general, model=model,
        preflight=preflight_of3_upstream(general, model),
        general_path=Path("general.yaml"), model_path=Path("of3.yaml"),
    )


# --- the mosaic scorer, shared by several test modules ----------------------

@pytest.fixture
def scorer_configs(tmp_path):
    """A validated (GeneralConfig, ScorerConfig) pair with real paths."""
    from bindocracy.config.models import GeneralConfig
    from bindocracy.tools.scorer.config import ScorerConfig

    target = tmp_path / "target.fasta"
    target.write_text(">t\nACDEFGHIKLMNPQRSTVWY\n")
    msa = tmp_path / "t.a3m"
    msa.write_text(">t\nACDEFGHIKLMNPQRSTVWY\n")
    container = tmp_path / "mosaic.sif"; container.write_bytes(b"x")
    wrapper = tmp_path / "exec.sh"; wrapper.write_text("#!/bin/sh\n")
    weights = tmp_path / "weights"; weights.mkdir()
    driver = tmp_path / "d.py"; driver.write_text("# fixture\n")
    (tmp_path / "scratch").mkdir()

    general = GeneralConfig.model_validate({
        "schema_version": 1,
        "campaign": {"name": "c"},
        "target": {"name": "t", "sequence_fasta": str(target),
                   "msa": str(msa), "chain_id": "A", "hotspots": []},
        "cluster": {"executor": "slurm", "account": "a", "default_partition": "p"},
    })
    model = ScorerConfig.model_validate({
        "schema_version": 1, "name": "s", "tool": "scorer",
        "design_set": str(tmp_path / "set.json"),
        "model": {"name": "boltz2", "recycling_steps": 3, "sampling_steps": 25,
                  "num_samples": 1, "use_target_msa": True},
        "readers": {"complex": True},
        "sharding": {"jobs": 1},
        "driver": {"script": str(driver)},
        "runtime": {"container": str(container), "weights": str(weights),
                    "exec_wrapper": str(wrapper),
                    "scratch": str(tmp_path / "scratch" / "m")},
        "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32, "walltime": "2:00:00"},
    })
    return general, model
