"""Build the command for one Proteina-Complexa task.

There is no driver here. Proteina-Complexa has a command-line override for
every value the harness owns -- the seed, the sample count, the search width,
the filter budget -- so a task is one `singularity run` and nothing else.

The one value with no override is where the output goes: `./inference/` and
`./evaluation_results/` are built relative to the process's working directory.
`--pwd <task dir>` is what places them, and it is also what keeps two tasks
from writing into each other.

There *is* a `++root_path` override, and it must not be used. `generate.py`
only calls `setup()` when `root_path` is None, and `setup()` is the sole caller
of `L.seed_everything(cfg.seed)` and the sole place `cfg.seed + job_id` is
applied. Setting it silently produces an unseeded run whose manifest records a
seed that did nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.proteina_complexa.config import ProteinaComplexaConfig

# Inside the image. The pipeline config composes the whole Hydra tree, which
# ships with the image and is covered by the container digest.
CONTAINER_PIPELINE = "/opt/proteina-complexa/configs/search_binder_local_pipeline.yaml"
CONTAINER_REGISTRY = "/opt/proteina-complexa/configs/targets/targets_dict.yaml"
CONTAINER_TARGET = "/mnt/bindocracy_target.pdb"

# Hydra names every output directory after the config it ran, so the stem is
# part of the output contract rather than a detail of the command.
CONFIG_STEM = "search_binder_local_pipeline"

# `++run_name` is not a campaign concept -- the harness names runs. It exists
# here only because it is part of the output directory name, so it is fixed and
# the directory is predictable at planning time. Tasks are already isolated
# from each other by `--pwd`.
RUN_NAME = "bindocracy"

# `binder_results_<config>_<job_id>.csv`. The job id is 0 because every task
# runs with `gen_njobs=1`: parallelism here is one Slurm job per task, not one
# Hydra job per GPU.
JOB_ID = 0

GENERATION_DIR = "inference"
EVALUATION_DIR = "evaluation_results"


def output_stem(task_name: str) -> str:
    """The directory name Hydra builds for this run, under both output roots."""
    return f"{CONFIG_STEM}_{task_name}_{RUN_NAME}"


def designs_file(task_name: str) -> str:
    """The evaluated table, relative to a task directory.

    This is the table with sequences in it. The generation-stage rewards table
    holds every candidate but encodes its sequence as integer `aatype`.
    """
    return f"{EVALUATION_DIR}/{output_stem(task_name)}/binder_results_{CONFIG_STEM}_{JOB_ID}.csv"


def rewards_file(task_name: str) -> str:
    """Every candidate the run generated, before the reward filter kept any."""
    return f"{GENERATION_DIR}/{output_stem(task_name)}/all_rewards_{CONFIG_STEM}.csv"


def successes_file(task_name: str) -> str:
    """The designs that cleared the evaluation thresholds, if the stage ran."""
    return f"{EVALUATION_DIR}/{output_stem(task_name)}/all_successes_protein_binder_self.csv"


def criteria_file(task_name: str) -> str:
    """The thresholds those successes were judged against."""
    return f"{EVALUATION_DIR}/{output_stem(task_name)}/success_criteria_protein_binder.json"


def proteina_complexa_launch_spec(
    general: GeneralConfig,
    model: ProteinaComplexaConfig,
    manifest: RunManifest,
    task_id: int,
) -> LaunchSpec:
    """One task: the archived registry, bound over the image's own."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived copy, not the authored one: the working tree may have moved
    # on since this run was planned.
    registry = run_dir / manifest.provenance["registry"].path
    target_pdb = general.target.structure_pdb

    node_tmp = _node_tmp(model, manifest, task_id)
    sampling = model.sampling

    argv = [
        "singularity", "run", "--cleanenv", "--nv",
        # The working directory IS the output directory; see the module note.
        "--pwd", str(task_dir),
        "--bind", f"{target_pdb}:{CONTAINER_TARGET}:ro",
        # Bound over the image's registry rather than added to it, so the run
        # can only see the campaign's target.
        "--bind", f"{registry}:{CONTAINER_REGISTRY}:ro",
        str(model.runtime.container),
        "design", CONTAINER_PIPELINE,
        f"++run_name={RUN_NAME}",
        f"++generation.task_name={model.registry.task_name}",
        # Every task samples from its own seed. Sharing one would draw the same
        # backbones twice and the run would return duplicates without any of
        # its counts changing.
        f"++seed={sampling.seed_base + task_id}",
        f"++generation.dataloader.dataset.nres.nsamples={sampling.samples_per_job}",
        f"++generation.dataloader.dataset.nrepeat_per_sample={sampling.nrepeat_per_sample}",
        f"++generation.dataloader.batch_size={sampling.batch_size}",
        "++generation.search.algorithm=best-of-n",
        f"++generation.search.best_of_n.replicas={sampling.replicas}",
        f"++generation.filter.filter_samples_limit={sampling.keep_per_job}",
        f"++generation.reward_model.reward_models.af2folding.seed={sampling.reward_seed}",
        # GPU counts, not task counts. The harness fans out over Slurm jobs, so
        # each one sees a single device.
        "++gen_njobs=1", "++eval_njobs=1",
        f"++ncpus_={model.resources.cpus}",
    ]

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(model, node_tmp),
        # `--pwd` does not create the directory, and the image writes its cache
        # and TMPDIR without creating either root.
        mkdirs=(task_dir, node_tmp, node_tmp / "complexa-cache"),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _node_tmp(model: ProteinaComplexaConfig, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-proteina-complexa-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _environment(model: ProteinaComplexaConfig, node_tmp: Path) -> dict[str, str]:
    """Node-local scratch, the runtime cache, and the JAX/torch truce.

    `SINGULARITYENV_*` survives `--cleanenv`, which is what makes it safe to
    keep the host's own environment out of the image.
    """
    environment = {
        # Must not be on Lustre; see docs/known-issues.md section 2.1.
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        "SINGULARITYENV_COMPLEXA_RUNTIME_CACHE": str(node_tmp / "complexa-cache"),
    }
    if not model.runtime.xla_preallocate:
        # AF2 is JAX and the generative model is torch, on one device. Left
        # alone JAX reserves ~75% of the GPU at import and starves torch, which
        # surfaces as an out-of-memory error from the wrong library.
        environment["SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # `--cleanenv` drops the variable Slurm sets to name the allocated device,
    # so forward it explicitly rather than let the container guess. Read here
    # rather than at planning because the allocation only exists now.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
