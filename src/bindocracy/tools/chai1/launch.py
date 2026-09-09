"""Build the command for one Chai-1 scoring task. Submission is Snakemake's job."""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.chai1.config import Chai1Config

METRICS_FILE = "metrics.jsonl"
# The image's own interpreter, not `python`. `singularity exec ... python`
# resolves against whatever PATH survives, and the image ships a system
# python3 alongside the pinned 3.11.11 venv that has torch in it.
CONTAINER_PYTHON = "/opt/chai-venv/bin/python"


def chai1_launch_spec(
    general: GeneralConfig, model: Chai1Config, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    driver = run_dir / manifest.provenance["driver"].path
    task_dir = run_dir / task.directory

    # The design set the run was PLANNED with, from the manifest rather than
    # the live config, so a resumed task cannot be redirected at a set that has
    # since been rebuilt under the same path.
    design_fasta = manifest.workflow["design_set_fasta"]

    # Chai writes a directory per fold and refuses a non-empty output dir
    # (`chai1.py:505`), so each task gets its own tree under the task dir.
    work = task_dir / "folds"

    argv = [
        "singularity", "exec", "--cleanenv", "--nv",
        str(model.runtime.container),
        CONTAINER_PYTHON, str(driver),
        "--design-set", str(design_fasta),
        "--target-fasta", str(general.target.sequence_fasta),
        "--target-chain", general.target.chain_id,
        "--work-dir", str(work),
        "--save-dir", str(task_dir),
        "--shard", str(task.task_id),
        "--num-shards", str(len(manifest.tasks)),
        "--task-id", str(task.task_id),
        "--num-trunk-recycles", str(model.protocol.num_trunk_recycles),
        "--num-diffn-timesteps", str(model.protocol.num_diffn_timesteps),
        "--num-diffn-samples", str(model.protocol.num_diffn_samples),
        "--num-trunk-samples", str(model.protocol.num_trunk_samples),
        "--recycle-msa-subsample", str(model.protocol.recycle_msa_subsample),
        "--seed", str(model.protocol.seed),
    ]
    if not model.protocol.use_esm_embeddings:
        argv.append("--no-esm-embeddings")
    if model.protocol.low_memory:
        argv.append("--low-memory")
    if model.protocol.use_target_msa and model.runtime.msa_directory is not None:
        argv.extend(("--msa-directory", str(model.runtime.msa_directory)))

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(model, task_dir),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
        mkdirs=(work, Path(model.runtime.scratch) / f"task-{task.task_id:04d}"),
    )


def _environment(model: Chai1Config, task_dir: Path) -> dict[str, str]:
    """`SINGULARITYENV_*` survives `--cleanenv`, which is what makes this work."""
    scratch = Path(model.runtime.scratch) / task_dir.name
    environment = {
        # The image routes torch, numba, triton, matplotlib, HF and CUDA caches
        # under this one directory (`chai1_offline.py`). Unset, it makes a
        # fresh `/tmp/chai1-*` per invocation and never removes it -- which on
        # a shared node is a slow leak rather than an error.
        "SINGULARITYENV_CHAI_RUNTIME_DIR": str(scratch),
        # Belt and braces: the image already sets both in %environment, but
        # --cleanenv means the driver's own imports should not be able to reach
        # the network even if that changes.
        "SINGULARITYENV_HF_HUB_OFFLINE": "1",
        "SINGULARITYENV_TRANSFORMERS_OFFLINE": "1",
    }
    # known-issues.md §2.6: --cleanenv discards the variable SLURM uses to name
    # the allocated GPU, and the container then sees every device on the node.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
