"""Build the command for one upstream OpenFold3 scoring task."""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.of3_upstream.config import OF3UpstreamConfig

METRICS_FILE = "metrics.jsonl"
CONTAINER_PYTHON = "python3"


def of3_upstream_launch_spec(
    general: GeneralConfig, model: OF3UpstreamConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    driver = run_dir / manifest.provenance["driver"].path
    task_dir = run_dir / task.directory
    work = Path(model.runtime.work_root) / f"task-{task.task_id:04d}"

    design_fasta = manifest.workflow["design_set_fasta"]

    argv = [
        "singularity", "exec", "--nv", "--cleanenv",
        "--bind", f"{work}:{work}",
        str(model.runtime.container),
        CONTAINER_PYTHON, str(driver),
        "--design-set", str(design_fasta),
        "--target-fasta", str(general.target.sequence_fasta),
        "--checkpoint", str(model.runtime.checkpoint),
        "--work-dir", str(work),
        "--save-dir", str(task_dir),
        "--shard", str(task.task_id),
        "--num-shards", str(len(manifest.tasks)),
        "--task-id", str(task.task_id),
        "--seed", str(model.protocol.seed),
        "--num-diffusion-samples", str(model.protocol.num_diffusion_samples),
    ]
    if model.protocol.use_target_msa and general.target.msa is not None:
        argv.extend(("--target-msa", str(general.target.msa)))

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
        mkdirs=(work,),
    )


def _environment() -> dict[str, str]:
    """`SINGULARITYENV_*` survives `--cleanenv`."""
    environment = {
        # OpenFold3 will download parameters into $OPENFOLD_CACHE if it is not
        # handed a checkpoint. It always is here, but a run must not be able to
        # reach the network for weights partway through a shard.
        "SINGULARITYENV_HF_HUB_OFFLINE": "1",
        "SINGULARITYENV_TRANSFORMERS_OFFLINE": "1",
    }
    # known-issues.md section 2.6: --cleanenv drops the variable SLURM uses to
    # name the allocated GPU.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
