"""Build the command for one AlphaFold 3 scoring task."""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.af3.config import AF3Config

METRICS_FILE = "metrics.jsonl"
# The image's venv interpreter. `singularity exec ... python` would resolve
# against whatever PATH survives, and the image has a system python3 beside the
# pinned 3.12 venv that holds jax and alphafold3.
CONTAINER_PYTHON = "python"
CONTAINER_MODELS = "/models"


def af3_launch_spec(
    general: GeneralConfig, model: AF3Config, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    driver = run_dir / manifest.provenance["driver"].path
    task_dir = run_dir / task.directory
    work = Path(model.runtime.work_root) / f"task-{task.task_id:04d}"

    design_fasta = manifest.workflow["design_set_fasta"]

    argv = [
        "singularity", "exec", "--nv", "--cleanenv",
        # The weights, read-only. Bound rather than embedded because the terms
        # restrict distribution; see config.py.
        "--bind", f"{model.runtime.model_dir}:{CONTAINER_MODELS}:ro",
        "--bind", f"{work}:{work}",
        str(model.runtime.container),
        CONTAINER_PYTHON, str(driver),
        "--design-set", str(design_fasta),
        "--target-fasta", str(general.target.sequence_fasta),
        "--model-dir", CONTAINER_MODELS,
        "--work-dir", str(work),
        "--save-dir", str(task_dir),
        "--shard", str(task.task_id),
        "--num-shards", str(len(manifest.tasks)),
        "--task-id", str(task.task_id),
        "--seed", str(model.protocol.seed),
        "--num-diffn-samples", str(model.protocol.num_diffn_samples),
    ]
    if model.protocol.use_target_msa and general.target.msa is not None:
        argv.extend(("--target-msa", str(general.target.msa)))

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(work),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
        mkdirs=(work, work / "jax_cache"),
    )


def _environment(work: Path) -> dict[str, str]:
    """`SINGULARITYENV_*` survives `--cleanenv`."""
    environment = {
        # The image sets these in %environment; repeated here so a bare
        # `singularity exec` cannot reach the network either.
        "SINGULARITYENV_HF_HUB_OFFLINE": "1",
        "SINGULARITYENV_TRANSFORMERS_OFFLINE": "1",
        # AF3 is a JAX program, and so is every mosaic scorer -- which means a
        # workflow that runs both inherits mosaic's
        # JAX_COMPILATION_CACHE_DIR=/jax_cache, a path that exists only inside
        # mosaic.sif because mosaic-exec.sh binds it there. AF3 then dies with
        # `NOT_FOUND: /jax_cache/xla_gpu_per_fusion_autotune_cache_dir`, which
        # names a directory nobody configured and reads like a broken image.
        # Observed 2026-09-09. Setting it explicitly to this task's own work
        # directory both fixes that and earns the cache: 434 designs of varying
        # length otherwise recompile per length.
        "SINGULARITYENV_JAX_COMPILATION_CACHE_DIR": str(work / "jax_cache"),
    }
    # known-issues.md section 2.6: --cleanenv drops the variable SLURM uses to
    # name the allocated GPU, and the container then sees every device.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
