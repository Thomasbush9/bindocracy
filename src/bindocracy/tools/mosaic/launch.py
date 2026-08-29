"""Build the command for one Mosaic task. Submission is Snakemake's job."""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.runs.launch import LaunchSpec, task_of
from bindocracy.runs.manifest import RunManifest


def mosaic_launch_spec(
    loaded: LoadedConfigs, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """Everything needed to run one Mosaic task of a planned run."""
    task = task_of(manifest, task_id)
    mosaic = loaded.model
    run_dir = manifest.directory
    # The archived driver, not the authored one: the working tree may have
    # moved on since this run was planned.
    driver = run_dir / manifest.provenance["driver"].path

    argv = (
        str(mosaic.runtime.exec_wrapper),
        "python",
        str(driver),
        "--target-fasta", str(loaded.general.target.sequence_fasta),
        "--target-msa", str(loaded.general.target.msa),
        "--binder-length", str(mosaic.sampling.binder_length),
        "--task-id", str(task.task_id),
        "--seed-base", str(mosaic.sampling.seed_base),
        "--n-designs", str(task.n_requested),
        "--max-runtime", str(mosaic.sampling.max_runtime_hours),
        "--save-dir", str(run_dir / task.directory),
        "--soft-steps", str(mosaic.sampling.optimizer.soft_steps),
        "--sharpen-steps", str(mosaic.sampling.optimizer.sharpen_steps),
        "--final-steps", str(mosaic.sampling.optimizer.final_steps),
    )
    return LaunchSpec(
        argv=argv,
        env=_environment(loaded),
        resources=mosaic_resources(loaded),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
    )


def _environment(loaded: LoadedConfigs) -> dict[str, str]:
    """What mosaic-exec.sh reads, plus the two container fixes from env.sh."""
    runtime = loaded.model.runtime
    return {
        "MOSAIC_SIF": str(runtime.container),
        "MOSAIC_WEIGHTS": str(runtime.weights),
        "MOSAIC_SCRATCH": str(runtime.scratch),
        # Shared XLA kernel cache: only the first run pays the compile.
        "SINGULARITYENV_JAX_COMPILATION_CACHE_DIR": "/jax_cache",
        # The host's RHEL CA path does not exist inside the Ubuntu image and
        # breaks httpx at import; point at the container's own bundle.
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }


def mosaic_resources(loaded: LoadedConfigs) -> dict[str, Any]:
    """Snakemake Slurm-executor resource names.

    Separate from `mosaic_launch_spec` because Snakemake needs a rule's
    resources while it builds the DAG, before any run manifest exists.
    """
    resources = loaded.model.resources
    cluster = loaded.general.cluster
    return {
        "slurm_account": cluster.account,
        "slurm_partition": cluster.default_partition,
        # The plugin owns --gres and rejects it in slurm_extra.
        "gres": f"gpu:{resources.gpus}",
        "cpus_per_task": resources.cpus,
        "mem_mb": resources.memory_gb * 1024,
        "runtime": resources.walltime_seconds // 60,
    }


