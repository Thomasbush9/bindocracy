"""Build the command for one Mosaic task without submitting anything.

Snakemake owns submission: it renders `LaunchSpec.command` in a job and passes
`LaunchSpec.resources` to the Slurm executor plugin. Keeping `sbatch` out of
here means job state, retries, and logs stay with one scheduler, and the whole
launch is unit-testable on a login node.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.runs.manifest import RunManifest, TaskPlan


@dataclass(frozen=True)
class LaunchSpec:
    argv: tuple[str, ...]
    env: dict[str, str]
    resources: dict[str, Any]
    log: Path
    outputs: tuple[Path, ...]

    @property
    def command(self) -> str:
        """One shell-safe string: `env K=V ... wrapper python driver ...`."""
        assignments = [f"{key}={value}" for key, value in sorted(self.env.items())]
        return shlex.join(["env", *assignments, *self.argv])


def mosaic_launch_spec(
    loaded: LoadedConfigs, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """Everything needed to run one Mosaic task of a planned run."""
    task = _task(manifest, task_id)
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


def _task(manifest: RunManifest, task_id: int) -> TaskPlan:
    for task in manifest.tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"run {manifest.run_id} has no task {task_id}")


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


def boltzgen_launch_spec(
    loaded: LoadedConfigs, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One BoltzGen task: `singularity run <sif> run <spec> --output <dir>`.

    BoltzGen has no wrapper script and no driver -- the container's runscript
    takes the archived design spec directly, so the spec is what gets bound in
    and archived, exactly where Mosaic archives its driver.
    """
    task = _task(manifest, task_id)
    model = loaded.model
    run_dir = manifest.directory
    spec = run_dir / manifest.provenance["spec"].path
    sampling = model.sampling

    argv = [
        "singularity", "run", "--cleanenv", "--nv", str(model.runtime.container),
        "run", str(spec),
        "--output", str(run_dir / task.directory),
        "--protocol", sampling.protocol,
        "--num_designs", str(sampling.num_designs),
        "--budget", str(sampling.budget),
        "--filter_biased", "true" if sampling.filter_biased else "false",
        "--devices", str(model.resources.gpus),
        "--num_workers", str(model.resources.cpus),
    ]
    if sampling.diffusion_batch_size is not None:
        argv += ["--diffusion_batch_size", str(sampling.diffusion_batch_size)]

    return LaunchSpec(
        argv=tuple(argv),
        env=_boltzgen_environment(loaded, manifest, task_id),
        resources=boltzgen_resources(loaded),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _boltzgen_environment(
    loaded: LoadedConfigs, manifest: RunManifest, task_id: int
) -> dict[str, str]:
    """TMPDIR must be node-local, and the container needs its own CA bundle.

    Triton compiles every GPU kernel inside a temporary directory, and on
    Lustre the cleanup fails with Errno 39 and kills the job on its first
    kernel -- BoltzGen died this way in the benchmark. The path is derived from
    the run and task so two concurrent jobs on one node cannot collide.
    See docs/known-issues.md sections 2.1 and 2.5.
    """
    node_tmp = (
        loaded.model.runtime.node_tmp_root
        / f"bindocracy-boltzgen-{manifest.run_id[:8]}-{task_id:04d}"
    )
    return {
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        "SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE": str(node_tmp / "boltzgen-cache"),
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }


def boltzgen_resources(loaded: LoadedConfigs) -> dict[str, Any]:
    """Same Slurm shape as Mosaic; the numbers come from this tool's config."""
    return mosaic_resources(loaded)
