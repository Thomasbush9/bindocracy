"""Build the command for one Mosaic task. Submission is Snakemake's job."""

from __future__ import annotations

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.mosaic.config import MosaicConfig


def mosaic_launch_spec(
    general: GeneralConfig, mosaic: MosaicConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """Everything needed to run one Mosaic task of a planned run."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    # The archived driver, not the authored one: the working tree may have
    # moved on since this run was planned.
    driver = run_dir / manifest.provenance["driver"].path

    argv = [
        str(mosaic.runtime.exec_wrapper),
        "python",
        str(driver),
        "--target-fasta",
        str(general.target.sequence_fasta),
        "--target-msa",
        str(general.target.msa),
        "--binder-length",
        str(mosaic.sampling.binder_length),
        "--task-id",
        str(task.task_id),
        "--seed-base",
        str(mosaic.sampling.seed_base),
        "--n-designs",
        str(task.n_requested),
        "--max-runtime",
        str(mosaic.sampling.max_runtime_hours),
        "--save-dir",
        str(run_dir / task.directory),
        "--soft-steps",
        str(mosaic.sampling.optimizer.soft_steps),
        "--sharpen-steps",
        str(mosaic.sampling.optimizer.sharpen_steps),
        "--final-steps",
        str(mosaic.sampling.optimizer.final_steps),
    ]
    # Keep the archived driver's CLI and its manifest in lockstep. Drivers
    # archived before epitope_idx was introduced do not accept this flag.
    if "epitope_idx" in manifest.workflow:
        argv.extend(("--epitope", _epitope(manifest)))
    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(mosaic),
        resources=slurm_resources(general.cluster, mosaic.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
    )


def _epitope(manifest: RunManifest) -> str:
    """The epitope indices this run was planned with, from the manifest alone.

    Called only for a manifest that records the key. Empty is still a value:
    it names the unconditioned loss in a current driver.
    """
    indices = manifest.workflow.get("epitope_idx") or ()
    return ",".join(str(int(index)) for index in indices)


def _environment(mosaic: MosaicConfig) -> dict[str, str]:
    """What mosaic-exec.sh reads, plus the two container fixes from env.sh."""
    runtime = mosaic.runtime
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
