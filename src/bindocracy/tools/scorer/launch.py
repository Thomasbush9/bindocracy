"""Build the command for one scoring task. Submission is Snakemake's job."""

from __future__ import annotations

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.scorer.config import ScorerConfig

METRICS_FILE = "metrics.jsonl"


def scorer_launch_spec(
    general: GeneralConfig, scorer: ScorerConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """Everything needed to score one shard of a planned run."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    driver = run_dir / manifest.provenance["driver"].path

    # The design set the run was PLANNED with, from the manifest rather than
    # from the live config. A campaign that rebuilt its design sets since must
    # not silently redirect a resumed task at a different set of designs.
    design_fasta = manifest.workflow["design_set_fasta"]

    argv = [
        str(scorer.runtime.exec_wrapper),
        "python",
        str(driver),
        "--design-set",
        str(design_fasta),
        "--target-fasta",
        str(general.target.sequence_fasta),
        "--model",
        scorer.model.name,
        "--recycling",
        str(scorer.model.recycling_steps),
        "--num-samples",
        str(scorer.model.num_samples),
        "--seed",
        str(scorer.model.seed),
        "--readers",
        ",".join(_readers(scorer)),
        "--shard",
        str(task.task_id),
        "--num-shards",
        str(len(manifest.tasks)),
        "--task-id",
        str(task.task_id),
        "--save-dir",
        str(run_dir / task.directory),
    ]
    if scorer.model.sampling_steps is not None:
        argv.extend(("--sampling-steps", str(scorer.model.sampling_steps)))
    if scorer.model.use_target_msa and general.target.msa is not None:
        argv.extend(("--target-msa", str(general.target.msa)))

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(scorer),
        resources=slurm_resources(general.cluster, scorer.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
    )


def _readers(scorer: ScorerConfig) -> list[str]:
    """The conditions the driver folds, in a fixed order.

    Complex first so a run killed by walltime has the metric that matters most
    for the designs it did reach, rather than a monomer number for all of them
    and an interface number for none.
    """
    enabled = []
    if scorer.readers.complex:
        enabled.append("complex")
    if scorer.readers.monomer:
        enabled.append("monomer")
    return enabled


def _environment(scorer: ScorerConfig) -> dict[str, str]:
    """What mosaic-exec.sh reads, plus the container fixes from known-issues §2."""
    runtime = scorer.runtime
    env = {
        "MOSAIC_SIF": str(runtime.container),
        "MOSAIC_WEIGHTS": str(runtime.weights),
        "MOSAIC_SCRATCH": str(runtime.scratch),
        "SINGULARITYENV_JAX_COMPILATION_CACHE_DIR": "/jax_cache",
        # The host's RHEL CA path does not exist inside the Ubuntu image and
        # breaks httpx at import, before offline mode is ever consulted.
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        # Every backend here resolves its alignments through mosaic's
        # `require_msa`, which raises rather than falling back to a public
        # server. Leaving the opt-out unset is what keeps that guarantee: the
        # fallback succeeds, returns plausible structures, and leaves nothing
        # in the output to say the alignment changed.
        "SINGULARITYENV_MOSAIC_ALLOW_MSA_SERVER": "0",
        # The weight caches are bound read-only, and huggingface_hub tries to
        # write a `refs/main` file while resolving a repo even when every blob
        # it needs is already present. That surfaces as
        # `OSError: [Errno 30] Read-only file system` from inside a model
        # constructor, which reads like a missing weight rather than a cache
        # write. Offline mode skips the resolution entirely.
        "SINGULARITYENV_HF_HUB_OFFLINE": "1",
        "SINGULARITYENV_TRANSFORMERS_OFFLINE": "1",
    }
    if runtime.dev_source is not None:
        # Binds over /opt/mosaic/src. Only needed until the image carries
        # the MSA-routing fix; preflight has already digested this tree and
        # the run records the digest beside the container.
        env["MOSAIC_DEV_SRC"] = str(runtime.dev_source)
    return env
