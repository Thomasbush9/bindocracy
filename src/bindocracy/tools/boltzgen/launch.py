"""Build the command for one BoltzGen task.

BoltzGen has no wrapper script and no driver -- the container's runscript takes
the archived design spec directly.
"""

from __future__ import annotations

from typing import Any

from bindocracy.config.load import LoadedConfigs
from bindocracy.runs.launch import LaunchSpec, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.mosaic.launch import mosaic_resources


def boltzgen_launch_spec(
    loaded: LoadedConfigs, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One BoltzGen task: `singularity run <sif> run <spec> --output <dir>`.

    BoltzGen has no wrapper script and no driver -- the container's runscript
    takes the archived design spec directly, so the spec is what gets bound in
    and archived, exactly where Mosaic archives its driver.
    """
    task = task_of(manifest, task_id)
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
