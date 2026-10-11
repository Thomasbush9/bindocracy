"""Build the command for one BoltzGen task.

BoltzGen consumes the resolved specification frozen in the run configuration.
The original specification is retained separately as provenance.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.boltzgen.config import BoltzGenConfig


def boltzgen_launch_spec(
    general: GeneralConfig, model: BoltzGenConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One BoltzGen task: `singularity run <sif> run <spec> --output <dir>`.

    The task-local spec is materialized from the frozen model, never reread from
    the original template or resolved relative to the worker's working directory.
    """
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    if model.spec.contents is None:
        raise ValueError("BoltzGen launch requires a resolved specification")
    spec = run_dir / task.directory / "spec.yaml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(yaml.safe_dump(model.spec.contents, sort_keys=False))
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

    node_tmp = _node_tmp(model, manifest, task_id)
    return LaunchSpec(
        argv=tuple(argv),
        env=_boltzgen_environment(model, manifest, task_id),
        mkdirs=(node_tmp, node_tmp / "boltzgen-cache"),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _node_tmp(model: BoltzGenConfig, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-boltzgen-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _boltzgen_environment(
    model: BoltzGenConfig, manifest: RunManifest, task_id: int
) -> dict[str, str]:
    """TMPDIR must be node-local, and the container needs its own CA bundle.

    Triton compiles every GPU kernel inside a temporary directory, and on
    Lustre the cleanup fails with Errno 39 and kills the job on its first
    kernel -- BoltzGen died this way in the benchmark. The path is derived from
    the run and task so two concurrent jobs on one node cannot collide.
    See docs/known-issues.md sections 2.1 and 2.5.
    """
    node_tmp = _node_tmp(model, manifest, task_id)
    return {
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        "SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE": str(node_tmp / "boltzgen-cache"),
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }
