"""Build the command for one PXDesign task.

Like BoltzGen, PXDesign needs no driver: every value the harness owns is a CLI
flag, so the archived spec is passed to the container's `pipeline` as it stands.

The flags that look redundant are not. `--preset` defaults to `custom`, which
configures no confidence filters; the eta schedule has to be repeated back
because the CLI overwrites the container's config with its own defaults; and
`--seeds` is what makes the run reproducible and two tasks distinct. See
docs/known-issues.md sections 1.5 and 1.6.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.pxdesign.config import PXDesignConfig

# PXDesign writes `./msa_cache/` relative to its working directory, so the
# working directory has to be the task's own -- two tasks sharing one would
# race on it, which is the failure in docs/known-issues.md section 6.1b.
CACHE_DIR = "pxdesign-cache"


def pxdesign_launch_spec(
    general: GeneralConfig, model: PXDesignConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One PXDesign task: `singularity run <sif> pipeline -i <spec> -o <task>`."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived spec, not the authored one: the working tree may have moved
    # on since this run was planned.
    spec = run_dir / manifest.provenance["spec"].path
    sampling = model.sampling

    argv = [
        "singularity", "run", "--cleanenv", "--nv",
        # /n is a system bind path here, so the only thing worth naming is the
        # working directory PXDesign resolves its cache against.
        "--pwd", str(task_dir),
        str(model.runtime.container),
        "pipeline",
        "-i", str(spec),
        "-o", str(task_dir),
        # Explicit because the CLI's own default is `custom`, which runs with
        # no confidence filters and still writes a normal-looking summary.csv.
        "--preset", sampling.preset,
        "--N_sample", str(sampling.designs_per_job),
        "--N_step", str(sampling.diffusion_steps),
        "--N_max_runs", "1",
        "--dtype", model.runtime.dtype,
        # Repeated back because the CLI emits every shared option
        # unconditionally and would otherwise overwrite the container's
        # intended schedule with const / 2.5 / 2.5.
        "--eta_type", sampling.eta_type,
        "--eta_min", str(sampling.eta_min),
        "--eta_max", str(sampling.eta_max),
        # One seed per run, and `--N_max_runs 1` means exactly one is expected.
        # Without it PXDesign seeds from the clock: unreproducible, and two
        # tasks would only differ by when they happened to start.
        "--seeds", str(sampling.seed_base + task_id),
        # Forwarded through click to the inner argparse.
        "--use_fast_ln", _flag(model.runtime.use_fast_ln),
        "--use_deepspeed_evo_attention", _flag(model.runtime.use_deepspeed_evo_attention),
    ]

    node_tmp = _node_tmp(model, manifest, task_id)
    return LaunchSpec(
        argv=tuple(argv),
        env=_pxdesign_environment(node_tmp),
        mkdirs=(node_tmp, node_tmp / CACHE_DIR),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _flag(value: bool) -> str:
    """The inner argparse reads these as strings, not as store_true flags."""
    return "True" if value else "False"


def _node_tmp(model: PXDesignConfig, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-pxdesign-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _pxdesign_environment(node_tmp: Path) -> dict[str, str]:
    """Every cache this image writes, moved off Lustre and off the home quota.

    PXDesign JITs custom kernels through Triton and DeepSpeed. Triton's
    temp-dir cleanup fails on Lustre with Errno 39 and kills the job on its
    first kernel; its autotune cache defaults to `~/.triton` on NFS, which the
    image itself warns about; and PXDESIGN_CACHE defaults under `$HOME`, which
    would put first-run kernel builds on the home quota.
    """
    return {
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        "SINGULARITYENV_PXDESIGN_CACHE": str(node_tmp / CACHE_DIR),
        "SINGULARITYENV_TRITON_CACHE_DIR": str(node_tmp / "triton"),
        # The host's RHEL CA path does not exist inside this image.
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        **_allocated_devices(),
    }


def _allocated_devices() -> dict[str, str]:
    """`--cleanenv` drops the variable Slurm sets to name the allocated GPU."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"SINGULARITYENV_CUDA_VISIBLE_DEVICES": devices} if devices else {}
