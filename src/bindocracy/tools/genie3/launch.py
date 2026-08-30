"""Build the command for one Genie 3 task.

Two things make this longer than BoltzGen's launch. Genie 3 has no CLI override
for its output root, its seed, or its sample count, so a driver renders one
config per task; and the image's JAX has no CUDA plugin, so three host overlays
have to be threaded in through the environment or the AF2 evaluation stage runs
on the CPU and never finishes. See docs/known-issues.md sections 2.3 and 2.4.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.genie3.config import Genie3Config

# The image's own interpreter. The runscript reaches Genie 3 the same way, and
# `singularity exec ... python` would resolve against whatever PATH survives.
CONTAINER_PYTHON = "/opt/conda/envs/genie3/bin/python"
# Genie 3's Trainer is built without `default_root_dir` or `logger=False`, so
# Lightning does makedirs('/opt/genie3/lightning_logs') on a read-only path and
# the run dies before generating anything. There is no config knob; binding
# writable space over the missing path is the least invasive fix.
CONTAINER_LIGHTNING_LOGS = "/opt/genie3/lightning_logs"
# What the container's PATH becomes. ptxas has to come first, and Genie 3's own
# environment has to stay reachable behind it.
CONTAINER_PATH_TAIL = "/opt/conda/envs/genie3/bin:/usr/local/bin:/usr/bin:/bin"

# Written inside the task directory by the driver, and read back by the adapter
# as the record of what this task actually ran.
RENDERED_CONFIG = "experiment.yaml"
LIGHTNING_LOGS = "lightning_logs"
GENIE3_LOGS = "genie3_logs"
CACHE = "_cache"


def genie3_launch_spec(
    general: GeneralConfig, model: Genie3Config, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    """One Genie 3 task: the archived driver, run against the archived template."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived copies, not the authored ones: the working tree may have
    # moved on since this run was planned.
    driver = run_dir / manifest.provenance["driver"].path
    template = run_dir / manifest.provenance["experiment"].path

    node_tmp = _node_tmp(model, manifest, task_id)
    lightning_logs = task_dir / LIGHTNING_LOGS

    argv = [
        "singularity", "exec", "--cleanenv", "--nv",
        # The only bind worth naming. `/n` is a system bind path here, so the
        # run directory, the problem set and the overlays are already visible;
        # this one exists to put writable space over a path inside the image.
        "--bind", f"{lightning_logs}:{CONTAINER_LIGHTNING_LOGS}",
        str(model.runtime.container),
        CONTAINER_PYTHON, str(driver),
        "--template", str(template),
        "--config-out", str(task_dir / RENDERED_CONFIG),
        "--rootdir", str(task_dir),
        "--log-dir", str(task_dir / GENIE3_LOGS),
        "--cache", str(task_dir / CACHE),
        # Every task diffuses from its own seed. Sharing one would make two
        # tasks generate the same backbones and the run would return duplicates
        # without any of its counts changing.
        "--seed", str(model.sampling.seed_base + task_id),
        "--n-sample", str(model.sampling.backbones_per_job),
        "--num-devices", str(model.resources.gpus),
    ]

    return LaunchSpec(
        argv=tuple(argv),
        env=_genie3_environment(model, node_tmp),
        # The bind source has to exist on the host, and Genie 3 writes into its
        # own cache and TMPDIR without creating either root.
        mkdirs=(node_tmp, lightning_logs, task_dir / CACHE, task_dir / GENIE3_LOGS),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _node_tmp(model: Genie3Config, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-genie3-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _genie3_environment(model: Genie3Config, node_tmp: Path) -> dict[str, str]:
    """The three overlays, plus the two container fixes every image here needs.

    `SINGULARITYENV_*` survives `--cleanenv`, which is what makes it safe to
    keep the host's own environment out of the image.
    """
    runtime = model.runtime
    environment = {
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
        # jax finds a CUDA backend through the `jax_plugins` namespace package.
        "SINGULARITYENV_PYTHONPATH": str(runtime.jax_plugin_overlay),
        # cuDNN goes here and NOT on PYTHONPATH: these wheels ship a real
        # nvidia/__init__.py, so an overlay copy would shadow nvidia.cublas and
        # nvidia.nccl along with it.
        "SINGULARITYENV_LD_LIBRARY_PATH": str(runtime.cudnn_overlay),
        # XLA JIT needs ptxas and nvvm/libdevice; the image's only copy is
        # vendored inside triton, where XLA does not look.
        "SINGULARITYENV_XLA_FLAGS": (
            f"--xla_gpu_cuda_data_dir={runtime.cuda_nvcc_overlay}"
        ),
        "SINGULARITYENV_PATH": f"{runtime.cuda_nvcc_overlay}/bin:{CONTAINER_PATH_TAIL}",
        # The host's RHEL CA path does not exist inside this Ubuntu image.
        "SINGULARITYENV_SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "SINGULARITYENV_CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }
    # `--cleanenv` drops the variable Slurm sets to name the allocated device,
    # so forward it explicitly rather than let the container guess. Read here
    # rather than at planning because the allocation only exists now.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
