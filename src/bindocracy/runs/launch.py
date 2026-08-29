"""What a tool must produce to have one task launched.

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

from bindocracy.config.models import ClusterConfig, ResourceConfig
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


def task_of(manifest: RunManifest, task_id: int) -> TaskPlan:
    for task in manifest.tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"run {manifest.run_id} has no task {task_id}")


def slurm_resources(cluster: ClusterConfig, resources: ResourceConfig) -> dict[str, Any]:
    """Map a tool's resource block onto the Slurm executor's resource names.

    Shared because every tool here wants one GPU node and differs only in the
    numbers. Snakemake needs these while it builds the DAG, before any run
    manifest exists, so this takes configs rather than a manifest.
    """
    return {
        "slurm_account": cluster.account,
        "slurm_partition": cluster.default_partition,
        # The plugin owns --gres and rejects it in slurm_extra.
        "gres": f"gpu:{resources.gpus}",
        "cpus_per_task": resources.cpus,
        "mem_mb": resources.memory_gb * 1024,
        "runtime": resources.walltime_seconds // 60,
    }
