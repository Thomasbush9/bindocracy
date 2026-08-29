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
