"""BindCraft 2's authored configuration.

BC2 is configured by **one** JSON document, unlike its predecessor's three. The
document is a campaign: the binder format and length, the loss weights, the
stage schedule, and the acceptance filters. Presets shipped inside the image
supply defaults under it, and the harness overrides a handful of values on the
command line.

Six settings are harness-owned and passed with `--set` rather than authored --
`project_folder`, `number_of_final_designs`, `max_trajectories`,
`campaign_seed`, `resume`, and the whole `targets` block -- and preflight
refuses a template that sets any of them itself. Two answers to "how many
designs did this run ask for" is worse than none, and an authored epitope is one
more thing that can drift away from the campaign's.

The document is folded into the stored config. It is the science: the filter
thresholds alone are the definition of `n_passed`, and they live in no other
queryable place.

The contrast with FreeBindCraft is that **no driver is needed**. BindCraft 2
exposes every harness-owned value as a `--set KEY=VALUE` override, including the
whole target block as JSON, so the archived document is passed to the container
as it stands. Verified against the image: a campaign whose template names no
target at all passes preflight when the target arrives through `--set`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, ToolConfig


class BindCraft2Document(ConfigModel):
    """The authored campaign JSON, archived and run from the archive.

    `contents` is filled in from `template` when the config is loaded, so the
    stored JSON holds the document itself rather than a path to it. Authors
    leave it out; a recovered config carries it and still validates.
    """

    template: Path
    contents: dict | None = None


class BindCraft2SamplingConfig(ConfigModel):
    """The two counters that end a campaign, and the seed that makes it repeat.

    A campaign stops when *either* `designs_per_job` accepted designs exist
    *or* `max_trajectories` trajectories have been spent. Whichever trips first
    ends it.
    """

    jobs: int = Field(default=1, gt=0)
    # `number_of_final_designs`. ACCEPTED designs, after every filter -- a
    # stopping condition, not a table size. Several workers run concurrently
    # and each accepts independently, so a task OVERSHOOTS rather than stopping
    # exactly: a budget of 10 returned 11 on the 2026-09-22 run. Never fewer
    # than asked unless the trajectory budget ran out first.
    designs_per_job: int = Field(gt=0)
    # `max_trajectories`. Trajectory ATTEMPTS, not designs and not successes:
    # a trajectory terminated at any design stage still counts against it. On
    # the 2026-09-22 run only 29 of 184 ran the whole way through, so budget
    # roughly six attempts per completed trajectory.
    max_trajectories: int = Field(gt=0)
    # `campaign_seed`, offset by the task id so two tasks provably differ.
    # BindCraft 2 draws every trajectory from this, which is the one clear
    # advance over FreeBindCraft's unseeded global RNG: a run here is
    # reproducible, and the plan records that it is.
    seed_base: int = Field(ge=0)
    # `BINDCRAFT_WORKERS_PER_GPU`. None leaves BC2's own packing, which fills
    # each card to its free memory -- up to seven workers. Pin it when the
    # allocation's host memory cannot hold that many: each worker holds its own
    # copy of the parameters at about 4 GB, and BC2 reads the NODE's memory
    # rather than the cgroup, so its own cap will not protect the job.
    workers_per_gpu: int | None = Field(default=None, gt=0)


class BindCraft2RuntimeConfig(ConfigModel):
    container: Path
    # Node-local scratch root. TMPDIR must NOT be on Lustre; see
    # docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class BindCraft2Config(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["bindcraft2"]
    settings: BindCraft2Document
    sampling: BindCraft2SamplingConfig
    runtime: BindCraft2RuntimeConfig
