"""FreeBindCraft's authored configuration.

BindCraft is driven by three JSON documents and no useful flags: a **target**
(what to design against, how long, how many), a **filter set** (what counts as
a design worth keeping), and an **advanced** profile (the 4-stage schedule, the
MPNN settings, the trajectory budget). Only four of its command-line arguments
are not paths, and none of them is the output directory, the design count, the
trajectory budget, or a seed.

So three keys are harness-owned and written per task by the driver --
`design_path`, `number_of_final_designs`, and `target_hotspot_residues` in the
target document, and `max_trajectories` in the advanced one -- and preflight
refuses a template that sets any of them itself. Two answers to "how many
designs did this run ask for" is worse than none, and an authored epitope is
one more thing that can drift away from the campaign's.

All three documents are folded into the stored config. They are the science:
the filter set alone is the definition of `n_passed`, and it lives in no other
queryable place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from bindocracy.config.models import ConfigModel, ToolConfig


class FreeBindCraftDocument(ConfigModel):
    """One authored JSON document, archived and run from the archive.

    `contents` is filled in from `template` when the config is loaded, so the
    stored JSON holds the document itself rather than a path to it. Authors
    leave it out; a recovered config carries it and still validates.
    """

    template: Path
    contents: dict | None = None


class FreeBindCraftDriverConfig(ConfigModel):
    """The driver renders this task's target and advanced JSON, then runs.

    BindCraft has no override for its output directory, so two tasks sharing
    one would resume each other's work -- the design loop skips any trajectory
    whose PDB already exists. The driver is what gives each task its own.
    """

    script: Path


class FreeBindCraftSamplingConfig(ConfigModel):
    """The two counters that end a task, and neither is a design count.

    The design loop stops when *either* `designs_per_job` accepted designs
    exist *or* `max_trajectories` successful hallucinations do. Whichever trips
    first ends the run, and only the first of them writes
    `final_design_stats.csv`.
    """

    jobs: int = Field(default=1, gt=0)
    # `number_of_final_designs`. Accepted designs, after every filter -- not
    # candidates. The run stops at the first check where at least this many
    # PDBs sit in `Accepted/`, so a run can finish with one or two more.
    designs_per_job: int = Field(gt=0)
    # `max_trajectories`. No default and not optional, though the tool's own
    # value is `false`: without it a task designs until walltime, and a
    # walltime kill is the one ending that leaves no record of why it stopped.
    # It counts *successful* hallucinations only -- trajectories that abort as
    # clashing or low-confidence are moved aside and never counted -- so the
    # number of attempts is always larger.
    max_trajectories: int = Field(gt=0)


class FreeBindCraftRuntimeConfig(ConfigModel):
    container: Path
    # Which metric `final_design_stats.csv` is ranked by. `ipSAE` is this
    # fork's addition; `i_pTM` is BindCraft's own default.
    rank_by: Literal["i_pTM", "ipSAE"] = "i_pTM"
    # Off by default: an HTML animation and four PNGs per trajectory are tens
    # of megabytes and minutes of wall time, and the run zips them at the end
    # into files nothing here reads. The advanced profile's own values say
    # true, so these are passed as `--no-plots` / `--no-animations`.
    save_plots: bool = False
    save_animations: bool = False
    # Node-local scratch root. TMPDIR must NOT be on Lustre; see
    # docs/known-issues.md section 2.1.
    node_tmp_root: Path = Path("/tmp")


class FreeBindCraftConfig(ToolConfig):
    schema_version: Literal[1]
    tool: Literal["freebindcraft"]
    target: FreeBindCraftDocument
    filters: FreeBindCraftDocument
    advanced: FreeBindCraftDocument
    driver: FreeBindCraftDriverConfig
    sampling: FreeBindCraftSamplingConfig
    runtime: FreeBindCraftRuntimeConfig
