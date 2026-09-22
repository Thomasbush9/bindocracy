"""Build the command for one BindCraft 2 task.

No driver, and that is the whole point of this module. FreeBindCraft needed one
because BindCraft 1 had no override for its output directory, its design count
or its epitope. BindCraft 2 exposes every one of them as `--set KEY=VALUE`,
including the entire target block as JSON, so the archived campaign document is
passed to the container as it stands and the harness's values ride on the
command line where they are visible in the log.

Verified against the image on 2026-09-22: a campaign document naming no target
at all passes BC2's own preflight when the target arrives through `--set`, with
the path, the chain and the epitope all resolved correctly.

`--pwd` matters only as a belt: every path the harness passes is absolute, and
BC2 resolves `project_folder` against the working directory when it is not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.bindcraft2.config import BindCraft2Config

# BindCraft 2's own output tree, under the task directory rather than beside the
# harness's files, so `status.json` and the logs cannot be mistaken for campaign
# output by a resumed run.
CAMPAIGN_DIR = "campaign"

# The three stage tables, in the layout a fresh campaign folder always uses.
# BC2 keeps a flat legacy layout for folders written by older versions; a task
# directory is always new, so the stage layout is the only one that can appear
# here. The adapter still accepts both.
TRAJECTORIES_FILE = f"{CAMPAIGN_DIR}/1_Trajectories/!_Trajectories.csv"
REFOLDED_FILE = f"{CAMPAIGN_DIR}/2_Refolded/!_Refolded.csv"
RANKED_FILE = f"{CAMPAIGN_DIR}/3_Ranked/!_Ranked.csv"

# Declared as the task's output, and it is the completion marker rather than the
# design list. It is the first table a campaign writes and the only one every
# task that ran at all produces: a task whose every trajectory terminated early
# writes no candidates and no accepted designs, which is a real outcome and not
# a Snakemake failure. The adapter reads all three stages regardless.
DESIGNS_FILE = TRAJECTORIES_FILE

# What BC2 writes beside the stage directories.
STATE_FILE = f"{CAMPAIGN_DIR}/.campaign_state.json"
METADATA_FILE = f"{CAMPAIGN_DIR}/campaign_metadata.json"
SUMMARY_FILE = f"{CAMPAIGN_DIR}/summary.csv"
WORKERS_DIR = f"{CAMPAIGN_DIR}/workers"


def bindcraft2_launch_spec(
    general: GeneralConfig,
    model: BindCraft2Config,
    manifest: RunManifest,
    task_id: int,
) -> LaunchSpec:
    """One task: `singularity exec <sif> bindcraft design <campaign> --set ...`."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived document, not the authored one: the working tree may have
    # moved on since this run was planned.
    settings = run_dir / manifest.provenance["settings"].path
    sampling = model.sampling

    argv = [
        "singularity", "exec", "--cleanenv", "--nv",
        "--pwd", str(task_dir),
        str(model.runtime.container),
        "bindcraft", "design", str(settings),
        # The campaign target, as one JSON value. This is what removes the need
        # for a driver: `targets` is a list, so no dotted key could reach it,
        # but the whole block passes as JSON because `--set` parses its value
        # with json.loads and falls back to a bare string.
        "--set", f"targets={_targets(manifest)}",
        # Its own directory per task. A campaign resumes by default, so two
        # tasks sharing one would continue each other's work and both would
        # report the same designs.
        "--set", f"project_folder={task_dir / CAMPAIGN_DIR}",
        # The stopping condition, and the budget that ends a task that cannot
        # reach it.
        "--set", f"number_of_final_designs={sampling.designs_per_job}",
        "--set", f"max_trajectories={sampling.max_trajectories}",
        # Offset by the task id, so two tasks of one run provably sample
        # different trajectories rather than differing by chance.
        "--set", f"campaign_seed={sampling.seed_base + task_id}",
        # Explicit, though it is BC2's default: a fresh task directory has
        # nothing to resume, and saying so means a rerun into a directory that
        # does have output continues it deliberately rather than by default.
        "--set", "resume=true",
    ]

    node_tmp = _node_tmp(model, manifest, task_id)
    return LaunchSpec(
        argv=tuple(argv),
        env=_bindcraft2_environment(model, node_tmp),
        # `--pwd` does not create the directory, and neither TMPDIR nor the
        # campaign folder's parent is created by anything else.
        mkdirs=(task_dir, node_tmp),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _targets(manifest: RunManifest) -> str:
    """The campaign target block this run was planned with, from the manifest.

    Read back rather than rebuilt, so a launch cannot derive an epitope
    different from the one the run recorded. Serialized with sorted keys so the
    command is identical for identical plans, which is what makes a launch
    comparable across reruns.
    """
    target = manifest.workflow.get("bindcraft2_target")
    if not target:
        raise ValueError(
            f"run {manifest.run_id} records no bindcraft2_target; it was planned "
            "before the target block was stored and cannot be launched from"
        )
    return json.dumps(target, sort_keys=True, separators=(",", ":"))


def _node_tmp(model: BindCraft2Config, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-bindcraft2-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _bindcraft2_environment(model: BindCraft2Config, node_tmp: Path) -> dict[str, str]:
    """Node-local scratch, the GPUs `--cleanenv` would hide, and the worker cap.

    Nothing here reaches the network and no weight cache needs redirecting: the
    AlphaFold parameters and all three ProteinMPNN variants are baked into the
    image, and compiled graphs go under the campaign's own output folder because
    `--nv` gives the container an `nvidia-smi` to name the card with.
    """
    environment = {
        # Must not be on Lustre; see docs/known-issues.md section 2.1.
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
    }
    # `--cleanenv` drops the variable Slurm sets to name the allocated devices,
    # and BC2 reads exactly that variable to decide how many workers to fan out
    # across. Without it the campaign finds no card at all. See
    # docs/known-issues.md section 2.6.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    if model.sampling.workers_per_gpu is not None:
        environment["SINGULARITYENV_BINDCRAFT_WORKERS_PER_GPU"] = str(
            model.sampling.workers_per_gpu
        )
    return environment
