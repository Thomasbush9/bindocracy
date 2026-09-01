"""Build the command for one FreeBindCraft task.

A driver is used for one reason: BindCraft has no override for its output
directory, its design count, or its epitope -- all three live inside the target
JSON -- and no override for the trajectory budget, which lives inside the
advanced JSON. Two tasks sharing an output directory would not race so much as
*resume* each other: the design loop skips any trajectory whose PDB already
exists, so the second task would quietly do the first one's leftovers and both
would report the same designs.

Everything else is `singularity exec`. `--pwd` matters only because BindCraft
resolves relative paths in its settings against the working directory, and
preflight has already refused relative paths; it is set so that a settings file
that somehow carries one resolves inside the task rather than wherever
Snakemake happened to run.
"""

from __future__ import annotations

import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.freebindcraft.config import FreeBindCraftConfig

# The image's own interpreter, as a symlink the image provides for this. The
# runscript (`singularity run`) is BindCraft itself, so a driver needs `exec`.
CONTAINER_PYTHON = "bindcraft-python"

# BindCraft's output tree, under the task directory rather than beside the
# harness's own files. The rendered settings sit in the task directory itself,
# where they are one level up from everything BindCraft writes.
DESIGN_DIR = "bindcraft"

# The five tables BindCraft writes into its design path.
DESIGNS_FILE = f"{DESIGN_DIR}/mpnn_design_stats.csv"
REJECTED_FILE = f"{DESIGN_DIR}/rejected_mpnn_full_stats.csv"
FINAL_FILE = f"{DESIGN_DIR}/final_design_stats.csv"
TRAJECTORY_FILE = f"{DESIGN_DIR}/trajectory_stats.csv"
FAILURE_FILE = f"{DESIGN_DIR}/failure_csv.csv"

# Where the accepted and rejected structures land, and the three trajectory
# outcomes. `Trajectory/Relaxed` is what `max_trajectories` counts.
ACCEPTED_DIR = f"{DESIGN_DIR}/Accepted"
REJECTED_DIR = f"{DESIGN_DIR}/Rejected"
TRAJECTORY_DIRS = {
    "successful": f"{DESIGN_DIR}/Trajectory/Relaxed",
    "clashing": f"{DESIGN_DIR}/Trajectory/Clashing",
    "low_confidence": f"{DESIGN_DIR}/Trajectory/LowConfidence",
}


def freebindcraft_launch_spec(
    general: GeneralConfig,
    model: FreeBindCraftConfig,
    manifest: RunManifest,
    task_id: int,
) -> LaunchSpec:
    """One task: the archived driver, rendering settings and running BindCraft."""
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    task_dir = run_dir / task.directory
    # The archived copies, not the authored ones: the working tree may have
    # moved on since this run was planned.
    driver = run_dir / manifest.provenance["driver"].path
    target = run_dir / manifest.provenance["target"].path
    filters = run_dir / manifest.provenance["filters"].path
    advanced = run_dir / manifest.provenance["advanced"].path
    runtime = model.runtime

    argv = [
        "singularity", "exec", "--cleanenv", "--nv",
        "--pwd", str(task_dir),
        str(runtime.container),
        CONTAINER_PYTHON, str(driver),
        "--target-template", str(target),
        "--advanced-template", str(advanced),
        "--filters", str(filters),
        "--settings-dir", str(task_dir),
        "--design-path", str(task_dir / DESIGN_DIR),
        "--final-designs", str(model.sampling.designs_per_job),
        "--max-trajectories", str(model.sampling.max_trajectories),
        # The campaign epitope, resolved at planning and carried in the
        # manifest so the launch cannot re-derive it differently from what was
        # recorded. Empty is a value: BindCraft reads it as no epitope at all.
        "--hotspots", _hotspots(manifest),
        "--rank-by", runtime.rank_by,
        "--plots" if runtime.save_plots else "--no-plots",
        "--animations" if runtime.save_animations else "--no-animations",
    ]

    node_tmp = _node_tmp(model, manifest, task_id)
    return LaunchSpec(
        argv=tuple(argv),
        env=_freebindcraft_environment(node_tmp),
        # `--pwd` does not create the directory, and OpenMM writes its OpenCL
        # scratch under TMPDIR without creating the root.
        mkdirs=(task_dir, task_dir / DESIGN_DIR, node_tmp),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs,),
    )


def _hotspots(manifest: RunManifest) -> str:
    """The epitope string this run was planned with, from the manifest alone."""
    hotspots = manifest.workflow.get("hotspot_string")
    if hotspots is None:
        raise ValueError(
            f"run {manifest.run_id} records no hotspot_string; it was planned "
            "before the epitope was stored and cannot be launched from"
        )
    return str(hotspots)


def _node_tmp(model: FreeBindCraftConfig, manifest: RunManifest, task_id: int) -> Path:
    """Node-local scratch for one task, unique so two jobs cannot collide."""
    return (
        model.runtime.node_tmp_root
        / f"bindocracy-freebindcraft-{manifest.run_id[:8]}-{task_id:04d}"
    )


def _freebindcraft_environment(node_tmp: Path) -> dict[str, str]:
    """Node-local scratch, and the GPU `--cleanenv` would otherwise hide.

    OpenMM's OpenCL JIT writes a dependency file per compiled kernel under
    TMPDIR, thousands of them over a campaign, and Lustre is the wrong disk for
    that. Everything else this image needs -- AF2 parameters, ProteinMPNN
    weights, DSSP, FASPR, sc-rs -- is inside it, so nothing here reaches the
    network and no cache variable needs redirecting.
    """
    environment = {
        # Must not be on Lustre; see docs/known-issues.md section 2.1.
        "TMPDIR": str(node_tmp),
        "SINGULARITYENV_TMPDIR": str(node_tmp),
    }
    # `--cleanenv` drops the variable Slurm sets to name the allocated device.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
