"""Build the command for one optimization task. Submission is Snakemake's job.

The wrapper driver runs inside the user's container and the user's script runs
inside the same one. That is deliberate: the wrapper imports nothing but the
standard library, precisely so it can run in whatever image the optimizer needs
without that image having to know about this project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bindocracy.config.models import GeneralConfig
from bindocracy.runs.launch import LaunchSpec, slurm_resources, task_of
from bindocracy.runs.manifest import RunManifest
from bindocracy.tools.optimize.config import OptimizeConfig

CHILDREN_FILE = "children.jsonl"
# Written beside the task so the driver can be handed plan-time lookups without
# a database connection inside the container.
RESOLVED_FILE = "resolved.json"


def optimize_launch_spec(
    general: GeneralConfig, model: OptimizeConfig, manifest: RunManifest, task_id: int
) -> LaunchSpec:
    task = task_of(manifest, task_id)
    run_dir = manifest.directory
    driver = run_dir / manifest.provenance["driver"].path
    archived_script = manifest.provenance.get("optimizer_script")
    script = run_dir / archived_script.path if archived_script else model.script
    task_dir = run_dir / task.directory

    # From the manifest, never the live config: a resumed task must not be
    # redirected at a design set that has been rebuilt under the same path.
    design_fasta = manifest.workflow["design_set_fasta"]
    design_manifest = manifest.workflow["design_set_manifest"]

    work = task_dir / "work"
    structures = task_dir / "structures"
    resolved = _write_resolved(manifest, task_dir)

    argv: list[str] = []
    if model.runtime.container is not None:
        argv.extend(("singularity", "exec", "--cleanenv", "--nv"))
        if model.runtime.dev_source is not None:
            # Same bind the scorer uses, and the reason `dev_source_sha256` is
            # recorded: with the image's own copy replaced, the container
            # digest no longer determines the result.
            argv.extend(("--bind", f"{model.runtime.dev_source}:/opt/mosaic/src/mosaic:ro"))
        argv.extend((str(model.runtime.container), model.runtime.container_python))
    else:
        # `sys.executable` is not available inside a container, and there is no
        # bare `python` on this cluster's PATH. On the host path the harness's
        # own interpreter is what the script should see.
        import sys

        argv.append(sys.executable)

    argv.extend((
        str(driver),
        "--design-set", str(design_fasta),
        "--design-set-manifest", str(design_manifest),
        "--script", str(script),
        "--target-fasta", str(general.target.sequence_fasta),
        "--target-chain", general.target.chain_id,
        "--resolved", str(resolved),
        "--work-dir", str(work),
        "--structure-dir", str(structures),
        "--save-dir", str(task_dir),
        "--shard", str(task.task_id),
        "--num-shards", str(len(manifest.tasks)),
        "--task-id", str(task.task_id),
        "--seed", str(model.seed),
        "--max-children", str(model.max_children),
        "--length-delta", str(model.length_delta),
        "--timeout-seconds", str(model.timeout_seconds),
        "--inputs-wanted", ",".join(model.inputs),
        "--declared-metrics", ",".join(sorted(model.metrics)),
    ))
    if general.target.msa is not None:
        argv.extend(("--target-msa", str(general.target.msa)))
    if general.target.structure_pdb is not None:
        argv.extend(("--target-structure", str(general.target.structure_pdb)))
    # 1-based FASTA positions, resolved and range-checked at PLAN time by
    # `preflight._resolve_hotspots` and carried in the manifest. The container
    # is never handed author numbering, because it has no way to map it.
    hotspots = manifest.workflow.get("hotspots") or []
    if hotspots:
        argv.extend(("--hotspots", ",".join(str(spot) for spot in hotspots)))
    if model.args:
        argv.extend(("--script-args", *model.args))

    return LaunchSpec(
        argv=tuple(argv),
        env=_environment(model, task_dir),
        resources=slurm_resources(general.cluster, model.resources),
        log=run_dir / task.log,
        outputs=(run_dir / task.designs, run_dir / task.status),
        mkdirs=(work, structures),
    )


def _write_resolved(manifest: RunManifest, task_dir: Path) -> Path:
    """Plan-time lookups, as a file the container can read.

    Poses and parent metrics are resolved from the database when the run is
    planned (see `preflight.py`). The container has no database connection and
    should not have one -- a driver that could read the campaign could also be
    handed a different answer than the plan recorded.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / RESOLVED_FILE
    payload = {
        "structures": manifest.workflow.get("parent_structures") or {},
        "metrics": manifest.workflow.get("parent_metrics") or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _environment(model: OptimizeConfig, task_dir: Path) -> dict[str, str]:
    """`SINGULARITYENV_*` survives `--cleanenv`, which is what makes this work."""
    environment: dict[str, str] = {}
    if model.runtime.scratch is not None:
        scratch = Path(model.runtime.scratch) / task_dir.name
        # Node-local and job-private. Every cache a folding library reaches for
        # goes here rather than into a fresh /tmp tree it never removes.
        for variable in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            environment[f"SINGULARITYENV_{variable}"] = str(scratch)
        # known-issues.md 2.5: a JAX cache path that exists only inside one
        # image is how the AF3 run died. Pinned to the task, which exists.
        environment["SINGULARITYENV_JAX_COMPILATION_CACHE_DIR"] = str(scratch / "jax")
    # known-issues.md 2.6: --cleanenv discards the variable SLURM uses to name
    # the allocated GPU, and the container then sees every device on the node.
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices:
        environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = devices
    return environment
