"""Render one task's BindCraft settings, then run BindCraft on what it wrote.

BindCraft's CLI takes three JSON paths and a few behaviour flags. The output
directory, design count, epitope and trajectory budget are all keys *inside*
those documents, and its plot flags can disable but not enable an authored
profile. A run that fans out over tasks therefore needs a rendered settings
pair per task. This writes one from the templates the harness archived and
then runs BindCraft on it.

The rendered files keep their templates' names, so the `TargetSettings` and
`AdvancedSettings` columns BindCraft stamps on every row name the authored
documents rather than something invented here. (BindCraft takes the stem with
`basename.split('.')[0]`, so a template filename with a dot in it is truncated;
that is its behaviour, not ours.)

Two container fixes live here because they have to happen next to the process
they fix:

* **File descriptors.** BindCraft raises its own soft limit but can only reach
  the inherited hard one. Below 65,536 the JAX/OpenMM combination runs out of
  descriptors partway through a campaign -- hours in, with the failure looking
  like an unrelated crash.
* **The OpenCL JIT's noise.** The relax step emits `Failed to read file:
  /tmp/dep-<hex>.d` for essentially every kernel it compiles. It is harmless
  and it buries the real log, so it is dropped here rather than left for
  whoever reads the log afterwards.

What this deliberately does not do is seed anything. BindCraft draws each
trajectory's seed and length from numpy's unseeded global RNG
(`np.random.randint` in `bindcraft.py`) and has no flag that changes it, so a
task is not reproducible and two tasks differ by chance rather than by design.
The plan records `reproducible: false` rather than papering over it with a
wrapper that reimplements the entry point.

Run inside freebindcraft.sif:

    singularity exec --cleanenv --nv --pwd <task> freebindcraft.sif \
        bindcraft-python run_freebindcraft.py \
            --target-template <archived target.json> \
            --advanced-template <archived advanced.json> \
            --filters <archived filters.json> \
            --settings-dir <task> --design-path <task>/bindcraft \
            --final-designs 4 --max-trajectories 8 --hotspots "" \
            --rank-by i_pTM

Output contract, consumed by `bindocracy.tools.freebindcraft.adapter`:

    <design-path>/mpnn_design_stats.csv         one row per fully scored design
    <design-path>/rejected_mpnn_full_stats.csv  one row per rejected design
    <design-path>/final_design_stats.csv        the accepted designs, ranked
    <design-path>/trajectory_stats.csv          one row per successful trajectory
    <design-path>/failure_csv.csv               one row of failure counters
    <design-path>/{Accepted,Rejected}/          the best model of each design

No `status.json`: this knows nothing its exit code does not. Whether a task
stopped because it had enough designs or because it ran out of trajectories is
visible in the output itself, and collection reads it there.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import resource
import subprocess
import sys
from pathlib import Path

# The BindCraft wrapper inside the image: `python -u bindcraft.py "$@"`.
BINDCRAFT = "/usr/local/bin/bindcraft"

# Below this the JAX/OpenMM combination exhausts descriptors partway through a
# campaign. Only the hard limit matters; the soft one is raised here.
MIN_OPEN_FILES = 65536

# One line per JIT-compiled OpenCL kernel, and there are thousands.
_NOISE = re.compile(r"Failed to read file: .*/dep-[0-9a-fA-F]+\.d")


def render_target(template: dict, *, design_path: str, final_designs: int, hotspots: str) -> dict:
    """The authored target with this task's three harness-owned keys set.

    Nothing else is touched. The harness refuses a template that sets any of
    these itself, so there is never an authored value being overwritten here.
    """
    target = copy.deepcopy(template)
    target["design_path"] = design_path
    target["number_of_final_designs"] = final_designs
    # Always written, including as the empty string. BindCraft turns "" into
    # `hotspot=None` and designs against the whole surface; leaving the key out
    # is a KeyError. Either way the campaign, not this file, decides.
    target["target_hotspot_residues"] = hotspots
    return target


def render_advanced(
    template: dict,
    *,
    max_trajectories: int,
    save_plots: bool,
    save_animations: bool,
) -> dict:
    """The authored profile with this task's harness-owned runtime values.

    It caps *successful* hallucinations: BindCraft counts the PDBs in
    `Trajectory/Relaxed/`, and trajectories that abort as clashing or
    low-confidence are moved aside and never counted. The two output toggles
    are written in both directions; BindCraft's CLI can disable them but has
    no corresponding flags that enable a profile where they are false.
    """
    advanced = copy.deepcopy(template)
    advanced["max_trajectories"] = max_trajectories
    advanced["save_design_trajectory_plots"] = save_plots
    advanced["save_design_animations"] = save_animations
    return advanced


def raise_open_files(minimum: int = MIN_OPEN_FILES) -> int:
    """Raise the soft descriptor limit to the hard one, or refuse to start."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < minimum:
        raise SystemExit(
            f"the inherited hard file-descriptor limit is {hard}, below the "
            f"{minimum} BindCraft needs. It cannot be raised from inside the "
            "job, and running anyway fails hours in with an unrelated-looking "
            "error."
        )
    target = hard if hard == resource.RLIM_INFINITY else max(soft, min(hard, minimum))
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return target


def bindcraft_command(args: argparse.Namespace, settings: Path, advanced: Path) -> list[str]:
    """The BindCraft invocation for this task.

    `--no-pyrosetta` is unconditional: PyRosetta is not installed in this
    image, and without the flag BindCraft prints that it fell back and carries
    on -- so the flag is what makes the mode a decision rather than an
    accident.
    """
    command = [
        BINDCRAFT,
        "--settings",
        str(settings),
        "--advanced",
        str(advanced),
        "--filters",
        str(Path(args.filters).resolve()),
        "--no-pyrosetta",
        "--rank-by",
        args.rank_by,
    ]
    if not args.plots:
        command.append("--no-plots")
    if not args.animations:
        command.append("--no-animations")
    return command


def run(command: list[str], environment: dict[str, str]) -> int:
    """Run BindCraft, dropping the OpenCL JIT's per-kernel noise from the log."""
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if not _NOISE.search(line):
            sys.stdout.write(line)
            sys.stdout.flush()
    return process.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FreeBindCraft task.")
    parser.add_argument("--target-template", required=True, help="Archived target JSON.")
    parser.add_argument("--advanced-template", required=True, help="Archived advanced JSON.")
    parser.add_argument("--filters", required=True, help="Archived filter set JSON.")
    parser.add_argument(
        "--settings-dir", required=True, help="Where this task's rendered settings are written."
    )
    parser.add_argument(
        "--design-path", required=True, help="This task's BindCraft output directory."
    )
    parser.add_argument(
        "--final-designs",
        type=int,
        required=True,
        help="number_of_final_designs: accepted designs, not candidates.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        required=True,
        help="Successful hallucinations before the loop stops.",
    )
    parser.add_argument(
        "--hotspots", required=True, help="target_hotspot_residues; empty means no epitope."
    )
    parser.add_argument("--rank-by", required=True, choices=("i_pTM", "ipSAE"))
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--animations", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    design_path = Path(args.design_path).resolve()
    settings_dir = Path(args.settings_dir).resolve()
    design_path.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)

    target_template = Path(args.target_template)
    advanced_template = Path(args.advanced_template)
    settings = settings_dir / target_template.name
    advanced = settings_dir / advanced_template.name

    settings.write_text(
        json.dumps(
            render_target(
                json.loads(target_template.read_text()),
                # BindCraft joins this with every output name, so the trailing
                # separator its own examples carry is neither required nor harmful.
                design_path=str(design_path),
                final_designs=args.final_designs,
                hotspots=args.hotspots,
            ),
            indent=2,
        )
        + "\n"
    )
    advanced.write_text(
        json.dumps(
            render_advanced(
                json.loads(advanced_template.read_text()),
                max_trajectories=args.max_trajectories,
                save_plots=args.plots,
                save_animations=args.animations,
            ),
            indent=2,
        )
        + "\n"
    )

    print(f"open files: soft limit raised to {raise_open_files()}", flush=True)

    return run(bindcraft_command(args, settings, advanced), dict(os.environ))


if __name__ == "__main__":
    sys.exit(main())
