"""Render one task's Genie 3 experiment config, then run it.

Genie 3's CLI takes a config file and nothing else: `--num-devices`,
`--log-dir` and the sharding flags are the only overrides it has, and the
output root, the seed and the sample count are all config keys. A run that fans
out over tasks needs a different value for each of those three per task, so
something has to write a config per task. This does, from the template the
harness archived, and then runs Genie 3 on what it wrote.

Everything else here is a container fix that has to happen next to the process
it fixes:

* **ColabFold's parameter cache.** `colabfold` resolves its data directory from
  `XDG_CACHE_HOME`, which must be writable and so cannot be the image's own.
  Pointed at an empty directory it starts downloading ~4 GB of AF2 parameters
  that are already in the image; `GENIE3_RUNTIME_CACHE` does not help, because
  the image expands it before Singularity injects user environment. Symlinking
  the image's parameters into the writable cache is what makes an offline run
  offline.
* **The working directory.** ProteinMPNN, IPSAE, TM-align and DSSP all have
  repository-relative paths, so Genie 3 only works from `/opt/genie3` — which
  is why the image's runscript hard-`cd`s there, and why this does too.

Run inside genie3.sif:

    singularity exec --cleanenv --nv <binds> genie3.sif \
        /opt/conda/envs/genie3/bin/python run_genie3.py \
            --template <archived experiment.yaml> --config-out <task>/experiment.yaml \
            --rootdir <task> --seed 7 --n-sample 1 \
            --log-dir <task>/genie3_logs --cache <task>/_cache --num-devices 1

Output contract, consumed by `bindocracy.tools.genie3.adapter`:

    <rootdir>/<selection>/results/info.csv   one row per design per AF2 model
    <rootdir>/<selection>/pdbs/             the diffused backbones
    <rootdir>/<selection>/structures/       the refolded complexes

No `status.json`: this knows nothing its exit code does not, and the harness
records the outcome around the process.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Inside the image. The parameters are present but unreachable at the default
# XDG_CACHE_HOME, which is read-only.
CONTAINER_PARAMS = Path("/opt/genie3/packages/.cache/xdg-cache/colabfold/params")
# One file that proves the parameters are really there, rather than an empty
# directory that would send colabfold to the network.
PARAMS_MARKER = "params_model_1_multimer_v3.npz"
# ProteinMPNN, IPSAE, TM-align and DSSP are all reached by repository-relative
# paths, so Genie 3 runs from here and nowhere else.
REPO_ROOT = Path("/opt/genie3")


def render(template: dict, *, rootdir: str, seed: int, n_sample: int) -> dict:
    """The authored experiment with this task's three harness-owned keys set.

    Nothing else is touched. The harness refuses a template that sets any of
    these itself, so there is never an authored value being overwritten here.
    """
    experiment = copy.deepcopy(template)
    experiment.setdefault("experiment", {})["seed"] = seed
    experiment.setdefault("paths", {})["rootdir"] = rootdir
    generation = experiment.setdefault("generation", {})
    generation.setdefault("dataset", {})["n_sample"] = n_sample
    return experiment


def prepare_cache(cache: Path) -> Path:
    """Point a writable XDG cache at the AF2 parameters inside the image."""
    colabfold = cache / "colabfold"
    colabfold.mkdir(parents=True, exist_ok=True)
    params = colabfold / "params"
    if params.is_symlink() or params.exists():
        params.unlink()
    params.symlink_to(CONTAINER_PARAMS)
    if not (params / PARAMS_MARKER).is_file():
        raise SystemExit(
            f"AF2 parameters are not visible at {params}: {PARAMS_MARKER} is "
            "missing. Without them colabfold would try to download 4 GB from a "
            "node with no route to the internet."
        )
    return params


def genie3_command(config: Path, log_dir: Path, num_devices: int) -> list[str]:
    executable = shutil.which("genie3")
    if executable is None:
        raise SystemExit(
            "genie3 is not on PATH. This driver runs inside genie3.sif, where it "
            "lives in /opt/conda/envs/genie3/bin."
        )
    return [executable, "run", "--config", str(config), "--log-dir", str(log_dir),
            "--num-devices", str(num_devices), "--verbose"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, help="Archived experiment YAML.")
    parser.add_argument("--config-out", required=True, help="Where to write this task's config.")
    parser.add_argument("--rootdir", required=True, help="This task's Genie 3 output root.")
    parser.add_argument("--log-dir", required=True, help="Genie 3's own run logs.")
    parser.add_argument("--cache", required=True, help="Writable XDG_CACHE_HOME.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-sample", type=int, required=True, help="Backbones for this task.")
    parser.add_argument("--num-devices", type=int, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    rootdir = Path(args.rootdir).resolve()
    log_dir = Path(args.log_dir).resolve()
    cache = Path(args.cache).resolve()
    config_out = Path(args.config_out).resolve()
    for directory in (rootdir, log_dir, cache, config_out.parent):
        directory.mkdir(parents=True, exist_ok=True)

    template = yaml.safe_load(Path(args.template).read_text())
    if not isinstance(template, dict):
        raise SystemExit(f"experiment template is not a mapping: {args.template}")
    experiment = render(
        template, rootdir=str(rootdir), seed=args.seed, n_sample=args.n_sample
    )
    config_out.write_text(yaml.safe_dump(experiment, sort_keys=False))

    prepare_cache(cache)

    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(cache)

    command = genie3_command(config_out, log_dir, args.num_devices)
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
