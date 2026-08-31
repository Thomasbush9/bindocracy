"""Seed Protein-Hunter's MSA cache for one task, then run its Boltz pipeline.

Protein-Hunter's Boltz pipeline has only `--msa_mode {single,mmseqs}`. `single`
folds the target with no alignment at all; `mmseqs` calls
https://api.colabfold.com, which a compute node cannot reach. There is no flag
for a precomputed a3m.

The way through is that `run_mmseqs2` is cache-first: it skips the HTTP call
entirely when `{prefix}_env/out.tar.gz` exists, and skips untarring when the
a3m files are already there. So this writes that cache by hand from the
campaign's own alignment and then asks for `--msa_mode mmseqs`.

Two details that are easy to get wrong, one loudly and one silently:

1.  The parser does `M = int(line[1:].rstrip())` on the FIRST header, so it must
    read `>101` -- a literal `>DIO3` raises ValueError. Only the first one;
    every later header is passed through untouched.
2.  `max_seqs` is hardcoded to 4096 downstream and overrides the caller, so the
    full alignment would be pushed through the MSA module on every one of a few
    hundred predictions. Subsampling to a few hundred costs almost nothing in
    accuracy and saves hours.

The cache is per task because `save_dir` is: two tasks sharing one would race on
the same directory, which is the failure in docs/known-issues.md section 6.1b.

Run inside protein_hunter.sif:

    singularity run --cleanenv --nv protein_hunter.sif python run_boltz_design.py \
        --save-dir <task> --a3m <target.a3m> --max-seqs 512 \
        --name dio3_cut --protein-seqs <SEQUENCE> --msa-mode mmseqs \
        --num-designs 40 --num-cycles 5 ...

Output contract, consumed by `bindocracy.tools.protein_hunter.adapter`:

    <save-dir>/summary_all_runs.csv    one wide row per trajectory
    <save-dir>/summary_high_iptm.csv   one row per (trajectory, cycle) that passed
    <save-dir>/high_iptm_pdb/          the co-folded complexes that passed

No `status.json`: this knows nothing its exit code does not, and the harness
records the outcome around the process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Inside the image; the runscript's `boltz` mode reaches the same file.
PIPELINE = "/opt/protein-hunter/boltz_ph/design.py"
# The binder is chain A, so the single target chain becomes chain B.
TARGET_CHAIN = "B"
DESIGN_DIR = "0_protein_hunter_design"
EMPTY_TARBALL = "out.tar.gz"
UNIREF = "uniref.a3m"
ENVIRONMENTAL = "bfd.mgnify30.metaeuk30.smag30.a3m"


def read_a3m(path: Path) -> list[tuple[str, list[str]]]:
    """Return [(header, [sequence lines])], preserving a3m insertion casing."""
    records: list[tuple[str, list[str]]] = []
    header: str | None = None
    body: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                records.append((header, body))
            header, body = line, []
        elif header is not None and line:
            body.append(line)
    if header is not None:
        records.append((header, body))
    return records


def seed_msa_cache(a3m: Path, env_dir: Path, max_seqs: int) -> Path:
    """Write the ColabFold cache `run_mmseqs2` consults before the network."""
    records = read_a3m(a3m)
    if not records:
        raise SystemExit(f"no sequences parsed from {a3m}")

    # The alignment is ordered by E-value, so a prefix is the natural subsample.
    kept = records[: max(1, max_seqs)]
    env_dir.mkdir(parents=True, exist_ok=True)

    # Presence alone blocks the HTTP call; the content is never read, because
    # the a3m files below already exist.
    (env_dir / EMPTY_TARBALL).write_bytes(b"")
    # Must exist, or the loader tries to untar the empty tarball above.
    (env_dir / ENVIRONMENTAL).write_text("")

    with (env_dir / UNIREF).open("w") as handle:
        for index, (header, body) in enumerate(kept):
            handle.write(">101\n" if index == 0 else f"{header}\n")
            for line in body:
                handle.write(f"{line}\n")

    missing = [name for name in (EMPTY_TARBALL, UNIREF, ENVIRONMENTAL)
               if not (env_dir / name).exists()]
    if missing:
        raise SystemExit(f"MSA cache incomplete in {env_dir}: {', '.join(missing)}")
    print(f"seeded {env_dir}: {len(records)} parsed, {len(kept)} written", flush=True)
    return env_dir


def pipeline_command(args: argparse.Namespace) -> list[str]:
    """The Boltz design pipeline, as the container's own runscript invokes it."""
    command = [
        sys.executable, PIPELINE,
        "--name", args.name,
        "--mode", "binder",
        "--protein_seqs", args.protein_seqs,
        "--msa_mode", args.msa_mode,
        "--num_designs", str(args.num_designs),
        "--num_cycles", str(args.num_cycles),
        "--min_protein_length", str(args.min_protein_length),
        "--max_protein_length", str(args.max_protein_length),
        "--percent_X", str(args.percent_x),
        "--temperature", str(args.temperature),
        "--diffuse_steps", str(args.diffuse_steps),
        "--recycling_steps", str(args.recycling_steps),
        "--high_iptm_threshold", str(args.high_iptm_threshold),
        "--high_plddt_threshold", str(args.high_plddt_threshold),
        "--save_dir", str(args.save_dir),
        "--gpu_id", str(args.gpu_id),
    ]
    if args.omit_aa:
        command += ["--omit_AA", args.omit_aa]
    # An epitope, when the campaign names one. Upstream reads this in three
    # places: a Boltz pocket constraint on generation, a resampling loop that
    # rejects binders which miss it, and a third condition on being kept.
    if args.contact_residues:
        command += [
            "--contact_residues", args.contact_residues,
            "--contact_cutoff", str(args.contact_cutoff),
            "--max_contact_filter_retries", str(args.max_contact_filter_retries),
        ]
        if not args.contact_filter:
            command.append("--no_contact_filter")
    # No --template_path, ever: a value that is not an existing file falls into
    # a branch that fetches from RCSB or AlphaFold and hangs on an offline node.
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True, help="This task's output root.")
    parser.add_argument("--a3m", default=None, help="Alignment to seed the cache from.")
    parser.add_argument("--max-seqs", type=int, default=512)
    parser.add_argument("--name", required=True)
    parser.add_argument("--protein-seqs", required=True, help="The target sequence.")
    parser.add_argument("--msa-mode", required=True, choices=("single", "mmseqs"))
    parser.add_argument("--num-designs", type=int, required=True)
    parser.add_argument("--num-cycles", type=int, required=True)
    parser.add_argument("--min-protein-length", type=int, required=True)
    parser.add_argument("--max-protein-length", type=int, required=True)
    parser.add_argument("--percent-x", type=int, required=True)
    parser.add_argument("--omit-aa", default="")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--diffuse-steps", type=int, required=True)
    parser.add_argument("--recycling-steps", type=int, required=True)
    parser.add_argument("--high-iptm-threshold", type=float, required=True)
    parser.add_argument("--high-plddt-threshold", type=float, required=True)
    parser.add_argument("--contact-residues", default="",
                        help="Comma-separated target residues, chains split by '|'.")
    parser.add_argument("--contact-cutoff", type=float, default=15.0)
    parser.add_argument("--max-contact-filter-retries", type=int, default=6)
    parser.add_argument("--contact-filter", dest="contact_filter",
                        action="store_true", default=True)
    parser.add_argument("--no-contact-filter", dest="contact_filter",
                        action="store_false")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.msa_mode == "mmseqs":
        if not args.a3m:
            raise SystemExit(
                "--msa-mode mmseqs needs --a3m to seed the cache from, or the "
                "pipeline calls api.colabfold.com"
            )
        seed_msa_cache(
            Path(args.a3m),
            save_dir / DESIGN_DIR / f"{TARGET_CHAIN}_env",
            args.max_seqs,
        )

    command = pipeline_command(args)
    print("+ " + " ".join(command[:6]) + " ...", flush=True)
    return subprocess.run(command, cwd=save_dir, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
