#!/usr/bin/env python3
"""Re-diffuse an existing binder's backbone with RFdiffusion, in two stages.

The idea, in the operator's words: **optimize the interface, then lock it and
optimize the remaining parts.** RFdiffusion has no loss terms to weight, so
"optimize" here means re-noise and re-denoise, and "lock" means hold residues
fixed as a motif. Both map onto primitives RFdiffusion already has:

    --stage interface   partial diffusion. The whole binder is re-noised to
                        timestep `--partial-t` and re-denoised against the
                        target with hotspots set. Re-samples the interface
                        while staying in the parent's neighbourhood.

    --stage scaffold    motif scaffolding. The binder residues that CONTACT
                        the target are carried through as fixed segments;
                        everything else is regenerated at the same length.

Run the two as two runs of this kit, `interface` first. Stage-1 children are
ordinary designs with a `parent_design_id`, so stage 2 takes them as parents
and the lineage is a chain in the database rather than a note in a README.

**This kit runs on the HARNESS interpreter, not inside a container.** It shells
out to two images, because neither can do the whole job: `rfd.sif` holds
RFdiffusion and no sequence designer, and RFdiffusion emits poly-glycine
backbones with no sequence at all. The contract requires a sequence per child,
so a separate inverse-folding step is not optional here -- it is what turns a
backbone into a design.

**Two hard operational facts, both logged.**

* `rfd.sif` **cannot run on an H100.** It carries torch 1.12.1+cu116, whose
  `arch_list` stops at sm_86 with no PTX to JIT forward from, so every CUDA op
  fails -- verified down to a bare host-to-device tensor copy. Schedule this on
  A100.
* Without `ppi.hotspot_res`, `model_runners.py` silently selects the
  **monomer** checkpoint and you get plausible backbones designed against
  nothing. This kit always passes `inference.ckpt_override_path` explicitly, so
  that cannot happen even if the hotspot list is empty.

**Status: NOT GPU-verified, and the contig strings are the risk.** The contract
plumbing is exercised by `--dry-run`, and the invocation follows
`docs/containers/rfdiffusion.md` and `docs/known-issues.md` sections 1.1 and
2.8. But nobody has run this file, and RFdiffusion's contig grammar is
position-sensitive in ways no dry run can check. Run ONE parent, read the
output PDB, and confirm the fixed segments are actually fixed before queuing a
set.

Check the I/O first -- no GPU, no containers, seconds:

    python scripts/validate_optimizer.py \
        --script examples/kits/rfd-refine/optimizer.py \
        --design-set sets/<digest>.json --target-fasta target.fasta \
        --declare n_binder_contacts:none --declare n_fixed:none \
        --declare n_substitutions:none --declare seq_identity:none \
        --declare mpnn_score:min --declare stage_id:none \
        --max-children 4 --script-args --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from bindocracy_io import RejectCandidate, run_optimization

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# Recorded on every child so the two stages stay separable in one table.
STAGE_IDS = {"interface": 0.0, "scaffold": 1.0}

# Without this the monomer checkpoint is selected silently -- known-issues 1.1.
COMPLEX_CKPT = "/app/RFdiffusion/models/Complex_base_ckpt.pt"


# ---------------------------------------------------------------------------
# The contract. Nothing above the next banner is about diffusion.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_IDS),
        default="interface",
        help="interface = partial diffusion; scaffold = fix the interface, rebuild the rest.",
    )
    parser.add_argument(
        "--partial-t",
        type=int,
        default=20,
        help=(
            "Timestep the binder is re-noised to for --stage interface. Low "
            "values stay near the parent; high values approach a fresh design "
            "and stop being refinement."
        ),
    )
    parser.add_argument("--diffuser-t", type=int, default=50,
                        help="Denoising steps for --stage scaffold.")
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=5.0,
        help=(
            "Heavy-atom distance defining which binder residues count as "
            "interface, and so which ones --stage scaffold holds fixed."
        ),
    )
    parser.add_argument("--num-designs", type=int, default=4,
                        help="Backbones per parent. Also the child count.")
    parser.add_argument("--mpnn-samples", type=int, default=8,
                        help="Sequences sampled per backbone; the best by MPNN score is kept.")
    parser.add_argument("--mpnn-temperature", type=float, default=0.1)

    # Fallbacks for the declared requirements, used only when
    # context['dependencies'] does not carry them.
    parser.add_argument("--rfd-sif", default="")
    parser.add_argument("--mpnn-sif", default="")
    parser.add_argument("--mpnn-python", default="python",
                        help="Interpreter inside the inverse-folding image.")

    parser.add_argument("--dry-run", action="store_true")
    return parser


def dependency(context: dict, name: str, fallback: str) -> str:
    """The path the harness resolved for a requirement this kit declares.

    `kit.yaml` names what this optimizer needs; the campaign says where those
    things are. A script that read a path out of its own arguments instead
    would be claiming to know a filesystem it has never seen, and a missing
    dependency would surface as a subprocess failing after the GPU was
    allocated rather than as a refusal at preflight.
    """
    resolved = (context.get("dependencies") or {}).get(name)
    return resolved or fallback


def main() -> int:
    setup: dict = {}

    def optimize_parent(parent, context, args):
        if not setup:
            setup["rfd"] = dependency(context, "rfdiffusion", args.rfd_sif)
            setup["mpnn"] = dependency(context, "proteinmpnn", args.mpnn_sif)
            structure_dir = Path(context["structure_dir"])
            (structure_dir / "poses").mkdir(parents=True, exist_ok=True)
            setup["structure_dir"] = structure_dir
            setup["work"] = Path(context["work_dir"])
            setup["work"].mkdir(parents=True, exist_ok=True)
            # 1-based positions in the target's FASTA. RFdiffusion wants them
            # as chain-letter labels against the chain it is handed, which is
            # why they are formatted rather than passed through.
            setup["hotspots"] = list(context.get("hotspots") or [])
            setup["target"] = context["target_sequence"]

        index = parent["index"]
        sequence_in = parent["sequence"]

        unknown = sorted(set(sequence_in) - set(CANONICAL))
        if unknown:
            raise RejectCandidate(f"parent has non-canonical residue(s) {unknown}")

        pose = parent.get("structure")
        if not pose and not args.dry_run:
            # `inputs: [sequence, structure]` means the run was planned over
            # folded parents. A row without one is a planning error surfacing
            # here, not a biological refusal -- but refusing keeps the shard
            # alive and names the design. Exempt under --dry-run, whose whole
            # job is to exercise the row shapes without real inputs: the
            # validator's --sequences mode supplies no pose, and a kit that
            # refused every parent there would prove nothing.
            raise RejectCandidate("no parent pose; this kit re-diffuses a backbone")

        started = time.time()
        if args.dry_run:
            children = _dry_run(sequence_in, context["seed"] + index, args)
        else:
            children = diffuse(
                sequence_in,
                pose=Path(pose),
                target_sequence=setup["target"],
                hotspots=setup["hotspots"],
                seed=context["seed"] + index,
                work=setup["work"] / f"parent-{index}",
                rfd_sif=setup["rfd"],
                mpnn_sif=setup["mpnn"],
                max_children=int(context.get("max_children") or args.num_designs),
                args=args,
            )

        if not children:
            raise RejectCandidate(
                f"RFdiffusion produced no usable backbone at stage {args.stage}"
            )

        for ordinal, child in enumerate(children):
            sequence = child["sequence"]
            substitutions = sum(
                1 for was, now in zip(sequence_in, sequence) if was != now
            )
            row = {
                "sequence": sequence,
                "metrics": {
                    "n_binder_contacts": float(child["n_interface"]),
                    "n_fixed": float(child["n_fixed"]),
                    "n_substitutions": float(substitutions),
                    "seq_identity": 1.0 - substitutions / max(1, len(sequence_in)),
                    "mpnn_score": float(child["mpnn_score"]),
                    "stage_id": STAGE_IDS[args.stage],
                },
                "seconds": round((time.time() - started) / len(children), 2),
            }
            # Relative to context['structure_dir']; an absolute path is refused
            # so a run directory can be moved.
            if child.get("pose"):
                relative = f"poses/{index}-{ordinal}.pdb"
                shutil.copyfile(child["pose"], setup["structure_dir"] / relative)
                row["structure"] = relative
            yield row

    return run_optimization(optimize_parent, parser=build_parser())


# ---------------------------------------------------------------------------
# The diffusion. Everything RFdiffusion-specific is below here.
# ---------------------------------------------------------------------------


def _load(path: Path):
    """A pose as biotite atoms, from either .cif or .pdb.

    Chai-1 and AlphaFold 3 write .cif; RFdiffusion reads only .pdb, so a
    conversion is unavoidable and is done here rather than assumed upstream.
    """
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    import biotite.structure as struc

    if path.suffix == ".cif":
        atoms = pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1)
    else:
        atoms = pdb.get_structure(pdb.PDBFile.read(str(path)), model=1)
    # Waters, ions and hydrogens would corrupt both the per-chain residue
    # counts that identify the target and the "heavy-atom" distances below.
    return atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]


def _split_chains(atoms, target_length: int):
    """Which chain is the target, by LENGTH rather than by letter.

    Chain letters are not portable between drivers: the mosaic driver builds
    [binder, target] so its target is chain B, while the Chai-1 and AlphaFold 3
    drivers write the target as chain A. The target's residue count is known
    from the context and is unambiguous.
    """
    import numpy as np

    chains = {}
    for chain_id in np.unique(atoms.chain_id):
        mask = atoms.chain_id == chain_id
        chains[str(chain_id)] = (mask, len(np.unique(atoms.res_id[mask])))

    target = [c for c, (_, n) in chains.items() if n == target_length]
    if len(target) != 1:
        raise RejectCandidate(
            f"cannot identify the target chain: lengths "
            f"{ {c: n for c, (_, n) in chains.items()} } against a "
            f"{target_length}-residue target"
        )
    target_id = target[0]
    binder = [c for c in chains if c != target_id]
    if len(binder) != 1:
        raise RejectCandidate(f"expected exactly one binder chain, found {binder}")
    return target_id, binder[0], chains


def _interface_positions(atoms, target_id, binder_id, cutoff: float) -> list[int]:
    """1-based binder positions with any heavy atom within `cutoff` of the target.

    Heavy-atom rather than CA-CA: a CA cutoff generous enough to catch a real
    side-chain contact also catches residues that are merely nearby, and what
    stage 2 must not destroy is the atoms that touch.
    """
    import numpy as np

    target = atoms[atoms.chain_id == target_id]
    binder = atoms[atoms.chain_id == binder_id]
    distances = np.linalg.norm(
        binder.coord[:, None, :] - target.coord[None, :, :], axis=-1
    )
    nearest = distances.min(axis=1)

    ordered = list(dict.fromkeys(int(r) for r in binder.res_id))
    position_of = {res_id: i + 1 for i, res_id in enumerate(ordered)}
    hit = {
        position_of[int(res_id)]
        for res_id, distance in zip(binder.res_id, nearest)
        if distance <= cutoff
    }
    return sorted(hit), ordered


def _segments(positions: list[int]) -> list[tuple[int, int]]:
    """Consecutive runs, so `[3,4,5,9]` becomes `[(3,5),(9,9)]`.

    RFdiffusion's contig grammar names ranges, not sets, so a fixed motif has
    to be expressed as runs.
    """
    runs: list[tuple[int, int]] = []
    for position in positions:
        if runs and position == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], position)
        else:
            runs.append((position, position))
    return runs


def _contigs(stage, *, target_id, binder_id, target_ids, binder_ids, fixed):
    """The contig string, and the count of residues actually held fixed.

    ⚠ RFdiffusion's contig grammar names residues by their PDB NUMBERING, not
    by position. `target_ids` / `binder_ids` are the real res_ids in file
    order, so `A1-201` is only correct when the pose happens to be numbered
    from 1 -- a predicted pose is typically numbered 0..200, where that string
    names a residue that does not exist and omits residue 0. `fixed` is a list
    of 1-based ORDINALS and is translated here, once, at the only boundary
    where the two conventions meet.
    """
    target_segment = f"{target_id}{target_ids[0]}-{target_ids[-1]}"
    binder_length = len(binder_ids)

    if stage == "interface":
        # Partial diffusion re-noises what it is given, so the contig must
        # describe the input exactly: the whole binder, same length, no
        # generated segments.
        binder_segment = f"{binder_id}{binder_ids[0]}-{binder_ids[-1]}"
        return f"[{target_segment}/0 {binder_segment}]", binder_length

    # Scaffolding: fixed runs stay, the gaps between them are regenerated at
    # exactly their original length so the child matches its parent and
    # `changes_length: false` holds.
    pieces: list[str] = []
    held = 0
    cursor = 1
    for start, end in _segments(fixed):
        if start > cursor:
            gap = start - cursor
            pieces.append(f"{gap}-{gap}")
        pieces.append(f"{binder_id}{binder_ids[start - 1]}-{binder_ids[end - 1]}")
        held += end - start + 1
        cursor = end + 1
    if cursor <= binder_length:
        gap = binder_length - cursor + 1
        pieces.append(f"{gap}-{gap}")
    return f"[{target_segment}/0 {'/'.join(pieces)}]", held


def _run(command: list[str], where: Path) -> None:
    """One subprocess, with its tail kept for the failure message.

    Not wrapped in a broad except: a container that will not start is a bug in
    the binding, not a parent this optimizer declined.
    """
    done = subprocess.run(
        command, cwd=str(where), capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError(
            f"{command[0]} exited {done.returncode}:\n" + "\n".join(tail)
        )


def diffuse(
    parent_sequence: str,
    *,
    pose: Path,
    target_sequence: str,
    hotspots: list[int],
    seed: int,
    work: Path,
    rfd_sif: str,
    mpnn_sif: str,
    max_children: int,
    args: argparse.Namespace,
) -> list[dict]:
    """One parent through RFdiffusion and then through inverse folding.

    `max_children` is the host's ceiling: `contract.py` rejects and counts every
    row whose ordinal reaches it, so asking RFdiffusion for more backbones than
    that would burn GPU on designs the database throws away without warning.
    """
    import biotite.structure.io.pdb as pdb

    if not rfd_sif:
        raise RuntimeError(
            "no `rfdiffusion` binding and no --rfd-sif; see kit.yaml requires"
        )

    designs = min(args.num_designs, max_children)
    work.mkdir(parents=True, exist_ok=True)
    (work / "schedules").mkdir(exist_ok=True)  # known-issues 2.8: mkdir, not makedirs

    atoms = _load(pose)
    target_id, binder_id, _ = _split_chains(atoms, len(target_sequence))
    interface, binder_ids = _interface_positions(
        atoms, target_id, binder_id, args.contact_cutoff
    )
    target_ids = list(
        dict.fromkeys(int(r) for r in atoms.res_id[atoms.chain_id == target_id])
    )
    if args.stage == "scaffold" and not interface:
        raise RejectCandidate(
            f"no binder residue within {args.contact_cutoff} A of the target; "
            "there is no interface to hold fixed"
        )

    inputs = work / "input.pdb"
    written = pdb.PDBFile()
    written.set_structure(atoms)
    written.write(str(inputs))

    contigs, held = _contigs(
        args.stage,
        target_id=target_id,
        binder_id=binder_id,
        target_ids=target_ids,
        binder_ids=binder_ids,
        fixed=interface,
    )

    command = [
        "singularity", "run", "--cleanenv", "--nv",
        "--bind", f"{work}:/work",
        rfd_sif,
        "inference.input_pdb=/work/input.pdb",
        "inference.output_prefix=/work/out",
        f"inference.num_designs={designs}",
        # Mandatory. Omitting hotspots alone would silently pick the monomer
        # checkpoint; pinning the checkpoint makes that impossible either way.
        f"inference.ckpt_override_path={COMPLEX_CKPT}",
        "inference.schedule_directory_path=/work/schedules",
        f"inference.seed={seed}",
        # QUOTED for the inner `eval`. The chain break `/0 ` contains a space
        # that cannot be formatted away, and an unquoted value would be
        # re-split into two argv entries, destroying Hydra parsing exactly as
        # a ${now:} interpolation does.
        f"'contigmap.contigs={contigs}'",
    ]
    if hotspots:
        # ⚠ context["hotspots"] are 1-BASED POSITIONS IN THE TARGET'S FASTA.
        # RFdiffusion wants PDB residue numbers, and a predicted pose is
        # numbered positionally (0..200 for a 201-residue target). Passing the
        # FASTA position straight through addresses the wrong residue, the
        # complex checkpoint is still selected, and you get plausible backbones
        # conditioned on the wrong epitope -- the exact bug class the harness's
        # own epitope function already paid for once.
        labels = ",".join(f"{target_id}{target_ids[spot - 1]}" for spot in hotspots)
        command.append(f"ppi.hotspot_res=[{labels}]")
    if args.stage == "interface":
        command.append(f"diffuser.partial_T={args.partial_t}")
    else:
        command.append(f"diffuser.T={args.diffuser_t}")

    # The image's runscript `eval`s its arguments, so nothing here may contain
    # a Hydra interpolation such as ${now:...} -- known-issues records that
    # breaking the command line.
    assert not any("${" in piece for piece in command), "Hydra interpolation in argv"
    _run(command, work)

    # Stage 2 locks the interface. RFdiffusion writes designed regions as
    # glycine, so the parent's residue identities are written back into the
    # backbone BEFORE inverse folding and those positions are then pinned --
    # otherwise "fixed" would pin whatever placeholder the diffuser emitted.
    locked = interface if args.stage == "scaffold" else []

    children: list[dict] = []
    for backbone in _numerically(work.glob("out_*.pdb")):
        # ⚠ RE-DERIVE THE CHAINS FROM THE OUTPUT. RFdiffusion assigns output
        # chains in CONTIG order, and _contigs always puts the target first
        # regardless of how the input was ordered -- and the input order itself
        # varies by driver (the mosaic driver writes the target as chain B,
        # Chai-1 and AF3 as chain A). Reusing the input's letters would, for a
        # mosaic-lineage pose, seed the parent's residues into the TARGET and
        # hand MPNN the target as the designed chain.
        out_atoms = _load(backbone)
        out_target, out_binder, _ = _split_chains(out_atoms, len(target_sequence))

        if locked:
            _write_identities(backbone, out_binder, locked, parent_sequence)
        sequence, score = inverse_fold(
            backbone,
            binder_id=out_binder,
            target_id=out_target,
            locked=locked,
            mpnn_sif=mpnn_sif,
            seed=seed,
            work=work,
            args=args,
        )
        if len(sequence) != len(parent_sequence):
            # `changes_length: false` is a claim the host enforces; catching it
            # here names the backbone instead of failing the whole shard.
            continue
        # The lock, checked rather than assumed. If `fix_pos` did not mean what
        # this kit thinks it means, the interface silently drifts and stage 2
        # becomes an expensive rerun of stage 1. NOTE this validates
        # ColabDesign's fix_pos, NOT RFdiffusion's contig interpretation --
        # both sides here are indexed by ordinal. Read the first output PDB to
        # confirm the motif landed where the contig asked.
        drifted = [p for p in locked if sequence[p - 1] != parent_sequence[p - 1]]
        if drifted:
            raise RuntimeError(
                f"{backbone.name}: inverse folding changed {len(drifted)} locked "
                f"position(s) {drifted[:8]}; fix_pos did not hold"
            )
        # The stored pose must carry the sequence it is stored beside: the raw
        # backbone is poly-glycine, and any later geometric metric that reads
        # residue identities off it would return a plausible number about a
        # sequence that does not exist.
        _write_identities(backbone, out_binder, range(1, len(sequence) + 1), sequence)
        children.append({
            "sequence": sequence,
            "mpnn_score": score,
            "n_interface": len(interface),
            "n_fixed": held if args.stage == "scaffold" else 0,
            "pose": backbone,
        })
    return children


def _numerically(paths):
    """`out_0, out_1, out_2, ... out_10`, not `out_0, out_1, out_10, out_2`.

    Lexicographic order would stop child ordinals tracking RFdiffusion's own
    design numbers above nine.
    """
    import re

    def key(path):
        digits = re.findall(r"\d+", path.stem)
        return (int(digits[-1]) if digits else 0, path.stem)

    return sorted(paths, key=key)


def _write_identities(backbone: Path, binder_id: str, positions, sequence: str) -> None:
    """Write residue NAMES into the binder chain at the given 1-based ordinals.

    Used twice, for two reasons. Before inverse folding: ProteinMPNN reads
    backbone geometry plus the residue name at any position it is told to keep,
    and RFdiffusion emits designed regions as glycine, so without this the lock
    would preserve a placeholder rather than the parent's interface. After it:
    so the pose stored beside a child carries that child's sequence.

    Only res_name is set. MPNN reads backbone atoms (N, CA, C, O) and the
    residue label, so side-chain atoms are neither present nor needed.
    """
    import biotite.structure.io.pdb as pdb

    atoms = _load(backbone)
    binder_mask = atoms.chain_id == binder_id
    ordered = list(dict.fromkeys(int(r) for r in atoms.res_id[binder_mask]))
    for position in positions:
        residue_id = ordered[position - 1]
        atoms.res_name[binder_mask & (atoms.res_id == residue_id)] = ONE_TO_THREE[
            sequence[position - 1]
        ]

    written = pdb.PDBFile()
    written.set_structure(atoms)
    written.write(str(backbone))


ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


# Run inside the inverse-folding image, which has ColabDesign and no
# bindocracy. Kept as a string rather than a file so the kit stays two files.
#
# `chain` is passed target-first, so the binder is always part index 1 of the
# returned "/"-joined sequence. `fix_pos` names the whole target chain plus,
# at stage `scaffold`, the binder runs to hold.
MPNN_SCRIPT = r"""
import json, sys
from colabdesign.mpnn import mk_mpnn_model

pdb, chains, fix_pos, out, batch, temperature, seed = sys.argv[1:8]
model = mk_mpnn_model(model_name="v_48_020", backbone_noise=0.0, seed=int(seed))
model.prep_inputs(
    pdb_filename=pdb,
    chain=chains,
    fix_pos=fix_pos or None,
    verbose=False,
)
sampled = model.sample_parallel(batch=int(batch), temperature=float(temperature))
json.dump(
    {
        "seq": [str(s) for s in sampled["seq"]],
        "score": [float(v) for v in sampled["score"]],
    },
    open(out, "w"),
)
"""


def inverse_fold(backbone, *, binder_id, target_id, locked, mpnn_sif, seed, work, args):
    """Backbone -> sequence, through ColabDesign's ProteinMPNN.

    RFdiffusion emits poly-glycine: it designs shape, not sequence. Something
    has to choose residues, and that something is an INVERSE-FOLDING model --
    it reads geometry and writes residues; it does not predict a fold. That is
    why `loss_models` in kit.yaml is `[]` and every folding model in the panel
    stays available to judge these children. A step here that scored candidates
    with a folding model would silently invalidate that claim, which is why the
    best sample is taken on MPNN's OWN score and nothing else is consulted.

    ⚠ The `fix_pos` grammar is the remaining unverified detail. The caller
    checks the returned sequence against the parent at every locked position
    and raises if any drifted, so a wrong guess here fails loudly on the first
    backbone instead of producing a stage-2 run that is really stage 1.
    """
    if not mpnn_sif:
        raise RuntimeError(
            "no `proteinmpnn` binding and no --mpnn-sif. RFdiffusion emits "
            "backbones only, so this kit cannot produce a sequence without one."
        )

    script = work / "mpnn_sample.py"
    script.write_text(MPNN_SCRIPT)
    out = work / f"{backbone.stem}.mpnn.json"

    fix = target_id
    if locked:
        fix += "," + ",".join(
            f"{binder_id}{start}-{end}" for start, end in _segments(locked)
        )

    _run(
        [
            "singularity", "exec", "--cleanenv", "--nv",
            "--bind", f"{work}:{work}",
            mpnn_sif, args.mpnn_python, str(script),
            str(backbone), f"{target_id},{binder_id}", fix, str(out),
            str(args.mpnn_samples), str(args.mpnn_temperature), str(seed),
        ],
        work,
    )

    sampled = json.loads(out.read_text())
    if not sampled.get("seq"):
        raise RuntimeError(f"inverse folding wrote no sequence to {out}")

    # Lowest MPNN score wins. Its own opinion of its own sequence, and the only
    # ranking signal this kit is allowed to use.
    best = min(range(len(sampled["seq"])), key=lambda i: sampled["score"][i])
    parts = sampled["seq"][best].split("/")
    if len(parts) < 2:
        raise RuntimeError(
            f"expected a two-chain sequence from {out}, got {len(parts)} part(s)"
        )
    return parts[1], sampled["score"][best]


def _dry_run(parent_sequence: str, seed: int, args: argparse.Namespace) -> list[dict]:
    """Contract plumbing without singularity, a GPU or a pose.

    The validator checks row shapes and costs seconds; starting RFdiffusion
    costs minutes and an A100 and proves nothing about the rows.
    """
    rng = random.Random(seed)
    alphabet = "ADEFGHIKLMNPQRSTVWY"
    interface = sorted(rng.sample(range(1, len(parent_sequence) + 1),
                                  k=min(12, len(parent_sequence))))
    children = []
    for _ in range(max(1, args.num_designs)):
        mutated = list(parent_sequence)
        movable = [i for i in range(len(mutated)) if (i + 1) not in interface]
        for position in rng.sample(movable, k=min(8, len(movable))):
            mutated[position] = rng.choice(alphabet)
        children.append({
            "sequence": "".join(mutated),
            "mpnn_score": round(rng.uniform(0.8, 1.6), 4),
            "n_interface": len(interface),
            "n_fixed": len(interface) if args.stage == "scaffold" else 0,
            "pose": None,
        })
    return children


if __name__ == "__main__":
    raise SystemExit(main())
