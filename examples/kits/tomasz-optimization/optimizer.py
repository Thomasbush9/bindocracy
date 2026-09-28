#!/usr/bin/env python3
"""Refine binders by WRAPPING the DIO3 campaign's own `run_design.py`.

This kit used to reimplement that script's soft-sharp cycling protocol. It no
longer does, and the reason is worth stating because it cost a GPU run to learn:
a reimplementation of a 900-line script silently diverges. The first version
diverged in six places, three of them invisible without a GPU, and the one that
mattered most was this -- `run_design.py --init-noise-random`'s own help text:

    Without this, --init-noise is a PURE FUNCTION of the seed sequence:
    x = (1-w)*seeded + w/n_tokens. Every replica of the same parent therefore
    starts from the IDENTICAL point, so spawning K children to diversify a
    converged design does nothing.

The reimplementation had exactly that bug. Asking for two children per parent
produced two children from the same starting point. So this file is now a
translator, not an optimizer: it turns the harness's contract into a
`run_design.py` invocation and turns that script's output back into child rows.
Nothing scientific happens here.

**Why the JSONL contract is written by hand** rather than through
`bindocracy_io.run_optimization`: that helper calls back once per parent, but
`run_design.py` vmaps B independent trajectories in ONE process and pays its JIT
compile once for the whole batch. Per-parent calls would pay it per parent. So
this reads every parent, groups them by binder length (the script takes a single
`--binder-length`), and invokes the script once per chunk.
`docs/custom-optimization.md` keeps the raw path valid for exactly this case.

**How a trajectory maps back to a parent.** `run_design.py:469` picks a seed
sequence with `seqs[(seed * batch + b) % len(seqs)]`. Writing the chunk's
`--init-fasta` with exactly `batch` entries makes that reduce to `seqs[b]`, for
any seed -- so the mapping is exact rather than inferred. Each parent is
repeated `--replicates` times consecutively, and `--init-noise-random` gives
each repeat its own Dirichlet draw keyed off the seed, which is what makes them
genuinely different children.

**Two traps honoured**, both recorded in `RUN_PARAMETERS.md`:
`designs_seed*.json` `results` is sorted by LOSS, not batch order, and the batch
item is `r["trajectory"]` -- so the join uses that field and never the
enumeration index.

**There is no `start_loss`.** `run_design.py` computes no parent baseline, and
inventing one here by re-running the objective would be a second implementation
of the thing this rewrite exists to stop duplicating. `cycle1_loss` is reported
instead as a within-run reference, and the parent's own quality is already
measured in the database by an independent scorer -- which is the comparison
that matters anyway.

Status: the wrapper is exercised by `--dry-run`; the script it wraps is the one
the source campaign ran for thirty rounds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # The three the harness always passes, in this order.
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)

    parser.add_argument(
        "--design-script",
        required=True,
        help="Absolute path to the campaign's run_design.py. Its sibling "
             "design_config.py is imported by name, so the two must stay together.",
    )
    parser.add_argument(
        "--models",
        default="boltz2",
        help="Comma-separated backends passed straight through to --models. Pick "
             "the one the parent's GENERATOR did not already fold against: af2 "
             "for Boltz-lineage parents, boltz2 for AF2-lineage ones.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Children per parent, each from its own Dirichlet-perturbed start. "
             "Clamped to the campaign's max_children.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="Trajectories vmapped in one invocation. GPU memory scales with "
             "this times the complex length; the source campaign lost 32 tasks "
             "to OOM at 4 once the complex passed ~296 residues.",
    )
    # The campaign's measured schedule: 3 x (90 + 30) beat a single 1 x (250+30)
    # by +37% at ipTM >= 0.7 for +28% GPU, and 3 x (250+30) was the WORST arm.
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--soft-steps", type=int, default=90)
    parser.add_argument("--sharp-steps", type=int, default=30)
    parser.add_argument("--init-noise", type=float, default=0.15)
    parser.add_argument("--cycle-noise", type=float, default=0.15)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument(
        "--esmc-weight",
        type=float,
        default=0.5,
        help=(
            "Weight on the ESM-C pseudolikelihood, clipped to [2,100]. 0.5 is "
            "the source campaign's value in all ~30 rounds. 0 disables it."
        ),
    )
    parser.add_argument(
        "--esmc-checkpoint",
        default="",
        help=(
            "PIN THIS TO A SNAPSHOT DIRECTORY, not to the repo name. "
            "`from_pretrained` resolves a NAME against the hub and will pull a "
            "newer revision into the cache, silently repointing refs/main. That "
            "happened here on 2026-09-26: the upstream repo was re-versioned, "
            "and the new revision renames the class (ESMCForMaskedLM -> "
            "EsmcForMaskedLM) and every tensor (esmc.embed.weight -> "
            "esmc.embed_tokens.weight, transformer.blocks.N -> layers.N), so "
            "this container's pinned class matches nothing, initializes random "
            "weights, and esmc_pll becomes NaN. The total loss then sits at "
            "mosaic's 1e6 sentinel and the run APPEARS to proceed while "
            "optimizing nothing -- the only symptom is a 'Some weights ... "
            "newly initialized' line buried in the model-loading output. "
            "A snapshot path resolves locally and cannot drift. Empty falls "
            "back to the repo name, which is how this got broken."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Diffusion samples per evaluation, for backends that have one. "
             "Ignored for af2, which asserts it is handed none.",
    )
    parser.add_argument(
        "--msa-clusters",
        type=int,
        default=128,
        help="af2 only. design_config's own default is 512/2048, the full "
             "scoring protocol; a gradient loop cannot afford it.",
    )
    parser.add_argument("--extra-msa", type=int, default=512)
    parser.add_argument(
        "--weights",
        default="",
        help="MOSAIC_WEIGHTS tree, holding plain-named caches (alphafold/, "
             "boltz/, hf/ ...). run_design.py builds its models with no "
             "data_dir and so reaches for ~/.alphafold, ~/.boltz and friends; "
             "mosaic-exec.sh normally bind-mounts them into a synthetic HOME, "
             "and this harness has no bind hook, so the wrapper rebuilds that "
             "HOME as symlinks instead. Without it, af2 finds no params and "
             "boltz2 finds no checkpoint.",
    )
    parser.add_argument(
        "--epitope",
        default="",
        help=(
            "Comma-separated 0-BASED target indices for BinderTargetContact and "
            "DistogramIPTMProxy. Empty falls back to context['hotspots'], which "
            "is usually WRONG here: that is the bare 3-hotspot list, which the "
            "source campaign's R17 measured as the tightest placement and the "
            "WORST yield. Its production rounds use a 24-residue consensus "
            "epitope, which cannot be derived from geometry -- "
            "pocket_residues(shell=6.0) gives the 43-residue CROP and the "
            "epitope is the subset of it contacted by >=30% of 1500 BoltzGen "
            "poses, so it has to be inherited from a run that generated it."
        ),
    )
    parser.add_argument("--hard-sharp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    context = json.loads(args.context.read_text())
    work = Path(context["work_dir"])
    structure_dir = Path(context["structure_dir"])
    (structure_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    parents = [
        json.loads(line)
        for line in args.inputs.read_text().splitlines()
        if line.strip()
    ]
    replicates = max(1, min(args.replicates, context.get("max_children") or 1))

    # One --binder-length per invocation, so length decides the grouping. The
    # design set is already ordered by length and shards are contiguous, so in
    # practice a shard spans few groups and pays few compiles.
    by_length: dict[int, list[dict]] = defaultdict(list)
    refusals: list[dict] = []
    for parent in parents:
        unknown = sorted(set(parent["sequence"]) - set(CANONICAL))
        if unknown:
            refusals.append({
                "parent_index": parent["index"],
                "failed": f"parent has non-canonical residue(s) {unknown}",
            })
            continue
        by_length[len(parent["sequence"])].append(parent)

    with args.outputs.open("w") as sink:
        for row in refusals:
            sink.write(json.dumps(row) + "\n")
        sink.flush()

        chunk_id = 0
        for length in sorted(by_length):
            group = by_length[length]
            # Whole parents per chunk: splitting a parent's replicates across
            # two invocations would give them unrelated seeds for no reason and
            # make the pair incomparable.
            per_chunk = max(1, args.max_batch // replicates)
            for start in range(0, len(group), per_chunk):
                chunk = group[start:start + per_chunk]
                seed = (
                    context["seed"]
                    + context.get("shard", 0) * 1_000_003
                    + chunk_id * 101
                )
                chunk_id += 1
                try:
                    children = run_chunk(
                        chunk, length, seed, replicates, context, work, args
                    )
                except Exception as error:  # noqa: BLE001
                    # Reported per parent so the shard continues and the counts
                    # say which parents were tried. Deliberately broad ONLY
                    # here, around a subprocess whose failure modes belong to
                    # the cluster rather than to this script.
                    for parent in chunk:
                        sink.write(json.dumps({
                            "parent_index": parent["index"],
                            "failed": f"{type(error).__name__}: {error}",
                        }) + "\n")
                    sink.flush()
                    continue
                for row in children:
                    sink.write(json.dumps(row, allow_nan=False) + "\n")
                sink.flush()
    return 0


# The bind map `singularity/mosaic-exec.sh` applies, reproduced as symlinks.
# Plain names in the weight tree, dotted names in HOME -- which is the whole
# reason the indirection exists rather than pointing HOME at the tree directly.
CACHE_MAP = {
    "alphafold": ".alphafold",
    "boltz": ".boltz",
    "protenix": ".protenix",
    "openfold3": ".openfold3",
    "hf": ".cache/huggingface",
}


def cache_home(work: Path, weights: str) -> dict[str, str]:
    """A synthetic HOME whose dotfile caches point into the weight tree.

    The harness's optimize launcher exposes no bind hook and no env field, so
    this cannot be done the way mosaic-exec.sh does it. But the wrapped script
    is a SUBPROCESS of this one, so its environment is ours to set -- which
    reaches the same place without patching either the harness or the script.
    """
    if not weights:
        return {}
    root = Path(weights)
    home = work / "cache-home"
    for plain, dotted in CACHE_MAP.items():
        source = root / plain
        if not source.exists():
            continue
        link = home / dotted
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source)
    home.mkdir(parents=True, exist_ok=True)
    return {"HOME": str(home), "HF_HOME": str(home / ".cache" / "huggingface")}


def _epitope(context, args):
    """The epitope both contact terms slice with, 0-based.

    `--epitope` wins when given. The fallback is `context["hotspots"]`, 1-based
    FASTA positions resolved and range-checked at plan time, converted here.
    """
    if args.epitope.strip():
        return sorted({int(tok) for tok in args.epitope.split(",") if tok.strip()})
    return [spot - 1 for spot in context.get("hotspots") or []] or None


def design_config(length, seeds_fasta, batch, seed, epitope, context, args):
    """The config `run_design.py --config` consumes, with the epitope INJECTED.

    This is the wrapped script's production path, not a shortcut: the source
    campaign's `pipeline_node.py` writes exactly such a file and injects
    `epitope_idx` into it, and `RUN_PARAMETERS.md` calls the resulting
    `optimize/config_seed*.json` "the run's actual parameters, after injection".
    Driving the script from its own CLI flags instead leaves every
    `epitope_idx` at None -- measured: the first wrapper run came back with
    `epitope_idx: [None, None]`, i.e. a contact term that rewards contact
    ANYWHERE on the target. Since `BinderTargetContact.epitope_idx` is the term
    RUN_PARAMETERS shows controls placement, that silently discards the whole
    point of selecting on-site parents.

    ⚠ `--config` bypasses `config_from_args`, so anything that would have come
    from a CLI flag has to be set HERE. `--seed`, `--out`, `--soft-sharp-cycles`,
    `--cycle-noise`, `--init-noise-random`, `--hard-sharp` and `--dump-cycles`
    are read from argv directly and still apply; `--batch`, `--models`,
    `--soft-steps`, `--sharp-steps`, `--init-fasta` and `--esmc-weight` do not.

    The weights are the constants held fixed across rounds 12-20.
    `BinderTargetContact` at 2.0 and UNCAPPED is the term that controls
    placement: clipping it doubled the distance to target and halved the
    usable count.
    """
    models = []
    for name in (m.strip() for m in args.models.split(",")):
        if not name:
            continue
        if name == "af2":
            # No sampling_steps: design_config.validate refuses it for af2,
            # because the model itself asserts it was handed none.
            params = {"recycling_steps": args.recycling_steps,
                      "max_msa_clusters": args.msa_clusters,
                      "max_extra_msa": args.extra_msa}
        else:
            params = {"recycling_steps": args.recycling_steps,
                      "sampling_steps": 25, "num_samples": args.num_samples}
        models.append({"name": name, "weight": 1.0, "params": params})

    sequence_models = []
    if args.esmc_weight > 0:
        sequence_models.append({
            "name": "esmc", "weight": args.esmc_weight,
            "params": {"checkpoint": args.esmc_checkpoint or "biohub/ESMC-300M",
                       "clip_lower": 2.0, "clip_upper": 100.0},
        })

    return {
        "target": {"fasta": context["target_sequence"],
                   "msa": context.get("target_msa"),
                   "use_msa": bool(context.get("target_msa"))},
        "binder": {"length": length, "no_cys": True,
                   "init_fasta": str(seeds_fasta), "init_noise": args.init_noise},
        "models": models,
        "losses": [
            {"name": "BinderTargetContact", "weight": 2.0,
             "params": {"contact_distance": 20.0, "epitope_idx": epitope}},
            {"name": "WithinBinderContact", "weight": 1.0, "params": {}},
            {"name": "HelixLoss", "weight": 1.0, "params": {}},
            {"name": "DistogramRadiusOfGyration", "weight": 1.0, "params": {}},
            # RESTRICTED, matching every one of the source campaign's runtime
            # configs. Earlier this was left unrestricted because the proxy
            # clipped to exactly 0 -- but that was a symptom of passing the
            # bare 3-hotspot list, not of the restriction: with a 24-residue
            # epitope the top-k pool is large enough. Note the term reads 0.00
            # in his logs too, so a zero here is his behaviour, not a fault.
            {"name": "DistogramIPTMProxy", "weight": 2.0,
             "params": {"contact_distance": 12.0, "epitope_idx": epitope}},
            {"name": "PLDDTLoss", "weight": 1.0, "params": {}},
            {"name": "IPSAE_min", "weight": 1.0, "params": {}},
        ],
        "sequence_models": sequence_models,
        "optimizer": {
            "soft": {"n_steps": args.soft_steps, "stepsize": 0.1,
                     "momentum": 0.9, "scale": 1.0, "logspace": False,
                     "max_gradient_norm": None},
            "sharp": {"n_steps": args.sharp_steps, "stepsize": 0.025,
                      "momentum": 0.5, "scale": 2.0, "logspace": False,
                      "max_gradient_norm": None},
        },
        "mpnn": {"weights": "soluble", "backbone_noise": 0.0, "terms": [
            {"name": "InverseFoldingSequenceRecovery", "weight": 1.0,
             "params": {"temp": 0.1}},
        ]},
        "run": {"batch": batch, "seed": seed},
    }


def run_chunk(chunk, length, seed, replicates, context, work, args):
    """One `run_design.py` invocation over `len(chunk) * replicates` trajectories."""
    batch = len(chunk) * replicates
    out = work / f"chunk-{seed}"
    out.mkdir(parents=True, exist_ok=True)

    # EXACTLY `batch` entries, each parent repeated `replicates` times in order.
    # That is what makes run_design.py:469's modulo reduce to seqs[b].
    seeds_fasta = out / "init.fasta"
    with seeds_fasta.open("w") as handle:
        for parent in chunk:
            for replicate in range(replicates):
                handle.write(f">p{parent['index']}_r{replicate}\n{parent['sequence']}\n")

    started = time.time()
    if args.dry_run:
        payload, cycles = _dry_run(chunk, replicates, seed, args)
    else:
        epitope = _epitope(context, args)
        config_path = out / "config.json"
        config_path.write_text(json.dumps(
            design_config(length, seeds_fasta, batch, seed, epitope, context, args),
            indent=2, sort_keys=True,
        ))
        command = [
            sys.executable, args.design_script,
            # --target is still required by argparse even when --config carries
            # it, so it is passed twice and the config is what is used.
            "--target", context["target_sequence"],
            "--config", str(config_path),
            "--out", str(out),
            "--seed", str(seed),
            # Read from argv directly, so they survive --config.
            "--init-noise-random",
            "--soft-sharp-cycles", str(max(1, args.cycles)),
            "--cycle-noise", str(args.cycle_noise),
            "--dump-cycles",
        ]
        if args.hard_sharp:
            command.append("--hard-sharp")

        environment = {**os.environ, **cache_home(work, args.weights)}
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, env=environment
        )
        # The script's log is the only record of per-term aux, so it is kept
        # beside the output whether or not the invocation succeeded.
        (out / "run_design.log").write_text((done.stdout or "") + (done.stderr or ""))
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError(
                f"run_design.py exited {done.returncode}:\n" + "\n".join(tail)
            )
        payload = json.loads((out / f"designs_seed{seed}.json").read_text())
        cycles_path = out / f"cycles_seed{seed}.json"
        cycles = json.loads(cycles_path.read_text()) if cycles_path.exists() else None

    elapsed = time.time() - started
    return _rows(chunk, replicates, seed, payload, cycles, elapsed, context, args)


def _rows(chunk, replicates, seed, payload, cycles, elapsed, context, args):
    """`designs_seed*.json` -> child rows.

    ⚠ `results` is sorted by LOSS, not batch order, and the batch item is
    `r["trajectory"]`. Using the enumeration index instead scrambles the join
    inside every batch -- a trap `RUN_PARAMETERS.md` records as having survived
    undetected in the source campaign's own analysis script for days.
    """
    structure_dir = Path(context["structure_dir"])
    per_cycle = _cycle_losses(cycles)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for result in payload.get("results", []):
        b = int(result["trajectory"])
        which, replicate = divmod(b, replicates)
        if which >= len(chunk):
            continue
        parent = chunk[which]
        sequence = result["sequence"]
        # `changes_length: false` is a claim the host enforces; skipping here
        # names the trajectory instead of failing the shard.
        if len(sequence) != len(parent["sequence"]):
            continue
        loss = float(result["loss"])
        if not math.isfinite(loss):
            continue

        relative = f"trajectories/{parent['index']}-{replicate}.json"
        (structure_dir / relative).write_text(json.dumps(
            {"trajectory": b, "seed": seed, "cycles": per_cycle.get(b, [])}, indent=1
        ))

        substitutions = sum(
            1 for was, now in zip(parent["sequence"], sequence) if was != now
        )
        metrics = {
            "loss": loss,
            "n_substitutions": float(substitutions),
            "seq_identity": 1.0 - substitutions / max(1, len(parent["sequence"])),
            "opt_replicate": float(replicate),
            "opt_seed": float(seed),
            "opt_cycles": float(max(1, args.cycles)),
            "hard_sharp": 1.0 if args.hard_sharp else 0.0,
            # Flags rather than one id, so a comma-separated --models stays
            # representable and a query can ask "did AF2 shape this design?"
            "lm_af2": 1.0 if "af2" in models else 0.0,
            "lm_boltz2": 1.0 if "boltz2" in models else 0.0,
        }
        history = per_cycle.get(b, [])
        if history:
            metrics["cycle1_loss"] = float(history[0])
            # Which cycle produced the winner. If this is almost always 1,
            # cycling did not earn its GPU.
            metrics["best_cycle"] = float(
                min(range(len(history)), key=lambda i: history[i]) + 1
            )
        rows.append({
            "parent_index": parent["index"],
            "child": replicate,
            "sequence": sequence,
            "metrics": metrics,
            "trajectory": relative,
            "seconds": round(elapsed / max(1, len(chunk) * replicates), 2),
        })
    return rows


def _cycle_losses(cycles):
    """trajectory -> [loss at each cycle boundary], from `cycles_seed*.json`."""
    out: dict[int, list[float]] = defaultdict(list)
    if not cycles:
        return out
    for entry in cycles.get("cycles", []) or []:
        for design in entry.get("designs", []) or []:
            out[int(design["trajectory"])].append(float(design["loss"]))
    return out


def _dry_run(chunk, replicates, seed, args):
    """The row shapes without mosaic, a GPU or the wrapped script.

    Mirrors `designs_seed*.json` including its sort-by-loss ordering, so the
    join in `_rows` is exercised rather than accidentally correct.
    """
    rng = random.Random(seed)
    alphabet = "ADEFGHIKLMNPQRSTVWY"
    results = []
    for which, parent in enumerate(chunk):
        for replicate in range(replicates):
            mutated = list(parent["sequence"])
            for position in rng.sample(range(len(mutated)), k=min(4, len(mutated))):
                mutated[position] = rng.choice(alphabet)
            results.append({
                "trajectory": which * replicates + replicate,
                "seed": seed,
                "loss": round(rng.uniform(10.0, 25.0), 4),
                "sequence": "".join(mutated),
            })
    results.sort(key=lambda r: r["loss"])
    cycles = [
        {"cycle": cycle + 1, "designs": [
            {"trajectory": r["trajectory"], "seed": seed,
             "loss": r["loss"] + (max(1, args.cycles) - cycle - 1) * 0.5,
             "sequence": r["sequence"]}
            for r in results
        ]}
        for cycle in range(max(1, args.cycles))
    ]
    return {"results": results}, {"cycles": cycles}


if __name__ == "__main__":
    raise SystemExit(main())
