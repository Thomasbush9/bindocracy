#!/usr/bin/env python3
"""Refine an existing binder with mosaic, driven by AlphaFold-2 MSA multimer.

The campaign's hallucination driver and `examples/optimizers/mosaic_refine.py`
both drive Boltz-2. This one drives **AF2 multimer with the campaign's target
alignment**, for two reasons that are really one:

* AF2 is the only backend here whose loss can see the target's a3m *and* whose
  gradients reach a binder PSSM, so "optimize against the alignment the
  campaign actually has" means AF2.
* Boltz-2 is the model most of these parents were generated against. Driving
  the children against it again would make every later Boltz-2 number a
  measurement of this optimizer. `loss_models: [af2]` is the honest claim, and
  it leaves eight scorers held out instead of seven.

The one line that makes this refinement rather than hallucination is the same
one as ever: the PSSM starts at the PARENT's one-hot rather than at Gumbel
noise, so simplex_APGM walks away from a sequence somebody already has.

**Status: GPU-verified, and it did not help.** Run
`optimize-af2refine-smoke23c` drove four parents on an H100 with the schedule
in `configs/optimize/af2_refine.yaml` (12 soft + 4 sharpen + 2 final steps).
Three of the four children scored worse than their parents and one was
unchanged, by +0.000 to +0.299 on the ranking loss. That is a statement about
an 18-step schedule, which is roughly a tenth of a real one, not about the
binders or about AF2. Raise the step counts before reading anything into a
child. Two bugs were found by running it and are fixed here: `simplex_APGM`
returns three values when handed a `trajectory_fn`, and `start_loss` has to be
the ranking objective rather than the training one or it does not subtract
from `loss`.

**Weights.** Declared in `kit.yaml` as the requirement `alphafold-params`,
because the optimize tool builds its own `singularity exec` line, does not go
through `mosaic-exec.sh`, and so binds nothing onto `~/.alphafold` inside the
image. The harness resolves that name against the campaign's bindings and
hands the path over as `context["dependencies"]["alphafold-params"]`. Until it
reads `kit.yaml`, the campaign passes the same path through `args`, so
`--af2-data-dir` remains the fallback and the two cannot disagree: the context
wins when present.

**MSA routing.** Declared as the optional requirement `mosaic-src`. The
shipped `mosaic.sif` predates the AF2 MSA work and its `models/af2.py` still
asserts "AF2 interface does not support MSA yet" at an interface, so the kit
verifies the capability by importing `mosaic.models.af2_msa` rather than by
trusting a version tag, and the overlay is what satisfies it on a stale image.
On today's harness that overlay is `runtime.dev_source`, and it binds onto
`/opt/mosaic/src/mosaic`, one level below the scorer's `MOSAIC_DEV_SRC`.

Check the I/O first. It needs no GPU, no mosaic and no weights, because
everything below the second banner is skipped by `--dry-run`:

    python scripts/validate_optimizer.py \
        --script optimizers/mosaic-af2-refine/optimizer.py \
        --design-set sets/<digest>.json \
        --target-fasta target.fasta \
        --declare loss:min --declare start_loss:min \
        --declare n_substitutions:none --declare opt_steps:none \
        --max-children 1 --script-args --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from bindocracy_io import RejectCandidate, run_optimization

# The 20 the design table accepts. A parent outside it is a refusal, not a bug.
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# The contract. Nothing above the next banner is about optimization.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Only this kit's own options.

    `--inputs`, `--outputs` and `--context` belong to `bindocracy_io`, which
    also owns row identity: it attaches `parent_index` and the child ordinal,
    so this script cannot get them wrong and cannot emit a duplicate pair.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--af2-data-dir",
        default="~/.alphafold",
        help=(
            "Fallback for the `alphafold-params` requirement, used only when "
            "context['dependencies'] does not carry it. Directory holding params/."
        ),
    )
    parser.add_argument("--soft-steps", type=int, default=12)
    parser.add_argument("--sharpen-steps", type=int, default=4)
    parser.add_argument("--final-steps", type=int, default=2)
    parser.add_argument("--recycling-steps", type=int, default=1)
    # Re-scoring the discrete child costs one forward pass, so it can afford
    # more recycles than the loop used. This number is what `loss` means.
    parser.add_argument("--rescore-recycling", type=int, default=3)
    # AF2's MSA stacks dominate both memory and step time. The defaults here
    # are a test-run budget, not the scoring protocol's (512 / 2048).
    parser.add_argument("--msa-clusters", type=int, default=128)
    parser.add_argument("--extra-msa", type=int, default=512)
    # How far to move off the parent. 0.0 pins the PSSM to it and leaves the
    # optimizer no gradient; 1.0 is a fresh start, which is hallucination.
    parser.add_argument("--epsilon", type=float, default=0.1)
    # Exercise the contract without jax, mosaic or a GPU.
    parser.add_argument("--dry-run", action="store_true")
    return parser


def dependency(context: dict, name: str, fallback: str) -> str:
    """The path the harness resolved for a requirement this kit declares.

    `kit.yaml` names what this optimizer needs; the campaign says where those
    things are; the harness resolves one against the other and puts the answer
    in the context. A script that read a path out of its own arguments instead
    would be claiming to know a filesystem it has never seen, and a missing
    dependency would surface as a model constructor failing after the GPU was
    allocated rather than as a refusal at preflight.

    The fallback exists because the harness does not read `kit.yaml` yet. It is
    the same value, passed the long way round through `args`.
    """
    resolved = (context.get("dependencies") or {}).get(name)
    return resolved or fallback


def main() -> int:
    # Per-process state. One model load and one feature build per binder
    # LENGTH, not per design: the length is baked into the AF2 feature shapes,
    # so a shard spanning many lengths pays a JIT recompile at each one. That
    # is why the design set is ordered by length and shards are contiguous.
    cache: dict[int, tuple] = {}
    setup: dict = {}

    def optimize_parent(parent, context, args):
        if not setup:
            args.af2_data_dir = dependency(
                context, "alphafold-params", args.af2_data_dir
            )
            structure_dir = Path(context["structure_dir"])
            (structure_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            setup["structure_dir"] = structure_dir
            # 1-based positions in the target's FASTA, resolved and
            # range-checked when the run was planned. mosaic slices a contact
            # matrix with 0-based ones.
            setup["epitope"] = [
                spot - 1 for spot in context.get("hotspots") or []
            ] or None

        index = parent["index"]
        sequence_in = parent["sequence"]
        # An expected refusal: a parent this optimizer cannot represent. Only
        # this one and the NaN below are caught. Everything else propagates and
        # fails the job on purpose -- catching broadly is how a bug in this
        # script once wrote four "refusals" and looked like a biological
        # result.
        unknown = sorted(set(sequence_in) - set(CANONICAL))
        if unknown:
            raise RejectCandidate(f"parent has non-canonical residue(s) {unknown}")

        started = time.time()
        sequence, loss, start_loss, trajectory = refine(
            sequence_in,
            target_sequence=context["target_sequence"],
            msa_path=context.get("target_msa"),
            epitope_idx=setup["epitope"],
            seed=context["seed"] + index,
            args=args,
            cache=cache,
        )
        if not (math.isfinite(loss) and math.isfinite(start_loss)):
            raise RejectCandidate(
                f"AF2 returned a non-finite loss (start {start_loss}, final {loss})"
            )

        # Relative to context['structure_dir']. An absolute path is refused so
        # that a run directory can be moved; known-issues.md records that
        # having already cost this campaign a day.
        relative = f"trajectories/{index}.jsonl"
        (setup["structure_dir"] / relative).write_text(
            "".join(
                json.dumps({"step": step, "value": value}) + "\n"
                for step, value in enumerate(trajectory)
            )
        )

        yield {
            "sequence": sequence,
            "metrics": {
                # `start_loss` rides along with `loss` on purpose, and both are
                # the RANKING objective: a final loss means nothing without the
                # number it started from, and only if the two subtract.
                "loss": loss,
                "start_loss": start_loss,
                "n_substitutions": sum(
                    1 for was, now in zip(sequence_in, sequence) if was != now
                ),
                "opt_steps": len(trajectory),
            },
            "trajectory": relative,
            "seconds": round(time.time() - started, 2),
        }

    return run_optimization(optimize_parent, parser=build_parser())


# ---------------------------------------------------------------------------
# The optimization. Everything AF2-specific is below here.
# ---------------------------------------------------------------------------


def refine(
    parent_sequence: str,
    *,
    target_sequence: str,
    msa_path: str | None,
    epitope_idx: list[int] | None,
    seed: int,
    args: argparse.Namespace,
    cache: dict[int, tuple],
) -> tuple[str, float, float, list[float]]:
    """Walk one parent downhill. (sequence, final loss, start loss, trajectory).

    `loss` and `start_loss` are both the RANKING objective -- ipTM plus the two
    ipSAE terms at `--rescore-recycling` recycles -- evaluated on the discrete
    child and on the parent. The trajectory carries the training objective,
    which is a different and much larger number; the two must not be mixed.
    """
    if args.dry_run:
        return _dry_run(parent_sequence, seed, args)

    import jax
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys
    from mosaic.models.af2 import AlphaFold2
    from mosaic.optimizers import simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain

    binder_length = len(parent_sequence)
    if binder_length not in cache:
        # multimer=True is the binder-plus-target case. `data_dir` is passed
        # rather than defaulted because this tool's launcher binds no weights.
        folder = AlphaFold2(
            data_dir=args.af2_data_dir,
            multimer=True,
            max_msa_clusters=args.msa_clusters,
            max_extra_msa=args.extra_msa,
        )
        mpnn = load_mpnn_sol(0.05)
        # -inf for Cys so MPNN does not propose what NoCys has already removed
        # from the optimizer's alphabet.
        bias = jnp.zeros((binder_length, 20)).at[:, TOKENS.index("C")].set(-1e6)

        objective = (
            # The only term the epitope reaches.
            sp.BinderTargetContact(epitope_idx=epitope_idx)
            + sp.WithinBinderContact()
            + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.001), bias=bias)
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss()
        )
        # THE MSA. `use_msa=True` with the campaign a3m is the whole point of
        # this optimizer, and it is also the line that needs `dev_source`: the
        # shipped image asserts rather than building per-chain MSA features.
        features, _ = folder.binder_features(
            binder_length=binder_length,
            chains=[
                TargetChain(
                    sequence=target_sequence,
                    use_msa=msa_path is not None,
                    msa_path=msa_path,
                )
            ],
        )
        # AF2 has no diffusion sampler, so there is no `build_multisample_loss`
        # here and no `num_samples`; a single trunk is all there is to average.
        loss_function = NoCys(
            folder.build_loss(
                loss=objective,
                features=features,
                recycling_steps=args.recycling_steps,
            )
        )
        ranking = folder.build_loss(
            loss=1.00 * sp.IPTMLoss()
            + 0.5 * sp.TargetBinderIPSAE()
            + 0.5 * sp.BinderTargetIPSAE(),
            features=features,
            recycling_steps=args.rescore_recycling,
        )
        cache[binder_length] = (loss_function, ranking)
    loss_function, ranking = cache[binder_length]

    # THE ONE LINE THAT MAKES THIS REFINEMENT RATHER THAN HALLUCINATION.
    # NoCys leaves a 19-token alphabet; cysteine is spliced back at zero
    # probability by NoCys.sequence once the walk is over.
    alphabet = [token for token in TOKENS[:20] if token != "C"]
    seeded = [alphabet.index(residue) for residue in parent_sequence.replace("C", "A")]
    one_hot = jax.nn.one_hot(jnp.array(seeded), len(alphabet))
    start = one_hot * (1.0 - args.epsilon) + args.epsilon / len(alphabet)

    # `start_loss` is measured with the RANKING loss, not with the training
    # objective, and that is the whole point of it. The first GPU run reported
    # start_loss ~10-16 beside loss ~-1, because one was the nine-term design
    # objective and the other was ipTM + ipSAE: two different quantities, so
    # their difference meant nothing and "did the optimizer improve anything"
    # could not be answered. Same function, same key as the child's re-score,
    # so the two numbers subtract.
    parent_tokens = jnp.array([TOKENS.index(residue) for residue in parent_sequence])
    start_value, _ = ranking(jax.nn.one_hot(parent_tokens, 20), key=jax.random.key(0))

    trajectory: list[float] = []
    record = lambda aux, _x: trajectory.append(float(aux["loss"]))

    # THREE values, not two: simplex_APGM returns (x, best_x) normally and
    # (x, best_x, trajectory) when a trajectory_fn is given. Its own list is
    # discarded here because `record` already appended to ours.
    _, pssm, _ = simplex_APGM(
        loss_function=loss_function,
        x=start,
        stepsize=0.2 * binder_length**0.5,
        n_steps=args.soft_steps,
        momentum=0.3,
        scale=1.00,
        logspace=False,
        max_gradient_norm=1.0,
        trajectory_fn=record,
    )
    for steps, scale in ((args.sharpen_steps, 1.25), (args.final_steps, 1.4)):
        if steps <= 0:
            continue
        pssm, _, _ = simplex_APGM(
            loss_function=loss_function,
            x=jnp.log(pssm + 1e-5),
            stepsize=0.5 * binder_length**0.5,
            n_steps=steps,
            momentum=0.0,
            scale=scale,
            logspace=True,
            max_gradient_norm=1.0,
            trajectory_fn=record,
        )

    tokens = NoCys.sequence(pssm).argmax(-1)
    sequence = "".join(TOKENS[token] for token in tokens)

    # Re-score the discrete child on the same features at more recycles than
    # the loop used. This number, not the loop's last value, is `loss`.
    final, _ = ranking(jax.nn.one_hot(tokens, 20), key=jax.random.key(0))
    return sequence, float(final), float(start_value), trajectory


def _dry_run(
    parent_sequence: str, seed: int, args: argparse.Namespace
) -> tuple[str, float, float, list[float]]:
    """Contract plumbing without jax, mosaic, weights or a GPU.

    The validator's whole value is that it costs seconds. Importing AF2 and
    loading five sets of multimer parameters costs minutes and a bound weight
    tree, and it proves nothing about the row shapes -- which is the only thing
    the validator checks. So the I/O path is exercised with a stand-in walk.
    """
    rng = random.Random(seed)
    alphabet = "ADEFGHIKLMNPQRSTVWY"
    mutated = list(parent_sequence)
    for position in rng.sample(range(len(mutated)), k=min(3, len(mutated))):
        mutated[position] = rng.choice(alphabet)
    steps = max(1, args.soft_steps + args.sharpen_steps + args.final_steps)
    start_value = 1.0
    trajectory = [start_value - 0.01 * step for step in range(steps)]
    return "".join(mutated), trajectory[-1], start_value, trajectory


if __name__ == "__main__":
    raise SystemExit(main())
