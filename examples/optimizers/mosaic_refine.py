#!/usr/bin/env python3
"""TEMPLATE: refine an existing binder with mosaic, instead of hallucinating one.

The difference from `drivers/mosaic/hallucinate_binders.py` is one line, and it
is the whole idea: that script starts the PSSM from Gumbel noise, so it invents
a binder. This one starts it from the PARENT's one-hot, so simplex_APGM walks
away from a sequence somebody already has rather than from nothing.

    _pssm = jnp.log(one_hot(parent) * (1 - e) + e / 20)

**Status: not GPU-verified.** The contract plumbing (argument parsing, the row
shapes, the failure rows, the trajectory) is exercised by
`tests/test_optimize.py` against a stub, and the mosaic calls are copied from
the hallucination driver that this campaign has run. But nobody has yet run
this file on a node, so treat the loss weights and the schedule as a starting
point rather than as tuned values -- and run it over two or three parents
before a whole set.

Check the I/O first, which needs no GPU and no mosaic:

    python scripts/validate_optimizer.py --script examples/optimizers/mosaic_refine.py \\
        --sequences <a parent> --target-fasta target.fasta \\
        --declare loss:min --declare start_loss:min --declare n_substitutions:none
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# The contract. Nothing below the next banner is about optimization.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The three the harness always passes, in this order.
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--context", required=True)
    # Whatever `args:` in the YAML adds.
    parser.add_argument("--soft-steps", type=int, default=50)
    parser.add_argument("--sharpen-steps", type=int, default=25)
    parser.add_argument("--final-steps", type=int, default=10)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=4)
    # How far to move off the parent. 0.0 pins the PSSM exactly to it, which
    # gives the optimizer no gradient to work with; 1.0 is a fresh start and
    # would make this the hallucination script.
    parser.add_argument("--epsilon", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = json.loads(Path(args.context).read_text())

    target_sequence = context["target_sequence"]
    msa_path = context.get("target_msa")
    structure_dir = Path(context["structure_dir"])
    structure_dir.mkdir(parents=True, exist_ok=True)
    # 1-based FASTA positions from the harness; mosaic wants 0-based indices.
    epitope_idx = [spot - 1 for spot in context.get("hotspots") or []] or None

    parents = [
        json.loads(line)
        for line in Path(args.inputs).read_text().splitlines()
        if line.strip()
    ]

    # One model load per process, not per design. The binder length is part of
    # the feature build, so a shard spanning many lengths pays a JIT recompile
    # per length -- which is why the design set is ordered by length and shards
    # are contiguous.
    built: dict[int, tuple] = {}

    with open(args.outputs, "w") as out:
        for parent in parents:
            index = parent["index"]
            started = time.time()
            try:
                sequence, loss, start_loss = refine(
                    parent["sequence"],
                    target_sequence=target_sequence,
                    msa_path=msa_path,
                    epitope_idx=epitope_idx,
                    seed=context["seed"] + index,
                    args=args,
                    cache=built,
                )
            except Exception as error:  # noqa: BLE001 - one parent must not kill the shard
                # Reported, not omitted. Absence and refusal look the same in a
                # query and mean opposite things.
                out.write(json.dumps({
                    "parent_index": index,
                    "failed": f"{type(error).__name__}: {error}",
                }) + "\n")
                continue

            substitutions = sum(
                1 for left, right in zip(parent["sequence"], sequence) if left != right
            )
            out.write(json.dumps({
                "parent_index": index,
                "child": 0,
                "sequence": sequence,
                "metrics": {
                    "loss": loss,
                    "start_loss": start_loss,
                    "n_substitutions": substitutions,
                },
                "seconds": round(time.time() - started, 2),
            }) + "\n")
            out.flush()
    return 0


# ---------------------------------------------------------------------------
# The optimization. Everything mosaic-specific is below here.
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
) -> tuple[str, float, float]:
    """Walk one parent downhill. Returns (sequence, final loss, start loss).

    `start_loss` is reported alongside the final one on purpose: a loss of 0.31
    means nothing without knowing it began at 0.33, and "the optimizer ran and
    improved nothing" is a result worth being able to count.
    """
    import jax
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import NoCys
    from mosaic.models.boltz2 import Boltz2
    from mosaic.optimizers import simplex_APGM
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain

    binder_length = len(parent_sequence)
    if binder_length not in cache:
        folder = Boltz2()
        mpnn = load_mpnn_sol(0.05)
        bias = jnp.zeros((binder_length, 20)).at[:, TOKENS.index("C")].set(-1e6)
        objective = (
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
        loss_function = NoCys(
            folder.build_multisample_loss(
                loss=objective,
                features=features,
                recycling_steps=args.recycling_steps,
                num_samples=args.num_samples,
            )
        )
        cache[binder_length] = (folder, loss_function)
    folder, loss_function = cache[binder_length]

    # THE ONE LINE THAT MAKES THIS REFINEMENT RATHER THAN HALLUCINATION.
    # NoCys leaves the optimizer a 19-token alphabet; cysteine is spliced back
    # in at zero probability by NoCys.sequence at the end.
    alphabet = [token for token in TOKENS[:20] if token != "C"]
    parent_index = [alphabet.index(residue) for residue in parent_sequence.replace("C", "A")]
    one_hot = jax.nn.one_hot(jnp.array(parent_index), len(alphabet))
    start = one_hot * (1.0 - args.epsilon) + args.epsilon / len(alphabet)

    start_loss, _ = loss_function(_padded(jnp, start), key=jax.random.key(seed))

    _, pssm = simplex_APGM(
        loss_function=loss_function,
        x=start,
        stepsize=0.2 * binder_length**0.5,
        n_steps=args.soft_steps,
        momentum=0.3,
        scale=1.00,
        logspace=False,
        max_gradient_norm=1.0,
    )
    for steps, scale in ((args.sharpen_steps, 1.25), (args.final_steps, 1.4)):
        pssm, _ = simplex_APGM(
            loss_function=loss_function,
            x=jnp.log(pssm + 1e-5),
            stepsize=0.5 * binder_length**0.5,
            n_steps=steps,
            momentum=0.0,
            scale=scale,
            logspace=True,
            max_gradient_norm=1.0,
        )

    pssm = NoCys.sequence(pssm)
    tokens = pssm.argmax(-1)
    sequence = "".join(TOKENS[token] for token in tokens)

    # Re-score the discrete child at a larger budget than the loop used, the
    # way the hallucination driver does. This number is what `loss` means.
    features, _ = folder.target_only_features(
        chains=[
            TargetChain(sequence=sequence, use_msa=False),
            TargetChain(
                sequence=target_sequence,
                use_msa=msa_path is not None,
                msa_path=msa_path,
            ),
        ]
    )
    ranking = folder.build_multisample_loss(
        loss=1.00 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE(),
        features=features,
        recycling_steps=3,
        num_samples=6,
    )
    final, _ = ranking(jax.nn.one_hot(tokens, 20), key=jax.random.key(0))
    return sequence, float(final.item()), float(start_loss.item())


def _padded(jnp, pssm):
    """The 19-token simplex back to 20 with zero for cysteine.

    `NoCys` does this internally for the optimizer, but the loss is evaluated
    directly here to get a starting value, so it has to be done by hand.
    """
    from mosaic.common import TOKENS

    position = TOKENS.index("C")
    left, right = pssm[:, :position], pssm[:, position:]
    zeros = jnp.zeros((pssm.shape[0], 1))
    return jnp.concatenate([left, zeros, right], axis=-1)


if __name__ == "__main__":
    raise SystemExit(main())
