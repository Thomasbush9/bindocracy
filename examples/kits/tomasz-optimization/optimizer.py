#!/usr/bin/env python3
"""Refine an existing binder with the DIO3 campaign's soft-sharp CYCLING protocol.

This is the protocol worked out over rounds 21-29 of the DIO3 campaign
(`tomasz/mosaic_test/pipelines/RUN_PARAMETERS.md`), packaged as a kit. Three
things distinguish it from `examples/optimizers/mosaic_refine.py`, and the
first is the reason the kit exists.

**1. Soft-sharp CYCLING, not one pass.** The campaign measured
`3 x (90 soft + 30 sharp)` against a single `1 x (250 + 30)` baseline on 48
parents at the catalytic pocket: +37% at ipTM >= 0.7, +40% at >= 0.8, and the
best `worst@5` in the sweep, for +28% GPU. Re-running *soft* genuinely
re-searches -- cycle 2 beat cycle 1 by 0.58 median loss, three times what
perturbing the sharp start bought -- and JAX compiles once, so later cycles are
nearly free (measured 290 s on cycle 1, then 22 s on cycles 2 and 3).

Between cycles the iterate is **argmax-ed to a one-hot vertex and then blended
back toward uniform**. The argmax is the point: it forces commitment to a
discrete sequence before the next soft phase. Carrying a soft blend forward is
not the same operation and does not reproduce the result.

Do not scale this up. `3 x (250 + 30)` -- 840 steps, three times the compute --
was the *worst* cycling arm on every column the campaign measured.

**2. An optional straight-through estimator on the sharp stage** (`--hard-sharp`).
During the soft phase the structure models never see a protein:
`losses/boltz2.py` writes the probability vector straight into `res_type`, so
the model folds the probability-weighted average of the amino-acid embeddings.
The optimizer reaches low-loss points no discrete sequence can occupy, and
sharpening is the bill -- measured at 7.77 loss units on one diagnosed
trajectory. `HardSequence` below closes that gap.

It is **off by default and must be piloted per site.** It gave 2.2x the designs
at ipTM >= 0.7 at the catalytic pocket and 1.5x at Y84, and was strictly worse
on every column at two other sites.

**3. The loss model is chosen per parent lineage** (`--model`). A design whose
generator already folded against Boltz-2 must not be refined against Boltz-2,
or every later Boltz-2 number measures this optimizer. `boltz2` for AF2-lineage
parents, `af2` for Boltz-lineage ones. See `kit.yaml`.

This departs from the source campaign, which used OpenFold3 as its second
model. `mosaic.models.of3` is a port and the port is broken -- the harness
deprecates it on evidence that it scores BELOW a sequence-only control -- so
driving a gradient through it would be driving one through noise. AF2 is the
substitute. Two consequences: AF2 has no diffusion sampler, so `--num-samples`
does nothing on that path; and AF2 already hardens PART of its input natively
(`models/af2.py:226`, "Do not touch this"), so `--hard-sharp` stacks a full
straight-through estimator on a partial one. Compare an `af2` STE arm only
against its own non-STE control, never against a `boltz2` one.

**Status: NOT GPU-verified.** The contract plumbing is exercised by
`--dry-run`. Nobody has run this file on a node. Run two or three parents and
read `start_loss` against `loss` before committing a set.

Check the I/O first -- no GPU, no mosaic, seconds:

    python scripts/validate_optimizer.py \
        --script examples/kits/tomasz-optimization/optimizer.py \
        --design-set sets/<digest>.json --target-fasta target.fasta \
        --declare loss:min --declare start_loss:min \
        --declare n_substitutions:none --declare seq_identity:none \
        --declare opt_cycles:none --declare best_cycle:none \
        --declare hard_sharp:none --declare loss_model_id:none \
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

# Recorded on every child so a query can separate populations that are not
# comparable. `loss` is the ranking objective and IS comparable across these;
# the training loss is not, which is why it never leaves the trajectory file.
MODEL_IDS = {"boltz2": 0.0, "af2": 1.0}

# AF2 has no diffusion sampler and asserts if handed one, so it takes a
# different builder and ignores --num-samples.
HAS_SAMPLER = {"boltz2": True, "af2": False}

# The campaign's measured schedule (design_config.py:393-396). Absolute
# stepsizes, NOT scaled by binder length -- that is this protocol's
# parameterization and changing it changes the protocol.
#
# `max_gradient_norm` is deliberately absent: the campaign leaves it None and
# simplex_APGM then defaults to sqrt(binder_length), about 8.9 for an 80-mer.
# Pinning it to 1.0 (as the length-scaled mosaic-af2-refine kit does, because
# ITS stepsize is length-scaled) would make every effective step ~9x smaller
# than the schedule that was measured, and the run would simply look like
# "cycling did not help".
SOFT = {"stepsize": 0.1, "momentum": 0.9, "scale": 1.0}
SHARP = {"stepsize": 0.025, "momentum": 0.5, "scale": 2.0}
# With a one-hot forward pass there is nothing left to squeeze, so the sharp
# stage drops to scale 1.0 and runs longer: the estimator has to move argmaxes
# rather than concentrate mass.
SHARP_STE_SCALE = 1.0


# ---------------------------------------------------------------------------
# The contract. Nothing above the next banner is about optimization.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_IDS),
        default="boltz2",
        help=(
            "Folding model that drives the loss. Pick the one the parent's "
            "GENERATOR did not already fold against: boltz2 for AF2-lineage "
            "parents (freebindcraft, genie3, proteina_complexa, pxdesign), "
            "af2 for Boltz-lineage parents (boltzgen, protein_hunter)."
        ),
    )
    parser.add_argument(
        "--af2-data-dir",
        default="~/.alphafold",
        help=(
            "Directory holding params/, for --model af2. The optimize launcher "
            "builds its own singularity line and binds no weight tree, so AF2 "
            "cannot find them at mosaic's default and has to be handed this."
        ),
    )

    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--soft-steps", type=int, default=90)
    parser.add_argument("--sharp-steps", type=int, default=30)
    parser.add_argument(
        "--cycle-noise",
        type=float,
        default=0.1,
        help=(
            "Blend back toward uniform after the between-cycle argmax. 0.0 "
            "restarts the soft phase from a bare vertex, which has no gradient."
        ),
    )

    parser.add_argument(
        "--hard-sharp",
        action="store_true",
        help=(
            "Sharp stage sees a one-hot sequence with soft gradients. Inverts "
            "at some sites; pilot before committing a round to it."
        ),
    )
    parser.add_argument(
        "--hard-sharp-steps",
        type=int,
        default=60,
        help="Sharp steps when --hard-sharp is set; overrides --sharp-steps.",
    )

    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--rescore-recycling", type=int, default=3)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help=(
            "Diffusion samples per loss evaluation. IGNORED for --model af2, "
            "which has no sampler. Memory scales with this times the complex "
            "length; a 201-residue target with a 95+ residue binder overflowed "
            "an 80 GB H100 at 4 in the source campaign."
        ),
    )
    # AF2's MSA stacks dominate memory and step time in a gradient loop, so
    # these sit below the scoring protocol's 512 / 2048. Ignored for boltz2.
    parser.add_argument("--msa-clusters", type=int, default=128)
    parser.add_argument("--extra-msa", type=int, default=512)
    # How far to start from the parent. 0.0 pins the PSSM to it and leaves the
    # optimizer no gradient; 1.0 is a fresh start, which is hallucination.
    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    # Per-process state. One model load and one feature build per binder
    # LENGTH, not per design: the length is baked into the feature shapes, so
    # a shard spanning many lengths pays a JIT recompile at each one. That is
    # why the design set is ordered by length and shards are contiguous.
    cache: dict[int, tuple] = {}
    setup: dict = {}

    def optimize_parent(parent, context, args):
        if not setup:
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
        # this one and the non-finite check below are caught. Everything else
        # propagates and fails the job on purpose -- catching broadly is how a
        # bug comes to look like a biological result.
        unknown = sorted(set(sequence_in) - set(CANONICAL))
        if unknown:
            raise RejectCandidate(f"parent has non-canonical residue(s) {unknown}")

        started = time.time()
        result = refine(
            sequence_in,
            target_sequence=context["target_sequence"],
            msa_path=context.get("target_msa"),
            epitope_idx=setup["epitope"],
            seed=context["seed"] + index,
            args=args,
            cache=cache,
        )
        if not (math.isfinite(result.loss) and math.isfinite(result.start_loss)):
            raise RejectCandidate(
                f"{args.model} returned a non-finite loss "
                f"(start {result.start_loss}, final {result.loss})"
            )

        # Relative to context['structure_dir']. An absolute path is refused so
        # a run directory can be moved; known-issues.md records that having
        # already cost this campaign a day. allow_nan=False so a diverged step
        # writes a loud error here rather than bare `NaN`, which Python emits
        # happily and a strict JSON reader rejects later.
        relative = f"trajectories/{index}.jsonl"
        (setup["structure_dir"] / relative).write_text(
            "".join(
                json.dumps(row, allow_nan=False) + "\n"
                for row in _finite(result.trajectory)
            )
        )

        substitutions = sum(
            1 for was, now in zip(sequence_in, result.sequence) if was != now
        )
        yield {
            "sequence": result.sequence,
            "metrics": {
                # `loss` and `start_loss` are BOTH the ranking objective, one
                # on the discrete child and one on the discrete parent. A final
                # loss means nothing without the number it started from, and
                # only if the two subtract. The training objective is a
                # different and much larger quantity, stays in the trajectory
                # file, and with --hard-sharp is not even comparable between
                # runs.
                "loss": result.loss,
                "start_loss": result.start_loss,
                # NOTE a parent cysteine is mapped to alanine before the walk
                # (NoCys leaves a 19-token alphabet), so a C in the parent
                # counts as a substitution even if the optimizer never moved
                # that position.
                "n_substitutions": float(substitutions),
                "seq_identity": 1.0 - substitutions / max(1, len(sequence_in)),
                # The cycles actually RUN, not the cycles requested: --cycles 0
                # still does one, and storing the request beside one cycle's
                # work would misreport it.
                "opt_cycles": float(result.cycles_run),
                "best_cycle": float(result.best_cycle),
                "hard_sharp": 1.0 if args.hard_sharp else 0.0,
                "loss_model_id": MODEL_IDS[args.model],
            },
            "trajectory": relative,
            "seconds": round(time.time() - started, 2),
        }

    return run_optimization(optimize_parent, parser=build_parser())


def _finite(rows):
    """Trajectory rows with non-finite values dropped rather than serialized.

    A diverged step is worth knowing about, but it belongs in the metrics path
    (which refuses the child) rather than in a JSONL file a later reader will
    choke on.
    """
    for row in rows:
        value = row.get("value")
        if value is None or math.isfinite(value):
            yield row


# ---------------------------------------------------------------------------
# The optimization. Everything mosaic-specific is below here.
# ---------------------------------------------------------------------------


class Result:
    """What one parent produced. A class rather than a tuple because the
    trajectory rows and the winning cycle are easy to transpose positionally."""

    __slots__ = ("sequence", "loss", "start_loss", "trajectory", "best_cycle",
                 "cycles_run")

    def __init__(self, sequence, loss, start_loss, trajectory, best_cycle, cycles_run):
        self.sequence = sequence
        self.loss = loss
        self.start_loss = start_loss
        self.trajectory = trajectory
        self.best_cycle = best_cycle
        self.cycles_run = cycles_run


# Built once per process, not once per parent. `eqx.filter_jit` keys its cache
# on the pytree treedef, which includes the CLASS object -- so defining a fresh
# subclass per parent would miss the cache every time and recompile the whole
# structure model, which is exactly what the per-length `cache` exists to avoid.
_HARD_SEQUENCE = None


def _hard_sequence_class():
    """`HardSequence`, vendored from the campaign's `run_design.py:270`.

        forward:  stop_gradient(hard - soft) + soft  ==  hard
        backward: d/dsoft == 1

    Applied at the OUTERMOST loss, so the structure model, ESM-C and the MPNN
    recovery term all see the same discrete sequence. Wrapping outside `NoCys`
    is correct: argmax over the 19 no-cysteine columns is the same choice as
    argmax over 20 with C excluded.

    CAVEAT, and the reason this belongs on the sharp stage only: the gradient
    is deliberately wrong. Moving the soft values changes nothing until some
    position's argmax flips, then the loss jumps. That makes it a refinement
    operator near a good solution rather than a search operator from a diffuse
    start -- which is the regime a refinement kit is in, since the start IS a
    sequence somebody has.
    """
    global _HARD_SEQUENCE
    if _HARD_SEQUENCE is None:
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        class HardSequence(eqx.Module):
            inner: object

            def __call__(self, sequence, key=None):
                hard = jax.nn.one_hot(
                    jnp.argmax(sequence, axis=-1), sequence.shape[-1]
                )
                ste = jax.lax.stop_gradient(hard - sequence) + sequence
                return self.inner(ste, key=key)

        _HARD_SEQUENCE = HardSequence
    return _HARD_SEQUENCE


def _assert_mosaic_source_is_current() -> None:
    """Fail at startup rather than at hour two of a queued job.

    Two capabilities this objective needs arrived in the host checkout AFTER
    the shared image was built (2026-07-22), and neither is expressible as a
    version tag -- which is why `kit.yaml` verifies files and this re-checks
    the capability itself:

      * `DistogramIPTMProxy.epitope_idx` (commit 6a32fe0, 2026-08-13). Without
        it the proxy rewards confident contact ANYWHERE on the target, which
        RUN_PARAMETERS records as silently optimizing a different objective.
        Verified absent from the shipped image.
      * `models/af2_msa.py` (2026-08-18), for `--model af2` only. The shipped
        image still asserts "AF2 interface does not support MSA yet".

    Checked by inspection rather than by trusting the binding, because a source
    overlay bound at the wrong DEPTH resolves silently: this tool binds onto
    /opt/mosaic/src/mosaic, one level below the scorer's MOSAIC_DEV_SRC.
    """
    import inspect

    import mosaic.losses.structure_prediction as sp

    if "epitope_idx" not in inspect.getsource(sp.DistogramIPTMProxy):
        raise SystemExit(
            "mosaic.losses.structure_prediction.DistogramIPTMProxy has no "
            "`epitope_idx`; this is the pre-2026-08-13 image, where the proxy "
            "rewards contact anywhere on the target and the objective is NOT "
            "the one this kit claims to run. Bind the `mosaic-src` overlay "
            "onto /opt/mosaic/src/mosaic -- see kit.yaml."
        )


def _assert_af2_takes_an_msa() -> None:
    try:
        import mosaic.models.af2_msa  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "mosaic.models.af2_msa is not importable, so AF2 cannot fold "
            "against the campaign alignment and will assert mid-run. Bind the "
            "`mosaic-src` overlay onto /opt/mosaic/src/mosaic -- see kit.yaml "
            f"-- or run with --model boltz2. ({error})"
        ) from error


def _build(binder_length, *, target_sequence, msa_path, epitope_idx, args):
    """The campaign's objective, its ranking objective, and a jitted evaluator.

    The training weights are the constants held fixed across rounds 12-20.
    `BinderTargetContact` at 2.0 and UNCAPPED is the term that controls
    placement: clipping it doubled the distance to target and halved the
    usable count.
    """
    import equinox as eqx
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.common import TOKENS
    from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    from mosaic.losses.transformations import ClippedLoss, NoCys
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.structure_prediction import TargetChain

    _assert_mosaic_source_is_current()

    if args.model == "af2":
        from mosaic.models.af2 import AlphaFold2

        _assert_af2_takes_an_msa()
        folder = AlphaFold2(
            data_dir=args.af2_data_dir,
            multimer=True,
            max_msa_clusters=args.msa_clusters,
            max_extra_msa=args.extra_msa,
        )
    else:
        from mosaic.models.boltz2 import Boltz2

        folder = Boltz2()

    # backbone_noise 0.0 and temp 0.1 are the campaign's values. The reference
    # AF2 kit uses 0.05 / 0.001; a 100x colder MPNN makes the recovery target a
    # near-argmax rather than an average, which is a materially different
    # designability pressure at weight 1.0.
    mpnn = load_mpnn_sol(0.0)
    # -inf for Cys so MPNN does not propose what NoCys removed from the
    # optimizer's alphabet.
    bias = jnp.zeros((binder_length, 20)).at[:, TOKENS.index("C")].set(-1e6)

    # STRUCTURE-CONDITIONED terms only. Everything here is called by the model
    # loss as `term(sequence=..., output=..., key=...)`, so a term that does
    # not accept `output` cannot live in this sum -- see the ESM-C note below.
    objective = (
        2.0 * sp.BinderTargetContact(epitope_idx=epitope_idx, contact_distance=20.0)
        + 1.0 * sp.WithinBinderContact()
        + 1.0 * sp.HelixLoss()
        + 1.0 * sp.DistogramRadiusOfGyration()
        + 2.0 * sp.DistogramIPTMProxy(epitope_idx=epitope_idx, contact_distance=12.0)
        + 1.0 * sp.PLDDTLoss()
        + 1.0 * sp.IPSAE_min()
        + 1.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.1), bias=bias)
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

    # AF2 has no diffusion sampler, so there is no multisample loss and no
    # num_samples: one trunk is all there is to average.
    def build(loss, recycling):
        if HAS_SAMPLER[args.model]:
            return folder.build_multisample_loss(
                loss=loss,
                features=features,
                recycling_steps=recycling,
                num_samples=args.num_samples,
            )
        return folder.build_loss(
            loss=loss, features=features, recycling_steps=recycling
        )

    # ESM-C IS APPLIED TO THE SOFT SEQUENCE DIRECTLY, OUTSIDE THE STRUCTURE
    # LOSS. It has to be: `Boltz2Loss.__call__` invokes its inner loss as
    # `self.loss(sequence=..., output=..., key=...)`, and
    # `ESMCPseudoLikelihood.__call__` is `(seq_standard_tokens, *, key)` --
    # no `output`, no `sequence=` keyword. Putting it inside raises TypeError
    # on the first gradient step, after every model is already on the GPU.
    # `run_design.py:421-422` adds sequence terms to the built structure loss
    # and `NoCys` wraps the sum, which is what this reproduces.
    sequence_term = 0.5 * ClippedLoss(
        ESMCPseudoLikelihood(load_esmc("biohub/ESMC-300M")), 2.0, 100.0
    )
    loss_function = NoCys(build(objective, args.recycling_steps) + sequence_term)

    # The ranking objective is built from THE SAME features as the training
    # loss, so the two describe the same complex. Building a second set from a
    # literal poly-alanine chain would give the diffusion and confidence
    # modules an alanine atom layout with a `res_type` claiming the designed
    # sequence -- a chimera whose ipTM would still subtract cleanly from
    # `start_loss` and so would never be caught.
    ranking = build(
        1.0 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE(),
        args.rescore_recycling,
    )
    ranking = NoCys(ranking)

    # Evaluation happens five-plus times per parent; eager forward passes over
    # a ~280-residue complex are far slower and hungrier than jitted ones.
    return loss_function, ranking, eqx.filter_jit(ranking)


def refine(
    parent_sequence: str,
    *,
    target_sequence: str,
    msa_path: str | None,
    epitope_idx: list[int] | None,
    seed: int,
    args: argparse.Namespace,
    cache: dict[int, tuple],
) -> Result:
    """Walk one parent downhill over `--cycles` soft-sharp cycles."""
    if args.dry_run:
        return _dry_run(parent_sequence, seed, args)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS
    from mosaic.losses.transformations import NoCys
    from mosaic.optimizers import simplex_APGM

    binder_length = len(parent_sequence)
    if binder_length not in cache:
        cache[binder_length] = _build(
            binder_length,
            target_sequence=target_sequence,
            msa_path=msa_path,
            epitope_idx=epitope_idx,
            args=args,
        )
    loss_function, ranking, jit_rank = cache[binder_length]

    sharp_loss = loss_function
    sharp_scale = SHARP["scale"]
    sharp_steps = args.sharp_steps
    if args.hard_sharp:
        sharp_loss = _hard_sequence_class()(loss_function)
        sharp_scale = SHARP_STE_SCALE
        sharp_steps = args.hard_sharp_steps
    jit_sharp = eqx.filter_jit(sharp_loss)

    # THE ONE LINE THAT MAKES THIS REFINEMENT RATHER THAN HALLUCINATION. The
    # hallucination driver starts the PSSM from Gumbel noise; this starts it at
    # the PARENT's one-hot, so the walk leaves a sequence somebody already has.
    # NoCys leaves a 19-token alphabet; cysteine is spliced back at zero
    # probability by NoCys.sequence once the walk is over.
    alphabet = [token for token in TOKENS[:20] if token != "C"]
    seeded = [alphabet.index(residue) for residue in parent_sequence.replace("C", "A")]
    one_hot = jax.nn.one_hot(jnp.array(seeded), len(alphabet))
    x = one_hot * (1.0 - args.epsilon) + args.epsilon / len(alphabet)

    def decode(pssm) -> str:
        return "".join(TOKENS[t] for t in NoCys.sequence(pssm).argmax(-1))

    trajectory: list[dict] = []
    best_x, best_value, best_cycle = None, None, 1
    key = jax.random.key(seed)
    cycles = max(1, args.cycles)

    for cycle in range(1, cycles + 1):
        stage = {"cycle": cycle, "stage": "soft"}
        offset = len(trajectory)
        record = lambda aux, _x, s=stage, o=offset: trajectory.append(
            {**s, "step": len(trajectory) - o, "value": float(aux["loss"])}
        )

        # ---- soft stage ----
        # simplex_APGM returns (final_iterate, best_iterate, trajectory).
        # Continue from the FINAL iterate: best_x is a snapshot whose momentum
        # history belongs to a different point, and mosaic itself records that
        # its recorded score was evaluated at the extrapolated point rather
        # than at x. run_design.py:547 takes the final iterate here for exactly
        # that reason; taking best_x instead quietly runs a different protocol.
        key, soft_key = jax.random.split(key)
        x_soft, _, _ = simplex_APGM(
            loss_function=loss_function,
            x=x,
            n_steps=args.soft_steps,
            stepsize=SOFT["stepsize"],
            momentum=SOFT["momentum"],
            scale=SOFT["scale"],
            logspace=False,
            # None, not 1.0 -- see the SOFT/SHARP note above.
            max_gradient_norm=None,
            # Passed explicitly: without a key simplex_APGM draws one from an
            # unseeded numpy RNG, and re-running a preempted shard would give
            # different children while the run record still named one seed.
            key=soft_key,
            trajectory_fn=record,
        )

        # ---- sharp stage ----
        stage = {"cycle": cycle, "stage": "sharp"}
        offset = len(trajectory)
        record = lambda aux, _x, s=stage, o=offset: trajectory.append(
            {**s, "step": len(trajectory) - o, "value": float(aux["loss"])}
        )
        key, sharp_key = jax.random.split(key)
        # Here the BEST iterate is what the reference keeps (run_design.py:581)
        # -- the sharp stage is not resumed from, it is the endpoint.
        _, cycle_x, _ = simplex_APGM(
            loss_function=sharp_loss,
            x=x_soft,
            n_steps=sharp_steps,
            stepsize=SHARP["stepsize"],
            momentum=SHARP["momentum"],
            scale=sharp_scale,
            logspace=False,
            max_gradient_norm=None,
            key=sharp_key,
            trajectory_fn=record,
        )

        # ---- keep the best across cycles ----
        # Re-evaluated rather than taken from the optimizer, which makes cycles
        # comparable to each other. Matches run_design.py:602-607.
        key, eval_key = jax.random.split(key)
        value = float(jit_sharp(cycle_x, key=eval_key)[0])
        if best_value is None or value < best_value:
            best_x, best_value, best_cycle = cycle_x, value, cycle

        # The decoded sequence at each cycle boundary. This is what makes
        # "how much identity survived" answerable THROUGH the optimization
        # rather than only end to end.
        trajectory.append(
            {
                "cycle": cycle,
                "stage": "endpoint",
                "value": value,
                "sequence": decode(cycle_x),
            }
        )

        # ---- re-soften for the next cycle ----
        # ARGMAX to a one-hot vertex, THEN blend toward uniform. The argmax is
        # the point -- it forces commitment to a discrete sequence before the
        # next soft phase. Carrying a soft blend forward is a different
        # operation and does not reproduce the measured result.
        if cycle < cycles:
            weight = args.cycle_noise
            hard = jax.nn.one_hot(jnp.argmax(cycle_x, axis=-1), cycle_x.shape[-1])
            x = (1.0 - weight) * hard + weight * (1.0 / cycle_x.shape[-1])

    sequence = decode(best_x)

    # Both numbers are the RANKING objective on a discrete sequence: the child
    # here, the parent below. Same function, same key, so they subtract -- and
    # so they stay comparable across --hard-sharp, which shifts the TRAINING
    # loss by several units and would otherwise make the pair meaningless.
    # NoCys-wrapped, so both take the 19-token simplex the walk used.
    def as_simplex(seq: str):
        return jax.nn.one_hot(
            jnp.array([alphabet.index(r) for r in seq.replace("C", "A")]),
            len(alphabet),
        )

    final, _ = jit_rank(as_simplex(sequence), key=jax.random.key(0))
    start, _ = jit_rank(as_simplex(parent_sequence), key=jax.random.key(0))

    return Result(
        sequence, float(final), float(start), trajectory, best_cycle, cycles
    )


def _dry_run(parent_sequence: str, seed: int, args: argparse.Namespace) -> Result:
    """Contract plumbing without jax, mosaic, weights or a GPU.

    The validator's whole value is that it costs seconds. Loading Boltz-2 or
    AF2 plus ESM-C costs minutes and proves nothing about the row shapes, which
    is the only thing the validator checks.
    """
    rng = random.Random(seed)
    alphabet = "ADEFGHIKLMNPQRSTVWY"
    mutated = list(parent_sequence)
    for position in rng.sample(range(len(mutated)), k=min(3, len(mutated))):
        mutated[position] = rng.choice(alphabet)

    sharp_steps = args.hard_sharp_steps if args.hard_sharp else args.sharp_steps
    cycles = max(1, args.cycles)
    trajectory: list[dict] = []
    value = 1.0
    for cycle in range(1, cycles + 1):
        for stage, count in (("soft", args.soft_steps), ("sharp", sharp_steps)):
            for step in range(max(0, count)):
                value -= 0.001
                trajectory.append(
                    {"cycle": cycle, "stage": stage, "step": step, "value": value}
                )
        trajectory.append(
            {
                "cycle": cycle,
                "stage": "endpoint",
                "value": value,
                "sequence": "".join(mutated),
            }
        )
    return Result("".join(mutated), value, 1.0, trajectory, cycles, cycles)


if __name__ == "__main__":
    raise SystemExit(main())
