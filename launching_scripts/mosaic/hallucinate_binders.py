"""Mosaic hallucination-based binder design, parameterised.

This is a lightly generalised copy of
`mosaic_setup/mosaic/hallucinate/hallucinate.py`. It is copied rather than
imported for one reason: the upstream script hard-codes both the target
sequence and

    MSA_PATH = ".../tbush/mosaic_setup/dio3_cut/.../DIO3.a3m"

which no longer exists — the tree moved under `binder_design/` — so the
upstream script cannot run as-is (see docs/known-issues.md). Everything that
matters scientifically (the loss, the three-stage APGM schedule, the ranking
re-fold) is preserved verbatim so results stay comparable to the runs the lab
has already done.

Run inside the mosaic container, one GPU per process:

    singularity/mosaic-exec.sh python hallucinate_binders.py \
        --target-fasta <fa> --msa-path <a3m> --binder-length 80 \
        --n-designs 40 --save-dir <dir>

Sequences are appended as each finishes, so a job killed by the walltime still
leaves everything completed up to that point.

The models and the loss are built ONCE per process. Boltz2() loads a 2.3 GB
torch checkpoint and converts it to Equinox, which costs minutes; rebuilding it
per design would dominate the run.
"""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.models.boltz2 import Boltz2
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain


def read_fasta(path: str) -> str:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return "".join(ln for ln in lines if not ln.startswith(">")).upper()


def build(target_sequence: str, msa_path: str, binder_length: int):
    """Load the models and build the design objective. Once per process."""
    folder = Boltz2()
    mpnn = load_mpnn_sol(0.05)

    # -inf for Cys in the binder for MPNN. NoCys below already removes cysteine
    # from the optimizer's alphabet; this stops MPNN proposing it either.
    bias = jnp.zeros((binder_length, 20)).at[:, TOKENS.index("C")].set(-1e6)

    sp_loss = (
        sp.BinderTargetContact()
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
        chains=[TargetChain(sequence=target_sequence, use_msa=True, msa_path=msa_path)],
    )

    loss = NoCys(
        folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=1,
            num_samples=4,
        )
    )
    return folder, loss


def design(folder, loss, seed, target_sequence, msa_path, binder_length):
    # NoCys makes the optimizer's alphabet 19 tokens, not 20 — cysteine is
    # spliced back in with zero probability by NoCys.sequence below.
    _pssm = np.random.uniform(low=0.25, high=0.75) * jax.random.gumbel(
        key=jax.random.key(seed),
        shape=(binder_length, 19),
    )

    _, pssm = simplex_APGM(
        loss_function=loss,
        x=jax.nn.softmax(_pssm),
        stepsize=0.2 * np.sqrt(binder_length),
        n_steps=100,
        momentum=0.3,
        scale=1.00,
        logspace=False,
        max_gradient_norm=1.0,
    )

    # sharpen the PSSM towards a discrete sequence
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        stepsize=0.5 * np.sqrt(binder_length),
        n_steps=50,
        momentum=0.0,
        scale=1.25,
        logspace=True,
        max_gradient_norm=1.0,
    )
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=15,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.4,
        logspace=True,
        max_gradient_norm=1.0,
    )

    pssm = NoCys.sequence(pssm)
    seq = pssm.argmax(-1)
    seq_str = "".join(TOKENS[i] for i in seq)

    # Re-fold the discrete binder alongside the target and score it. This is a
    # separate, more expensive prediction (3 recycles, 6 samples) than the one
    # used inside the optimisation loop, and it is what the reported score means.
    boltz_features, _ = folder.target_only_features(
        chains=[
            TargetChain(sequence=seq_str, use_msa=False),
            TargetChain(sequence=target_sequence, use_msa=True, msa_path=msa_path),
        ]
    )
    ranking_loss = folder.build_multisample_loss(
        loss=1.00 * sp.IPTMLoss()
        + 0.5 * sp.TargetBinderIPSAE()
        + 0.5 * sp.BinderTargetIPSAE(),
        features=boltz_features,
        recycling_steps=3,
        num_samples=6,
    )
    loss_value, _ = ranking_loss(jax.nn.one_hot(seq, 20), key=jax.random.key(0))
    return seq_str, loss_value.item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fasta", required=True)
    ap.add_argument("--msa-path", required=True)
    ap.add_argument("--binder-length", type=int, default=80)
    ap.add_argument("--n-designs", type=int, default=40,
                    help="stop after this many designs (0 = time-bounded)")
    ap.add_argument("--max-runtime", type=float, default=11.0,
                    help="hours; set below the job walltime so the last design "
                         "finishes and gets written")
    ap.add_argument("--array-id", type=int, default=0,
                    help="names the output file and seeds the RNG, so parallel "
                         "tasks explore different inits")
    ap.add_argument("--save-dir", required=True)
    a = ap.parse_args()

    target_sequence = read_fasta(a.target_fasta)
    os.makedirs(a.save_dir, exist_ok=True)
    out_path = f"{a.save_dir}/designs_{a.array_id}.txt"
    jsonl_path = f"{a.save_dir}/designs_{a.array_id}.jsonl"

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}", flush=True)
    print(f"target: {len(target_sequence)} aa   binder: {a.binder_length} aa", flush=True)

    t_build = time.time()
    folder, loss = build(target_sequence, a.msa_path, a.binder_length)
    build_s = time.time() - t_build
    print(f"models + loss built in {build_s:.1f}s", flush=True)

    start = time.time()
    n = 0
    max_runtime_sec = a.max_runtime * 3600
    while time.time() - start < max_runtime_sec:
        if a.n_designs and n >= a.n_designs:
            break
        # Distinct per (task, design) so no two trajectories share an init.
        seed = a.array_id * 100_000 + n
        t0 = time.time()
        seq, loss_value = design(
            folder, loss, seed, target_sequence, a.msa_path, a.binder_length
        )
        dt = time.time() - t0
        n += 1
        with open(out_path, "a") as f:
            f.write(f">{loss_value:.4f}\n{seq}\n")
        # A machine-readable sibling of the FASTA-ish file above: the benchmark
        # needs per-design wall-clock, which the .txt format cannot carry.
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({
                "index": n, "seed": seed, "sequence": seq,
                "score": loss_value, "seconds": round(dt, 1),
            }) + "\n")
        print(f"[{n}] loss={loss_value:.4f} ({dt:.0f}s) {seq}", flush=True)

    total_min = (time.time() - start) / 60
    print(f"done: {n} designs in {total_min:.1f} min "
          f"({total_min / max(n, 1):.1f} min/design) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
