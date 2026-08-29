"""Mosaic hallucination-based binder design: one task, one process, one GPU.

A lightly generalised copy of `mosaic_setup/mosaic/hallucinate/hallucinate.py`.
It is copied rather than imported because the upstream script hard-codes the
target sequence, the binder length, and an MSA path that no longer exists (see
docs/known-issues.md). Everything that matters scientifically — the nine-term
loss, the three-stage APGM schedule, the ranking re-fold — is preserved
verbatim so results stay comparable to the runs the lab has already done.

Nothing campaign-specific is hard-coded here. Every run-dependent value arrives
as an argument, which is what lets the harness archive this file per run and
execute the archived copy.

Run inside the mosaic container:

    singularity/mosaic-exec.sh python hallucinate_binders.py \
        --target-fasta <fa> --target-msa <a3m> --binder-length 80 \
        --task-id 0 --seed-base 0 --n-designs 40 --max-runtime 11 \
        --save-dir <dir>

Output contract, consumed by `bindocracy.adapters.mosaic`:

    <save-dir>/designs.jsonl   one JSON object per completed design, appended
                               and fsynced, so a task killed by the walltime
                               still leaves every fully written record
    <save-dir>/status.json     written once, atomically, when the task stops

The process exits 0 whenever it managed to write `status.json`. Success is
`status.json` plus the designs it accounts for, never the exit code.

The models and the loss are built ONCE per process. Boltz2() loads a 2.3 GB
torch checkpoint and converts it to Equinox, which costs minutes; rebuilding it
per design would dominate the run.
"""

import argparse
import json
import os
import time
import traceback
from datetime import UTC, datetime
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

DESIGNS_FILE = "designs.jsonl"
STATUS_FILE = "status.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


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


def design(folder, loss, seed, target_sequence, msa_path, binder_length, schedule):
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
        n_steps=schedule["soft"],
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
        n_steps=schedule["sharpen"],
        momentum=0.0,
        scale=1.25,
        logspace=True,
        max_gradient_norm=1.0,
    )
    pssm, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(pssm + 1e-5),
        n_steps=schedule["final"],
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


def write_status(save_dir: Path, payload: dict) -> None:
    """Write status.json atomically so a reader never sees a half file."""
    tmp = save_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, save_dir / STATUS_FILE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-fasta", required=True)
    ap.add_argument("--target-msa", required=True)
    ap.add_argument("--binder-length", type=int, required=True)
    ap.add_argument("--task-id", type=int, required=True,
                    help="names the designs and offsets the seeds, so parallel "
                         "tasks explore different inits")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--n-designs", type=int, required=True)
    ap.add_argument("--max-runtime", type=float, required=True,
                    help="hours; set below the job walltime so the last design "
                         "finishes and gets written")
    ap.add_argument("--save-dir", required=True)
    # The APGM schedule: soft optimization, then two sharpening passes. The
    # defaults are the lab's benchmarked settings, at ~7 min per design.
    ap.add_argument("--soft-steps", type=int, default=100)
    ap.add_argument("--sharpen-steps", type=int, default=50)
    ap.add_argument("--final-steps", type=int, default=15)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    designs_path = save_dir / DESIGNS_FILE

    started_at = now()
    attempted = 0
    produced = 0
    error = None

    schedule = {"soft": a.soft_steps, "sharpen": a.sharpen_steps, "final": a.final_steps}

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}", flush=True)
    try:
        target_sequence = read_fasta(a.target_fasta)
        print(f"target: {len(target_sequence)} aa   binder: {a.binder_length} aa", flush=True)
        print(f"schedule: {schedule}", flush=True)

        t_build = time.time()
        folder, loss = build(target_sequence, a.target_msa, a.binder_length)
        print(f"models + loss built in {time.time() - t_build:.1f}s", flush=True)

        deadline = time.monotonic() + a.max_runtime * 3600
        with designs_path.open("a") as out:
            while produced < a.n_designs and time.monotonic() < deadline:
                index = produced
                # Distinct per (task, design) so no two trajectories share an init.
                seed = a.seed_base + a.task_id * 100_000 + index
                attempted += 1
                t0 = time.time()
                sequence, ranking_loss = design(
                    folder, loss, seed, target_sequence, a.target_msa,
                    a.binder_length, schedule,
                )
                seconds = time.time() - t0
                out.write(json.dumps({
                    "native_id": f"task-{a.task_id:04d}-design-{index:06d}",
                    "sequence": sequence,
                    "seed": seed,
                    "ranking_loss": ranking_loss,
                    "completed_at": now(),
                    "seconds": round(seconds, 1),
                }) + "\n")
                out.flush()
                os.fsync(out.fileno())
                produced += 1
                print(f"[{produced}] loss={ranking_loss:.4f} ({seconds:.0f}s) {sequence}",
                      flush=True)
    except Exception as exc:  # noqa: BLE001 -- any failure must still write status.json
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    if error is not None or produced == 0:
        status = "failed"
    elif produced >= a.n_designs:
        status = "succeeded"
    else:
        status = "partial"

    write_status(save_dir, {
        "task_id": a.task_id,
        "status": status,
        "started_at": started_at,
        "finished_at": now(),
        "n_attempted": attempted,
        "n_produced": produced,
        "output_file": DESIGNS_FILE,
        "error": error,
    })
    print(f"{status}: {produced}/{a.n_designs} designs -> {designs_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
