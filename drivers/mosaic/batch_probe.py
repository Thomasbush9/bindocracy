#!/usr/bin/env python3
"""Does batching the mosaic optimizer buy anything, and where does it OOM?

`batched_simplex_APGM` advances B sequences per step via vmap. vmap does not
reduce FLOPs -- it improves utilization -- so if a single Boltz2 forward and
backward at this target size already saturates the GPU, B=10 costs ten times as
long per step and the batching has bought nothing but a memory ceiling. Nobody
has measured which regime this target is in, and a population-based optimizer is
designed very differently depending on the answer.

This probe answers it in one job. It builds the real loss (imported from
`hallucinate_binders.py`, so the nine terms, sample count and recycling match
what a real run uses), then times a handful of optimization steps at each batch
size and reports seconds per step per sequence.

    singularity/mosaic-exec.sh python batch_probe.py \\
        --target-fasta <fa> --target-msa <a3m> --binder-length 80 \\
        --batch-sizes 1,2,4,8,16 --steps 5

Read the `s/step/seq` column. Flat means batching is free throughput and the
population should be micro-batched as large as memory allows. Rising linearly
means the device is already saturated at B=1, batching buys nothing, and the
driver should keep the simple member loop.

The `reference` row times plain `simplex_APGM` on one sequence -- today's code
path. `batched_simplex_APGM` round-trips through numpy for the simplex
projection on every step, so B=1 batched is not guaranteed to match it, and the
member loop is only the cheaper option if it actually is.

Compilation is paid per batch size, because each B is a new shape. It is timed
separately and excluded: a real run pays it once, so it belongs in neither the
throughput number nor the decision.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mosaic.optimizers import batched_simplex_APGM, simplex_APGM

# The loss must be the one a real run uses, or the timings describe nothing.
# The driver is a script rather than a package, so it is loaded by path.
_DRIVER = Path(__file__).resolve().parent / "hallucinate_binders.py"
_spec = importlib.util.spec_from_file_location("hallucinate_binders", _DRIVER)
_driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_driver)


def peak_gib() -> float | None:
    """Peak device memory, when the backend reports it."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 -- not all backends implement this
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return peak / 1024**3 if peak else None


def init_population(batch: int, binder_length: int, seed: int) -> jnp.ndarray:
    """The driver's own initialisation, widened to a batch.

    19 tokens, not 20: NoCys removes cysteine from the optimizer's alphabet and
    splices it back with zero probability at the end.
    """
    keys = jax.random.split(jax.random.key(seed), batch)
    raw = jnp.stack([
        np.random.uniform(low=0.25, high=0.75)
        * jax.random.gumbel(key=key, shape=(binder_length, 19))
        for key in keys
    ])
    return jax.nn.softmax(raw, axis=-1)


def time_steps(call, n_steps: int) -> tuple[float, float]:
    """Return (compile+first-step seconds, seconds per subsequent step).

    The optimizer prints a line per sequence per step; that output is the
    driver's trace format and is pure noise here, so it is swallowed.
    """
    sink = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(sink):
        call(1)
    warm = time.time() - t0

    t0 = time.time()
    with contextlib.redirect_stdout(sink):
        call(n_steps)
    steady = (time.time() - t0) / n_steps
    return warm, steady


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-fasta", required=True)
    ap.add_argument("--target-msa", required=True)
    ap.add_argument("--binder-length", type=int, required=True)
    ap.add_argument("--epitope", default="",
                    help="comma-separated 0-based indices, as the driver takes them")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--steps", type=int, default=5,
                    help="timed steps per batch size, after the compile step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    batches = [int(piece) for piece in args.batch_sizes.split(",") if piece.strip()]

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}", flush=True)
    target = _driver.read_fasta(args.target_fasta)
    epitope = _driver.parse_epitope(args.epitope, len(target))
    print(f"target: {len(target)} aa   binder: {args.binder_length} aa", flush=True)
    print(f"epitope: {epitope if epitope else 'none (whole target)'}", flush=True)

    t0 = time.time()
    _, loss = _driver.build(target, args.target_msa, args.binder_length, epitope)
    print(f"models + loss built in {time.time() - t0:.1f}s\n", flush=True)

    # The soft stage's settings, which is where a cull would fire.
    stepsize = 0.2 * np.sqrt(args.binder_length)
    shared = dict(loss_function=loss, stepsize=stepsize, momentum=0.3, scale=1.00,
                  logspace=False, max_gradient_norm=1.0)

    rows = []

    x = init_population(1, args.binder_length, args.seed)[0]
    warm, steady = time_steps(
        lambda n: simplex_APGM(x=x, n_steps=n, **shared), args.steps)
    rows.append({"batch": 1, "path": "simplex_APGM", "compile_s": round(warm, 1),
                 "s_per_step": round(steady, 2), "s_per_step_per_seq": round(steady, 2),
                 "speedup": 1.0, "peak_gib": peak_gib()})
    reference = steady

    for batch in batches:
        x = init_population(batch, args.binder_length, args.seed)
        try:
            warm, steady = time_steps(
                lambda n, x=x: batched_simplex_APGM(x=x, n_steps=n, **shared), args.steps)
        except Exception as exc:  # noqa: BLE001 -- an OOM is a result, not a failure
            text = str(exc)
            oom = "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()
            rows.append({"batch": batch, "path": "batched", "error":
                         "OOM" if oom else type(exc).__name__})
            print(f"b={batch:<4} {'OOM' if oom else type(exc).__name__} "
                  f"-- stopping the sweep here", flush=True)
            if oom:
                break
            raise
        per_seq = steady / batch
        rows.append({"batch": batch, "path": "batched", "compile_s": round(warm, 1),
                     "s_per_step": round(steady, 2),
                     "s_per_step_per_seq": round(per_seq, 3),
                     "speedup": round(reference / per_seq, 2), "peak_gib": peak_gib()})
        print(f"b={batch:<4} {steady:7.2f} s/step   {per_seq:6.3f} s/step/seq   "
              f"{reference / per_seq:5.2f}x   compile {warm:6.1f}s", flush=True)

    print(f"\n{'batch':>6} {'path':>14} {'s/step':>9} {'s/step/seq':>12} "
          f"{'speedup':>9} {'peak GiB':>10}")
    for row in rows:
        if "error" in row:
            print(f"{row['batch']:>6} {row['path']:>14} {row['error']:>9}")
            continue
        peak = f"{row['peak_gib']:.1f}" if row["peak_gib"] else "-"
        print(f"{row['batch']:>6} {row['path']:>14} {row['s_per_step']:>9.2f} "
              f"{row['s_per_step_per_seq']:>12.3f} {row['speedup']:>8.2f}x {peak:>10}")

    usable = [r for r in rows if r.get("path") == "batched" and "error" not in r]
    if usable:
        best = max(usable, key=lambda r: r["speedup"])
        print(f"\nbest throughput at b={best['batch']}: {best['speedup']:.2f}x "
              f"per sequence against the plain optimizer")
        if best["speedup"] < 1.3:
            print("under 1.3x -- the device is already saturated at b=1. Keep the "
                  "member loop; micro-batching is complexity without throughput.")
        else:
            print(f"worth micro-batching: use b={best['batch']} and size the "
                  "population independently of it.")

    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
