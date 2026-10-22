"""Mosaic binder design with a BATCHED optimizer and soft/sharp CYCLES.

This is the lab's current strategy (tomasz's `mosaic_test/mosaic/run_design.py`,
as run for the DIO3 R32/R35 1000-trajectory runs) rewritten against
bindocracy's driver contract. `hallucinate_binders.py` remains the single-
trajectory three-stage driver; this is the alternative schedule, and the two
differ in three ways that matter:

  * PARALLELISM. `hallucinate_binders.py` runs one trajectory at a time through
    `simplex_APGM`, N designs sequentially. This runs B trajectories at once
    through `batched_simplex_APGM`, which vmaps them over a shared loss object
    and ONE JIT compile. The compile is the dominant fixed cost (~3 min), so B
    trajectories cost barely more than 1 up to the memory ceiling.

  * SCHEDULE. Not soft -> sharpen -> final (once), but C repetitions of
    [soft -> sharp], re-softening between cycles: argmax the sharp endpoint to a
    one-hot vertex, blend it back toward uniform, and run soft again. The argmax
    is the point -- it forces commitment to a discrete sequence before the next
    soft phase. Upstream reached the same place accidentally by re-seeding one
    round's decoded designs into the next; the cycle loop makes it explicit and
    skips the model reload between rounds.

  * STEPSIZE CONVENTION. `hallucinate_binders.py` scales stepsize by sqrt(L)
    (0.2*sqrt(L) ~ 1.73 at L=75). The lab's cycle config uses ABSOLUTE
    stepsizes, 0.05 soft and 0.025 sharp -- ~35x smaller. The defaults below are
    the lab's absolute values, because that is the schedule this file ports.
    Do not reintroduce the sqrt(L) factor without re-tuning both stages.

ONE DELIBERATE DEVIATION from the lab's recipe: they sum TWO structure
backends, boltz2 and of3, at weight 1.0 each. This ports the loss onto boltz2
alone, because of3 submits the design to the ColabFold MSA server on every
build (visible in their job logs) -- an external dependency inside the
optimization loop, which bindocracy's drivers do not take. Adding of3 back is a
two-line change in `build`, at the cost of that network call.

The cycle boundary is also the natural seam for selection. Each member
currently re-softens from ITS OWN argmax, which makes the B trajectories
independent -- B parallel runs that never interact. Replacing "its own" with "a
parent chosen by rank" is the whole of an evolutionary scheme, and it is one
gather: see `--select` and `drivers/mosaic/evolution.py`.

Run inside the mosaic container:

    singularity/mosaic-exec.sh python cycle_binders.py \
        --target-fasta <fa> --target-msa <a3m> --binder-length 75 \
        --task-id 0 --seed-base 0 --n-designs 4 --max-runtime 11 \
        --save-dir <dir> --batch 4 --cycles 3

Output contract is `hallucinate_binders.py`'s, unchanged, so
`bindocracy.adapters.mosaic` consumes this driver's runs without knowing which
schedule produced them:

    <save-dir>/designs.jsonl   one JSON object per completed design
    <save-dir>/status.json     written once, atomically, when the task stops

Each row additionally carries `cycle_loss` (the in-loop objective at the
winning cycle) next to the required `ranking_loss` (the independent re-fold).
Those two are NOT interchangeable: `cycle_loss` is the quantity being optimized
and correlates with the re-fold partly because both are Boltz2.
"""

import argparse
import json
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mosaic.losses.structure_prediction as sp
import numpy as np
from mosaic.common import TOKENS, LossTerm
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.models.boltz2 import Boltz2
from mosaic.optimizers import batched_eval, batched_simplex_APGM
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain

DESIGNS_FILE = "designs.jsonl"
STATUS_FILE = "status.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_fasta(path: str) -> str:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return "".join(ln for ln in lines if not ln.startswith(">")).upper()


def read_fasta_records(path: str) -> list[str]:
    """Every sequence in a multi-FASTA, in file order."""
    return [
        ln.strip().upper()
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.startswith(">")
    ]


class RootCysteines(LossTerm):
    """A fixed-shape 19-token search with root-specific, immutable C positions."""

    loss: object
    cysteines: jax.Array

    def sequence(self, sequence):
        full = NoCys.sequence(sequence)
        fixed = jax.nn.one_hot(TOKENS.index("C"), len(TOKENS))
        return jnp.where(self.cysteines[:, None], fixed, full)

    def __call__(self, sequence, *, key):
        value, aux = self.loss(self.sequence(sequence), key=key)
        value = eqx.error_if(value, ~jnp.isfinite(value), "nonfinite campaign objective")
        return value, aux

    def for_root(self, sequence: str):
        mask = jnp.asarray([aa == "C" for aa in sequence])
        if mask.shape != self.cysteines.shape:
            raise ValueError("root length differs from the compiled loss")
        allowed = jnp.where(
            mask[:, None],
            jax.nn.one_hot(TOKENS.index("C"), len(TOKENS), dtype=bool),
            jnp.arange(len(TOKENS))[None, :] != TOKENS.index("C"),
        )
        bias = jnp.where(allowed, 0.0, -1e6)

        def biases(tree):
            terms = jax.tree_util.tree_leaves(
                tree.loss, is_leaf=lambda node: isinstance(node, InverseFoldingSequenceRecovery)
            )
            return tuple(term.bias for term in terms
                         if isinstance(term, InverseFoldingSequenceRecovery))

        if len(biases(self)) != 1:
            raise ValueError("expected exactly one inverse-folding recovery term")
        updated = eqx.tree_at(biases, self, (bias,))
        return eqx.tree_at(lambda tree: tree.cysteines, updated, mask)


def build(
    target_sequence: str,
    msa_path: str,
    binder_length: int,
    epitope_idx: list[int] | None = None,
    num_samples: int = 1,
    *,
    root_constraints: bool = False,
):
    """Load Boltz2 and build the lab's seven-term cycle objective. Once per process.

    This is the R32/R35 loss stack, not `hallucinate_binders.py`'s nine-term
    one. The differences are deliberate and worth stating, because the two
    drivers' losses are not comparable:

      * Two distogram-only terms carry most of the weight -- `DistogramIPTMProxy`
        at 2.0 and `BinderTargetContact` at 2.0. The hallucinate loss has no
        distogram term at all and leans on 10.0x inverse-folding recovery.
      * `HelixLoss` and `DistogramRadiusOfGyration` constrain the binder's own
        shape (secondary structure, compactness). The hallucinate loss has no
        equivalent and relies on `WithinBinderPAE` instead.
      * `IPSAE_min` replaces the separate Target->Binder / Binder->Target IPSAE
        terms, taking the worse direction rather than averaging the two.
      * MPNN enters at 1.0, not 10.0, and at temp 0.1, not 0.001.

    `num_samples` is 1 in the lab's config (with `deterministic=True` on the
    backend) against hallucinate's 4. A confidence term is enabled either way,
    so JAX cannot prune the structure module and the step cost is set by the
    forward pass, not by this number.
    """
    folder = Boltz2()
    mpnn = load_mpnn_sol(0.05)

    # -inf for Cys in the binder for MPNN. NoCys below already removes cysteine
    # from the optimizer's alphabet; this stops MPNN proposing it either.
    bias = jnp.zeros((binder_length, 20)).at[:, TOKENS.index("C")].set(-1e6)

    sp_loss = (
        # The two terms the epitope reaches.
        2.0 * sp.BinderTargetContact(epitope_idx=epitope_idx, contact_distance=20.0)
        + 2.0 * sp.DistogramIPTMProxy(epitope_idx=epitope_idx, contact_distance=12.0)
        # The binder's own fold: contact order, helicity, compactness.
        + 1.0 * sp.WithinBinderContact(
            max_contact_distance=14.0,
            min_sequence_separation=8,
            num_contacts_per_residue=25,
        )
        + 1.0 * sp.HelixLoss(max_distance=6.0, target_value=-2.0)
        + 1.0 * sp.DistogramRadiusOfGyration(target_radius=None)
        # Confidence.
        + 1.0 * sp.PLDDTLoss()
        + 1.0 * sp.IPSAE_min()
        # Designability. Weight 1.0 here, where hallucinate_binders.py uses 10.0.
        + 1.0 * InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(0.1), bias=bias)
    )

    features, _ = folder.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(sequence=target_sequence, use_msa=True, msa_path=msa_path)],
    )

    wrapper = RootCysteines if root_constraints else NoCys
    kwargs = {"cysteines": jnp.zeros(binder_length, dtype=bool)} if root_constraints else {}
    loss = wrapper(
        folder.build_multisample_loss(
            loss=sp_loss,
            features=features,
            recycling_steps=1,
            num_samples=num_samples,
        ),
        **kwargs,
    )
    return folder, loss


def initial_population(
    key: jax.Array,
    batch: int,
    binder_length: int,
    n_tokens: int,
    seed: int,
    init_fasta: str | None,
    init_noise: float,
    noise_random: bool,
) -> jnp.ndarray:
    """[B, L, K] starting simplex points, from noise or from proposed sequences.

    Seeding from a FASTA is how the lab's runs start: the 1000 best of a 20000-
    sequence pre-screen, so the optimizer refines a plausible point rather than
    searching from noise. Two details are load-bearing:

      * The alphabet is 19 tokens under NoCys, with C REMOVED, so the one-hot
        must be built in that alphabet. Building it in TOKENS puts every residue
        from index 4 onward in the wrong column.
      * A hard vertex of the simplex has no room to move under a projected-
        gradient step, so the one-hot is blended off the vertex. Toward uniform
        it is a pure function of the seed, which makes replicas of one parent
        IDENTICAL; `noise_random` draws a Dirichlet per trajectory instead,
        which is what makes K children of one parent genuinely different. That
        distinction is the difference between a resample operator that works and
        one that silently clones.
    """
    if not init_fasta:
        # Near-uniform with noise, so the B trajectories diverge instead of
        # collapsing onto one path.
        x = jax.random.uniform(key, (batch, binder_length, n_tokens), minval=0.0, maxval=1.0)
        return x / x.sum(-1, keepdims=True)

    seqs = read_fasta_records(init_fasta)
    if not seqs:
        raise SystemExit(f"no sequences in {init_fasta}")
    alphabet = TOKENS.replace("C", "") if n_tokens == 19 else TOKENS

    rows = []
    for b in range(batch):
        # Offset by seed so parallel tasks refine DIFFERENT proposals rather
        # than every task starting from the same first B sequences.
        s = seqs[(seed * batch + b) % len(seqs)]
        if len(s) != binder_length:
            raise SystemExit(
                f"--init-fasta sequence {b} is {len(s)} aa, expected {binder_length}"
            )
        idx = [alphabet.index(c) if c in alphabet else alphabet.index("A") for c in s]
        rows.append(np.eye(n_tokens, dtype=np.float32)[idx])
    seeded = jnp.asarray(np.stack(rows))

    if noise_random:
        return (1.0 - init_noise) * seeded + init_noise * jax.random.dirichlet(
            key, jnp.ones(n_tokens), shape=seeded.shape[:-1]
        )
    return (1.0 - init_noise) * seeded + init_noise * (1.0 / n_tokens)


def resoften(x: jnp.ndarray, noise: float) -> jnp.ndarray:
    """Argmax to a one-hot vertex, then blend back toward uniform.

    The mutation operator of the cycle schedule, and the only place between
    cycles where information is intentionally DESTROYED: everything the soft
    phase learned about the shape of the distribution is collapsed to the
    winning token per position. Carrying the soft blend forward would not be the
    same thing -- it would be a warm restart, not a commitment.
    """
    hard = jax.nn.one_hot(jnp.argmax(x, axis=-1), x.shape[-1])
    return (1.0 - noise) * hard + noise * (1.0 / x.shape[-1])


def optimize_cycle(loss, x, key, schedule, *, label="cycle"):
    """Run the shared soft/sharp optimizer, returning both resumable endpoints."""
    _, sharp_key = jax.random.split(key)
    t0 = time.time()
    print(f"{label}: soft stage starting ({schedule['soft_steps']} steps)", flush=True)
    soft, _ = batched_simplex_APGM(
        loss_function=loss, x=x, key=key,
        n_steps=schedule["soft_steps"], stepsize=schedule["soft_stepsize"],
        momentum=schedule["soft_momentum"], scale=schedule["soft_scale"],
        logspace=False, max_gradient_norm=None,
    )
    print(f"{label}: soft stage finished ({time.time() - t0:.1f}s); sharp starting", flush=True)
    final, best = batched_simplex_APGM(
        loss_function=loss, x=soft, key=sharp_key,
        n_steps=schedule["sharp_steps"], stepsize=schedule["sharp_stepsize"],
        momentum=schedule["sharp_momentum"], scale=schedule["sharp_scale"],
        logspace=False, max_gradient_norm=None,
    )
    print(f"{label}: sharp stage finished ({time.time() - t0:.1f}s total)", flush=True)
    return soft, final, best


def run_cycles(
    loss,
    x: jnp.ndarray,
    key: jax.Array,
    eval_keys: jax.Array,
    schedule: dict,
    n_cycles: int,
    cycle_noise: float,
    select=None,
    mutate=None,
) -> tuple[jnp.ndarray, np.ndarray, list[dict]]:
    """C repetitions of [soft -> sharp], re-softening between them.

    Returns the best iterate per member over all cycles, its loss, and a
    per-cycle log.

    Two optional hooks at the cycle boundary, which together are the
    evolutionary seam:

      * `select(losses, rng) -> parent index per slot`, called BEFORE
        re-softening. Defaults to the identity -- every member its own parent,
        which is the lab's current behaviour and keeps the B trajectories
        independent.
      * `mutate(x, parents, losses, rng) -> x`, called INSTEAD of `resoften`
        once parents are gathered. Defaults to `resoften`. This is where a
        per-position or per-member noise schedule belongs, because it is the
        only hook that sees the PSSM itself.
    """
    batch = x.shape[0]
    overall_x, overall_vals = None, None
    log: list[dict] = []
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))

    for cyc in range(n_cycles):
        tag = f"cycle {cyc + 1}/{n_cycles}: "

        _, _, cyc_x = optimize_cycle(loss, x, key, schedule, label=tag)
        key, _ = jax.random.split(key)
        # Fixed evaluation keys make the cycle endpoints comparable.
        cyc_vals = np.asarray(batched_eval(loss, cyc_x, eval_keys)[0])

        # ---- keep the best across cycles ----
        if overall_vals is None:
            overall_x, overall_vals = cyc_x, cyc_vals
        else:
            take = cyc_vals < overall_vals
            overall_x = jnp.where(jnp.array(take)[:, None, None], cyc_x, overall_x)
            overall_vals = np.where(take, cyc_vals, overall_vals)
        print(f"{tag}this cycle: "
              + "  ".join(f"[{b}] {cyc_vals[b]:.4f}" for b in range(batch))
              + "   best overall: "
              + "  ".join(f"[{b}] {overall_vals[b]:.4f}" for b in range(batch)),
              flush=True)
        log.append({
            "cycle": cyc + 1,
            "cumulative_steps": (cyc + 1) * (schedule["soft_steps"] + schedule["sharp_steps"]),
            "losses": [float(v) for v in cyc_vals],
        })

        # ---- select, then re-soften for the next cycle ----
        if cyc < n_cycles - 1:
            parents = np.arange(batch)
            if select is not None:
                parents = np.asarray(select(cyc_vals, rng), dtype=int)
                if parents.shape != (batch,):
                    raise SystemExit(
                        f"select returned {parents.shape} parents, expected ({batch},)"
                    )
                replaced = int((parents != np.arange(batch)).sum())
                if replaced:
                    log[-1]["replaced"] = replaced
                    log[-1]["parents"] = parents.tolist()
                    print(f"{tag}selection replaced {replaced}/{batch} trajectories",
                          flush=True)
            gathered = cyc_x[jnp.asarray(parents)]
            if mutate is None:
                x = resoften(gathered, cycle_noise)
                print(f"{tag}re-softened from argmax (cycle noise {cycle_noise})",
                      flush=True)
            else:
                x = jnp.asarray(mutate(gathered, parents, cyc_vals, rng))
                if x.shape != gathered.shape:
                    raise SystemExit(
                        f"mutate returned {x.shape}, expected {gathered.shape}"
                    )
                print(f"{tag}mutated {batch} trajectories", flush=True)

    return overall_x, overall_vals, log


def refold_and_score(folder, seq_str: str, target_sequence: str, msa_path: str) -> float:
    """Re-fold the discrete binder alongside the target and score it.

    A separate, more expensive prediction (3 recycles, 6 samples) than the one
    inside the loop, and what `ranking_loss` means in the output contract. It is
    the same model family as the in-loop objective, so it is not an independent
    critic -- treat agreement between the two as partly self-correlation.
    """
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
    value, _ = ranking_loss(
        jax.nn.one_hot(np.array([TOKENS.index(c) for c in seq_str]), 20),
        key=jax.random.key(0),
    )
    return value.item()


def preflight() -> None:
    """Fail loudly if the container's mosaic is older than this driver needs.

    `/opt/mosaic/mosaic.sif` ships a mosaic predating epitope conditioning on
    the distogram proxy, and `mosaic-exec.sh` only overrides it when
    MOSAIC_DEV_SRC is set -- which `bindocracy.tools.mosaic.launch` does NOT
    currently inject. Without this check the run dies 30 s in with a bare
    `TypeError: DistogramIPTMProxy.__init__() got an unexpected keyword
    argument 'epitope_idx'`, which says nothing about the cause.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(sp.DistogramIPTMProxy)}
    if "epitope_idx" not in fields:
        raise SystemExit(
            "this container's mosaic has no epitope_idx on DistogramIPTMProxy, "
            "so the epitope cannot reach the distogram term.\n"
            "Set MOSAIC_DEV_SRC=<mosaic checkout>/src to bind a newer source "
            "over the image, as the lab's own runs do."
        )


def parse_epitope(value: str, target_length: int) -> list[int] | None:
    """Read `--epitope` into 0-based target indices, or refuse it.

    Checked here as well as in the harness because this is the last place the
    numbers exist before they become a silent array slice: an out-of-range
    index would be clipped by JAX rather than raised, and the run would
    condition on whatever position it landed on.
    """
    indices = [int(item) for item in value.split(",") if item.strip()]
    if not indices:
        return None
    outside = [index for index in indices if not 0 <= index < target_length]
    if outside:
        raise SystemExit(
            f"epitope indices {outside} are outside a target of "
            f"{target_length} residues. They are 0-based positions in the "
            "target sequence, not residue numbers."
        )
    return sorted(set(indices))


def write_status(save_dir: Path, payload: dict) -> None:
    """Write status.json atomically so a reader never sees a half file."""
    tmp = save_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, save_dir / STATUS_FILE)


def load_hook(spec: str, flag: str):
    """Resolve `module:function` to a callable, or None for the lab's behaviour.

    Imported from THIS directory, so an EA lives in its own file and this
    driver stays the schedule it ports. The archived-per-run copy of the driver
    sits next to the archived hook module, so a run stays reproducible.
    """
    if not spec:
        return None
    import importlib.util

    module_name, _, func_name = spec.partition(":")
    if not func_name:
        raise SystemExit(f"{flag} takes module:function, e.g. evolution:select")
    path = Path(__file__).resolve().parent / f"{module_name}.py"
    if not path.is_file():
        raise SystemExit(f"--select module not found: {path}")
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    if not hasattr(module, func_name):
        raise SystemExit(f"{path} has no {func_name}")
    return getattr(module, func_name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # The eleven flags the harness always passes, unchanged, so this file can be
    # named as a MosaicDriverConfig.script without touching the launcher.
    ap.add_argument("--target-fasta", required=True)
    ap.add_argument("--target-msa", required=True)
    ap.add_argument("--binder-length", type=int, required=True)
    ap.add_argument("--task-id", type=int, required=True,
                    help="names the designs and offsets the seeds, so parallel "
                         "tasks explore different inits")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--n-designs", type=int, required=True,
                    help="designs this task produces. With --batch B they come "
                         "B at a time, so the task runs ceil(n/B) populations.")
    ap.add_argument("--max-runtime", type=float, required=True,
                    help="hours; set below the job walltime so the last "
                         "population finishes and gets written")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epitope", default="",
                    help="comma-separated 0-based indices into the target "
                         "sequence. Empty means no epitope, which rewards "
                         "contact anywhere on the target.")
    # Accepted and mapped onto the cycle schedule so the harness's existing
    # three-stage flags do not make this driver unlaunchable. --sharpen-steps
    # becomes the sharp stage; --final-steps has no cycle equivalent and is
    # ignored with a warning rather than silently dropped.
    ap.add_argument("--soft-steps", type=int, default=90)
    ap.add_argument("--sharpen-steps", type=int, default=30)
    ap.add_argument("--final-steps", type=int, default=0,
                    help="ignored; the cycle schedule has no third stage")

    # The cycle schedule proper.
    ap.add_argument("--batch", type=int, default=4,
                    help="trajectories optimized at once on one GPU, vmapped "
                         "over one JIT compile. The lab's runs use 4.")
    ap.add_argument("--cycles", type=int, default=3,
                    help="repetitions of [soft -> sharp]. 1 is a single pass.")
    ap.add_argument("--cycle-noise", type=float, default=0.15,
                    help="re-softening weight between cycles: the argmax "
                         "one-hot is blended this far back toward uniform")
    # Absolute, NOT scaled by sqrt(L). See the module docstring.
    ap.add_argument("--soft-stepsize", type=float, default=0.05)
    ap.add_argument("--soft-momentum", type=float, default=0.7)
    ap.add_argument("--soft-scale", type=float, default=1.0)
    ap.add_argument("--sharp-stepsize", type=float, default=0.025)
    ap.add_argument("--sharp-momentum", type=float, default=0.5)
    ap.add_argument("--sharp-scale", type=float, default=2.0)
    ap.add_argument("--num-samples", type=int, default=1,
                    help="in-loop structure samples; the lab's config uses 1")

    # Seeding from a pre-screened proposal pool, as the R32/R35 runs do.
    ap.add_argument("--init-fasta", default="",
                    help="multi-FASTA of proposed binders to refine. Empty "
                         "starts from noise.")
    ap.add_argument("--init-noise", type=float, default=0.15)
    ap.add_argument("--init-noise-random", action="store_true",
                    help="blend seeds toward a per-trajectory Dirichlet draw "
                         "instead of toward uniform. Required for replicas of "
                         "one parent to differ at all.")

    ap.add_argument("--select", default="",
                    help="module:function in this directory, called at each "
                         "cycle boundary as select(losses, rng) -> parent index "
                         "per slot. Empty keeps every member its own parent.")
    ap.add_argument("--mutate", default="",
                    help="module:function in this directory, called after the "
                         "parent gather as mutate(x, parents, losses, rng) -> x, "
                         "replacing the default argmax re-soften. Empty uses "
                         "the lab's re-soften at --cycle-noise.")
    ap.add_argument("--dump-cycles", action="store_true",
                    help="write the per-cycle loss log to cycles.jsonl")
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

    schedule = {
        "soft_steps": a.soft_steps, "soft_stepsize": a.soft_stepsize,
        "soft_momentum": a.soft_momentum, "soft_scale": a.soft_scale,
        "sharp_steps": a.sharpen_steps, "sharp_stepsize": a.sharp_stepsize,
        "sharp_momentum": a.sharp_momentum, "sharp_scale": a.sharp_scale,
    }
    total_steps = a.cycles * (a.soft_steps + a.sharpen_steps)

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}", flush=True)
    try:
        preflight()
        if a.final_steps:
            print(f"warning: --final-steps {a.final_steps} ignored; the cycle "
                  "schedule has no third stage", flush=True)
        select = load_hook(a.select, "--select")
        mutate = load_hook(a.mutate, "--mutate")
        target_sequence = read_fasta(a.target_fasta)
        epitope_idx = parse_epitope(a.epitope, len(target_sequence))
        print(f"target: {len(target_sequence)} aa   binder: {a.binder_length} aa", flush=True)
        print(f"batch: {a.batch}   cycles: {a.cycles}   "
              f"steps/trajectory: {total_steps}", flush=True)
        print(f"epitope: {epitope_idx if epitope_idx else 'none (whole target)'}", flush=True)
        print(f"selection: {a.select or 'none (each member its own parent)'}", flush=True)
        print(f"mutation: {a.mutate or f'none (argmax re-soften @ {a.cycle_noise})'}",
              flush=True)

        t_build = time.time()
        folder, loss = build(
            target_sequence, a.target_msa, a.binder_length, epitope_idx, a.num_samples
        )
        # NoCys makes the optimizer's alphabet 19 tokens, not 20 -- cysteine is
        # spliced back in with zero probability by NoCys.sequence.
        n_tokens = 19
        print(f"models + loss built in {time.time() - t_build:.1f}s", flush=True)
        print("  no-cys: optimizing over 19 tokens "
              "(cysteine excluded by construction)", flush=True)

        deadline = time.monotonic() + a.max_runtime * 3600
        cycles_path = save_dir / "cycles.jsonl"
        with designs_path.open("a") as out:
            population = 0
            while produced < a.n_designs and time.monotonic() < deadline:
                # Distinct per (task, population) so no two populations share an
                # init. Members within a population differ by their RNG key.
                seed = a.seed_base + a.task_id * 100_000 + population
                batch = min(a.batch, a.n_designs - produced)
                attempted += batch

                key = jax.random.key(seed)
                key, init_key = jax.random.split(key)
                x = initial_population(
                    init_key, batch, a.binder_length, n_tokens, seed,
                    a.init_fasta or None, a.init_noise, a.init_noise_random,
                )
                # Fixed across cycles, so a loss at cycle 1 is comparable to one
                # at cycle 3 and ranking across them is meaningful.
                eval_keys = jax.random.split(jax.random.key(seed + 1), batch)

                t0 = time.time()
                best_x, best_vals, log = run_cycles(
                    loss, x, key, eval_keys, schedule, a.cycles,
                    a.cycle_noise, select, mutate,
                )
                elapsed = time.time() - t0

                full = np.asarray(NoCys.sequence(best_x))
                for b in range(batch):
                    seq_str = "".join(TOKENS[i] for i in full[b].argmax(-1))
                    ranking_loss = refold_and_score(
                        folder, seq_str, target_sequence, a.target_msa
                    )
                    index = produced
                    out.write(json.dumps({
                        "native_id": f"task-{a.task_id:04d}-design-{index:06d}",
                        "sequence": seq_str,
                        "seed": seed,
                        "ranking_loss": ranking_loss,
                        "cycle_loss": float(best_vals[b]),
                        "population": population,
                        "member": b,
                        "cycles": a.cycles,
                        "steps": total_steps,
                        "completed_at": now(),
                        "seconds": round(elapsed / batch, 1),
                    }) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    produced += 1
                    print(f"[{produced}] loss={ranking_loss:.4f} "
                          f"cycle={best_vals[b]:.4f} {seq_str}", flush=True)

                if a.dump_cycles:
                    with cycles_path.open("a") as clog:
                        clog.write(json.dumps({
                            "task_id": a.task_id, "population": population,
                            "seed": seed, "batch": batch, "cycles": log,
                        }) + "\n")
                population += 1
    except Exception as exc:  # noqa: BLE001 -- any failure must still write status.json
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    if error is not None or produced == 0:
        status = "failed"
    elif produced >= a.n_designs:
        status = "completed"
    else:
        status = "partial"

    write_status(save_dir, {
        "status": status,
        "task_id": a.task_id,
        "attempted": attempted,
        "produced": produced,
        "requested": a.n_designs,
        "batch": a.batch,
        "cycles": a.cycles,
        "steps_per_trajectory": total_steps,
        "selection": a.select or None,
        "mutation": a.mutate or None,
        "started_at": started_at,
        "finished_at": now(),
        "error": error,
    })
    print(f"status={status} produced={produced}/{a.n_designs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
