"""User-authored EA operators for Mosaic's cycle optimizer.

Operators act on the population supplied by the driver, whether its members
originate from one binder or several distinct inputs. Root design IDs, target
identity, and lineage bookkeeping belong to the driver. Indices refer to slots
in that supplied population, not persistent candidate IDs.

Use only the supplied numpy.random.Generator for randomness. Do not reseed it,
use global random state, or modify input arrays in place.

Integration:
    select(losses, rng) builds an identity parent map and replaces culled slots
    with indices returned by resample(). The driver then gathers parent PSSMs.
    mutate(gathered, parents, losses, rng) replaces the driver's default
    argmax/re-soften operation by calling perturb() on ALL gathered members.
    Survivors also need a discrete restart before the next soft phase.
    make_hooks() binds cull and noise settings to those run_cycles signatures.

cull_steps() describes cycle-boundary schedules; selection timing changes only
when a driver explicitly consumes its output.
"""
from functools import partial

import numpy as np



def cull(losses, fraction, rng): 
    """Partition a population into survivors and members to replace.

    Inputs:
        losses: Finite real array of shape (B,), B >= 1. LOWER is better.
            Values are comparable fitness evaluations for the current cycle.
        fraction: Finite fraction in [0, 1). Cull floor(B * fraction) members;
            do not silently round up. Small populations can therefore yield
            zero replacements, e.g. B=4 and fraction=0.1.
        rng: numpy.random.Generator, used to break fitness ties reproducibly.

    Returns:
        (survivor_idx, culled_idx): One-dimensional integer NumPy arrays of
        ORIGINAL population indices, with lengths B - n_culled and n_culled.
        Each index appears exactly once across the two arrays; their union is
        range(B), they are disjoint, and at least one survivor remains.
        Cull the largest losses; array ordering is otherwise unspecified.

    Edge cases:
        fraction=0 returns all indices as survivors and an empty culled array.
        Reject empty/non-1D losses, nonfinite losses, or an invalid fraction
        with ValueError rather than silently ranking invalid predictions.
        Do not mutate losses. The caller owns the surviving candidate data.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError(f"losses must be 1-D and non-empty, got shape {losses.shape}")
    if not np.all(np.isfinite(losses)):
        raise ValueError("losses must all be finite; a NaN would sort unpredictably")
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    # Rank, do not threshold. A value threshold cannot honour the floor(B*f)
    # count, and on ties "<= t" and ">= t" both match, which would put one
    # member in BOTH halves and let a culled slot parent its own replacement.
    n_cull = int(np.floor(losses.size * fraction))

    # Permute before sorting so equal losses do not always cull the same slot.
    # A stable sort over the permuted order makes the tie-break a reproducible
    # function of rng alone.
    perm = rng.permutation(losses.size)
    order = perm[np.argsort(losses[perm], kind="stable")]

    # order is best-first (lower loss is better), so the culled members are the
    # TAIL. n_cull == 0 yields an empty tail, which is the fraction=0 case and
    # needs no branch of its own.
    n_keep = losses.size - n_cull
    return order[:n_keep], order[n_keep:]
    

def resample(survivors, n, rng):
    """Choose parents for replacement slots, sampling survivors with replacement.

    Inputs:
        survivors: One-dimensional array of unique, nonnegative integer
            ORIGINAL population indices, normally returned by cull().
        n: Nonnegative integer number of replacement children to create.
        rng: numpy.random.Generator; use it for reproducible parent draws.

    Returns:
        Integer NumPy array of shape (n,), with every value drawn from
        survivors. Return population indices, NOT offsets into survivors.
        Repeated parents are allowed; use uniform sampling for the initial
        policy. This function chooses parents; it does not copy or perturb
        their sequences.

    Edge cases:
        n=0 returns an empty integer array, including when survivors is empty.
        Reject malformed survivor indices, invalid n, or empty survivors when
        n>0 with ValueError. Do not modify survivors.
    """
    survivors = np.asarray(survivors)
    if survivors.ndim != 1:
        raise ValueError(f"survivors must be 1-D, got shape {survivors.shape}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"n must be a nonnegative integer, got {n!r}")
    if survivors.size:
        if survivors.dtype.kind not in "iu":
            raise ValueError("survivors must contain integer population indices")
        if np.any(survivors < 0) or np.unique(survivors).size != survivors.size:
            raise ValueError("survivors must contain unique, nonnegative indices")
    if n > 0 and survivors.size == 0:
        raise ValueError("cannot choose parents from an empty survivor set")

    # An untyped empty list becomes float64; still return integer indices for
    # the documented empty-survivor/no-children case.
    if n == 0:
        return np.empty(0, dtype=int)
    
    return rng.choice(survivors, size=n, replace=True)

def perturb(pssm, low, high, rng):
    """Randomly re-soften parent sequences into valid optimization start points.

    Inputs:
        pssm: Finite real probability array of shape (B, L, K): population,
            sequence length, alphabet size. Entries are nonnegative and each
            residue row sums to one within floating-point tolerance.
            The hook passes already-gathered parents, so rows may be identical.
        low, high: Finite bounds with 0 <= low <= high <= 1 for the fraction
            mixed away from each parent's discrete sequence. Draw a mixing
            fraction per child; these are NOT logit or gradient-noise bounds.
        rng: numpy.random.Generator for mixing fractions and independent
            per-child/per-position random probability distributions.

    Returns:
        Floating-point probability array of the SAME shape (B, L, K), accepted
        by jnp.asarray(). Commit each parent position to its argmax token, then
        mix that one-hot sequence with random simplex-valued noise.
        Entries remain finite/nonnegative and every residue row sums to one.
        Identical parents receive independent draws, not a broadcast mutation.

    Edge cases and ownership:
        low=high fixes the mixing fraction, not the random distribution.
        low=high=0 returns the parents' argmax one-hot sequences.
        Reject invalid probabilities, dimensions, or bounds with ValueError.
        Do not modify pssm, change L/K, or infer amino-acid identities from
        column numbers. The driver owns alphabet and residue constraints.
        This operator acts on every supplied row; the hook decides which rows
        receive it. There is no automatic driver re-softening after this hook.
    """
    pssm = np.asarray(pssm, dtype=float)
    if pssm.ndim != 3 or 0 in pssm.shape:
        raise ValueError(f"pssm must be 3-D (B, L, K) and non-empty, got {pssm.shape}")
    if not np.all(np.isfinite(pssm)):
        raise ValueError("pssm must be finite")
    if np.any(pssm < 0.0):
        raise ValueError("pssm must be nonnegative; it is a probability array")
    if not np.allclose(pssm.sum(axis=-1), 1.0, atol=1e-4):
        worst = float(np.abs(pssm.sum(axis=-1) - 1.0).max())
        raise ValueError(f"pssm rows must sum to 1; worst deviation {worst:.3g}")
    for name, bound in (("low", low), ("high", high)):
        if not np.isfinite(bound):
            raise ValueError(f"{name} must be finite, got {bound!r}")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"need 0 <= low <= high <= 1, got low={low}, high={high}")

    B, L, K = pssm.shape

    # Commit first. The argmax is the whole point of re-softening: it forces a
    # discrete sequence before the next soft phase, rather than carrying the
    # parent's soft distribution forward as a warm restart.
    hard = np.zeros_like(pssm)
    np.put_along_axis(hard, pssm.argmax(axis=-1)[..., None], 1.0, axis=-1)

    # One mixing fraction per CHILD, so a child is perturbed uniformly along
    # its length. uniform(low, low) returns low, which is why low == high
    # fixes the fraction without disabling the noise draw below.
    weight = rng.uniform(low, high, size=B)[:, None, None]

    # Dirichlet(1) is the uniform distribution over the simplex, so each
    # (child, position) gets an independent random probability vector. Drawing
    # per position -- not once per child -- is what makes K children gathered
    # from ONE parent diverge instead of cloning.
    noise = rng.dirichlet(np.ones(K), size=(B, L))

    out = (1.0 - weight) * hard + weight * noise
    # Rows sum to 1 analytically; renormalise so the postcondition holds
    # exactly rather than to float tolerance.
    return out / out.sum(axis=-1, keepdims=True)


def select(losses, rng, *, fraction=0.5):
    """Keep survivors in place and assign a surviving parent to each culled slot."""
    losses = np.asarray(losses)
    if losses.dtype.kind not in "iuf":
        raise ValueError("losses must contain real fitness values")
    survivors, culled = cull(losses, fraction, rng)
    parents = np.arange(survivors.size + culled.size)
    parents[culled] = resample(survivors, culled.size, rng)
    return parents


def mutate(gathered_pssm, parents, losses, rng, *, low=0.10, high=0.20):
    """Re-soften every already-gathered parent without gathering indices again.

    parents and losses describe the original population. They are validated
    for lineage consistency, but the PSSMs are already in destination-slot
    order. Surviving members need the same restart as replacement children.
    Probability validation and independent noise draws belong to perturb().
    """
    gathered_pssm = np.asarray(gathered_pssm)
    parents = np.asarray(parents)
    losses = np.asarray(losses)
    if gathered_pssm.ndim != 3 or 0 in gathered_pssm.shape:
        raise ValueError(
            f"pssm must be 3-D (B, L, K) and non-empty, got {gathered_pssm.shape}"
        )
    batch = gathered_pssm.shape[0]
    if parents.shape != (batch,) or parents.dtype.kind not in "iu":
        raise ValueError(f"parents must be an integer array of shape ({batch},)")
    if np.any(parents < 0) or np.any(parents >= batch):
        raise ValueError(f"parents must be population indices in [0, {batch})")
    if losses.shape != (batch,) or losses.dtype.kind not in "iuf":
        raise ValueError(f"losses must be a real array of shape ({batch},)")
    if not np.all(np.isfinite(losses)):
        raise ValueError("losses must all be finite")
    if gathered_pssm.dtype.kind not in "iuf":
        raise ValueError("pssm must contain real probabilities")
    return perturb(gathered_pssm, low, high, rng)


def make_hooks(cull_fraction, noise_low, noise_high):
    """Bind campaign settings to run_cycles' select and mutate hook signatures."""
    return (
        partial(select, fraction=cull_fraction),
        partial(mutate, low=noise_low, high=noise_high),
    )


def cull_steps(schedule, every): 
    """Translate a cycle-based selection interval into cumulative step numbers.

    Inputs (selection timing changes only when a driver consumes this output):
        schedule: One-dimensional sequence of strictly increasing positive
            integer cumulative gradient-step counts at cycle ends, INCLUDING
            the final cycle. Example: [120, 240, 360] for three 120-step cycles.
            This is not run_cycles' soft/sharp settings dictionary; the caller
            constructs these endpoints from the cycle count and stage lengths.
        every: Positive integer interval in COMPLETED CYCLES, not gradient
            steps. Select after every-th cycle only when another cycle follows.

    Returns:
        Ascending integer NumPy array of cumulative step counts where selection
        should run. Never include step zero or the final endpoint, because no
        continuation remains to redistribute after the last cycle.
        For [120, 240, 360]: every=1 gives [120, 240]; every=2 gives [240].

    Edge cases:
        Empty/single-cycle schedules, or an interval longer than the available
        nonfinal cycles, return an empty integer array.
        Reject noninteger/nonpositive/nonincreasing endpoints or invalid every
        with ValueError. Do not modify schedule.
        The driver must explicitly consume this output; implementing this
        function alone will not change the current selection timing.
    """
    endpoints = np.asarray(schedule)
    if endpoints.ndim != 1:
        raise ValueError(f"schedule must be 1-D, got shape {endpoints.shape}")
    # Checked before the dtype test: np.asarray([]) is float64, and an empty
    # schedule is a documented no-op rather than a malformed one.
    if endpoints.size == 0:
        return np.array([], dtype=int)
    if endpoints.dtype.kind not in "iuf":
        raise ValueError(f"schedule must be numeric, got dtype {endpoints.dtype}")
    if endpoints.dtype.kind == "f":
        if not np.all(np.isfinite(endpoints)):
            raise ValueError("schedule endpoints must be finite")
        if not np.all(endpoints == np.floor(endpoints)):
            raise ValueError("schedule endpoints must be whole step counts")
    endpoints = endpoints.astype(int)
    if np.any(endpoints <= 0):
        raise ValueError("schedule endpoints must be positive step counts")
    if np.any(np.diff(endpoints) <= 0):
        raise ValueError("schedule endpoints must be strictly increasing")
    if isinstance(every, bool) or not isinstance(every, (int, np.integer)) or every < 1:
        raise ValueError(f"every must be a positive integer number of cycles, got {every!r}")

    # Cycle i (1-based) ends at endpoints[i - 1]. Only cycles with a SUCCESSOR
    # can be selection points, so the range stops at size - 1: after the final
    # endpoint there is no continuation left to redistribute compute into.
    cycles = np.arange(1, endpoints.size)
    selected = cycles[cycles % every == 0]
    return endpoints[selected - 1]

    
