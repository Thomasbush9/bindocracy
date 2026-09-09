#!/usr/bin/env python
"""Score a shard of a design set with one structural model, inside mosaic.sif.

Invoked by the harness as:

    singularity/mosaic-exec.sh python score_designs.py \
        --design-set <dir>/designs.fasta --target-fasta ... --target-msa ... \
        --model boltz2 --recycling 3 --sampling-steps 25 --num-samples 1 \
        --readers complex,monomer --shard 0 --num-shards 1 --save-dir <dir>

Output contract, identical in shape to every other driver here:

    <save-dir>/metrics.jsonl   one JSON object per (design, condition, sample),
                               appended and fsynced as it goes
    <save-dir>/status.json     written once, atomically, when the task stops

The process **exits 0 whenever it managed to write status.json**. Success is
status.json plus the rows it accounts for, never the exit code -- one bad
design must not cost the shard.

PROVENANCE. `load_model`, `build_features`, `fold`, `metrics` and the two
protocol-normalisation tables below are transcribed from
`mosaic/benchmark/model_matrix.py`, the runner that produced the matched
benchmark in `mosaic_setup/benchmark/BENCHMARK.md`. They are copied rather than
imported because a driver runs from the archived copy in the run directory and
must not depend on a host checkout that can move under it -- and because the
archived copy is then the record of exactly what ran. If the benchmark's
protocol changes, this file has to be updated deliberately, which is the
intended cost.

NOT copied: `screen_proposals.py`'s approach. It builds features with
`binder_features`, which stubs binder sidechains to poly-glycine -- the single
largest protocol confound BENCHMARK.md identifies -- and assumes one binder
length where a campaign design set has many.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

METRICS_FILE = "metrics.jsonl"
STATUS_FILE = "status.json"
# One subdirectory per condition beneath it, so a complex pose and the monomer
# pose of the same design never collide.
STRUCTURES_DIR = "structures"

CA = 1
IFACE_CUTOFF = 8.0  # angstrom, CA-CA, for interface pLDDT
TOKENS = "ARNDCQEGHILKMFPSTWYV"

# Trunk-pass normalisation. Every backend takes `recycling_steps` and they do
# not agree on what it counts: OpenFold3 runs `recycling_steps + 1` passes and
# ESMFold2's sister library does the same. So `--recycling 3` gave three models
# three passes and two of them four. Subtracting one here means the number the
# harness records is the number of passes that actually ran.
RECYCLING_OFFSET = {"of3": -1, "esmfold2": -1}

# Backends with no diffusion sampler. AF2 asserts it is not handed one.
NO_SAMPLER = {"af2"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_model(name, variant=None):
    """Constructors matched to the checkpoints actually on disk.

    `variant` applies to protenix only. Naming a checkpoint that is not on
    disk would reach for a download, and offline mode turns that into an
    unrelated-looking error from inside a constructor, so the accepted set is
    closed here as well as in the config.
    """
    if name == "af2":
        from mosaic.models.af2 import AlphaFold2

        return AlphaFold2()
    if name == "boltz1":
        from mosaic.models.boltz1 import Boltz1

        return Boltz1()
    if name == "boltz2":
        from mosaic.models.boltz2 import Boltz2

        return Boltz2()
    if name == "of3":
        from mosaic.models.of3 import OF3

        return OF3()
    if name == "protenix":
        # Two checkpoints are on disk. They are different models, not different
        # speeds of one: mini defaults to 2 diffusion steps and base to 20, and
        # the benchmark scored mini at 0.680 AUC partly for that reason.
        if variant in (None, "mini"):
            from mosaic.models.protenix import ProtenixMini

            return ProtenixMini()
        if variant == "base":
            from mosaic.models.protenix import ProtenixBase

            return ProtenixBase()
        raise ValueError(
            f"protenix variant {variant!r} has no weights under MOSAIC_WEIGHTS; "
            "mini and base are the two on disk"
        )
    if name == "promera":
        # jpromera publishes converted equinox weights, so this is a plain
        # load with no torch conversion. `subsample` is left at the library
        # default; it caps MSA rows per pass and is a protocol knob we do not
        # expose yet, so it must not be varied silently.
        from mosaic.models.promera import JPromeraModel

        return JPromeraModel()
    if name == "esmfold2":
        # Full, not Fast. Fast has no MSA encoder and raises if any chain sets
        # use_msa, so it cannot take the target MSA on equal terms.
        from mosaic.models.esmfold2 import ESMFold2Full

        return ESMFold2Full()
    raise ValueError(f"unknown model {name}")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """(header, sequence) pairs, in file order."""
    header, chunks, out = None, [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        else:
            chunks.append(line)
    if header is not None:
        out.append((header, "".join(chunks)))
    return out


# Backends whose `model_output` applies the PSSM through
# `apply_binder_sequence`, which asserts on a target-only feature pack. They
# must be featurised down the binder-design path instead. The difference is not
# cosmetic: `binder_features` stubs the binder to a placeholder residue and
# lets the PSSM carry the sequence, so the binder's sidechain reference atoms
# are generic rather than real. Recorded per run in `feature_path`.
BINDER_FEATURE_PATH = {"promera"}


def build_features(model, *, name, seq, target, msa_path, condition):
    """One feature pack, with the binder's real sidechain reference atoms.

    `target_only_features` bakes the actual binder sequence in, so it is
    rebuilt per design. Shapes do not change, so that costs a featurisation
    but no recompile -- and it is what the design and ranking paths use.

    Promera is the exception; see BINDER_FEATURE_PATH.
    """
    from mosaic.structure_prediction import TargetChain

    target_chain = TargetChain(
        sequence=target, use_msa=msa_path is not None,
        msa_path=str(msa_path) if msa_path else None,
    )
    if name in BINDER_FEATURE_PATH:
        # binder_features(binder_length, target_chains): the binder is
        # prepended internally, so only the target is passed.
        return model.binder_features(
            len(seq), [target_chain] if condition == "complex" else []
        )

    chains = [TargetChain(sequence=seq, use_msa=False)]
    if condition == "complex":
        chains.append(target_chain)
    # Both halves. The writer turns a prediction's coordinates into a
    # gemmi.Structure (`models/*.py::predict`), and dropping it -- which this
    # driver used to do with a bare [0] -- is what made a scoring run
    # unrepeatable: the metrics survived and the pose they described did not,
    # so no epitope, contact or clash measurement could ever be added after
    # the fact without folding everything again.
    return model.target_only_features(chains=chains)


def fold(name, model, pssm, features, *, sample, key, recycling, sampling):
    import jax

    kw = {}
    if name in NO_SAMPLER:
        # AF2 is deterministic, so the key does nothing and the parameter set
        # is the only axis. Pinned to 0 for reproducibility.
        kw["model_idx"] = 0
        k = key
    else:
        k = jax.random.fold_in(key, sample)
        kw["sampling_steps"] = sampling
    return model.model_output(
        PSSM=pssm,
        features=features,
        recycling_steps=recycling + RECYCLING_OFFSET.get(name, 0),
        key=k,
        **kw,
    )


def save_structure(output, path: Path, save_dir: Path) -> str | None:
    """Write one predicted pose as PDB, returning its task-relative path.

    Deliberately NOT the per-model `writer` that `target_only_features`
    returns. AF2's is `None` (`models/af2.py:440`), so a writer-based
    implementation silently produced no structures for AF2 while every other
    model worked -- observed 2026-09-09.

    `atom37_coords`, `full_sequence`, `asym_id` and `residue_idx` are populated
    by *every* wrapper (`losses/structure_prediction.py:18-38`), and
    `full_sequence` is guaranteed to be in mosaic-20 order, which is
    AlphaFold's `restypes` order. So one writer serves all of them, and -- more
    useful than uniformity for its own sake -- every model's structure comes
    out with the same atom ordering and chain convention, which is what makes
    two models' poses directly comparable without a per-model reader.

    Chain 0 is the binder and chain 1 the target, matching the order
    `build_features` constructs them in.

    A failure here must not lose the metrics: the numbers are the run's purpose
    and the pose is an addition to it, so this returns None and the caller
    records that.
    """
    import numpy as np

    try:
        from mosaic.alphafold.common import protein, residue_constants

        coords = np.asarray(output.atom37_coords, dtype=np.float64)
        aatype = np.asarray(output.full_sequence).argmax(-1).astype(np.int32)
        # Which atoms this residue type has at all, minus any the model left
        # exactly at the origin (how the wrappers mark an absent atom).
        mask = residue_constants.restype_atom37_mask[aatype].astype(np.float64)
        mask = mask * (np.abs(coords).sum(-1) > 1e-6)
        plddt = np.asarray(output.plddt, dtype=np.float64)
        # AF2's PDB writer puts b-factors per atom; pLDDT is per residue.
        b_factors = np.repeat(plddt[:, None], coords.shape[1], axis=1) * mask

        pose = protein.Protein(
            atom_positions=coords,
            atom_mask=mask,
            aatype=aatype,
            residue_index=np.asarray(output.residue_idx, dtype=np.int32),
            chain_index=np.asarray(output.asym_id, dtype=np.int32),
            b_factors=b_factors,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(protein.to_pdb(pose))
        return str(path.relative_to(save_dir))
    except Exception as exc:  # noqa: BLE001 - a missing pose must not fail a fold
        print(f"    could not write structure {path.name}: "
              f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return None


def interface_plddt(out, binder_length):
    """Mean pLDDT over binder residues in contact with the target.

    NaN when nothing is in contact. A failed dock has no interface, and
    averaging over an empty set would silently report zero -- which reads as a
    measured bad interface rather than an absent one.
    """
    import numpy as np

    ca = np.asarray(out.atom37_coords[:, CA])
    binder, target = ca[:binder_length], ca[binder_length:]
    distances = np.linalg.norm(binder[:, None, :] - target[None, :, :], axis=-1)
    contact = (distances < IFACE_CUTOFF).any(axis=1)
    if not contact.any():
        return float("nan")
    return float(np.asarray(out.plddt[:binder_length])[contact].mean())


def metrics(pssm, out, *, condition, binder_length, key):
    """Every metric for one fold. Aux dicts carry the natural sign.

    Names are prefixed by the caller, not here, so this stays the same
    function whichever model produced `out`.
    """
    import mosaic.losses.structure_prediction as sp
    import numpy as np

    def aux(term, field):
        return float(term(pssm, out, key=key)[1][field])

    if condition == "monomer":
        # No second chain exists, so every interface quantity is undefined.
        return {
            "mono_plddt": float(np.asarray(out.plddt).mean()),
            "mono_ptm": aux(sp.BinderPTMLoss(), "binder_ptm"),
            "mono_pae": aux(sp.WithinBinderPAE(), "bb_pae"),
            "mono_rg": aux(sp.ActualRadiusOfGyration(target_radius=0.0), "actual_rg"),
        }

    # The composite the design and ranking paths rank on. Those LossTerms
    # return the negated score, so what is recorded here is the score itself
    # and every metric keeps the natural "higher is better" sign. PAE stays
    # raw -- lower is better -- because a pre-negated column cannot be joined
    # against one that is not.
    iptm = aux(sp.IPTMLoss(), "iptm")
    tb_ipsae = aux(sp.TargetBinderIPSAE(), "tb_ipsae")
    bt_ipsae = aux(sp.BinderTargetIPSAE(), "bt_ipsae")
    return {
        "rank_composite": iptm + 0.5 * tb_ipsae + 0.5 * bt_ipsae,
        "ipsae_min": aux(sp.IPSAE_min(), "ipsae_min"),
        "bt_ipsae": bt_ipsae,
        "tb_ipsae": tb_ipsae,
        "iptm": iptm,
        "bt_iptm": aux(sp.BinderTargetIPTM(), "bt_iptm"),
        "binder_ptm": aux(sp.BinderPTMLoss(), "binder_ptm"),
        "binder_plddt": aux(sp.PLDDTLoss(), "plddt"),
        "complex_plddt": float(np.asarray(out.plddt).mean()),
        "iplddt": interface_plddt(out, binder_length),
        "bt_pae": aux(sp.BinderTargetPAE(), "bt_pae"),
        "tb_pae": aux(sp.TargetBinderPAE(), "tb_pae"),
        "ptm_energy": aux(sp.pTMEnergy(), "pTMEnergy"),
    }


def write_status(save_dir: Path, payload: dict) -> None:
    tmp = save_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, save_dir / STATUS_FILE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design-set", required=True, help="design-set FASTA")
    p.add_argument("--target-fasta", required=True)
    p.add_argument("--target-msa", default=None)
    p.add_argument("--model", required=True,
                   choices=["af2", "boltz1", "boltz2", "of3", "protenix",
                            "esmfold2", "promera"])
    p.add_argument("--recycling", type=int, required=True,
                   help="trunk passes, normalised per backend")
    p.add_argument("--sampling-steps", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--readers", default="complex",
                   help="comma-separated: complex, monomer")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--clear-cache-every", type=int, default=25,
                   help="bound JAX's compiled-kernel pool; Protenix OOMs without it")
    p.add_argument("--max-runtime", type=float, default=None, help="hours")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--variant", default=None,
                   help="protenix checkpoint: mini or base")
    p.add_argument("--save-structures", action="store_true",
                   help="write each predicted pose as mmCIF beside the metrics")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = save_dir / STRUCTURES_DIR
    started = now()
    t_start = time.time()

    status = {
        "task_id": a.task_id,
        "status": "failed",
        "started_at": started,
        "finished_at": None,
        "n_attempted": 0,
        "n_produced": 0,
        "output_file": METRICS_FILE,
        "error": None,
        # Which featurisation the binder went through. Not cosmetic: the
        # binder-design path stubs the binder's sidechains, so two models on
        # different paths are not measuring quite the same object.
        "feature_path": (
            "binder_features" if a.model in BINDER_FEATURE_PATH else "target_only_features"
        ),
    }

    try:
        import jax
        import jax.numpy as jnp

        print(f"jax {jax.__version__} backend={jax.default_backend()} "
              f"devices={jax.devices()}", flush=True)
        if jax.default_backend() != "gpu":
            # known-issues 2.3: a CPU JAX does not fail, it just never
            # finishes. Refuse rather than burn the walltime.
            raise SystemExit("jax resolved to CPU; refusing to score on CPU")

        readers = [r.strip() for r in a.readers.split(",") if r.strip()]
        for reader in readers:
            if reader not in ("complex", "monomer"):
                raise SystemExit(f"unknown reader {reader!r}")

        target = "".join(
            line.strip()
            for line in Path(a.target_fasta).read_text().splitlines()
            if line.strip() and not line.startswith(">")
        )

        entries = read_fasta(Path(a.design_set))
        indexed = []
        for header, seq in entries:
            index = int(header.split()[0])
            indexed.append((index, seq))
        # Contiguous shard, so a task spans few binder lengths and therefore
        # few JIT compilations. Matches DesignSet.shard exactly.
        per = -(-len(indexed) // a.num_shards)
        mine = indexed[a.shard * per : (a.shard + 1) * per]

        n_samples = a.num_samples
        if a.model == "af2" and n_samples > 1:
            print(f"NOTE: af2 is deterministic; --num-samples {n_samples} would "
                  "fold the same structure repeatedly. Forcing 1.", flush=True)
            n_samples = 1

        resolved_recycling = a.recycling + RECYCLING_OFFSET.get(a.model, 0)
        if resolved_recycling < 1:
            raise SystemExit(
                f"--recycling {a.recycling} leaves {a.model} with "
                f"recycling_steps={resolved_recycling}; it must run one pass"
            )

        # The line to diff when comparing two runs. The raw arguments are not
        # enough: --recycling means a different number of passes per backend.
        print(
            f"PROTOCOL model={a.model} readers={','.join(readers)} "
            f"trunk_passes={a.recycling} (recycling_steps={resolved_recycling}) "
            f"sampling_steps={'n/a' if a.model in NO_SAMPLER else a.sampling_steps} "
            f"num_samples={n_samples} seed={a.seed} "
            f"target_msa={a.target_msa} target_len={len(target)} "
            f"shard={a.shard}/{a.num_shards} designs={len(mine)}",
            flush=True,
        )

        t0 = time.time()
        model = load_model(a.model, getattr(a, "variant", None))
        print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

        key = jax.random.key(a.seed)
        deadline = t_start + a.max_runtime * 3600 if a.max_runtime else None

        n_attempted = 0
        # Per condition, not pooled. Pooling made a run whose every complex
        # fold raised report `succeeded` with 20/20 produced, because the
        # monomer fold had worked -- the interface numbers the run existed to
        # measure were simply absent. Observed on af2, 2026-09-07.
        produced: dict[str, set[int]] = {reader: set() for reader in readers}
        current_length = None

        with (save_dir / METRICS_FILE).open("w") as out_file:
            for index, seq in mine:
                if deadline and time.time() > deadline:
                    print("max runtime reached; stopping cleanly", flush=True)
                    break

                # Every distinct length compiles its own kernels and JAX
                # preallocates a fixed pool, so it fills monotonically as
                # groups accumulate. Clearing at each boundary is what let
                # Protenix finish a 20-length benchmark it had died partway
                # through.
                if len(seq) != current_length:
                    jax.clear_caches()
                    current_length = len(seq)
                    print(f"  length {current_length}", flush=True)
                elif a.clear_cache_every and n_attempted % a.clear_cache_every == 0:
                    jax.clear_caches()

                n_attempted += 1
                bad = [aa for aa in seq if aa not in TOKENS]
                if bad:
                    record = {"index": index, "condition": None, "replicate": 0,
                              "metrics": {}, "seconds": 0.0,
                              "failed": f"non-standard residues: {sorted(set(bad))}"}
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    continue

                pssm = jax.nn.one_hot(
                    jnp.array([TOKENS.index(aa) for aa in seq]), 20
                )

                for condition in readers:
                    try:
                        features, _writer = build_features(
                            model, name=a.model, seq=seq, target=target,
                            msa_path=a.target_msa if condition == "complex" else None,
                            condition=condition,
                        )
                        for sample in range(n_samples):
                            t_fold = time.time()
                            output = fold(
                                a.model, model, pssm, features,
                                sample=sample, key=key,
                                recycling=a.recycling,
                                sampling=a.sampling_steps,
                            )
                            values = metrics(
                                pssm, output, condition=condition,
                                binder_length=len(seq), key=key,
                            )
                            values = {
                                k: (None if v is None or not math.isfinite(v) else v)
                                for k, v in values.items()
                            }
                            structure = None
                            if a.save_structures:
                                structure = save_structure(
                                    output,
                                    structures_dir / condition
                                    / f"design-{index:06d}_s{sample}.pdb",
                                    save_dir,
                                )
                            record = {
                                "index": index,
                                "condition": condition,
                                "replicate": sample,
                                "metrics": values,
                                "seconds": round(time.time() - t_fold, 3),
                                "failed": None,
                                # Relative to the task directory.
                                # The adapter turns this into an artifact row so
                                # a later epitope or clash pass can read the pose
                                # instead of folding it again.
                                "structure": structure,
                            }
                            out_file.write(json.dumps(record) + "\n")
                            out_file.flush()
                            os.fsync(out_file.fileno())
                            if any(v is not None for v in values.values()):
                                produced[condition].add(index)
                    except Exception as exc:  # noqa: BLE001 - one design must not lose the shard
                        print(f"  design {index} [{condition}] FAILED: "
                              f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                        record = {"index": index, "condition": condition,
                                  "replicate": 0, "metrics": {}, "seconds": 0.0,
                                  "failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()

                if n_attempted % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"    {n_attempted}/{len(mine)} designs, "
                          f"{elapsed / 60:.1f}m elapsed", flush=True)

        status["n_attempted"] = n_attempted
        # A design counts as produced only when EVERY enabled reader measured
        # it. A run that folded 20 monomers and no complexes has not scored 20
        # designs; it has scored none of what it was asked for.
        complete = set.intersection(*produced.values()) if produced else set()
        status["n_produced"] = len(complete)
        status["produced_by_reader"] = {r: len(v) for r, v in produced.items()}
        empty = sorted(r for r, v in produced.items() if not v)

        if empty:
            status["status"] = "failed"
            status["error"] = (
                f"reader(s) {empty} produced nothing; the run measured none of "
                "what it was configured to measure"
            )
        elif len(complete) >= len(mine):
            status["status"] = "succeeded"
        else:
            status["status"] = "partial"

    except Exception as exc:  # noqa: BLE001 - always leave a status behind
        status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    status["finished_at"] = now()
    write_status(save_dir, status)
    print(f"status={status['status']} attempted={status['n_attempted']} "
          f"produced={status['n_produced']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
