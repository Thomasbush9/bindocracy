#!/usr/bin/env python3
"""Co-fold a shard of a design set with Chai-1, inside chai1.sif.

Runs in the container and cannot import bindocracy. The contract with the host
is the same one the mosaic scorer uses -- a `metrics.jsonl` of
`{index, replicate, condition, metrics, seconds}` rows plus a `status.json` --
so `Chai1OutputAdapter` is the scorer's adapter with a different tool name.

Two things here are transcribed rather than imported, and both are checked
against the container by `tests/test_chai1.py`:

  * the offline policy, from the image's own `chai1_offline.py`. Importing that
    module would run its `cli()`, so the policy is applied here directly. It
    must be installed BEFORE `chai_lab` is imported, because the guarded names
    are captured as function aliases at import time.
  * the token layout. `run_inference` returns per-token pLDDT and PAE but no
    token-to-chain map, so the driver assumes tokens are the two protein chains
    concatenated in FASTA order and asserts the total. For standard protein
    residues Chai tokenizes one token per residue; a mismatch means that
    stopped being true, and the assert turns a silently mis-sliced metric into
    a failed design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    """ISO-8601 UTC, matching what `TaskStatus` parses and what the mosaic
    scorer's driver writes. A bare `time.time()` float also validates, but the
    two drivers' status files would then disagree on the same field."""
    return datetime.now(UTC).isoformat()

# --- offline policy, before any chai_lab import -----------------------------

def install_offline_policy() -> None:
    """Refuse network access from inside the fold.

    The image already sets HF_HUB_OFFLINE and denies the two server flags at
    the CLI, but this driver calls `run_inference` directly and so bypasses
    that CLI entirely. Chai only *warns* when an alignment is missing, so a
    silent fetch is the difference between scoring against the campaign MSA and
    scoring against whatever a public server returns -- the failure that made
    OpenFold3 and Protenix incomparable.
    """
    os.environ.setdefault("CHAI_DOWNLOADS_DIR", "/opt/chai-lab/downloads")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    runtime = Path(os.environ.get("CHAI_RUNTIME_DIR", "/tmp/chai1-driver"))
    for variable, name in {
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "HF_HOME": "huggingface",
        "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        location = runtime / name
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)

    import requests
    from chai_lab.utils import paths

    def require_local(http_url, path):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise FileNotFoundError(f"Required offline asset is absent: {path}")

    def deny_http(self, method, url, *args, **kwargs):
        raise RuntimeError(f"HTTP is disabled in this image: {method} {url}")

    paths.download_if_not_exists = require_local
    requests.sessions.Session.request = deny_http


# --- design set -------------------------------------------------------------

def read_fasta_entries(path: Path) -> list[tuple[int, str]]:
    """(index, sequence) from the design-set FASTA, in file order.

    Only the leading index is parsed. `runs/designset.py:_render_fasta` is
    explicit that everything after it is for the human reading the file and a
    driver must not key on it -- a tool name is not stable and a `design_id`
    has no business inside the container. The host rejoins index to design_id
    through the design-set manifest.
    """
    entries: list[tuple[int, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((int(header.split()[0]), "".join(chunks)))
            header, chunks = line[1:], []
        else:
            chunks.append(line)
    if header is not None:
        entries.append((int(header.split()[0]), "".join(chunks)))
    return entries


def shard_of(entries, shard: int, num_shards: int):
    """Contiguous, matching `DesignSet.shard`.

    Contiguous rather than strided keeps each shard's lengths adjacent, which
    is what limits recompiles and, for Chai, keeps ESM batch shapes stable.
    """
    total = len(entries)
    per = -(-total // num_shards)
    start = shard * per
    return entries[start : min(start + per, total)]


# --- metrics ----------------------------------------------------------------

def read_single_fasta(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def _relative(path: Path, save_dir: Path) -> str:
    """A task-relative path, or the absolute one when it lies outside.

    An absolute path recorded inside a container is wrong the moment the
    campaign moves, so the adapter is given something it can resolve against
    the run directory.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(save_dir).resolve()))
    except ValueError:
        return str(path)


def write_chai_fasta_monomer(path: Path, sequence: str) -> None:
    """One chain, no target. Chai-1's analogue of the mosaic scorer's monomer
    reader: it folds the sequence alone, which is what a "does this model
    produce the right fold" check needs."""
    path.write_text(f">protein|name=monomer\n{sequence}\n")


def write_chai_fasta(path: Path, target_sequence: str, binder_sequence: str) -> None:
    """Target first, binder second.

    Chai names asym units A, B, C... in input order, so this makes the target
    chain A -- the same letter the campaign's `target.chain_id` uses, which is
    what any later epitope mapping will be written against.
    """
    path.write_text(
        f">protein|name=target\n{target_sequence}\n"
        f">protein|name=binder\n{binder_sequence}\n"
    )


def metrics_from_monomer(candidates, i: int) -> dict:
    """A single-chain fold. Only the whole-chain numbers exist -- there is no
    interface, so no ipTM, no directional PAE and no clash-between-chains.
    Emitting those as zero would read as a measured bad interface rather than
    an absent one."""
    import torch

    ranking = candidates.ranking_data[i]
    plddt = candidates.plddt[i].to(torch.float32)
    pae = candidates.pae[i].to(torch.float32)
    return {
        "mono_plddt": float(plddt.mean().item()),
        "mono_ptm": float(ranking.ptm_scores.complex_ptm.reshape(-1)[0].item()),
        "mono_pae": float(pae.mean().item()),
    }


def metrics_from_candidate(candidates, i: int, n_target: int, n_binder: int) -> dict:
    """One candidate structure's numbers.

    Chain order matches `write_chai_fasta`: target is chain 0, binder chain 1.
    """
    import torch

    ranking = candidates.ranking_data[i]
    plddt = candidates.plddt[i].to(torch.float32)
    pae = candidates.pae[i].to(torch.float32)

    n_tokens = plddt.shape[0]
    if n_tokens != n_target + n_binder:
        raise ValueError(
            f"token count {n_tokens} != target {n_target} + binder {n_binder}; "
            "Chai's per-residue tokenization assumption no longer holds and every "
            "per-chain slice below would be wrong"
        )
    target = slice(0, n_target)
    binder = slice(n_target, n_tokens)

    def scalar(value) -> float:
        return float(value.reshape(-1)[0].item())

    ptm = ranking.ptm_scores
    per_chain_ptm = ptm.per_chain_ptm.reshape(-1)
    n_chains = per_chain_ptm.shape[0]
    # `[query_chain, key_chain]` -- ptm.py:147 says so, and the diagonal equals
    # per_chain_ptm, which is how the ordering was confirmed against a real
    # fold. Chain 0 is the target and chain 1 the binder, per write_chai_fasta.
    pair_iptm = ptm.per_chain_pair_iptm.reshape(-1, n_chains, n_chains)[0]
    bt_iptm = float(pair_iptm[1, 0].item())
    tb_iptm = float(pair_iptm[0, 1].item())

    clashes = ranking.clash_scores
    # `chain_chain_clashes[i, j]` holds INTRA-chain clashes on the diagonal and
    # inter-chain clashes off it. Observed on a real fold as [[17, 0], [0, 11]]
    # beside has_inter_chain_clashes=False: counting nonzero entries without
    # masking the diagonal reports a design's clashes with itself as an
    # interface problem.
    chain_clashes = clashes.chain_chain_clashes.reshape(-1, n_chains, n_chains)[0]
    off_diagonal = chain_clashes.clone()
    off_diagonal.fill_diagonal_(0)

    return {
        "aggregate_score": scalar(ranking.aggregate_score),
        "complex_ptm": scalar(ptm.complex_ptm),
        # Chai's interface_ptm is the MAX over chains (ptm.py:100), so it is
        # optimistic by construction; both directions and their minimum are
        # stored beside it.
        "iptm": scalar(ptm.interface_ptm),
        "bt_iptm": bt_iptm,
        "tb_iptm": tb_iptm,
        "iptm_min": min(bt_iptm, tb_iptm),
        "binder_ptm": float(per_chain_ptm[1].item()),
        "complex_plddt": float(plddt.mean().item()),
        "binder_plddt": float(plddt[binder].mean().item()),
        # Chai reports PAE in angstroms, same orientation as the registry
        # expects (raw, lower is better).
        "bt_pae": float(pae[binder, target].mean().item()),
        "tb_pae": float(pae[target, binder].mean().item()),
        "has_clashes": float(bool(clashes.has_inter_chain_clashes.any().item())),
        "n_clashing_chain_pairs": float((off_diagonal > 0).sum().item() / 2.0),
        "binder_intra_clashes": float(chain_clashes[1, 1].item()),
    }
    # Deliberately absent: iplddt and the ipSAE family. iplddt needs interface
    # residues from coordinates, and ipSAE is mosaic's computation over its own
    # PAE convention. Emitting either from a different definition under the same
    # registry key would make the columns look joinable when they are not.


# --- main -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The driver's argument surface, separated from main().

    `tests/test_chai1.py` parses the argv that `launch.py` builds with *this*
    parser. Duplicating the flag list in the test would only prove the test
    agrees with itself; a connector that grows a flag the driver never learned
    would still launch a GPU job that dies in argparse.
    """
    p = argparse.ArgumentParser(description="Score a design-set shard with Chai-1")
    p.add_argument("--design-set", type=Path, required=True)
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-chain", default="A")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--num-trunk-recycles", type=int, required=True)
    p.add_argument("--num-diffn-timesteps", type=int, required=True)
    p.add_argument("--num-diffn-samples", type=int, required=True)
    p.add_argument("--num-trunk-samples", type=int, required=True)
    p.add_argument("--recycle-msa-subsample", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--msa-directory", type=Path, default=None)
    p.add_argument("--no-esm-embeddings", action="store_true")
    p.add_argument("--low-memory", action="store_true")
    p.add_argument("--monomer", action="store_true",
                   help="fold each sequence alone, with no target chain")
    return p


def main() -> int:
    args = build_parser().parse_args()

    install_offline_policy()
    from chai_lab.chai1 import run_inference

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.save_dir / "metrics.jsonl"
    status_path = args.save_dir / "status.json"

    target_sequence = read_single_fasta(args.target_fasta)
    entries = shard_of(
        read_fasta_entries(args.design_set), args.shard, args.num_shards
    )

    started, t_start = now(), time.time()
    produced: set[int] = set()
    attempted = 0
    failures: dict[str, int] = {}

    with metrics_path.open("w") as sink:
        for index, binder_sequence in entries:
            attempted += 1
            began = time.time()
            fold_dir = args.work_dir / f"design-{index:06d}"
            try:
                if fold_dir.exists():
                    # run_inference asserts an empty output directory, so a
                    # retried task must not inherit the previous attempt's.
                    for child in sorted(fold_dir.rglob("*"), reverse=True):
                        child.unlink() if child.is_file() else child.rmdir()
                fold_dir.mkdir(parents=True, exist_ok=True)
                fasta = args.work_dir / f"design-{index:06d}.fasta"
                if args.monomer:
                    write_chai_fasta_monomer(fasta, binder_sequence)
                else:
                    write_chai_fasta(fasta, target_sequence, binder_sequence)

                candidates = run_inference(
                    fasta_file=fasta,
                    output_dir=fold_dir,
                    use_esm_embeddings=not args.no_esm_embeddings,
                    use_msa_server=False,
                    msa_directory=args.msa_directory,
                    num_trunk_recycles=args.num_trunk_recycles,
                    num_diffn_timesteps=args.num_diffn_timesteps,
                    num_diffn_samples=args.num_diffn_samples,
                    num_trunk_samples=args.num_trunk_samples,
                    recycle_msa_subsample=args.recycle_msa_subsample,
                    seed=args.seed,
                    low_memory=args.low_memory,
                )

                # Every candidate is a replicate. Stored as rows, never
                # collapsed here: the mean of six samples and one sample that
                # happened to look good are different facts.
                for replicate in range(len(candidates.cif_paths)):
                    values = (
                        metrics_from_monomer(candidates, replicate)
                        if args.monomer
                        else metrics_from_candidate(
                            candidates, replicate,
                            len(target_sequence), len(binder_sequence),
                        )
                    )
                    sink.write(
                        json.dumps(
                            {
                                "index": index,
                                "replicate": replicate,
                                "condition": "monomer" if args.monomer else "complex",
                                "metrics": values,
                                "seconds": round(time.time() - began, 2),
                                # Relative to the task directory, matching the
                            # mosaic driver, so one adapter turns both into
                            # artifact rows.
                            "structure": _relative(
                                candidates.cif_paths[replicate], args.save_dir
                            ),
                            }
                        )
                        + "\n"
                    )
                sink.flush()
                if candidates.cif_paths:
                    produced.add(index)
            except Exception as error:  # noqa: BLE001 - one design must not kill a shard
                reason = f"{type(error).__name__}: {error}"[:200]
                failures[type(error).__name__] = failures.get(type(error).__name__, 0) + 1
                print(f"design {index} failed: {reason}", file=sys.stderr)
                traceback.print_exc()
                sink.write(
                    json.dumps({"index": index, "failed": reason})
                    + "\n"
                )
                sink.flush()

    # A shard that measured nothing is a failure even if the process exited 0.
    # Reporting success with an empty reader is the bug that let AF2 claim 20
    # designs scored while every complex fold had failed.
    ok = bool(entries) and len(produced) == len(entries)
    status_path.write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "status": "succeeded" if ok else ("partial" if produced else "failed"),
                "n_requested": len(entries),
                "n_attempted": attempted,
                "n_produced": len(produced),
                "produced_by_reader": {"complex": len(produced)},
                "failures": failures,
                "output_file": "metrics.jsonl",
                "started_at": started,
                "finished_at": now(),
                "seconds": round(time.time() - t_start, 2),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
