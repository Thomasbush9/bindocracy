#!/usr/bin/env python3
"""Co-fold a shard of a design set with upstream OpenFold3, inside openfold3.sif.

The official `openfoldconsortium/openfold3` image, not mosaic's `jopenfold3`
port. The two are not interchangeable: on GFP, mosaic's OF3 returns pLDDT 38.5
and a structure 24 A from consensus, while this one returns 88.7 and 3.9 A --
and the two disagree with each other by 24.6 A. Metrics are therefore stored
under `of3_upstream_*`, never `of3_*`.

Emits the same `metrics.jsonl` contract as every other scorer here, so one
adapter reads all four.

**The MSA filename is load-bearing.** `parse_msas_direct` keeps only files whose
*basename* is a key of `max_seq_counts` and silently `continue`s past the rest
(`core/data/io/sequence/msa.py:274`). A correctly-formatted a3m named anything
else is dropped, the MSA dict comes back empty, and the run dies several frames
later on `sorted(...)[0]` with an IndexError that names nothing relevant.
Observed 2026-09-10. So the alignment is staged as `colabfold_main.a3m`.

That is the third variant of this trap in this campaign -- Boltz keys the MSA
off the a3m's first header, Chai off a sha256 of the sequence, OpenFold3 off the
filename -- and all three fail quietly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

TOKENS = "ARNDCQEGHILKMFPSTWYV"
# One of the thirteen basenames OpenFold3's parser accepts. Ours are ColabFold
# searches, so this is the honest key as well as an accepted one.
MSA_BASENAME = "colabfold_main.a3m"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_fasta_entries(path: Path) -> list[tuple[int, str]]:
    """(index, sequence). Only the leading index is parsed -- `designset.py`
    is explicit that the rest of the header is for the human."""
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
    per = -(-len(entries) // num_shards)
    return entries[shard * per : min((shard + 1) * per, len(entries))]


def read_single_fasta(path: Path) -> str:
    return "".join(
        ln.strip() for ln in path.read_text().splitlines() if not ln.startswith(">")
    )


def stage_msa(a3m: Path, work: Path) -> Path:
    """Copy the alignment to the one filename OpenFold3 will not discard."""
    staged = work / "msa"
    staged.mkdir(parents=True, exist_ok=True)
    destination = staged / MSA_BASENAME
    if not destination.exists():
        shutil.copyfile(a3m, destination)
    return destination


def build_query(
    entries, target: str, msa: Path | None, seed: int, monomer: bool
) -> dict:
    """One query per design. Target is chain A, binder chain B.

    Chain A for the target matches `general.target.chain_id` and both other
    co-folding drivers, so all three agree on which chain is which and a later
    epitope mapping is written once.
    """
    queries = {}
    for index, binder in entries:
        if monomer:
            chains = [{
                "molecule_type": "protein", "chain_ids": ["A"], "sequence": binder,
            }]
            if msa is not None:
                chains[0]["main_msa_file_paths"] = [str(msa)]
        else:
            target_chain = {
                "molecule_type": "protein", "chain_ids": ["A"], "sequence": target,
            }
            if msa is not None:
                target_chain["main_msa_file_paths"] = [str(msa)]
            chains = [
                target_chain,
                # The binder gets no alignment: a de novo binder has no
                # homologs, which is what every other scorer here assumes too.
                {"molecule_type": "protein", "chain_ids": ["B"], "sequence": binder},
            ]
        queries[f"design-{index:06d}"] = {
            "chains": chains,
            # No paired MSA: there is only one aligned chain, so pairing has
            # nothing to pair with.
            "use_paired_msas": False,
        }
    return {"seeds": [seed], "queries": queries}


def plddt_from_cif(path: Path, binder_chain: str = "B") -> tuple[float, float]:
    """(complex, binder) mean pLDDT from the mmCIF B-factor column."""
    total: list[float] = []
    binder: list[float] = []
    columns: list[str] = []
    in_loop = False
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            in_loop = True
            columns.append(line.strip().split(".")[1])
            continue
        if in_loop and line.startswith(("ATOM", "HETATM")):
            parts = line.split()
            if len(parts) < len(columns):
                continue
            row = dict(zip(columns, parts, strict=False))
            try:
                value = float(row.get("B_iso_or_equiv", "nan"))
            except ValueError:
                continue
            total.append(value)
            if binder_chain in (row.get("label_asym_id"), row.get("auth_asym_id")):
                binder.append(value)
        elif in_loop and line.startswith("#"):
            in_loop = False

    def mean(values):
        return float(sum(values) / len(values)) if values else float("nan")

    return mean(total), mean(binder)


def metrics_from_confidences(
    conf: dict, complex_plddt: float, binder_plddt: float, monomer: bool
) -> dict:
    """OpenFold3's aggregated confidences onto the shared registry keys.

    A monomer fold has no interface, so the ipTM family is absent rather than
    zero -- emitting zero would read as a measured bad interface instead of an
    absent one.
    """
    if monomer:
        return {
            "mono_plddt": complex_plddt,
            "mono_ptm": conf.get("ptm"),
        }

    chain_ptm = conf.get("chain_ptm") or {}
    # `chain_pair_iptm` is keyed "(A, B)" and holds one value for a two-chain
    # complex, so the two directions are not separable here as they are for
    # Chai-1 and AF3.
    pair = conf.get("chain_pair_iptm") or {}
    pair_value = next(iter(pair.values()), None) if pair else None

    return {
        "complex_ptm": conf.get("ptm"),
        "iptm": conf.get("iptm"),
        "bt_iptm": pair_value,
        "binder_ptm": chain_ptm.get("B"),
        "complex_plddt": complex_plddt,
        "binder_plddt": binder_plddt,
        "aggregate_score": conf.get("sample_ranking_score"),
        "has_clashes": float(bool(conf.get("has_clash"))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Score a design-set shard with OpenFold3")
    p.add_argument("--design-set", type=Path, required=True)
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-diffusion-samples", type=int, default=1)
    p.add_argument("--monomer", action="store_true",
                   help="fold each sequence alone, with no target chain")
    args = p.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    structures = args.save_dir / "structures" / ("monomer" if args.monomer else "complex")
    metrics_path = args.save_dir / "metrics.jsonl"
    status_path = args.save_dir / "status.json"

    target = read_single_fasta(args.target_fasta)
    msa = stage_msa(args.target_msa, args.work_dir) if args.target_msa else None
    print(f"target {len(target)} aa; alignment {msa}", flush=True)

    entries = shard_of(read_fasta_entries(args.design_set), args.shard, args.num_shards)
    query = build_query(entries, target, msa, args.seed, args.monomer)
    query_path = args.work_dir / "query.json"
    query_path.write_text(json.dumps(query, indent=2))

    started, t_start = now(), time.time()
    predictions = args.work_dir / "predictions"

    # One invocation for the whole shard: OpenFold3 loads a 2.2 GB checkpoint
    # and builds its featurisation pipeline once, so per-design invocations
    # would pay that repeatedly.
    result = subprocess.run(
        ["run_openfold", "predict",
         "--query-json", str(query_path),
         "--inference-ckpt-path", str(args.checkpoint),
         "--num-diffusion-samples", str(args.num_diffusion_samples),
         "--num-model-seeds", "1",
         "--use-msa-server", "false",
         "--use-templates", "false",
         "--output-dir", str(predictions)],
        capture_output=True, text=True, check=False,
    )
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr, flush=True)

    produced: set[int] = set()
    failures: dict[str, int] = {}

    with metrics_path.open("w") as sink:
        for index, _binder in entries:
            name = f"design-{index:06d}"
            found = sorted((predictions / name).rglob("*_confidences_aggregated.json"))
            if not found:
                # A design OpenFold3 skipped. It logs the reason per query and
                # continues, so one bad design does not lose the shard.
                failures["no_prediction"] = failures.get("no_prediction", 0) + 1
                sink.write(json.dumps(
                    {"index": index, "failed": "no prediction written"}) + "\n")
                continue
            for replicate, conf_path in enumerate(found):
                cif = Path(str(conf_path).replace(
                    "_confidences_aggregated.json", "_model.cif"))
                complex_plddt = binder_plddt = float("nan")
                relative = None
                if cif.is_file():
                    complex_plddt, binder_plddt = plddt_from_cif(cif)
                    structures.mkdir(parents=True, exist_ok=True)
                    kept = structures / f"{name}_s{replicate}.cif"
                    shutil.copyfile(cif, kept)
                    relative = str(kept.relative_to(args.save_dir))
                values = metrics_from_confidences(
                    json.loads(conf_path.read_text()),
                    complex_plddt, binder_plddt, args.monomer,
                )
                values = {k: v for k, v in values.items() if v is not None}
                sink.write(json.dumps({
                    "index": index,
                    "condition": "monomer" if args.monomer else "complex",
                    "replicate": replicate, "metrics": values,
                    "seconds": round(time.time() - t_start, 2),
                    "structure": relative,
                }) + "\n")
            produced.add(index)
        sink.flush()

    reader = "monomer" if args.monomer else "complex"
    ok = bool(entries) and len(produced) == len(entries)
    status_path.write_text(json.dumps({
        "task_id": args.task_id,
        "status": "succeeded" if ok else ("partial" if produced else "failed"),
        "n_requested": len(entries), "n_attempted": len(entries),
        "n_produced": len(produced),
        "produced_by_reader": {reader: len(produced)},
        "failures": failures, "output_file": "metrics.jsonl",
        "exit_code": result.returncode,
        "started_at": started, "finished_at": now(),
        "seconds": round(time.time() - t_start, 2),
    }, indent=2) + "\n")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
