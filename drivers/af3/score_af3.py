#!/usr/bin/env python3
"""Co-fold a shard of a design set with AlphaFold 3, inside af3.sif.

Runs in the container. Emits the same `metrics.jsonl` + `status.json` contract
as the mosaic and Chai-1 drivers, so the same adapter reads all three.

**No genetic databases.** AF3's data pipeline wants ~630 GB of BFD, UniRef and
MGnify. We never run it: the campaign already has a ColabFold alignment for the
target, and AF3 takes it directly as `unpairedMsa` (docs/input.md:459). So this
always passes `--norun_data_pipeline`, and the alignment reaching the model is
the same one every other scorer here folds against -- which is the whole point
of having a campaign MSA.

The binder chain is MSA-free (`unpairedMsa: ""`), matching every other scorer:
a de novo binder has no homologs by construction.
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

RUN_ALPHAFOLD = "/app/alphafold/run_alphafold.py"
TOKENS = "ARNDCQEGHILKMFPSTWYV"


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_fasta_entries(path: Path) -> list[tuple[int, str]]:
    """(index, sequence), parsing only the leading index from each header.

    `runs/designset.py:_render_fasta` is explicit that the rest of the header is
    for the human and a driver must not key on it.
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
    per = -(-len(entries) // num_shards)
    return entries[shard * per : min((shard + 1) * per, len(entries))]


def read_single_fasta(path: Path) -> str:
    return "".join(
        ln.strip() for ln in path.read_text().splitlines() if not ln.startswith(">")
    )


def fold_input(name: str, target: str, binder: str, target_a3m: str, seed: int) -> dict:
    """Target as chain A, binder as chain B.

    Chain A for the target matches `general.target.chain_id` and the Chai-1
    driver, so the two co-folding scorers put the same molecule in the same
    chain and a later epitope mapping is written once.
    """
    return {
        "name": name,
        "modelSeeds": [seed],
        "sequences": [
            {"protein": {"id": "A", "sequence": target,
                         "unpairedMsa": target_a3m, "pairedMsa": "",
                         "templates": []}},
            # MSA-free, as everywhere else here.
            {"protein": {"id": "B", "sequence": binder,
                         "unpairedMsa": "", "pairedMsa": "", "templates": []}},
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def plddt_from_cif(path: Path, n_target: int) -> tuple[float, float]:
    """(complex, binder) mean pLDDT from the mmCIF B-factor column.

    AF3 writes pLDDT per atom there. Averaged over atoms rather than residues,
    which is what the file supports without a residue map; both models are
    treated the same way so the numbers stay comparable within this driver.
    """
    total: list[float] = []
    binder: list[float] = []
    in_loop = False
    columns: list[str] = []
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
            if row.get("label_asym_id") == "B" or row.get("auth_asym_id") == "B":
                binder.append(value)
        elif in_loop and line.startswith("#"):
            in_loop = False
    def mean(values):
        return float(sum(values) / len(values)) if values else float("nan")

    return mean(total), mean(binder)


def metrics_from_summary(summary: dict, complex_plddt: float, binder_plddt: float) -> dict:
    """AF3's summary confidences onto the shared registry keys.

    Chain order is [A=target, B=binder], set by `fold_input`.
    """
    pair_iptm = summary.get("chain_pair_iptm") or [[None, None], [None, None]]
    pair_pae = summary.get("chain_pair_pae_min") or [[None, None], [None, None]]
    chain_ptm = summary.get("chain_ptm") or [None, None]

    def cell(matrix, i, j):
        try:
            value = matrix[i][j]
            return None if value is None else float(value)
        except (IndexError, TypeError):
            return None

    bt_iptm, tb_iptm = cell(pair_iptm, 1, 0), cell(pair_iptm, 0, 1)
    directional = [v for v in (bt_iptm, tb_iptm) if v is not None]

    return {
        "complex_ptm": summary.get("ptm"),
        "iptm": summary.get("iptm"),
        "aggregate_score": summary.get("ranking_score"),
        "bt_iptm": bt_iptm,
        "tb_iptm": tb_iptm,
        "iptm_min": min(directional) if directional else None,
        "binder_ptm": float(chain_ptm[1]) if len(chain_ptm) > 1 and chain_ptm[1] is not None else None,
        "complex_plddt": complex_plddt,
        "binder_plddt": binder_plddt,
        "bt_pae": cell(pair_pae, 1, 0),
        "tb_pae": cell(pair_pae, 0, 1),
        "has_clashes": float(bool(summary.get("has_clash"))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Score a design-set shard with AlphaFold 3")
    p.add_argument("--design-set", type=Path, required=True)
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, required=True,
                   help="ColabFold a3m for the target; passed as unpairedMsa")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-diffn-samples", type=int, default=5)
    args = p.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    structures = args.save_dir / "structures" / "complex"
    metrics_path = args.save_dir / "metrics.jsonl"
    status_path = args.save_dir / "status.json"

    target = read_single_fasta(args.target_fasta)
    target_a3m = args.target_msa.read_text()
    print(f"target {len(target)} aa, a3m {target_a3m.count('>')} sequences", flush=True)

    entries = shard_of(read_fasta_entries(args.design_set), args.shard, args.num_shards)
    started, t_start = now(), time.time()
    produced: set[int] = set()
    failures: dict[str, int] = {}
    attempted = 0

    with metrics_path.open("w") as sink:
        for index, binder in entries:
            attempted += 1
            began = time.time()
            name = f"design-{index:06d}"
            job = args.work_dir / name
            try:
                bad = sorted({aa for aa in binder if aa not in TOKENS})
                if bad:
                    raise ValueError(f"non-standard residues {bad}")
                job.mkdir(parents=True, exist_ok=True)
                json_path = job / "fold_input.json"
                json_path.write_text(
                    json.dumps(fold_input(name, target, binder, target_a3m, args.seed))
                )
                out_dir = job / "out"
                result = subprocess.run(
                    [sys.executable, RUN_ALPHAFOLD,
                     f"--json_path={json_path}", f"--model_dir={args.model_dir}",
                     f"--output_dir={out_dir}", "--norun_data_pipeline"],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"run_alphafold exit {result.returncode}: "
                        f"{result.stderr.strip()[-300:]}"
                    )

                summaries = sorted(out_dir.rglob("*_summary_confidences.json"))
                if not summaries:
                    raise RuntimeError("no summary_confidences.json produced")

                for replicate, summary_path in enumerate(summaries):
                    cif = Path(str(summary_path).replace(
                        "_summary_confidences.json", "_model.cif"))
                    complex_plddt = binder_plddt = float("nan")
                    relative = None
                    if cif.is_file():
                        complex_plddt, binder_plddt = plddt_from_cif(cif, len(target))
                        structures.mkdir(parents=True, exist_ok=True)
                        kept = structures / f"{name}_s{replicate}.cif"
                        shutil.copyfile(cif, kept)
                        relative = str(kept.relative_to(args.save_dir))
                    values = metrics_from_summary(
                        json.loads(summary_path.read_text()), complex_plddt, binder_plddt
                    )
                    values = {k: v for k, v in values.items() if v is not None}
                    sink.write(json.dumps({
                        "index": index, "condition": "complex", "replicate": replicate,
                        "metrics": values, "seconds": round(time.time() - began, 2),
                        "structure": relative,
                    }) + "\n")
                sink.flush()
                produced.add(index)
                print(f"  {name}: {len(summaries)} sample(s) "
                      f"in {time.time() - began:.0f}s", flush=True)
            except Exception as error:  # noqa: BLE001 - one design must not kill a shard
                reason = f"{type(error).__name__}: {error}"[:300]
                failures[type(error).__name__] = failures.get(type(error).__name__, 0) + 1
                print(f"  {name} FAILED: {reason}", file=sys.stderr, flush=True)
                sink.write(json.dumps({"index": index, "failed": reason}) + "\n")
                sink.flush()

    ok = bool(entries) and len(produced) == len(entries)
    status_path.write_text(json.dumps({
        "task_id": args.task_id,
        "status": "succeeded" if ok else ("partial" if produced else "failed"),
        "n_requested": len(entries), "n_attempted": attempted,
        "n_produced": len(produced),
        "produced_by_reader": {"complex": len(produced)},
        "failures": failures, "output_file": "metrics.jsonl",
        "started_at": started, "finished_at": now(),
        "seconds": round(time.time() - t_start, 2),
    }, indent=2) + "\n")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
