#!/usr/bin/env python3
"""Seed Protein-Hunter's ColabFold MSA cache so the run stays offline.

Protein-Hunter's Boltz pipeline has only `--msa_mode {single,mmseqs}`. `single`
folds the target with no MSA at all; `mmseqs` calls
https://api.colabfold.com, which a compute node cannot reach. There is no flag
for a precomputed a3m.

The way through is that `run_mmseqs2` is cache-first: it skips the HTTP call
entirely when `{prefix}_env/out.tar.gz` exists, and skips untarring when the
a3m files are already there. So we write the cache by hand and then ask for
`--msa_mode mmseqs`.

Two details that are easy to get wrong and fail loudly/silently:

1.  The parser does `M = int(line[1:].rstrip())` on the FIRST header, so it must
    read `>101` — a literal `>DIO3` raises ValueError. Only the first one; every
    later header is passed through untouched.
2.  `max_seqs` is hardcoded to 4096 downstream and overrides the caller, so the
    full ~3000-sequence alignment would be pushed through the MSA module on
    every one of ~240 predictions. Subsampling to a few hundred costs almost
    nothing in accuracy and saves hours.

Usage:
    seed_msa_cache.py --a3m <in.a3m> --env-dir <save_dir>/0_protein_hunter_design/B_env --max-seqs 512

`B` is the chain id because the binder is chain A and the single target chain
becomes chain B.
"""

# Deliberately free of type annotations and f-string-only syntax beyond 3.6:
# the FASRC login and compute nodes ship /usr/bin/python3 == 3.6.8, and this
# script runs on the host, outside any container.

import argparse
from pathlib import Path


def read_a3m(path):
    """Return [(header, [sequence lines])], preserving a3m insertion casing."""
    records = []
    header = None
    body = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                records.append((header, body))
            header, body = line, []
        elif header is not None and line:
            body.append(line)
    if header is not None:
        records.append((header, body))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a3m", required=True, type=Path)
    ap.add_argument("--env-dir", required=True, type=Path)
    ap.add_argument("--max-seqs", type=int, default=512)
    a = ap.parse_args()

    records = read_a3m(a.a3m)
    if not records:
        raise SystemExit(f"FATAL: no sequences parsed from {a.a3m}")

    # Keep the query (record 0) plus the next max_seqs-1 hits. The alignment is
    # already ordered by E-value, so a prefix is the natural subsample.
    kept = records[: max(1, a.max_seqs)]

    a.env_dir.mkdir(parents=True, exist_ok=True)

    # Presence alone blocks the HTTP call; content is never read because the
    # a3m files below already exist.
    (a.env_dir / "out.tar.gz").write_bytes(b"")
    # Must exist or the loader tries to untar it.
    (a.env_dir / "bfd.mgnify30.metaeuk30.smag30.a3m").write_text("")

    with (a.env_dir / "uniref.a3m").open("w") as fh:
        for i, (header, body) in enumerate(kept):
            fh.write(">101\n" if i == 0 else f"{header}\n")
            for line in body:
                fh.write(f"{line}\n")

    print(f"seeded {a.env_dir}")
    print(f"  parsed {len(records)} sequences, wrote {len(kept)}")
    print(f"  query header rewritten to '>101' (was {records[0][0][:40]!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
