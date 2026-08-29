#!/usr/bin/env python3
"""Stage RFdiffusion backbones for Caliby and write its fixed-chain CSV.

Two Caliby behaviours make this script necessary rather than optional.

1.  `input_cfg.pdb_dir` globs `{pdb_dir}/*`, NOT `*.pdb`. RFdiffusion writes
    `<prefix>_<i>.pdb`, `<prefix>_<i>.trb` and a `traj/` subdirectory all into
    one place, and handing a .trb to the structure parser crashes the run. So
    the PDBs are copied into a directory that contains nothing else.

2.  Chain fixing is done through `pos_constraint_csv`, and a `pdb_key` missing
    from that CSV does NOT raise. Caliby prints "No fixed positions found" and
    redesigns the whole complex — including the target. Silent, and fatal for a
    binder benchmark. The CSV is therefore generated from the actual directory
    listing, and the row count is asserted against the file count.

A bare chain letter in `fixed_pos_seq` fixes that entire chain, which also
sidesteps Caliby's label_seq_id-vs-auth_seq_id numbering trap that explicit
ranges like `B1-201` would expose us to.

NOTE ON CHAIN IDS: RFdiffusion relabels the output so the BINDER is chain A and
the TARGET is chain B. That is the reverse of the input. We therefore fix
chain B.

Deliberately annotation-free: the FASRC hosts ship /usr/bin/python3 == 3.6.8.
"""

import argparse
import os
import shutil
from pathlib import Path


def chains_in(pdb_path):
    seen = []
    with open(str(pdb_path)) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                c = line[21]
                if c not in seen:
                    seen.append(c)
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="directory of RFdiffusion .pdb output")
    ap.add_argument("--pattern", default="*.pdb")
    ap.add_argument("--stage-dir", required=True, help="clean dir of structures only")
    ap.add_argument("--csv", required=True, help="pos_constraint_csv to write")
    ap.add_argument("--fixed-chain", default="B",
                    help="chain to hold fixed (the TARGET). RFdiffusion makes "
                         "the binder chain A and the target chain B.")
    a = ap.parse_args()

    src = Path(a.src)
    pdbs = sorted(src.glob(a.pattern))
    if not pdbs:
        raise SystemExit("FATAL: no structures matched %s/%s" % (src, a.pattern))

    stage = Path(a.stage_dir)
    if stage.exists():
        shutil.rmtree(str(stage))
    stage.mkdir(parents=True)

    rows = []
    bad = []
    for p in pdbs:
        chains = chains_in(p)
        if a.fixed_chain not in chains:
            bad.append((p.name, chains))
            continue
        shutil.copy2(str(p), str(stage / p.name))
        rows.append(p.stem)

    if bad:
        raise SystemExit(
            "FATAL: chain %r absent from %d structure(s), e.g. %s (chains %s).\n"
            "  Check which chain RFdiffusion assigned to the target."
            % (a.fixed_chain, len(bad), bad[0][0], bad[0][1]))

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w") as fh:
        fh.write("pdb_key,fixed_pos_seq,fixed_pos_scn\n")
        for key in rows:
            # fixed_pos_scn must be a subset of fixed_pos_seq. It conditions on
            # the target's sidechain coordinates where they are resolved, and is
            # a no-op where they are not -- RFdiffusion emits backbone only, so
            # it costs nothing here and is correct if the input ever gains them.
            fh.write("%s,%s,%s\n" % (key, a.fixed_chain, a.fixed_chain))

    staged = len(list(stage.iterdir()))
    if staged != len(rows):
        raise SystemExit("FATAL: staged %d files but wrote %d CSV rows" % (staged, len(rows)))

    print("staged %d structures -> %s" % (staged, stage))
    print("wrote %d constraint rows -> %s (fixed chain %s)" % (len(rows), a.csv, a.fixed_chain))


if __name__ == "__main__":
    main()
