#!/usr/bin/env python3
"""Inventory what each tool actually produced.

Deliberately an INVENTORY, not a unifier: it reports where each tool put its
binder sequences and structures and how many there are, so that a decision about
common post-processing can be made from facts rather than from the upstream
READMEs (several of which are wrong about their own output paths).

    inventory.py [--outputs <dir>] [--tsv <path>]

For each tool it prints the design count, the file that holds the sequences, and
the structure directory. No tool in this suite writes a FASTA; sequences live in
CSV columns or are implicit in structures.

Written for Python 3.6 (the FASRC host interpreter), stdlib only.
"""

import argparse
import csv
import glob
import os
import sys

DEFAULT_OUTPUTS = ("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
                   "binder_design/outputs")


def count_lines_startswith(path, prefix):
    n = 0
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(prefix):
                    n += 1
    except IOError:
        return 0
    return n


def csv_rows(path, seq_col=None):
    """Return (n_rows, n_nonempty_seq). seq_col may be a list of candidates."""
    try:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    except IOError:
        return (0, 0)
    if not rows:
        return (0, 0)
    col = None
    if seq_col:
        for c in seq_col:
            if c in rows[0]:
                col = c
                break
    if col is None:
        return (len(rows), 0)
    return (len(rows), sum(1 for r in rows if (r.get(col) or "").strip()))


def report(name, count, seq_loc, struct_loc, note=""):
    return {
        "tool": name,
        "designs": count,
        "sequences_in": seq_loc,
        "structures_in": struct_loc,
        "note": note,
    }


def collect(out):
    j = os.path.join
    res = []

    # --- Mosaic: FASTA-like .txt plus a jsonl sibling with per-design timing.
    d = j(out, "mosaic", "designs")
    n = sum(count_lines_startswith(p, ">") for p in glob.glob(j(d, "designs_*.txt")))
    res.append(report("mosaic", n, j(d, "designs_*.txt"), "(none - sequence only)",
                      "header is the ranking score; designs_*.jsonl adds seconds/design"))

    # --- BoltzGen: sequences only in the final metrics CSV, and the column is
    # `designed_sequence` -- NOT `sequence`, which is what the docs imply.
    d = j(out, "boltzgen", "results", "final_ranked_designs")
    cands = sorted(glob.glob(j(d, "final_designs_metrics_*.csv")))
    n, nseq = csv_rows(cands[-1], ["designed_sequence", "sequence"]) if cands else (0, 0)
    res.append(report("boltzgen", nseq or n,
                      cands[-1] if cands else j(d, "final_designs_metrics_<budget>.csv"),
                      j(d, "final_*_designs/"),
                      "column is 'designed_sequence'; no FASTA anywhere, CIF only"))

    # --- FreeBindCraft: MPNN stats CSV is the reliable one (see known-issues).
    d = j(out, "freebindcraft")
    mp = j(d, "mpnn_design_stats.csv")
    fin = j(d, "final_design_stats.csv")
    n, nseq = csv_rows(mp, ["Sequence", "sequence"])
    nfin, _ = csv_rows(fin, ["Sequence", "sequence"])
    traj = len(glob.glob(j(d, "Trajectory", "Relaxed", "*.pdb")))
    res.append(report("freebindcraft", n,
                      mp + (" (+ final_design_stats.csv)" if nfin else ""),
                      j(d, "Accepted/") + " , " + j(d, "MPNN/Relaxed/"),
                      "%d successful trajectories; final_design_stats rows=%d"
                      % (traj, nfin)))

    # --- Genie 3: the one tool that does write per-design FASTA. The reducer
    # publishes them under dio3_cut/sequences/, but mid-run (or if the reduce
    # stage did not complete) they only exist inside the eval shard, so look in
    # both rather than reporting zero.
    d = j(out, "genie3", "dio3_cut")
    fa = glob.glob(j(d, "sequences", "*.fasta"))
    loc = j(d, "sequences/*.fasta")
    if not fa:
        shard = j(d, "eval_shards", "shard_*_of_*", "devices", "device_*",
                  "sequences", "*.fasta")
        fa = glob.glob(shard)
        if fa:
            loc = shard + "  (reduce stage has not published these yet)"
    pdbs = len(glob.glob(j(d, "pdbs", "*.pdb")))
    nseq = sum(count_lines_startswith(p, ">") for p in fa)
    af2 = len(glob.glob(j(d, "eval_shards", "shard_*_of_*", "devices", "device_*",
                          "structures", "*.pdb")))
    info = j(d, "results", "info.csv")
    if os.path.exists(info):
        af2 = max(af2, csv_rows(info)[0])
    res.append(report("genie3", pdbs, loc, j(d, "pdbs/"),
                      "%d MPNN sequences, %d AF2 model outputs; "
                      "metrics in results/info.csv" % (nseq, af2)))

    # --- Proteina-Complexa: one complex PDB per design, in its own job dir.
    # Chain A is the TARGET and chain B the binder -- the reverse of
    # RFdiffusion. The `n_<N>` in the filename is total residues (target +
    # binder), which is a quick way to read off the binder length.
    d = j(out, "proteina_complexa", "inference")
    npdb, rew = 0, ""
    for r in glob.glob(j(d, "*dio3_cut_v1")):
        npdb += len(glob.glob(j(r, "job_*", "*.pdb")))
        c = glob.glob(j(r, "rewards_*.csv"))
        if c:
            rew = c[0]
    res.append(report("proteina_complexa", npdb,
                      j(d, "*/job_*/*.pdb") + " (chain B)",
                      j(d, "*/job_*/"),
                      "chain A = target, chain B = binder (reverse of RFdiffusion); "
                      "rewards_*.csv 'aatype' is integer indices, not letters"))

    # --- Protein-Hunter: everything worth having is in the summary CSVs.
    d = j(out, "protein_hunter", "dio3_cut_boltz")
    allc = j(d, "summary_all_runs.csv")
    n, nseq = csv_rows(allc, ["best_seq"])
    hi, _ = csv_rows(j(d, "summary_high_iptm.csv"), ["sequence"])
    res.append(report("protein_hunter", n, allc,
                      j(d, "0_protein_hunter_design/run_*/"),
                      "%d rows cleared the ipTM/pLDDT gate (summary_high_iptm.csv)" % hi))

    # --- PXDesign: summary.csv is padded to N_sample with failures.
    d = glob.glob(j(out, "pxdesign", "run01", "out", "design_outputs", "*"))
    n, succ, loc = 0, 0, ""
    if d:
        loc = j(d[0], "summary.csv")
        n, _ = csv_rows(loc)
        try:
            with open(loc) as fh:
                for r in csv.DictReader(fh):
                    for k in r:
                        if "success" in k.lower() and str(r[k]).strip() in ("1", "True", "true"):
                            succ += 1
                            break
        except IOError:
            pass
    res.append(report("pxdesign", n, loc or "(not produced)",
                      (j(d[0], "orig_designed/") if d else ""),
                      "rows are PADDED with failures; %d row(s) flagged success" % succ))

    # --- RFdiffusion: backbones only, no sequences at all.
    d = j(out, "rfdiffusion", "designs")
    n = len(glob.glob(j(d, "dio3_cut_*.pdb")))
    res.append(report("rfdiffusion", n, "(none - poly-glycine backbones)", d,
                      "binder is chain A, target chain B; .trb holds the config"))

    # --- Caliby: sequences only in the CSV; files are CIF despite the column name.
    p = j(out, "caliby", "run1", "seq_des_outputs.csv")
    n, nseq = csv_rows(p, ["seq"])
    res.append(report("caliby", nseq or n, p, j(out, "caliby", "run1", "samples/"),
                      "'seq' is chains joined by ':'; out_pdb column holds .cif paths"))

    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=DEFAULT_OUTPUTS)
    ap.add_argument("--tsv", default=None)
    a = ap.parse_args()

    rows = collect(a.outputs)

    w = max(len(r["tool"]) for r in rows)
    print("%-*s  %8s  %s" % (w, "tool", "designs", "sequences live in"))
    print("-" * (w + 60))
    for r in rows:
        print("%-*s  %8d  %s" % (w, r["tool"], r["designs"],
                                 r["sequences_in"].replace(a.outputs + "/", "")))
        if r["note"]:
            print("%-*s            %s" % (w, "", r["note"]))

    if a.tsv:
        with open(a.tsv, "w") as fh:
            fh.write("tool\tdesigns\tsequences_in\tstructures_in\tnote\n")
            for r in rows:
                fh.write("%s\t%d\t%s\t%s\t%s\n" % (
                    r["tool"], r["designs"], r["sequences_in"],
                    r["structures_in"], r["note"]))
        print("\nwrote %s" % a.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
