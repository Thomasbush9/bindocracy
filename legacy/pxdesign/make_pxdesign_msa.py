#!/usr/bin/env python3
"""
Convert a single ColabFold-style .a3m into the two-file MSA directory that
PXDesign / Protenix expects for an offline precomputed MSA:

    <outdir>/non_pairing.a3m
    <outdir>/pairing.a3m

Usage:
    python make_pxdesign_msa.py IN.a3m OUTDIR [--target-seq SEQ_OR_FASTA]
"""
import argparse
import os
import string
import sys

DELETE = str.maketrans("", "", string.ascii_lowercase)


def read_a3m(path):
    heads, seqs = [], []
    cur = None
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if cur is not None:
                    heads.append(cur)
                    seqs.append("".join(buf))
                cur = line
                buf = []
            elif line:
                buf.append(line)
    if cur is not None:
        heads.append(cur)
        seqs.append("".join(buf))
    return heads, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a3m")
    ap.add_argument("outdir")
    ap.add_argument("--target-seq", default=None,
                    help="Expected full-length target sequence (literal, or path to a FASTA). "
                         "The a3m query MUST match this exactly.")
    args = ap.parse_args()

    heads, seqs = read_a3m(args.a3m)
    if not heads:
        sys.exit("ERROR: empty a3m")

    query = seqs[0].translate(DELETE)
    if "-" in query:
        sys.exit("ERROR: query row contains gaps; it must be ungapped.")
    width = len(query)
    print(f"query header : {heads[0]}")
    print(f"query length : {width}")

    if args.target_seq:
        exp = args.target_seq
        if os.path.exists(exp):
            exp = "".join(l.strip() for l in open(exp) if not l.startswith(">"))
        exp = exp.strip()
        if exp != query:
            sys.exit(
                f"ERROR: a3m query ({len(query)} aa) != target sequence ({len(exp)} aa).\n"
                "  PXDesign requires the MSA to be built on the FULL-LENGTH target chain\n"
                "  sequence exactly as it appears in the target CIF."
            )
        print("target sequence: MATCHES a3m query")

    # Validate every row has the same match-state width, drop bad/duplicate rows.
    kept_h, kept_s, dropped_width, dropped_dup = [], [], 0, 0
    for h, s in zip(heads[1:], seqs[1:]):
        if s.translate(DELETE) == query:
            dropped_dup += 1
            continue
        if len(s.translate(DELETE)) != width:
            dropped_width += 1
            continue
        kept_h.append(h)
        kept_s.append(s)

    os.makedirs(args.outdir, exist_ok=True)

    np_path = os.path.join(args.outdir, "non_pairing.a3m")
    with open(np_path, "w") as f:
        f.write(">query\n%s\n" % seqs[0])
        for h, s in zip(kept_h, kept_s):
            f.write("%s\n%s\n" % (h, s))

    # PXDesign requires pairing.a3m to exist. For a monomeric target + de-novo
    # binder it is never actually consumed (both stages see a single MSA-bearing
    # entity), and we have no UniRef100_<acc>_<taxid>/ headers to pair on, so we
    # write the query-only dummy that Protenix itself writes in this situation.
    p_path = os.path.join(args.outdir, "pairing.a3m")
    with open(p_path, "w") as f:
        f.write(">query\n%s\n" % query)

    print(f"\nwrote {np_path}  ({len(kept_h) + 1} sequences)")
    print(f"wrote {p_path}  (1 sequence, dummy)")
    if dropped_dup:
        print(f"  dropped {dropped_dup} row(s) identical to query")
    if dropped_width:
        print(f"  dropped {dropped_width} row(s) with mismatched match-state width")


if __name__ == "__main__":
    main()
