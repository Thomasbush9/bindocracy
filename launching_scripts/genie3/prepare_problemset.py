"""Offline Genie3 binder-design problem-set prep (no ColabFold MSA server).

Only valid when evaluation.folding.mode == 'template' (or 'singleseq'),
which never reads target_msa_filepath.

Usage (inside the genie3 container):
  python prepare_offline.py \
      --pdb  /abs/path/target.pdb \
      --key  dio3_cut \
      --name DIO3-cut \
      --hotspot "A45,A48,A52" \
      --binder-min 60 --binder-max 120 \
      --outdir /abs/path/dataset          # writes <outdir>/problems + <outdir>/targets
"""

import argparse
import json
import os

import numpy as np

from genie3.generation.np.protein_constants import PROTEIN_RESTYPES, RESTYPE_3_TO_1
from genie3.generation.utils.interface.extended import compute_extended_interface
from genie3.generation.utils.pdb_utils import parse_pdb


def main(a):
    os.makedirs(os.path.join(a.outdir, "problems"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "targets", "pdb"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "targets", "fasta"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "targets", "msa"), exist_ok=True)

    hotspots = [h.strip() for h in a.hotspot.split(",") if h.strip()]

    # --- read + validate ATOM lines ---
    pdb_lines = []
    missing = set(hotspots)
    with open(a.pdb) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[16].strip() != "" or line[26].strip() != "":
                raise SystemExit("altloc / insertion codes are not supported")
            if line[12:16].strip() == "CA":
                missing.discard(f"{line[21]}{int(line[22:26])}")
            pdb_lines.append(line)
    if missing:
        raise SystemExit(f"hotspots not found in PDB: {sorted(missing)}")

    # --- per-chain sequence + renumber to 1..N, rename chains to B, C, ... ---
    structure = parse_pdb(a.pdb)
    chain_tags, seqs, remap = [], [], {}
    for i, chain in enumerate(structure.chains):
        new_cid = chr(ord("A") + i + 1)          # first target chain -> 'B'
        start = chain.residues[0].index
        end = chain.residues[-1].index
        seq = ["-"] * (end - start + 1)
        for res in chain.residues:
            j = res.index - start
            seq[j] = RESTYPE_3_TO_1[PROTEIN_RESTYPES[np.argmax(res.restype)]]
            remap[f"{chain.name}{res.index}"] = f"{new_cid}{j + 1}"
        seq = "".join(seq)
        if "-" in seq:
            raise SystemExit(
                f"chain {chain.name} has missing residues; renumber/repair the PDB first"
            )
        seqs.append(seq)
        chain_tags.append(f"{new_cid}1-{end - start + 1}")

    # --- FASTA (complex + per chain) ---
    fasta = os.path.join(a.outdir, "targets", "fasta", f"{a.key}.fasta")
    with open(fasta, "w") as fh:
        fh.write(f">{a.key}\n{':'.join(seqs)}")
    fasta_by_chain = []
    for i, tag in enumerate(chain_tags):
        p = os.path.join(a.outdir, "targets", "fasta", f"{a.key}-chain_{tag[0]}.fasta")
        fasta_by_chain.append(p)
        with open(p, "w") as fh:
            fh.write(f">{a.key}-chain_{tag[0]}\n{seqs[i]}")

    # --- PDB with renumbered/renamed chains ---
    pdb_out = os.path.join(a.outdir, "targets", "pdb", f"{a.key}.pdb")
    with open(pdb_out, "w") as fh:
        head = [f"REMARK 999 KEY    {a.key}\n", f"REMARK 999 NAME   {a.name}\n"]
        head += [
            "REMARK 999 TARGET {} {} {}\n".format(
                t[0], t[1:].split("-")[0].rjust(4), t[1:].split("-")[1].rjust(4)
            )
            for t in chain_tags
        ]
        body = []
        for line in pdb_lines:
            tag = remap[f"{line[21]}{int(line[22:26])}"]
            body.append(line[:21] + tag[0] + str(int(tag[1:])).rjust(4) + line[26:])
        fh.write("".join(head + body))

    pdb_by_chain = []
    for tag in chain_tags:
        p = os.path.join(a.outdir, "targets", "pdb", f"{a.key}-chain_{tag[0]}.pdb")
        pdb_by_chain.append(p)
        with open(p, "w") as fh:
            fh.write("".join(l for l in open(pdb_out) if l.startswith("ATOM") and l[21] == tag[0]))

    # --- problem JSON (absolute paths; msa paths are placeholders, unused in template mode) ---
    new_hotspots = [remap[h] for h in hotspots]
    info = {
        "key": a.key,
        "name": a.name,
        "target_pdb_filepath": os.path.abspath(pdb_out),
        "target_fasta_filepath": os.path.abspath(fasta),
        "target_msa_filepath": os.path.join(
            os.path.abspath(a.outdir), "targets", "msa", f"{a.key}.a3m"
        ),
        "target_pdb_filepath_by_chain": [os.path.abspath(p) for p in pdb_by_chain],
        "target_fasta_filepath_by_chain": [os.path.abspath(p) for p in fasta_by_chain],
        "target_msa_filepath_by_chain": [
            os.path.join(os.path.abspath(a.outdir), "targets", "msa", f"{a.key}-chain_{t[0]}.a3m")
            for t in chain_tags
        ],
        "target_chain_and_residues": chain_tags,
        "target_interface_residues": {
            "hotspot": new_hotspots,
            "extended": compute_extended_interface(
                target_pdb_filepath=os.path.abspath(pdb_out),
                target_hotspot_residues=new_hotspots,
                version_num=1,
            ),
        },
        "binder_min_length": a.binder_min,
        "binder_max_length": a.binder_max,
    }
    out = os.path.join(a.outdir, "problems", f"{a.key}.json")
    with open(out, "w") as fh:
        json.dump(info, fh, indent=4)
    print("wrote", out)
    print("chains:", chain_tags)
    print("hotspot ->", new_hotspots)
    print("extended:", len(info["target_interface_residues"]["extended"]), "residues")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--hotspot", required=True, help="comma-separated, e.g. A45,A48,A52")
    p.add_argument("--binder-min", type=int, default=60)
    p.add_argument("--binder-max", type=int, default=120)
    p.add_argument("--outdir", required=True)
    main(p.parse_args())
