# OpenFold3 (upstream)

Image: `/n/holylfs06/.../singularity_dev/images/openfold3.sif` (9 GB)
Definition: not ours — ProtForge's build of `openfoldconsortium/openfold3:stable`
v0.4, Apache-2.0, from `aqlaboratory/openfold-3`.

A registered tool: `tool: of3_upstream`, see
`examples/of3_upstream.example.yaml`.

## Why this exists separately from `of3`

mosaic's `jopenfold3` port does not fold correctly, and the GFP control shows
it plainly:

| | GFP pLDDT | RMSD to consensus |
|---|---|---|
| upstream (`of3_upstream`) | **88.7** | **3.9 Å** |
| mosaic's port (`of3`) | 38.5 | 24.0 Å |

3.9 Å places it among af2 (3.9), af3 (4.0) and chai1 (3.7). The two differ
**from each other by 24.6 Å** and disagree on complexes too (ipTM 0.374 vs
0.248). mosaic's OF3 scored 0.597 on Nipah-G — below the 0.642 a sequence-only
control reaches — and that number was measuring the port, not the model.

`of3` is now deprecated: refused for new runs, still loadable from archived
manifests so historical runs stay relaunchable. Metrics are stored as
`of3_upstream_*`, never `of3_*`; with a 24 Å disagreement one column holding
both would be worse than useless.

## The filename decides whether the alignment exists

`parse_msas_direct` keeps only alignment files whose **basename** is one of
thirteen it knows — `colabfold_main`, `uniref90_hits`, `mgnify_hits` and so on
— and silently `continue`s past everything else
(`core/data/io/sequence/msa.py:274`). A correctly formatted a3m named anything
else is dropped, the MSA dict comes back empty, and the run dies frames later
on `sorted(...)[0]` with an `IndexError` naming nothing relevant.

The driver stages the campaign alignment as `colabfold_main.a3m`. The accepted
set is in `tools/of3_upstream/config.py` so preflight can explain itself.

This is the third variant of the trap in this campaign: Boltz keys the MSA off
the a3m's first header, Chai off a sha256 of the sequence, OpenFold3 off the
filename. All three fail quietly.

## Weights and the MSA server

The image ships no weights; `runtime.checkpoint` points at the original
`of3-p2-155k.pt` (2.2 GB). `use_msa_server: true` is refused by the config —
letting OpenFold3 search its own alignment would fold the target against a
different MSA from every other scorer.
