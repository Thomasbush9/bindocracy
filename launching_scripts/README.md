# launching_scripts

Reference launchers for every binder-design tool exercised in the alpha
benchmark.

Each subdirectory is one tool and contains the configs it consumes plus one
`run_<tool>.sbatch`. The scripts are meant to be read as much as run: the
comments record *why* a flag is set, and in several cases the flag exists only
to avoid a silent failure. Copy a directory and edit it for a new target.

```text
common/                  shared environment, resource collection, GPU tracing
00_target/               fold DIO3-cut once; everything else consumes the result
mosaic/                  hallucination (sequence-space, no target structure)
boltzgen/                structure-conditioned diffusion + inverse folding
freebindcraft/           AF2 hallucination + MPNN, PyRosetta-free
genie3/                  backbone diffusion + MPNN + AF2, one workflow
proteina_complexa/       flow matching + AF2 reward search
protein_hunter/          Boltz-2 co-fold / MPNN iterative refinement
pxdesign/                diffusion + AF2-IG and Protenix filters
rfdiffusion/             backbone diffusion (backbones only, no sequences)
caliby/                  inverse folding: backbones -> binder sequences
```

## Running one

```bash
cd launching_scripts/<tool>
sbatch run_<tool>.sbatch
```

Every script sources `common/env.sh` for paths, target identity, campaign size
and cluster identity. Nothing else is assumed about your shell. Outputs always
land in `outputs/<tool>/`; nothing is written outside `binder_design/`.

## Order of operations

`00_target/predict_target.sbatch` must run first. It folds the DIO3-cut sequence
with Mosaic/Boltz-2 and publishes `outputs/_shared/dio3_cut.{cif,pdb}`. Seven of
the nine tools consume that structure — folding it once is what makes the
comparison about the design method rather than about whichever folder each tool
would have called internally.

Two tools need a further one-off preparation step, both idempotent and both
runnable on the login node:

| Tool | Prepare | Produces |
|---|---|---|
| PXDesign | `pxdesign/make_pxdesign_msa.py` | `outputs/_shared/msa/dio3_cut/0/{non_pairing,pairing}.a3m` |
| Genie 3 | `genie3/prepare_problemset.py` | `outputs/_shared/genie3_dataset/` |

Protein-Hunter seeds its own MSA cache from inside its sbatch.

`caliby/` runs last: it is an inverse-folding step and consumes the backbones
RFdiffusion produces.

## How the target is treated

- **Sequence + MSA** — Mosaic, Protein-Hunter. No structure needed.
- **Structure** — BoltzGen, FreeBindCraft, Genie 3, Proteina-Complexa,
  PXDesign, RFdiffusion, Caliby.

The target is 201 residues, chain A, numbered 1..201. It is **not** 198; several
configs encode the length (`A1-201` for Proteina-Complexa, the contig
`A1-201/0 80-80` for RFdiffusion) and would misbehave silently if it changed.

## Hotspots

Every tool that permits it runs **without** hotspots, matching the lab's
previous DIO3 campaigns (`hotspots: ""` in both
`mosaic_setup/mosaic/pipelines/presets/*_dio3.yaml`) and letting each generator
choose its own surface. Genie 3 is the sole exception — it has no hotspot-free
binder mode — and uses `A110,A112,A131`, the solvent-exposed rim of the
catalytic pocket. The archived alpha benchmark records how those were chosen and
why Genie 3's designs are therefore not strictly comparable.

## Conventions worth keeping

- **`--cleanenv` everywhere**, with `SINGULARITYENV_*` used to pass back only
  what is needed. Host conda and `PYTHONPATH` otherwise contaminate these
  images, and the host's RHEL `SSL_CERT_FILE` breaks any image that imports
  `httpx`. `common/env.sh` forwards `CUDA_VISIBLE_DEVICES` so `--cleanenv` does
  not cost you SLURM's GPU assignment.
- **`TMPDIR` on node-local disk**, never Lustre. Triton cannot clean up its
  kernel-compile temp directories on `/n/netscratch` and the job dies with
  `Errno 39`. `common/env.sh` provides `node_tmp`.
- **One job per tool, 40 designs, no arrays.** An array would hide per-design
  cost behind the scheduler, and per-design cost is the thing being measured.
  Each script's header notes what a production run should do instead.
- **Absolute paths.** They survive a change of submit directory and map cleanly
  into the container.

## Resource measurement

`common/collect_stats.sh <tool> <jobid>` writes `outputs/<tool>/resources/`
with `jobstats`, `sacct` and a one-line TSV summary. `common/gpu_trace.sh` is
sourced by each sbatch and samples `nvidia-smi` every 15 s, because `jobstats`
reports a single memory maximum and its utilisation sampler is too coarse for
short jobs — it reported "GPU utilization 0% <-- GPU was not used" for a job
whose GPU memory peaked at 76 GB. Read
[`docs/harness-design.md`](../docs/harness-design.md) before trusting any
GPU-memory figure: the JAX-based tools preallocate.
