# Genie 3

Image: `images/genie3.sif`
Repository: `genie3/`

Genie 3 generates protein backbones and can run generation, sequence design,
and structure evaluation as one configured workflow.

## Through the harness

Genie 3 is a registered tool, so a campaign runs it the same way it runs
anything else: a `tool: genie3` model config, an entry in a workflow index, and
`uv run snakemake`. See `examples/genie3.example.yaml` and its experiment
template. The harness handles everything the manual path below needs handling
for — the three JAX overlays, the writable `lightning_logs`, the ColabFold
parameter cache, and a per-task output root — and refuses to start if any of
them is missing. The manual commands here remain the way to debug one stage.

## Quick start

Copy and edit one of the repository examples:

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
WORK="$BINDER_ROOT/runs/genie3-test"

mkdir -p "$WORK"
cp "$BINDER_ROOT/genie3/examples/binder_design/experiment.yaml" "$WORK/experiment.yaml"
# Edit every input/output path in experiment.yaml to an absolute path.

cd "$WORK"
singularity run --cleanenv --nv "$BINDER_ROOT/images/genie3.sif" \
  run -c "$WORK/experiment.yaml"
```

Other example configurations live under `examples/unconditional/` and
`examples/motif_scaffolding/`.

## Commands

```bash
singularity run --nv genie3.sif --help
singularity run --nv genie3.sif generate -c experiment.yaml
singularity run --nv genie3.sif evaluate -c experiment.yaml
singularity run --nv genie3.sif evaluate --reduce -c experiment.yaml
singularity run --nv genie3.sif status -c experiment.yaml
```

`run` is the all-in-one path. The individual commands are useful for debugging,
resuming, or scheduling stages separately. For multiple local devices, add
`--num-devices N`; distributed shards use `--shard-id ID --num-shards TOTAL`.
Consult `genie3 --help` for the exact placement of these flags in the installed
version.

## The image's JAX is CPU-only — read this before running evaluation

`genie3.sif` ships jax/jaxlib 0.6.2 **without** `jax-cuda12-plugin`, so
`jax.default_backend()` is `'cpu'`. Genie 3's diffusion and ProteinMPNN stages
are PyTorch and use the GPU normally; the ColabFold/AF2 evaluation stage is JAX
and runs on the CPU, where it never finishes. Nothing errors — you just see a
live process and an idle GPU.

Three host overlays fix it without a rebuild. Build once, then use the launcher:

```bash
cd bindocracy/launching_scripts/genie3
./build_jax_overlay.sh          # login node, needs network
sbatch verify_jax_gpu.sbatch    # confirms cpu -> gpu, and that torch still works
sbatch run_genie3.sbatch
```

`run_genie3.sbatch` asserts all three overlays are present before starting,
because falling back to CPU silently is the expensive failure. Two further
requirements it handles: `--log-dir` must be set (the default is under the
read-only `/opt/genie3`), and `/opt/genie3/lightning_logs` must be bind-mounted
to writable space. Details and rationale in
[`known-issues.md` §2.3](../known-issues.md).

The durable fix is `jax[cuda12]` in the image definition; until then, record the
overlay paths beside the SIF checksum, because the image alone no longer
determines the result.

## Weights and outputs

Genie 3, ColabFold/AlphaFold multimer, and ProteinMPNN assets are embedded. A
normal run should not download weights. The image is read-only, so configurations
must direct all outputs and caches to writable absolute paths. Note that weight
completeness and runnability are different questions — see the JAX section above.
