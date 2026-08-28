# Mosaic

Image: `images/mosaic.sif`
Repository and wrapper: `mosaic_setup/mosaic/`

Mosaic optimizes binder sequences against several differentiable structure
backends at once. Unlike most suite images, its weights live outside the SIF.

## Always use the wrapper

The wrapper creates a private container home and binds the external Boltz, AF2,
Protenix, Hugging Face, JAX cache, and output directories correctly.

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
MOSAIC_ROOT="$BINDER_ROOT/mosaic_setup/mosaic"

export MOSAIC_SIF="$BINDER_ROOT/images/mosaic.sif"
export MOSAIC_WEIGHTS="$BINDER_ROOT/mosaic_setup/weights"
export MOSAIC_SCRATCH=/absolute/path/to/writable/mosaic_scratch

mkdir -p "$MOSAIC_SCRATCH"
cd "$MOSAIC_ROOT"
./singularity/mosaic-exec.sh python /opt/mosaic/singularity/selftest.py
```

Use `--show` before a new command to print the resolved Singularity invocation
without running it:

```bash
./singularity/mosaic-exec.sh --show python run_design.py --help
```

Do not bypass the wrapper, and do not omit its private `-H` home setup.

## Direct design example

```bash
./singularity/mosaic-exec.sh python run_design.py \
  --target /absolute/path/to/target.fasta \
  --target-msa /absolute/path/to/target.a3m \
  --binder-length 80 \
  --models boltz2,af2 \
  --batch 2 \
  --seed 0 \
  --soft-steps 100 \
  --sharp-steps 25 \
  --out /absolute/path/to/results
```

For production arrays, prefer the repository's `singularity/campaign.sbatch`.
When passing `MODELS` through `sbatch --export`, use `+` between model names,
for example `MODELS=boltz2+af2`; commas delimit exported variables in SLURM.
Use absolute paths and verify the backends listed in the first task's log.

Mosaic is memory-intensive. H100 80 GB is the practical baseline; H200 is
preferable for the ESM-C 6B model. A precomputed target MSA avoids runtime
queries to the ColabFold service.

For the planned config-to-DuckDB generation workflow, including the output
contract, run manifest, adapter, ingestion boundary, tests, and Snakemake rule
graph, see [Mosaic generation: end-to-end implementation plan](../mosaic-generation-plan.md).
