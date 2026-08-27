# PXDesign

Image: `images/pxdesign.sif`
Repository: `PXDesign/`

PXDesign couples diffusion-based binder generation to Protenix and AF2-IG
confidence filters.

## Input YAML

A minimal target description has this shape:

```yaml
target:
  file: /absolute/path/to/target.cif
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]
      msa: /absolute/path/to/msa/0
binder_length: 80
```

Validate the exact schema against the examples and `check-input` command in the
installed repository.

## GPU check and production run

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/pxdesign.sif"
WORK=/absolute/path/to/pxdesign_run

mkdir -p "$WORK/cache" "$WORK/out"
export SINGULARITYENV_PXDESIGN_CACHE="$WORK/cache"

singularity run --cleanenv --nv "$SIF" check
singularity run --cleanenv --nv "$SIF" pipeline \
  --preset extended \
  -i "$WORK/target.yaml" \
  -o "$WORK/out" \
  --N_sample 100 \
  --dtype bf16 \
  --use_fast_ln True \
  --use_deepspeed_evo_attention True
```

Use A100, H100, or H200 where possible. On V100, use `--dtype fp32` and
`--use_deepspeed_evo_attention False`. This image does not support Blackwell
GPUs because its pinned JAX and custom kernels predate that architecture.

All inference weights are embedded. `prepare-msa` contacts an external MSA
service, so precompute the MSA where network access is allowed and then run the
GPU pipeline offline.
