# Caliby

Image: `images/caliby.sif`
Repository: `caliby/`

Caliby provides inverse folding/sequence design, ensemble design, scoring,
sidechain packing, and Protpardelle ensemble generation.

## Available commands

```bash
singularity run caliby.sif --help
```

The image dispatches `seq-design`, `seq-design-ensemble`,
`generate-ensembles`, `score`, `score-ensemble`, `sidechain-pack`,
`clean-pdbs`, and `python`.

## Sequence-design example

Caliby uses Hydra `key=value` overrides:

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/caliby.sif"
INPUT=/absolute/path/to/input_pdbs
OUTPUT=/absolute/path/to/caliby_results

mkdir -p "$OUTPUT"
singularity run --cleanenv --nv "$SIF" seq-design \
  ckpt_name_or_path=soluble_caliby_v1 \
  input_cfg.pdb_dir="$INPUT" \
  sampling_cfg_overrides.num_seqs_per_pdb=4 \
  out_dir="$OUTPUT"
```

Inspect a subcommand's Hydra options with:

```bash
singularity run "$SIF" seq-design --help
```

Structural ensembles generally benefit from at least 32 members; 8–16 can be
used for a cheaper exploratory pass.

## Weights and limitation

Caliby variants, the sidechain packer, ProteinMPNN, and Protpardelle assets are
embedded. The optional AlphaFold2 self-consistency extra is not installed, so
do not enable `run_self_consistency_eval=true` in this image.
