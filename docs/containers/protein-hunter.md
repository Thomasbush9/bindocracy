# Protein-Hunter

Image: `images/protein_hunter.sif`
Repository: `Protein-Hunter/`

Protein-Hunter exposes separate Boltz and Chai binder-design pipelines.

## Discover the installed arguments

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/protein_hunter.sif"

singularity run "$SIF" --help
singularity run --nv "$SIF" boltz --help
singularity run --nv "$SIF" chai --help
```

## Boltz example

Run from the intended writable output directory:

```bash
WORK=/absolute/path/to/protein_hunter_boltz
mkdir -p "$WORK"
cd "$WORK"

singularity run --cleanenv --nv "$SIF" boltz \
  --name my_target \
  --mode binder \
  --protein_seqs "TARGET_SEQUENCE"
```

The repository README contains fuller Boltz and Chai examples for trial counts,
length bounds, MSA mode, confidence thresholds, and plotting. Confirm options
against the image's `--help`, because these pipelines have many version-specific
flags.

## Included and excluded features

Boltz2, Chai/Chai-ESM, and ProteinMPNN/LigandMPNN assets are embedded.
AlphaFold3 validation and PyRosetta post-processing are intentionally absent.
Do not pass `--use_alphafold3_validation`; the runscript rejects it early.
