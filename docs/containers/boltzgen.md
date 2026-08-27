# BoltzGen

Image: `images/boltzgen.sif`
Repository: `boltzgen/`

BoltzGen performs structure-conditioned biomolecular design. The image runs the
`boltzgen` executable directly.

## Quick start

Prepare a design YAML using the upstream examples, then run:

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/boltzgen.sif"
WORK=/absolute/path/to/boltzgen_run

mkdir -p "$WORK/results"
cd "$WORK"
singularity run --cleanenv --nv "$SIF" run "$WORK/design.yaml" \
  --output "$WORK/results" \
  --protocol protein-anything \
  --num_designs 100 \
  --budget 10
```

Equivalent explicit form:

```bash
singularity exec --cleanenv --nv "$SIF" boltzgen run design.yaml \
  --output results --protocol protein-anything
```

Useful protocols include `protein-anything`, `peptide-anything`,
`protein-small_molecule`, `nanobody-anything`, and `antibody-anything`. Check
the installed interface with `singularity run "$SIF" --help` and use a small
`--num_designs` smoke test before a large campaign.

Standard BoltzGen checkpoints and data are embedded, and offline mode is the
default. Runtime/JIT caches should still point to writable scratch.
