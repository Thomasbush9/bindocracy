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

## Collected structure identity

The harness associates each ranked CIF with its design using the numeric
`final_rank` and exact native `file_name`, independent of rank zero-padding
(`rank01_…` and `rank001_…`). Conflicting files for the same rank and filename
fail collection rather than silently selecting a pose.

Older collections can omit ranks 1–99 when native filenames use three digits.
The generated designs and structures are still present: repair collection and
re-ingest through an explicit backed-up recovery, then verify exact binder and
target sequences. Do not rerun generation or bypass an existing ingestion-digest
conflict to repair these associations.
