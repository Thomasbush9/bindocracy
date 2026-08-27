# Proteina-Complexa

Image: `images/proteina_complexa.sif`
Repository: `Proteina-Complexa/`

Proteina-Complexa packages local pipelines for binder design, ligand-binder
design, atomic motif embedding (AME), and motif scaffolding.

## Inspect targets and commands

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/proteina_complexa.sif"

singularity run "$SIF" --help
singularity run "$SIF" target list
singularity run "$SIF" validate --help
singularity run "$SIF" status --help
```

## Local binder-design example

```bash
WORK=/absolute/path/to/complexa_run
mkdir -p "$WORK"
cd "$WORK"

singularity run --cleanenv --nv "$SIF" design \
  /opt/proteina-complexa/configs/search_binder_local_pipeline.yaml \
  ++run_name=my_binder \
  ++generation.task_name=02_PDL1
```

Other embedded local configurations include:

- `search_ligand_binder_local_pipeline.yaml`
- `search_ame_local_pipeline.yaml`
- `search_motif_local_pipeline.yaml`

The `++...` values are Hydra overrides. Start with a known target and a small
run, then inspect `status` before scaling.

## Custom target data

If the tool expects targets at `/data`, bind a host target directory there:

```bash
export SINGULARITYENV_COMPLEXA_DATA_PATH=/data

singularity run --cleanenv --nv \
  --bind /absolute/path/to/targets:/data \
  "$SIF" design /opt/proteina-complexa/configs/search_binder_local_pipeline.yaml \
  ++run_name=my_run ++generation.task_name=my_target
```

Complexa, AF2, ESM2, RF3, ProteinMPNN, LigandMPNN, Foldseek, and supporting
pipeline assets are embedded. Outputs and runtime caches remain writable host
data; the large SIF itself is immutable and intended to run offline.
