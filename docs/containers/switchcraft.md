# SwitchCraft

Image: `images/switchcraft.sif`
Repository: `switchcraft/`

SwitchCraft designs multistate proteins, including positive/negative allostery,
induced binding, ligand discrimination, and motif switching.

## Entrypoint

Use `singularity run`: the image runscript changes to `/opt/switchcraft` and
launches `switchcraft.py`. The `%help` text embedded in this version shows an
incomplete `singularity exec ... --config` example; `exec` requires an explicit
program, whereas `run` uses the runscript.

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/switchcraft.sif"
WORK=/absolute/path/to/switchcraft_run

mkdir -p "$WORK"
cp "$BINDER_ROOT/switchcraft/tasks/pos_allostery.yaml" "$WORK/design.yaml"
# Edit motifs, ligands, states, and output paths in the copied YAML.

cd "$WORK"
singularity run --cleanenv --nv "$SIF" \
  --config "$WORK/design.yaml"
```

Equivalent explicit form for debugging:

```bash
singularity exec --cleanenv --nv "$SIF" \
  python /opt/switchcraft/switchcraft.py --config "$WORK/design.yaml"
```

Template tasks in the repository are:

- `neg_allostery.yaml`
- `pos_allostery.yaml`
- `induced_binding.yaml`
- `ligand_discrimination.yaml`
- `motif_switching.yaml`

Each YAML completely specifies states, ligands, and losses. Copy a template
instead of editing the repository version. Boltz and LigandMPNN weights are
embedded, but output paths must be writable.
