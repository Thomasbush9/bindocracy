# FreeBindCraft

Image: `images/freebindcraft.sif`
Repository: `FreeBindCraft/`

This image runs BindCraft with the FreeBindCraft PyRosetta bypass. Relaxation
and replacement metrics use OpenMM/FASPR/sc-rs/FreeSASA.

## Quick start

Create a target JSON following the examples in `FreeBindCraft/settings_target/`.
Its `design_path` must be writable and `starting_pdb` must be visible in the
container. Absolute `/n/...` paths are recommended.

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="$BINDER_ROOT/images/freebindcraft.sif"
TARGET=/absolute/path/to/my_target.json

ulimit -n 65536
singularity exec --cleanenv --nv "$SIF" bindcraft-gpu-check
singularity exec --cleanenv --nv "$SIF" bindcraft \
  --settings "$TARGET" \
  --no-pyrosetta
```

`singularity run --nv "$SIF" ...` invokes the same BindCraft wrapper. The
default filters and advanced settings are embedded at
`/opt/freebindcraft/settings_filters/` and
`/opt/freebindcraft/settings_advanced/`; provide host files to override them.

## Important limitations

- `--nv` is mandatory.
- PyRosetta is not installed. Keep `--no-pyrosetta`; Rosetta-only metrics use
  placeholders, so review filters accordingly.
- The inherited hard file-descriptor limit must be at least 65,536.
- If OpenMM initialization fails, retry with
  `SINGULARITYENV_OPENMM_PLATFORM_ORDER=CUDA,OpenCL`.
- Relative paths in the target JSON resolve from the launch directory.

AF2/ColabDesign and MPNN assets are embedded; normal jobs are offline.
