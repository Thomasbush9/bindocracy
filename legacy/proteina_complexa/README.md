# Proteina-Complexa runtime target contract

The image is target-agnostic. It owns Proteina-Complexa's reusable Hydra config
tree, dependencies, model weights, and executables. The launcher adds exactly
two read-only runtime binds:

- the requested target PDB at `/mnt/bindocracy_target.pdb`;
- one small target registry at the in-image `targets_dict.yaml` location.

The default registry is [`target_registry.yaml`](target_registry.yaml), which
describes the DIO3 smoke target. A different target requires no image rebuild:

```bash
PROTEINA_TARGET_NAME=my_target \
PROTEINA_TARGET_PDB=/path/to/my_target.pdb \
PROTEINA_TARGET_REGISTRY=/path/to/my_target_registry.yaml \
sbatch run_proteina_complexa.sbatch
```

The registry key must match `PROTEINA_TARGET_NAME`, and its `target_path` must
be `/mnt/bindocracy_target.pdb`. Binder length can be changed independently
with `PROTEINA_BINDER_MIN` and `PROTEINA_BINDER_MAX`.

This keeps tool-owned defaults in the SIF and target/campaign state in the
harness, without restoring the former 80-file config bind.
