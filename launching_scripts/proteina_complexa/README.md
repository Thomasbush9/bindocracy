# Proteina-Complexa configs

`configs/` is a **verbatim copy of `/opt/proteina-complexa/configs/` from
`proteina_complexa.sif`**, with exactly one edit.

## Why it is vendored

Proteina-Complexa resolves its target from `configs/targets/targets_dict.yaml`,
and the copy inside the image is read-only. `run_step` invokes Hydra with
`--config-path <parent of the pipeline yaml>`, so a host-side copy composes
correctly and `complexa target list` sees our dictionary too. There is no
override that adds a target without a writable config tree.

Regenerate it with:

```bash
singularity exec /n/holylfs06/.../images/proteina_complexa.sif \
    cp -r /opt/proteina-complexa/configs <here>/configs
chmod -R u+w <here>/configs
```

## The one edit

`configs/targets/targets_dict.yaml` gains a `dio3_cut` entry at the end. Three
things about it are deliberate:

- **`target_path` is absolute.** It takes priority over the
  `$DATA_PATH/target_data/<source>/<filename>.pdb` fallback, and that fallback
  cannot be steered here anyway: `SINGULARITYENV_COMPLEXA_DATA_PATH` never
  reaches `DATA_PATH`, because Singularity injects user environment *after*
  `%environment` runs. An absolute path sidesteps the whole problem.
- **`target_input: "A1-201"` is an explicit range.** The docs say a bare chain
  id `A` works; `AtomSelectionStack.from_contig` uses the regex
  `([A-Za-z]+)(\d+)-(\d+)` and raises `ValueError` on it.
- **`hotspot_residues: []`** means unconstrained placement, matching the rest of
  the benchmark.

Everything else in the tree is upstream and unmodified. Note that the shipped
benchmark targets (`02_PDL1` and friends) keep **relative** paths like
`./assets/target_data/...` and therefore only resolve when the working directory
is `/opt/proteina-complexa`; that does not affect `dio3_cut`.
