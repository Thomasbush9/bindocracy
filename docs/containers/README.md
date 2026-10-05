# Container catalog

All image paths below are relative to:

```text
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
```

| Guide | Entry style | GPU | Weights |
|---|---|---|---|
| [Genie 3](genie3.md) | `singularity run ... <genie3 args>` | Yes | Embedded |
| [FreeBindCraft](freebindcraft.md) | `singularity exec ... bindcraft` | Yes | Embedded |
| [BindCraft2 configuration](../../examples/bindcraft2.example.yaml) | `singularity exec ... bindcraft design` | Yes | Embedded |
| [Caliby](caliby.md) | `singularity run ... <subcommand>` | Usually | Embedded for supported modes |
| [BoltzGen](boltzgen.md) | `singularity run ... <boltzgen args>` | Yes | Embedded |
| [Protein-Hunter](protein-hunter.md) | `singularity run ... boltz|chai` | Yes | Embedded for supported modes |
| [Proteina-Complexa](proteina-complexa.md) | `singularity run ... <complexa args>` | Yes | Embedded |
| [PXDesign](pxdesign.md) | `singularity run ... <subcommand>` | Yes | Embedded |
| [SwitchCraft](switchcraft.md) | explicit Python command | Yes | Embedded |
| [Mosaic](mosaic.md) | required host wrapper | Yes | External |
| [RFdiffusion](rfdiffusion.md) | direct RFdiffusion runscript | Yes | External |
| [Chai-1](chai1.md) | `singularity run ... fold` | Yes | Embedded |
| [AlphaFold 3](af3.md) | `run_alphafold.py` | Yes | **External — licence, not convenience** |
| [OpenFold3](openfold3.md) | `run_openfold predict` | Yes | External |

Read [running containers](../running-containers.md) before launching a campaign,
especially the login-node policy and reproducibility checklist.

## BindCraft2 settings compatibility

Offline harness preflight validates campaign and nested loss/filter **keys**
against the qualified BC2 1.0.1 consumer registry. The packaged
[`settings_schema.json`](../../src/bindocracy/tools/bindcraft2/settings_schema.json)
records the upstream revision, source hashes, and extraction APIs. It rejects
unknown keys such as `_comment`; it does not strip authored settings or reject
all underscore-prefixed names by convention.

This is not complete value validation, preset expansion/existence checking, or
GPU/runtime qualification. Refresh and qualify the snapshot when supporting a
different upstream consumer. The opt-in configuration-only checks use the actual
image on CPU, without `--nv` or launching design workers:

```bash
BINDCRAFT2_CONTAINER=/absolute/path/to/bindcraft2.sif uv run pytest -q \
  tests/test_bindcraft2_launch.py tests/test_bindcraft2_preflight.py \
  -k 'upstream_config_consumer or qualified_upstream_consumer'
```
