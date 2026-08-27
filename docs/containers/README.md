# Container catalog

All image paths below are relative to:

```text
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
```

| Guide | Entry style | GPU | Weights |
|---|---|---|---|
| [Genie 3](genie3.md) | `singularity run ... <genie3 args>` | Yes | Embedded |
| [FreeBindCraft](freebindcraft.md) | `singularity exec ... bindcraft` | Yes | Embedded |
| [Caliby](caliby.md) | `singularity run ... <subcommand>` | Usually | Embedded for supported modes |
| [BoltzGen](boltzgen.md) | `singularity run ... <boltzgen args>` | Yes | Embedded |
| [Protein-Hunter](protein-hunter.md) | `singularity run ... boltz|chai` | Yes | Embedded for supported modes |
| [Proteina-Complexa](proteina-complexa.md) | `singularity run ... <complexa args>` | Yes | Embedded |
| [PXDesign](pxdesign.md) | `singularity run ... <subcommand>` | Yes | Embedded |
| [SwitchCraft](switchcraft.md) | explicit Python command | Yes | Embedded |
| [Mosaic](mosaic.md) | required host wrapper | Yes | External |
| [RFdiffusion](rfdiffusion.md) | direct RFdiffusion runscript | Yes | External |

Read [running containers](../running-containers.md) before launching a campaign,
especially the login-node policy and reproducibility checklist.
