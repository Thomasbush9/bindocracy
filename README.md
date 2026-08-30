# Bindocracy

Operational documentation for the protein-design containers installed under:

```text
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
```

The images are in `images/`. Run GPU workloads through SLURM on a compute node;
use the login node only to edit inputs, inspect help, and submit jobs.

## Start here

- **[What the alpha run says about building the harness](docs/harness-design.md)**
  — the actual conclusion: nine requirements the library has to absorb, each one
  traced to something that went wrong
- **[Known issues and traps](docs/known-issues.md)** — read before configuring
  anything; several tools produce confident, well-formed output when misconfigured
- **[Campaign database and output-adapter contract](docs/database.md)** — the
  single-target schema, empty-database command, and normalized parser interface
- **[Generation workflow](workflow/README.md)** — the Snakemake rule graph that
  takes any registered tool's config to designs in DuckDB, and how to run it
- **[Adding a tool](docs/adding-a-tool.md)** — the five things a tool supplies,
  and the one line that registers it
- **[drivers/](drivers/)** — code the harness runs inside a container
- **[legacy/](legacy/)** — the hand-written `sbatch` launchers from the alpha
  benchmark, kept as reference and not used by the harness
- [Running containers on the cluster](docs/running-containers.md)
- [Weights and offline-readiness report](docs/weights.md)
- [Container catalog](docs/containers/README.md)

## Container guides

Four of these run through the harness today — Mosaic, BoltzGen, Genie 3 and
PXDesign. The rest are documented images you drive by hand; adding one is
[one package and one line](docs/adding-a-tool.md).

| Tool | Image | Primary use | Harness |
|---|---|---|---|
| [Genie 3](docs/containers/genie3.md) | `genie3.sif` | Backbone generation and end-to-end design/evaluation | `tool: genie3` |
| [FreeBindCraft](docs/containers/freebindcraft.md) | `freebindcraft.sif` | BindCraft-style binder design without PyRosetta | — |
| [Caliby](docs/containers/caliby.md) | `caliby.sif` | Sequence design, scoring, packing, and ensembles | — |
| [BoltzGen](docs/containers/boltzgen.md) | `boltzgen.sif` | Structure-conditioned protein/peptide design | `tool: boltzgen` |
| [Protein-Hunter](docs/containers/protein-hunter.md) | `protein_hunter.sif` | Boltz- and Chai-based binder design pipelines | — |
| [Proteina-Complexa](docs/containers/proteina-complexa.md) | `proteina_complexa.sif` | Local multi-stage binder, ligand, AME, and motif workflows | — |
| [PXDesign](docs/containers/pxdesign.md) | `pxdesign.sif` | Diffusion binder design with structure filters | `tool: pxdesign` |
| [SwitchCraft](docs/containers/switchcraft.md) | `switchcraft.sif` | Multistate and allosteric protein design | — |
| [Mosaic](docs/containers/mosaic.md) | `mosaic.sif` | Differentiable multi-model binder optimization | `tool: mosaic` |
| [RFdiffusion](docs/containers/rfdiffusion.md) | `rfd.sif` | Diffusion-based protein backbone generation | — (see §2.2) |

## Scope and conventions

These runbooks document the images currently in `images/`, not arbitrary future
upstream releases. Commands use absolute paths deliberately: they survive
changes in the submit directory and map cleanly into Singularity. Replace the
example input and output paths with your own writable locations.

For each image, the quickest source-of-truth checks are:

```bash
IMAGE_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
singularity inspect --helpfile "$IMAGE_ROOT/genie3.sif"
singularity inspect --runscript "$IMAGE_ROOT/genie3.sif"
```

The upstream repositories remain next to `images/` and contain deeper,
tool-specific scientific documentation.
