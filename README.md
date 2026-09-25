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
- **[Campaign workflow](workflow/README.md)** — frozen execution plans, explicit
  approval, controller/worker lifecycle, and the shared Snakemake rule graph
  that takes registered tools to DuckDB
- **[Stored ranking and cohort selection](docs/scoring-stage.md#11-stored-ranking-and-cohort-selection)**
  — reproducible global or per-generator head/tail selections and frozen handoffs
- **[Adding a tool](docs/adding-a-tool.md)** — the five things a tool supplies,
  and the one line that registers it
- **[Scoring functions](docs/scoring-functions.md)** — conditions (things that
  fold) against functions (things computed from the result), and how to add a
  metric with a script instead of a plugin
- **[Custom optimization](docs/custom-optimization.md)** — improve the designs a
  filter chose, with a script the campaign does not have to know about; the
  whole generate → score → filter → optimize loop
- **[drivers/](drivers/)** — code the harness runs inside a container
- **[legacy/](legacy/)** — the hand-written `sbatch` launchers from the alpha
  benchmark, kept as reference and not used by the harness
- [Running containers on the cluster](docs/running-containers.md)
- [Weights and offline-readiness report](docs/weights.md)
- [Container catalog](docs/containers/README.md)

## Machine-readable CLI and readiness

Place `--json` **before** the command. Successful command execution writes one
`{"schema_version":1,"ok":true,"result":...}` envelope to stdout; argument,
validation and execution errors write
`{"schema_version":1,"ok":false,"error":{"code":...,"message":...,"details":[...]}}`
to stderr and exit nonzero. Validation details include field locations.
Inspect both the process exit code and the result's domain status: a completed
readiness inspection or qualification can report `unknown`, `blocked` or
`inconclusive` and still exit nonzero. `--help` is human-only; combining it with
`--json` is rejected with a structured argument error.

```bash
uv run bindocracy --json tools list
uv run bindocracy --json tools describe mosaic
uv run bindocracy --json config schema --tool mosaic
uv run bindocracy --json config schema --kind site
uv run bindocracy --json config check \
  --general /path/to/general.yaml --model /path/to/mosaic.yaml
uv run bindocracy --json designset preview /path/to/campaign.duckdb \
  --tool mosaic --distinct-sequences --limit 50
uv run bindocracy --json filter metrics /path/to/campaign.duckdb \
  --run EXACT_MEASURING_RUN_ID
```

Discovery reflects registered plugins and their actual Pydantic schemas, not a
claim that their containers or weights are installed. Shared schema kinds are
`general`, `site`, `index`, `filter`, `rank`, `query` and `function`.
`config check --no-preflight` checks schemas only and does not claim resolved
configuration identities. `designset preview` uses the same selection, exact
filter-run resolution and limit-before-deduplication semantics as `build`,
without writing a design set; an empty preview is valid.

- **Readiness:** [`campaign check INDEX --site SITE`](docs/env-setup.md) checks
  declared inputs, resources, dependencies and frozen-plan consistency without
  creating executions. Scheduler readiness stays unknown unless bounded,
  read-only probes are explicitly requested with `--probe-site`.
- **Results:** [`campaign report`, `campaign export`, and `campaign audit`](docs/navigating-the-database.md)
  provide run-scoped summaries, long-form CSV/Parquet and read-only integrity
  checks. Metrics and replicate evidence are never silently pooled across runs.
- **Inputs:** [`target materialize`](docs/chai1.md) creates validated Chai,
  PXDesign, PDB and mmCIF representations from existing inputs, with hashes and
  provenance. It performs no search, folding or GPU work.
- **Deployment:** [CI and `campaign qualify`](workflow/README.md#verification-and-opt-in-deployment-qualification)
  separate offline verification from explicit, digest-approved, tightly bounded
  live-cluster qualification. Passing tests does not qualify a deployment.

## Container guides

Seven of these run through the harness today — Mosaic, BoltzGen, Genie 3,
PXDesign, Protein-Hunter, Proteina-Complexa and FreeBindCraft. The rest are
documented images you drive by hand; adding one is
[one package and one line](docs/adding-a-tool.md).

| Tool | Image | Primary use | Harness |
|---|---|---|---|
| [Genie 3](docs/containers/genie3.md) | `genie3.sif` | Backbone generation and end-to-end design/evaluation | `tool: genie3` |
| [FreeBindCraft](docs/containers/freebindcraft.md) | `freebindcraft.sif` | BindCraft-style binder design without PyRosetta | `tool: freebindcraft` |
| [Caliby](docs/containers/caliby.md) | `caliby.sif` | Sequence design, scoring, packing, and ensembles | — |
| [BoltzGen](docs/containers/boltzgen.md) | `boltzgen.sif` | Structure-conditioned protein/peptide design | `tool: boltzgen` |
| [Protein-Hunter](docs/containers/protein-hunter.md) | `protein_hunter.sif` | Boltz- and Chai-based binder design pipelines | `tool: protein_hunter` |
| [Proteina-Complexa](docs/containers/proteina-complexa.md) | `proteina_complexa.sif` | Local multi-stage binder, ligand, AME, and motif workflows | `tool: proteina_complexa` (binder only) |
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
