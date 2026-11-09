# Bindocracy

Operational documentation for the protein-design containers installed under:

```text
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
```

The images are in `images/`. Run GPU workloads through SLURM on a compute node;
use the login node only to edit inputs, inspect help, and submit jobs.

## Start here

- **[Autonomous-agent benchmark](docs/agent-benchmark.md)** — canonical organizer
  and agent workflow, from environment capture and workspace initialization to
  budgeted attempts, immutable submissions and held-out evaluation
- **[Remaining benchmark work](docs/benchmark-next-steps.md)** — prioritized
  deployment and verification gates, with acceptance criteria
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
- **[Campaign Watch](#campaign-watch)** — lightweight local TUI over SSH for Mosaic
  and generator campaigns; per-arm/tool progress, conditional ETA and sampled GPU utilization
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

## Campaign Watch

The read-only dashboard lives in [`scripts/campaign_watch/`](scripts/campaign_watch/).
The local client needs Python 3.12, OpenSSH and Textual, **not** Bindocracy's model
or workflow dependencies. A separate CPU collector on the cluster publishes a
small JSON snapshot; the client fetches only that file over SSH. Neither the client
nor the collector submits, cancels or modifies scientific jobs.

### Install and run locally

From a local clone of this repository:

```bash
python3.12 -m venv ~/.venvs/bindocracy-watch
~/.venvs/bindocracy-watch/bin/python -m pip install \
  -r scripts/campaign_watch/requirements.txt

HOST=your-existing-ssh-alias
ssh -F scripts/campaign_watch/ssh_config "$HOST" true
~/.venvs/bindocracy-watch/bin/python scripts/campaign_watch/tui.py \
  --host "$HOST" --ssh-config scripts/campaign_watch/ssh_config \
  --snapshot /n/netscratch/kempner_bsabatini_lab/Lab/tbush/campaign-watch-prototype/snapshot.json
```

The snapshot path above is the existing Chai1/confhard campaign monitor; use another
absolute snapshot path for another deployment. Authenticate interactively once
with `ssh`; subsequent fetches reuse the connection without password prompts.
The supplied SSH configuration includes `~/.ssh/config` and preserves host-key
checking. Omit `--host` and `--ssh-config` to inspect a local snapshot instead.

Update with `git pull --ff-only`, rerun the requirements-install command, then
restart the TUI. No tarball or `scp` is needed.

Controls: `j/k` or arrows move through rows or scroll details; `l` or Enter drills
from campaigns into stages, then arms, generators and details for generation.
Mosaic stages open details directly. `h` or Escape backs up one level. Tab switches
panes, `r` refreshes, and `q` quits. Selection survives snapshot refreshes.
`87% [8/95]` means mean GPU utilization across **8 sampled tasks
out of 95 running tasks**, not all 95 GPUs. Details show range, memory, coverage,
and observation age. These are Slurm accounting readings, not instantaneous device
polls or statistically representative campaign-wide estimates. Missing, partial,
stale and disconnected data remain distinguishable from a measured zero.
Native numeric job steps take precedence over CPU-only `.batch` wrappers. Each
allocation contributes at most one sample; multiple native steps use an explicitly
unweighted mean, not a device-weighted estimate. Missing or invalid native readings
stay unknown rather than falling back to a misleading wrapper zero.

### Collector deployment on the cluster

**Do not launch a second writer for an existing snapshot.** The current monitor
does not need restarting just to use the Git-managed client. For a new deployment,
keep registrations, snapshots and logs outside Git, for example under
`/n/netscratch/kempner_bsabatini_lab/Lab/tbush/campaign-watch`.
Copy `scripts/campaign_watch/campaigns.example.json` there as `campaigns.json` and
set the real campaign root and planned scoring counts.

From the cluster's repository checkout, with that state directory already created:

```bash
STATE=/n/netscratch/kempner_bsabatini_lab/Lab/tbush/campaign-watch
export CAMPAIGN_WATCH_PYTHON="$PWD/.venv/bin/python"
sbatch --output="$STATE/collector-%j.log" scripts/campaign_watch/collect.sbatch \
  "$PWD/scripts/campaign_watch" \
  --config "$STATE/campaigns.json" --output "$STATE/snapshot.json" --interval 60
```

The launcher requests one CPU and 2 GiB on `kempner_priority`, for up to 72 hours,
with no automatic resubmission. Renew deliberately for longer campaigns; never
overlap writers. Override site-specific Slurm options at submission.
Pass `--once` after the collector arguments for a one-snapshot qualification,
using a **separate output path** from any running monitor. Python defaults to
`python3` if `CAMPAIGN_WATCH_PYTHON` is unset; the collector itself uses only the
standard library plus the trusted campaign validator.

### Registering generation campaigns

Registration freezes the monitoring inventory; it does **not** render execution
bundles, submit jobs, or change scientific outputs. DIO3 uses four production
bundles (126 arm/generator components) plus a separate qualification bundle
(14 components). The generation-only registration excludes downstream stages:

```bash
STATE=/n/netscratch/kempner_bsabatini_lab/Lab/tbush/campaign-watch-generators
ROOT=/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-generators-20261105
PROPOSAL=/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-generators-proposal-20261105
mkdir -p "$STATE"
python3 scripts/campaign_watch/register_generation.py \
  --allocation "$PROPOSAL/allocation.json" --root "$ROOT" \
  --manifest "$STATE/live-manifest.json" --config "$STATE/live-campaigns.json" \
  --name "DIO3 generators" --generation-only \
  --qualification-bundle bundles/qualification \
  --bundle wave1_catalytic=bundles/wave1_catalytic \
  --bundle wave2_sites=bundles/wave2_sites \
  --bundle wave3_major=bundles/wave3_major \
  --bundle wave4_minor=bundles/wave4_minor
python3 scripts/campaign_watch/collector.py \
  --config "$STATE/live-campaigns.json" --output "$STATE/snapshot.json" --offline
```

`--offline` is a bounded, metadata-only preview: no scheduler commands and no
Slurm allocation required. It does not claim historical journal states are live.
For continuous scheduler/GPU observations, launch the CPU collector above using
this state directory and `--config "$STATE/live-campaigns.json"`; point the local
TUI at its `snapshot.json`. Never run an offline writer against a live snapshot.

The generation adapter reads each bundle's `control/status.json`, frozen plans,
run manifests, worker journals, and task status/collection summaries. Individual
worker job IDs are tracked, not just arrays. Controller allocations are excluded
from worker/GPU totals. It reports preparing/generating/generated/ingested phases,
reported production, task coverage, scheduler failures and partial output by arm
and generator. Repeated metadata references are deduplicated by run/task identity;
this is **not** cross-run sequence deduplication or coordinate validation.
Optional atomic `control/stages/<tier>.json` records expose the CPU stage job and
its success dependency. Queued dependencies and failed stage jobs remain visible,
without counting CPU controllers as generator jobs or GPUs. Worker activity and
already-produced output remain visible even if a stage controller fails.

Generator counts are tool-reported, never labeled unique validated designs.
Missing counts and coordinate readiness remain `?`, and overshoot is retained.
BindCraft2/FreeBindCraft native requests are accepted-design stopping targets,
not raw-candidate quotas; those two quantities are not compared for completion.
Current generator collections publish compact `collected-summary.json` files.
The monitor checks their run identity and source-file metadata before using them.
Snakemake can touch the canonical output after that summary is published. A stale
or missing sidecar therefore triggers a bounded read of the canonical file's
leading run metadata (up to 1 MiB), not its design/metric inventories. Legacy
layouts retain the small-file fallback. Reads are cached by file identity and
timestamps; an unreadable or changing source leaves counts unknown.
Canonical run totals are counted once even when per-task counts are unavailable;
the monitor does not invent a per-task breakdown. Counts update when canonical
collection publishes them, not continuously within native model steps;
scheduler/GPU observations refresh independently.
Per-component ETA requires at least five successful same-arm/tool elapsed samples
and running workers, and is suppressed on failures, ambiguous retries or unknown
scheduler evidence. It estimates remaining **planned-task execution** at current
concurrency, not an output quota; queue wait is excluded. Tier-wide ETA remains
unknown. GPU utilization is campaign-scoped, not attributed to individual tools.

The independent-generation recovery overlay is optional and read-only. With no
`control/recovery/index.json`, the original view is unchanged. Index schema v1
contains `campaign_root` (the registered absolute root) and at most 256 `units`,
each with `id`, `stage_id`, `arm`, `generator`, `target`, and absolute `status_path`.
IDs must match an original independent unit (`stage_id--arm--generator`), targets
must equal its original nominal target, and status paths must be exactly
`<root>/control/recovery/components/<id>.json`, without escaping-root symlinks.
Each status (at most 64 KiB) repeats schema version and identity/target and supplies
`state` (`planned`, `qualifying`, `running`, `complete`, `stopped`, `blocked`, or `failed`),
nonnegative integer `retained_raw_candidates` and `completed_batches`,
`active_job_ids` (at most 1,024 unique individual single-GPU allocation ID strings,
not controller or array IDs), nullable text `error`, timezone-aware ISO8601
`updated_at`, and nonempty text `count_basis`.
Optional nullable fields `accounting_warning` and `controller_warning` carry
reconciliation failures; `controller_job_id` is an individual CPU allocation ID
string, excluded from worker/GPU totals. `reconciled_at` is a timezone-aware
ISO8601 timestamp of the last successful arm reconciliation (null means never).
Older publishers may omit these fields.

The publisher's retained count includes original output plus terminal immutable
recovery batches; it **replaces**, rather than adds to, original counts. Recovery
phases, generated/target, active allocations, batch count, count basis and errors
appear through existing snapshot fields/details. Raw overshoot remains visible,
but recovery progress is capped at 100%. Old failures do not override recovery;
new allocation IDs join the existing scheduler and GPU accounting queries.
Live original GPU jobs remain visible while recovery waits for their arm; queued
canaries are shown as queued rather than running.
Counts are labeled **last-reconciled**, not freshly scanned output. Accounting or
controller warnings, a missing/exited controller, or reconciliation older than
15 minutes produce a degraded row even when the monitor itself is fresh. The TUI
shows `DEGRADED / MONITOR FRESH`, with the actual error in details. Running and
queued GPU totals require current scheduler evidence; published active IDs alone
are not evidence of activity. Terminal IDs awaiting reconciliation remain visible,
and a nonterminal active controller with no worker jobs is labeled idle rather
than generating. Other units retain their independent running/completed states;
an affected unit makes the tier need attention without hiding live GPU totals.
Durably complete units do not become stalled merely because their controller
exits or their reconciliation timestamp ages. Published warnings still prevent a
healthy completion display.
Recovery ETA is unavailable: old task timing is not a calibration for new batches.
Active statuses expire after 15 minutes without a heartbeat; timestamps over
60 seconds in the future are rejected. Valid `complete` statuses are durable after
the supervisor exits, but require count >= target, no active allocations, and no
error. The publisher must use terminal original runs and immutable finished
batches for that completion claim. An intentional `stopped` status also remains
durable, with reconciled counts, no active allocations/error, and a nonempty
`stop_reason`. It retains partial output without claiming the quota was met.
Update the local TUI before loading snapshots containing this new state.
Missing, malformed, stale or contradictory status never implies completion.
No recovery directory scans, native candidate counts or job commands are performed.

### Scope and safety

Snapshot schema v2 supports generation and Mosaic; the TUI still accepts v1 Mosaic
snapshots. Malformed campaign records retain their last-good data without hiding
healthy campaigns. Frozen Mosaic batch-two campaigns use `campaign.json`,
`tasks.json`, `parents.json`, `launch.json`, arm recipes, and the campaign's trusted
`scripts/collect.py` validator resolved through `runtime_root`. Only register
campaigns you trust: that validator is imported Python, not sandboxed.
Generation geometry, parent selection, optimization, ESMFold2, Protenix and
analysis rows are explicit **not-submitted plans**, not live downstream adapters.
Counts are units within a stage, never summed across generation and downstream stages.

Live scheduler collection runs only in a CPU allocation, with incremental reads,
bounded file/command sizes, atomic snapshots, and a 15-second scheduler deadline. GPU
sampling queries at most eight spread-out running tasks per refresh; it never
SSHes into GPU nodes. The client fetches at most 1 MiB every 30 seconds by default.
Live state, logs, SSH credentials and debug captures are not part of the repository.

## DIO3 confhard2 optimization preparation

`scripts/prepare_dio3_optimization.py` is the campaign-specific transition from the
stopped DIO3 generation archive to a reviewable optimization cohort. Run preparation,
validation, collection and launch planning in a CPU allocation on
`kempner_priority`; `kemp_cpu_id59` permits at most 4 CPUs and 16 GiB per node.
Preparation does not submit GPU work.

To decouple optimization from full artifact preservation, first run
`scripts/prepare_dio3_sampling_db.py --smoke`, then the same script with `--workers 4`,
using the generation checkout's Python environment. This builds the separate
`archive/optimization_candidates.duckdb` CampaignStore from the complete stopped
population. Publication requires reconciliation of all 126 strata against the
final census (751,080 raw production candidates, including labelled canaries).
Native sequences avoid unnecessary coordinate parsing; structure-only candidates
still use the archival parser. Conflicting/invalid native sequence observations
are not replaced from coordinates; unrecoverable sequences remain invalid.
This catalog is not an artifact backup.

```bash
python scripts/prepare_dio3_optimization.py prepare \
  --database /n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-generators-20261105/archive/optimization_candidates.duckdb
python scripts/prepare_dio3_optimization.py validate
python scripts/prepare_dio3_optimization.py run --task-id 0 --describe
python scripts/prepare_dio3_optimization.py launch
```

The output root is
`/n/netscratch/kempner_bsabatini_lab/Lab/tbush/dio3-confhard2-20261109`.
Preparation refuses to overwrite a frozen or partial cohort. It samples the
archive's `generation_candidates` view: 80 parents per generator–arm stratum,
7 generators × 18 arms = 10,080 parents and 20,160 children. Sampling is
deterministic, family-first and score-blind. Known families take precedence;
unknown lineage uses explicitly labelled sequence proxies, never invented native
families. Parent sequences are distinct within each stratum. Cross-stratum
duplicates are reported rather than silently removed.

Only production, non-canary, canonical 75-aa sequences with matching target and
hotspots are eligible. Cysteines are retained. There is no intrinsic-score, Chai1,
distance or native acceptance filter. Any stratum shortage stops preparation and
writes `reports/feasibility.json`; variants cannot silently replace missing families.

Every arm uses the full 201-aa target and its generation-specific hotspots,
validated against target residue identities and converted from 1-based generation
indices to 0-based optimizer indices. Approved generation source revisions retain
the original manifest anchor and their checked revision receipts. The frozen
capsule includes the DB snapshot, parents, seeds, child IDs, hotspot mapping,
recipes and checksummed prior runtime. Shared external model assets remain required.
The recipe is confhard contact weight 2.0, three 90-soft/30-sharp cycles,
hard sharpening, two replicates, sharpening noise 0.1 and `no_cys=false`.

Qualification uses 18 full-schedule parents, one per arm, covering all seven
generators. The default cap is eight GPUs; this approved launch uses `--gpu-cap 18`.
These are production parents: successful children are reused, not extra trajectories.
Production gates only its own arm.
All earlier submitted arrays must be positively terminal before another launch;
absence from `squeue` alone is insufficient. Production admits all eligible parents
in balanced arrays, normally capped at 48 + 48 GPUs on
`kempner_h100_priority`. Parent failures do not cancel unrelated parents.
Retries retain failed attempts and reuse validated successes.

`launch` prints a plan by default. Actual submission additionally requires
`--submit --approval FILE`, with explicit user authorization recorded against the
campaign root, provenance hash and permitted modes. Preparation creates no approval.
Use `--root .../dio3-confhard2-20261109/qualification` to inspect the qualification
plan. No scoring jobs or scoring dependencies are created.

The campaign's `control/launch_approved.py` records the explicit chat authorization
after freezing the cohort. It submits qualification, then an `afterany` CPU
controller that admits passing arms to production and queues optimization-only
collection. Neither full archival ingestion nor packed LF backup gates GPU launch.
The existing legacy Mosaic watcher adapter does not yet support this campaign's
18-arm, mapped-array launch protocol; do not register it as a legacy campaign.

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
`general`, `site`, `index`, `filter`, `rank`, `query`, `function`, `kit`, `kit-lock`,
`environment`, `task`, `workspace` and `attempt`.
`config check --no-preflight` checks schemas only and does not claim resolved
configuration identities. `designset preview` uses the same selection, exact
filter-run resolution and limit-before-deduplication semantics as `build`,
without writing a design set; an empty preview is valid.

- **Shared kits:** [`kit init/inspect/check/publish/list/fetch/resolve/smoke`](docs/adding-an-optimizer.md)
  author runnable optimizer/scoring bundles, distribute pinned revisions and
  resolve reviewed code into site-specific locks. Explicit bounded smoke executes
  code; static checks and fetching do not. Neither digest approval nor read-only
  mounts provide publisher authentication or a sandbox.
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

Use `bindocracy tools list` for the installed harness connectors. This table is
a container catalog, not an exhaustive registry. A new core connector follows
[the tool contract](docs/adding-a-tool.md); benchmark agents instead extend
permitted scientific behavior through configurations and locked kits.

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
