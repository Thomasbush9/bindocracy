# Campaign workflow

Snakemake owns worker submission. The connector builds one task's command,
environment, resources, and expected outputs; Snakemake's Slurm executor plugin
submits it and tracks state. The campaign CLI submits the controller and journals
the plugin's worker submissions without introducing another workflow engine.

```text
prepare_run --> generate[task=0..N] --> collect --> ingest --> generation_complete
```

No rule names a tool. A model config declares its own `tool:`, and that tool's
plugin supplies the task count, the command, the resources, and the adapter.

## Files

| File | Role |
|---|---|
| `Snakefile` | The rule graph |
| `campaign.example.yaml` | Template execution index — copy it beside your campaign data |
| `profiles/slurm/` | Executor settings; account and partition come from the configs |

Real indices and real configs live outside the repository, beside the campaign
data, because their absolute paths belong to one machine. See `examples/`.

The workflow config is an index, not a configuration. The general and model YAML
it points at are the durable scientific inputs, and they are what the database
stores.

## Plan, review, and launch

Start from an ordinary execution index and a site policy copied from
[`examples/campaign-site.example.yaml`](../examples/campaign-site.example.yaml).
Worker resources remain in each tool's config; controller resources and campaign
concurrency limits live in the site policy. A GPU-backed controller must be
requested explicitly, and its GPUs count toward `max_total_gpus`.

```bash
# No scheduler submission or database writes. Creates frozen configs and run manifests.
uv run bindocracy campaign plan /path/to/index.yaml \
  --site /path/to/site.yaml --output-dir /path/to/plan
uv run bindocracy campaign show /path/to/plan/plan.json

# Only after reviewing the printed scope, resources, and full plan.
uv run bindocracy campaign submit /path/to/plan/plan.json \
  --approve sha256:PASTE_THE_REVIEWED_DIGEST

uv run bindocracy campaign status /path/to/plan/plan.json
uv run bindocracy campaign cancel /path/to/plan/plan.json
uv run bindocracy campaign resume /path/to/plan/plan.json \
  --approve sha256:PASTE_THE_REVIEWED_DIGEST
```

The plan records exact frozen candidate membership/digest, generator and source-run
counts, prediction multiplicity when the plugin declares it, shard counts, and
controller/worker resource limits. Generation request slots are **not** promised
output counts. Source runs are not automatically treated as selection cohorts;
the original selection query remains in the plan.

Planning copies the common Snakefile verbatim and adds `campaign_plan` to its
frozen index. The common workflow verifies that index, uses the frozen task and
resource mappings, and keeps the usual prepare → generate → collect → ingest
stages. `scripts/check_index.py` also accepts the frozen index.

### Identity and recovery

- **Frozen, not live YAML.** Editing the authored YAML does not change an existing
  plan. Snapshot/manifest tampering or changes to referenced inputs and harness
  Python source prevent execution. Small inputs/scripts use SHA-256; large
  container/checkpoint assets use recorded filesystem identities, not full binary
  hashes. Preserve those assets separately.
- **One plan owns each run directory.** Plans cannot be relocated or rebound.
  A different execution, source revision, or resource policy needs a new plan
  directory and new run names. An incomplete planning attempt retains its
  diagnostic files and any reserved run ownership rather than silently reusing it.
- **Exact approval.** `submit` and `resume` require the full reviewed digest.
  Repeated `submit` returns the existing attempt instead of creating another.
- **Durable jobs.** `control/journal.json` records submission intents, controller
  and worker IDs, run/task associations, and recovery archives. Attempt-specific
  controller scripts and Slurm logs live under `control/<attempt>/`.
- **Isolated cancellation.** Cancellation closes the submission gate first, then
  cancels only allocations confirmed by their exact scheduler tags. Delayed
  accounting can leave an unknown/cancelling state: repeat `status`/`cancel`.
  An unknown submission outcome blocks resume; do not delete the journal to retry.
- **Preserved results.** Resume requires all prior jobs to be terminal. Valid
  completed tasks keep their outputs; interrupted shards/logs move into
  `.campaign-recovery/<attempt>/` before restarting cleanly. Recorded terminal
  partial/failed scientific outcomes are not silently retried or rewritten.
  Damaged already-ingested runs are refused.
- **Environment hygiene.** The controller removes inherited Slurm submission
  settings and GPU visibility before launching Snakemake. Workers keep their own
  scheduler allocation environment.

Lifecycle management currently targets one Slurm cluster with accessible
`squeue` and `sacct` comments. Federated submission IDs and job arrays are not
accepted. The campaign CLI enforces the approval/lifecycle interface; direct
Snakemake invocation below remains a lower-level interface.

## Direct Snakemake invocation

```bash
# 1. Dry run: expect one generate job per task, one ingestion per run.
uv run snakemake --dry-run --configfile /path/to/your-index.yaml

# 2. For real, through Slurm.
uv run snakemake --configfile /path/to/your-index.yaml --profile workflow/profiles/slurm
```

An index names its executions, so re-running a configuration means adding
another entry with another name rather than copying the config file:

```yaml
runs:
  - name: mosaic-run-12
    config: /path/to/configs/mosaic/hallucinate.yaml
```

## What each run leaves behind

```text
runs/<name>/
|-- run.json          run identity, task plan, archived-input hash, and the
|                     whole config pair as JSON — a run is replayable from
|                     this alone, with no config file on the share
|-- campaign-binding.json  owning plan and execution identity, when campaign-planned
|-- provenance/       the exact driver that ran, archived and executed
|-- tasks/0000/       designs.jsonl and status.json, written by the driver
|-- logs/
|-- collected.json    the validated staging bundle
`-- ingested.json     proof the bundle reached DuckDB
```

Two properties are deliberate:

- **Restart-safe.** Re-running reuses the existing `run.json`, so the `run_id`
  and the archived inputs never change. Re-ingesting an identical bundle is a
  no-op; a changed bundle for a known run is an error, never a second copy.
- **Partial work survives.** The driver appends and fsyncs each design, and
  reports its own outcome in `status.json`. A task killed by the walltime still
  contributes its completed designs, and the run is recorded as `partial`.

- **A launch reads the manifest, never the live YAML.** Editing a config after
  planning cannot change what an existing run executes; it re-runs
  `prepare_run`, which either accepts the manifest or refuses it loudly.

To execute the same configuration a second time, add another `runs:` entry with
another name: a `run_id` identifies an execution, not a configuration.
