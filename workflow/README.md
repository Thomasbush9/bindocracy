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

## Verification and opt-in deployment qualification

GitHub Actions runs the committed `uv.lock` environment on Python 3.12 with
`uv sync --locked --group dev`, `uv run --locked ruff check src tests scripts`,
and the full CPU pytest suite. It needs no Slurm credentials, containers,
checkpoints, or GPUs. Tests use injected transports where appropriate;
passing CI is **not** evidence that a real cluster deployment has been qualified.

`campaign qualify` observes the existing submit/status/cancel/resume lifecycle and
Slurm transport. It accepts an already-frozen plan, never builds one implicitly,
and requires a fresh execution: no earlier journal attempts, task contents, task
logs, collected/ingested markers, completion markers, or recovery archives.
Use a dedicated canary run root and database, new run names, and a new plan
directory. Never select a production execution with outcomes you intend to keep
running, and do not operate the same plan from another terminal during qualification.

### Preparing a canary

1. Select a **real** installed connector and its normal validated configuration.
   Set exactly one run, one task, at most two requested/generated records, and,
   where declared, at most two input designs/predictions. Use ordinary
   `campaign plan` and `campaign show` as above; inspect the complete frozen
   execution closure and digest. Do not edit `plan.json` or manifests afterward.
2. The default policy is CPU-only: zero controller and worker GPUs, controller
   at most 2 CPUs/4 GiB/10 minutes, worker at most 2 CPUs/4096 MiB/5 minutes, and
   site `max_workers: 1`. The current bundled scientific connectors require GPUs;
   a CPU canary therefore needs a genuine CPU-capable external connector using the
   existing `BINDOCRACY_PLUGINS` interface and a resource model accepting zero GPUs.
   Its resolved resources must satisfy normal campaign planning, including
   `gres: gpu:0`. Confirm that representation is supported by the local executor
   and Slurm. The test-only toy connector is not a production canary.
3. For an existing supported GPU connector instead, **separately opt in** with
   `--allow-gpu`. The policy still permits only one run/task and one concurrent
   worker. Each allocation may request at most one GPU and the site's total GPU
   ceiling must be at most two. Controller limits are 2 CPUs/8 GiB/15 minutes;
   worker limits are 8 CPUs/64 GiB/10 minutes. This option never removes the exact
   digest approval requirement and does not authorize any unreviewed launch.
   A small requested workload is not a guarantee that a model will finish within
   these limits; a timeout must remain an inconclusive result.

No path above installs a fake scientific tool or silently reduces an existing
plan's resources. A configuration that cannot meet the canary limits is refused.
The current site schema requires a positive `max_total_gpus` even for a CPU-only
plan; it is a ceiling, not a GPU allocation.

### Inspecting, approving, and interpreting results

```bash
# No scheduler commands or submissions; returns approval_required/inconclusive.
uv run bindocracy --json campaign qualify /path/to/canary/plan.json

# Only after separately approving this exact plan and its CPU resource scope.
uv run bindocracy --json campaign qualify /path/to/canary/plan.json \
  --approve sha256:PASTE_THE_REVIEWED_DIGEST --timeout 900 --poll-interval 5

# Alternative for a separately reviewed GPU canary; never implicit.
uv run bindocracy --json campaign qualify /path/to/gpu-canary/plan.json \
  --allow-gpu --approve sha256:PASTE_THE_REVIEWED_DIGEST

# Use ANOTHER fresh plan to exercise cancellation and resume explicitly.
uv run bindocracy --json campaign qualify /path/to/resume-canary/plan.json \
  --mode cancel-resume --approve sha256:PASTE_THE_REVIEWED_DIGEST
```

The command needs `sbatch`, `squeue`, `sacct`, and `scancel`, a valid account and
partition, and ordinary shared-filesystem/controller/worker prerequisites.
Both `squeue` comments (`%k`) and `sacct` allocation comments (`Comment%256`) must
preserve the full campaign tag. Slurm accounting must expose terminal allocation
states in a timely manner; an accepted submission or an empty accounting response
is never completion evidence. This is a single-cluster, non-array qualification.

Reports are retained under `PLAN_DIR/control/qualification-*.json`, alongside the
normal journal and logs. They include lifecycle observations, exact owned
controller/worker tags as observed separately in queue/accounting responses,
terminal accounting states, and cleanup results. `passed` requires completed
campaign artifacts, successful status and an existing output for every task,
and accounted `COMPLETED` controller **and** worker jobs. Failed task outcomes
cannot qualify merely because their Slurm allocations completed.
`failed` means an observed terminal campaign or task failure; `inconclusive`
covers interruption, scheduler/transport errors, missing accounting and timeouts.
Injected evidence is explicitly labelled and always has `live_qualified: false`.
A passed infrastructure check does not establish scientific validity of outputs.
An inconclusive/failed report exits nonzero, including the unapproved inspection.

`cancel-resume` waits for an active tagged worker, cancels through the ordinary
campaign gate, requires accounted worker cancellation and terminal prior jobs,
then resumes the same frozen DAG and verifies completion. A canary that finishes
before cancellation can be observed is inconclusive, not a cancellation pass.
Polling is bounded (default 900 seconds, maximum 1800; intervals 1–30 seconds);
each scheduler subprocess is bounded to 15 seconds and the remaining deadline.
Interruption, errors, and timeouts trigger at most 60 seconds of best-effort,
tag-scoped cancellation for jobs submitted by this invocation. There is no
generic `scancel`. Unknown cleanup remains explicit: inspect the journal and
repeat ordinary `campaign status`/`campaign cancel` when accounting recovers.
SIGINT/SIGTERM are handled; SIGKILL, node loss, and inaccessible filesystems cannot
guarantee cleanup or report persistence.

Scientific outputs and logs are never deleted by qualification. Ordinary resume
retains completed outcomes and archives interrupted task evidence. Until run on
a real approved cluster plan, worker submission from the Slurm executor,
queue/comment visibility, accounting latency, completion/ingestion, cancellation,
resume, and environment propagation remain **unverified integration paths**.
Even a successful CPU canary does not qualify GPU visibility, GPU libraries,
containers, checkpoints, federated Slurm, or real scientific model correctness.

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
