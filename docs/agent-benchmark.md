# Autonomous-agent benchmark: canonical workflow

This guide joins organizer provisioning, an agent's editable workspace, managed
execution, immutable submission and organizer evaluation. It uses the existing
Snakemake campaign engine; the benchmark runner supervises an external agent,
not a second scientific workflow engine.

## Ownership and prerequisites

| Organizer-owned | Agent-editable |
| --- | --- |
| Installed harness, Python environment, containers and input assets | Run configurations and orchestration/selection scripts |
| Task prompt, target identity, resource policy and attempt records | Scoring-function and optimizer kits allowed by the task |
| Scheduler credentials/broker, accepted archives, evaluator and reports | Workspace database, results and proposed submissions |

A content hash identifies bytes; it does not grant permissions, authenticate a
publisher, isolate code, or establish scientific validity. Production requires
the distinct identities, controlled scheduler, isolated workers and protected
filesystem view in [External benchmark attempts](benchmark-runner.md). An
attestation JSON describes prerequisites; it does not install or prove them.
Use `--development` only for explicitly unisolated local CPU work. It disables
benchmark scheduler admission and is not an adversarial benchmark environment.

Install from the committed lock before capturing the environment:

```sh
uv sync --locked --group dev
uv run bindocracy --json tools list
uv run bindocracy --json config schema --kind environment
uv run bindocracy --json config schema --kind task
```

All CLI examples use `--json` before the command when machine output is wanted.
A zero CLI exit status means the command ran; also inspect the returned domain
status. A failed attempt or inconclusive scientific qualification is not success.

## 1. Organizer: capture the actual installation

Prepare an environment YAML using real installed paths. Its fields are:

- `schema_version: 1`, a `name`, and the installed `harness_revision`;
- `harness_path`: the protected checkout containing `src`, `drivers`, `workflow`
  and `pyproject.toml`; `lock_path`: its `uv.lock`;
- `site`: the ordinary campaign site policy, including controller resources,
  `max_workers` and `max_total_gpus` (zero is valid for CPU-only workloads);
- optional `worker_partition`, separate from the controller partition;
- `tool_configs`: tool name to an installed, valid configuration template;
- optional named `assets` and kit `bindings` for installed dependencies;
- optional `qualification`: organizer evidence to retain, not an asserted pass.

Template paths are resolved relative to the environment document; paths inside
a template are relative to that template. Every allowed tool needs a captured
template. The benchmark accepts implementations from the captured harness, not
external `BINDOCRACY_PLUGINS` overrides. Extend scientific behavior through kits
rather than modifying or replacing the harness during an attempt.

```sh
bindocracy --json workspace environment capture organizer/environment.yaml \
  --output organizer/environment.json
bindocracy --json workspace environment check organizer/environment.json
```

Capture hashes actual harness sources, the lock, declared assets, typed template
inputs and referenced local specification files. Directory inventories reject
symlinks, special files and unreadable subtrees. Outputs must be outside
inventoried source directories. Capture does not download, install, execute or
qualify anything, and it never overwrites an existing manifest. Both commands
report `runtime_qualified: false`; retain separate real qualification evidence.

## 2. Organizer: freeze a task and initialize each attempt

A task YAML contains:

```yaml
schema_version: 1
task_id: example-task
version: "1"
prompt: >-
  Use the supplied workspace and permitted tools. Preserve provenance and
  submit an ordered candidate bundle before the deadline. Do not modify the
  harness or evaluator.
target:
  name: supplied-target
  sequence_fasta: inputs/target.fasta
  chain_id: A
budget:
  mode: allocation
  walltime_seconds: 3600
  max_gpus: 1
  max_cpus: 16
  max_memory_gb: 128
submission:
  min_candidates: 1
  max_candidates: 16
  require_structures: false
  distinct_sequences: true
allowed_tools: [function, optimize]
allowed_extensions: [configuration, orchestration, scoring_function, optimizer, selection]
internet: none
failure_policy: no_compensation
```

The target file and installed templates must actually exist; this is a policy
example, not supplied scientific input. Add required target representations and
choose an installed initial workload appropriate to the task. For cumulative
accounting, use `mode: compute` with an explicit `gpu_seconds` and/or
`cpu_seconds` cap. Budgets include the agent's one CPU/one GiB allocation,
controller and workers concurrently, and reserve each job's full requested
walltime. Requests exceeding remaining time or budget are refused, not shortened.

```sh
bindocracy --json workspace init work/attempt-001 \
  --environment organizer/environment.json --task organizer/task.yaml \
  --attempt-id attempt-001
bindocracy --json workspace check work/attempt-001
```

Initialization creates `campaign.duckdb`, `general.yaml`, `site.yaml`,
`bindings.yaml`, `workflow.yaml`, `prompt.txt`, policy snapshots, usable copies of
installed tool templates, and `configs/`, `kits/`, `plans/`, `runs/`, `logs/` and
`submissions/`. Function-template database paths are rebound to the new workspace.
The database starts empty: initialization does not invent candidates, populate
results or execute the starter workflow. A scoring/optimization stage still needs
a real populated design set; stage dependent workloads rather than launching an
index whose prerequisites have not been produced.

The destination must not exist, even as an empty directory. Publication uses an
exclusive reservation and atomic POSIX rename, including on filesystems without
`RENAME_NOREPLACE`. A crash can leave an empty reservation; it is never silently
adopted. Input/source directories are never made writable or modified.

Task identity includes the exact prompt, policy and target contents, not an
agent-specific target pathname. Environment identity records the installation.
Use a new attempt ID and workspace for every agent/task/repetition. `workspace
check` verifies immutable inputs and the identical prompt, not scientific results.

## 3. Organizer: start the external agent

For a local software check only:

```sh
bindocracy benchmark smoke work/attempt-001 organizer/local-smoke
```

For a real benchmark, use `benchmark run` with the qualified isolation and
scheduler arrangement described in [the runner guide](benchmark-runner.md).
Supply `--agent-name` and a nonnegative `--repeat`, and put the noninteractive
agent argv after `--`. The control directory must be new and outside the workspace.
The agent receives EOF on stdin, workspace/attempt identity and the authenticated
broker endpoint. It should read `prompt.txt`, emit its interaction transcript to
stdout/stderr and report provider usage explicitly; unreported usage remains
unknown rather than zero.

The runner captures raw streams and timestamped events, enforces the deadline,
closes admission, kills its owned process group and cancels exact owned scheduler
jobs. Unknown submissions remain reserved pending authoritative terminal
accounting. Independent OS/scheduler confinement is still required for escaped
processes, alternate credentials and malicious code.

## 4. Agent: inspect, execute, select and extend

Read the supplied prompt and policy first. Use the CLI's actual schemas and
resource-managed campaign lifecycle rather than hand-written scheduler calls:

```sh
bindocracy --json tools describe function
bindocracy --json config schema --kind function
bindocracy --json campaign check work/attempt-001/workflow.yaml \
  --site work/attempt-001/site.yaml
bindocracy --json campaign plan work/attempt-001/workflow.yaml \
  --site work/attempt-001/site.yaml --output-dir work/attempt-001/plans/initial
```

Review the returned scope and exact digest; pass that digest to `campaign submit`.
Inside an attempt, submit/status/resume/cancel use the organizer broker. Frozen
plans retain attempt/task/environment identity and frozen input membership.
Workers never requery live scoring inputs. Collection and cross-process serialized
DuckDB ingestion preserve actual successes, partial results and failures.

Use normal `designset preview/build`, filtering and ranking for explicit selection.
Metric inputs identify exact measuring run IDs, metric names and replicates;
`missing_metrics: error|skip` is explicit. Missing/failed values are not silently
pooled, imputed or converted to valid scores. See [Scoring functions](scoring-functions.md).

Both extension kinds have complete runnable CPU authoring examples:

```sh
bindocracy kit init work/attempt-001/kits/my-score --kind scoring_function
bindocracy kit init work/attempt-001/kits/my-optimizer --kind optimizer
bindocracy --json kit inspect work/attempt-001/kits/my-score
```

`kit resolve` freezes a reviewed digest, dependencies and request into a lock and
ordinary tool config. Follow [the authoring guide](adding-an-optimizer.md) for the
request and binding formats. Scoring function names use lowercase letters,
digits and underscores; set `function.name` separately if the run name contains
hyphens. Kits retain sibling modules/data, not just their entrypoint. Editing the
source yields a different revision and does not alter an existing frozen bundle.
Raw scoring source overlays require a verified kit lock. GPU-required optimizer
kits reject CPU-only allocations.

`kit check` is static. `kit smoke --local-cpu` explicitly runs bounded trusted CPU
code on supplied inputs; within a benchmark it uses the existing reserved agent
allocation. It reports execution, contract validity and refusals separately.
`--plan-only --site ... --database ...` prepares a bounded campaign instead and
reports `executed: false`. It is not a passing smoke. GPU execution always uses
an approved, allocated campaign; no host GPU fallback is authorized.

For scoring, `function plan DATABASE --general ... --config ... --site ...
--output-dir ...` builds the same campaign route. The old synchronous `function
run` command is removed. Ordinary optimizer configs use `tool: optimize` in an
execution index. Never alter a frozen manifest to revise a run: create a new run
and plan instead.

## 5. Agent: propose and commit a submission

```sh
bindocracy submission build work/attempt-001/campaign.duckdb \
  work/attempt-001/submissions/final --workspace work/attempt-001 \
  --design-set work/attempt-001/selected.json
bindocracy submission check work/attempt-001/submissions/final
bindocracy submission accept work/attempt-001/submissions/final
```

Use the real JSON path returned by `designset build`, not a guessed filename.
If structures are required, add an explicit `--structures-from` measuring run.
`build` freezes exact order, target, lineage/configurations and executable kit
provenance. `check` is portable and score-free; it does not predict quality.

Only successful organizer acceptance before cutoff commits a submission. The
broker copies and revalidates the bundle into the protected immutable archive;
a file left in the workspace is not a submission. The latest accepted bundle is
official, never the best-scoring historical one. Detailed format and cutoff rules
are in [Submissions and evaluation](benchmark-submissions.md).

## 6. Organizer: evaluate and compare complete rosters

After the attempt stops, evaluate its accepted archive with an organizer-owned,
content-pinned external protocol. No default biological judge is invented:

```sh
bindocracy benchmark evaluate organizer/attempt-001 \
  --protocol organizer/protocol.json --output organizer/evaluations/attempt-001
bindocracy benchmark aggregate organizer/roster.json \
  --protocol organizer/protocol.json \
  --report organizer/evaluations/attempt-001/report.json \
  --baseline scripted --output organizer/aggregate.json
```

The roster lists **every** attempt record, including missing/failed submissions.
Repeat `--report` for available evaluations. Freeze task versions/content IDs,
protocol, score range, failure score and direction before running comparisons.
Aggregation reports outcomes, task-balanced means, hierarchical bootstrap
intervals and paired task/repetition differences against the named baseline.
A single attempt's degenerate interval is not meaningful uncertainty evidence.

`scripts/benchmark_baseline.py` executes one predeclared index and selection under
the same external runner/broker. Supply a fixed `--index` and optional `--query`,
use `--agent-name scripted`, and keep task/resource policy identical. It submits
real configured workloads; development mode correctly refuses its Slurm path.
It is not a simulated scientific result or an adaptive agent.

Archive broader search/control evidence separately with `submission
archive-evidence`, including evaluation directories explicitly. The accepted
candidate bundle remains a small, fixed scientific handoff rather than an
unbounded experiment-directory upload.

## Verification boundary

CPU tests and local smoke exercise the actual contracts, frozen workflow,
subprocess supervision, publication, database ingestion, acceptance and evaluator
transport. They do not establish model quality, experimental success, GPU/container
compatibility, live Slurm accounting or hostile-agent isolation. Use
`campaign qualify` only after separately approving its exact deployment canary.
See [the remaining-work plan](benchmark-next-steps.md) before production claims.
