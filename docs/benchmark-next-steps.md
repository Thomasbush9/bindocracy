# Benchmark harness: remaining-work plan

Updated 2026-10-03. Start with [the canonical workflow](agent-benchmark.md).
This plan separates implemented software from deployment work and unverified
scientific/operational assumptions. A local smoke pass is not a production launch
approval.

## Implemented foundation

- Versioned task/environment/workspace/attempt contracts; content inventories and
  task identity independent of agent-specific target paths.
- Non-destructive initialization, ordinary campaign configurations and attempt
  identity through plans, runs, scheduler associations and submissions.
- Runnable optimizer/scoring kits, full bundle locking, explicit CPU smoke and
  resource-managed scoring with frozen metric/run/replicate inputs.
- Cross-process DuckDB coordination; frozen workers do not query mutable input DBs.
- External agent supervision, authenticated local broker, deadline cleanup and
  durable resource reservations with conservative unknown-outcome accounting.
- Portable score-free bundles, organizer cutoff/acceptance, external pinned
  evaluator protocol, complete-roster aggregation and a fixed scripted baseline.
- Shared POSIX directory publication using the existing Lustre-compatible
  reservation/rename convention, rather than requiring Linux `renameat2` support.

Recent corrections include CPU optimizer accelerator visibility, rejecting
GPU-required kits on CPU-only allocations, refusing unlocked scoring overlays,
portable timestamp serialization, specification-relative input paths, unreadable
asset refusal, and capture-output/source overlap refusal. Exact registry-list
assertions were removed rather than repinned to another incidental tool count.

## Evidence and current limits

Observed locally:

- Real CLI environment capture/init/check and an initialized DuckDB workspace.
- A complete multi-file scoring kit executed through a frozen local Snakemake
  campaign, followed by collection and actual database ingestion.
- A runnable CPU optimizer kit, including a normal refusal rather than fabricated
  output; separate static checks and execution reports.
- External agent command, brokered immutable acceptance, organizer-owned bounded
  evaluator process and aggregation with `scientific: false`.
- Publication and overwrite refusal on the actual project filesystem.
- Focused regression tests and Python lint. The final full-suite result is reported
  in the implementation handoff rather than frozen as a rapidly stale test count.

Not qualified by those checks:

- Production privilege separation, hostile-agent containment or network policy.
- Live Slurm authorization, accounting, cancellation/recovery or GPU allocations.
- Installed container/model compatibility or scientific predictive validity.
- A live scheduler execution of `scripts/benchmark_baseline.py`; its CLI was
  exercised, but no cluster workload was submitted.
- Provider billing completeness, infrastructure failover or power-loss durability.

## P0: gates before an enforceable benchmark

### 1. Provision and qualify the trust boundary

**Owner:** organizer/platform operator, with security review.

**Relevant code:** `benchmark/runner.py`, `campaign/control.py`,
`benchmark/submission.py`; [deployment prerequisites](benchmark-runner.md).

**Work:** provision separate organizer/agent identities, protected harness/control
and evaluator installations, process-tree/CPU/memory/network confinement, controlled
scheduler credentials, isolated scientific workers and a broker filesystem view
that cannot expose privileged paths through agent-controlled path races. Review
both host-agent and scheduled-worker execution; protecting only the host agent is
insufficient. Do not treat attestation booleans or read-only mounts as a sandbox.

**Acceptance:** an independently run adversarial deployment suite demonstrates
that the agent and its worker code cannot alter trusted state, read held-out
assets, submit through alternate credentials/routes, escape resource confinement,
or retain running descendants/jobs beyond enforced termination. Preserve the
actual UID/mount/network/scheduler configuration and evidence with the environment.

**Dependency:** site authority and infrastructure configuration. This cannot be
established by another same-UID local subprocess test.

### 2. Qualify the real scheduler/resource path

**Owner:** cluster operator and harness maintainer.

**Relevant code:** `benchmark/budget.py`, `campaign/control.py`,
`campaign/qualify.py`, `workflow/Snakefile`.

**Work:** run separately approved small CPU and GPU canaries on the real deployment.
Exercise submit/status/cancel/resume, lost submission replies, delayed accounting,
controller loss and restart recovery. Check the scheduler's actual GRES/AllocTRES,
comment/tag preservation, CPU/memory limits, requeue policy and single-cluster
assumptions. Include the agent and controller overhead in allocation/compute
budgets; do not silently shrink a scientific workload to pass qualification.

**Acceptance:** reconcile every admitted intent to its exact job identity and
terminal allocated-resource seconds; unknown outcomes remain reserved until
resolved. Demonstrate refusal beyond concurrent/cumulative/time ceilings,
cancellation confined to owned jobs, and preservation of previously ingested
results. Archive scheduler evidence and qualification reports. Only then declare
that site/runtime qualified.

**Dependency:** gate 1, installed assets, authorized queue/account and explicit
approval of each canary digest. No live submissions were performed here.

### 3. Freeze and independently validate the actual evaluation protocol

**Owner:** benchmark/scientific owner, separate from agent implementation.

**Relevant code:** `benchmark/evaluation.py`, `benchmark/submission.py`;
[protocol contract](benchmark-submissions.md).

**Work:** supply the real organizer evaluator and its complete pinned runtime/asset
closure. Predeclare task versions/content IDs, permitted inputs, score range,
direction, failure score, reproducibility policy and held-out access restrictions.
Validate the evaluator independently on organizer-controlled reference cases and
keep its assets/results inaccessible during attempts. The supplied deterministic
CPU example is transport validation, not a biological evaluator.

**Acceptance:** repeatable reports for immutable accepted bundles; rejection of
changed evaluator assets and invalid result identities; explicit scientific,
invalid-input, agent and infrastructure outcomes; documented evidence supporting
interpretation of the chosen measurements. Do not promote software pass/fail to
a claim of experimental success.

**Dependency:** an approved scientific protocol and reference evidence; neither
can be inferred from harness unit tests.

## P1: gates before interpreting agent comparisons

### 4. Freeze the task roster and run the scripted baseline end to end

**Owner:** benchmark operator.

**Work:** choose the complete task × agent × repetition roster before execution.
Freeze identical prompts/policies per task, provider/model settings, allowed tools,
network policy and environment. Predeclare the baseline's fixed workflow/query;
run `scripts/benchmark_baseline.py` under the same runner and budget as agents.
Use fresh workspaces, run names and attempt IDs; stage downstream workloads only
after their actual inputs exist. Preserve every failed/missing attempt.

**Acceptance:** the baseline produces genuine accepted submissions on the qualified
site; complete-roster aggregation includes all slots and pairs comparisons by
task/repetition. Report task-balanced estimates and uncertainty rather than a
best run. Explain differing environments rather than silently treating different
hardware/software installations as equivalent.

**Dependency:** P0 gates. A local single-attempt smoke is not an agent leaderboard.

### 5. Connect real agent/provider usage and transcript adapters

**Owner:** agent integration owner and benchmark operator.

**Work:** wrap each selected noninteractive agent in the existing argv interface.
Emit a complete interaction transcript to captured streams and stable, idempotent
usage events. Record model/provider/version/settings; reconcile reported usage
against trusted provider billing where available. Keep credentials outside the
agent's submission and evidence exports.

**Acceptance:** ordinary completion, provider failure, interrupted streams and
replayed usage events yield correct attempt outcomes; missing token/cost fields
remain unknown. Published cost comparisons state reporting coverage and separate
agent-supplied reports from independently reconciled usage.

**Dependency:** actual agent entrypoints and provider credentials. Do not add a
new embedded agent runtime or guess token/cost totals from output length.

### 6. Exercise crash/storage boundaries on the deployment filesystem

**Owner:** infrastructure and harness maintainers.

**Work:** fault-inject at reservation persistence, scheduler transport, DB ingestion,
submission copying/publication, acceptance-pointer update and runner recovery.
Test disk-full, permission changes, interrupted connections, actual filesystem
locking behavior and restart recovery. Separately assess directory fsync and
storage guarantees for power loss; atomic visibility is not the same claim as
power-loss durability. Define operator handling for abandoned empty publication
reservations without automatically adopting or deleting unrelated directories.

**Acceptance:** no duplicate scheduler work or partial accepted bundle, no deletion
of intervening user files, deterministic refusal/recovery states, and preservation
of valid results. Document which guarantees depend on shared-storage configuration
and which failure modes remain unresolved.

## P2: improvements after correctness and qualification

### 7. Profile the real immutable-input workload

**Owner:** harness maintainer.

Measure source/asset hashing, plan construction, submission size/copy time, DB
lock contention and repeated status/verification on representative deployments.
Capture latency and I/O evidence before choosing optimizations. If repeated hashing
is material, prefer an organizer-owned immutable/content-addressed installation
with explicit verification boundaries; do not replace content identity with an
unchecked mtime cache.

**Acceptance:** benchmark evidence shows the bottleneck and improvement; tampering,
source replacement and exact provenance remain detectable. No second workflow
engine, generic retry framework or speculative caching layer is required.

### 8. Establish evidence retention and upgrade procedures

**Owner:** benchmark operator and maintainer.

Define retention/access policies for interaction logs, provider reports, full
experiment archives, accepted bundles, protocols and aggregate reports. Verify
restoration of a portable submission without its original workspace. Record
schema/tool/runtime changes and use new environment/protocol versions rather than
editing the identity of an already running comparison. Define how organizer
adjudication is recorded if a task selects that policy; retain the original
outcome and explain any exclusion or rerun.

**Acceptance:** an independent operator can reconstruct a reported result from
retained trusted artifacts, identify every failure/exclusion, and restore or audit
an old bundle without executing its untrusted code.

## Recommended execution order

1. Provision isolation and controlled scheduler access; review the threat model.
2. Qualify CPU then explicitly approved GPU scheduler canaries.
3. Freeze and independently validate the organizer evaluator in parallel with
   provider integration, using separate protected assets.
4. Run the fixed scripted baseline and operational fault cases.
5. Freeze the full comparison roster and execute repetitions.
6. Publish complete outcomes, uncertainty, cost coverage and qualification limits.
7. Optimize only measured bottlenecks and establish retention/upgrade operations.

Keep the existing module boundaries: the runner supervises agents, the campaign
engine schedules scientific work, kits provide permitted extensions, acceptance
freezes candidates, and the organizer's evaluator supplies scientific judgments.
