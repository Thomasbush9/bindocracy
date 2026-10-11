# External benchmark attempts

The organizer runs `bindocracy benchmark`; the autonomous agent is an external argv process, not code loaded into the organizer's Python interpreter. The agent's permitted scientific edits are configurations, orchestration, selection, scoring functions and optimizer kits. The harness, acceptance service and evaluator are organizer-controlled.

## Local CPU development (not isolation)

After `workspace init` has created a workspace, run a genuine CPU command:

```sh
bindocracy benchmark smoke /absolute/workspace /absolute/organizer/smoke-attempt
bindocracy benchmark run /absolute/workspace /absolute/organizer/attempt \
  --development --agent-name my-agent --repeat 0 -- /absolute/agent executable-argument
bindocracy benchmark status /absolute/organizer/attempt
```

The attempt directory must not exist and must be outside the workspace. `smoke` launches Python, captures both output streams, records the attempt identity and exits. It neither submits Slurm jobs nor qualifies a tool/container/GPU runtime. Development records explicitly say `unisolated/development`; scheduler admission is disabled. A same-UID process can bypass or modify any same-UID filesystem policy. Development results are not adversarially enforceable benchmark results.

`run` uses argv without shell interpolation, a new process session/group, UTC timestamps plus a monotonic cutoff, separate raw `stdout.log`/`stderr.log`, and timestamped `events.jsonl`. A process group is terminated even when the main command succeeds with background children. Escaping a process group is prevented only by externally provisioned process-tree confinement, not by Python subprocess management.

## Production prerequisites

Production refuses to launch without a distinct `--agent-uid`, `--broker-socket`, and organizer-authored `--isolation` JSON containing **all** of:

```json
{
  "protected_control": true,
  "protected_harness": true,
  "isolated_agent": true,
  "isolated_scheduler_jobs": true,
  "controlled_scheduler": true,
  "network_policy": true
}
```

These are explicit prerequisites/attestations, **not a sandbox installation or proof**. Provision and independently qualify them before making these assertions:

- The organizer identity alone may mutate attempt records, accepted submissions, the budget ledger and trusted code. Protect the harness, Python environment, executable resolution, environment/task inputs and broker configuration. The agent must not possess that identity's credentials.
- Launch the agent under its distinct identity inside a process-tree/resource confinement mechanism (for example, a separately administered container/cgroup/VM). The local agent allocation is one CPU, no GPUs and one GiB memory, included in the concurrent ceilings. CPU smoke executes within that existing allocation, not an additional unaccounted host allocation. Enforce internet policy and prohibit escape/new sessions from escaping confinement.
- A dedicated controlled scheduler principal/account and scheduler access policy must deny the agent arbitrary `sbatch`, `srun`, alternate clusters, alternate submission hosts/credentials and requeue paths. A shell variable or PATH shim cannot do this. Only the authenticated organizer broker and trusted controller may authorize new allocations.
- Trusted controller and worker bootstrap run the immutable harness. Before any agent-authored scientific code executes, externally enforced job isolation must deny that code access to organizer writes/credentials. Merely scheduling jobs as the organizer UID is unsafe. Provision the broker's filesystem view so agent-controlled symlinks/path races cannot resolve into organizer state; the existing scientific harness is not a privileged hostile-filesystem sandbox.
- The broker socket lives in an externally provisioned directory traversable by the agent, not inside a mode-0700 control directory. Linux `SO_PEERCRED` authenticates the configured agent UID. Socket permissions are not the sole identity check. Its parent directory must not let another identity replace the socket.

When the runner is root and no launcher prefix is supplied, it drops supplementary groups and launches with the configured agent UID/default GID. Otherwise supply `--launcher launcher.json`, a JSON argv prefix invoking an organizer-provisioned privilege-separation launcher (for example an exact restricted sudo rule). That launcher must preserve the benchmark environment variables and actually enter the configured agent identity/confinement. No unrestricted sudo rule should be granted to the agent. The organizer controls the launch command and all launcher files.

The production scheduler arrangement must support the trusted bootstrap's accounting access independently of the confined scientific process. This project does not provision site-specific Slurm identities, SPANK/job confinement, ACLs, containers or credential firewalls. Do not enable production based solely on successfully creating directories or supplying the attestation JSON.

## Broker and submission flow

The runner exposes a Unix-domain JSON-lines broker and supplies `BINDOCRACY_BROKER`, `BINDOCRACY_WORKSPACE`, `BINDOCRACY_ATTEMPT_ID`, `BINDOCRACY_ATTEMPT_DIR`, and `BINDOCRACY_LOCAL_CPU_ALLOCATION=agent` to the agent. The variables propagate identity; they are not security credentials. Frozen plans retain attempt identity and organizer ledger location. Agent campaign submit/resume/status/cancel calls automatically use the broker. Trusted controller/worker transport hooks use the durable ledger directly. Benchmark campaign journals and launch scripts live below the organizer attempt directory, not in the agent-owned plan directory.

`submission accept` uses the broker when launched in an attempt. The broker accepts only a source below the assigned workspace; acceptance copies and revalidates all bytes into the organizer's accepted archive and checks the running state and cutoff before committing. The agent cannot choose the organizer directory or acceptance clock. The latest accepted snapshot, not a subsequently mutated workspace, is evaluated separately after the attempt.

The organizer copies task/environment/workspace manifests and target FASTA outside the workspace before launch. These records are read-only files; their actual immutability depends on the permission/identity prerequisites above, not `chmod` against their owner.

## Accounting and admission

`BudgetLedger` uses process-safe `flock`, fsync and atomic replacement. Reservations are persisted **before** scheduler transport, with exact attempt+plan+controller-attempt+intent tags. Controller CPU/GPU/memory allocations count, as do all workers and the agent's one-CPU local allocation. Automatic scheduler requeue is disabled for admitted jobs. Resource requests exceeding the remaining cutoff are rejected, not silently shortened; configure controller/worker walltimes accordingly.

- Allocation mode enforces deadline and concurrent GPU/CPU/memory ceilings. Optional cumulative limits still apply.
- Compute mode additionally requires at least one explicit GPU-second/CPU-second ceiling. Charge is allocated resource count times elapsed allocation seconds, not utilization or device FLOPs.
- Admission reserves the full requested walltime. It cannot oversubscribe available cumulative budget or active allocation ceilings. Identical reservation IDs are idempotent; conflicting retries fail.
- A lost `sbatch` reply remains `UNKNOWN` and fully reserved. Neither a missing queue row nor a cancellation request authorizes a refund or resubmission. Exact tag reconciliation records recovered job IDs. Only terminal `sacct` allocation accounting settles GPU/CPU seconds and frees unused reservation capacity.
- Scheduler accounting above the reserved allocation closes the gate and records actual consumption; it is never silently capped to manufacture compliance. Site-enforced allocation limits are required to prevent scheduler overruns. Cancellation/termination latency can incur additional real usage; it remains recorded, not hidden.

The local allocation reserves one CPU and one GiB for the remaining attempt duration, so compute budgets must include that overhead. Local subprocess wall elapsed time is allocation charge; model inference cost is a separate explicit record. Unknown remote outcomes remain reserved through runner recovery.

## Recovery, cancellation and evidence

```sh
bindocracy benchmark cancel /absolute/organizer/attempt
bindocracy benchmark recover /absolute/organizer/attempt
bindocracy benchmark status /absolute/organizer/attempt --reconcile
```

`cancel` closes the durable gate first, marks terminal attempt state, signals the recorded process group, then cancels only scheduler jobs matching exact registered tags and identities. Scheduler transport has finite timeouts; each cleanup pass has a bounded work window and reports unresolved accounting rather than waiting indefinitely or releasing unknown reservations. Run `status --reconcile` after accounting becomes available. Ordinary `status` is local/read-only and does not contact Slurm.

`recover` refuses an active runner lock. After a runner crash it closes the attempt and cleans up rather than restarting a possibly already-executed command. PID/start-time/boot identity prevents signalling a reused PID. If the leader identity is gone, descendant status is explicitly unknown: external process-tree confinement is the authority for escaped/orphaned descendants. A machine reboot invalidates the old monotonic ledger gate.

Startup holds the runner lock before publishing attempt state. A trusted child waits on an organizer pipe before executing agent argv: its PID/start identity is persisted before release, and organizer death before release closes the pipe without executing the agent. The agent receives EOF on stdin; interactive agent frontends should be wrapped in their noninteractive/API mode.

Provider usage is ingested explicitly:

```json
{"event_id":"request-123","provider":"provider-name","model":"model-id","input_tokens":1200,"output_tokens":300,"cost_usd":0.004}
```

```sh
bindocracy benchmark usage /absolute/organizer/attempt /path/provider-usage.json
```

`event_id`, `provider` and `model` are required. Token/cost fields are optional, nonnegative and never inferred. Duplicate identical records are idempotent. Totals cover **reported records only**; omitted values remain null/unknown, and unreported calls are unknown. Agent-supplied usage is a report, not trusted provider billing evidence. No fake zero-token or zero-cost inference is produced. `--agent-name` and optional nonnegative `--repeat` support reproducible comparison labels.
