# Benchmark submissions and held-out evaluation

A submission is a **fixed, ordered candidate set**, not the whole search history.
A successful `submission check` establishes file integrity, portable provenance,
and policy compliance. It does **not** establish binding, affinity, selectivity,
experimental success, or the validity of any computational scientific method.
Likewise, kit checks and the CPU example below are software qualification only.

## Trust and deployment

Provision separate organizer and agent operating-system identities. The organizer
must exclusively control the runner installation, attempt directory, accepted
archives, evaluator executable/scripts/configuration/model assets, protocol JSON,
and reports. Do not give the agent a writable copy of any of those paths that the
organizer subsequently imports or executes. The agent's permitted editable surface
is configuration, orchestration, selection, scoring functions and optimizer kits,
not the scientific harness or held-out grader.

The runner's authenticated broker accepts submission requests for its own running
attempt. Agent-side `submission accept` uses `BINDOCRACY_BROKER` automatically;
it does not grant access to the control directory or accept an arbitrary attempt
identity. The organizer may call the direct acceptance function or CLI with an
explicit control directory. Environment variables and content hashes are **not**
a sandbox, a publisher signature, or scheduler authorization. The process-group
walltime/output controls below do not prevent a malicious process from escaping
into another session, reading other readable files, using the network, or launching
scheduler jobs. Use organizer-provisioned OS/container/cgroup/network and scheduler
policy for those restrictions. In particular, mount only accepted scientific inputs
and approved grader assets into a held-out evaluator's isolated environment; never
mount the mutable agent workspace as a scientific input.

## Build and score-free validation

```sh
bindocracy submission build work/campaign.duckdb work/submissions/final \
  --workspace work --design-set work/design_sets/selected.json \
  --structures-from held-in-fold-run
bindocracy submission check work/submissions/final
bindocracy submission accept work/submissions/final
```

The workspace is required for benchmark acceptance, but optional for ordinary
portable nonbenchmark exports. Without `--design-set`, the normal DB default
selection is frozen (sequence-bearing candidates, length then identity order).
With a design set, its exact membership and order are retained and checked against
the real database, including sequence, length, source tool/run and native identity.
There is no implicit ranking, score thresholding, or top-k selection.

`--structures-from` names one evaluator run ID or unambiguous name. Each member
must have exactly one `predicted_structure` with condition `complex`, replicate
`0` in that run. The referenced bytes and any recorded checksum/size must agree.
There is no implicit choice between replicas or folds. Task policy determines
minimum/maximum count, distinct-sequence requirements, and whether structures
are mandatory. Only canonical, nonempty protein sequences are accepted.

A bundle contains:

- `manifest.json`: schema/software versions, ordered members, count, target content
  identity, attempt/task/environment identity, policy, kit identities, and file hashes;
- `candidates.fasta`: exactly the same members, in exactly the same order;
- `target/`: sequence FASTA and supplied PDB/CIF target representations;
- `structures/`: explicitly selected candidate structures;
- `provenance/design_set.json`: frozen selection query and membership;
- `provenance/lineage.json`: recursive parent-design and parent-run closure and
  complete canonical general/model source configurations, their IDs, hashes,
  original source URIs, revisions, runtime/container and resource metadata;
- `provenance/kits/`: full kit files, execution locks, all content-inventoried source
  overlays and declared harness source dependencies. Containers/weights/databases
  with filesystem-stat identities are recorded explicitly as external runtime
  dependencies, not falsely called content-pinned or copied into candidate bundles;
- standalone custom-script bytes where a historical configuration uses a script
  rather than a kit; use a kit for multi-file custom code;
- a workspace provenance snapshot when building in a workspace.

The manifest `digest` is lowercase hexadecimal SHA-256 of the canonical JSON
manifest excluding its own `digest` field. Canonical JSON sorts keys, uses compact
separators, and preserves Unicode. `files` maps portable POSIX relative names to
lowercase hexadecimal SHA-256 of exact bytes. The manifest itself is not in `files`.
Kit locks retain their existing `sha256:`-prefixed digest convention.

Validation reads **only the bundle**, never the original database/workspace or
external paths mentioned as historical provenance. It checks the complete file
inventory, hashes, manifest digest, FASTA/order/count/duplicates, target identity,
lineage closure/config IDs and complete declared kit code closure. It refuses
symlinks, special files, path escapes and undeclared files. It never executes kit
code or a scorer. Hashes detect changes relative to a manifest, not whether an agent
truthfully described its history; authentic experimental evidence comes from the
organizer's independent execution records.

## Atomic acceptance and cutoff

Direct organizer command:

```sh
bindocracy submission accept /agent/work/submissions/final \
  --attempt-dir /organizer/attempts/attempt-001
```

`accept_submission(source, attempt_dir, *, now=None)` locks `attempt.lock`, loads
the organizer's `attempt.json`, `task.json`, `workspace.json` and protected
`target.fasta`, checks the running attempt/task/environment identity and organizer
submission policy, and safely copies the bundle into a private staging directory.
It revalidates the copy and source and checks the deadline **after** the copy.
It checks again after archive publication before committing acceptance; equality
with the deadline is too late. When the runner's budget ledger is present, each
admission check also enforces its monotonic cutoff and closed-gate policy: rolling
back the wall clock cannot extend an attempt. `now` is a trusted callable-clock
test seam, not an agent/CLI argument.

Completed copies live at `accepted/<digest>/` with read-only file/directory modes.
The exclusive lock is shared with runner state transitions. Accepted metadata
includes the digest, archive path, UTC acceptance timestamp, attempt ID and task
digest. The atomic `attempt.json.latest_submission` update is the authoritative
commit point; `acceptance_history/` retains acceptance events. A crash can leave an
unreferenced complete archive/history event, but never a partially copied accepted
bundle. Only the pointer in the attempt record selects the official submission.
Repeated acceptance replaces that pointer; earlier digest archives remain intact.
Permission separation, not mode bits alone, prevents an agent from modifying an
archive. The runner and evaluator never choose an earlier, better-scoring bundle.

## Organizer evaluator protocol

```sh
bindocracy benchmark evaluate /organizer/attempts/attempt-001 \
  --protocol /organizer/protocol.json --output /organizer/evaluations/attempt-001
```

Callable interface:

```python
from bindocracy.benchmark.evaluation import evaluate_submission
report = evaluate_submission(attempt_directory, protocol_json, new_output_directory)
```

Evaluation requires a stopped attempt and resolves only its **latest accepted**
archive. The output directory must be new and outside the agent workspace. An
arbitrary candidate path, bundled script or agent-selected evaluator is not an
acceptable grading input. The protocol is trusted, organizer-supplied JSON:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `protocol_id`, `version` | Organizer-declared scientific protocol identity |
| `argv` | Nonempty argument vector; no shell interpolation; `argv[0]` is an absolute executable |
| `files` | Absolute grader asset paths mapped to SHA-256 digests; executable and every file argument must be pinned; include imported modules, model/config files and other relevant scientific assets |
| `task_versions`, `task_digests` | Matching maps from task IDs to declared versions and captured content-based task digests |
| `timeout_seconds` | Finite positive command walltime bound |
| `max_output_bytes` | Positive combined stdout/stderr/result limit; default 16 MiB; polled while the process runs |
| `score_min`, `score_max` | Finite valid score range |
| `failure_score` | Predeclared finite in-range score for missing/invalid/infrastructure/agent outcomes in aggregation, not a fabricated evaluator measurement |
| `direction` | `max` or `min`; default `max` |
| `scientific` | Whether this is a real scientific protocol; **false for smoke demonstrations** |
| `environment` | Explicit trusted environment additions; mutable agent environment is not inherited |

All fields participate in the protocol digest. Use one frozen protocol document
for comparable repetitions. File hashes are rechecked before and after execution;
grader files inside the agent workspace are refused. Pin the trusted deployment's
full relevant asset closure, not just a thin launcher which imports mutable code.
Process isolation and trust in that closure remain organizer responsibilities.

The harness runs:

```text
<argv...> --bundle /organizer/attempts/ID/accepted/DIGEST --result OUTPUT/result.json
```

Working directory and `HOME` are the evaluation output directory. Standard input
is closed. `BINDOCRACY_EVALUATION_PROTOCOL` names the saved organizer protocol JSON;
`BINDOCRACY_EVALUATION_PROTOCOL_DIGEST` gives its canonical digest.
`PYTHONDONTWRITEBYTECODE=1` avoids adding Python caches to code inventories.
The harness saves stdout/stderr, terminates the process group at timeout or output
limit (and remaining descendants after leader exit), and saves `report.json` even
for ordinary grader failures. These bounds are not GPU/Slurm qualification and
do not authorize scheduling a job.

### Exact external evaluator result contract

Write one JSON object to the supplied `--result` path and exit zero:

```json
{
  "schema_version": 1,
  "bundle_digest": "the accepted manifest digest",
  "protocol_id": "the declared protocol ID",
  "protocol_version": "the declared protocol version",
  "protocol_digest": "the supplied protocol digest",
  "status": "ok",
  "score": 0.42,
  "candidate_results": [
    {"index": 0, "status": "ok", "metrics": {"declared_measurement": 0.42}}
  ],
  "details": {"method": "Describe the actual scientific computation and qualifications"}
}
```

This illustrates the shape, **not a supplied scientific score or judge**. Every
accepted candidate must appear exactly once, in contiguous manifest order, even
when its scientific result is negative. Top-level and per-candidate status must be
`ok` or `scientific_failure`. Scores/metric values must be real finite JSON numbers,
not numeric strings, booleans, null, NaN or infinity. The top-level score must lie in
the protocol's range. `details` is optional JSON metadata (also finite). Unknown
result/candidate fields, duplicate JSON keys and mismatched identities are refused.
The evaluator—not the harness—defines actual metrics and their scientifically
justified reduction to the score. A valid negative scientific result can have a
finite score and `scientific_failure`; command crashes/timeouts are infrastructure
failures, not negative binding results.

Reports distinguish `completed`, `scientific_failure`, `missing_submission`,
`agent_failure`, `invalid_submission`, `infrastructure_failure` and
`invalid_evaluator_output`, preserving original attempt state, declared agent,
repeat, usage, acceptance metadata, protocol digest, logs and errors. A failure
report has `score: null`, not invented scientific metrics. During aggregation only,
the explicit protocol failure policy supplies its included score.

### Runnable CPU smoke demonstration (not a scientific judge)

Save this as an organizer-owned `length_smoke.py`. It computes actual sequence
lengths from the accepted bundle, to exercise transport and result validation only.
Length is **not** a binding metric. Do not report its result as scientific success.

```python
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", required=True, type=Path)
parser.add_argument("--result", required=True, type=Path)
args = parser.parse_args()
manifest = json.loads((args.bundle / "manifest.json").read_text())
protocol = json.loads(Path(os.environ["BINDOCRACY_EVALUATION_PROTOCOL"]).read_text())
rows = [
    {"index": c["index"], "status": "ok", "metrics": {"length": len(c["sequence"])}}
    for c in manifest["candidates"]
]
result = {
    "schema_version": 1,
    "bundle_digest": manifest["digest"],
    "protocol_id": protocol["protocol_id"],
    "protocol_version": protocol["version"],
    "protocol_digest": os.environ["BINDOCRACY_EVALUATION_PROTOCOL_DIGEST"],
    "status": "ok",
    "score": sum(row["metrics"]["length"] for row in rows) / len(rows),
    "candidate_results": rows,
    "details": {"scientific_validation": False, "purpose": "CPU contract smoke only"},
}
args.result.write_text(json.dumps(result, allow_nan=False))
print("Software contract demonstration only; no binding inference")
```

Generate its protocol from a stopped organizer attempt (change the two path
arguments, rather than inserting fake hashes):

```python
import hashlib
import json
import sys
from pathlib import Path
from bindocracy.benchmark.evaluation import EvaluationProtocol

attempt_dir = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
python = Path(sys.executable).resolve()
attempt = json.loads((attempt_dir / "attempt.json").read_text())
task = json.loads((attempt_dir / "task.json").read_text())
protocol = EvaluationProtocol(
    protocol_id="length-contract-smoke", version="1", scientific=False,
    argv=(str(python), str(script)),
    files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (python, script)},
    task_versions={task["task_id"]: task["version"]},
    task_digests={task["task_id"]: attempt["task_digest"]},
    timeout_seconds=10, score_min=0, score_max=100000, failure_score=0,
)
Path("protocol.json").write_text(protocol.model_dump_json(indent=2))
```

## Repetitions, baseline and uncertainty

Declare the full experimental roster **before** running agents, including explicit
agent labels and repeat IDs. A baseline is an ordinary agent label using exactly
the same task/protocol and attempt-accounting rules. Roster JSON is a list of all
`attempt.json` paths, relative to the roster file or absolute:

```sh
bindocracy benchmark aggregate /organizer/roster.json \
  --protocol /organizer/protocol.json \
  --report /organizer/evaluations/attempt-001/report.json \
  --report /organizer/evaluations/baseline-001/report.json \
  --baseline baseline --seed 17 --bootstrap-samples 2000 \
  --output /organizer/aggregate.json
```

Callable interface is `aggregate_reports(reports, attempts=complete_roster,
protocol=protocol, bootstrap_samples=2000, seed=0, baseline=None)`.
Missing reports are explicit `missing_report` outcomes; failed/missing attempts
remain in denominators at `failure_score`. Every agent must have roster slots for
every protocol task. Duplicate attempts/reports, out-of-roster reports and differing
protocol or task identities are refused. The organizer must preserve the original
roster: no program can detect an organizer silently omitting an unrecorded trial.
There is no best-of-repeat, best-of-submission, or available-case dropping of failures.

The reported mean is the mean of **per-task means**, not the pooled attempt mean.
A deterministic hierarchical percentile bootstrap samples tasks with replacement,
then attempts within each sampled task, and reports a 95% interval. It describes
uncertainty over this task/attempt population, not proof of generalization beyond
the task sampling design; a single task/repetition cannot establish broad precision.
With `--baseline`, comparisons require complete identical `(task_id, repeat)` slots,
compute agent-minus-baseline differences, and bootstrap paired differences. Missing
results still use the declared failure score. Reported usage and attempt/outcome
counts remain explicit; telemetry is not silently converted to scientific quality.

## Full experiment evidence is a separate archive

```sh
bindocracy submission archive-evidence /organizer/attempts/attempt-001 \
  /organizer/evidence/attempt-001 \
  --evaluation /organizer/evaluations/attempt-001
```

Archive only stopped attempts. This copies the complete control and workspace trees
(including the campaign DB, logs, configurations, runs, budget/attempt records and
all local candidate bundles), with a separate hashed `experiment_evidence` manifest.
Its destination must be new and outside both source trees. Symlinks/special files
and changes during an individual file copy are refused; stop all writers first to
obtain a consistent experiment snapshot. Paths outside these trees, such as remote
scheduler logs or externally stored run outputs, must first be gathered by the
organizer into the evidence tree. Repeat `--evaluation` to include every evaluator
output tree (reports, logs and frozen protocols) under `evaluations/` in the archive.
An evidence archive is deliberately not
accepted by `submission check` and is never a substitute for the fixed candidate
submission.
