# Mosaic generation: end-to-end implementation plan

This plan covers Mosaic from authored configuration through generated designs
stored in DuckDB. Filtering, clustering, evaluation by held-out models, and
optimization are deliberately out of scope.

## Goal

Given one workflow configuration that points to a general campaign config and
one or more Mosaic configs, the system should:

1. Validate and store each general + Mosaic configuration pair.
2. Create a persistent identity and directory for each generation run.
3. Archive the exact Mosaic driver script used by the run.
4. Run the requested Mosaic tasks through Snakemake and Slurm.
5. Parse all task outputs into validated Bindocracy records.
6. Ingest those records through a serialized DuckDB writer.
7. Make every generated sequence, native loss, artifact, and provenance link
   queryable from the campaign database.

The intended flow is:

```text
workflow config
      |
      v
plan run + archive inputs
      |
      v
N Mosaic generation tasks
      |
      v
collect one validated staging bundle
      |
      v
serialized DuckDB ingestion
```

## Architectural decisions

### Snakemake owns job submission

The Mosaic connector should construct the command, environment, resources, and
expected outputs for one task. It should not call `sbatch` itself. Snakemake
should submit each task using its Slurm executor plugin so that it can track job
state, retries, logs, resources, and failures without a nested scheduler.

### Output adapters do not write DuckDB

The Mosaic adapter reads a completed run directory and returns a
`CollectedRun`. It serializes that result to a staging JSON file. A later,
serialized rule is the only process that writes those results to DuckDB.

### A run ID identifies an execution, not a configuration

Identical configurations may be executed more than once. Every real execution
therefore receives a new `run_id`. A restarted Snakemake execution must reuse
the `run_id` already saved in its run manifest rather than generating another.

### Preserve partial and failed work

Mosaic writes designs as they finish. Collection should retain valid completed
designs when a task times out or fails and mark the run `partial` or `failed` as
appropriate. Failed attempts are useful campaign history and should not become
invisible merely because they produced no designs.

## Phase 1: define the Mosaic output contract

Update the selected `hallucinate.py` driver before implementing its parser.
The current draft still hard-codes the target sequence, target MSA, and binder
length. The driver must accept all run-dependent values through arguments or a
resolved input file:

- Target FASTA or target sequence
- Target MSA
- Binder length
- Task ID
- Base seed
- Number of designs
- Maximum runtime
- Output directory

Prefer append-only JSON Lines over the current two-line FASTA-like format. One
completed design should produce one record:

```json
{"native_id":"task-0000-design-000003","sequence":"ACDEFG","seed":3,"ranking_loss":0.1842}
```

Required fields are:

| Field | Meaning |
|---|---|
| `native_id` | Stable identifier within the run |
| `sequence` | Completed binder sequence |
| `seed` | Actual seed used for this design |
| `ranking_loss` | Mosaic's native post-design loss |

The driver should flush each JSON line after writing it so a terminated task
does not lose earlier designs.

Each task should also write `status.json` atomically when it finishes. It should
contain at least:

- Task ID
- Start and finish timestamps
- `succeeded`, `partial`, or `failed` status
- Attempted and produced counts
- Process exit information or error summary
- Output filename

Do not use the row number after filtering as the native identity. A useful
scheme is `task-{task_id:04d}-design-{design_index:06d}`.

### Phase 1 completion criteria

- No campaign-specific target, MSA, binder length, or seed remains hard-coded.
- A one-design local or container invocation produces valid `designs.jsonl` and
  `status.json` files.
- An interrupted output file still contains every completely written record.

## Phase 2: create the persistent run manifest

Before submitting GPU work, create a run directory such as:

```text
runs/<run_id>/
|-- run.json
|-- provenance/
|   |-- general.yaml
|   |-- mosaic.yaml
|   `-- hallucinate.py
|-- tasks/
|   |-- 0000/
|   |   |-- designs.jsonl
|   |   `-- status.json
|   `-- 0001/
|       |-- designs.jsonl
|       `-- status.json
|-- logs/
`-- collected.json
```

`run.json` is the stable handoff between planning, generation, collection, and
ingestion. It should include:

- `run_id`
- `general_config_id` and `model_config_id`
- Run name, tool, and `kind: generate`
- Creation timestamp
- Expected task IDs and output paths
- Requested designs per task
- Resolved resource request
- Paths and checksums for the archived configs and driver
- Container path and, when available, its digest
- Code revision
- Snakemake execution metadata

Copy the driver into `provenance/` and execute that archived copy. This makes
the actual code used by the run inspectable even when the Mosaic working tree
changes later.

The planning operation must be restart-safe: if a valid manifest already
exists, reuse it; do not silently replace its run ID or archived inputs.

## Phase 3: implement the Mosaic output adapter

Implement `MosaicOutputAdapter` in `src/bindocracy/adapters/mosaic.py`. It should
receive the run directory and its `RunRecord`, then:

1. Discover the expected task outputs from `run.json`.
2. Validate every complete JSONL record.
3. Reject invalid sequences, non-numeric losses, duplicate native IDs, and
   records whose task identity disagrees with their directory.
4. Retain identical sequences when their native IDs differ. Sequence hashes can
   be used later for cross-run comparison and deduplication.
5. Emit one `DesignRecord` per valid record.
6. Emit one `MetricRecord` per design with:
   - `name="mosaic_ranking_loss"`
   - `direction="min"`
   - The generation run's `run_id`
7. Emit `ArtifactRecord`s for native output files, task status files, logs, and
   the archived driver.
8. Set `n_requested`, `n_attempted`, and `n_produced` on the run.
9. Derive the final run status from expected tasks, status files, and parsed
   designs.
10. Return one validated `CollectedRun` without opening DuckDB.

Record identifiers produced by collection should be deterministic from stable
inputs such as `run_id`, `native_id`, metric name, and artifact path. Reparsing
an unchanged run directory must produce the same records. The run ID itself
must still originate from the persistent manifest because repeated executions
of the same config are distinct runs.

### Adapter tests

Add small committed output fixtures and cover:

- One valid task and design
- Several tasks and designs
- Empty output
- A truncated final JSON line
- Invalid sequence
- Missing or non-numeric loss
- Duplicate native ID
- Duplicate sequence with different native IDs
- Missing task output
- Failed task with no designs
- Partial task with completed designs
- Deterministic re-collection

## Phase 4: stage and ingest collected output

Add a CLI boundary resembling:

```bash
uv run bindocracy collect mosaic \
  runs/<run_id>/run.json \
  --output runs/<run_id>/collected.json

uv run bindocracy ingest \
  campaign.duckdb \
  runs/<run_id>/collected.json
```

`collect` should validate the manifest and serialize `CollectedRun` as JSON.
`ingest` should deserialize and validate that bundle before opening DuckDB.
Ingestion must remain one transaction in foreign-key order.

Make ingestion restart-safe before putting it under Snakemake. Re-running the
same ingestion rule must either be an explicit no-op for an identical bundle or
produce a clear integrity error without partial writes. It must never create a
second copy with new random IDs.

### Store integration test

Using a temporary DuckDB:

1. Insert one general + Mosaic config pair.
2. Parse a fixture run directory.
3. Serialize and deserialize its staging bundle.
4. Ingest it.
5. Assert the expected run, designs, metrics, and artifacts.
6. Query `design_history` and verify both config IDs.
7. Repeat ingestion and verify the documented restart behavior.

## Phase 5: implement the Mosaic launch connector

The connector consumes the validated general config, Mosaic config, run
manifest, and task ID. It returns a launch specification containing:

- An argument vector, not a shell-interpolated command string
- Environment variables required by `mosaic-exec.sh`
- CPU, memory, GPU, partition, account, and walltime resources
- Log path
- Expected task output paths

The command should execute the archived driver, for example conceptually:

```text
mosaic-exec.sh python <run>/provenance/hallucinate.py
  --target-fasta <target>
  --target-msa <msa>
  --binder-length <length>
  --task-id <task>
  --seed-base <seed>
  --n-designs <count>
  --max-runtime <hours>
  --save-dir <run>/tasks/<task>
```

Keep Slurm submission outside the connector. Unit-test the resulting argument
vector, environment, resources, and expected outputs without requiring a GPU.

## Phase 6: add the generation-only Snakemake workflow

Use a small orchestration config that points to the scientific configs rather
than duplicating their contents:

```yaml
database: /absolute/path/to/campaign.duckdb
run_root: /absolute/path/to/runs

general_config: /absolute/path/to/general_config.yaml

models:
  mosaic:
    - /absolute/path/to/configs/mosaic/config_01.yaml
```

This workflow config is an execution index. The general and model configs
remain the durable scientific inputs stored in the database.

Initial rule graph:

```text
prepare_mosaic_run
        |
        +--> mosaic_generate[task=0] --+
        +--> mosaic_generate[task=1] --+--> collect_mosaic --> ingest_mosaic
        `--> mosaic_generate[task=N] --+                         |
                                                                 v
                                                      generation_complete
```

Suggested rule responsibilities:

1. `prepare_mosaic_run`
   - Validate/load the config pair.
   - Create or reuse the run manifest.
   - Archive configs and driver.
   - Register the planned run through a serialized database-writing step if
     failed attempts must be visible immediately.
2. `mosaic_generate`
   - One wildcard job per Mosaic task.
   - Use resources from the validated configs.
   - Run through Snakemake's Slurm executor.
3. `collect_mosaic`
   - Aggregate all expected task directories.
   - Preserve partial output.
   - Write `collected.json` atomically.
4. `ingest_mosaic`
   - Be the only DuckDB writer at this stage.
   - Insert designs, metrics, and artifacts and finalize the run status.
5. `generation_complete`
   - Produce a small completion marker only after successful ingestion.

If separate register and finalize rules both write DuckDB, give them the same
exclusive Snakemake resource so they can never overlap. Single-writer means
serialized access, not necessarily that the database can only be opened once
in the entire workflow.

## Phase 7: test the workflow

Use progressively more realistic tests.

### 1. Unit tests

Run the adapter, manifest, connector, and identifier tests through `uv`:

```bash
uv run pytest -q
uv run ruff check src tests
```

### 2. Local mock end-to-end test

Use a fixture generator that writes the exact Mosaic output contract without a
container or GPU. Run the complete Snakefile locally and verify the temporary
database. This tests dependencies, paths, staging, and serialization cheaply.

### 3. Snakemake dry run

```bash
uv run snakemake --dry-run --printshellcmds \
  --configfile workflow/campaign.yaml
```

Verify the expected number of Mosaic task jobs and exactly one final ingestion
path.

### 4. Real Slurm smoke test

Use a dedicated Mosaic smoke config:

```yaml
sampling:
  jobs: 1
  designs_per_job: 1
  max_runtime_hours: 0.25
```

Submit through Snakemake's Slurm executor, inspect the log and native output,
then query DuckDB.

## Generation-only definition of done

The vertical slice is complete when one real one-design Mosaic smoke run leaves:

- One paired row in `configs`
- One generation row in `runs` with accurate status and counts
- One sequence row in `designs`
- One `mosaic_ranking_loss` row in `metrics`
- Artifacts for the native output, task status, log, and archived driver
- The expected general and model config IDs in `design_history`
- A restart-safe run manifest and ingestion path
- Passing unit, store-integration, local workflow, and Slurm smoke tests

Useful verification queries:

```sql
SELECT
  run_id,
  model_config_id,
  status,
  n_requested,
  n_attempted,
  n_produced
FROM runs
WHERE tool = 'mosaic' AND kind = 'generate'
ORDER BY created_at DESC;
```

```sql
SELECT
  design_id,
  native_id,
  sequence,
  general_config_id,
  model_config_id,
  producing_run_name
FROM design_history
ORDER BY created_at;
```

```sql
SELECT
  d.native_id,
  m.name,
  m.value,
  m.direction
FROM designs AS d
JOIN metrics AS m USING (design_id)
WHERE m.name = 'mosaic_ranking_loss';
```

## Recommended next coding session

Implement the smallest testable slice before adding Snakemake:

1. Finalize the Mosaic JSONL and task-status output contract.
2. Implement `MosaicOutputAdapter`.
3. Add adapter fixtures and unit tests.
4. Add staging-bundle serialization and the store integration test.

Only after those pass should the launch connector and Snakefile be added. This
keeps scheduler problems separate from parser and database problems.
