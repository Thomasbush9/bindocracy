# Generation workflow

Snakemake owns submission. The connector builds one task's command, environment,
resources, and expected outputs; Snakemake's Slurm executor plugin submits it and
tracks state, retries, and logs. Nothing here calls `sbatch`.

```text
prepare_mosaic_run --> mosaic_generate[task=0..N] --> collect_mosaic
                                                          |
                                                          v
                                   generation_complete <-- ingest_mosaic
```

## Files

| File | Role |
|---|---|
| `Snakefile` | The rule graph |
| `campaign.yaml` | Execution index: database, run root, and which configs to run |
| `smoke.yaml` | The same, for a one-design Slurm smoke test |
| `profiles/slurm/` | Executor settings; account and partition come from the configs |

The workflow config is an index, not a configuration. The general and model YAML
it points at are the durable scientific inputs, and they are what the database
stores.

## Run it

```bash
# 1. Dry run: expect one mosaic_generate job per sampling.jobs, one ingestion.
uv run snakemake --dry-run --configfile workflow/smoke.yaml

# 2. One design on real Slurm.
uv run snakemake --configfile workflow/smoke.yaml --profile workflow/profiles/slurm

# 3. The full campaign, once the smoke run is in the database.
uv run snakemake --configfile workflow/campaign.yaml --profile workflow/profiles/slurm
```

## What each run leaves behind

```text
runs/<model>/
|-- run.json          run identity, task plan, driver hash, and the whole
|                     config pair as JSON — a run is replayable from this
|                     alone, with no config file on the share
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

To execute the same configuration a second time, use a new run directory: a
`run_id` identifies an execution, not a configuration.
