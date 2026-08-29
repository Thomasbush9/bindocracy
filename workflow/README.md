# Generation workflow

Snakemake owns submission. The connector builds one task's command, environment,
resources, and expected outputs; Snakemake's Slurm executor plugin submits it and
tracks state, retries, and logs. Nothing here calls `sbatch`.

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

## Run it

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
