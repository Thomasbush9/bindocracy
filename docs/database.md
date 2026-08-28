# Campaign database

One DuckDB file represents one target campaign. The target sequence, hotspots,
cluster settings, and campaign identity live in the general configuration rather
than being repeated in every table row.

## Tables

| Table | One row represents |
|---|---|
| `configs` | One paired general + model configuration, with complete JSON for both |
| `runs` | One generation, evaluation, filtering, clustering, optimization, or ranking execution |
| `designs` | One candidate produced by a generation or optimization run |
| `artifacts` | One file associated with a run or design |
| `metrics` | One numeric measurement of one design by one run |
| `decisions` | One filter, cluster, selection, rank, or benchmark-label outcome |

Optimization does not introduce another kind of candidate. An optimized sequence
is another `designs` row whose `parent_design_id` identifies its input and whose
`run_id` identifies the optimization strategy that produced it.

The held-out evaluator writes its raw outputs to `metrics`. A final rank belongs
in `decisions` only when it is frozen together with a `scope_id`, because ranks
change when the candidate set changes.

## Create an empty database

```bash
bindocracy init-db runs/dio3-cut/campaign.duckdb
```

The command refuses to overwrite an existing file. Use `--if-not-exists` when an
idempotent workflow rule should accept a database that is already initialized.

## Load the general and Mosaic configs

```bash
bindocracy config load-mosaic runs/dio3-cut/campaign.duckdb \
  --general /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/configs/general_config.yaml \
  --model /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/configs/mosaic/config_01.yaml
```

This command validates the YAML with Pydantic, checks the referenced target,
script, container, and weights, and inserts one row containing both complete JSON
documents. The stable `general_config_id` groups model configurations using the
same general parameters; the stable `model_config_id` identifies the exact pair
and is what a future run references. Source paths and hashes of the original YAML
bytes are retained as provenance. The command initializes the database if needed
and does not write to any other domain table. Repeating it is safe.

Inspect commonly used parameters directly:

```sql
SELECT
  general_config_id,
  model_config_id,
  general_config_json->'target'->>'name' AS target,
  model_config_json->'sampling'->>'binder_length' AS binder_length,
  model_config_json->'resources'->>'walltime' AS walltime
FROM configs;
```

Recover either JSON document as YAML using its corresponding ID:

```bash
bindocracy config export runs/dio3-cut/campaign.duckdb \
  <general_config_id-or-model_config_id> recovered.yaml
```

The Python equivalents are `config_yaml_from_db()` for a YAML string and
`recover_config_yaml()` for writing a file.

## Output-adapter contract

An output adapter reads files but never opens DuckDB. It receives the `RunRecord`
created for the job and returns one `CollectedRun`:

```python
from pathlib import Path

from bindocracy.adapters import CollectionError, OutputAdapter
from bindocracy.store import ArtifactRecord, CollectedRun, DesignRecord, RunRecord


class ExampleAdapter(OutputAdapter):
    tool = "example"

    def succeeded(self, run_dir: Path) -> bool:
        return (run_dir / "designs.csv").is_file()

    def collect(self, run_dir: Path, run: RunRecord) -> CollectedRun:
        output = run_dir / "designs.csv"
        if not output.is_file():
            raise CollectionError(f"missing expected output: {output}")

        designs = []
        # Parse the tool-specific file here. Assign a stable native_id from the
        # output, not a row number that can change when failed rows are removed.
        for native_id, sequence in parse_tool_output(output):
            designs.append(
                DesignRecord(
                    run_id=run.run_id,
                    native_id=native_id,
                    candidate_type="sequence",
                    sequence=sequence,
                )
            )

        artifact = ArtifactRecord(
            run_id=run.run_id,
            kind="native_design_table",
            uri=str(output.relative_to(run_dir)),
            media_type="text/csv",
        )
        return CollectedRun(run=run, designs=tuple(designs), artifacts=(artifact,))
```

The adapter should preserve tool-native identifiers and scores, distinguish
padded/failed rows from produced designs, identify binder chains explicitly, and
emit paths relative to the campaign or run directory. `DesignRecord` normalizes
sequence case/whitespace and computes its length and SHA-256 hash.

Native model scores use the generation run's `run_id`. A later common-scoring
pass creates an `evaluate` run and emits `MetricRecord` objects that reference
the already existing design IDs. Filtering and clustering similarly create new
runs but emit `DecisionRecord` objects rather than new designs.

## Serialized ingestion

Container jobs write their native outputs independently. Collection rules turn
those outputs into normalized staging bundles. A single Snakemake ingestion rule
then calls `CampaignStore.ingest()` for each bundle. Each bundle is inserted in
one transaction in foreign-key order:

```text
configs -> run -> designs -> artifacts -> metrics -> decisions
```

This keeps DuckDB single-writer, makes failed ingestion atomic, and allows the
database to be rebuilt from immutable run directories and staging bundles.
