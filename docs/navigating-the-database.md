# Navigating the campaign database

Everything the harness has ever done is in one DuckDB file. This is how to ask
it questions.

```bash
duckdb /n/holylfs06/.../binder_design/campaign.duckdb   # the real campaign
duckdb /n/holylfs06/.../binder_design/scoring_test.duckdb  # where all scoring work has happened so far
```

Nothing in the scoring stage has written to `campaign.duckdb` yet. Every number
described below currently lives in `scoring_test.duckdb`.

## The shape of it

Six tables. The whole design is that **a run is the unit of everything** — every
row anywhere is owned by exactly one run, and a run records what produced it.

```
configs ──< runs ──< designs      the candidates a run created
                 ├─< metrics      the numbers a run measured
                 ├─< decisions    the verdicts a run reached
                 └─< artifacts    the files a run wrote
```

| table | one row is | current size |
|---|---|---|
| `runs` | one invocation of one tool | 25 |
| `designs` | one candidate binder | 3,302 |
| `metrics` | one measurement of one design, at one replicate | 13,365 |
| `decisions` | one verdict — filter pass/fail, cluster, rank | 3,506 |
| `artifacts` | one file, optionally tied to one design | — |
| `configs` | the exact YAML pair a run was planned from | 26 |

`design_history` and `metric_history` are views, not tables.

## Start here: what has been run

```sql
SELECT name, tool, kind, status, n_requested, n_produced, created_at
FROM runs ORDER BY created_at DESC LIMIT 20;
```

`kind` separates what a run *did*: `generate` made candidates, `evaluate`
measured them, `filter`/`cluster`/`rank` decided about them. A scoring run is
`evaluate` and creates **no designs** — it measures candidates that already
exist, which is why `designs` does not grow when you score.

The three counts mean different things and the distinction is deliberate:
`n_requested` is how many were asked for, `n_attempted` how many were tried,
`n_produced` how many came back usable. A fold that failed is attempted and not
produced.

## Designs

```sql
-- Where did the candidates come from?
SELECT r.tool, r.name, COUNT(*) AS designs
FROM designs d JOIN runs r ON r.run_id = d.run_id
GROUP BY 1, 2 ORDER BY designs DESC;

-- One design, in full
SELECT design_id, native_id, length, sequence FROM designs WHERE design_id = '...';
```

`native_id` is what the tool called it; `design_id` is what the harness calls
it. Joining across tools only works on `design_id`.

## Metrics

The stored name is always `<model>_<metric>` — `boltz2_iptm`,
`chai1_iptm`, `protenix_base_iptm`. That prefix is structural, not a
convention: the same design scored by two models produces two rows, and a
column holding both would be meaningless.

```sql
-- What has been measured, and by what?
SELECT split_part(name, '_', 1) AS model, COUNT(DISTINCT design_id) AS designs,
       COUNT(*) AS rows
FROM metrics GROUP BY 1 ORDER BY rows DESC;

-- Every metric for one design
SELECT name, value, replicate, status FROM metrics
WHERE design_id = '...' ORDER BY name, replicate;
```

Three things to know before you aggregate:

**Replicates are rows.** Nothing is averaged at write time. Six samples of one
design are six rows with `replicate` 0–5. Averaging is your decision, made at
read time, and `AVG(value)` silently mixes replicates unless you say otherwise.

**A failed measurement is stored, not dropped.** `status = 'failed'` with
`value` NULL means something ran and produced nonsense; an absent row means
nothing ran. Only the first is a reason to distrust the model rather than the
harness. Filter with `WHERE status = 'ok'` when you want numbers.

**Direction is recorded per metric.** `direction` is `max`, `min` or `none`.
PAE metrics are stored raw — lower is better — rather than pre-negated, so
sorting requires reading `direction` rather than assuming.

```sql
-- Best designs by one model, correctly: only ok rows, averaged across replicates
SELECT design_id, AVG(value) AS iptm, COUNT(*) AS n_replicates
FROM metrics WHERE name = 'boltz2_iptm' AND status = 'ok'
GROUP BY 1 ORDER BY iptm DESC LIMIT 20;
```

## Comparing models on the same designs

This is what the scoring stage exists for. Pivot the metric name:

```sql
SELECT design_id,
       AVG(CASE WHEN name = 'boltz2_iptm'        THEN value END) AS boltz2,
       AVG(CASE WHEN name = 'chai1_iptm'         THEN value END) AS chai1,
       AVG(CASE WHEN name = 'af3_iptm'           THEN value END) AS af3,
       AVG(CASE WHEN name = 'protenix_base_iptm' THEN value END) AS protenix
FROM metrics WHERE status = 'ok' GROUP BY 1
HAVING boltz2 IS NOT NULL AND chai1 IS NOT NULL
ORDER BY boltz2 DESC LIMIT 25;
```

A caution that the schema cannot enforce: two models agreeing on a *number* is
not two models agreeing on a *structure*. Measured on nine models, the median
binder RMSD between two predictions of the same design was 22.7 Å. Use the
saved structures (below) when the question is about the pose.

## Structures

Every predicted pose is an artifact row tied to the design it is of. This is
what makes a later epitope, contact or clash pass a read rather than a re-fold:

```sql
SELECT a.design_id, a.uri, a.metadata->>'model' AS model,
       a.metadata->>'condition' AS condition, a.metadata->>'replicate' AS replicate
FROM artifacts a
WHERE a.kind = 'predicted_structure' AND a.design_id = '...';
```

`uri` is relative to the run's `output_uri`, so the absolute path is
`runs.output_uri || '/' || artifacts.uri`. Other artifact kinds worth knowing:
`scored_metrics` (the raw JSONL), `task_status`, `log`, and the archived
`driver_script` / `general_config` / `model_config`.

## Provenance: what actually produced a number

Every run carries the configs it was planned from, byte for byte:

```sql
SELECT c.tool, c.model_name, c.model_config_hash,
       r.container_digest, r.code_revision
FROM runs r JOIN configs c ON c.model_config_id = r.model_config_id
WHERE r.run_id = '...';
```

`workflow_metadata` on a scoring run is where the useful detail sits:

```sql
SELECT name,
       workflow_metadata->>'model'            AS model,
       workflow_metadata->>'protocol_sha256'  AS protocol,
       workflow_metadata->>'design_set_digest' AS design_set,
       workflow_metadata->>'scope_id'         AS scope
FROM runs WHERE kind = 'evaluate';
```

**`protocol_sha256` is the comparability key.** Two runs sharing it measured the
same quantity the same way, whatever designs they scored — which is what makes
incremental scoring safe. Two runs that differ on it are not comparable however
similar the YAML looks. It covers the model and its knobs and deliberately
excludes the design set.

**`design_set_digest` and `scope_id`** identify the frozen candidate set. A rank
decision must be scoped to one, because "best of these 400" and "best of these
3,302" are different claims.

## Decisions

```sql
SELECT kind, name, COUNT(*) FILTER (WHERE passed) AS passed, COUNT(*) AS total
FROM decisions GROUP BY 1, 2;
```

Filter runs write one decision per rule *plus* one for the whole set, so a
design that failed can be asked which rule failed it. A design with no metric
for a thresholded name gets an explicit failing verdict rather than vanishing —
missing is a failure, never a pass.

`bindocracy rank apply` writes a `rank` run with scoped ordering decisions and
named `filter` decisions for cohort membership. This deliberately reuses the
existing selector:

```bash
uv run bindocracy designset build campaign.duckdb \
  --passed-filter top50 --filter-run RANK_RUN_ID --out-dir selected
```

`--filter-run` identifies the source of filter **decisions**, not only runs whose
kind is `filter`: native generation/evaluation verdicts and rank memberships are
also valid. Reused run names are refused when ambiguous; use the exact run ID.
Ranking preserves all original designs/metrics and stores exclusion reasons.
See [stored ranking policies](scoring-stage.md#11-stored-ranking-and-cohort-selection)
for coverage, deduplication, group-wise ordering, and head/tail selection.

## Two habits worth having

**Read through `store/query.py`, not raw SQL, when writing code.** It takes a
connection rather than a `CampaignStore`, so it cannot write by accident, and
it returns frozen `DesignRow` / `MetricRow` rather than the records the write
path uses.

**Open read-only when exploring:**

```python
import duckdb
con = duckdb.connect("scoring_test.duckdb", read_only=True)
```

A second writer is refused; a reader is not.
