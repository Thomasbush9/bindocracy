# The scoring stage

Status: **scoring, filtering, frozen selection, and stored ranking have CLI
interfaces (2026-09-24).** Registered scorers run through the shared Snakemake
workflow as `evaluate` runs. `designset build`, `filter apply`, and `rank apply`
provide CPU-side selection and persisted decisions. Sections 1–8 preserve the
original design discussion; §9 covers target preparation, §10 records the earlier
scientific priorities, and §11 documents the current ranking interface.

`harness-design.md` closes by naming what is missing: *"No cross-tool quality
comparison exists. Every score comes from a different scorer, mostly the same
model that produced the design. A common scorer over all designs is the next
step."* This is that step.

The scale of the problem, read from `campaign.duckdb`:

| | |
|---|---|
| designs | 3,302, all distinct sequences, zero cross-tool duplicates |
| designs passing their own tool's filter | 57 (1.7%), under six mutually incomparable filter names |
| designs with any metric another tool also has | **0** |
| distinct binder lengths | 78, over 60–149 residues |

A filter over today's metrics demonstrates the point better than the table
does. Applying two real thresholds — FreeBindCraft's interface pTM and
Genie 3's ipTM — to all 3,302 designs leaves **3,116 unscored and 0 passing**,
not because the designs are bad but because no two tools measured the same
thing.

---

## 1. Four run kinds, and only one of them is a tool

`runs.kind` already accepts `evaluate`, `filter`, `cluster` and `rank`, and
`docs/database.md` already prescribes the shape: a common scorer creates an
`evaluate` run emitting `MetricRecord`s against existing design IDs; filtering
and clustering create runs emitting `DecisionRecord`s rather than new designs.
No schema change is needed.

What is *not* obvious, and is the main thing to review, is that these four are
two different kinds of thing:

| Stage | Shape | Why |
|---|---|---|
| `evaluate` | a real `ToolPlugin` | Runs a model on a GPU inside a container, fans out over shards, writes a run directory an adapter parses. Exactly what the plugin seam describes. |
| `filter` | a library function plus a CLI command | Reads the database, evaluates arithmetic, writes verdicts. No container, no tasks, no run directory. |
| `cluster` | probably a plugin | Foldseek and an exhaustive all-vs-all are real compute, but CPU. Undecided; see §7. |
| `rank` | a library function plus `rank apply` | Scoped ordering and named cohort memberships; same persistence path as `filter`. |

`docs/adding-a-tool.md` says to name a missing contract rather than work around
one. The missing contract is a **database-to-database run**: something that
produces records without producing a run directory. Forcing filtering through
`ToolPlugin` would mean inventing a container it does not need, a task fan-out
over work that takes under a second, and a `ResourceConfig` declaring a GPU it
never allocates.

The way out costs nothing: a filter builds a `CollectedRun` in process, writes
a staging bundle, and the existing `bindocracy ingest` writes it. The
single-writer rule, the idempotence key and the atomic transaction all keep
working untouched.

## 2. A design set is a file, not a query

This is the one genuinely new mechanism, and everything else leans on it.

Every generation tool's inputs are files named in its config, digested by
`plan_run` and verified before launch. A scorer's input is a query result,
which has none of those properties: it changes whenever a generation run lands,
it cannot be digested, and it cannot be re-read later to find out what was
scored.

So a design set is materialised before any run is planned:

```text
design_sets/<digest>.fasta    what the scorer reads
design_sets/<digest>.json     what it means: query, counts, provenance
```

Content-addressed, which buys three things at once:

- Two scoring runs over the same candidates share one file and one identity.
- A rank decision gets a `scope_id` that genuinely names the candidate set it
  ranked. The schema requires one and it is otherwise easy to fake.
- Re-running the same query after new designs land produces a **different**
  digest, so the change is visible rather than silent.

Building one is a CLI step run once. The scoring run that follows is an
ordinary run whose config names the path, so **no new Snakemake machinery is
needed** — `ToolPlan.inputs` digests it like any other input.

### Entries are ordered by length, and shards are contiguous

Every JAX structure model recompiles per binder length. The campaign holds 78
distinct lengths, so a strided shard would hand every task all 78
compilations. Ordering entries by length and cutting contiguous blocks gives,
measured on the real set at eight shards:

```text
shard 0:  413 designs,  15 lengths      shard 4:  413 designs,  17 lengths
shard 1:  413 designs,   8 lengths      shard 5:  413 designs,   7 lengths
shard 2:  413 designs,   9 lengths      shard 6:  413 designs,   9 lengths
shard 3:  413 designs,  10 lengths      shard 7:  411 designs,  10 lengths
```

Seven to seventeen compilations per task instead of seventy-eight.

## 3. Metrics are stored per replicate, and the name carries the scorer

Two conventions, both load-bearing.

**The stored name is `<model>_<metric>`** — `esmfold2_iptm`, `boltz2_iplddt`.
An unprefixed `iptm` would silently mean whichever model wrote it last, which
is how a held-out judge stops being held out. `adapters/scoring.py` makes the
prefix structural: a `MetricSpec` holds the bare key and only ever produces a
stored name through `stored_name(model)`.

**Replicates are rows, not aggregates.** The schema's
`UNIQUE (run_id, design_id, name, replicate)` already supports this, so a
six-sample scoring run writes six rows per metric per design at
`replicate=0..5`. Nothing is collapsed at write time, because "ipTM above 0.6"
is ambiguous across six samples and mean, median and min are different filters.
Which one applies is stated by the filter set.

**Direction is declared once.** A metric stored with the wrong direction sorts
backwards and nothing about the row says so. `spec_for()` refuses an
unregistered key rather than defaulting to `none`; adding a metric is one line
and forces somebody who knows the answer to state the direction. PAE metrics
are stored raw — lower is better — rather than pre-negated, because a database
that stores an already-flipped number cannot be joined against one that does
not.

## 4. A filter set is a stored policy document

Every threshold in the campaign today lives inside the tool that applied it,
which is why nobody can say what `pxdesign_af2ig` required without reading
PXDesign's source. A `FilterSet` is embedded whole in the filter run's
`model_config_json`, so `bindocracy config export` recovers the exact
thresholds that produced any verdict, and changing a number changes the config
hash and therefore produces a new run rather than reinterpreting an old one.

Three rules are enforced by the models rather than left to whoever writes YAML:

1. **A threshold names its aggregation.** A config that does not choose is
   rejected.
2. **A missing metric is a failure, not a pass.** The alternative silently
   admits every design the scorer failed on, which is the exact shape of the
   silent-success bugs `known-issues.md` is organised around. It also means a
   design nothing ever scored gets an explicit failing verdict instead of being
   absent — absence and rejection look identical in a query and mean opposite
   things.
3. **A rule records why.** Each verdict carries the observed value beside the
   threshold that rejected it:

```json
{"metric": "esmfold2_iptm", "aggregate": "mean", "op": ">=",
 "threshold": 0.7, "observed": 0.42, "n_replicates": 6,
 "passed": false, "missing": false}
```

Each design gets one decision per rule plus one for the set. The per-rule rows
make a filter debuggable — "which requirement did this miss" is the question
actually asked — and the set-level row is what selection reads. Storing only
the second would make every rejection look identical.

A replicate whose status is not `ok` is **dropped**, not treated as zero.
A failed fold is an absence of evidence; averaging a zero into six samples
would quietly move a design down the ranking as though it had been measured
and found bad.

## 5. What was drafted

| File | What it is |
|---|---|
| `store/query.py` | The read path. Takes a connection, not a `CampaignStore`, so it cannot write by accident. Returns frozen dataclasses, not `DesignRecord`s — a row read back is not a record being written. |
| `runs/designset.py` | Building, writing and sharding a frozen design set. |
| `adapters/scoring.py` | `MetricSpec` registry and `metric_records()`. The 13 interface and 4 monomer metric names match `mosaic/benchmark/model_matrix.py` exactly, so a stored column and a benchmark column are the same quantity. |
| `filters/models.py` | `Threshold`, `FilterRule`, `FilterSet`. |
| `filters/apply.py` | Pure functions over records. Nothing opens a database or launches anything, so a threshold's behaviour is testable with a list of floats. |
| `filters/config.py` | `FilterConfig` and the `CollectedRun` builder for a filter pass. |

One library change was made: **`ToolPlan` gained a `kind` field defaulting to
`RunKind.GENERATE`, and `plan_run` reads it** instead of hard-coding. That is
two lines, and all 498 existing tests pass unchanged.

One library change was considered and **rejected**: widening
`ResourceConfig.gpus` to `ge=0`. Filtering never goes through SLURM, so it
needs no `ResourceConfig` at all. Widening it now would put a GPU field on a
thing that has no GPU. Revisit only if clustering becomes a scheduled CPU job.

## 6. What is left

The original implementation checklist follows. The CLI surface is now available:
`bindocracy designset build`, `bindocracy filter apply`, and `bindocracy rank apply`;
see §11 for stored ranking and reusable cohort handoffs.

1. **CLI surface — implemented.** Explicit database selections, threshold policies,
   and ranking/cohort policies no longer require ad hoc SQL/Python glue.
2. **Sequence metrics and the negative controls.** Length, net charge and
   molecular weight are specified in `adapters/scoring.py` and not computed.
   They cost no GPU and they are the bar every real metric must clear — the
   existing benchmark found four of six models failing to beat net charge
   alone.
3. **The scorer plugin.** Base it on `mosaic/benchmark/model_matrix.py`, not on
   `mosaic/screen_proposals.py`: the latter stubs binder sidechains to poly-G,
   which the benchmark identified as its largest protocol confound, and assumes
   one binder length where this library has 78.
4. **Epitope contact.** Specified in `EPITOPE_METRICS`, not computed. Needs the
   scored structures to exist first. Biotite is already installed, so no
   container.
5. **Tests.** `test_tool_seam.py` and `test_third_tool.py` should gain an
   `evaluate`-kind case, and the filter engine deserves table-driven tests over
   fabricated metric rows.

## 7. Decisions I would like settled before implementing

- **Is `cluster` a plugin or a library function?** Foldseek is real compute and
  wants a container; the exhaustive all-vs-all Smith-Waterman is a few minutes
  of vectorised arithmetic and does not. They may not want to be one stage.
- **Does a scoring run write one structure per design, or per replicate?**
  Six samples over 3,302 designs is roughly 10 GB of mmCIF. Cheap, but only
  worth keeping if something will read it.
- **Should `DesignQuery.distinct_sequences` default on?** It is off, and today
  it would change nothing — the campaign has zero cross-tool duplicates. The
  day two tools converge, off means scoring the same sequence twice.
- **Where do design sets live?** The draft writes `design_sets/<digest>.*`
  beside the database. Under `runs/` would tie them to a run they are meant to
  outlive.

---

## 8. What the first run found (2026-09-07)

Six models, 20 designs spanning all seven generation tools, complex and monomer.
The path works end to end: AF2's run reached the database as an `evaluate` run
with metrics attached to existing design IDs, using no new workflow machinery.
Four real problems surfaced, and they are worth keeping because three of them
are silent.

### 8.1 `mosaic.sif` predates the MSA-routing fix — **the serious one**

`BENCHMARK.md` records that OpenFold3, Protenix and ESMFold2 once ignored a
supplied `msa_path` and queried the ColabFold server instead, and that this was
fixed by routing every backend through `mosaic.msa.require_msa`. **The fix is in
the host checkout, not in the image.** The container's
`of3.py::_compute_msas` is the pre-fix version:

```python
msa_chains = [tc for tc in chains if tc.use_msa]
if not msa_chains:
    return augment_main_msa_with_query_sequence(query_set, settings)
```

It selects every MSA-wanting chain and submits them all, never consulting
`msa_path` and never calling `require_msa` — which the image *does* ship, at
`/opt/mosaic/src/mosaic/msa.py`, unused by this caller. Observed directly:
six ColabFold submissions in one OpenFold3 run.

So **OpenFold3 and Protenix folded against a server alignment while Boltz-1 and
Boltz-2 folded against the campaign's local 3,032-sequence a3m** (verified in
their logs: `n_msa 3001` for the target, `n_msa 1` for the binder). Their
numbers from this run are not comparable with each other, which is exactly the
confound the benchmark set out to remove.

The benchmark runs avoided it by setting `MOSAIC_DEV_SRC`, which binds the host
`src/` over the image's copy — visible in the audit log as `note: DEV SOURCE
... overrides the image's src`. The scoring runs did not set it.

Two ways out, and the choice is a real one:

- **Set `MOSAIC_DEV_SRC` in the scorer's launch environment.** Immediate. But
  `mosaic-exec.sh` announces it loudly precisely because a run with it set is
  not reproducible from the image alone. If taken, the harness must digest that
  source tree and record it beside the SIF checksum — the same rule
  `known-issues.md` §2.3 sets for the Genie 3 overlays.
- **Rebuild `mosaic.sif` from current source.** Durable, and keeps the property
  that the image determines the result. Roughly a ten-minute rebuild.

### 8.2 AF2 refuses an MSA at an interface

`AssertionError: AF2 interface does not support MSA yet`, on all 20 designs.
AF2 takes an alignment for a single chain and refuses one at an interface.
Note the trap in the sources: `models/af2.py:343` reads `chain.use_msa` and
loads the a3m, which reads as support, while `design_config.py` marks af2
`accepts_msa: False`, which reads as a stale note contradicting the code.
Both were misleading; only running it was decisive.

### 8.3 ESMFold2 dies on a read-only weight cache

`OSError: [Errno 30] Read-only file system` on a HuggingFace `refs/main` write,
raised from inside a model constructor so it reads like a missing weight rather
than a cache write. The caches are bound read-only by design. Fixed with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the launch environment.

### 8.4 Pooled success accounting hid a total failure — a harness bug

AF2 reported `succeeded, n_produced=20` while **every complex fold raised**.
The driver counted a design as produced if any enabled reader measured it, so
20 working monomer folds masked 20 broken interface folds — the run reported
success for measuring none of what it was configured to measure. Fixed: a
design counts as produced only when every enabled reader measured it, and a
reader that produced nothing fails the run. `produced_by_reader` is now in the
task status so the split is visible rather than inferred.

## 9. Optional target MSA preparation

Run this explicitly **before** loading scoring configs or invoking Snakemake.
Set `target.msa` in the general YAML to the absolute path to publish. The scorer's
`model.use_target_msa` still determines whether that run consumes the alignment.
No search is submitted during config loading, DAG evaluation, or dry-runs.

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design

uv run bindocracy target prepare-msa \
  --general "$BINDER_ROOT/configs/general_config.yaml" \
  --script "$BINDER_ROOT/mosaic_setup/mosaic/singularity/msa-search.sbatch" \
  --image /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/msa.sif \
  --database /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db \
&& uv run snakemake --snakefile workflow/Snakefile \
  --configfile "$BINDER_ROOT/workflows/scoring_smoke_22.yaml" \
  --profile workflow/profiles/slurm
```

The script is Mosaic's existing import of ProtForge's local
`colabfold_search --gpu 1` invocation. It searches local databases through
`msa.sif`; it neither needs `mosaic.sif` nor submits sequences to a public server.
Only the target receives an MSA; de novo binders remain single-sequence.

- A matching existing A3M is reused, never overwritten. A wrong-target A3M is
  rejected. To deliberately rerun a search, choose a new `target.msa` path.
- A new search uses `sbatch --wait`, one GPU, and the campaign account and
  partition. Defaults are 8 CPUs, 64 GB, and eight hours; override with
  `--cpus`, `--memory-gb`, and `--walltime`.
- The command archives a normalized target FASTA and the imported script in a
  private `.OUTPUT.a3m-*/` directory beside the destination. Its `search.log`
  and `preparation.json` retain command, asset paths, submission result, and
  input/output hashes. Image and database contents are not hashed; version
  those assets separately.
- Publication is atomic and exclusive, after successful job completion and
  query validation. Failed, missing, wrong-target outputs or a target changed
  during the search are not published. Concurrent invocations share a lock.
- Inspect the search log for the launcher's query-only alignment warning:
  a successful search need not find homologs. Query identity is checked; MSA
  depth and biological quality are not guaranteed.
- Scoring subsequently records the A3M digest using its existing input
  manifest and checks it before launch.

This does not remove the existing AF2 complex-wrapper restriction:
`af2` still requires `use_target_msa: false`. The other five configured
backends accept the target alignment with the corrected Mosaic source;
`esmfold2` selects Full, not Fast.

For the MSA-routing fixes, rebuilding `mosaic.sif` is **not mandatory** when
`runtime.dev_source` points to the corrected host `src/` tree. The scorer
records that tree's Python-source digest. Preserve an immutable copy of the
tree for replay; a hash alone does not preserve its bytes. Rebuild to bake the
fix into a self-contained release, or when dependencies/system libraries
change. Target FASTA/A3M changes and host scoring-driver changes do not require
an image rebuild.

## 10. Deferred priorities before adding new losses

Recorded 2026-09-08 at campaign review. These are **next-work decisions**, not
claims that the corresponding measurements or commands exist. The current work
stages larger Protenix and Chai assets and clones DeepMind AF3; rebuilding Mosaic
and changing the scoring panel remain separate decisions.

| Order | Priority | Current gap and acceptance criterion |
|---|---|---|
| 1 | Preserve scored structures | `drivers/scorer/score_designs.py` stores scalar metrics, not the predicted complexes. Save a chain-labelled structure per retained design/condition/replicate, linked to its metrics and run. Retain residue mapping back to the campaign target and compare scored pose with designed pose. |
| 2 | Make reader support truthful | `epitope` and `inverse_folding` are accepted config flags but the launcher's reader list only contains `complex` and `monomer`. Reject unsupported flags before allocation until implemented; never accept an experiment that silently omits its requested measurements. |
| 3 | Measure epitope and physical interface geometry | Compute actual hotspot coverage, target/binder contacts, clashes and buried surface area from saved structures; assess pose consistency across samples/models. Conditioning on an epitope is not evidence of contacting it. Calibrate geometry thresholds to the construct rather than declaring a universal cutoff. |
| 4 | Protect a held-out evaluator | Record which models/checkpoints contributed generation and optimization losses. Select evaluation protocols before inspecting outcomes. A model used to tune the loss is not held out, and related model families are not independent votes. |
| 5 | Add controls and developability measurements | Compute charge, hydrophobicity, low complexity, sequence liabilities and exposed hydrophobic patches. Compare against composition-matched binder controls and relevant off-target/paralog counter-screens. Record sampling coverage/failures alongside aggregates so one surviving replicate is not mistaken for consistent support. Neural confidence, sequence plausibility and protein binding affinity are different quantities. |
| 6 | Select a diverse experimental panel | Cluster by sequence, binder fold and target-aligned binding pose/contact footprint. Select cluster representatives with explicit quotas, rather than filling a plate with near-identical top-scoring variants. Store cluster/selection decisions against the frozen candidate set. |
| 7 | Validate biological target context | Recheck cropped targets against the full construct, glycans, oligomer partners, membranes and required cofactors. Record inaccessible/forbidden surfaces and residue mappings. An exposed surface of an isolated cropped chain need not be accessible in the biological target. |
| 8 | Finish selection workflow commands | Scoring already runs through the generic Snakemake tool workflow. Expose design-set building, filter application, candidate export and diversity-aware selection; connect generation to a frozen evaluation set and recorded selection policy without ad hoc SQL scripts. |
| 9 | Close experimental feedback and asset provenance | Preserve stable design IDs through ordering and assay import. Link expression, solubility, enrichment and binding measurements, with units and assay conditions, back to computational protocols. Pin image and checkpoint manifests and preserve overlay source bytes; calibrate thresholds against experimental outcomes rather than silently changing historical decisions. |

Recommended next implementation: **structure export, truthful reader validation,
then epitope/physical-interface measurements**. Additional optimization losses
should wait until those outputs can reveal whether a score improvement is a
better binding hypothesis or merely an exploited model preference.

## 11. Stored ranking and cohort selection

[`examples/ranking.example.yaml`](../examples/ranking.example.yaml) defines a
global Chai mean-iPTM top-50 policy and shows per-generator recipe overrides.
The source must be an explicit frozen DesignSet; evaluator runs must be named
explicitly. The command does not generate or fold sequences.

```bash
uv run bindocracy rank apply /path/to/campaign.duckdb \
  --general /path/to/general.yaml --policy /path/to/ranking.yaml \
  --output-dir /path/to/rankings

# Use the printed ranking run ID, not an ambiguous reused policy name.
uv run bindocracy designset build /path/to/campaign.duckdb \
  --passed-filter top50 --filter-run RANK_RUN_ID --out-dir /path/to/selected
```

Each metric input declares its replica subset, aggregation (`mean`, `median`,
`min`, or `max`), and optional minimum coverage. By default every specified replica
must be finite and successful. This distinguishes a native aggregate at replica 0
from five independent stored replica rows. Per-input evaluator scopes can narrow,
but never widen, the top-level evaluator list.

Ranking is lexicographic, with explicit direction for each priority and ascending
design ID as the final tie-breaker. A priority may use the conservative minimum
of several named inputs after their individual aggregation. No weighted score or
cross-tool normalization is inferred.

- `group_by: global` ranks one population; `generator` ranks each producing tool
  independently and accepts `by_generator` recipe overrides.
- Exact-sequence deduplication retains the best-ranked eligible observation within
  each scope, not whichever database row happened to come first.
- Named `head` and/or `tail` cohorts never overlap. `shortage: error` refuses an
  undersized eligible population; explicit `truncate` fills head first, then tail
  from remaining candidates. Missing, failed, nonfinite, or ambiguous measurements
  are excluded with reasons, never classified as the worst designs.
- Native-pass gating is not implicit. Freeze that eligibility first with
  `designset build --passed-filter NAME --filter-run RUN_ID`. The source run may
  be a native generation/evaluation run, a filter run, or a rank run, provided it
  actually carries filter decisions. Ambiguous names remain errors.

The ranking run stores scoped `rank` decisions and `filter` membership decisions
for each cohort and their union (default name `selected`). It creates no new
designs and does not alter measurements or earlier verdicts. Resolved policy,
source membership, and consumed metrics identify the run; identical reapplication
is a no-op, while a changed metric snapshot or policy produces a separate run.

The content-addressed output directory contains `collected.json`, `config.json`
(including the frozen source and resolved policy), `summary.txt`, and per-cohort
JSON/FASTA DesignSets. Empty cohorts are explicitly empty, not a fallback to the
whole source; the existing `designset build` command still refuses empty queries.
Use the selected manifest in a scoring/optimization config, then review a
[frozen campaign plan](../workflow/README.md) before submitting any compute.
