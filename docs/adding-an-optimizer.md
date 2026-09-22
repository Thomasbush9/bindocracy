# Adding a custom optimizer

How to get from "I have an idea for improving designs" to children in the
database, in the order the work actually happens.

[custom-optimization.md](custom-optimization.md) is the **contract**: the exact
rows, the six rejection rules, why `loss_models` exists. Read it once. This
page is the **procedure**, and it assumes you have.

The unit you build is a **kit**: a directory holding a manifest and an
entrypoint. Everything true of the optimizer wherever it runs goes in the
manifest; everything true only of your machine stays in the campaign config.
That line is the whole design, and the test for which side a field falls on is:
*would a second site have to change it?*

```
mosaic-af2-refine/
|-- kit.yaml        what it is, what it needs, what it costs
`-- optimizer.py    the entrypoint
```

Two worked kits ship under `examples/kits/`. Copy the nearer one.

---

## 1. Write the entrypoint

Start from `tests/fixtures/optimize/point_mutate.py`. It is the shortest thing
that exercises every part of the contract and it is covered by the test suite,
so it is the one example that cannot silently rot.

```python
from bindocracy_io import RejectCandidate, run_optimization

def optimize(parent, context, args):
    ...
    yield {"sequence": child, "metrics": {"loss": value}}

if __name__ == "__main__":
    run_optimization(optimize)
```

The helper owns the transport: it parses `--inputs`, `--outputs` and
`--context`, loads the context once, validates that every metric is a finite
number, and **attaches `parent_index` and the child ordinal itself**. You
cannot name a parent outside your shard and cannot emit a duplicate
`(parent_index, child)` pair, because you never write either.

Three things are still yours, and each is a place scripts go wrong.

**Distinguish a refusal from a bug.** Raise `RejectCandidate` when *this
parent* cannot be optimized and the run should carry on. Let everything else
propagate. Catching broadly is tempting and it is how a broken optimizer comes
to look like a biological result: a version of `mosaic-af2-refine` with a
two-line unpacking bug wrapped every parent in `except Exception`, wrote four
tidy refusal rows, and produced a run that read as "AF2 declined to improve
these binders" when it had never run a gradient step. Keep the caught set
narrow enough to name, one clause per reason.

**Report `start_loss` in the same units as `loss`.** A final loss means
nothing without the number it started from, and only if the two subtract. The
same optimizer once reported `start_loss` around 12 beside a `loss` around
-1.3, because one was the nine-term training objective and the other the
ranking objective. Both numbers were correct and their difference was
meaningless.

**Write artifacts relative to `context["structure_dir"]`.** An absolute path is
rejected so the run directory can be moved. `known-issues.md` records that
having already cost this campaign a day.

A script that reads and writes the JSONL rows by hand is still valid and always
will be, which is what lets a kit wrap a binary or a script in another
language. `examples/kits/surface-tuner/` is kept on that path on purpose, so
the raw contract keeps a user. Use the helper unless you cannot.

---

## 2. Write the manifest

```yaml
kit_schema: 1
name: mosaic-af2-refine
version: 0.3.0
kind: optimizer
entrypoint: optimizer.py

requires: [...]        # see below
inputs: [sequence]
loss_models: [af2]     # proposed; the campaign restates it
capabilities: {changes_length: false, max_children: 1}
metrics:
  loss: {direction: min, description: ...}
cost: {compile_seconds: 172, per_parent_seconds: 19.4}
args: {soft-steps: 12}
```

**Metric directions belong here, not in the campaign config.** You are the only
person who knows which way your numbers point, and a direction stored backwards
sorts backwards with nothing in the row to say so. Until the harness reads
`kit.yaml`, the campaign config restates them and `check_kit.py` fails if the
two disagree.

**`loss_models` is proposed here and restated there.** The field exists so a
later selection can exclude the models that already had a say in a design. A
value inherited from a file somebody else wrote is a weaker claim than one the
operator affirmed, so the campaign has to say it again. `[]` is a real claim
that the loss saw no structure predictor at all, not an unfilled default.

**Declare a cost model, never a walltime.** Two measured constants: what a
process pays once, and what it pays per parent. The campaign multiplies them by
the parent count and the shard count, neither of which you can see. A four-hour
reservation is right for four parents and wrong for four hundred; only the
campaign knows which it has.

### Dependencies

Name what you need. Do not name where it lives.

```yaml
requires:
  - name: mosaic
    kind: container
    version: ">=2026.08.18"
    verify: {import: mosaic.models.af2_msa}
    why: AF2 multimer with a per-chain MSA.
```

`verify` decides; `version` is only a hint for finding a candidate. That order
is deliberate. `mosaic.sif` carries no usable version, and the thing that
actually mattered here, whether AF2 could take a per-chain alignment, was a
capability no tag expressed. An import that either works or does not is a
stronger check than a number somebody remembered to bump.

| `kind` | verified by | supplied by |
|---|---|---|
| `container` | an import that runs inside it | the campaign, or a store |
| `weights` | files that must exist under it | the campaign, or a store |
| `source_overlay` | files under the tree, bound over the image | the campaign |
| `harness_module` | a file in the bindocracy checkout | the harness itself |

Mark a requirement `optional: true` when it only patches around a stale
dependency, and `provided_by: harness` when a store should never be asked for
it. `bindocracy-io` is the latter: if it is missing, the harness is out of
date, and fetching a copy would paper over that.

---

## 3. Bind the dependencies on your machine

One file per site, campaign-owned. `examples/kits/bindings.example.yaml` is a
template.

```yaml
mosaic:
  kind: container
  path: /absolute/path/to/images/mosaic.sif
  interpreter: python
alphafold-params:
  kind: weights
  path: /absolute/path/to/weights/alphafold
```

Nothing here is portable, and nothing here belongs in a kit.

---

## 4. Check it before you queue anything

```bash
python scripts/check_kit.py --kit examples/kits/mosaic-af2-refine \
    --bindings /path/to/bindings.yaml \
    --config /path/to/configs/optimize/af2_refine.yaml
```

Three questions, a second, no GPU. Is the manifest well formed? Is everything
it needs installed here, verified rather than assumed? Has the campaign config
drifted from the manifest? A requirement with no binding is reported as
something a store could supply, not as a crash.

It also prints the walltime its cost model implies for this design set and
shard count, next to what the config reserved.

Then check the script against the contract, which drives the real driver and
applies the real validation:

```bash
python scripts/validate_optimizer.py --script examples/kits/<kit>/optimizer.py \
    --design-set sets/<digest>.json --target-fasta target.fasta \
    --declare loss:min --max-children 1
```

Give the entrypoint a `--dry-run` that returns plausible children without
loading a model. Importing AF2 and five sets of multimer parameters costs
minutes and proves nothing about the row shapes, which is all the validator
checks.

---

## 5. Deploy, configure, run

Copy the kit beside your campaign data and point the config at its entrypoint.
The repository holds templates; a campaign holds the copy it runs, the way
`mosaic_refine.py` has always worked. A store will one day replace the copy
step, and nothing above changes when it does.

```yaml
name: af2refine            # also the metric prefix: loss -> af2refine_loss
tool: optimize
design_set: /abs/sets/<digest>.json
script: /abs/optimizers/mosaic-af2-refine/optimizer.py
loss_models: [af2]         # restated, never inherited
metrics: {...}             # restated until the harness reads kit.yaml
resources: {gpus: 1, cpus: 16, memory_gb: 96, walltime: "04:00:00"}
```

Then the ordinary path: `config load`, snakemake, `collect`, `ingest`. The run
records your script's sha256 and archives its bytes beside the driver, so a run
is replayable from its own directory.

**Score the children with a model that is not in `loss_models`.** That is the
one rule that survives everything else on this page.

---

## Things that will bite you

**Editing a config re-plans every run in the same index.** The model YAML is a
declared input to `prepare_run`, so changing it re-runs preparation for every
entry that points at it, mints new run IDs over finished run directories, and
resubmits their jobs. Give a new run its own index entry, or expect to cancel.

**A dry run proves the rows, not the science.** Both bugs that cost this
campaign a GPU job passed every dry run: one was an arity mismatch in a mosaic
call, the other a units mismatch between two metrics. Run two or three parents
before a whole set, and read `start_loss` against `loss` when they land.

**Compilation is paid per binder length, not per task.** A shard spanning many
lengths pays it again at each one. The design set is ordered by length and
shards are contiguous precisely so it is paid as rarely as possible; a cost
model that ignores this will under-reserve a mixed shard.
