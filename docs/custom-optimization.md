# Custom optimization

Improve designs the campaign already has, with a script the campaign does not
have to know about.

An optimization run is the first thing in this harness that is **both** shapes
at once: it reads a frozen design set like a scorer, and it writes designs like
a generator. That is the whole reason it is a separate contract from
[scoring functions](scoring-functions.md) rather than a reuse of one.

|  | scoring function | optimizer |
|---|---|---|
| reads | one design | one design |
| writes | numbers about it | **new designs** |
| cardinality | 1:1 on `index` | **n→m** on `(parent_index, child)` |
| run-level inputs | repeated per row | one `context.json` |
| worst case if wrong | a column a query can ignore | sequences in `designs` forever |

That last row is why this contract is stricter. A bad custom scoring function
adds a useless number. A bad custom optimizer puts sequences into the campaign
that look exactly like designs a model produced, permanently, and nothing
downstream can tell them apart.

---

## The whole loop

```bash
# 1. Score the candidates (already the case for most of them)
#    -> metrics land in the database

# 2. Decide what is worth the GPU time. A stored, versioned policy.
bindocracy filter metrics campaign.duckdb            # what can be tested
bindocracy filter apply campaign.duckdb \
    --general general.yaml --filter filter.yaml --output-dir runs/filter-01

# 3. Freeze what it chose into a content-addressed set
bindocracy designset build campaign.duckdb --out-dir sets/ \
    --passed-filter worth_optimizing --filter-run <run_id>

# 4. Check your script against the contract. No GPU, seconds.
python scripts/validate_optimizer.py --script my_optimizer.py \
    --design-set sets/<digest>.json --target-fasta target.fasta \
    --declare loss:min --max-children 4

# 5. Run it, then collect and ingest as with any tool
bindocracy config load campaign.duckdb --general general.yaml --model optimize.yaml
# ... snakemake ...
bindocracy collect runs/<run>/run.json --output runs/<run>/collected.json
bindocracy ingest campaign.duckdb runs/<run>/collected.json

# 6. The children are now designs with parents, ready to be scored
#    BY A MODEL THAT WAS NOT IN loss_models
```

Step 3 names the filter **run**, not just the rule. Re-filtering under changed
thresholds writes a second filter run *beside* the first rather than replacing
it, so both carry the config's name — and selecting on a name two runs share
would union two contradictory policies, which silently gives you the looser
one. `designset build` refuses an ambiguous name and prints the run IDs.

---

## What your script is invoked as

```
<python> your_script.py --inputs IN.jsonl --outputs OUT.jsonl --context CTX.json [your args]
```

Those three always come first, in that order. Anything in `args:` in the YAML
follows.

`<python>` is the interpreter inside your container (`runtime.container_python`),
or the harness's own if you declared no container.

### `--context CTX.json` — written once

Everything that does not vary per design. It is a separate file rather than a
field on every row because an optimizer needs the target, its alignment, the
epitope, a seed and an output directory, and a hundred copies of an a3m path is
an invitation to read one parent's copy and apply it to another.

```json
{
  "target_sequence": "MA...",
  "target_chain": "A",
  "target_msa": "/abs/target.a3m",
  "target_structure": "/abs/target.pdb",
  "hotspots": [110, 112, 131],
  "seed": 42,
  "max_children": 4,
  "length_delta": 0,
  "work_dir": "/abs/runs/<run>/task-0000/work",
  "structure_dir": "/abs/runs/<run>/task-0000/structures",
  "shard": 0,
  "num_shards": 8,
  "n_parents": 12
}
```

**`hotspots` are 1-based positions in the target's FASTA.** Not author
numbering, not a chain letter. mosaic wants 0-based, so subtract one:

```python
epitope_idx = [spot - 1 for spot in context["hotspots"]] or None
```

This convention was settled the hard way by the epitope scoring function, and
it is resolved and range-checked when the run is *planned*, not in the
container: residue ids in a predicted pose are positional (`0..200` for a
201-residue target) and carry none of the numbering a crystal structure has,
and the drivers here do not agree on which chain is the target. See
[scoring-functions.md](scoring-functions.md).

### `--inputs IN.jsonl` — one line per parent

Only the fields you asked for in `inputs:`. `index` and `length` are always
there.

```json
{"index": 0, "length": 62, "sequence": "MKT...",
 "structure": "/abs/poses/boltz2/000000.pdb", "source_model": "boltz2",
 "metrics": {"boltz2_iptm": 0.81}, "tool": "mosaic", "native_id": "..."}
```

**`index` is shard-local and starts at 0 in every task.** It is not a design
id, and it is not a position in the whole set. Echo it back as
`parent_index`; the driver translates it.

| `inputs:` value | what you get |
|---|---|
| `sequence` | the parent's amino-acid sequence |
| `structure` | `structure` + `source_model`; needs `structures_from:` |
| `metrics` | `metrics`, the names in `metric_inputs:`, averaged over replicates |
| `provenance` | `tool`, `run_name`, `native_id` |

Ask for the least you need. A run that does not ask for `structure` can be
planned over designs nothing has folded; one that does is refused at plan time
if any parent has no pose (or set `allow_unfolded: true`).

### `--outputs OUT.jsonl` — one line per child

```json
{"parent_index": 0, "child": 0, "sequence": "MKTVL...",
 "metrics": {"loss": 0.31, "n_steps": 200},
 "structure": "poses/0-0.pdb", "trajectory": "traj/0.jsonl", "seconds": 84.2}
```

and for a parent you tried and could not improve:

```json
{"parent_index": 3, "failed": "loss did not decrease in 200 steps"}
```

| field | rule |
|---|---|
| `parent_index` | required; the shard-local `index` you were handed |
| `child` | ordinal within that parent, `0..max_children-1`. Default 0 |
| `sequence` | the 20 canonical amino acids only |
| `metrics` | numbers, and only keys declared in the YAML |
| `structure`, `trajectory` | **relative to `context["structure_dir"]`** |
| `seconds` | optional, recorded |
| `failed` | a string; makes the row a refusal and needs no sequence |

Emit nothing for a parent you chose not to touch. Emit `failed` for one you
tried. The difference is recorded — `n_parents_untouched` against `n_failed` —
because absence and refusal look the same in a query and mean opposite things.

---

## Six rules, and why each one is refused rather than tidied

A bad row is **rejected and counted**; it does not abort the file. One bad
sequence out of five hundred children is a bug in one branch, and throwing away
the other 499 would cost a GPU-day to punish a typo. What never happens is the
third option — storing it anyway.

1. **`parent_index` must be in the shard.** Otherwise the child attaches to a
   different design, and a design table where some children have the wrong
   parent is worse than one missing a few.
2. **`(parent_index, child)` must be unique.** Otherwise one of the two is
   lost and which one depends on file order.
3. **`child` below `max_children`.** Keyed on the ordinal rather than arrival
   order, so "excess" does not depend on how the file was written. Refused
   rather than truncated, because a runaway loop and a productive optimizer
   look identical after truncation.
4. **Sequences use the 20 canonical amino acids.** An `X` is a position the
   optimizer failed to fill, `B`/`Z`/`J` are ambiguity codes meaning it did not
   decide, and `U`/`O` cannot be ordered by the ordinary route.
5. **Every metric is declared, with a direction.** A number stored the wrong
   way round sorts backwards and nothing in the row says so, and only whoever
   wrote the script knows which way it points.
6. **Structure paths are relative.** An absolute path does not survive the run
   directory being moved — which [known-issues.md](known-issues.md) records as
   having already cost this campaign a day.

Your script's **sha256 goes into the run**, so editing it between two runs
cannot make them look comparable. The raw file your script wrote is kept beside
the normalized one, so any disagreement between what you wrote and what was
stored can be settled by reading both.

---

## `loss_models` — the field that matters most

```yaml
loss_models: [boltz2]   # or [] to claim the loss saw no folding model
```

Required, and not for bookkeeping.

If an optimizer drives a design against Boltz-2 ipTM and the children are later
ranked by Boltz-2 ipTM, **the ranking measures the optimizer, not the binder.**
Nothing in the database could detect that after the fact, because until this
field existed no run recorded which models fed a loss. It is stored on the run
*and* on every child's metadata, because a design outlives the query that found
it.

So the rule for what comes next: **score the children with a model that is not
in `loss_models`.** The panel has nine, and on the Nipah-G benchmark Chai-1
(0.828) is level with ESMFold2 — which is the reason it was worth adding, since
ESMFold2 is the intended held-out judge and had no alternate.

`[]` is a claim, not a default. It says the loss consulted no structure
predictor at all — a charge optimizer, a motif remover, a language model. Write
it explicitly. A name that is not a model this campaign scores with is refused
at preflight, because a misspelling would exclude nothing later while looking
like it had.

---

## What gets stored

One `DesignRecord` per accepted child, with:

- `parent_design_id` — the lineage, so "which designs are original" is a query
- `native_id` = `<parent native_id>.<optimizer name><child>`, so the lineage is
  legible in a FASTA without a join, and so re-collecting the same run produces
  the same `design_id` and ingestion stays idempotent
- `metadata.loss_models` — which models already had a say
- `metadata.n_substitutions` and `metadata.length_delta` — how far the child
  moved. A child identical to its parent is a real result (the optimizer found
  nothing) and is counted as `n_unchanged` rather than discovered later as a
  duplicate sequence

Metrics land as `<optimizer name>_<metric>` — `refine_charge_loss`, not `loss`.
The prefix is the optimizer's name for the same reason `protenix_mini` and
`protenix_base` are separate columns: `<name>_loss` is the optimizer's *opinion*
of a child, never a measurement of it, and the two must not be joinable by
accident. A declared metric may not reuse a registered name, so an optimizer
reporting its internal ipTM estimate must call it something else (`opt_iptm`).

Structures and trajectories become `artifacts` rows joined to the child, so a
later geometric metric is a read rather than a re-fold.

---

## Check it before you queue it

```bash
python scripts/validate_optimizer.py \
    --script my_optimizer.py \
    --sequences MKTVLIWAFG... KVFGRCELAA... \
    --target-fasta target.fasta \
    --declare loss:min --declare n_steps:none \
    --max-children 2
```

It drives the **same driver** the cluster runs and applies the **same
validation** the adapter applies, so a script that passes here fails on the
cluster only for reasons that are actually about the cluster. It prints what
would be stored, and every rejected row with the reason and a hint.

`--design-set sets/<digest>.json` instead of `--sequences` runs it over the
real frozen set, which is what the run will see.

---

## Worked examples

**`tests/fixtures/optimize/point_mutate.py`** — complete, minimal, imports
nothing from this project, and covered by the test suite. Not a useful
optimizer; it is the shortest thing that exercises every part of the contract
(n→m children, a declared metric, a failure row, a written trajectory, a parent
it declines to touch). Copy it and replace `optimize()`.

**`examples/optimizers/mosaic_refine.py`** — the real thing, against mosaic.
The difference from the hallucination driver is one line:

```python
# hallucinate: start from noise, invent a binder
_pssm = uniform(0.25, 0.75) * gumbel(key, (binder_length, 19))

# refine: start from the PARENT, walk away from a sequence somebody has
start = one_hot(parent) * (1 - epsilon) + epsilon / len(alphabet)
```

Its contract plumbing is exercised by the validator; its mosaic calls are
copied from the hallucination driver this campaign has run. **Nobody has run
this file on a node yet**, so treat the loss weights and the schedule as a
starting point, and run two or three parents before a whole set.

---

## Things to think about before a real run

**Diversity stops being optional.** Optimization pulls a population toward one
motif. Clustering is reserved in the metric registry and nothing computes it,
so round two will happily rank forty near-identical variants and the campaign
will learn one thing instead of forty.

**Length changes cost the next stage.** `length_delta: 0` keeps children the
same length as their parents. Non-zero is fine and sometimes the point, but
every JAX scorer recompiles per binder length, and the design set is ordered by
length precisely to keep that cost down.

**Keep the trajectory.** A per-parent JSONL of loss-per-step is nearly free and
is the only way to tell "converged" from "wandered" after the fact. Report
`start_loss` beside `loss` for the same reason: a final loss of 0.31 means
nothing without knowing it began at 0.33.

**One parent must not kill the shard.** Catch per-parent, write a `failed` row,
continue. Optimization cost is variable in a way scoring cost is not.

**Round depth.** Optimizing children gives a parent chain, which is what you
want, but the generation number is not stored as a column — it is reconstructed
by walking `parent_design_id`. Worth adding if rounds go deep.
