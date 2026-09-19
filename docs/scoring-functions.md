# Scoring functions

A **condition** is something that gets folded. A **function** is something
computed from the result. They were one config block until 2026-09-11, which
put three different cost classes behind one name:

| | what it costs |
|---|---|
| `readers.complex`, `readers.monomer` | a GPU pass per design, per model |
| `functions.epitope` | geometry over a pose already on disk — **free** |
| `functions.sequence` | sequence only — **free** |
| `functions.inverse_folding` | loads a second model |

That conflation is not a tidiness complaint. `epitope` and `inverse_folding`
were accepted by the config, validated, launched, and then dropped by the
launcher, which only ever emitted `complex` and `monomer`. A run asking for
them **succeeded and measured nothing**, with no error anywhere. They are now
refused by preflight, with a message saying where they went.

The practical consequence of the split: a function reads what the scorer
already produced, so it can be added later and **backfilled across every
structure ever saved**, on a CPU, in minutes. Adding a folding model means
re-folding. These are different enough to deserve different paths.

## Adding one

Three tiers, cheapest first.

### Tier 3 — a custom function (no code in this repository)

A script that reads a JSONL of designs and writes a JSONL of metrics. Any
language, any container, or none. No plugin, no `register()` line, nothing
under `src/`.

```yaml
functions:
  sequence: true            # built-in
  custom:
    - name: sasa
      script: /abs/path/to/buried_sasa.py
      container: images/mosaic.sif       # optional; absent = the harness venv
      inputs: [structure, sequence]
      metrics:
        buried_sasa:  {direction: max, unit: angstrom^2}
        n_hbonds:     {direction: max}
```

The script is invoked as:

```
<python> your_script.py --inputs IN.jsonl --outputs OUT.jsonl [your args...]
```

**Input**, one line per design — only the fields you asked for in `inputs`:

```json
{"index": 0, "design_id": "…", "sequence": "MKT…", "structure": "/abs/pose.pdb"}
```

**Output**, one line per design:

```json
{"index": 0, "metrics": {"buried_sasa": 812.4, "n_hbonds": 7}}
{"index": 1, "failed": "no interface found"}
```

`tests/fixtures/functions/net_charge.py` is a complete working example, and it
imports nothing from this project.

### Tier 2 — a mosaic-backed model

A value in the scorer's `model` field, not a new plugin. Roughly six edit sites
across `tools/scorer/config.py` and `drivers/scorer/score_designs.py`: the
literal, the two capability tables, the loader branch, the CLI choices, and
whichever quirk table the model turns out to need.

### Tier 1 — a new container

A plugin, the Chai-1 / AlphaFold 3 / OpenFold3 pattern. Six files and one
`register()` line, exactly as `adding-a-tool.md` describes.

### Not a function at all — changing the design

A function computes a number *about* a design. Something that produces a new
design is an optimizer, which is a different contract: n→m rather than 1:1,
keyed on `(parent_index, child)`, and writing to the `designs` table rather
than to `metrics`. See [custom-optimization.md](custom-optimization.md).

## Four rules the contract enforces

**Declare the direction.** A custom metric has no default direction and the
config will not validate without one. This is the single rule worth being
strict about: a number stored the wrong way round sorts backwards and nothing
in the row says so. The person who wrote the script is the only one who knows
which way it points.

**You cannot redefine a registered metric.** `iptm` means one thing. A custom
function declaring it is refused — the same reason `protenix_mini` and
`protenix_base` are separate columns, and mosaic's OF3 and the upstream one.

**Return only what you declared.** A metric that appears in the output but not
the config aborts the run rather than being stored with an unknown meaning.

**Your bytes are hashed and archived.** The script's sha256 goes into every
metric row's `details`, and a copy is kept beside the output. A run records
which bytes scored it, not which path they were read from, so editing a script
between two runs cannot make them look comparable.

## What a function does not get

A `design_id` — it keys on `index`, the design-set position, for the same
reason the design-set FASTA carries an index rather than an ID. The join back
to a design happens host-side.

A design without a required input is **counted, not dropped**: a function
asking for `structure` cannot score a design that was never folded, and that
is a fact worth recording rather than a silent omission.

## Built-ins

`epitope` and `sequence` go through this same runner, as the same type, invoked
the same way. That is deliberate: the extension point is exercised by the
harness itself rather than merely offered to others, so the contract is proven
by use rather than by assertion. They are also the two worked examples — copy
`drivers/functions/sequence_metrics.py` and change the middle.

A built-in declares no metrics, because its metrics are already in the registry
with a fixed meaning. A custom function must declare, because nobody else knows
what its numbers mean.

### `sequence` — the negative-control arm

Seven sequence-only properties: length, net charge, molecular weight,
hydrophobic fraction, cysteines, N-linked sequons, longest homopolymer run.
No structure, no model, no GPU.

These exist because they are the bar. On the labelled Nipah-G set, **binder
length alone separates binders from non-binders at AUC 0.642**, and five of the
nine folding models score within 0.06 of that. A confidence metric that does
not beat these has not earned its GPU time — and until they are stored beside
every real metric, that comparison cannot be made without re-deriving them.

Net charge deliberately excludes histidine, matching the benchmark's own
control, so a number here and a number there are the same quantity.

### `epitope` — does the binder touch what it was aimed at?

Four metrics from heavy-atom contacts at a 5 Å cutoff: coverage, contact count,
interface size, and mean distance to the nearest hotspot.

Two things about the inputs were not obvious, and both were wrong on the first
try — caught by running it against a real pose rather than reading a format:

**Hotspots are 1-based positions in the target's FASTA** — not author
numbering, and no chain letter. Chain letters are not portable: the mosaic
driver builds `[binder, target]`, so its target is chain B, while the Chai-1
and AlphaFold 3 drivers write the target as chain A. `A110` names the binder in
one and the target in the other. And residue ids in a predicted pose are
positional and 0-based (`0..200` for a 201-residue target), not the numbering a
crystal structure carries. The caller resolves author numbering against
`target.structure_pdb` — which the harness already does elsewhere — and passes
positions.

**The binder is identified by length**, not by chain letter, for the same
reason.

Metrics are prefixed by **the model that produced the pose**, not by the
function: `boltz2_epitope_coverage`, `chai1_epitope_coverage`. Two models
disagree about where the binder sits by a median 22.7 Å, and one unprefixed
column would hide exactly that.

A hotspot past the end of the target raises rather than scoring 0.0 —
out-of-range numbering is a mistake, and reporting it as a miss would read as a
real measurement. Coverage with no hotspots named is NaN, which the driver
drops so no row is stored, rather than a 0.0 claiming the binder missed an
epitope nobody specified.

### First result

Run against the three DIO3 designs already folded by eight models, with the
campaign's own hotspots (`A110, A112, A131`, which map to FASTA positions
110/112/131 — the target PDB is numbered 1..201, so the mapping is the
identity here):

**24 model-design pairs. Coverage above zero in two of them**, both Promera,
and both a single hotspot of three. The other twenty-two place the binder
13–30 Å from the nearest hotspot while forming a real interface of 10–36
target residues somewhere else.

So the models disagree with each other about where the binder sits — by a
median 22.7 Å — and agree that it is not on the epitope. That independently
corroborates what run 19 found: the tool that enforces an epitope was the one
that appeared to miss it. Conditioning on an epitope is not evidence of
contacting one.

Read it as three arbitrary designs from one generator, not as a verdict on
epitope conditioning in general. It is the measurement the harness could not
make until structures were kept.
