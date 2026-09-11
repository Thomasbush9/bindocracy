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

`epitope` and `sequence` are built-in functions that go through this same
runner. That is deliberate — the extension point is exercised by the harness
itself rather than merely offered to others, so the contract is proven by use.

`sequence` is also the negative-control arm: on the labelled Nipah-G set,
length alone reaches AUC 0.642 and five of nine scorers sit within 0.06 of it.
