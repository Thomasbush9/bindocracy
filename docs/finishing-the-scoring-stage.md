# Finishing the scoring stage

**Written 2026-09-10.** What is done, what is left, and the order to do it in.
Ordered by what makes the numbers mean something, not by effort.

## Where it stands

Nine scorers run, across four containers, on one seam: a frozen design set goes
in, a `metrics.jsonl` comes out, one adapter reads all of them.

| | |
|---|---|
| scorers wired and GPU-verified | 9 |
| plugins | 5 (`scorer`, `chai1`, `af3`, `of3_upstream`, `optimize`) |
| metric specs registered | 35 |
| structures saved and pointed at from the DB | yes |
| selection wired (`designset build`, `filter apply`) | yes, 2026-09-12 |
| tests | 716 |
| written to `campaign.duckdb` | **nothing yet** |

**Update 2026-09-12.** Items 2 and 8 below are done, and item 3's first half
(the sequence-only control arm) landed with the scoring functions. The
selection commands exist because optimization needed them: `docs/custom-optimization.md`
is the stage that consumes a filter's verdicts.

Two measurements exist for the panel, and they disagree usefully:
`docs/benchmark-nipah.md` has the AUCs and the GFP control folds.

---

## 1. Replace OpenFold3 with its own implementation · **done**

The case is made in `benchmark-nipah.md`: mosaic's OF3 fails both tests
independently — 0.597 on the labelled set, below the sequence-only control bar,
and 24 Å from consensus on GFP — and no parameter reaches it.

Cheapest of the three replacements because the expensive half exists:
`openfold3.sif` (official `openfoldconsortium/openfold3:stable` v0.4,
Apache-2.0) and the original `of3-p2-145k.pt` / `of3-p2-155k.pt` checkpoints
are already on disk.

Critically, upstream OF3 **can take our alignment**: the query schema has
`main_msa_file_paths` per chain, so it folds against the campaign a3m with
`--use-msa-server false`. Without that this would have reintroduced the
confound the MSA work removed.

Stored as `of3_upstream_*`, never as `of3_*`. They are different models and one
column holding both would silently average two implementations.

**Done.** Plugin `of3_upstream` (config, preflight, launch, adapter, driver),
fourth on the shared seam, 14 tests. On GFP it returns pLDDT 88.7 at 3.9 Å from
consensus against mosaic's 38.5 at 24.0 Å — the two differ from each other by
24.6 Å. `of3` is deprecated: refused for new runs, still loadable from archived
manifests so historical runs stay relaunchable.

**Still outstanding:** `of3_upstream` has not been scored on the Nipah-G 434, so
the panel's AUC table still carries mosaic's 0.597 for OpenFold3. That number
now has a known cause and should be replaced rather than trusted.

## 2. Make `readers.epitope` and `readers.inverse_folding` honest · **done**

Both are now refused by preflight with a message naming where they went, and
`epitope` is implemented as a scoring function over poses already on disk. See
`docs/scoring-functions.md`, including what it measured on its first run.

## 3. Negative controls

The largest gap. Every design scored so far was meant to bind, so no threshold
here has a false-positive rate. `adapters/scoring.py` already reserves the seven
sequence-only specs; nothing computes them.

Two pieces:

- **Sequence metrics** — charge, length, hydrophobic fraction, cysteines,
  glycosylation sequons, low-complexity runs. No GPU. These are also the
  control bar: on Nipah-G, length alone reaches 0.642 and five of nine scorers
  sit within 0.06 of it.
- **Decoys** — the same binders against an unrelated target, and scrambled
  binders against the real one. This is the piece that turns "ipTM 0.8" into a
  claim with a denominator.

## 4. Epitope geometry

Now unblocked: structures are saved and pointed at from `artifacts`, so this is
a read rather than a re-fold. `EPITOPE_METRICS` reserves four specs —
coverage, contact count, interface size, offset — and biotite is installed.

Run 19's finding was that the tool enforcing the epitope was the one that
missed it. Nothing in the harness has been able to check that claim
independently until now.

## 5. Protect the judge

ESMFold2 is meant to be held out, and two things need settling before it
arbitrates anything:

- **It must not gate a filter upstream.** Nothing records which models
  contributed to generation or optimisation losses, so this cannot currently be
  audited.
- **Its GFP result deserves an explanation.** 0.833 on the benchmark — second
  best — and pLDDT 42 on a beta barrel it should find trivial, plateauing at
  0.59 with five times the trunk budget. Excellent at the task, weak on the
  control. One direct comparison against upstream ESMFold2 settles whether that
  is the model or mosaic's port.

## 6. Diversity

Cluster by sequence, by fold, and by target-aligned contact footprint. Without
it, ranking returns a plate of near-identical variants and the campaign learns
one thing instead of a hundred. Reserved in the registry; nothing computes it.

## 7. Rebuild `mosaic.sif`

Still blocked on the same prerequisite: **ten files under
`mosaic_setup/mosaic/src` are uncommitted**, and they are the MSA routing
itself plus the `tinyprot` bind. Building today stamps a commit hash onto an
image that commit does not describe. Full proposal in `mosaic-rebuild.md`,
including a `%test` that fails the build if the routing is dead — the defect
the current image shipped with silently.

Retires `runtime.dev_source` and unblocks AF2-with-MSA in the image rather than
only in the bind.

## 8. Selection workflow · **done for the parts optimization needs**

`bindocracy designset build` freezes a query into a content-addressed set;
`bindocracy filter apply` applies a stored policy and writes verdicts;
`bindocracy filter metrics` lists what a policy may test. The filter run's
`run_id` is derived from the policy, the set and the evaluator runs, so
re-applying the same policy is a no-op rather than a second set of identical
verdicts — a filter has no run directory to key restart-safety on.

Two refusals came out of building it, both for the same reason. `passed_filter`
requires `filter_runs`, and a filter run *name* is refused when two runs share
it: re-filtering under changed thresholds writes a second run beside the first,
and the union of a strict filter and a loose one is the loose one.

Still open here: candidate export, and diversity-aware selection (item 6).

## 9. Ingest into the real campaign

Everything so far lives in `scoring_test.duckdb`. Moving to `campaign.duckdb`
should happen once the readers are honest (2) and the controls exist (3) —
scoring 3,302 designs into the campaign before either is spending GPU hours on
numbers whose meaning is not yet established.

---

## What I would not do next

**Add more models.** Nine is already more than the evidence supports: five of
them sit within 0.06 of what net charge and length achieve with no GPU at all.
A tenth scorer does not fix that; controls do.

**Run the full 3,302-design panel.** At six samples across the panel this is
tens of GPU-hours, and until the controls exist it produces a ranking with no
false-positive rate attached. The cascade exists for evidential hygiene, not
for GPU savings — score deeply once the thresholds mean something.
