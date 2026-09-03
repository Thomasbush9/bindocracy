# The first full comparison run, and what it cost to get there

**Run 19, 2026-09-01.** One target, one epitope, seven tools, two jobs each,
50 designs asked of every tool. The first time every registered tool in the
harness designed against the same thing at the same time.

- Target: chain A of `outputs/_shared/dio3_cut.pdb`, 201 residues.
- Epitope: **A110, A112, A131** — the solvent-exposed rim of the DIO3
  catalytic pocket. All three sit below the median burial of the chain and
  span 7–16 Å CA–CA. They are the residues the Genie 3 problem set was already
  built around (as B110/B112/B131 after its renumbering), so using them is what
  makes this one comparison rather than seven.
- Index: `workflows/full_run_19.yaml` (six tools) and
  `workflows/full_run_19_mosaic.yaml` (Mosaic, added after the fix below).
- Database: `campaign.duckdb`.

Configs live beside the campaign data, not here:
`configs/general_config.epitope.yaml` and `configs/<tool>/*_run19.yaml`.

**Scope: this run tests the tools, not the binders.** Fifty designs is nowhere
near a design campaign — PXDesign alone needs 10,000+ for 10–100 dual-filter
passes — and nothing here was meant to be ordered. Every number below is a
diagnostic of what a tool did, not a measure of design quality, and a low pass
rate is not a finding on its own. What is worth reading is where a tool
reported something it did not measure, silently ignored what it was told, or
returned something that is not a design at all.

---

## What did not work

### 1. Mosaic refused the epitope — **fixed, then run**

`preflight_mosaic` raised on any campaign naming hotspots. The refusal was
deliberate and it named its own fix: `sp.BinderTargetContact` takes an
`epitope_idx` and `drivers/mosaic/hallucinate_binders.py` built it without one,
so a Mosaic run would have rewarded contact anywhere on the target while the
other six designed against a patch.

Fixed in commit *Mosaic: the epitope its loss was always able to take*. The
driver takes `--epitope` as 0-based indices into the target sequence; preflight
maps campaign author residue numbers through the ordered target PDB chain and
refuses an ambiguous FASTA/PDB mapping. The bounds check is repeated inside
the driver, because that is the last place the numbers exist before they
become an array slice and JAX clips an out-of-range index rather than raising.

The separate cropped-target indexing bug in
[known-issues §1.4](../known-issues.md) silently conditioned 23 of 26 epitope
entries on the wrong residues. This mapping is checked against both the PDB
and FASTA instead of repeating that assumption.

Mosaic was launched from its own index after the fix, rather than restarting
the six already on the queue: the change touches `tools/mosaic/` and
`drivers/mosaic/` only, so nothing else in the run depends on it.

### 2. An epitope means four different things

Not a failure, but the single most important caveat on any comparison drawn
from this run. `epitope_enforcement` in each run's `workflow_metadata` records
it per tool:

| Tool | How the epitope enters | Checked after generation? |
|---|---|---|
| Protein-Hunter | Boltz pocket constraint, a resampling loop that rejects binders which miss it, **and** a condition on the hit gate | **yes** |
| BoltzGen | `binding_types` in the design spec, at generation | no |
| PXDesign | `hotspots` in the input spec, at diffusion | no |
| Genie 3 | `cond_strategy: extended` — the triple plus a 12-residue patch around it | no |
| Proteina-Complexa | a hotspot mask on the target, during sampling | no |
| FreeBindCraft | `i_con` loss restricted to the hotspots, **20 Å**, hallucination only | no |
| Mosaic | `BinderTargetContact` sliced to the epitope columns, **20 Å**, hallucination only | no |

Only Protein-Hunter enforces it. For the other six, "conditioned on A110/A112/
A131" means the search was biased towards the patch, not that the designs
touch it — the FreeBindCraft validation run the day before accepted two designs
that landed on a neighbouring patch entirely. **Epitope coverage has to be
measured from the structures; it is not a property any of these runs reports.**

### 3. FreeBindCraft cannot be asked for a number of designs

Its `designs_per_job` is a stopping condition, not a table size, and the run
ends at whichever of `number_of_final_designs` and `max_trajectories` trips
first. Asking for 25 per job with a budget of 40 successful trajectories means
the answer is somewhere between roughly 20 and 45, and the run will very likely
stop on the budget rather than on the design count. That is recorded — the
task's `ranked` flag is false and its trajectory census says why — but it means
FreeBindCraft is the one tool whose "50" is nominal.

### 4. Nothing here is reproducible for two tools

FreeBindCraft and Protein-Hunter both draw their seeds from unseeded global
RNGs and have no flag that changes it. Both record `reproducible: false`. The
other five take a seed base and split it per task.

### 5. Eight of FreeBindCraft's metrics are constants

PyRosetta is not in the image, so `dG`, `Binder_Energy_Score`, `PackStat` and
the hydrogen-bond counts are fixed values chosen to pass. They are not emitted
as metrics, and the thresholds they made inert are named in the run's
`inert_filters`. For `relaxed_filters.json` that is four of its thresholds.

### 6. Pre-existing: BoltzGen returns memorised natural protein

[known-issues §1.4b](../known-issues.md). Roughly a third of the archived
BoltzGen benchmark, and 7–10 of 10 designs in `boltzgen_run10`/`run11`, are
human ubiquitin rather than designs. Not addressed here and not caused by this
run's settings; any comparison involving BoltzGen's numbers has to check for it.

---

## Results

Filled in below as runs land. Counts are `n_requested / n_produced / n_passed`
straight from the `runs` table; `n_passed` is each tool's own verdict and means
something different for each of them.

### The counts

Every tool was asked for 50. Every tool succeeded; none needed a retry.

| Tool | produced | passed (its own verdict) | wall |
|---|---|---|---|
| Mosaic | 50 | — (it has no filters) | 2h53 |
| BoltzGen | 50 | 4 | 17 min |
| Genie 3 | 50 | 1 | 55 min |
| PXDesign | 50 | 0 | 8 min |
| Protein-Hunter | 50 | 7 | 9 min |
| Proteina-Complexa | 50 | 1 | 18 min |
| FreeBindCraft | **1500** | 14 | 7h52 |

`n_passed` is not comparable across this column — it is each tool's own filter
set, and PXDesign's 0 of 50 is what its `extended` preset does at this scale
(upstream needs 10,000+ designs for 10–100 dual-filter passes), while
Mosaic simply never judges its output.

FreeBindCraft's 1500 is the honest number and the reason its row looks nothing
like the others: it asked 1500 MPNN sequences to clear its base AF2 filters,
1443 of them did not, 57 were scored in full and 14 accepted. Both tasks spent
their 40-trajectory budget (88 and 81 attempts) rather than reaching 25
designs, which is the `ranked: false` outcome §3 predicted.

### Does anything actually bind the epitope?

Measured from the complexes, because no tool reports it: a heavy atom of the
binder within 4.5 Å of A110, A112 or A131.

| Tool | complexes kept | ≥1 epitope residue | all three |
|---|---|---|---|
| Proteina-Complexa | 50 | **50** | 44 |
| PXDesign | 50 | **50** | 28 |
| FreeBindCraft | 57 | **54** | 7 |
| BoltzGen | 50 | 12 | 5 |
| Genie 3 | 50 | 5 | 3 |
| Protein-Hunter | 7 (its hits only) | **0** | 0 |
| Mosaic | 0 | — | — |

Read the "complexes kept" column before the others: the tools do not keep the
same things. BoltzGen, PXDesign, Proteina-Complexa and Genie 3 write a
structure per design; FreeBindCraft writes one for each design that survived
its base filters (14 accepted plus 43 rejected); Protein-Hunter writes one only
for designs that cleared its threshold gate; **Mosaic writes none at all**.

Three results fall out of this.

**The one tool that enforces the epitope is the only one that misses it
entirely.** All 7 of Protein-Hunter's hits satisfy its own gate — CA–CA within
15 Å for at least 2 of the 3 residues — and not one of them puts a heavy atom
within 4.5 Å of any of them. 15 Å CA–CA means "in the neighbourhood", not
"bound". Its resampling filter and its hit gate are doing what they say; what
they say is much weaker than it sounds.

**Biasing the search beats enforcing a loose criterion.** Proteina-Complexa and
PXDesign, which only condition their sampling and check nothing afterwards,
land on the epitope in 50 of 50 designs each. FreeBindCraft's 20 Å `i_con`
bias gets 54 of 57.

**Mosaic's conditioning cannot be verified at all.** It records sequences and
one ranking loss and keeps no structure, so there is nothing to measure. The
epitope reached the loss — the driver logged `epitope: [109, 111, 130]`, which
is where PDB residues A110/A112/A131 map on this contiguous chain — but
whether it changed where the binders
sit is unanswerable from what the run wrote.

### BoltzGen is still returning ubiquitin

[known-issues §1.4b](../known-issues.md) reproduces exactly. 11 of BoltzGen's
50 designs are ≥80% identical to human ubiquitin, one of them at 99%. No other
tool in the run exceeds 18% identity to it. That also explains the bottom of
the coverage table: a memorised natural protein is not binding the epitope, and
BoltzGen's 12 of 50 is the arithmetic of roughly a fifth of its output being
ubiquitin.

### An adapter defect the run exposed — fixed, and re-run

Both FreeBindCraft tasks stopped on their trajectory budget, which is the case
the adapter claimed leaves `final_design_stats.csv` empty. It does not:
BindCraft appends a row for every accepted design with the `Rank` column blank
and fills the ranks in only on the way out. The counts were right either way,
but the test fixture wrote a shape BindCraft never produces and the adapter
discarded names it should have been checking.

Fixed in *FreeBindCraft: the ranked table is unranked, not absent*; the names
are now a third witness against the accepted set. An ingested run cannot be
re-collected into the same row by design, so `freebindcraft-epitope-run19b` is
a fresh run collected by the corrected adapter — and, since FreeBindCraft has
no seed, an honest replicate as well.


---

## What "50 designs" meant per tool

| Tool | 25 per job is | Produced is |
|---|---|---|
| Mosaic | designs, exactly | the same, or fewer if the walltime cut it short |
| BoltzGen | `num_designs` = `budget` = 25, filtering off | exactly 25 |
| Genie 3 | 25 backbones, each refolded by 5 AF2 models | 25 designs, 125 rows |
| PXDesign | `--N_sample 25`, table padded to 25 | exactly 25, most of them failures |
| Protein-Hunter | 5 trajectories × 5 cycles | exactly 25 |
| Proteina-Complexa | 25 drawn, 25 kept (filter is a no-op) | up to 25 |
| FreeBindCraft | a stopping condition, capped by 40 trajectories | anywhere from 0 to ~45 |
