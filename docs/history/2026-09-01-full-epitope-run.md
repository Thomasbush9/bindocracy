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
maps campaign residue numbers with `seqid - 1` and refuses a hotspot on another
chain or past the end of the FASTA. The bounds check is repeated inside the
driver, because that is the last place the numbers exist before they become an
array slice and JAX clips an out-of-range index rather than raising.

Indexing by enumeration position instead of `seqid - 1` is
[known-issues §1.4](../known-issues.md), which silently conditioned 23 of 26
epitope entries on the wrong residues. Not repeated.

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

<!-- RESULTS -->

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
