# Alpha benchmark: nine tools, one target, 40 binders each

**Target:** DIO3-cut, 201 aa, single chain.
**Date:** 2026-08-26/27.
**Cluster:** FASRC Kempner, `kempner_h100`, account `kempner_bsabatini_lab`.
**Brief:** [`initial_prompt.md`](initial_prompt.md).

The question this run answers is *how do I launch each of these tools correctly,
what comes out, and what should I request from SLURM next time* — not which tool
designs the best binder. Nothing here has been validated experimentally, and 40
designs is a pilot for every one of these tools (upstream RFdiffusion suggests
1,000–10,000 backbones per target; PXDesign's own README targets 10,000+ designs
to land 10–100 dual-filter passes).

Read [`known-issues.md`](known-issues.md) alongside this. Several tools produce
confident, well-formed output when they are misconfigured, and that document is
the list of ways that happens.

---

## Method

**One target structure for everybody.** The brief supplies DIO3-cut as a
sequence plus an MSA. Seven of the nine tools are structure-conditioned, so the
target was folded **once** — Mosaic/Boltz-2 with the precomputed MSA, 4 recycles,
50 sampling steps — and published as `outputs/_shared/dio3_cut.{cif,pdb}`. Mean
pLDDT 86.3, 87% of residues above 70. Folding it once is what makes the
comparison about the design method instead of about whichever folder each tool
would otherwise have called internally.

Note the target is **201 residues**, not 198. Several configs encode the length
(`A1-201`, the contig `A1-201/0 80-80`) and would misbehave quietly if it were
wrong.

**One job per tool, 40 designs, no arrays.** An array would hide per-design cost
behind the scheduler, and per-design cost is the thing being measured. Each
launcher's header records what a production run should do instead.

**No hotspots, wherever the tool allows it.** This matches the lab's previous
DIO3 campaigns (`hotspots: ""` in both
`mosaic_setup/mosaic/pipelines/presets/*_dio3.yaml`) and lets each generator
choose its own surface. It is also the only setting under which these tools are
comparable, since they do not express epitopes the same way.

Genie 3 is the sole exception: it has **no hotspot-free binder mode** — its
reducer reads `target_interface_residues['hotspot']` unconditionally. It was
given `A110, A112, A131`, chosen as follows. DIO3 is a deiodinase, so the
biologically motivated epitope is the catalytic site; the catalytic motif
`NFGSCTCPPF` sits at residues 61–70 with the catalytic cysteine at 65. That
residue is buried in the predicted structure (23 neighbours within 10 Å of its
Cβ, against a chain median of 17), so the hotspots are the solvent-exposed rim
of its pocket — residues with burial 9–12 lying within 14 Å of C65. A binder
there would occlude substrate access. **This is a defensible default, not an
experimentally mapped epitope**, and it means Genie 3's designs are not strictly
comparable to the hotspot-free runs.

**How resources were measured.** `jobstats` and `sacct` for allocation and host
memory; a 15-second `nvidia-smi` trace for GPU utilisation and memory. The trace
exists because `jobstats` is not trustworthy here on its own — it reported
`GPU utilization 0% <-- GPU was not used` for the target-folding job whose GPU
memory it simultaneously reported at 76 GB. Two caveats on any GPU-memory number:

- **JAX preallocates ~75% of the device.** Mosaic, Genie 3's AF2 stage and
  Proteina-Complexa's AF2 reward will all report ~76 GB regardless of demand
  unless `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set. Which convention a run
  used is recorded in the header of its `gpu_trace.csv`.
- **PyTorch caching allocators** hold freed blocks, so the peak is an upper
  bound on demand, not a measurement of it.

---

## What each tool consumes and produces

The single most useful thing learned: **only Genie 3 writes a real FASTA.**
Everywhere else the sequences live in a CSV column, or have to be read off a
structure, and no two tools name the column the same way. Any unified
post-processing needs a per-tool extractor —
`launching_scripts/common/inventory.py` records where each one is, and
`seq_qc.py` contains a working extractor for all of them.

| Tool | Target as | Produces | Sequences are in |
|---|---|---|---|
| Mosaic | sequence + MSA | sequences only | `designs_*.txt` (FASTA-*like*: header is the score), `designs_*.jsonl` |
| BoltzGen | structure (CIF) | CIF + metrics | `final_designs_metrics_<budget>.csv`, column **`designed_sequence`** |
| FreeBindCraft | structure (PDB) | PDB + metrics | `mpnn_design_stats.csv`, column `Sequence` |
| Genie 3 | problem set (built from PDB) | PDB + **FASTA** + metrics | `sequences/*.fasta`, as `binder:target` |
| Proteina-Complexa | structure (PDB) | one complex PDB per design | **chain B of the PDB** (`aatype` in the CSV is integer indices, not letters) |
| Protein-Hunter | **sequence only** | PDB + metrics | `summary_all_runs.csv`, column `best_seq` |
| PXDesign | structure (CIF) + MSA dir | CIF + metrics | `summary.csv`, column `sequence` |
| RFdiffusion | structure (PDB) | PDB backbones, **no sequence** | — (poly-glycine) |
| Caliby | complex PDBs | CIF + CSV | `seq_des_outputs.csv`, column `seq` (chains joined by `:`) |

Two traps in that table are worth stating twice. **Chain conventions are
inconsistent**: RFdiffusion writes the binder as chain A and the target as chain
B, while Proteina-Complexa does exactly the reverse. And **Proteina's `aatype`
column looks like a sequence but is not** — it is a comma-separated list of
integer residue indices.

"40 designs" also means something different in each tool — see
[`known-issues.md` §3](known-issues.md), which is required reading before
comparing any two counts. The short version: FreeBindCraft's cap counts only
*successful* trajectories; PXDesign pads `summary.csv` to 40 rows with failures;
BoltzGen's `--budget` filters after generation; Proteina-Complexa deduplicates.

---

## Results

### Ranking by time to 40 binders

All on one H100 80 GB except RFdiffusion (A100 40 GB — it cannot run on H100,
see [`known-issues.md` §2.2](known-issues.md)). "Designs" is what actually landed
on disk.

| # | Tool | Wall clock | min/design | Designs | Outcome |
|---|---|---|---|---|---|
| 1 | **Caliby** | **1 min 21 s** | 0.03 | 40/40 | inverse folding only — needs backbones as input |
| 2 | **PXDesign** | **15 min 24 s** | 0.39 | 40/40 | complete, both filter stages ran |
| 3 | **BoltzGen** | **23 min 08 s** | 0.58 | 38/40 | 2 dropped by hard filters |
| 4 | **Proteina-Complexa** | **25 min 56 s** | 0.65 | 40/40 | complete; ipSAE columns all zero (§1.7) |
| 5 | **RFdiffusion** | **41 min 09 s** | 1.03 | 40/40 | backbones only; **A100** |
| 6 | **Protein-Hunter** | **59 min 44 s** | 1.49 | 40/40 | 34 with a best sequence, 6 lost to the alanine cap |
| 7 | **Genie 3** | **1 h 04 min** | 1.61 | 40/40 | full pipeline, after the JAX fix (§2.3) |
| 8 | **Mosaic** | **4 h 40 min** | 7.0 | 40/40 | **95% GPU utilisation** — slow but genuinely GPU-bound |
| 9 | **FreeBindCraft** | **7 h 24 min** | 11.1 | 41 accepted | stopped on the *accepted* cap, not the trajectory cap |

**FreeBindCraft ended the way I predicted it would not**, which is worth
recording. §3 of `known-issues.md` warns that a trajectory-capped run usually
never reaches `number_of_final_designs`, so `final_design_stats.csv` and
`Accepted/Ranked/` never get written. Here the opposite happened: the acceptance
rate was high enough that the **accepted** counter tripped first, at 34
successful trajectories rather than the 40 allowed, and the file exists with 41
rows. Both branches are real; which one you hit depends on the target, so a
harness must handle either and cannot assume.

The full accounting: 49 trajectories attempted → 34 successful (11 clashing, 4
low-confidence) → 326 MPNN designs evaluated → 41 accepted. That is 13.1 min per
successful trajectory, 10.8 min per accepted design.

**Mosaic is the instructive one.** It is the slowest tool that finished and also
the best-utilised: 95.1% mean GPU utilisation, 7.0 min per design with very
little variance (405–694 s, and the 694 s outlier is the first design paying JIT
compilation). It is not wasting the GPU — it is doing 7 minutes of real gradient
work per design, optimising a soft sequence through Boltz-2 with a nine-term
loss. Nothing about its schedule is fixable by better batching; the only lever is
fewer optimisation steps or a cheaper loss.

That is worth separating from the other slow entries, where low utilisation
*does* indicate reclaimable time.

**Genie 3's line is the whole pipeline, and it is very unevenly distributed:**
generation 383 s (9.6 s/design), ProteinMPNN 18 s (0.45 s/sequence), and AF2
refolding **3363 s** — 84 s per sequence, 16.8 s per model output, at 5 models ×
20 recycles. Generation is the fastest in the suite by a wide margin; the AF2
evaluation is 88% of the wall clock. Dropping to `num_models: 1,
num_recycles: 3` would cut the run to roughly 15 minutes at the cost of
departing from the calibrated v0 filter thresholds.

Before the JAX fix this stage ran on the CPU and made no measurable progress in
33 minutes; afterwards it completed in 56. That is the single largest change
this benchmark produced.

**Caliby** is not a generator; it turned RFdiffusion's 40 backbones into 40
sequences, so the honest cost of that pipeline is RFdiffusion + Caliby ≈ **42
minutes**, which is still second only to PXDesign among routes that yield real
sequences.

### Self-reported design quality — do not cross-compare these

Each number below comes from a *different* scorer, and in most cases from the
same model that produced the design, evaluating its own work. They say "did this
tool's own objective converge", not "which tool makes better binders". Ranking
binders across tools requires one common scorer applied to all of them, which is
the natural next step (ProtForge, or Mosaic's folding backends).

| Tool | Metric (the tool's own) | Value |
|---|---|---|
| Protein-Hunter | Boltz-2 ipTM of its best cycle | median 0.89, max 0.97 (n=34) |
| FreeBindCraft | AF2 i_pTM of accepted designs | median 0.82, max 0.89 (n=41); pLDDT median 0.90 |
| Proteina-Complexa | binder self-consistency RMSD | median 0.51 Å; **39/40 under 2 Å** |
| Genie 3 | AF2-multimer, best of 5 models per design | ipTM median 0.28, max 0.85, 8/40 above 0.5; complex scRMSD median 14.8 Å, 2/40 under 2.5 Å; **V0 successes: 0/40** |
| PXDesign | AF2-IG ipTM | median 0.50; 16/40 pass AF2-IG-easy, **5/40 pass both filters** |
| Caliby | Potts energy U | median −982 |
| Mosaic | composite ranking loss (lower better) | median −1.75 (partial) |
| BoltzGen | design→target ipTM | **median 0.18** — see below |

### Sequence composition — the one comparison that *is* fair

Unlike the scores above, amino-acid composition can be measured identically for
every tool. `launching_scripts/common/seq_qc.py` reproduces this.

| Tool | n | length | Shannon entropy | most common residue | flagged |
|---|---|---|---|---|---|
| FreeBindCraft | 163 | 70–150 | **3.69** | E 20% | 25/163 |
| Proteina-Complexa | 40 | 70–109 | **3.62** | L 16% | **2/40** |
| Caliby | 40 | 80 | 3.56 | E 19% | **0/40** |
| BoltzGen | 38 | 70–90 | 3.32 | A 20% | 13/38 |
| Genie 3 | 40 | 65–119 | 3.28 | E 22% | 11/40 |
| Protein-Hunter | 34 | 67–120 | 3.26 | E 25% | 17/34 |
| PXDesign | 40 | 80 | 3.19 | A 23% | 22/40 |
| Mosaic | 40 | 80 | **3.18** | A 22% | **28/40** |

*Flagged* = any single residue exceeds 25% of the sequence, or entropy falls
below 3.0 bits. A natural globular protein sits at roughly **4.1–4.2 bits**.

**Every tool here is well below that**, which is expected for de novo binders
(they are mostly idealised helical bundles), but the spread is large and it
tracks something real. Proteina-Complexa's designs look like natural protein —
e.g. `NSALAWNNLGVVYKNQGDLLEAAKCYKKALELKPNDTEIHNNYLAVLDSLAKNGLPLHALEERQKVLSLR`,
recognisably TPR-like. The alanine-rich end of the table is the classic
gradient-based-hallucination failure mode: sequences that satisfy the scoring
model while being poor candidates for expression.

**Mosaic sits furthest into it — 28 of 40 flagged, the worst in the suite** —
and that is the expected cost of its method rather than a defect: it optimises a
soft sequence directly against a differentiable objective, which is exactly the
setting in which composition drifts toward whatever the scorer likes. PXDesign is
close behind at 22/40. Both are worth re-running with an explicit composition
penalty before any of these designs are ordered.

Two practical consequences. **Caliby's 0/40 is not a coincidence** — inverse
folding onto a fixed backbone regularises composition in a way that direct
sequence optimisation does not, which is an argument for the
RFdiffusion → Caliby route over sequence-space hallucination. And
**Protein-Hunter's hardcoded 20% alanine cap is doing real work**; it is the
only tool that guards against this internally, and it is why 6 of its 40 runs
have a `NaN` best score rather than a bad sequence.

**Genie 3 is now the one tool with a real, independent verdict on its own
output, and it is sobering.** Because the JAX fix let AF2 actually run, we get
the full v0 evaluation instead of a proxy — and **no design passed**. The
binding site is the reason: median hotspot coverage is **0.00**, with only 3/40
designs achieving full coverage of the `A110/A112/A131` patch they were
conditioned on. Complex scRMSD is poor too (median 14.8 Å) even though binder
pLDDT is respectable (median 78). The designs are plausible proteins that
largely did not land on the requested epitope.

Two readings, and it is worth not collapsing them. Forty designs against strict
filters is a small sample — upstream Genie 3 expects far more — so 0/40 is not
by itself evidence the tool is bad. But the hotspot-coverage number is a
different signal from the success count: it says the conditioning did not take,
which is a question about the epitope choice and the conditioning strategy
(`cond_strategy: extended`), not about sample size. That is the first thing to
investigate before scaling this tool up.

**BoltzGen's 0.18 is a consequence of how this benchmark was configured, not a
verdict on the tool.** To keep the design count comparable, it ran
`--num_designs 40 --budget 40 --filter_biased false`, which applies *no*
selection pressure — every generated backbone was kept. BoltzGen is built to
oversample and then diversity-filter (upstream's own example is 200 → 40). A
production run should generate several hundred and let `--budget` do its job.
The same caution applies in reverse to Protein-Hunter's 0.89: that is Boltz-2
scoring a design that was itself optimised against Boltz-2, so it is
in-sample and optimistic.

---

## Recommended SLURM requests

Derived from measurement, with headroom. `min/design` scales roughly linearly,
so multiply the walltime by `N/40` for a bigger campaign.

| Tool | GPU | cpus | mem | time (40) | Notes |
|---|---|---|---|---|---|
| Caliby | h100 (any ≥8 GB) | 8 | 16G | 00:20:00 | 4.1 GiB peak; cost scales with `num_seqs_per_pdb × potts_sweeps` |
| PXDesign | h100/a100/h200 | 16 | 64G | 01:00:00 | 6.2 GiB peak; **not Blackwell** (jaxlib 0.4.29) |
| BoltzGen | h100 | 8 | 64G | 01:00:00 | 7.6 GiB GPU, 27.6 G host RAM — the host figure is the binding one |
| Proteina-Complexa | h100 | 16 | 64G | 01:30:00 | 15.9 GiB with JAX preallocation OFF, i.e. real demand |
| RFdiffusion | **a100** | 8 | 32G | 01:30:00 | 6.9 GiB, 78% mean GPU util — the best-utilised job here |
| Genie 3 | h100 | 16 | 32G | 00:30:00 | generation only until the JAX build is fixed; 3.3 GiB |
| Protein-Hunter | h100 | 8 | 32G | 02:00:00 | 13.8 GiB; MPNN subprocess wants the CPUs |
| Mosaic | h100 80 GB | 8 | 128G | 12:00:00 | JAX preallocates; H200 preferable for ESM-C 6B |
| FreeBindCraft | h100 | 8 | 64G | 36:00:00 | reports 61 GiB but that is JAX preallocation, not demand |

Two systematic notes on the memory column:

- **Every "GPU GiB" for a JAX tool is an allocator reservation.** Only
  Proteina-Complexa ran with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so only its
  15.9 GiB is a measurement of demand. FreeBindCraft's 61.4 GiB is AF2/JAX
  reserving 75% of the card; its real requirement is far lower.
- **Host RAM was over-requested everywhere.** The largest observed `MaxRSS` was
  BoltzGen's 27.6 G against a 96 G request. The table above cuts most requests
  substantially; fewer wasted GB means better queue position.

GPU utilisation is low for most tools (10–54% mean) because these pipelines
alternate GPU inference with CPU-side work — MPNN subprocesses, PDB parsing,
relaxation. That is inherent, not a misconfiguration, and it means asking for
more CPUs often helps more than a bigger GPU.

---

## Where the outputs are

```text
outputs/
  _shared/            target CIF/PDB, the PXDesign MSA dir, the Genie 3 problem set,
                      inventory.tsv
  <tool>/
    logs/             SLURM stdout
    resources/        jobstats.txt, sacct.txt, summary.tsv, gpu_trace.csv
    ...               tool-specific output tree
```

`launching_scripts/common/inventory.py` prints the current design count and the
exact file holding each tool's sequences; `summarize.py` prints the resource
table. Both are stdlib-only and safe to re-run at any time.

### Exact paths for post-processing

Chain conventions differ and matter — **RFdiffusion makes the binder chain A and
the target chain B; Proteina-Complexa does the reverse.**

```text
mosaic/designs/designs_0.txt              >{score}\n{binder_seq}  (FASTA-like)
mosaic/designs/designs_0.jsonl            {index,seed,sequence,score,seconds}

boltzgen/results/final_ranked_designs/
    final_designs_metrics_40.csv          column `designed_sequence`
    final_40_designs/rank<NN>_*.cif       complexes, ranked
    before_refolding/                     pre-refold copies

freebindcraft/
    mpnn_design_stats.csv                 column `Sequence` (+ AF2 metrics)
    final_design_stats.csv                accepted designs only
    Trajectory/{Relaxed,Clashing,LowConfidence}/
    Accepted/*.pdb , MPNN/Relaxed/*.pdb

genie3/dio3_cut/
    pdbs/dio3_cut_<i>.pdb                 backbone, chain A = binder (UNK)
    sequences/dio3_cut_<i>.fasta          "binder_seq:target_seq", header has MPNN score
    (harvested from eval_shards/shard_0_of_1/devices/device_0/sequences/)

proteina_complexa/inference/<run>/
    job_0_n_<total>_id_<i>_bon_*/*.pdb    complex; chain A = TARGET, chain B = binder
    rewards_*.csv                         column `aatype`, plus AF2 reward terms
proteina_complexa/evaluation_results/<run>/
    binder_results_*.csv                  scRMSD etc (ipSAE columns are all 0 — see §1.7)

protein_hunter/dio3_cut_boltz/
    summary_all_runs.csv                  `best_seq`, per-cycle iptm/plddt/seq
    summary_high_iptm.csv                 rows clearing the gate (79 here, multi-cycle)
    high_iptm_pdb/*.pdb                   NB: .pdb, not .cif as the README claims

pxdesign/run01/out/design_outputs/dio3_cut/
    summary.csv                           column `sequence`; PADDED to N_sample with failures
    orig_designed/rank_<k>.cif            all selected
    passing-AF2-IG-easy/ , passing-Protenix-basic/

rfdiffusion/designs/
    dio3_cut_<i>.pdb                      backbone only, binder = chain A, poly-GLY
    dio3_cut_<i>.trb                      pickle: resolved config, plddt, contig maps
    traj/                                 50-model trajectories (write_trajectory=False to skip)

caliby/run1/
    seq_des_outputs.csv                   `seq` = chains joined by ':', `U` = Potts energy
    samples/<key>_sample<N>.cif           NB: column is named out_pdb but files are .cif
```

Launchers and configs are in [`../launching_scripts/`](../launching_scripts/),
one directory per tool, each with a heavily commented `run_<tool>.sbatch`.

---

## What I would do next

1. **Score everything with one model.** Nothing above supports a cross-tool
   quality claim. Run all ~350 designs through a single scorer (ProtForge's
   Boltz/OpenFold stages, or Mosaic's) and rank on that.
2. **Fix the two broken images.** Rebuild `rfd.sif` against a torch/DGL with
   sm_90 so it runs on H100, and rebuild `genie3.sif` with `jax[cuda12]` so its
   AF2 stage uses the GPU. Both are one-line dependency changes and both
   currently cost a whole tool.
3. **Re-run BoltzGen the way it is meant to be used** — oversample and let
   `--budget` filter. The 0.18 median ipTM here is an artifact of forcing a
   40→40 pass-through.
4. **Decide the epitope.** Everything except Genie 3 ran hotspot-free, so each
   tool chose its own surface and they are very unlikely to agree. If DIO3's
   catalytic pocket is the intended site, set hotspots consistently across tools
   before scaling — and apply the `EPITOPE_INDEXING_BUG.md` fix first, since
   that path is exactly where it bites.
5. **Trim the resource requests** to the table above before a large campaign.
   Host RAM was over-requested by 3–5× everywhere.
