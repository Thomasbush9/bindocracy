# What the alpha run says about building the harness

This is the actual output of the benchmark. The 40-binder runs were never about
binder quality — nothing here has been validated, and 40 designs is a pilot for
every one of these tools. They were about finding out **what a library has to
absorb** to drive nine of these things from one place.

Each section below is a requirement, and each one is there because something in
`known-issues.md` proved it was needed. Ordered by how much of the design it
constrains.

---

## 1. A `Target` is not one thing, and the harness must own the conversion

The brief supplied DIO3-cut as a sequence plus an MSA. What the tools actually
wanted:

| Representation | Tools |
|---|---|
| sequence only | Mosaic, Protein-Hunter |
| structure (PDB) | FreeBindCraft, Proteina-Complexa, RFdiffusion, Caliby |
| structure (mmCIF) | BoltzGen, PXDesign |
| structure **+ a two-file MSA directory** | PXDesign |
| a **problem-set directory** (JSON + per-chain PDB/FASTA/MSA trees) | Genie 3 |
| an **entry appended to a Hydra target registry** | Proteina-Complexa |

Two of those cannot be produced by a format conversion at all. Genie 3 needs a
problem set whose official builder calls the ColabFold server and crashes
offline; Proteina-Complexa needs a target registered inside a config tree that
lives read-only inside the image and has to be copied out first.

**Requirement.** One `Target` object, built once from sequence + MSA, that can
*materialise* each of these representations on demand and cache them. Folding the
target once — rather than letting each tool call whatever folder it ships — is
also what makes any cross-tool comparison mean anything. In this run that was
`00_target/predict_target.sbatch`, producing `dio3_cut.{cif,pdb}` at pLDDT 86.3.

Two details that must live in the Target, not in per-tool scripts, because
getting them wrong is silent:

- **Residue numbering.** The target is 201 residues, not the 198 first assumed.
  `A1-201` and the contig `A1-201/0 80-80` both encode it. A Target that knows
  its own length and chain id removes a whole class of off-by-N.
- **Chain identity.** RFdiffusion emits the **binder as chain A** and the target
  as chain B; Proteina-Complexa does exactly the reverse. Downstream code must
  ask the adapter which chain is which, never assume.

## 2. "Generate N designs" is not portable — model three numbers, not one

Every tool interprets a count differently, and the differences are not cosmetic:

| Tool | What the number controls | What you actually get |
|---|---|---|
| BoltzGen | `--num_designs` generated, `--budget` kept | 38 of 40 survived hard filters |
| FreeBindCraft | `max_trajectories` = **successful** hallucinations | rejected trajectories don't count, so attempts > N |
| PXDesign | `--N_sample`, then padded | `summary.csv` always has exactly N rows, **padded with failures** |
| Proteina-Complexa | `nsamples × nrepeat × replicas`, then filtered | dedup can leave fewer |
| Protein-Hunter | trajectories × cycles | 34/40 produced a best sequence; 6 lost to an alanine cap |
| Genie 3 | `n_sample` **per problem** | 40 generated, **0** passed the v0 filter |

**Requirement.** The schema needs **attempted / produced / passed** as three
distinct fields, plus the filter definition that produced the third. A single
`n_designs` field would have recorded "40" for PXDesign and for Genie 3 and
hidden that one had 5 dual-filter passes and the other had none.

## 3. Output adapters are unavoidable — there is no common format

Only Genie 3 writes a FASTA. Everywhere else sequences are a CSV column, and no
two tools name it the same: `designed_sequence` (BoltzGen), `Sequence`
(FreeBindCraft), `best_seq` (Protein-Hunter), `sequence` (PXDesign), `seq`
(Caliby, chains joined by `:`). Proteina-Complexa has an `aatype` column that
*looks* like a sequence and is actually comma-separated integer indices — its
real sequence has to be read off chain B of a PDB.

**Requirement.** One adapter per tool, whose job is `run_dir -> [Design]` with
sequence, structure path, tool-native scores, and provenance. Write them against
observed output, not upstream docs: the Protein-Hunter README documents an output
directory (`high_iptm_cif`) that the code does not create, and PXDesign documents
a `--return_topk` flag that is dead code.

`launching_scripts/common/inventory.py` and `seq_qc.py` are the throwaway version
of this and are worth reading as a spec for the real one.

## 4. Cross-cutting environment hygiene belongs in one place

Three separate tools were broken by two host-environment problems that have
nothing to do with protein design:

- **`TMPDIR` on Lustre kills any Triton-JIT tool.** Triton compiles kernels in a
  temp dir and the cleanup `rmtree` cannot complete on `/n/netscratch`, raising
  `Errno 39` on the first kernel. BoltzGen died 9 seconds into a 6-step pipeline.
- **The host's RHEL `SSL_CERT_FILE` breaks `httpx` inside every Ubuntu image**,
  at import, before offline mode is consulted.
- **`--cleanenv` is correct but drops SLURM's `CUDA_VISIBLE_DEVICES`**, so it has
  to be forwarded deliberately.

**Requirement.** A single launch layer that sets node-local `TMPDIR`, the
container's CA bundle, and the GPU assignment for every tool. In this run that is
`common/env.sh`; in the library it should be impossible for a tool driver to
opt out of it.

## 5. Preflight assertions are the highest-value code in the harness

The expensive failures were all silent. Ranked by what they would have cost:

| Silent failure | What you would have seen |
|---|---|
| RFdiffusion selects the **monomer** checkpoint when hotspots are omitted | 40 plausible backbones designed against nothing |
| Caliby redesigns the **target** when a `pdb_key` is missing from its CSV | a complete, well-formed output CSV |
| Genie 3's JAX runs on **CPU** | a live process, an idle GPU, and no error, forever |
| PXDesign's `--preset` defaults to `custom` | a normal-looking `summary.csv` with no filters applied |
| PXDesign's CLI reverts its own eta schedule | different sampling than the config says |
| Mosaic's epitope indices on a cropped target | three campaign rounds aimed ~100 residues off |

None of these raise. Every one is cheap to assert *before* the GPU work starts:
check the resolved checkpoint against the requested mode; check the constraint
file covers every input; check `jax.default_backend() == "gpu"`; echo the
resolved config back and diff it against the requested one.

**Requirement.** Each tool driver declares preconditions, and the harness refuses
to launch if any fail. The launchers here do this ad hoc — `run_genie3.sbatch`
asserts its three overlays exist, `run_caliby.sbatch` verifies after the fact that
the fixed chain is byte-identical between input and output — and it should be a
first-class part of the driver interface instead.

## 6. Exit status is not a success signal

Genie 3's fixed run logged `🎉 Run completed`, wrote 40 sequences, 40 backbones
and 200 AF2 outputs — and `sacct` recorded **FAILED**. The cause was a trailing
`find … | head -30` in our own script: under `set -o pipefail` the SIGPIPE from
`head` fails the pipeline, and a job's exit status is its last command's. Had the
harness trusted `sacct`, an hour of correct GPU work would have been discarded.

**Requirement.** Success is defined per tool as *the expected artifacts exist and
parse*, with exit status as corroboration only. The adapter from §3 already knows
what those artifacts are, so it should own the predicate.

## 7. Images are not self-describing: probe capability, don't read the manifest

A weight audit that asks only "are the checkpoints present" passes both of the
worst images in this suite:

- **`rfd.sif`** has all nine RFdiffusion checkpoints embedded and **cannot execute
  a single CUDA kernel on H100** — torch 1.12.1+cu116, `arch_list` ending at
  sm_86, no PTX to JIT forward from. It works on A100.
- **`genie3.sif`** has every weight and shipped a **CPU-only JAX**.

Both were documented as fine. `CONTAINER_WEIGHTS_REPORT.md` additionally
described the *wrong artifact* for RFdiffusion — the `rfdiff.def` ESMFold2
variant that was never built — rather than the installed SIF.

**Requirement.** Per-image capability metadata generated by *running* a probe on
the target GPU class, not by reading a definition file: GPU arch support, the
backend each framework actually resolves to, and a minimal end-to-end smoke
design. Store it beside the SIF checksum.
`launching_scripts/rfdiffusion/diagnose_h100.sbatch` and
`genie3/verify_jax_gpu.sbatch` are the pattern.

Corollary: **record the SIF checksum with every campaign**, and where an overlay
is in play (Genie 3's three JAX overlays) record those too — the image alone no
longer determines the result.

## 8. Resource requests should be measured, then reused

Measured against requested, across the seven completed jobs:

- **Host RAM: 608 GB reserved, 83 GB used — 14%.** Every job over-requested,
  the worst by 18×.
- **GPU memory: nothing needed more than ~16 GiB.** A 40 GB card would serve
  every tool in this suite. The large numbers are allocator behaviour, not
  demand: JAX preallocates ~75% of the device unless
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which is why FreeBindCraft reports
  61.8 GiB and Proteina-Complexa — the one job run with preallocation off —
  reports a truthful 15.9 GiB.
- **GPU utilisation is low by nature**, 13–78% mean. These pipelines alternate
  GPU inference with CPU-side work (MPNN subprocesses, PDB parsing, relaxation).
  Extra CPUs often help more than a bigger GPU.

**Requirement.** Collect `sacct` plus a sampled `nvidia-smi` trace on every run
automatically, and derive the next request from history rather than from a
guess. Two measurement traps to bake in: `jobstats`' utilisation sampler is too
coarse for short jobs — it reported "GPU utilization 0% <-- GPU was not used" for
a job whose memory it simultaneously reported at 76 GB — and any GPU-memory
figure must be labelled with the allocator convention that produced it.

## 9. Can we use the GPU harder by putting more sequences in a job?

Partly — and the distinction matters, because batching fixes one cause of low
utilisation and does nothing for the other two.

**The enabling fact is memory headroom.** Nothing measured needs more than
~16 GiB on an 80 GB card, so there is roughly a 4–5× budget to spend on larger
batches before memory becomes the constraint. That is why this is worth doing at
all.

### Cause A — batch size 1. Fixable, and we hit it.

BoltzGen logged:

```
Using diffusion batch size: 1
Number of diffusion batches: 40
```

That was **an artefact of this benchmark, not the tool**:
`--diffusion_batch_size` "defaults to 1 if `--num_designs` is less than 100, and
10 otherwise". Asking for 40 designs put it on the slow path, and its 54% mean
utilisation is partly our doing. Verified levers that exist today:

| Tool | Knob | What it batches |
|---|---|---|
| BoltzGen | `--diffusion_batch_size` | diffusion samples per trunk run (default 1 below 100 designs, 10 above) |
| Caliby | `sampling_cfg_overrides.batch_size` | **structures**, and the Potts MCMC is vectorised across that dim — the real throughput lever |
| Proteina-Complexa | `++generation.dataloader.batch_size` | generation samples (set to 8 here) |
| PXDesign | none exposed | `--N_sample` only |

For these, a bigger job is genuinely a faster job per design.

### Cause B — CPU work in the critical path. Batching will not help.

Several tools idle the GPU while a **subprocess** runs on the CPU, and no batch
size changes that:

- **Protein-Hunter** shells out to LigandMPNN ~200 times, with MPNN's own
  `batch_size` hardcoded to 1 — one sequence per cycle.
- **Genie 3** launches `colabfold_batch` as a subprocess per evaluation stage;
  its AF2 work is 88% of the run.
- **FreeBindCraft** interleaves OpenMM relaxation, FASPR and DSSP with AF2.

For these the lever is **more CPUs**, or upstream code changes — not a larger
batch. It is also why the harness should record CPU count alongside GPU: on this
suite, extra CPUs often buy more than a bigger GPU.

### Cause C — inherently sequential and already saturated. Nothing to reclaim.

- **Mosaic: 95.1% mean utilisation**, the highest measured, at 7.0 min per design
  with almost no variance (405–694 s, and the outlier is the first design paying
  JIT compilation). It is the slowest tool that finished *and* the best utilised
  — it is doing seven minutes of real gradient work per design, optimising a soft
  sequence through Boltz-2 against a nine-term loss. The only levers are fewer
  optimisation steps or a cheaper loss.
- **RFdiffusion: 78%**, one denoising trajectory per design.

The distinction matters more than the raw number: "slow" and "wasting the GPU"
are different diagnoses with different fixes, and the harness should record
utilisation precisely so they are not confused. Mosaic is slow and efficient;
Proteina-Complexa is fast and only 13% utilised.

### Measured, 2026-08-27 — the sweep

Raw data in `outputs/_shared/batch_sweep.json`; figure 5 in `docs/figures/`.
Sampled at 2 s rather than the usual 15 s, because a two-minute job's peak is
invisible at 15 s.

**Caliby — the clean case, and the two knobs behave completely differently.**
40 structures, 8 sequences each:

| `batch_size` | peak GPU | wall clock | GiB per structure |
|---|---|---|---|
| 4 | 4.1 GiB | 2.9 min | 1.02 |
| 8 | 7.5 GiB | 2.4 min | 0.94 |
| 16 | 14.5 GiB | 2.3 min | 0.91 |
| 40 | 35.0 GiB | 2.4 min | 0.88 |

Memory is **linear** in `batch_size` at ~0.9 GiB per structure. Wall clock hits
its knee at **8** and never improves again — batch 40 costs 4.7× the memory of
batch 8 and is *slower*. Meanwhile the other knob, `num_seqs_per_pdb`, is a
sequential loop over one precomputed set of Potts parameters, and behaves exactly
as that implies: 32 sequences instead of 8 took **4× longer at identical
14.5 GiB**.

So the sizing rule for this tool is: **put `batch_size` at the knee, then scale
output with the sequential knob, which is free in memory.** At batch 16 / 32
sequences it produced 1,280 sequences in 7.2 min — **6× the throughput of the
benchmark configuration** at 14.5 GiB.

**BoltzGen — the same knob name, the opposite sizing rule.** All four points
re-measured at 2 s so they are mutually comparable:

| `--diffusion_batch_size` | trunk runs | wall clock | peak GPU | GPU util |
|---|---|---|---|---|
| 1 *(implicit default)* | 40 | 22.9 min | 7.3 GiB | 56% |
| 10 | 4 | **16.2 min** | 5.9 GiB | 60% |
| 20 | 2 | 15.9 min | 6.4 GiB | 61% |
| 40 | 1 | **15.0 min** | 7.1 GiB | 61% |

**Peak memory is flat — and non-monotonic** (7.3 / 5.9 / 6.4 / 7.1 GiB): the
largest batch is *cheaper* than the smallest. The trunk dominates and a
281-token sample adds almost nothing, so the variation is allocator noise rather
than scaling. Memory is simply not the constraint for this tool.

What *is* the constraint is diminishing returns in time: 1 → 10 buys 29%, and
10 → 40 buys only another 7%. So the rule is "batch ≥ 10, and there is no memory
reason not to go higher", which is nothing like Caliby's "stop at the knee or you
waste memory". **Two tools, the same word in the config, opposite advice.**

Note also that utilisation rose only 56% → 61% while wall clock fell 34%. The
gain came from 40 trunk runs collapsing to 1, not from filling the GPU.

**A measurement caveat worth keeping.** The original benchmark run sampled at
15 s and reported 5.3 GiB for batch 10; at 2 s the same configuration peaks at
5.9 GiB. A coarse sampler biases *small* batches to look cheaper than they are,
which is precisely backwards for a curve meant to answer "what fits".

**Proteina-Complexa — the counter-example.** Doubling the generation batch from
8 to 16 bought **9% wall clock for 55% more memory** (15.9 → 24.6 GiB), and
utilisation crept from 13% to 17%. Its idle time is the per-sample AF2 reward,
not the generation batch, so batching cannot reach it. This is the tool that
*looks* like it has the most headroom and has the least.

### What this implies for the harness

- **Store a measured knee per tool, not a batch size.** The useful artefact is
  "memory is ~0.9 GiB per structure and time stops improving at 8", because that
  answers *what fits in this card* for any future target. A single remembered
  number does not survive a change of target size.
- **Distinguish parallel knobs from sequential ones in the tool's capability
  record.** Caliby's two knobs look alike in the config and are opposites in
  cost: one is linear in memory and flat in time past the knee, the other is
  linear in time and flat in memory. A driver that models them as one "batch"
  concept will size jobs wrong.
- **Expose batch size as a first-class per-tool capability**, with its default
  *and its coupling to the design count* recorded — BoltzGen's silent 1-vs-10
  switch at `num_designs=100` is exactly the kind of thing a driver should know
  and a caller should not have to.
- **Optimise throughput, not utilisation.** BoltzGen got 28% faster with its
  utilisation unchanged; Proteina-Complexa's utilisation rose and bought almost
  nothing. Utilisation is a diagnostic, not the objective.
- **Prefer fewer, larger jobs where a batch knob exists.** It also amortises
  model load and JIT compilation, which is a fixed per-job cost (Mosaic ~30 s to
  build models; BoltzGen loads five checkpoints).
- **Classify each tool A/B/C** so the sizing logic knows whether "run more per
  job" is a real answer for it.
- **Re-measure after changing batch size.** Utilisation is only worth improving
  if per-design wall clock actually falls; a bigger batch that spills into
  slower memory can be net-negative. The trace collection in §8 already gives
  the before/after.

One caveat before optimising hard: on this cluster a GPU is allocated
exclusively, so low utilisation wastes the allocation but does not slow anyone
else's job. Throughput per GPU-hour is the thing to optimise, and that is not
always the same as peak utilisation.

## 10. Scheduling realities the harness should know about

- **One job per tool, not arrays, when profiling.** An array hides per-design
  cost behind the scheduler, and per-design cost is what sizing needs.
- **Cluster reservations silently defer long jobs.** A 24 h request during the
  window before an 08:00 maintenance reservation does not queue — it waits until
  after the window. The harness should check reservations and either fit the
  walltime or say plainly that the job will not start today.
- **Resume-safety varies and should be declared.** FreeBindCraft skips designs
  whose PDB already exists (so a walltime kill costs only a resubmit — but a
  *completed* run re-submitted exits immediately); RFdiffusion's
  `inference.cautious=True` silently no-ops a rerun; Protein-Hunter has no seed
  at all, so its runs are **not reproducible and cannot be split across jobs**.

---

## The shape this implies

```
Target(sequence, msa)                  # folds once, caches every representation
  ├── .as_pdb() / .as_cif()
  ├── .as_msa_dir(style="pxdesign")
  ├── .as_problem_set(style="genie3", hotspots=...)
  └── .register(style="proteina")

ToolDriver                             # one per tool
  ├── .preconditions(target, request) -> [Check]      # §5 — refuse, don't warn
  ├── .render(target, request) -> config files + argv # §1, §2
  ├── .resources() -> measured profile                # §8
  ├── .succeeded(run_dir) -> bool                     # §6 — artifacts, not exit code
  └── .collect(run_dir) -> [Design]                   # §3 — chain-aware

Launcher                               # §4, §9
  └── owns TMPDIR, CA bundle, CUDA_VISIBLE_DEVICES, walltime vs reservations,
      and the sacct + nvidia-smi trace
```

The nine `run_<tool>.sbatch` files in `launching_scripts/` are each one instance
of `ToolDriver` written by hand. They are worth keeping as the reference for what
the abstraction has to cover, because every comment in them marks a place where a
tool did something the others do not.

## What is still unknown

The benchmark deliberately did not answer these, and the harness design should
not pretend otherwise:

- **No cross-tool quality comparison exists.** Every score in
  `benchmark-alpha.md` comes from a different scorer, mostly the same model that
  produced the design. A common scorer over all ~350 designs is the next step.
- **The epitope question is open.** Eight tools ran hotspot-free and each chose
  its own surface; they are very unlikely to agree. Genie 3, the one tool given
  hotspots, achieved **median 0.00 coverage** of them — the conditioning did not
  take, which is a finding about conditioning strategy, not about sample size.
- **Two images need rebuilding** before their tools are first-class: `rfd.sif`
  for sm_90, `genie3.sif` for `jax[cuda12]`.
