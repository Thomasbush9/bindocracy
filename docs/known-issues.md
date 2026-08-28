# Known issues and traps

Everything here was hit or verified during the archived alpha benchmark.
Issues are ordered by how much damage they can do, not by which tool they
belong to.

The dangerous ones are at the top because they **do not raise**. A tool that
crashes costs an hour; a tool that silently designs against the wrong model,
the wrong chain, or the wrong epitope costs a campaign.

## What was fixed on 2026-08-27

| # | Issue | Fix | Verified by |
|---|---|---|---|
| §2.3 | `genie3.sif` JAX was CPU-only, so AF2 never finished | three host overlays (jax-cuda12 plugin, cuDNN 9.8, ptxas) bound in at run time | `verify_jax_gpu.sbatch`: backend `cpu`→`gpu`, cuDNN conv passes, torch unaffected; AF2 then ran at 100% GPU |
| §1.4 | Mosaic epitope indices wrong for cropped targets | index by `seqid - 1` in `proposals.epitope_from_complex()` | identical 19-residue epitope to the old code on a full target; correct by construction on crops |
| §5.2 | 25 Mosaic scripts pointed at a moved directory | split rewrite — `images/` went up a level, the rest moved with the tree | every resulting path checked to exist; all 25 re-parsed |
| §5.1 | `weights.md` + `CONTAINER_WEIGHTS_REPORT.md` wrong about `rfd.sif` | correction blocks added to both | image listing + a completed 40-design run |
| §2.1 | `TMPDIR` on Lustre killed Triton-JIT tools | `node_tmp()` in `common/env.sh`, used by every launcher | BoltzGen went from dying in 9 s to completing in 23 min |
| §2.4 | Genie 3 crashed writing `lightning_logs` to a read-only path | bind writable space over the missing path | run proceeds past setup |
| §2.5 | Host `SSL_CERT_FILE` broke httpx in every image | overridden in `common/env.sh` | BoltzGen imports and runs |
| §2.6 | `--cleanenv` dropped SLURM's `CUDA_VISIBLE_DEVICES` | forwarded in `common/env.sh` | verified SLURM renumbers the allocated GPU to index 0 in-container |
| §2.7 | `rfd.sif` `eval`s args, breaking Hydra `${now:}` | name the run dir from `$SLURM_JOB_ID` | run proceeds |
| §2.8 | RFdiffusion could not create its schedule dir | `inference.schedule_directory_path` to writable space | run proceeds |
| §6.1a | our own launchers reported FAILED after succeeding (`pipefail` + `head`) | `\|\| true` on the trailing summary listings | Genie 3's completed run no longer exits 1 |

**Still open and needing an image rebuild:** §2.2 (`rfd.sif` cannot run on H100 —
currently worked around by submitting to A100) and the durable version of §2.3
(`jax[cuda12]` in the Genie 3 definition, replacing the overlays). §1.7
(Proteina-Complexa ipSAE all zero) is undiagnosed.

---

## 1. Silent-wrong-answer bugs

### 1.1 RFdiffusion picks the MONOMER checkpoint when you omit hotspots

`model_runners.py` only auto-selects `Complex_base_ckpt.pt` when
`ppi.hotspot_res is not None`. Run binder design without hotspots — which is
otherwise legitimate, since the complex model was trained with 80–100% of
hotspots masked — and it silently falls back to `Base_ckpt.pt`, the
**unconditional monomer model**. You get 40 plausible backbones designed
against nothing.

**Fix:** always pass
`inference.ckpt_override_path=/app/RFdiffusion/models/Complex_base_ckpt.pt`
explicitly. Setting hotspots also fixes it, by making auto-selection work.

### 1.2 Caliby silently redesigns the target when a `pdb_key` is missing

Chain fixing is done through `pos_constraint_csv`. If a structure's `pdb_key`
(the filename stem) has no row in that CSV, Caliby does **not** error. It prints
`No fixed positions found` and redesigns the entire complex — target included.
For binder design that destroys the run while producing a full, well-formed
output CSV.

**Fix:** generate the CSV from the actual directory listing and assert the row
count equals the file count (`caliby/stage_inputs.py` does this). Verify after
the fact that the fixed chain is byte-identical between `input_seq` and `seq` in
`seq_des_outputs.csv`; `run_caliby.sbatch` does that check.

### 1.3 Caliby's default checkpoint is monomer-only

`caliby` (the default) and `soluble_caliby` are documented as trained on
monomers only. `soluble_caliby_v1` is the only sequence-design checkpoint
trained on interfaces. Using the default on binder–target complexes is a
methodological error with no visible symptom.

### 1.4 Mosaic epitope indices are wrong for cropped targets — **FIXED**

Pre-existing, documented by the lab in `mosaic_setup/EPITOPE_INDEXING_BUG.md`,
and still unfixed in the checked-out tree until now.
`proposals.epitope_from_complex()` indexed by enumeration position rather than
seqid, so any generator handed a **cropped** target (in practice any BoltzGen run
using `hotspots` + `hotspot_shell`) injected pocket-local indices into a loss
that reads them as full-target indices. Silent. It affected 23 of 26 epitope
entries across three campaign rounds.

**Fixed 2026-08-27** in `mosaic_setup/mosaic/proposals.py`, applying the
one-line change the lab's own document prescribes: index by
`cra.residue.seqid.num - 1` instead of by position within the (possibly cropped)
chain, and drop the now-unused `tgt_pos` map.

Verified to be a no-op on full targets, which is the property that matters for
backward compatibility: run against a Proteina-Complexa complex (chain A =
target, contiguous seqids 1..201), old and new implementations return the
**identical** 19-residue epitope. On a crop the new form maps straight back to
the full-target position, which the old one could not.

This benchmark was unaffected either way — it ran hotspot-free and did not use
the merge/optimize path.

### 1.5 PXDesign silently reverts its own sampling schedule

Upstream changed the default eta schedule to `piecewise_65 / 1.0 / 2.5`, and the
container has that config. But the CLI's `build_argv` emits *every* key in its
shared options dict unconditionally, so the click defaults
(`const / 2.5 / 2.5`) overwrite the config on every run. You get the old
sampling behaviour unless you pass the intended values back explicitly.

**Fix:** pass `--eta_type piecewise_65 --eta_min 1.0 --eta_max 2.5`.

### 1.6 PXDesign's `--preset` defaults to `custom`, which configures no filters

The docstring says extended is the default. It is not. `custom` runs with no
confidence filters at all, so the run completes and produces a `summary.csv`
that looks normal and means much less.

### 1.7 Proteina-Complexa reports every ipSAE as exactly 0.0

In the benchmark run, all twelve `self_complex_*_ipSAE*` columns of
`binder_results_*.csv` are `0.0` for all 40 designs, while the scRMSD columns in
the same file are populated and sensible (median binder scRMSD 0.52 Å, 39/40
under 2 Å). Interface ipSAE is the metric you would most want for ranking
binders, so a column of zeros is easy to mistake for "no design has interface
confidence" rather than "this metric did not run".

The generation-stage rewards CSV *does* carry real
`af2folding_avg_ipsae` values, so rank from `rewards_*.csv` until the evaluation
path is understood. **Not diagnosed** — flagged so it is not read as a result.

### 1.8 Protein-Hunter's chai pipeline silently overwrites previous runs

`check()` tests `os.path.exists(self.jobname)` — the bare jobname in the CWD —
while output actually goes to `./results_chai/<jobname>`. The collision check
therefore never fires and the auto-rename never happens. A rerun overwrites.
`--jobname` is also stripped by `re.sub(r"\W+", "")`, so `dio3-cut` and
`dio3.cut` both become `dio3cut`. Use a fresh working directory per chai run.
The `boltz` pipeline has `--save_dir` and does not have this problem — which is
one reason this benchmark uses it.

---

## 2. Hard failures we hit

### 2.1 `TMPDIR` on Lustre kills any Triton-JIT tool — `Errno 39`

**This is the one most likely to bite you again.** Triton compiles every GPU
kernel inside a `tempfile.TemporaryDirectory()`. On `/n/netscratch` the cleanup
`rmtree` races the filesystem's delayed unlink and raises

```
OSError: [Errno 39] Directory not empty: /n/netscratch/.../tmp/tmp_h4h90bf
```

which propagates out of `compile_module_from_src` and kills the job on its first
kernel. BoltzGen died this way ~9 s into step 1 of 6. Every PyTorch tool here is
exposed.

**Fix:** `TMPDIR` must be node-local. `launching_scripts/common/env.sh` provides
`node_tmp`, which puts it under `/tmp` on the compute node. The cost is that JIT
caches do not survive the job; that is a few minutes of recompilation and worth
paying.

### 2.2 rfd.sif cannot run on H100 — no sm_90 kernels, at all

`RuntimeError: CUDA error: no kernel image is available for execution on the
device`. The image carries torch 1.12.1+cu116, DGL 1.0.2+cu116 and Python 3.9;
CUDA 11.6 predates Hopper entirely. Verified end to end on an H100 by
`launching_scripts/rfdiffusion/diagnose_h100.sbatch` — **every** probe fails,
down to a bare host-to-device tensor copy:

```
arch_list = ['sm_37','sm_50','sm_60','sm_70','sm_75','sm_80','sm_86']
FAIL  tensor copy H2D        FAIL  elementwise add     FAIL  matmul (cuBLAS)
FAIL  conv2d (cuDNN)         FAIL  softmax             FAIL  dgl update_all (SpMM)
```

Note there is **no `compute_*` PTX entry** in `arch_list`, so there is nothing
for the driver to JIT forward to sm_90. This is not a marginal or op-specific
incompatibility and no environment variable will work around it. The same run
succeeds on A100 (sm_80). See §4.

### 2.3 genie3.sif ships a CPU-ONLY JAX, so its AF2 evaluation never uses the GPU

The most consequential defect found, because **it does not fail — it just runs
on the CPU forever.**

```
jax     0.6.2
jaxlib  0.6.2
jax_cuda12_plugin  MISSING
jax.default_backend() -> 'cpu'
```

From jax 0.4.30 onwards, GPU support lives in the separate
`jax-cuda12-plugin` / `jax-cuda12-pjrt` packages. They are absent, so JAX falls
back to CPU silently. Genie 3's own diffusion and its ProteinMPNN stage are
PyTorch and run on the GPU normally; only the ColabFold/AF2 folding stage is
JAX, and that stage is where the run stops making progress.

Observed: generation finished in 383 s (9.6 s/sample, 40 samples) and inverse
folding in 18 s, then `colabfold_batch` sat at **613% CPU with the GPU at 0%
utilisation and 1.6 GB held** for 33+ minutes with no output. 40 sequences × 5
AF2 models × 20 recycles at 281 residues on CPU would take days. The job was
cancelled at 41 minutes.

Note the shape of the failure: `nvidia-smi` shows an idle GPU and a live,
busy process. Nothing in the log says "CPU". If a JAX-based stage is ever
mysteriously slow, check `jax.default_backend()` first.

#### FIXED 2026-08-27, without rebuilding the image

`launching_scripts/genie3/build_jax_overlay.sh` creates three host overlays that
are bound in at run time; `genie3_jax_env()` in `common/env.sh` emits the flags,
and `run_genie3.sbatch` asserts all three are present before starting. After the
fix, `jax.default_backend()` is `gpu` and AF2 runs at 100% GPU utilisation.

| Overlay | Why | How it is applied |
|---|---|---|
| `jax-cuda12-plugin` + `jax-cuda12-pjrt`, pinned to jaxlib **0.6.2** | supplies the `jax_plugins` namespace package jax uses to find a CUDA backend | `PYTHONPATH` |
| `nvidia-cudnn-cu12` **9.8.0.87** | the plugin requires cuDNN ≥ 9.8; the image has 9.7.1.26 | `LD_LIBRARY_PATH` |
| `nvidia-cuda-nvcc-cu12` **12.8.61** | XLA JIT needs `ptxas` + `nvvm/libdevice`; the image's only copy is vendored inside triton, where XLA does not look | `XLA_FLAGS=--xla_gpu_cuda_data_dir=` |

Three things that are easy to get wrong here:

- **The plugin must match jaxlib exactly.** A mismatch fails at import with an
  unhelpful message. `build_jax_overlay.sh` reads the image's jaxlib version and
  refuses to run if it differs from the pin.
- **cuDNN must go on `LD_LIBRARY_PATH`, not `PYTHONPATH`.** These wheels ship a
  real `nvidia/__init__.py`, so `nvidia` is a *regular* package, not a PEP 420
  namespace. Putting an overlay copy ahead of the image on `PYTHONPATH` would
  shadow the entire `nvidia` package and hide `nvidia.cublas`, `nvidia.nccl` and
  the rest. Overriding at the dynamic-linker level touches only libcudnn.
- **torch shares the same `LD_LIBRARY_PATH`.** Genie 3's diffusion and
  ProteinMPNN stages are torch 2.7.1+cu128, built against cuDNN 9.7.1. cuDNN is
  ABI-stable within major version 9, and 9.8.0.87 was pinned (rather than the
  latest 9.24) to stay close; `verify_jax_gpu.sbatch` explicitly checks that a
  torch cuDNN convolution still works.

The overlays are a **stopgap**. The durable fix is `jax[cuda12]` in the image
definition. Until then, record the overlay paths beside the SIF checksum for any
campaign that used them: the image alone no longer determines the result.

Worth checking the same way in any other image whose AF2 stage seems slow:
`pxdesign.sif` uses jaxlib 0.4.29 (old enough that GPU support is still built
into jaxlib) and demonstrably used the GPU, so it is not affected. The one-line
test is
`singularity exec --nv <sif> python -c "import jax; print(jax.default_backend())"`.

### 2.4 Genie 3 writes `lightning_logs` into a read-only path

`OSError: [Errno 30] Read-only file system: '/opt/genie3/lightning_logs'`.
The runscript hard-`cd`s to `/opt/genie3` because ProteinMPNN, IPSAE, TM-align
and DSSP all have repo-relative paths, and `workflow.py` builds its
`Trainer(...)` without `default_root_dir` or `logger=False`, so Lightning's
TensorBoardLogger does `makedirs('<cwd>/lightning_logs')`. There is no config
knob.

**Fix:** bind writable space over the missing path —
`--bind $OUT/lightning_logs:/opt/genie3/lightning_logs`. Singularity's underlay
creates the mountpoint even though `/opt` is read-only squashfs.
(`--writable-tmpfs` does **not** work: `/opt/genie3` is root-owned and the
overlay preserves its permissions.)

### 2.5 Host `SSL_CERT_FILE` leaks into every container and breaks httpx

FASRC hosts export `SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt`, a RHEL path
absent from these Ubuntu-based images. Singularity passes it straight through
and `httpx`/`requests` die inside `ssl.create_default_context` at import time —
before offline mode is ever consulted. BoltzGen fails this way on every
invocation.

**Fix:** override to `/etc/ssl/certs/ca-certificates.crt`. `common/env.sh` does
this for the whole suite.

### 2.6 `--cleanenv` drops SLURM's `CUDA_VISIBLE_DEVICES`

`--cleanenv` is the right default — host conda and `PYTHONPATH` contaminate
these images — but it also discards the variable SLURM uses to name the
allocated GPU.

**Fix:** `common/env.sh` forwards it as `SINGULARITYENV_CUDA_VISIBLE_DEVICES`.

### 2.7 rfd.sif `eval`s its arguments, so Hydra's `${now:...}` explodes

`hydra.run.dir=.../\${now:%Y-%m-%d}` gives
``eval: 1:812: `%' must follow an expression`` and then a bogus `FATAL: stat ...
no such file or directory` on the first argument, because argument parsing has
already been destroyed. Name the directory from `$SLURM_JOB_ID` instead.

### 2.8 RFdiffusion cannot create its own schedule directory

`model_runners.py` calls `os.mkdir` (not `makedirs`) on
`/app/RFdiffusion/schedules`, which does not exist in the image and could not be
created there anyway.

**Fix:** `inference.schedule_directory_path=<writable dir>` whose parent exists.

### 2.9 The host Python is 3.6.8

`/usr/bin/python3` on the login and compute nodes is Python 3.6.8. Any helper
script that runs outside a container must avoid `from __future__ import
annotations`, `X | None` unions and the walrus operator. `python3.12` exists at
`/usr/bin/python3.12` if you need it, but writing 3.6-compatible helpers is less
fragile.

---

## 3. Counting traps — "40 designs" means something different in each tool

Read this before comparing any two tools' output counts.

| Tool | What "40" produces | The trap |
|---|---|---|
| BoltzGen | `--num_designs 40` backbones | `--budget` is the post-filter set size, and `--filter_biased true` (default) drops composition outliers. `--num_designs 40 --budget 40` has no headroom; intended usage oversamples (e.g. 200 → 40). |
| FreeBindCraft | `max_trajectories: 40` **or** `number_of_final_designs: 40`, whichever trips first | Counts only **successful** trajectories (PDBs in `Trajectory/Relaxed/`); Clashing and LowConfidence runs are moved elsewhere and not counted, so attempts exceed the cap. **Measured: 49 attempted → 34 successful → 326 MPNN designs → 41 accepted.** The *accepted* cap tripped first, so `final_design_stats.csv` and `Accepted/Ranked/` WERE written (41 rows). The opposite branch — trajectory cap first, those files absent, rank from `mpnn_design_stats.csv` — is equally possible on a harder target. A harness must handle both and cannot assume which. |
| PXDesign | `--N_sample 40` | The CLI appends `--min_total_return 40 --max_success_return 40`, so `summary.csv` always has exactly 40 rows — **padded with failed designs** if fewer pass. Real hits are the rows where `AF2-IG-easy-success` / `Protenix-success` are set. |
| Proteina-Complexa | `nsamples × nrepeat_per_sample × replicas` generated, then filtered to `filter_samples_limit` | `dedup_sequence: true` drops identical sequences before top-N, so you can finish with fewer than 40. |
| Protein-Hunter | 40 trajectories × `num_cycles` sequences | ProteinMPNN emits exactly **one** sequence per cycle (`batch_size` hardcoded to 1). A 20%-alanine cap silently excludes designs from `best_*` and writes their `best_iptm` as `NaN`. |
| Genie 3 | `n_sample: 40` **per problem** | With one problem that is the total. Binder length comes from the problem JSON, not the YAML. |
| RFdiffusion | `inference.num_designs=40` backbones | Backbones only, poly-glycine — no sequences. Upstream `inference.cautious=True` skips existing PDBs; the launcher explicitly sets it to `False`, so a fixed-prefix rerun overwrites them. |
| Caliby | `num_seqs_per_pdb × n_structures` | Sequences live only in `seq_des_outputs.csv`; despite the `out_pdb` column name the files are `.cif`. |
| Mosaic | `--n-designs 40` | Straightforward — one sequence per design, appended as it finishes. |

**No tool in this suite writes a FASTA.** BoltzGen, Protein-Hunter, PXDesign and
Caliby all put sequences in a CSV column; RFdiffusion and Genie 3 emit structures.
Any unified post-processing has to extract from CSVs per tool.

---

## 4. RFdiffusion on H100: diagnosis and options

The installed `rfd.sif` is a straight conversion of `rosettacommons/rfdiffusion`
carrying **torch 1.12.1+cu116, DGL 1.0.2+cu116, Python 3.9**. CUDA 11.6 has no
notion of Hopper, and this torch build ships cubins only — no PTX — so forward
JIT is not available either. Every CUDA operation fails (§2.2).

The only previously verified success on this cluster
(`RFDiffusion/outputs/2026-08-06/20-13-16/`) was on an **A100 MIG**, which is
consistent: sm_80 is covered by the shipped cubins. This benchmark reproduced
that — the identical command runs on a full A100.

**This is the one place the benchmark deviates from the `kempner_h100`-only
policy**, because on H100 the tool cannot run at all. RFdiffusion was submitted
as:

```bash
sbatch --partition=kempner --gres=gpu:nvidia_a100-sxm4-40gb:1 \
       --time=06:00:00 run_rfdiffusion.sbatch
```

One caution from doing so: the first A100 node tried returned
`CUDA error: all CUDA-capable devices are busy or unavailable` after successfully
identifying the GPU. Resubmitting to a different node worked, so treat that as a
transient node fault rather than a second incompatibility. (It is *not* caused by
`SINGULARITYENV_CUDA_VISIBLE_DEVICES` forwarding: SLURM's cgroup isolation
renumbers the allocated GPU to index 0 both on the host and inside the container,
which was verified directly.)

Options, best first:

1. **Rebuild the image** against a torch/DGL build with sm_90 support. The
   adjacent `RFDiffusion/rfdiff.def` already describes a newer variant; it was
   never built, and the weights question it raises is moot (see §5.1).
2. **Run RFdiffusion on A100** and everything else on H100. This is now the
   explicit partition/GRES default in `run_rfdiffusion.sbatch`.
3. Drop RFdiffusion. Genie 3 and Proteina-Complexa both cover backbone diffusion
   and both run natively on H100.

---

## 5. Documentation that is wrong

### 5.1 `docs/weights.md` and `CONTAINER_WEIGHTS_REPORT.md` were wrong about rfd.sif — **CORRECTED**

Both say RFdiffusion's weights are external and need a `/models` bind. The
installed image ships **all nine checkpoints** at `/app/RFdiffusion/models`
(`Complex_base_ckpt.pt`, `Complex_beta_ckpt.pt`, `Base_ckpt.pt`,
`Complex_Fold_base_ckpt.pt`, `InpaintSeq_ckpt.pt`, `InpaintSeq_Fold_ckpt.pt`,
`ActiveSite_ckpt.pt`, `Base_epoch8_ckpt.pt`,
`RF_structure_prediction_weights.pt`). No bind is required.

The reports describe `RFDiffusion/rfdiff.def`, the ESMFold2 variant that was
never built. Those are two different artifacts and should be documented
separately.

**Corrected 2026-08-27** in `bindocracy/docs/weights.md`,
`bindocracy/docs/containers/rfdiffusion.md`, and
`binder_design/CONTAINER_WEIGHTS_REPORT.md`, each with a correction block rather
than a silent edit, so the original claim and its refutation stay visible.

The general lesson is worth keeping: **a weight audit that only asks "are the
checkpoint files present" passes all three of the worst problems found here** —
`rfd.sif` has every checkpoint and cannot execute a single CUDA kernel on H100;
`genie3.sif` has every checkpoint and shipped a CPU-only JAX. Audits should
record "ran a smoke job on the target GPU class" alongside "weights present".

`sha256(Complex_base_ckpt.pt)` =
`76e4e260aefee3b582bd76b77ab95d2592e64f00c51bf344968ab9239f3250bc`
`sha256(Complex_beta_ckpt.pt)` =
`5a0b1cafc23c60b1aabcec1e49391986ac4fd02cc1b6b4cc41714ca9fe882e9e`

### 5.2 Mosaic's own launcher scripts pointed at a path that no longer exists — **FIXED**

25 files across `mosaic_setup/mosaic/` referenced

```bash
/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
```

The tree moved under `binder_design/`, so that directory does not exist.
`MOSAIC_SIF` and `MOSAIC_WEIGHTS` resolved to nothing and `mosaic-exec.sh` failed
with "image not found" — loudly, which was lucky. Worse,
`hallucinate/hallucinate.py` hard-coded the same stale prefix for `MSA_PATH`, so
it could not run as-is at all.

**Fixed 2026-08-27.** The rewrite is not a single prefix substitution, because
the tree split in two when it moved:

| Old | New |
|---|---|
| `.../tbush/mosaic_setup/images/` | `.../tbush/binder_design/**images**/` |
| `.../tbush/mosaic_setup/` (everything else) | `.../tbush/binder_design/mosaic_setup/` |

The SIFs went up a level while `weights/`, `dio3_cut/`, `targets/`, `msa/` and
`benchmark/` moved with the tree. Scripts that composed `$SETUP/images/...`
therefore got a new `IMAGES=` variable rather than a prefix rewrite — a blind
substitution would have produced
`binder_design/mosaic_setup/images/mosaic.sif`, which does not exist.

All 25 files re-checked with `bash -n` / `py_compile`, and every resulting
absolute path verified to exist on disk. Note two of them
(`singularity/model-matrix.sbatch`, `singularity/model-matrix-labelled.sbatch`)
already had uncommitted local edits; the path fix was applied on top and does
not touch anything else in them.

`launching_scripts/mosaic/hallucinate_binders.py` remains a parameterised copy
rather than a call into the repo, since the upstream script still hard-codes the
target sequence and MSA path.

### 5.3 Upstream docs that do not match the installed code

- **BoltzGen** README uses `secondary_structure: HHHLLLEEE`, but the parser
  rejects `E`; sheet is `s`/`S`. `total_len` in `constraints` is only honoured
  as `constraints[0]`.
- **Protein-Hunter** README documents `--no_potentials`; it is not a flag and
  passing it aborts argparse. It is derived from whether `--contact_residues`
  is set. The README also says the boltz pipeline writes `high_iptm_cif`; it
  writes `high_iptm_pdb` with `.pdb` files.
- **Proteina-Complexa** `docs/INFERENCE.md` says `target_input: A` (bare chain)
  works. `AtomSelectionStack.from_contig` uses the regex
  `([A-Za-z]+)(\d+)-(\d+)` and raises `ValueError`. Always give an explicit
  range.
- **Caliby** README documents
  `++sampling_cfg_overrides.save_potts_params=true`; that key does not exist in
  this code version and is silently ignored.
- **PXDesign** `--return_topk` is defined but never read — dead code.

### 5.4 `SINGULARITYENV_COMPLEXA_DATA_PATH` cannot work

The Proteina-Complexa helpfile documents it, but Singularity injects user
environment **after** `%environment` runs, so
`export DATA_PATH=${COMPLEXA_DATA_PATH:-…}` never sees it. Setting
`SINGULARITYENV_DATA_PATH` directly does work. An absolute `target_path` in the
target dict sidesteps it entirely, which is what this benchmark does.

The same ordering problem breaks `GENIE3_RUNTIME_CACHE`; use `XDG_CACHE_HOME`.

---

## 6. Measurement caveats

### 6.1 `jobstats` GPU utilisation is unreliable for short jobs

It reported `GPU utilization 0% <-- GPU was not used` for the target-folding job,
whose GPU memory peaked at 76 GB in the same report. Its sampler is too coarse.
`launching_scripts/common/gpu_trace.sh` samples `nvidia-smi` every 15 s instead.

### 6.1b Concurrent runs of one tool cannot share staging state

Submitting a five-point Caliby batch sweep at once killed four of the five jobs
within seconds:

```
FileNotFoundError: [Errno 2] No such file or directory:
  .../outputs/caliby/inputs/dio3_cut_13.pdb
```

`stage_inputs.py` `rmtree`s its staging directory before refilling it, and all
five jobs pointed at the same path, so each deleted the files the others were
mid-copy. Nothing about it is Caliby-specific — any per-run preparation step
that writes to a fixed path has the same failure the moment two runs overlap.

**Fixed** by scoping both the staging directory and the constraint CSV by
`RUN_TAG` (`inputs-<tag>/`, `fixed_target-<tag>.csv`).

The harness lesson is broader than the fix: **the unit of isolation is the run,
not the tool.** Any derived input — a staged directory, a constraint file, an MSA
cache, a Hydra output dir — has to be addressed by run identity, or the first
parameter sweep discovers it the hard way. Note also that the benchmark itself
never caught this, because it ran one job per tool.

### 6.1a A job can report FAILED after succeeding — `set -o pipefail` + `head`

Genie 3's fixed rerun logged `🎉 Run completed`, wrote all 40 sequences, 40
backbones and 200 AF2 outputs — and `sacct` recorded **FAILED, ExitCode 1:0**.

The cause was in *our* launcher, not the tool. The last line was a summary
listing:

```bash
find "$OUT/dio3_cut" -maxdepth 2 | head -30
```

Under `set -euo pipefail`, `head` exits after 30 lines, `find` takes SIGPIPE,
`pipefail` propagates the non-zero status, and `set -e` fails the script — after
every piece of real work has completed. The exit status of a job is the exit
status of its *last* command, so a cosmetic listing can overwrite a successful
run's verdict.

**Fixed** by appending `|| true` to the trailing listings in all five affected
launchers (boltzgen, genie3, proteina_complexa, pxdesign, rfdiffusion).

Two lessons worth keeping: never let a decorative command be the last statement
of a job script, and **do not trust `sacct` State alone** — check the log for the
tool's own completion message before concluding a run failed. Genie 3's real
result would have been discarded on the strength of that FAILED.

### 6.2 GPU-memory figures are allocator reservations, not demand

JAX preallocates ~75% of the device by default, so Mosaic, PXDesign, Genie 3's
AF2 stage and Proteina-Complexa's AF2 reward all report ~76 GB regardless of
what they need. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to make the number
mean something (at some cost in speed and fragmentation). The
Proteina-Complexa launcher does this, because AF2/JAX and Complexa/PyTorch share
one device there and the default would starve torch. Which convention a run used
is recorded in the header of its `gpu_trace.csv`.
