# Mosaic

Image: `images/mosaic.sif`
Repository and wrapper: `mosaic_setup/mosaic/`

Mosaic optimizes binder sequences against several differentiable structure
backends at once. Unlike most suite images, its weights live outside the SIF.

## Always use the wrapper

The wrapper creates a private container home and binds the external Boltz, AF2,
Protenix, Hugging Face, JAX cache, and output directories correctly.

```bash
BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
MOSAIC_ROOT="$BINDER_ROOT/mosaic_setup/mosaic"

export MOSAIC_SIF="$BINDER_ROOT/images/mosaic.sif"
export MOSAIC_WEIGHTS="$BINDER_ROOT/mosaic_setup/weights"
export MOSAIC_SCRATCH=/absolute/path/to/writable/mosaic_scratch

mkdir -p "$MOSAIC_SCRATCH"
cd "$MOSAIC_ROOT"
./singularity/mosaic-exec.sh python /opt/mosaic/singularity/selftest.py
```

Use `--show` before a new command to print the resolved Singularity invocation
without running it:

```bash
./singularity/mosaic-exec.sh --show python run_design.py --help
```

Do not bypass the wrapper, and do not omit its private `-H` home setup.

## Direct design example

```bash
./singularity/mosaic-exec.sh python run_design.py \
  --target /absolute/path/to/target.fasta \
  --target-msa /absolute/path/to/target.a3m \
  --binder-length 80 \
  --models boltz2,af2 \
  --batch 2 \
  --seed 0 \
  --soft-steps 100 \
  --sharp-steps 25 \
  --out /absolute/path/to/results
```

For production arrays, prefer the repository's `singularity/campaign.sbatch`.
When passing `MODELS` through `sbatch --export`, use `+` between model names,
for example `MODELS=boltz2+af2`; commas delimit exported variables in SLURM.
Use absolute paths and verify the backends listed in the first task's log.

Mosaic is memory-intensive. H100 80 GB is the practical baseline; H200 is
preferable for the ESM-C 6B model. A precomputed target MSA avoids runtime
queries to the ColabFold service.

Generation runs through the harness: see
[the generation workflow](../../workflow/README.md) and
[adding a tool](../adding-a-tool.md). The original plan is kept as
[history](../history/mosaic-generation-plan.md).

## Retained per-input descendant comparison

[`scripts/mosaic_ea_campaign.py`](../../scripts/mosaic_ea_campaign.py) runs the
balanced DIO3 cohort through
[`drivers/mosaic/campaign_ea.py`](../../drivers/mosaic/campaign_ea.py).
This is a separate, frozen optimization phase; it does not overwrite generation
results or write to the campaign databases. Its input is the completed balanced
phase's `reports/candidates.json`: 150 binders per generator per target,
2,100 original inputs, all 75 residues.

This is **not** population-wide selection across input designs. The 2026-10-14
run was stopped after its already-admitted 256 inputs; queued tasks remain held.
Its validated subset exports are under `reports/comparison-50pct/`. The original
full-cohort collector was superseded by a zero-GPU subset collector.

Each input retains its own population of four descendants. Production uses:

- Three cycles of 90 soft and 30 sharp steps; absolute step sizes 0.05/0.025,
  momentum 0.7/0.5, and scales 1/2.
- The current Boltz2 cycle objective, including ProteinMPNN; one in-loop sample
  and one recycle. This is not the lab's separate multi-model recipe.
- Initial independent Dirichlet re-softening at 0.15. At steps 120 and 240,
  `evolution.select` replaces the worst half with sampled survivors;
  `evolution.mutate` commits and independently re-softens every gathered member,
  drawing each member's noise weight from 0.10–0.20.
- Existing cysteine positions fixed exactly; no new cysteines. The MPNN bias
  respects the same position constraints. Cysteines are never changed to alanine.
- Final discrete-sequence Boltz2 ranking with three recycles and six samples.
  The minimum-ranking-loss offspring wins per input. This is internal ranking,
  not held-out validation; ESMFold and Protenix are not used by this phase.

The controller freezes the cohort, baseline predictions, target FASTAs/MSAs,
driver/hooks, source overlay, settings, and source/image/checkpoint hashes.
Canaries use a separate output tree and three shortened cycles, exercising
one C-containing and one C-free input on each target. Production submission
requires validated canary artifacts and two successful terminal Slurm jobs.

```bash
python scripts/mosaic_ea_campaign.py --phase "$NEW_PHASE" prepare \
  --source "$COMPLETED_BALANCED_PHASE" --repo "$BINDOCRACY_REPO"
python "$NEW_PHASE/provenance/mosaic_ea_campaign.py" \
  --phase "$NEW_PHASE" submit-canaries
# After both canaries finish:
python "$NEW_PHASE/provenance/mosaic_ea_campaign.py" \
  --phase "$NEW_PHASE" validate-canaries
python "$NEW_PHASE/provenance/mosaic_ea_campaign.py" \
  --phase "$NEW_PHASE" submit
python "$NEW_PHASE/provenance/mosaic_ea_campaign.py" \
  --phase "$NEW_PHASE" monitor
```

Production consists of two arrays, each throttled to eight single-H100 jobs;
each task handles up to four original inputs in one persistent model process.
The default 12-hour limit bounds each task, not the whole campaign. Record any
measured scheduling-only walltime reduction in `control/resource-amendment.json`;
do not rewrite the frozen scientific manifest after qualification. The
2026-10-14 launch reduced this limit to four hours after the real-target canaries.
A dependent collector
starts only after both arrays terminate and uses one H100 allocation for at most
30 minutes, without executing a GPU model. This preserves the 16-GPU concurrency
cap and allows collection to survive the interactive session. A native job
failure or one-hour log stall makes the live monitor cancel the remaining arrays
and persist a failure report. `provenance/watch_phase.py` independently reports
completion, faults, or stale monitoring.

Per-input output is `results/<arm>/<original_design_id>/`:

- `input.json`, `loss_metadata.json`: original identity, generator/lineage,
  target, settings, seeds, constraints, and objective definitions.
- `cycles.jsonl`: original baseline and every preselection cycle, including
  decoded sequences, losses/components, immutable node IDs, parents, and
  replacement mappings.
- `baseline.npz`, `cycle-*.npz`, `final.npz`: numeric populations, elite states,
  continuation states, evaluation keys, and RNG state.
- `result.json`: all four final offspring, their actual winning checkpoint
  ancestry, internal rankings, selected winner, and artifact checksums.

Candidate IDs are scoped by the phase; population slots are not persistent
lineage identities. Exact sequence duplicates remain explicit offspring records,
not independent discoveries. Completed roots are skipped only after identity,
settings, artifact, and ancestry validation. Interrupted roots restart from their
recorded deterministic seed; cycle checkpoints are retained for analysis, not
automatically resumed mid-root.

The collector validates every root before publishing `reports/final-report.json`,
plus `winners` and `offspring` in JSON, CSV, and FASTA. The original common scores
remain in `inputs/baseline-{candidates,predictions}.json` for paired downstream
comparison. `reports/monitor-status.json`, `control/submission.json`, and
`control/collector.json` retain live status and the Slurm job identities.
