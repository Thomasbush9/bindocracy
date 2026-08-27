#!/bin/bash
#
# Mosaic — hallucination-based binder design against DIO3-cut.
# Library: mosaic_setup/mosaic  |  Image: images/mosaic.sif
#
# WHAT IT DOES
#   Gradient-optimises a soft (PSSM) binder sequence directly through Boltz-2,
#   with a composite loss (interface contact, PAE, ipTM, pLDDT, plus a
#   ProteinMPNN inverse-folding-recovery term), then sharpens it to a discrete
#   sequence in three APGM stages and re-folds it for a ranking score.
#   No target structure is needed: Mosaic folds the target from sequence + MSA
#   as part of the design graph. This is the only tool in the suite that
#   consumes the target as sequence rather than geometry.
#
# WHY NOT AN ARRAY
#   The benchmark wants per-design cost, so all 40 designs run in one process on
#   one GPU. Mosaic is single-device throughout (no pmap/shard_map anywhere in
#   src/), so a second GPU would sit idle; for production, scale with
#   `--array=0-N%10` and --n-designs per task instead.
#
# WALLTIME
#   ~8 min/design measured previously by the lab, plus ~5-10 min to build the
#   models. 40 designs therefore needs roughly 6 h; 12 h is requested to leave
#   headroom, and --max-runtime 11 stops designing before SLURM kills the job so
#   the last sequence is written.
#
# Submit:
#     cd .../bindocracy/launching_scripts/mosaic && sbatch run_mosaic.sbatch
#
#SBATCH --job-name=bd-mosaic
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/mosaic/logs/mosaic-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/mosaic/logs/mosaic-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=mosaic
OUT="$OUT_ROOT/$TOOL"
mkdir -p "$OUT/logs" "$OUT/designs" "$OUT/resources"

MOSAIC_REPO="$BINDER_ROOT/mosaic_setup/mosaic"

# Mosaic's weights live OUTSIDE the SIF, unlike every other image here. If
# MOSAIC_WEIGHTS is wrong the wrapper does not fail — it just skips the bind and
# the run tries to download 2+ GB mid-job. Assert instead.
export MOSAIC_SIF="$IMAGE_ROOT/mosaic.sif"
export MOSAIC_WEIGHTS="$BINDER_ROOT/mosaic_setup/weights"
export MOSAIC_SCRATCH="$(tool_scratch mosaic)"
[[ -d "$MOSAIC_WEIGHTS/boltz" ]] || { echo "FATAL: no boltz weights at $MOSAIC_WEIGHTS/boltz"; exit 1; }

# /jax_cache is bound writable by mosaic-exec.sh. Sharing compiled XLA kernels
# means only the first run pays the several-minute first-iteration compile.
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# NOTE: the 2026-08-26 benchmark run predates this line, so Mosaic is the one
# tool with no gpu_trace.csv — its GPU figures come from jobstats alone, which
# for a JAX process means the ~75% preallocation rather than real demand.
gpu_trace_start "$OUT/resources/gpu_trace.csv"

"$MOSAIC_REPO/singularity/mosaic-exec.sh" python \
    "$REPO_ROOT/launching_scripts/mosaic/hallucinate_binders.py" \
    --target-fasta   "$TARGET_FASTA" \
    --msa-path       "$TARGET_MSA" \
    --binder-length  "$BINDER_LENGTH" \
    --n-designs      "$N_BINDERS" \
    --max-runtime    11 \
    --array-id       0 \
    --save-dir       "$OUT/designs"

gpu_trace_stop
echo "end=$(date -Is)"
