#!/bin/bash
#
# BoltzGen — structure-conditioned de novo binder design.
# Library: boltzgen/  |  Image: images/boltzgen.sif
#
# WHAT IT DOES
#   Diffuses a binder backbone against the target's geometry, inverse-folds it
#   to a sequence, refolds the complex, scores it, and diversity-filters the
#   pool down to --budget designs. The `protein-anything` protocol runs two
#   folding passes (folding + design_folding), which dominates the runtime.
#
# CONFIG
#   dio3_cut_binder.yaml — binder length range 70..90 plus chain A of the
#   predicted DIO3-cut CIF. Length lives in the YAML, NOT on the command line.
#
# --num_designs VS --budget
#   num_designs is how many backbones are generated; budget is how many survive
#   into final_ranked_designs/. They are set equal here, with
#   --filter_biased false, so the benchmark generates exactly 40 and keeps
#   exactly 40 — comparable to the other tools' "40 designs".
#   A PRODUCTION run should oversample instead (the intended usage is e.g.
#   --num_designs 200 --budget 40 with filtering left on), because
#   --filter_biased true is what drops amino-acid-composition outliers.
#
# GPU UTILISATION — READ BEFORE SIZING A CAMPAIGN
#   --diffusion_batch_size "defaults to 1 if --num_designs is less than 100, and
#   10 otherwise". A 40-design job therefore runs at batch size 1 and leaves the
#   GPU half idle (measured: 54% mean). That is an artefact of the benchmark
#   size, not the tool. Override it:
#       RUN_TAG=b10 DIFFUSION_BATCH_SIZE=10 sbatch run_boltzgen.sbatch
#   RUN_TAG redirects output so a sweep cannot overwrite the canonical run.
#
# The two default checkpoints split the work 50/50, so 40 designs means 20 from
# boltzgen1_diverse and 20 from boltzgen1_adherence.
#
# Submit:
#     cd .../bindocracy/launching_scripts/boltzgen && sbatch run_boltzgen.sbatch
#
#SBATCH --job-name=bd-boltzgen
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/boltzgen/logs/boltzgen-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/boltzgen/logs/boltzgen-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=boltzgen
# RUN_TAG isolates a variant run (e.g. a batch-size sweep) from the benchmark
# results. Empty = the canonical run.
RUN_TAG="${RUN_TAG:-}"
OUT="$OUT_ROOT/$TOOL${RUN_TAG:+-$RUN_TAG}"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources" "$OUT/results"

SIF="$IMAGE_ROOT/boltzgen.sif"
SCRATCH="$(tool_scratch $TOOL)"

# TMPDIR must be node-local: Triton's kernel-compile temp dirs cannot be cleaned
# up on Lustre and the resulting Errno 39 kills the job. See node_tmp() in
# common/env.sh. The runscript also runs under `set -eu` and mkdir -p's
# XDG_CACHE_HOME, MPLCONFIGDIR, NUMBA_CACHE_DIR, TORCH_HOME and
# TORCH_EXTENSIONS_DIR underneath BOLTZGEN_RUNTIME_CACHE, so an unwritable value
# there kills the job on its first line, before any useful log.
export TMPDIR="$(node_tmp $TOOL)"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE="$TMPDIR/boltzgen-cache"
mkdir -p "$SINGULARITYENV_BOLTZGEN_RUNTIME_CACHE"

[[ -f "$TARGET_CIF" ]] || { echo "FATAL: target CIF missing: $TARGET_CIF (run 00_target first)"; exit 1; }

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

# Weights are embedded (/opt/boltzgen/model-cache/hub) and the image sets
# HF_HUB_OFFLINE=1 itself, so no network is used. SSL_CERT_FILE is still
# forwarded by env.sh: the host's RHEL bundle path otherwise leaks in and kills
# httpx during import, before offline mode is ever consulted.
singularity run --cleanenv --nv "$SIF" \
    run "$HERE/dio3_cut_binder.yaml" \
    --output        "$OUT/results" \
    --protocol      protein-anything \
    --num_designs   "$N_BINDERS" \
    --budget        "$N_BINDERS" \
    --filter_biased false \
    --devices       1 \
    --num_workers   8 \
    ${DIFFUSION_BATCH_SIZE:+--diffusion_batch_size "$DIFFUSION_BATCH_SIZE"}

gpu_trace_stop
echo "end=$(date -Is)"
find "$OUT/results" -maxdepth 2 -name "*.csv" -o -maxdepth 2 -type d | head -30 || true
