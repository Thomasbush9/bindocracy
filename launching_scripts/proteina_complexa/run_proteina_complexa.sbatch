#!/bin/bash
#
# Proteina-Complexa — flow-matching binder generation with AF2 reward search.
# Library: Proteina-Complexa/  |  Image: images/proteina_complexa.sif (40 GB)
#
# WHAT IT DOES
#   Four stages driven by one Hydra pipeline config: `generate` samples binder
#   backbones against the target, `filter` ranks them by an AF2-multimer reward
#   and keeps the top N, `evaluate` refolds the survivors (AF2 + ESMFold
#   designability/codesignability), `analyze` tabulates. The AF2 reward is what
#   makes this expensive: it runs on every generated design, then again on the
#   survivors during evaluate.
#
# TARGET REGISTRATION
#   configs/ is copied out of the read-only image (see README.md in this dir)
#   and configs/targets/targets_dict.yaml carries an appended `dio3_cut` entry.
#   run_step passes --config-path <parent of the yaml>, so the host copy
#   composes correctly and `complexa target list` sees our dict too.
#
# DESIGN-COUNT ARITHMETIC
#   generate writes nsamples x nrepeat_per_sample x replicas structures.
#   40 x 1 x 1 = 40 generated, and filter_samples_limit=40 keeps all of them, so
#   this is exactly 40 designs with no search overhead — comparable to the other
#   tools. A REAL campaign should use search: replicas=2..4 with a filter limit
#   of 40 is the intended best-of-n usage and is where this tool's quality comes
#   from. Note filter.dedup_sequence=true drops identical sequences before
#   top-N, so slightly fewer than 40 can survive.
#
# Submit:
#     cd .../launching_scripts/proteina_complexa && sbatch run_proteina_complexa.sbatch
#
#SBATCH --job-name=bd-proteina
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
# 10h fits before the 2026-08-27 08:00 maintenance reservation; 12h is the
# safer production figure for 40 designs with AF2 in the reward loop.
#SBATCH --time=10:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/proteina_complexa/logs/pcx-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/proteina_complexa/logs/pcx-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=proteina_complexa
OUT="$OUT_ROOT/$TOOL"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources"

SIF="$IMAGE_ROOT/proteina_complexa.sif"
export TMPDIR="$(node_tmp $TOOL)"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_COMPLEXA_RUNTIME_CACHE="$TMPDIR/complexa-cache"
mkdir -p "$SINGULARITYENV_COMPLEXA_RUNTIME_CACHE"

# AF2 is JAX and Complexa is PyTorch, on the same device. JAX otherwise
# preallocates ~75% of the H100 and starves torch. This also makes the GPU-memory
# figure in resources/ mean actual demand rather than allocator reservation.
export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false

[[ -f "$TARGET_PDB" ]] || { echo "FATAL: target PDB missing: $TARGET_PDB"; exit 1; }
grep -q "dio3_cut:" "$HERE/configs/targets/targets_dict.yaml" \
    || { echo "FATAL: dio3_cut not registered in the host config copy"; exit 1; }

# All outputs are written relative to CWD, so the working directory IS the
# output directory here.
cd "$OUT"

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace${RUN_NAME:+-$RUN_NAME}.csv"

# gen_njobs/eval_njobs are GPU counts and must both be 1 on a single-GPU job.
singularity run --cleanenv --nv --pwd "$OUT" "$SIF" \
    design "$HERE/configs/search_binder_local_pipeline.yaml" \
    ++run_name="${RUN_NAME:-dio3_cut_v1}" \
    ++generation.task_name=dio3_cut \
    ++generation.dataloader.dataset.nres.nsamples="$N_BINDERS" \
    ++generation.dataloader.dataset.nres.low=70 \
    ++generation.dataloader.dataset.nres.high=110 \
    ++generation.dataloader.dataset.nrepeat_per_sample=1 \
    ++generation.dataloader.batch_size="${GEN_BATCH_SIZE:-8}" \
    ++generation.search.algorithm=best-of-n \
    ++generation.search.best_of_n.replicas=1 \
    ++generation.filter.filter_samples_limit="$N_BINDERS" \
    ++gen_njobs=1 ++eval_njobs=1 \
    ++ncpus_=16 ++seed=5

gpu_trace_stop
echo "end=$(date -Is)"
find "$OUT/inference" -maxdepth 2 -name "*.csv" 2>/dev/null | head -10 || true
