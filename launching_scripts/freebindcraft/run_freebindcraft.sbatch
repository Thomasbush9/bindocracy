#!/bin/bash
#
# FreeBindCraft — BindCraft binder hallucination without PyRosetta.
# Library: FreeBindCraft/  |  Image: images/freebindcraft.sif
#
# WHAT IT DOES
#   AF2 multimer hallucination of a binder against a target PDB (4-stage
#   schedule), then ProteinMPNN redesign of each surviving backbone, then AF2
#   re-prediction and filtering. Relaxation and the interface metrics that
#   normally come from PyRosetta are replaced by OpenMM / FASPR / sc-rs /
#   FreeSASA.
#
# CONFIGS
#   dio3_cut_target.json          the target: chain A of the predicted DIO3-cut
#                                 structure, binder 65-150 aa, no hotspots.
#   dio3_cut_advanced_max40.json  a copy of the image's
#                                 settings_advanced/default_4stage_multimer.json
#                                 with exactly ONE edit: max_trajectories 40.
#                                 The container's settings tree is read-only, so
#                                 the file has to live on the host.
#   filters                       settings_filters/relaxed_filters.json from
#                                 inside the image (absolute default paths were
#                                 baked in at build time, so this resolves from
#                                 any CWD).
#
# THE TWO COUNTERS — read this before interpreting the output
#   `max_trajectories` (advanced) caps *successful* hallucinations: it counts
#   PDBs in Trajectory/Relaxed/, and trajectories that abort as Clashing or
#   LowConfidence are moved elsewhere and never counted. So 40 here means 40
#   good backbones and MORE than 40 attempts.
#   `number_of_final_designs` (target) caps *accepted* designs after filtering.
#   Whichever trips first ends the loop. If the run stops on trajectories with
#   fewer than 40 accepted, `final_design_stats.csv` and `Accepted/Ranked/` are
#   never written — rank from `mpnn_design_stats.csv` instead. That is expected
#   here, not a failure.
#
# RESUME
#   The loop skips any design whose PDB already exists under Trajectory*/, so
#   re-submitting into the same design_path continues instead of redoing work.
#   That also means a *completed* 40-trajectory run exits immediately on
#   re-submission.
#
# Submit:
#     cd .../bindocracy/launching_scripts/freebindcraft && sbatch run_freebindcraft.sbatch
#
#SBATCH --job-name=bd-freebindcraft
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# 10h fits before the 2026-08-27 08:00 maintenance reservation. A PRODUCTION run
# should ask for 36:00:00 — 40 successful trajectories at 10-30 min each is
# 7-20 h, plus the rejected attempts that do not count towards the cap. Because
# the loop is resume-safe, a walltime kill here costs nothing but a resubmit.
#SBATCH --time=10:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/freebindcraft/logs/fbc-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/freebindcraft/logs/fbc-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=freebindcraft
OUT="$OUT_ROOT/$TOOL"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources"

SIF="$IMAGE_ROOT/freebindcraft.sif"
SCRATCH="$(tool_scratch $TOOL)"
export TMPDIR="$SCRATCH/tmp"

# bindcraft.py raises its own soft fd limit, but can only reach the inherited
# HARD limit. Below ~65536 the JAX/OpenMM combination runs out of descriptors
# partway through a campaign.
ulimit -n 65536 || echo "WARNING: could not raise fd limit; hard limit is $(ulimit -Hn)"

[[ -f "$TARGET_PDB" ]] || { echo "FATAL: target PDB missing: $TARGET_PDB (run 00_target first)"; exit 1; }

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

# --nv is mandatory: without a visible GPU the wrapper exits "No GPU device found".
# The OpenCL JIT emits a "Failed to read file: /tmp/dep-*.d" line for essentially
# every kernel; it is harmless and would otherwise bury the real log.
singularity exec --cleanenv --nv "$SIF" bindcraft \
    --settings "$HERE/dio3_cut_target.json" \
    --advanced "$HERE/dio3_cut_advanced_max40.json" \
    --filters  /opt/freebindcraft/settings_filters/relaxed_filters.json \
    --no-pyrosetta \
    2>&1 | stdbuf -oL grep -v -E 'Failed to read file: .*/dep-[0-9a-fA-F]+\.d' || true

gpu_trace_stop
echo "end=$(date -Is)"
ls -la "$OUT" || true
