#!/bin/bash
#
# RFdiffusion — binder backbone diffusion against a target.
# Library: RFDiffusion/  |  Image: images/rfd.sif
#
# WHAT IT DOES
#   Diffuses an 80-residue binder backbone against the full 201-residue target.
#   Output is BACKBONE ONLY with a poly-glycine binder — no sequence. Turning
#   these into binders needs an inverse-folding pass; in this suite that is
#   Caliby (see launching_scripts/caliby), which is run on RFdiffusion's output.
#
# WEIGHTS ARE EMBEDDED — the existing docs are wrong
#   docs/weights.md and CONTAINER_WEIGHTS_REPORT.md say rfd.sif has external
#   weights needing a /models bind. That describes RFDiffusion/rfdiff.def, the
#   ESMFold2 variant that was never built. The INSTALLED image is a conversion
#   of rosettacommons/rfdiffusion and ships all nine checkpoints at
#   /app/RFdiffusion/models. No bind is required.
#
# THE CHECKPOINT TRAP — the most important line in this script
#   ckpt_override_path is NOT optional here. RFdiffusion only auto-selects
#   Complex_base_ckpt.pt when ppi.hotspot_res is not None. This benchmark runs
#   without hotspots, so auto-selection would silently fall back to
#   Base_ckpt.pt — the MONOMER model — and produce plausible-looking garbage.
#   Running without hotspots is otherwise legitimate: the complex model was
#   trained with 80-100% of hotspots masked, so the all-masked case is
#   in-distribution. The cost is no epitope control.
#
# THIS IMAGE CANNOT RUN ON H100 — SUBMIT IT TO A100
#   Measured, not predicted. On kempner_h100 the run dies with
#       RuntimeError: CUDA error: no kernel image is available for execution
#   and torch reports
#       arch_list = sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86
#   with NO compute_* PTX entry, so there is nothing for the driver to JIT
#   forward to sm_90. The image carries torch 1.12.1+cu116 / DGL 1.0.2+cu116 /
#   Python 3.9, and CUDA 11.6 predates Hopper entirely.
#
#   The SBATCH lines below therefore describe the H100 default used by the rest
#   of the suite, and are OVERRIDDEN on the command line for this one tool:
#
#       sbatch --partition=kempner --gres=gpu:nvidia_a100-sxm4-40gb:1 \
#              --time=06:00:00 run_rfdiffusion.sbatch
#
#   sm_80 is covered by the shipped cubins, and 40 GB is ample for a 281-residue
#   complex. The proper fix is to rebuild the image against a torch/DGL with
#   sm_90 support; see docs/known-issues.md §4.
#
#   Run diagnose_h100.sbatch to re-check after any image rebuild.
#
# Submit:
#     cd .../launching_scripts/rfdiffusion && sbatch run_rfdiffusion.sbatch
#
#SBATCH --job-name=bd-rfdiffusion
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/rfdiffusion/logs/rfd-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/rfdiffusion/logs/rfd-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=rfdiffusion
OUT="$OUT_ROOT/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources" "$OUT/schedules" "$OUT/designs"

# NUM_DESIGNS=1 for the smoke test; defaults to the benchmark's 40.
NUM_DESIGNS="${NUM_DESIGNS:-$N_BINDERS}"

SIF="$IMAGE_ROOT/rfd.sif"
SCRATCH="$(tool_scratch $TOOL)"
export TMPDIR="$(node_tmp $TOOL)"

[[ -f "$TARGET_PDB" ]] || { echo "FATAL: target PDB missing: $TARGET_PDB"; exit 1; }

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} num_designs=$NUM_DESIGNS start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

# schedule_directory_path MUST be writable and its parent must exist:
# model_runners.py calls os.mkdir (not makedirs) and would otherwise try to
# create /app/RFdiffusion/schedules inside the read-only image.
# hydra.run.dir is overridden because Hydra's default is `outputs/<timestamp>`
# RELATIVE TO CWD, which would litter the submit directory. It is named from the
# SLURM job id and NOT from Hydra's ${now:%Y-%m-%d} interpolation: this image's
# runscript `eval`s its arguments, so the shell reaches the `%` first and the
# whole command line dies with "`%` must follow an expression".
# contigs `A1-201/0 80-80`: whole target chain, chain break, then a fixed
# 80-residue binder. Use `70-90` in place of `80-80` to sample the length.
singularity run --cleanenv --nv \
    --home "$SCRATCH/home:/home/rfd" \
    --bind "$BINDER_ROOT" \
    "$SIF" \
    inference.input_pdb="$TARGET_PDB" \
    inference.output_prefix="$OUT/designs/dio3_cut" \
    inference.num_designs="$NUM_DESIGNS" \
    inference.model_directory_path=/app/RFdiffusion/models \
    inference.ckpt_override_path=/app/RFdiffusion/models/Complex_base_ckpt.pt \
    inference.schedule_directory_path="$OUT/schedules" \
    'contigmap.contigs=[A1-201/0 80-80]' \
    denoiser.noise_scale_ca=0 \
    denoiser.noise_scale_frame=0 \
    hydra.run.dir="$OUT/hydra/job_${SLURM_JOB_ID:-manual}"

gpu_trace_stop
echo "end=$(date -Is)"
ls "$OUT/designs"/*.pdb 2>/dev/null | wc -l | xargs echo "backbones written:" || true
