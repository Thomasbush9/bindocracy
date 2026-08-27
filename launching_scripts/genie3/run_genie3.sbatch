#!/bin/bash
#
# Genie 3 — backbone diffusion + ProteinMPNN + AF2 evaluation, as one workflow.
# Library: genie3/  |  Image: images/genie3.sif
#
# WHAT IT DOES
#   `genie3 run` = generate -> evaluate -> evaluate --reduce, as three CHILD
#   PROCESSES, so torch and JAX never hold VRAM at the same time. Generation
#   produces UNK backbones (chain A = binder, chain B = target); ProteinMPNN
#   assigns sequences; AF2-multimer refolds them; the reducer applies the v0
#   success filters (complex scRMSD < 2.5 A, binder pTM > 0.8, min interface PAE
#   < 1.5, hotspot coverage).
#
# HOTSPOTS ARE MANDATORY HERE
#   Unlike every other tool in this benchmark, Genie 3 has no hotspot-free
#   binder mode — the reducer reads target_interface_residues['hotspot']
#   unconditionally. The set A110/A112/A131 is the solvent-exposed rim of the
#   DIO3 catalytic pocket (the catalytic Cys65 itself is buried: 23 neighbours
#   within 10 A against a chain median of 17), so a binder there would occlude
#   substrate access. This is a defensible default, NOT an experimentally
#   mapped epitope — revisit it before a real campaign, and note that it makes
#   Genie 3's designs not strictly comparable to the hotspot-free runs.
#
# TWO PATH TRAPS
#   1. --log-dir is effectively mandatory. The default is `logs/runs` relative
#      to the container's CWD /opt/genie3, which is read-only.
#   2. XDG_CACHE_HOME must be redirected AND the AF2 params symlinked into it.
#      colabfold resolves its data dir from XDG_CACHE_HOME; left at the default
#      it finds an empty directory and tries to download ~4 GB of AF2 params.
#      GENIE3_RUNTIME_CACHE does NOT work for this — the image expands it before
#      Singularity injects user env.
#   3. The image's JAX is CPU-only and must be overlaid. jax/jaxlib 0.6.2 ship
#      without jax-cuda12-plugin, so jax.default_backend() is 'cpu' and the AF2
#      stage runs on the CPU where it never finishes — with no error. Three host
#      overlays fix it (plugin, cuDNN >= 9.8, ptxas/libdevice); genie3_jax_env()
#      in common/env.sh emits the flags. Build them once with
#      build_jax_overlay.sh and confirm with verify_jax_gpu.sbatch.
#   4. /opt/genie3/lightning_logs must be bind-mounted to writable space.
#      The runscript hard-cds to /opt/genie3 (ProteinMPNN, IPSAE, TM-align and
#      DSSP all have repo-relative paths), and workflow.py builds its Trainer
#      without default_root_dir or logger=False, so Lightning's TensorBoardLogger
#      does makedirs('<cwd>/lightning_logs') on a read-only path and the run dies
#      with Errno 30 before generating anything. There is no config knob for it;
#      binding over the missing path is the least invasive fix (Singularity's
#      underlay creates the mountpoint).
#
# Submit:
#     cd .../launching_scripts/genie3 && sbatch run_genie3.sbatch
#
#SBATCH --job-name=bd-genie3
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/genie3/logs/genie3-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/genie3/logs/genie3-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=genie3
OUT="$OUT_ROOT/$TOOL"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources"

SIF="$IMAGE_ROOT/genie3.sif"
export TMPDIR="$(node_tmp $TOOL)"

# Redirect the cache, then point colabfold at the AF2 params already inside the
# image. The symlink target is a path INSIDE the container, so it must be
# created from a container process, not from the host.
CACHE="$OUT/_cache"
mkdir -p "$CACHE/colabfold"
singularity exec --cleanenv --bind "$BINDER_ROOT" "$SIF" \
    ln -sfn /opt/genie3/packages/.cache/xdg-cache/colabfold/params "$CACHE/colabfold/params"
singularity exec --cleanenv --bind "$BINDER_ROOT" "$SIF" \
    test -e "$CACHE/colabfold/params/params_model_1_multimer_v3.npz" \
    || { echo "FATAL: AF2 params not visible at $CACHE/colabfold/params"; exit 1; }

[[ -f "$OUT_ROOT/_shared/genie3_dataset/problems/dio3_cut.json" ]] \
    || { echo "FATAL: problem set missing (run prepare_problemset.py)"; exit 1; }

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

mkdir -p "$OUT/lightning_logs"

# The image's JAX has no CUDA plugin, so without these the AF2 evaluation stage
# runs on the CPU and never finishes. See genie3_jax_env() in common/env.sh and
# docs/known-issues.md §2.3. Assert rather than silently fall back to CPU: that
# failure mode costs a whole job before anyone notices.
[[ -d "$JAX_CUDA_OVERLAY/jax_plugins" ]] \
    || { echo "FATAL: JAX CUDA overlay missing. Run genie3/build_jax_overlay.sh"; exit 1; }
[[ -f "$CUDNN_OVERLAY/libcudnn.so.9" ]] \
    || { echo "FATAL: cuDNN overlay missing at $CUDNN_OVERLAY"; exit 1; }
[[ -x "$NVCC_OVERLAY/bin/ptxas" ]] \
    || { echo "FATAL: ptxas missing at $NVCC_OVERLAY/bin"; exit 1; }

singularity run --cleanenv --nv --bind "$BINDER_ROOT" \
    --bind "$OUT/lightning_logs:/opt/genie3/lightning_logs" \
    --env XDG_CACHE_HOME="$CACHE" \
    --env TMPDIR="$TMPDIR" \
    $(genie3_jax_env) \
    "$SIF" \
    run -c "$HERE/experiment.yaml" \
        --num-devices 1 \
        --log-dir "$OUT/logs/runs" \
        --verbose

gpu_trace_stop
echo "end=$(date -Is)"
find "$OUT/dio3_cut" -maxdepth 2 2>/dev/null | head -30 || true
