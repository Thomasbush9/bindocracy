#!/usr/bin/env bash
#
# Shared environment for every launcher in launching_scripts/.
#
# Source this, never execute it:
#
#     source /n/holylfs06/.../bindocracy/launching_scripts/common/env.sh
#
# It defines only paths and cluster identity. It deliberately sets no
# tool-specific variable, so a change here can never silently alter what a
# single tool does.

# --- Roots -------------------------------------------------------------------
export BINDER_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
export IMAGE_ROOT="$BINDER_ROOT/images"
export OUT_ROOT="$BINDER_ROOT/outputs"
export REPO_ROOT="$BINDER_ROOT/bindocracy"

# Scratch. Container HOMEs, JAX/XLA caches and torch inductor caches go here,
# never onto the lab share: they are large, high-churn, and worthless to keep.
export SCRATCH_ROOT="/n/netscratch/bsabatini_lab/Everyone/${USER}/bindocracy"

# --- Target ------------------------------------------------------------------
# DIO3 "cut" construct, 201 aa, single chain A, seqids 1..201. The FASTA and the
# precomputed ColabFold-style a3m are the two primitives; the CIF is produced by
# 00_target/ and is what the structure-conditioned tools consume.
#
# NO HOTSPOTS. Every tool that will accept it runs without an epitope
# restraint, matching the lab's previous DIO3 campaigns (`hotspots: ""` in both
# mosaic_setup/mosaic/pipelines/presets/*_dio3.yaml) and letting each generator
# choose its own surface. That is also the only setting under which the tools
# are comparable, since they do not all express hotspots the same way.
export TARGET_NAME=dio3_cut
export TARGET_DIR="$BINDER_ROOT/mosaic_setup/dio3_cut"
export TARGET_FASTA="$TARGET_DIR/dio3_cut.fasta"
export TARGET_MSA="$TARGET_DIR/msa_out/sequences/dio3_cut/msa/DIO3.a3m"
export TARGET_SEQ="DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFMARMSAFQRLVTKYQRDVDFLIIYIEEAHPSDGWVTTDSPYIIPQHRSLEDRVSAARVLQQGAPGCALVLDTMANSSSSAYGAYFERLYVIQSGTIMYQGGRGPDGYQVSELRTWLERYDEQLHGARPRRV"
export TARGET_LEN=201
export TARGET_CHAIN=A

# --- Overlays ----------------------------------------------------------------
# genie3.sif ships jax/jaxlib 0.6.2 with NO jax-cuda12-plugin, so JAX silently
# runs on the CPU and the AF2 evaluation stage never finishes. Two host trees fix
# that without rebuilding the image:
#
#   JAX_CUDA_OVERLAY  jax-cuda12-plugin + jax-cuda12-pjrt, both pinned to the
#                     image's jaxlib 0.6.2. Goes on PYTHONPATH: jax discovers a
#                     CUDA backend through the `jax_plugins` namespace package.
#   CUDNN_OVERLAY     nvidia-cudnn-cu12 9.8.0.87. Goes on LD_LIBRARY_PATH, NOT
#                     PYTHONPATH. The plugin needs cuDNN >= 9.8 and the image has
#                     9.7.1.26, but the image's `nvidia` is a REGULAR package
#                     (it has __init__.py), so shadowing it on PYTHONPATH would
#                     hide nvidia.cublas / nvidia.nccl too. Overriding at the
#                     dynamic-linker level touches only libcudnn. 9.8.0.87 is
#                     pinned rather than latest to stay close to the 9.7.1 that
#                     torch 2.7.1+cu128 was built against; cuDNN is ABI-stable
#                     within major version 9.
#
# Rebuild with launching_scripts/genie3/build_jax_overlay.sh; verify with
# launching_scripts/genie3/verify_jax_gpu.sbatch. This is a STOPGAP — the real
# fix is `jax[cuda12]` in the image definition. Record both overlay paths beside
# the SIF checksum for any campaign that used them: the image alone no longer
# determines the result.
#   NVCC_OVERLAY      nvidia-cuda-nvcc-cu12 12.8.61, pointed at with XLA_FLAGS.
#                     XLA JIT-compiles kernels and needs ptxas plus
#                     nvvm/libdevice; the image only has the copy vendored
#                     inside triton, which XLA does not look at. (The wheel has
#                     no `nvlink`; XLA logs a search for it and then links
#                     through the driver instead, so it is not needed.)
export JAX_CUDA_OVERLAY="$BINDER_ROOT/overlays/genie3-jax-cuda"
export CUDNN_OVERLAY="$BINDER_ROOT/overlays/genie3-cudnn/nvidia/cudnn/lib"
export NVCC_OVERLAY="$BINDER_ROOT/overlays/genie3-cuda-nvcc/nvidia/cuda_nvcc"

# Emit the singularity --env flags that turn genie3.sif's JAX into a GPU JAX.
# Use as:  singularity exec ... $(genie3_jax_env) image.sif ...
# Verified together by launching_scripts/genie3/verify_jax_gpu.sbatch.
genie3_jax_env() {
    printf -- '--env PYTHONPATH=%s --env LD_LIBRARY_PATH=%s ' \
        "$JAX_CUDA_OVERLAY" "$CUDNN_OVERLAY"
    printf -- '--env XLA_FLAGS=--xla_gpu_cuda_data_dir=%s ' "$NVCC_OVERLAY"
    printf -- '--env PATH=%s/bin:/opt/conda/envs/genie3/bin:/usr/local/bin:/usr/bin:/bin ' \
        "$NVCC_OVERLAY"
}

# Produced by launching_scripts/00_target/. Consumed by RFdiffusion,
# FreeBindCraft, BoltzGen, PXDesign, Protein-Hunter, Genie3 and Caliby.
export TARGET_CIF="$OUT_ROOT/_shared/${TARGET_NAME}.cif"
export TARGET_PDB="$OUT_ROOT/_shared/${TARGET_NAME}.pdb"

# --- Campaign size -----------------------------------------------------------
# One job per tool, 40 binders, so that per-tool wall-clock and GPU memory are
# directly comparable. Do not turn these into arrays for this benchmark: an
# array hides the per-design cost behind the scheduler.
export N_BINDERS=40
export BINDER_LENGTH=80

# --- SLURM -------------------------------------------------------------------
export SLURM_ACCOUNT_NAME=kempner_bsabatini_lab
export SLURM_PARTITION_NAME=kempner_h100

# --- Singularity -------------------------------------------------------------
# FASRC hosts export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt, a RHEL path
# that does not exist in these Ubuntu-based images. Singularity passes it
# straight through, and httpx/requests then die inside
# ssl.create_default_context before any work starts — BoltzGen fails this way on
# every invocation. Override with the container's own bundle. Harmless for the
# fully offline images.
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# --cleanenv is the right default — host conda/PYTHONPATH contaminate these
# images — but it also drops the CUDA_VISIBLE_DEVICES that SLURM sets to name
# the allocated device. Forward it explicitly so --cleanenv stays safe.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

# --- Cluster maintenance -----------------------------------------------------
# Reservation `upgrade_stage4` takes essentially every kempner_h100 node from
# 2026-08-27 08:00 to 17:00. SLURM will not START a job whose walltime would
# overlap it, so during that window a long --time silently means "queued until
# tomorrow evening" rather than "runs now". The benchmark scripts therefore
# request walltimes that fit before the reservation; the comment in each script
# records the walltime a production run should use instead.

# Per-tool scratch on the shared filesystem: container HOMEs and weight caches
# that are worth keeping between jobs. Call as: tool_scratch boltzgen
tool_scratch() {
    local d="$SCRATCH_ROOT/$1"
    mkdir -p "$d/cache" "$d/home"
    echo "$d"
}

# Per-job TMPDIR on NODE-LOCAL disk. Call as: TMPDIR="$(node_tmp boltzgen)"
#
# TMPDIR MUST NOT BE ON LUSTRE. Triton compiles every GPU kernel inside a
# tempfile.TemporaryDirectory(), and on /n/netscratch the cleanup rmtree hits
#     OSError: [Errno 39] Directory not empty
# because the distributed filesystem has not finished unlinking the files it was
# just asked to delete. The exception propagates out of compile_module_from_src
# and kills the job on its first kernel — BoltzGen job 42120043 died this way,
# ~9 s into step 1 of 6. Every PyTorch tool here (BoltzGen, Protein-Hunter,
# PXDesign, Proteina-Complexa) JITs Triton kernels and is exposed to it.
#
# /tmp on a compute node is real local disk, so the whole class of problem goes
# away. The cost is that JIT caches do not survive the job, which is a few
# minutes of recompilation and worth paying.
node_tmp() {
    local d="/tmp/bindocracy-${1}-${SLURM_JOB_ID:-$$}"
    mkdir -p "$d"
    echo "$d"
}
