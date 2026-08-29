#!/bin/bash
#
# Does the jax-cuda12 overlay give genie3.sif a GPU-capable JAX?
#
# genie3.sif ships jax/jaxlib 0.6.2 with no jax-cuda12-plugin, so
# jax.default_backend() is 'cpu' and the ColabFold/AF2 evaluation stage runs on
# the CPU forever (docs/known-issues.md §2.3). overlays/genie3-jax-cuda holds the
# matching plugin + pjrt wheels, installed with --no-deps because the image
# already carries a complete CUDA 12.8 stack from torch 2.7.1+cu128.
#
#     sbatch verify_jax_gpu.sbatch
#
#SBATCH --job-name=bd-genie3-jaxchk
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/genie3/logs/jaxchk-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/genie3/logs/jaxchk-%j.out

set -uo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "=============== WITHOUT the overlay (expected: cpu) ==============="
singularity exec --cleanenv --nv "$IMAGE_ROOT/genie3.sif" \
    /opt/conda/envs/genie3/bin/python -c "
import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())" 2>&1 | tail -3

echo
echo "=============== WITH the overlay (expected: gpu) ==============="
# The conv below is the part that actually matters: plain matmul goes through
# cuBLAS and passes even with the image's too-old cuDNN, while AF2's real work
# hits cuDNN and fails with "DNN library initialization failed".
singularity exec --cleanenv --nv --bind "$BINDER_ROOT" $(genie3_jax_env) \
    "$IMAGE_ROOT/genie3.sif" \
    /opt/conda/envs/genie3/bin/python -c "
import jax, jax.numpy as jnp
print('backend:', jax.default_backend())
print('devices:', jax.devices())
x = jnp.ones((2048, 2048))
print('matmul  ok, sum =', float((x @ x).sum()))
c = jax.lax.conv_general_dilated(jnp.ones((1,1,32,32)), jnp.ones((1,1,3,3)),
                                 (1,1), 'SAME')
print('cuDNN conv ok, sum =', float(c.sum()))
" 2>&1 | tail -6

echo
echo "=============== torch still healthy with the newer cuDNN? ==============="
# Genie 3's diffusion and ProteinMPNN stages are torch, and they inherit the same
# LD_LIBRARY_PATH. torch 2.7.1+cu128 links cuDNN 9.7.1; this confirms 9.8.0.87
# does not break it.
singularity exec --cleanenv --nv --bind "$BINDER_ROOT" $(genie3_jax_env) \
    "$IMAGE_ROOT/genie3.sif" \
    /opt/conda/envs/genie3/bin/python -c "
import torch
print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version())
a = torch.randn(1,1,32,32, device='cuda')
w = torch.randn(1,1,3,3, device='cuda')
print('torch cuDNN conv ok, sum =', float(torch.nn.functional.conv2d(a,w).sum()))
" 2>&1 | tail -4
