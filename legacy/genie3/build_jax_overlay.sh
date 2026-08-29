#!/usr/bin/env bash
#
# Give genie3.sif a GPU-capable JAX without rebuilding the image.
#
# THE PROBLEM
#   genie3.sif ships jax 0.6.2 and jaxlib 0.6.2 but not jax-cuda12-plugin. From
#   jax 0.4.30 onwards the CUDA backend lives in that separate plugin, so
#   jax.default_backend() returns 'cpu'. Genie 3's diffusion and ProteinMPNN
#   stages are PyTorch and are unaffected; its ColabFold/AF2 evaluation stage is
#   JAX and silently runs on the CPU, where it never finishes. Nothing errors.
#
# THE FIX
#   Install the matching plugin wheels into a host directory and put that
#   directory on PYTHONPATH inside the container. jax discovers CUDA through the
#   `jax_plugins` namespace package, which is exactly what these wheels provide,
#   so no code change and no image rebuild is needed.
#
#   --no-deps is deliberate: the wheels declare a pile of nvidia-*-cu12
#   dependencies, but the image already carries a complete CUDA 12.8 stack
#   (cublas 12.8.3, cudnn 9.7.1, nccl 2.26.2, ... from torch 2.7.1+cu128).
#   Installing them again would add gigabytes and risk shadowing the versions
#   torch is linked against. Overlay size with --no-deps is ~350 MB.
#
#   The wheels are built per CPython version, so they MUST be installed with the
#   container's own interpreter (3.10) rather than the host's 3.6.
#
# THIS IS A STOPGAP. The durable fix is `pip install jax[cuda12]==0.6.2` in the
# image definition. Record this overlay alongside the SIF checksum in any
# campaign that used it, because the image alone no longer determines behaviour.
#
# Usage (login node, needs network):
#     ./build_jax_overlay.sh
# Then verify on a GPU:
#     sbatch verify_jax_gpu.sbatch
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"

JAX_VERSION="${JAX_VERSION:-0.6.2}"

# Pin to whatever jaxlib the image actually has: a plugin/jaxlib mismatch fails
# at import with an unhelpful message, so read it rather than assume.
IMAGE_JAXLIB=$(singularity exec --cleanenv "$IMAGE_ROOT/genie3.sif" \
    /opt/conda/envs/genie3/bin/python -c "import jaxlib; print(jaxlib.__version__)")
if [[ "$IMAGE_JAXLIB" != "$JAX_VERSION" ]]; then
    echo "FATAL: image jaxlib is $IMAGE_JAXLIB but this script targets $JAX_VERSION." >&2
    echo "       Set JAX_VERSION=$IMAGE_JAXLIB and re-run." >&2
    exit 1
fi

pip_into() {   # pip_into <target dir> <spec...>
    local target="$1"; shift
    mkdir -p "$target"
    singularity exec --cleanenv --bind "$BINDER_ROOT" "$IMAGE_ROOT/genie3.sif" \
        /opt/conda/envs/genie3/bin/pip install --no-cache-dir --no-deps \
            --target "$target" "$@"
}

# 1. The plugin itself. Goes on PYTHONPATH; supplies the `jax_plugins`
#    namespace package that jax uses to discover a CUDA backend.
pip_into "$JAX_CUDA_OVERLAY" \
    "jax-cuda12-plugin==${JAX_VERSION}" "jax-cuda12-pjrt==${JAX_VERSION}"

# 2. cuDNN. The plugin needs >= 9.8; the image has 9.7.1.26. This tree is used
#    via LD_LIBRARY_PATH and deliberately NOT PYTHONPATH: these wheels ship a
#    real `nvidia/__init__.py`, so putting one on PYTHONPATH ahead of the image
#    would shadow the whole `nvidia` package and hide nvidia.cublas / nccl.
#    Pinned to 9.8.0.87 rather than latest to stay near torch 2.7.1+cu128's
#    9.7.1; cuDNN is ABI-stable within major version 9, and verify_jax_gpu
#    checks torch still works.
pip_into "$(dirname "$(dirname "$(dirname "$CUDNN_OVERLAY")")")" \
    "nvidia-cudnn-cu12==9.8.0.87"

# 3. ptxas + nvvm/libdevice for XLA's JIT, pointed at with
#    --xla_gpu_cuda_data_dir. The image's only copy is vendored inside triton,
#    where XLA does not look.
pip_into "$(dirname "$(dirname "$NVCC_OVERLAY")")" \
    "nvidia-cuda-nvcc-cu12==12.8.61"

echo
for d in "$JAX_CUDA_OVERLAY" "$CUDNN_OVERLAY" "$NVCC_OVERLAY"; do
    printf '  %-70s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
[[ -d "$JAX_CUDA_OVERLAY/jax_plugins" ]] \
    || { echo "FATAL: no jax_plugins/ — jax will not discover the backend"; exit 1; }
[[ -f "$CUDNN_OVERLAY/libcudnn.so.9" ]] \
    || { echo "FATAL: no libcudnn.so.9 in $CUDNN_OVERLAY"; exit 1; }
[[ -x "$NVCC_OVERLAY/bin/ptxas" ]] \
    || { echo "FATAL: no ptxas in $NVCC_OVERLAY/bin"; exit 1; }
echo
echo "All three overlays present. Confirm on a GPU with:"
echo "    sbatch verify_jax_gpu.sbatch"
