#!/bin/bash
#
# Which library in rfd.sif lacks an sm_90 kernel?
#
# The 40-design run dies with "CUDA error: no kernel image is available for
# execution on the device". The message is asynchronous, so the reported line is
# not the guilty one. This isolates the layers: plain torch tensor ops, then a
# conv/gemm, then DGL — the image carries torch 1.12.1+cu116 and DGL 1.0.2+cu116,
# and CUDA 11.6 predates Hopper entirely, so the only thing that could save
# either is forward JIT from embedded PTX.
#
#     sbatch diagnose_h100.sbatch
#
#SBATCH --job-name=bd-rfd-diag
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/rfdiffusion/logs/diag-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/rfdiffusion/logs/diag-%j.out

set -uo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

singularity exec --cleanenv --nv "$IMAGE_ROOT/rfd.sif" python -c '
import torch, traceback
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

def probe(name, fn):
    try:
        fn(); torch.cuda.synchronize(); print("OK   ", name)
    except Exception as e:
        print("FAIL ", name, "->", type(e).__name__, str(e).splitlines()[0])

probe("tensor copy H2D", lambda: torch.ones(8, device="cuda"))
probe("elementwise add", lambda: torch.ones(1024, device="cuda") + 1)
probe("matmul (cuBLAS)", lambda: torch.randn(512,512,device="cuda") @ torch.randn(512,512,device="cuda"))
probe("conv2d (cuDNN)", lambda: torch.nn.functional.conv2d(
        torch.randn(1,3,32,32,device="cuda"), torch.randn(4,3,3,3,device="cuda")))
probe("softmax", lambda: torch.softmax(torch.randn(64,64,device="cuda"), -1))

try:
    import dgl
    print("dgl", dgl.__version__)
    def dgl_op():
        g = dgl.graph((torch.tensor([0,1,2]), torch.tensor([1,2,0]))).to("cuda")
        g.ndata["h"] = torch.randn(3, 8, device="cuda")
        import dgl.function as fn
        g.update_all(fn.copy_u("h","m"), fn.sum("m","h2"))
        return g.ndata["h2"]
    probe("dgl update_all (SpMM)", dgl_op)
except Exception:
    print("dgl import FAILED"); traceback.print_exc()

try:
    import se3_transformer  # RFdiffusion SE(3) layers
    print("se3_transformer import OK")
except Exception as e:
    print("se3_transformer import:", type(e).__name__, e)
' 2>&1
