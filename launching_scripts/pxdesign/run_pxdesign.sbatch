#!/bin/bash
#
# PXDesign — diffusion binder generation with AF2-IG and Protenix confidence filters.
# Library: PXDesign/  |  Image: images/pxdesign.sif
#
# WHAT IT DOES
#   `pipeline` = generate binder backbones by diffusion against the target, then
#   score them with AF2-IG (complex + monomer) and, in `extended` mode, with
#   Protenix as a second orthogonal filter. Output is a ranked summary plus the
#   structures that passed each filter.
#
# THREE FLAG TRAPS, all deliberate below
#   1. --preset defaults to `custom`, not `extended`, despite the docstring.
#      `custom` configures NO filters. It must be passed explicitly.
#   2. The CLI emits every shared option unconditionally, so the click defaults
#      (--eta_type const --eta_min 2.5 --eta_max 2.5) silently overwrite the
#      config's intended piecewise_65 / 1.0 / 2.5 schedule. Passing them back
#      explicitly is what actually gets the intended sampling.
#   3. --N_sample 40 does NOT mean 40 good binders. The CLI appends
#      --min_total_return 40 --max_success_return 40, so summary.csv always has
#      exactly 40 rows, PADDED WITH FAILURES if fewer pass. Read the
#      `AF2-IG-easy-success` / `Protenix-success` columns to find real hits.
#      Upstream targets 10,000+ designs to land 10-100 dual-filter passes, so a
#      40-design job is a pilot, not a campaign.
#
# GPU ARCHITECTURE
#   Pinned JAX 0.4.29 and a custom Protenix LayerNorm kernel support sm_70-sm_90
#   only. H100 (sm_90) is fine; a Blackwell node would fail. Do not relax the
#   partition constraint.
#
# Submit:
#     cd .../launching_scripts/pxdesign && sbatch run_pxdesign.sbatch
#
#SBATCH --job-name=bd-pxdesign
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# 10h fits before the 2026-08-27 08:00 maintenance reservation; 12h is the
# safer production figure (the first run also pays one-time JIT kernel builds).
#SBATCH --time=10:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/pxdesign/logs/pxd-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/pxdesign/logs/pxd-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=pxdesign
OUT="$OUT_ROOT/$TOOL"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
WORK="$OUT/run01"
mkdir -p "$OUT/logs" "$OUT/resources" "$WORK"

SIF="$IMAGE_ROOT/pxdesign.sif"
# Node-local TMPDIR: this image JITs custom kernels (--use_fast_ln,
# --use_deepspeed_evo_attention) via Triton/CUTLASS, and Triton's temp-dir
# cleanup fails on Lustre with Errno 39. See node_tmp() in common/env.sh.
export TMPDIR="$(node_tmp $TOOL)"
export SINGULARITYENV_TMPDIR="$TMPDIR"
# JIT-compiled torch extensions land here. Default is ~/.cache/pxdesign, which
# would put first-run kernel builds on the home quota.
export SINGULARITYENV_PXDESIGN_CACHE="$TMPDIR/pxdesign-cache"
mkdir -p "$SINGULARITYENV_PXDESIGN_CACHE"

MSA_DIR="$OUT_ROOT/_shared/msa/$TARGET_NAME/0"
[[ -f "$TARGET_CIF" ]] || { echo "FATAL: target CIF missing: $TARGET_CIF"; exit 1; }
[[ -f "$MSA_DIR/non_pairing.a3m" && -f "$MSA_DIR/pairing.a3m" ]] \
    || { echo "FATAL: MSA dir incomplete: $MSA_DIR (run make_pxdesign_msa.py)"; exit 1; }

# Belt and braces for the Protenix filter stage: ptx.py calls
# populate_msa_with_cache, which shells out to `protenix msa` (network) for any
# entity that needs an MSA and is not cached. It *should* skip here — the binder
# has use_msa=False and the target's precomputed dir is propagated — but that
# propagation depends on asym-set matching. Seeding the cache by sequence makes
# the offline path unconditional. The key is the stripped, uppercased sequence.
cd "$WORK"
mkdir -p "$WORK/msa_cache"
python3 -c "
import json
seq = ''.join(l.strip() for l in open('$TARGET_FASTA') if not l.startswith('>')).upper()
json.dump({seq: '$MSA_DIR'}, open('$WORK/msa_cache/cache.json', 'w'))
print('seeded msa_cache/cache.json for %d aa target' % len(seq))
"

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

singularity run --cleanenv --nv --pwd "$WORK" "$SIF" \
    pipeline \
    --preset  extended \
    -i        "$HERE/dio3_cut.yaml" \
    -o        "$WORK/out" \
    --N_sample "$N_BINDERS" \
    --N_step  400 \
    --dtype   bf16 \
    --use_fast_ln True \
    --use_deepspeed_evo_attention True \
    --eta_type piecewise_65 --eta_min 1.0 --eta_max 2.5

gpu_trace_stop
echo "end=$(date -Is)"
find "$WORK/out/design_outputs" -maxdepth 2 | head -30 || true
