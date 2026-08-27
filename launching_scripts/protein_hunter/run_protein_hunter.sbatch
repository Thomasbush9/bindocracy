#!/bin/bash
#
# Protein-Hunter (Boltz pipeline) — iterative binder design from target SEQUENCE.
# Library: Protein-Hunter/  |  Image: images/protein_hunter.sif
#
# WHAT IT DOES
#   Per design: start from a mostly-X binder sequence, then loop `num_cycles`
#   times of {Boltz-2 co-fold of binder+target -> LigandMPNN redesign of the
#   binder}. Designs whose ipTM/pLDDT clear the thresholds are copied into
#   high_iptm_*. This is a sequence-space search, not diffusion.
#
# WHY boltz AND NOT chai
#   Chai carries an extra ESM-2 3B model per fold, has no --save_dir (output goes
#   to ./results_chai/<jobname> and a rerun silently overwrites), and upstream
#   marks it "under active development". Boltz is cheaper and scriptable.
#
# INPUT IS A SEQUENCE
#   Unlike the rest of the suite this tool needs no target structure. Do NOT
#   pass --template_path: a value that is not an existing file falls into a
#   branch that wgets from RCSB/AlphaFold, which hangs on an offline node.
#
# OFFLINE MSA — see seed_msa_cache.py
#   --msa_mode has only {single, mmseqs}; mmseqs means api.colabfold.com. We
#   pre-seed the ColabFold cache directory that run_mmseqs2 checks first, so
#   `mmseqs` resolves entirely from disk. `single` is the zero-setup fallback,
#   at the cost of folding the 201 aa target with no MSA at all.
#
# Submit:
#     cd .../launching_scripts/protein_hunter && sbatch run_protein_hunter.sbatch
#
#SBATCH --job-name=bd-proteinhunter
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# 8h fits before the 2026-08-27 08:00 maintenance reservation; 12h is the
# comfortable production figure (~240 Boltz-2 predictions + ~200 MPNN calls).
#SBATCH --time=08:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/protein_hunter/logs/ph-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/protein_hunter/logs/ph-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=protein_hunter
OUT="$OUT_ROOT/$TOOL"
HERE="$REPO_ROOT/launching_scripts/$TOOL"
SAVE="$OUT/dio3_cut_boltz"
mkdir -p "$OUT/logs" "$OUT/resources" "$SAVE"

SIF="$IMAGE_ROOT/protein_hunter.sif"
SCRATCH="$(tool_scratch $TOOL)"

# LigandMPNN opens a TemporaryDirectory per call (~200 calls) and the runscript
# runs under `set -eu`, mkdir -p'ing its cache tree under $TMPDIR. An unwritable
# TMPDIR kills the job on its first line.
export TMPDIR="$SCRATCH/tmp"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_PROTEIN_HUNTER_RUNTIME_CACHE="$SCRATCH/cache/ph"
mkdir -p "$SINGULARITYENV_PROTEIN_HUNTER_RUNTIME_CACHE"

# --- Seed the MSA cache before the GPU work ---------------------------------
# The single target chain is chain B (the binder is chain A), so the cache
# directory run_mmseqs2 consults is <save_dir>/0_protein_hunter_design/B_env.
ENV_DIR="$SAVE/0_protein_hunter_design/B_env"
python3 "$HERE/seed_msa_cache.py" \
    --a3m "$TARGET_MSA" --env-dir "$ENV_DIR" --max-seqs 512

# Fail here rather than discovering mid-job that it fell back to the network.
for f in out.tar.gz uniref.a3m bfd.mgnify30.metaeuk30.smag30.a3m; do
    [[ -e "$ENV_DIR/$f" ]] || { echo "FATAL: MSA cache incomplete, missing $f"; exit 1; }
done
echo "MSA cache seeded: $(grep -c '^>' "$ENV_DIR/uniref.a3m") sequences"

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
gpu_trace_start "$OUT/resources/gpu_trace.csv"

# NOTE on flags: --omit_AA is capitalised here and lowercase (--omit_aa) in the
# chai pipeline. There is no --seed anywhere, so this run is NOT reproducible
# and cannot be split across jobs by seed. MPNN emits exactly one sequence per
# cycle (batch_size is hardcoded to 1), so 40 designs x 5 cycles = 200 sequences.
singularity run --cleanenv --nv "$SIF" boltz \
    --name                 "$TARGET_NAME" \
    --mode                 binder \
    --protein_seqs         "$TARGET_SEQ" \
    --msa_mode             mmseqs \
    --num_designs          "$N_BINDERS" \
    --num_cycles           5 \
    --min_protein_length   65 \
    --max_protein_length   120 \
    --percent_X            90 \
    --omit_AA              C \
    --temperature          0.1 \
    --diffuse_steps        200 \
    --recycling_steps      3 \
    --high_iptm_threshold  0.7 \
    --high_plddt_threshold 0.7 \
    --save_dir             "$SAVE" \
    --gpu_id               0

gpu_trace_stop
echo "end=$(date -Is)"
ls -la "$SAVE" || true
