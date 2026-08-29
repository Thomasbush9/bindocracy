#!/bin/bash
#
# Caliby — inverse folding: RFdiffusion backbones -> binder sequences.
# Library: caliby/  |  Image: images/caliby.sif
#
# WHAT IT DOES
#   Reads each target+binder complex, holds the target chain fixed, and designs
#   the binder chain by annealed Potts MCMC. This is the step that turns
#   RFdiffusion's poly-glycine backbones into actual binders — RFdiffusion emits
#   geometry only, so without an inverse-folding pass it contributes no
#   sequences to the benchmark at all.
#
# CHECKPOINT CHOICE IS A CORRECTNESS ISSUE, NOT A PREFERENCE
#   `caliby` (the default) and `soluble_caliby` are documented as trained on
#   MONOMERS ONLY. `soluble_caliby_v1` is the only sequence-design checkpoint
#   trained on interfaces, so it is the only defensible choice for
#   binder-target complexes. Using the default here would be a methodological
#   error that produces perfectly plausible output.
#
# HOW CHAINS ARE HELD FIXED
#   There is no fixed_chains/design_chains key. It is done through
#   pos_constraint_csv, where a BARE CHAIN LETTER fixes that whole chain.
#   Anything not listed is redesigned. A pdb_key missing from the CSV does not
#   error — Caliby silently redesigns the entire complex, target included — so
#   stage_inputs.py generates the CSV from the directory listing and asserts the
#   counts match.
#
# ONE SEQUENCE PER BACKBONE
#   num_seqs_per_pdb=1 over 40 backbones gives exactly 40 binder sequences,
#   matching the other tools. Caliby's real lever is raising it: cost scales
#   linearly in num_seqs_per_pdb x potts_sweeps because the samples are a
#   sequential loop over one precomputed set of Potts parameters, while the
#   expensive encoder runs once per BATCH of structures. For production, raise
#   num_seqs_per_pdb and batch_size together.
#
# Submit (after RFdiffusion has finished):
#     cd .../launching_scripts/caliby && sbatch run_caliby.sbatch
#
#SBATCH --job-name=bd-caliby
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/caliby/logs/caliby-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/caliby/logs/caliby-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh
source "$REPO_ROOT/launching_scripts/common/gpu_trace.sh"

TOOL=caliby
RUN_TAG="${RUN_TAG:-run1}"
export OUT="$OUT_ROOT/$TOOL"   # exported: the verification heredoc below reads it
HERE="$REPO_ROOT/launching_scripts/$TOOL"
mkdir -p "$OUT/logs" "$OUT/resources"

SIF="$IMAGE_ROOT/caliby.sif"
export TMPDIR="$(node_tmp $TOOL)"

SRC="$OUT_ROOT/rfdiffusion/designs"
[[ -d "$SRC" ]] || { echo "FATAL: no RFdiffusion output at $SRC — run rfdiffusion first"; exit 1; }

# Clean input directory + fixed-chain CSV. See stage_inputs.py for why both are
# mandatory rather than conveniences.
#
# BOTH ARE SCOPED BY RUN_TAG. stage_inputs.py rmtree's its staging directory
# before refilling it, so two concurrent runs sharing one path delete each
# other's files mid-copy -- four of a five-job sweep died that way with
# "FileNotFoundError ... inputs/dio3_cut_13.pdb". Per-run state must be per-run.
STAGE="$OUT/inputs-$RUN_TAG"
CSV="$OUT/fixed_target-$RUN_TAG.csv"
python3 "$HERE/stage_inputs.py" \
    --src "$SRC" --pattern 'dio3_cut_*.pdb' \
    --stage-dir "$STAGE" \
    --csv "$CSV" \
    --fixed-chain B

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export RUN_TAG
gpu_trace_start "$OUT/resources/gpu_trace${RUN_TAG:+-$RUN_TAG}.csv"

# potts_temperature needs `++` (it lives in configs/seq_des/inference.yaml, not
# in seq_des.yaml's override block). The default 0.01 is very cold and would
# give near-identical sequences; upstream recommends 0.1-0.2 for diversity.
# hydra.run.dir is overridden because its default is the RELATIVE path
# `hydra_outputs`, created under whatever CWD the job happens to have.
# run_self_consistency_eval is left false: AF2 is not installed in this image.
singularity run --cleanenv --nv --bind "$BINDER_ROOT" \
    --env CALIBY_RUNTIME_CACHE="$TMPDIR/caliby-cache" \
    "$SIF" seq-design \
    ckpt_name_or_path=soluble_caliby_v1 \
    input_cfg.pdb_dir="$STAGE" \
    pos_constraint_csv="$CSV" \
    sampling_cfg_overrides.num_seqs_per_pdb="${NUM_SEQS:-1}" \
    sampling_cfg_overrides.batch_size="${BATCH_SIZE:-4}" \
    'sampling_cfg_overrides.omit_aas=[C]' \
    ++sampling_cfg_overrides.potts_sampling_cfg.potts_temperature=0.1 \
    num_workers=2 \
    seed=0 \
    out_dir="$OUT/$RUN_TAG" \
    hydra.run.dir="$OUT/$RUN_TAG/hydra"

gpu_trace_stop
echo "end=$(date -Is)"

# Free assertion that the target really was held fixed: chain 0 of the designed
# sequence must be byte-identical to chain 0 of the input sequence. Caliby joins
# chains with ':' in alphabetical order of chain id, so index 0 is chain A.
# RFdiffusion makes chain A the BINDER, so index 1 is the fixed target here.
python3 - <<'PY'
import csv, os, sys
p = os.path.join(os.environ["OUT"], os.environ.get("RUN_TAG", "run1"),
                 "seq_des_outputs.csv")
if not os.path.exists(p):
    sys.exit("no seq_des_outputs.csv at %s" % p)
rows = list(csv.DictReader(open(p)))
bad = [r["example_id"] for r in rows
       if r["seq"].split(":")[1] != r["input_seq"].split(":")[1]]
print("designs: %d   target-chain unchanged: %d" % (len(rows), len(rows) - len(bad)))
if bad:
    print("WARNING: target chain was altered in %d design(s): %s" % (len(bad), bad[:5]))
PY
