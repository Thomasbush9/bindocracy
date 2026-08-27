#!/bin/bash
#
# Fold DIO3-cut once and publish the result as both CIF and PDB.
#
# WHY THIS EXISTS: the campaign target is given only as a sequence + MSA, but
# most of the suite is structure-conditioned — RFdiffusion, FreeBindCraft,
# BoltzGen, PXDesign, Protein-Hunter and Caliby all want a target *structure*.
# Folding it once here means every tool designs against the same geometry, so
# the comparison is about the design method and not about which folder each
# tool happened to call internally.
#
# Uses Mosaic's predict_target.py, which builds `target_only_features` rather
# than `binder_features`. That distinction matters: binder_features stubs
# sidechains to UNK/G because they are not differentiably defined for a soft
# sequence, and a target with no sidechains is useless as a docking scaffold.
#
# Submit:
#     cd .../bindocracy/launching_scripts/00_target
#     sbatch predict_target.sbatch
#
#SBATCH --job-name=bd-target
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/_shared/logs/target-%j.out
#SBATCH --error=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/outputs/_shared/logs/target-%j.out

set -euo pipefail
source /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/bindocracy/launching_scripts/common/env.sh

MOSAIC_REPO="$BINDER_ROOT/mosaic_setup/mosaic"

# mosaic-exec.sh needs these three; it binds the external weight tree onto the
# in-container HOME. Mosaic is the one image in the suite whose weights are NOT
# embedded, so an unset MOSAIC_WEIGHTS silently degrades to "download at run
# time" rather than failing.
export MOSAIC_SIF="$IMAGE_ROOT/mosaic.sif"
export MOSAIC_WEIGHTS="$BINDER_ROOT/mosaic_setup/weights"
export MOSAIC_SCRATCH="$(tool_scratch mosaic)"

export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

mkdir -p "$OUT_ROOT/_shared"

echo "host=$(hostname -s) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$MOSAIC_REPO/singularity/mosaic-exec.sh" python "$MOSAIC_REPO/predict_target.py" \
    --target "$TARGET_FASTA" \
    --target-msa "$TARGET_MSA" \
    --out "$TARGET_CIF"

# Most of the suite (RFdiffusion, FreeBindCraft, ProteinMPNN-family tools) reads
# PDB, not mmCIF. Convert with gemmi from the same image so the atom naming and
# chain id come from one place.
"$MOSAIC_REPO/singularity/mosaic-exec.sh" python -c "
import gemmi, sys
st = gemmi.read_structure('$TARGET_CIF')
st.setup_entities()
st.remove_ligands_and_waters()
for model in st:
    for chain in model:
        chain.name = 'A'
st.write_pdb('$TARGET_PDB')
print('wrote $TARGET_PDB', len(st[0]['A']), 'residues')
"

ls -la "$OUT_ROOT/_shared"
