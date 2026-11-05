#!/bin/bash
#SBATCH --job-name=campaign-watch
#SBATCH --partition=kempner_priority
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=campaign-watch-%j.log

# Read-only CPU monitor. Slurm spools this file elsewhere: pass its source directory.
# sbatch collect.sbatch /absolute/path/to/scripts/campaign_watch \
#   --config /absolute/state/campaigns.json --output /absolute/state/snapshot.json
set -euo pipefail
watch_dir="${1:?Pass the absolute scripts/campaign_watch directory first}"
shift
if [[ "${watch_dir}" != /* || ! -f "${watch_dir}/collector.py" ]]; then
    printf '%s\n' 'Expected an absolute directory containing collector.py' >&2
    exit 2
fi
: "${SLURM_JOB_ID:?This wrapper must run as a Slurm CPU job}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -c 0
# Slurm enforces the memory/wall-time limits; there is no automatic resubmission.
ulimit -t 259200
python_bin="${CAMPAIGN_WATCH_PYTHON:-python3}"
exec "${python_bin}" -B "${watch_dir}/collector.py" "$@"
