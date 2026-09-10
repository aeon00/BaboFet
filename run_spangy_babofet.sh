#!/bin/bash
#SBATCH --job-name=spangy_babofet
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_%A_%a.out
#SBATCH --error=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_%A_%a.err

# Code lives in the git repo; data and logs stay in the working tree.
#
# Usage:
#   mkdir -p /envau/work/meca/users/dienye.h/python_files/Babofet/logs
#   sbatch run_spangy_babofet.sh                       # all subjects
#   sbatch run_spangy_babofet.sh sub-Aziza sub-Borgne  # subset
#
# Then, once the array finishes, build the per-subject + pooled CSVs:
#   sbatch --dependency=afterany:<JOBID> merge_spangy_babofet.sh
# (or use submit_spangy_babofet.sh, which chains both automatically)

set -euo pipefail

REPO=/envau/work/meca/users/dienye.h/BaboFet
SCRIPT="$REPO/spangy_babofet_all_subjects.py"

# Environment - this order matters on the cluster
module purge
module load all
source ~/.bashrc
conda activate slam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg

cd "$REPO"

echo "Host: $(hostname)"
echo "Repo: $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'n/a'))"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "Started: $(date)"

if [ "$#" -gt 0 ]; then
    python -u "$SCRIPT" --subjects "$@"
else
    python -u "$SCRIPT"
fi

echo "Finished: $(date)"
