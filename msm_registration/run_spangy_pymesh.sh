#!/bin/bash
#SBATCH --job-name=spangy_pymesh
#SBATCH --array=0-29%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_pymesh_%A_%a.out
#SBATCH --error=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_pymesh_%A_%a.err

# SPANGY on the Pymesh meshes (the ones registered with MSM).
# 90 meshes over 30 tasks = 3 meshes per task, at most 10 tasks at once.
#
# Usage:
#   mkdir -p /envau/work/meca/users/dienye.h/python_files/Babofet/logs
#   sbatch run_spangy_pymesh.sh                        # all subjects
#   sbatch run_spangy_pymesh.sh sub-Aziza sub-Borgne   # subset
#   sbatch --export=ALL,SPANGY_ARGS=--onref run_spangy_pymesh.sh   # surfaces on reference mesh
#
# Then, once the array finishes, build the per-subject + pooled CSVs:
#   sbatch --dependency=afterany:<JOBID> --wrap \
#     "source ~/.bashrc; conda activate babofet; python /envau/work/meca/users/dienye.h/BaboFet/msm_registration/spangy_babofet_pymesh.py --merge"

set -eo pipefail

REPO=/envau/work/meca/users/dienye.h/BaboFet
SCRIPT="$REPO/msm_registration/spangy_babofet_pymesh.py"

# Environment - this order matters on the cluster
module purge
module load all
source ~/.bashrc
conda activate babofet

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg

cd "$REPO"

echo "Host: $(hostname)"
echo "Repo: $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'n/a'))"
echo "Python: $(command -v python)"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "Started: $(date)"

# Extra options for the python script, e.g. SPANGY_ARGS=--onref:
#   sbatch --export=ALL,SPANGY_ARGS=--onref run_spangy_pymesh.sh
SPANGY_ARGS=${SPANGY_ARGS:-}
echo "Options: ${SPANGY_ARGS:-none}"

if [ "$#" -gt 0 ]; then
    python -u "$SCRIPT" $SPANGY_ARGS --subjects "$@"
else
    python -u "$SCRIPT" $SPANGY_ARGS
fi

echo "Finished: $(date)"