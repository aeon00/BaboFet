#!/bin/bash
#SBATCH --job-name=spangy_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_merge_%j.out
#SBATCH --error=/envau/work/meca/users/dienye.h/python_files/Babofet/logs/spangy_merge_%j.err

set -euo pipefail

SCRIPT=/envau/work/meca/users/dienye.h/python_files/Babofet/spangy_babofet_all_subjects.py

module purge
module load all
source ~/.bashrc
conda activate slam

if [ "$#" -gt 0 ]; then
    python -u "$SCRIPT" --merge --subjects "$@"
else
    python -u "$SCRIPT" --merge
fi
