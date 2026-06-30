#!/bin/bash
#SBATCH --job-name=spangy_borgne
#SBATCH --array=0-7                  # adjust: one task per ~N files; 8 tasks = 8 chunks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # eigenpairs on large meshes are memory-hungry
#SBATCH --time=04:00:00
#SBATCH --output=/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy/logs/slurm_%A_%a.out
#SBATCH --error=/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy/logs/slurm_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────────
module load all
source ~/.bashrc 
conda activate babofet

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p /envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy/logs

python /envau/work/meca/users/dienye.h/BaboFet/spangy_process_subs.py