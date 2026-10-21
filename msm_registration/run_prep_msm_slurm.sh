#!/usr/bin/env bash
# =============================================================================
# run_prep_msm_slurm.sh — run prep_msm_surface.sh over all baboon surfaces.
#
# Step 1 (login node): build the job list
#     bash run_prep_msm_slurm.sh list
# Step 2: submit one array task per surface
#     bash run_prep_msm_slurm.sh submit
#
# Adjust the CONFIG block to your folder layout / file naming.
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG ----------------------------------------
MESH_ROOT=/envau/work/meca/users/dienye.h/Pymesh/mesh
# e.g. sub-Aziza/ses-01/anat/sub-Aziza_ses-01_manualseg_extract_surf_hemi_left_wm.gii
MESH_GLOB="$MESH_ROOT/sub-*/ses-*/anat/*_hemi_*_wm.gii"   # input surfaces
LEFT_REGEX='_hemi_left_'                        # how the left hemisphere is named
RIGHT_REGEX='_hemi_right_'                      # how the right hemisphere is named
EXCLUDE_REGEX='(inflated|sphere|pial)'          # skip derived surfaces
OUT_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_inputs
CODE_DIR=/envau/work/meca/users/dienye.h/BaboFet/msm_prep
JOBLIST=$OUT_ROOT/joblist.txt
LOGDIR=$OUT_ROOT/logs
MODULES="freesurfer workbench"                  # names on your cluster
# Python env with slam + nibabel. conda in a batch job needs conda.sh sourced first.
# If babofet is a venv instead: PY_ACTIVATE="source /path/to/babofet/bin/activate"
PY_ACTIVATE='source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate babofet'
export COMPUTE_DPFSTAR=1 DPF_ALPHA=500 DPF_ADAPTATION=volume_hull
# -----------------------------------------------------------------------------

mode=${1:-}

if [[ "$mode" == list ]]; then
    mkdir -p "$OUT_ROOT" "$LOGDIR"
    : > "$JOBLIST"
    shopt -s nullglob
    for f in $MESH_GLOB; do
        name=$(basename "$f")
        [[ "$name" =~ $EXCLUDE_REGEX ]] && continue
        if   [[ "$name" =~ $LEFT_REGEX ]];  then hemi=L
        elif [[ "$name" =~ $RIGHT_REGEX ]]; then hemi=R
        else echo "skip (no hemisphere in name): $f" >&2; continue
        fi
        sub=$(echo "$f" | grep -o 'sub-[^/]*' | head -1)
        ses=$(echo "$f" | grep -o 'ses-[^/]*' | head -1)
        echo "$f $hemi $OUT_ROOT/$sub/$ses ${sub}_${ses}_hemi-${hemi}" >> "$JOBLIST"
    done
    echo "$(wc -l < "$JOBLIST") surfaces listed in $JOBLIST"
    cut -d' ' -f2- "$JOBLIST" | awk '{print "  " $3 "  (" $1 ")"}' | head -5
    exit 0
fi

if [[ "$mode" == submit ]]; then
    n=$(wc -l < "$JOBLIST")
    sbatch --array=1-"$n" \
           --job-name=msm_prep \
           --cpus-per-task=1 --mem=8G --time=04:00:00 \
           --output="$LOGDIR/%x_%A_%a.out" \
           "$0" task
    exit 0
fi

if [[ "$mode" == task ]]; then
    # shellcheck disable=SC2086
    module load $MODULES 2>/dev/null || true
    if [[ -n "$PY_ACTIVATE" ]]; then set +u; eval "$PY_ACTIVATE"; set -u; fi
    export PYTHON=$(command -v python3)
    : "${FS_LICENSE:?Set FS_LICENSE to your FreeSurfer license file}"
    line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
    read -r IN HEMI OUT PREFIX <<< "$line"
    bash "$CODE_DIR/prep_msm_surface.sh" "$IN" "$HEMI" "$OUT" "$PREFIX"
    exit 0
fi

echo "Usage: $0 {list|submit}" >&2
exit 1
