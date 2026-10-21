#!/usr/bin/env bash
# Re-run QC on every prepared surface (no FreeSurfer, takes seconds per surface).
#   bash rerun_qc.sh                 # all subjects
#   bash rerun_qc.sh sub-Aziza       # one subject
# Prints a one-line summary per surface and rewrites each <prefix>.qc.txt.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_inputs
export NEED_FREESURFER=0
source "$SCRIPT_DIR/env_setup.sh" > /dev/null || exit 1

n_ok=0; n_fail=0
for sphere in "$OUT_ROOT"/${1:-sub-*}/ses-*/*.sphere.surf.gii; do
    p=${sphere%.sphere.surf.gii}
    args=(--anat "$p.anat.surf.gii" --sphere "$sphere" --sulc "$p.sulc.func.gii"
          --curv "$p.curv.func.gii" --distortion "$p.sphere_distortion.func.gii")
    [[ -f "$p.dpfstar.func.gii" ]] && args+=(--dpfstar "$p.dpfstar.func.gii")
    if "$PYTHON" "$SCRIPT_DIR/qc_msm_inputs.py" "${args[@]}" > "$p.qc.txt" 2>&1; then
        echo "PASS  $(basename "$p")"; n_ok=$((n_ok + 1))
    else
        echo "FAIL  $(basename "$p")"; grep '^  - ' "$p.qc.txt"; n_fail=$((n_fail + 1))
    fi
done
echo "---- $n_ok passed, $n_fail failed"
