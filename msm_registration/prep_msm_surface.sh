#!/usr/bin/env bash
# =============================================================================
# prep_msm_surface.sh
# Prepare ONE hemisphere surface (.surf.gii) for MSM registration.
#
# Usage:
#   prep_msm_surface.sh <in_surf.surf.gii> <L|R> <out_dir> [prefix]
#
# Produces in <out_dir>:
#   <prefix>.anat.surf.gii        anatomical surface (copy of input, structure set)  -> --inanat / --refanat
#   <prefix>.inflated.surf.gii    inflated surface (QC / visualisation)
#   <prefix>.sphere.surf.gii      sphere, centred at origin, radius 100            -> --inmesh / --refmesh
#   <prefix>.sulc.func.gii        sulcal depth (FreeSurfer convention)              -> --indata / --refdata
#   <prefix>.curv.func.gii        mean curvature on the anatomical surface          -> optional data / later levels
#   <prefix>.dpfstar.func.gii     DPF-star sulcal depth (slam)                      -> alternative --indata / --refdata
#   <prefix>.dpfstar.func.json    DPF-star parameters and scale factor lc
#   <prefix>.sphere_distortion.func.gii   areal distortion anat -> sphere (QC)
#   <prefix>.qc.txt               QC summary (flipped triangles, radius, ranges)
#
# Requirements: FreeSurfer (mris_*; FS_LICENSE set), Connectome Workbench
# (wb_command), python3 with nibabel + numpy, and slam for DPF-star.
# The input must be a closed, genus-0 triangle mesh (one hemisphere).
# =============================================================================
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <in_surf.surf.gii> <L|R> <out_dir> [prefix]" >&2
    exit 1
fi

IN=$(readlink -f "$1")
HEMI=$2
OUT=$3
PREFIX=${4:-$(basename "$(basename "$IN" .gii)" .surf)}

# ---- options (override via environment) ------------------------------------
SMOOTH_ITERS=${SMOOTH_ITERS:-3}       # mris_smooth iterations (recon-all uses 3)
SPHERE_QUICK=${SPHERE_QUICK:-0}       # 1 -> mris_sphere -q (faster, more distortion)
SULC_HCP_SIGN=${SULC_HCP_SIGN:-0}     # 1 -> multiply sulc by -1 (HCP convention)
KEEP_WORK=${KEEP_WORK:-1}             # 0 -> delete FreeSurfer working dir at the end
SEED=${SEED:-1234}
COMPUTE_DPFSTAR=${COMPUTE_DPFSTAR:-1} # 1 -> also compute DPF-star texture (needs slam)
DPF_ALPHA=${DPF_ALPHA:-500}           # DPF-star alpha (scale of interest)
DPF_ADAPTATION=${DPF_ADAPTATION:-volume_hull}  # volume_hull | volume | surface
PYTHON=${PYTHON:-python3}             # python with nibabel, numpy (and slam for DPF-star)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$HEMI" in
    L) FSH=lh; STRUCT=CORTEX_LEFT ;;
    R) FSH=rh; STRUCT=CORTEX_RIGHT ;;
    *) echo "Hemisphere must be L or R (got '$HEMI')" >&2; exit 1 ;;
esac

for cmd in mris_convert mris_smooth mris_inflate mris_sphere wb_command "$PYTHON"; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done
[[ -f "$IN" ]] || { echo "Input not found: $IN" >&2; exit 1; }

mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
WORK="$OUT/fswork_${PREFIX}"
mkdir -p "$WORK"

log() { echo "[$(date +%H:%M:%S)] [$PREFIX] $*"; }

# ---- 1. GIFTI -> FreeSurfer surface -----------------------------------------
# FreeSurfer derives the hemisphere (and the sulc file name) from the "lh."/"rh."
# prefix, so the working files are named lh.* / rh.*.
# Resume: if a previous run already got through mris_sphere, reuse its outputs
# (set REDO_FS=1 to force recomputation).
if [[ "${REDO_FS:-0}" != 1 && -s "$WORK/$FSH.sphere" && -s "$WORK/$FSH.sulc" \
      && -s "$WORK/$FSH.inflated" && -s "$WORK/$FSH.white" ]]; then
    log "Reusing existing FreeSurfer outputs in $WORK (steps 1-4 skipped)"
else
    log "Converting to FreeSurfer format"
    mris_convert "$IN" "$WORK/$FSH.white"

    # ---- 2. Light smoothing (as recon-all does before inflation) -------------
    log "Smoothing ($SMOOTH_ITERS iterations)"
    mris_smooth -n "$SMOOTH_ITERS" -nw -seed "$SEED" "$WORK/$FSH.white" "$WORK/$FSH.smoothwm"

    # ---- 3. Inflation (also writes <hemi>.sulc next to the output) ----------
    log "Inflating"
    ( cd "$WORK" && mris_inflate "$FSH.smoothwm" "$FSH.inflated" )
    [[ -f "$WORK/$FSH.sulc" ]] || { echo "mris_inflate did not produce $FSH.sulc" >&2; exit 1; }

    # ---- 4. Spherical mapping ------------------------------------------------
    log "Mapping to sphere (this is the slow step)"
    SPH_OPTS=(-seed "$SEED")
    [[ "$SPHERE_QUICK" == 1 ]] && SPH_OPTS+=(-q)
    ( cd "$WORK" && mris_sphere "${SPH_OPTS[@]}" "$FSH.inflated" "$FSH.sphere" )
fi

# ---- 5. Back to GIFTI --------------------------------------------------------
log "Converting outputs to GIFTI"
mris_convert "$WORK/$FSH.inflated" "$OUT/$PREFIX.inflated.surf.gii"
mris_convert "$WORK/$FSH.sphere"   "$WORK/$PREFIX.sphere_raw.surf.gii"
# sulc: converted with nibabel, not `mris_convert -c` (which prepends "lh."/"rh."
# to the output file name). Values are written in vertex order, as FreeSurfer stores them.
"$PYTHON" - "$WORK/$FSH.sulc" "$OUT/$PREFIX.sulc.func.gii" <<'EOF'
import sys
import nibabel as nib
import numpy as np
values = nib.freesurfer.read_morph_data(sys.argv[1]).astype(np.float32)
nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
    values, intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32")]), sys.argv[2])
print(f"sulc: {values.size} values written to {sys.argv[2]}")
EOF
# remove a misnamed file left by an earlier version of this script, if any
rm -f "$OUT/$FSH.$PREFIX.sulc.func.gii"

# Anatomical surface for MSM strain regularisation = the original input
# (same vertices/order as the sphere; mris_convert preserves vertex order).
cp "$IN" "$OUT/$PREFIX.anat.surf.gii"

# ---- 6. MSM-ready sphere: centred at origin, radius exactly 100 --------------
log "Recentring sphere and setting radius to 100"
wb_command -surface-modify-sphere "$WORK/$PREFIX.sphere_raw.surf.gii" 100 \
    "$OUT/$PREFIX.sphere.surf.gii" -recenter

# ---- 7. Metadata (lets Workbench / MSM know what each file is) ---------------
wb_command -set-structure "$OUT/$PREFIX.anat.surf.gii"     "$STRUCT" -surface-type ANATOMICAL
wb_command -set-structure "$OUT/$PREFIX.inflated.surf.gii" "$STRUCT" -surface-type INFLATED
wb_command -set-structure "$OUT/$PREFIX.sphere.surf.gii"   "$STRUCT" -surface-type SPHERICAL
wb_command -set-structure "$OUT/$PREFIX.sulc.func.gii"     "$STRUCT"

if [[ "$SULC_HCP_SIGN" == 1 ]]; then
    log "Flipping sulc sign (HCP convention)"
    wb_command -metric-math "var * -1" "$OUT/$PREFIX.sulc.func.gii" \
        -var var "$OUT/$PREFIX.sulc.func.gii"
fi

# ---- 8. Curvature + distortion ----------------------------------------------
log "Computing mean curvature and sphere distortion"
wb_command -surface-curvature "$OUT/$PREFIX.anat.surf.gii" -mean "$OUT/$PREFIX.curv.func.gii"
wb_command -set-structure "$OUT/$PREFIX.curv.func.gii" "$STRUCT"
wb_command -surface-distortion "$OUT/$PREFIX.anat.surf.gii" "$OUT/$PREFIX.sphere.surf.gii" \
    "$OUT/$PREFIX.sphere_distortion.func.gii"

# ---- 9. DPF-star (computed on the same anatomical mesh as the sphere) --------
QC_EXTRA=()
if [[ "$COMPUTE_DPFSTAR" == 1 ]]; then
    log "Computing DPF-star (alpha=$DPF_ALPHA, adaptation=$DPF_ADAPTATION)"
    "$PYTHON" "$SCRIPT_DIR/compute_dpf_star.py" \
        "$OUT/$PREFIX.anat.surf.gii" "$OUT/$PREFIX.dpfstar.func.gii" \
        --alpha "$DPF_ALPHA" --adaptation "$DPF_ADAPTATION"
    wb_command -set-structure "$OUT/$PREFIX.dpfstar.func.gii" "$STRUCT"
    QC_EXTRA=(--dpfstar "$OUT/$PREFIX.dpfstar.func.gii")
fi

# ---- 10. QC ------------------------------------------------------------------
log "Running QC"
"$PYTHON" "$SCRIPT_DIR/qc_msm_inputs.py" \
    --anat "$OUT/$PREFIX.anat.surf.gii" \
    --sphere "$OUT/$PREFIX.sphere.surf.gii" \
    --sulc "$OUT/$PREFIX.sulc.func.gii" \
    --curv "$OUT/$PREFIX.curv.func.gii" \
    --distortion "$OUT/$PREFIX.sphere_distortion.func.gii" \
    ${QC_EXTRA[@]+"${QC_EXTRA[@]}"} \
    | tee "$OUT/$PREFIX.qc.txt"

[[ "$KEEP_WORK" == 0 ]] && rm -rf "$WORK"
log "Done -> $OUT"