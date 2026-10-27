#!/usr/bin/env bash
# =============================================================================
# run_msm_slurm.sh — longitudinal MSM registration of all prepared surfaces.
#
# Each available session is registered to the NEXT available session of the
# same animal and hemisphere (ses-01 -> ses-02, ses-02 -> ses-03, ...).
# Surfaces whose QC did not pass are skipped, so the pair jumps over them
# (e.g. Aziza L: ses-05 -> ses-07 while ses-06 is excluded).
#
#   bash run_msm_slurm.sh check    # environment check (FSL/msm, workbench, python)
#   bash run_msm_slurm.sh list     # build the pair list and show it
#   bash run_msm_slurm.sh submit   # one SLURM array task per pair
#   bash run_msm_slurm.sh summary  # after the jobs: table of registration QC
#
# TEXTURE=sulc (default), TEXTURE=dpfstar, or TEXTURE=both (one job per texture).
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG ----------------------------------------
IN_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_inputs
OUT_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_registered
CODE_DIR=/envau/work/meca/users/dienye.h/BaboFet/msm_registration
CONF=${CONF:-$CODE_DIR/msm_baboon_pairwise.conf}
TEXTURE=${TEXTURE:-sulc}
PAIRLIST=$OUT_ROOT/pairlist_${TEXTURE}.txt
LOGDIR=$OUT_ROOT/logs
export CONDA_ENV=${CONDA_ENV:-babofet}
# -----------------------------------------------------------------------------

mode=${1:-}
case "$TEXTURE" in sulc|dpfstar) TEXTURES=("$TEXTURE") ;; both) TEXTURES=(sulc dpfstar) ;;
    *) echo "TEXTURE must be sulc, dpfstar or both" >&2; exit 1 ;; esac

load_env() {
    export NEED_FREESURFER=0 NEED_MSM=1
    source "$CODE_DIR/env_setup.sh"
}

if [[ "$mode" == check ]]; then
    load_env && echo "Environment OK"
    [[ -f "$CONF" ]] && echo "config: $CONF" || { echo "MISSING config: $CONF" >&2; exit 1; }
    exit 0
fi

if [[ "$mode" == list ]]; then
    [[ -f "$CONF" ]] || { echo "MISSING config: $CONF" >&2; exit 1; }
    mkdir -p "$OUT_ROOT" "$LOGDIR"
    : > "$PAIRLIST"
    shopt -s nullglob
    for subdir in "$IN_ROOT"/sub-*; do
        sub=$(basename "$subdir")
        for hemi in L R; do
            usable=()
            for sesdir in "$subdir"/ses-*; do          # glob is sorted: ses-01, ses-02, ...
                ses=$(basename "$sesdir")
                p=$sesdir/${sub}_${ses}_hemi-${hemi}
                if [[ ! -f "$p.sphere.surf.gii" ]]; then continue; fi
                if ! grep -q "QC PASSED" "$p.qc.txt" 2>/dev/null; then
                    echo "skip (QC not passed): $(basename "$p")" >&2; continue
                fi
                for t in "${TEXTURES[@]}"; do
                    [[ -f "$p.$t.func.gii" ]] || { echo "skip (no $t): $(basename "$p")" >&2; continue 2; }
                done
                usable+=("$p")
            done
            for ((k = 0; k + 1 < ${#usable[@]}; k++)); do
                for t in "${TEXTURES[@]}"; do
                    echo "${usable[k]} ${usable[k+1]} $t" >> "$PAIRLIST"
                done
            done
        done
    done
    echo "$(wc -l < "$PAIRLIST") registrations listed in $PAIRLIST"
    awk '{n=split($1,a,"/"); m=split($2,b,"/"); print "  " a[n] "  ->  " b[m] "  [" $3 "]"}' "$PAIRLIST" | head -8
    echo "  ..."
    exit 0
fi

if [[ "$mode" == submit ]]; then
    [[ -s "$PAIRLIST" ]] || { echo "Run '$0 list' first (TEXTURE=$TEXTURE)" >&2; exit 1; }
    n=$(wc -l < "$PAIRLIST")
    sbatch --array=1-"$n" \
           --job-name=msm_reg_${TEXTURE} \
           --cpus-per-task=1 --mem=8G --time=08:00:00 \
           --output="$LOGDIR/%x_%A_%a.out" \
           --export=ALL,TEXTURE="$TEXTURE",CONF="$CONF" \
           "$0" task
    exit 0
fi

if [[ "$mode" == task ]]; then
    load_env
    read -r IN REF TEX <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIRLIST")"
    in_name=$(basename "$IN"); ref_name=$(basename "$REF")
    sub=${in_name%%_ses-*}; hemi=${in_name##*_hemi-}
    in_ses=$(echo "$in_name" | grep -o 'ses-[0-9]*'); ref_ses=$(echo "$ref_name" | grep -o 'ses-[0-9]*')
    ODIR=$OUT_ROOT/$TEX/$sub/hemi-$hemi
    OUT=$ODIR/${sub}_hemi-${hemi}_${in_ses}_to_${ref_ses}
    mkdir -p "$ODIR"
    echo "[msm] $in_name -> $ref_name  texture=$TEX  conf=$CONF"

    msm --inmesh="$IN.sphere.surf.gii"   --refmesh="$REF.sphere.surf.gii" \
        --indata="$IN.$TEX.func.gii"     --refdata="$REF.$TEX.func.gii" \
        --inanat="$IN.anat.surf.gii"     --refanat="$REF.anat.surf.gii" \
        --conf="$CONF" --out="$OUT." --verbose

    # ---- registration QC ----------------------------------------------------
    # areal distortion caused by the registration (registered vs original sphere)
    wb_command -surface-distortion "$IN.sphere.surf.gii" "$OUT.sphere.reg.surf.gii" \
        "$OUT.reg_distortion.func.gii"
    # input texture brought onto the reference mesh, before (rigid sphere) and after MSM
    wb_command -metric-resample "$IN.$TEX.func.gii" "$IN.sphere.surf.gii" "$REF.sphere.surf.gii" \
        BARYCENTRIC "$OUT.$TEX.on_ref.before.func.gii"
    wb_command -metric-resample "$IN.$TEX.func.gii" "$OUT.sphere.reg.surf.gii" "$REF.sphere.surf.gii" \
        BARYCENTRIC "$OUT.$TEX.on_ref.after.func.gii"
    "$PYTHON" - "$REF.$TEX.func.gii" "$OUT.$TEX.on_ref.before.func.gii" \
        "$OUT.$TEX.on_ref.after.func.gii" "$OUT.reg_distortion.func.gii" "$OUT.regqc.txt" <<'EOF'
import sys
import nibabel as nib
import numpy as np
ld = lambda f: np.asarray(nib.load(f).agg_data(), dtype=float).ravel()
ref, before, after, dist = (ld(f) for f in sys.argv[1:5])
r0, r1 = np.corrcoef(ref, before)[0, 1], np.corrcoef(ref, after)[0, 1]
d = dist[np.isfinite(dist)]
line = (f"corr_before={r0:.3f} corr_after={r1:.3f} "
        f"distortion_p95={np.percentile(np.abs(d), 95):.3f} distortion_max={np.abs(d).max():.3f}")
print("[regqc]", line)
open(sys.argv[5], "w").write(line + "\n")
EOF
    echo "[msm] done -> $OUT.*"
    exit 0
fi

if [[ "$mode" == summary ]]; then
    printf "%-45s %-8s %8s %8s %8s %8s\n" registration texture before after dist_p95 dist_max
    shopt -s nullglob
    for t in "${TEXTURES[@]}"; do
        for f in "$OUT_ROOT"/$t/sub-*/hemi-*/*.regqc.txt; do
            read -r b a p m < <(sed 's/[a-z_0-9]*=//g' "$f")
            printf "%-45s %-8s %8s %8s %8s %8s\n" "$(basename "$f" .regqc.txt)" "$t" "$b" "$a" "$p" "$m"
        done
    done
    echo "Missing results = failed or still running jobs: check $LOGDIR"
    exit 0
fi

echo "Usage: [TEXTURE=sulc|dpfstar|both] $0 {check|list|submit|summary}" >&2
exit 1
