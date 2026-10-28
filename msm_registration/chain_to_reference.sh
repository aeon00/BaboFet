#!/usr/bin/env bash
# =============================================================================
# chain_to_reference.sh — combine the session-to-session MSM registrations of
# each animal and hemisphere so that EVERY session maps onto one reference
# session of that animal (default: the latest = oldest gestational age).
#
#   bash chain_to_reference.sh                      # all animals, sulc registrations
#   bash chain_to_reference.sh sub-Aziza            # one animal
#   TEXTURE=dpfstar bash chain_to_reference.sh      # use the DPF-star registrations
#   REF=first bash chain_to_reference.sh            # reference = earliest session
#
# How the links are combined (wb_command -surface-sphere-project-unproject):
#   forward  a -> b : sphere currently in a-space, project to sphere_a,
#                     unproject from reg(a->b)            -> now in b-space
#   backward b -> a : sphere currently in b-space, project to reg(a->b),
#                     unproject from sphere_a             -> now in a-space
# Each session starts from its own sphere and follows the links to the
# reference, so every output keeps that session's own vertices.
#
# Output, per animal/hemisphere, in  msm_chained/<texture>/<sub>/hemi-<H>/ :
#   <sub>_hemi-<H>_ses-XX_in_ses-RR.sphere.surf.gii   session XX on the reference sphere
#   <sub>_hemi-<H>_ses-XX_in_ses-RR.distortion.func.gii   areal distortion of the full chain
#   <sub>_hemi-<H>_ses-XX_in_ses-RR.<texture>.on_ref.func.gii  QC: texture moved onto ref mesh
#   <sub>_hemi-<H>_chain_qc.txt                       correlation with the reference + distortion
#
# To move ANY texture of session XX (e.g. a SPANGY band map) onto the reference mesh:
#   wb_command -metric-resample tex_sesXX.func.gii <..._ses-XX_in_ses-RR.sphere.surf.gii> \
#       <ref sphere from msm_inputs> ADAP_BARY_AREA out.func.gii \
#       -area-surfs <sesXX anat.surf.gii> <ref anat.surf.gii>
# =============================================================================
set -euo pipefail

# ----------------------------- CONFIG ----------------------------------------
IN_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_inputs
REG_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_registered
OUT_ROOT=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_chained
CODE_DIR=/envau/work/meca/users/dienye.h/BaboFet/msm_registration
TEXTURE=${TEXTURE:-sulc}          # which registrations to chain: sulc | dpfstar
REF=${REF:-last}                  # last (oldest gestational age) | first
# -----------------------------------------------------------------------------

export NEED_FREESURFER=0
source "$CODE_DIR/env_setup.sh" > /dev/null

sphere_of() { echo "$IN_ROOT/$1/$2/${1}_${2}_hemi-${3}.sphere.surf.gii"; }   # sub ses hemi
anat_of()   { echo "$IN_ROOT/$1/$2/${1}_${2}_hemi-${3}.anat.surf.gii"; }
tex_of()    { echo "$IN_ROOT/$1/$2/${1}_${2}_hemi-${3}.${TEXTURE}.func.gii"; }

shopt -s nullglob
n_done=0
for regdir in "$REG_ROOT/$TEXTURE"/${1:-sub-*}/hemi-*; do
    sub=$(basename "$(dirname "$regdir")"); hemi=${regdir##*hemi-}
    tag="${sub} hemi-${hemi}"

    # ---- read the links: NEXT[a]=b, PREV[b]=a, REG[a]=file ------------------
    declare -A NEXT=() PREV=() REG=()
    for f in "$regdir"/*_to_ses-*.sphere.reg.surf.gii; do
        a=$(basename "$f" | grep -o 'ses-[0-9]*' | sed -n 1p)
        b=$(basename "$f" | grep -o 'ses-[0-9]*' | sed -n 2p)
        NEXT[$a]=$b; PREV[$b]=$a; REG[$a]=$f
    done
    (( ${#NEXT[@]} )) || { echo "[$tag] no registrations found, skipped"; continue; }

    # ---- order the chain: start = a session that is nobody's target ---------
    start=""
    for a in "${!NEXT[@]}"; do [[ -z "${PREV[$a]:-}" ]] && start=$a; done
    chain=("$start"); s=$start
    while [[ -n "${NEXT[$s]:-}" ]]; do s=${NEXT[$s]}; chain+=("$s"); done
    if (( ${#chain[@]} != ${#NEXT[@]} + 1 )); then
        echo "[$tag] WARNING: links do not form one unbroken chain (${chain[*]}); skipped" >&2
        unset NEXT PREV REG; continue
    fi
    if [[ "$REF" == first ]]; then ref=${chain[0]}; else ref=${chain[-1]}; fi
    echo "[$tag] chain: ${chain[*]}  ->  reference $ref"

    odir=$OUT_ROOT/$TEXTURE/$sub/hemi-$hemi
    mkdir -p "$odir"
    ref_sphere=$(sphere_of "$sub" "$ref" "$hemi")
    tmp=$(mktemp -d)

    for s in "${chain[@]}"; do
        out=$odir/${sub}_hemi-${hemi}_${s}_in_${ref}
        cur=$(sphere_of "$sub" "$s" "$hemi")
        pos=$s; step=0
        # walk from s to the reference, one link at a time
        while [[ "$pos" != "$ref" ]]; do
            step=$((step + 1)); nxt=$tmp/${s}_step${step}.surf.gii
            if [[ "$REF" == first ]]; then           # backward: pos -> PREV[pos]
                a=${PREV[$pos]}
                wb_command -surface-sphere-project-unproject "$cur" \
                    "${REG[$a]}" "$(sphere_of "$sub" "$a" "$hemi")" "$nxt"
                pos=$a
            else                                     # forward: pos -> NEXT[pos]
                wb_command -surface-sphere-project-unproject "$cur" \
                    "$(sphere_of "$sub" "$pos" "$hemi")" "${REG[$pos]}" "$nxt"
                pos=${NEXT[$pos]}
            fi
            cur=$nxt
        done
        cp "$cur" "$out.sphere.surf.gii"
        wb_command -set-structure "$out.sphere.surf.gii" \
            "$([[ $hemi == L ]] && echo CORTEX_LEFT || echo CORTEX_RIGHT)" -surface-type SPHERICAL

        # QC: distortion of the whole chain, and this session's texture on the ref mesh
        wb_command -surface-distortion "$(sphere_of "$sub" "$s" "$hemi")" "$out.sphere.surf.gii" \
            "$out.distortion.func.gii"
        wb_command -metric-resample "$(tex_of "$sub" "$s" "$hemi")" "$out.sphere.surf.gii" \
            "$ref_sphere" ADAP_BARY_AREA "$out.$TEXTURE.on_ref.func.gii" \
            -area-surfs "$(anat_of "$sub" "$s" "$hemi")" "$(anat_of "$sub" "$ref" "$hemi")"
        echo "  $s -> $ref  ($step link(s))"
    done
    rm -rf "$tmp"

    # ---- chain QC table ------------------------------------------------------
    "$PYTHON" - "$odir" "$sub" "$hemi" "$ref" "$(tex_of "$sub" "$ref" "$hemi")" "$TEXTURE" \
        "${chain[@]}" <<'EOF' | tee "$odir/${sub}_hemi-${hemi}_chain_qc.txt"
import sys
import nibabel as nib
import numpy as np
odir, sub, hemi, ref, ref_tex, tex, *chain = sys.argv[1:]
ld = lambda f: np.asarray(nib.load(f).agg_data(), dtype=float).ravel()
r = ld(ref_tex)
print(f"{sub} hemi-{hemi}, reference {ref}, texture {tex}")
print(f"{'session':<8} {'links':>5} {'corr_with_ref':>13} {'dist_p95':>9} {'dist_max':>9}")
k_ref = chain.index(ref)
for k, s in enumerate(chain):
    p = f"{odir}/{sub}_hemi-{hemi}_{s}_in_{ref}"
    t = ld(f"{p}.{tex}.on_ref.func.gii")
    d = np.abs(ld(f"{p}.distortion.func.gii")); d = d[np.isfinite(d)]
    print(f"{s:<8} {abs(k - k_ref):>5} {np.corrcoef(r, t)[0, 1]:>13.3f} "
          f"{np.percentile(d, 95):>9.3f} {d.max():>9.3f}")
EOF
    unset NEXT PREV REG
    n_done=$((n_done + 1))
done
echo "---- $n_done animal/hemisphere chains written to $OUT_ROOT/$TEXTURE"
