#!/usr/bin/env bash
# Example: register ses-01 -> ses-02 (left hemisphere) with the prepared files.
#
#   TEXTURE=sulc    bash run_msm_example.sh   (default)
#   TEXTURE=dpfstar bash run_msm_example.sh
#   CONF=msm_strain_sulc.conf TEXTURE=dpfstar bash run_msm_example.sh
#
# Input and reference always use the same texture.
set -euo pipefail

TEXTURE=${TEXTURE:-sulc}                       # sulc | dpfstar
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONF=${CONF:-$SCRIPT_DIR/msm_baboon_pairwise.conf}

D=/envau/work/meca/users/dienye.h/python_files/Babofet/msm_inputs/sub-Borgne
IN=$D/ses-01/sub-Borgne_ses-01_hemi-L
REF=$D/ses-02/sub-Borgne_ses-02_hemi-L
OUT=$D/msm_out/${TEXTURE}/ses-01_to_ses-02_hemi-L
mkdir -p "$(dirname "$OUT")"

for f in "$IN.$TEXTURE.func.gii" "$REF.$TEXTURE.func.gii"; do
    [[ -f "$f" ]] || { echo "Missing texture: $f" >&2; exit 1; }
done

# --inanat/--refanat are required by the strain config (--regoption=3) and
# harmless otherwise.
msm --inmesh="$IN.sphere.surf.gii"  --refmesh="$REF.sphere.surf.gii" \
    --indata="$IN.$TEXTURE.func.gii" --refdata="$REF.$TEXTURE.func.gii" \
    --inanat="$IN.anat.surf.gii"    --refanat="$REF.anat.surf.gii" \
    --conf="$CONF" \
    --out="$OUT." \
    --verbose

# Outputs: $OUT.sphere.reg.surf.gii (registered sphere) and $OUT.transformed_and_reprojected.func.gii
# Resample any ses-01 metric onto the ses-02 mesh with:
# wb_command -metric-resample <metric> "$OUT.sphere.reg.surf.gii" "$REF.sphere.surf.gii" \
#     ADAP_BARY_AREA <out> -area-surfs "$IN.anat.surf.gii" "$REF.anat.surf.gii"
