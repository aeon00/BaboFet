#!/bin/bash
# =============================================================================
# env_setup.sh — environment for the MSM prep / registration jobs on the INT cluster.
# Sourced by run_prep_msm_slurm.sh (task mode) and run_msm_example.sh.
# Order that works on INT: purge -> load all -> tools -> ~/.bashrc -> conda env.
#
# Settings (export before submitting to override):
#   FS_MODULE   exact FreeSurfer module name from `module avail freesurfer`
#               (or `module avail 2>&1 | grep -i free`)
#   FS_LICENSE  path to your FreeSurfer license.txt (auto-detected if unset)
#   CONDA_ENV   conda env with slam + nibabel (default: babofet)
#   NEED_FREESURFER  1 for the prep stage (default), 0 for MSM-only jobs
# =============================================================================

# Module and bashrc scripts often reference unset variables -> relax strict mode here.
_had_u=0; [[ $- == *u* ]] && _had_u=1; set +u
_had_e=0; [[ $- == *e* ]] && _had_e=1; set +e

module purge
module load all
module load FSL/6.0.6.2 workbench/1.5.0

if [ "${NEED_FREESURFER:-1}" = 1 ]; then
    if [ -n "${FS_MODULE:-}" ]; then
        module load "$FS_MODULE"
    else
        # try common names; stop once FreeSurfer is actually usable
        # (some module versions return success even when the load fails)
        for m in freesurfer FreeSurfer freesurfer/7.4.1 FreeSurfer/7.4.1 freesurfer/7.3.2 FreeSurfer/7.3.2; do
            [ -n "${FREESURFER_HOME:-}" ] || command -v mris_convert >/dev/null 2>&1 && break
            module load "$m" 2>/dev/null
        done
    fi
    if [ -n "${FREESURFER_HOME:-}" ] && [ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]; then
        source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null
    fi
fi

source ~/.bashrc
conda activate "${CONDA_ENV:-babofet}"

# FSL setup (defines FSLDIR and puts msm on PATH if the module does not do it)
if [ -n "${FSLDIR:-}" ] && [ -f "$FSLDIR/etc/fslconf/fsl.sh" ]; then
    source "$FSLDIR/etc/fslconf/fsl.sh"
fi

# FreeSurfer license: use FS_LICENSE if set, otherwise look in the usual places.
if [ "${NEED_FREESURFER:-1}" = 1 ] && [ -z "${FS_LICENSE:-}" ]; then
    for f in "${FREESURFER_HOME:-/nonexistent}/license.txt" "${FREESURFER_HOME:-/nonexistent}/.license" \
             "$HOME/license.txt" "$HOME/.freesurfer/license.txt"; do
        if [ -f "$f" ]; then export FS_LICENSE="$f"; break; fi
    done
fi

[ $_had_u = 1 ] && set -u
[ $_had_e = 1 ] && set -e
unset _had_u _had_e

# ---- report what was found, fail early with a clear message -----------------
_missing=0
_tools="wb_command python3"
[ "${NEED_FREESURFER:-1}" = 1 ] && _tools="$_tools mris_convert mris_smooth mris_inflate mris_sphere"
[ "${NEED_MSM:-0}" = 1 ] && _tools="$_tools msm"
for t in $_tools; do
    if p=$(command -v "$t"); then echo "[env] $t -> $p"
    else echo "[env] MISSING: $t" >&2; _missing=1; fi
done
if [ "${NEED_FREESURFER:-1}" = 1 ]; then
    if [ -n "${FS_LICENSE:-}" ] && [ -f "$FS_LICENSE" ]; then echo "[env] FS_LICENSE -> $FS_LICENSE"
    else echo "[env] MISSING: FreeSurfer license (export FS_LICENSE=/path/to/license.txt)" >&2; _missing=1; fi
fi
python3 -c "import nibabel, slam" 2>/dev/null && echo "[env] python: nibabel + slam OK" \
    || { echo "[env] MISSING: nibabel/slam in conda env ${CONDA_ENV:-babofet}" >&2; _missing=1; }
if [ $_missing = 1 ]; then
    echo "[env] Environment incomplete — see messages above." >&2
    return 1 2>/dev/null || exit 1
fi
unset _missing _tools
