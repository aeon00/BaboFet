#!/usr/bin/env python3
"""
resample_to_ref.py — put every session's anatomical surface onto its animal's
reference mesh (same vertices and triangles as the reference session), using the
chained MSM spheres. Run once on the cluster after chain_to_reference.sh.

Each output keeps the session's OWN shape (size, folding at that age); only the
triangulation is the reference's. SPANGY can then be run directly on these
surfaces, and all its outputs share one vertex numbering per animal/hemisphere.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`):
    $PYTHON resample_to_ref.py                     # all animals (sulc registrations)
    $PYTHON resample_to_ref.py --sub sub-Aziza
    $PYTHON resample_to_ref.py --texture dpfstar   # use the DPF-star chain instead

Output:
    <BASE>/surfaces_on_ref/<texture>/<sub>/hemi-<H>/
        <sub>_<ses>_hemi-<H>_on_<ref>.anat.surf.gii
        <sub>_hemi-<H>_resample_qc.txt
The QC table compares each resampled surface with the original one:
  area_ratio   total area after / before (below 1 = the surface was flattened a bit)
  bad_tri_%    triangles with a minimum angle below 5 degrees (badly shaped)
Both matter for SPANGY: lost area means lost fine folding (high bands); badly
shaped triangles disturb curvature and the Laplace-Beltrami eigenvectors.
"""
import argparse
import glob
import os
import re
import subprocess

import nibabel as nib
import numpy as np

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"


def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True)


def load_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float64),
            np.asarray(g.agg_data("triangle"), np.int64))


def area_and_bad(xyz, tri, min_deg=5.0):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    def ang(p, q, r):          # angle at p
        u, v = q - p, r - p
        cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    min_ang = np.minimum(np.minimum(ang(a, b, c), ang(b, c, a)), ang(c, a, b))
    return area, 100.0 * np.mean(min_ang < min_deg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", default="sulc", help="which chained registrations to use")
    p.add_argument("--sub", default="sub-*")
    args = p.parse_args()

    chain_root = f"{BASE}/msm_chained/{args.texture}"
    in_root = f"{BASE}/msm_inputs"
    out_root = f"{BASE}/surfaces_on_ref/{args.texture}"   # sulc and dpfstar kept apart

    for hdir in sorted(glob.glob(f"{chain_root}/{args.sub}/hemi-*")):
        sub = os.path.basename(os.path.dirname(hdir))
        hemi = hdir[-1]
        struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
        spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
        if not spheres:
            continue
        ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
        ref_sphere = f"{in_root}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.sphere.surf.gii"
        odir = f"{out_root}/{sub}/hemi-{hemi}"
        os.makedirs(odir, exist_ok=True)

        lines = [f"{sub} hemi-{hemi}, reference {ref} (registrations: {args.texture})",
                 f"{'session':<8} {'n_orig':>8} {'n_ref':>8} {'area_ratio':>10} "
                 f"{'bad_tri_%':>10} {'bad_tri_%_orig':>14}"]
        for sph in spheres:
            ses = re.search(r"_(ses-\d+)_in_", sph).group(1)
            anat = f"{in_root}/{sub}/{ses}/{sub}_{ses}_hemi-{hemi}.anat.surf.gii"
            out = f"{odir}/{sub}_{ses}_hemi-{hemi}_on_{ref}.anat.surf.gii"
            wb("-surface-resample", anat, sph, ref_sphere, "BARYCENTRIC", out)
            wb("-set-structure", out, struct, "-surface-type", "ANATOMICAL")

            xo, to = load_surf(anat)
            xr, tr = load_surf(out)
            a_o, bad_o = area_and_bad(xo, to)
            a_r, bad_r = area_and_bad(xr, tr)
            lines.append(f"{ses:<8} {len(xo):>8} {len(xr):>8} {a_r / a_o:>10.3f} "
                         f"{bad_r:>10.2f} {bad_o:>14.2f}")

        table = "\n".join(lines)
        print(table + "\n")
        with open(f"{odir}/{sub}_hemi-{hemi}_resample_qc.txt", "w") as fh:
            fh.write(table + "\n")


if __name__ == "__main__":
    main()