#!/usr/bin/env python3
"""
prepare_route_viewer.py — copy, for each SPANGY route, every inflated surface and
its matching textures into one folder per route, with names the viewer hub pairs
automatically (same sub-*, ses-*, hemi-* in the mesh and texture names).

Routes (see compare_spangy_routes.py):
  ORIG     SPANGY on the original mesh  -> that session's own inflated surface
  RESTEX   ORIG textures resampled onto the reference mesh
           -> the REFERENCE session's inflated surface, copied once per session
  ONREF    SPANGY on the surface resampled onto the reference mesh
           -> the REFERENCE session's inflated surface, copied once per session

RESTEX and ONREF use the same inflated file for every session of an animal and
hemisphere; it is copied under each session's name so the viewer pairs it with
that session's texture.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`, or any
python with nibabel):
    $PYTHON prepare_route_viewer.py                       # sulc registrations
    $PYTHON prepare_route_viewer.py --texture both        # sulc and DPF-star
    $PYTHON prepare_route_viewer.py --sub sub-Aziza --also-anat
    $PYTHON prepare_route_viewer.py --link                # symlinks instead of copies

Output:
    <BASE>/route_viewer/
        ORIG/            meshes/  textures/  manifest.csv
        RESTEX_<tex>/    meshes/  textures/  manifest.csv
        ONREF_<tex>/     meshes/  textures/  manifest.csv
    meshes/   <sub>_<ses>_hemi-<H>.inflated.surf.gii   (+ .anat.surf.gii with --also-anat)
    textures/ <sub>_<ses>_hemi-<H>.dom_band.gii
              <sub>_<ses>_hemi-<H>.filt_mean_curv.gii  (when available)
In the viewer: mesh directory = <route>/meshes, texture directory = <route>/textures,
mesh filter "inflated", texture filter "dom_band" (or "curv").
"""
import argparse
import csv
import glob
import os
import re
import shutil

import nibabel as nib

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"


def n_vertices(path):
    img = nib.load(path)
    for d in img.darrays:
        if d.intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]:
            return d.data.shape[0]
    return img.darrays[0].data.shape[0]


class Route:
    def __init__(self, name, out_root, link):
        self.name, self.link = name, link
        self.dir = f"{out_root}/{name}"
        self.mesh_dir, self.tex_dir = f"{self.dir}/meshes", f"{self.dir}/textures"
        os.makedirs(self.mesh_dir, exist_ok=True)
        os.makedirs(self.tex_dir, exist_ok=True)
        self.rows, self.missing, self.bad = [], [], []

    def put(self, src, dst):
        if os.path.lexists(dst):
            os.remove(dst)
        if self.link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)

    def add(self, label, mesh, textures, anat=None):
        """mesh: inflated surface; textures: {kind: path}; all must share vertices."""
        if not os.path.isfile(mesh):
            self.missing.append((label, "inflated", mesh))
            return
        nv = n_vertices(mesh)
        present = {}
        for kind, path in textures.items():
            if not os.path.isfile(path):
                if kind == "dom_band":
                    self.missing.append((label, kind, path))
                continue
            nt = n_vertices(path)
            if nt != nv:
                self.bad.append((label, kind, nt, nv))
                continue
            present[kind] = path
        if "dom_band" not in present:
            return
        self.put(mesh, f"{self.mesh_dir}/{label}.inflated.surf.gii")
        if anat and os.path.isfile(anat) and n_vertices(anat) == nv:
            self.put(anat, f"{self.mesh_dir}/{label}.anat.surf.gii")
        for kind, path in present.items():
            self.put(path, f"{self.tex_dir}/{label}.{kind}.gii")
        self.rows.append({"label": label, "n_vertices": nv, "inflated_from": mesh,
                          **{f"{k}_from": v for k, v in present.items()},
                          "anat_from": anat if anat else ""})

    def finish(self):
        if self.rows:
            keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k != "label", k))
            with open(f"{self.dir}/manifest.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        print(f"{self.name:<16} {len(self.rows):>4} sessions  -> {self.dir}")
        for label, kind, path in self.missing:
            print(f"    missing {kind:<9} {label}: {path}")
        for label, kind, nt, nv in self.bad:
            print(f"    MISMATCH {kind:<8} {label}: texture {nt} vs inflated {nv} vertices -> skipped")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", choices=["sulc", "dpfstar", "both"], default="sulc",
                   help="registrations for RESTEX / ONREF")
    p.add_argument("--sub", default="sub-*")
    p.add_argument("--out", default=f"{BASE}/route_viewer")
    p.add_argument("--also-anat", action="store_true",
                   help="also copy the folded surface: original mesh (ORIG) or the "
                        "session's surface resampled onto the reference (RESTEX/ONREF)")
    p.add_argument("--link", action="store_true", help="symlinks instead of copies")
    args = p.parse_args()
    textures = ["sulc", "dpfstar"] if args.texture == "both" else [args.texture]
    inputs = f"{BASE}/msm_inputs"

    # ---- ORIG: every session with SPANGY on its original mesh ----------------
    orig = Route("ORIG", args.out, args.link)
    for infl in sorted(glob.glob(f"{inputs}/{args.sub}/ses-*/*_hemi-*.inflated.surf.gii")):
        label = os.path.basename(infl).replace(".inflated.surf.gii", "")
        sub = label.split("_")[0]
        sp = f"{BASE}/{sub}/analysis/spangy_pymesh"
        orig.add(label, infl,
                 {"dom_band": f"{sp}/textures/spangy_dom_band_{label}.gii",
                  "filt_mean_curv": f"{sp}/curvature/mean/filt_mean_curv_{label}.gii"},
                 anat=infl.replace(".inflated.surf.gii", ".anat.surf.gii") if args.also_anat else None)
    orig.finish()

    # ---- RESTEX and ONREF: every chained session, on the reference mesh ------
    for tex in textures:
        restex = Route(f"RESTEX_{tex}", args.out, args.link)
        onref = Route(f"ONREF_{tex}", args.out, args.link)
        for hdir in sorted(glob.glob(f"{BASE}/msm_chained/{tex}/{args.sub}/hemi-*")):
            sub, hemi = os.path.basename(os.path.dirname(hdir)), hdir[-1]
            spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
            if not spheres:
                continue
            ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
            ref_infl = f"{inputs}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.inflated.surf.gii"
            for sph in spheres:
                ses = re.search(r"_(ses-\d+)_in_", sph).group(1)
                label = f"{sub}_{ses}_hemi-{hemi}"
                anat = (f"{BASE}/surfaces_on_ref/{tex}/{sub}/hemi-{hemi}/{label}_on_{ref}.anat.surf.gii"
                        if args.also_anat else None)
                rs = f"{BASE}/spangy_on_ref/{tex}/{sub}/hemi-{hemi}/{label}_on_{ref}"
                restex.add(label, ref_infl,
                           {"dom_band": f"{rs}.dom_band.func.gii",
                            "filt_mean_curv": f"{rs}.filt_mean_curv.func.gii"}, anat=anat)
                od = f"{BASE}/{sub}/analysis/spangy_onref_{tex}"
                onref.add(label, ref_infl,
                          {"dom_band": f"{od}/textures/spangy_dom_band_{label}.gii",
                           "filt_mean_curv": f"{od}/curvature/mean/filt_mean_curv_{label}.gii"},
                          anat=anat)
        restex.finish()
        onref.finish()


if __name__ == "__main__":
    main()
