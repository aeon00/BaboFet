#!/usr/bin/env python3
"""
build_spangy_timelapse.py — run ONCE on the cluster. For each animal/hemisphere,
brings every session's SPANGY texture onto the reference mesh (via the chained
MSM spheres) and packs everything into one .npz file for the local viewer.

Per animal/hemisphere the .npz holds:
  faces    (F, 3)    triangles of the reference mesh
  coords   (S, V, 3) each session's OWN anatomy, resampled onto the reference
                     mesh (same vertices for all sessions -> can be morphed)
  values   (S, V, C) each session's SPANGY texture on the reference mesh
  sessions (S,)      session names, ref, sub, hemi, kind

Usage (cluster, after `source env_setup.sh` with NEED_FREESURFER=0):
  python build_spangy_timelapse.py                 # all animals, dominant-band maps
  python build_spangy_timelapse.py --sub sub-Fabienne
Default --pattern (local dominant band from spangy_babofet_pymesh.py, i.e. SPANGY
run on the same Pymesh meshes that were registered):
  <BASE>/{sub}/analysis/spangy_pymesh/textures/spangy_dom_band_{sub}_{ses}_hemi-{hemi}.gii
--onref: SPANGY was run on the surfaces from resample_to_ref.py (spangy_babofet_pymesh.py
  --onref); textures and surfaces are packed as they are, nothing is resampled.
--save-gii: also keep the resampled textures and surfaces as GIFTI in <BASE>/spangy_on_ref/.
Older textures (other meshes):
  --pattern "<BASE>/{sub}/analysis/spangy/textures/spangy_dom_band_{sub}_{ses}_hemi_hemi-{hemiword}_level_0.45.gii"
Placeholders in --pattern: {sub} {ses} {hemi} (L/R) {hemiword} (left/right).
--kind label       categorical maps (local dominant band): nearest-weight resampling
--kind continuous  numeric maps (band power, curvature...): area-weighted resampling
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"


def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True)


def load_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float32),
            np.asarray(g.agg_data("triangle"), np.int32))


def load_metric(path):
    d = nib.load(path).agg_data()
    d = np.column_stack(d) if isinstance(d, tuple) else np.asarray(d)
    return d.reshape(d.shape[0], -1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", help="path pattern of the SPANGY texture",
                   default=f"{BASE}/{{sub}}/analysis/spangy_pymesh/textures/"
                           "spangy_dom_band_{sub}_{ses}_hemi-{hemi}.gii")
    p.add_argument("--kind", choices=["label", "continuous"], default="label")
    p.add_argument("--texture", default="sulc", help="which registrations were chained")
    p.add_argument("--sub", default="sub-*")
    p.add_argument("--name", default=None, help="tag in output file name (default: kind)")
    p.add_argument("--check", action="store_true",
                   help="only check that every texture matches its mesh; write nothing")
    p.add_argument("--onref", action="store_true",
                   help="use SPANGY run on the reference-mesh surfaces (spangy_onref): "
                        "nothing is resampled")
    p.add_argument("--save-gii", action="store_true",
                   help="also keep the resampled textures/surfaces as GIFTI in spangy_on_ref/")
    args = p.parse_args()
    n_bad = 0
    if args.onref and args.name is None:
        args.name = f"{args.kind}_onref"

    chain_root = f"{BASE}/msm_chained/{args.texture}"
    in_root = f"{BASE}/msm_inputs"
    out_dir = f"{BASE}/spangy_timelapse"
    os.makedirs(out_dir, exist_ok=True)
    tag = args.name or args.kind

    for hdir in sorted(glob.glob(f"{chain_root}/{args.sub}/hemi-*")):
        sub = os.path.basename(os.path.dirname(hdir))
        hemi = hdir[-1]
        hemiword = "left" if hemi == "L" else "right"
        spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
        if not spheres:
            continue
        ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
        ref_sphere = f"{in_root}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.sphere.surf.gii"
        ref_anat = f"{in_root}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.anat.surf.gii"
        _, faces = load_surf(ref_anat)

        sessions, coords, values = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            for sph in spheres:
                ses = re.search(r"_(ses-\d+)_in_", sph).group(1)
                tex = args.pattern.format(sub=sub, ses=ses, hemi=hemi, hemiword=hemiword)
                anat = f"{in_root}/{sub}/{ses}/{sub}_{ses}_hemi-{hemi}.anat.surf.gii"
                if args.onref:
                    # SPANGY already ran on the surface resampled onto the reference
                    # mesh: texture and surface are used as they are, no resampling.
                    surf = (f"{BASE}/surfaces_on_ref/{sub}/hemi-{hemi}/"
                            f"{sub}_{ses}_hemi-{hemi}_on_{ref}.anat.surf.gii")
                    tex = (f"{BASE}/{sub}/analysis/spangy_onref/textures/"
                           f"spangy_dom_band_{sub}_{ses}_hemi-{hemi}.gii")
                    missing = [f for f in (surf, tex) if not os.path.isfile(f)]
                    if missing:
                        print(f"  [{sub} {hemi} {ses}] missing {missing[0]} -> skipped")
                        continue
                    xyz, val = load_surf(surf)[0], load_metric(tex)
                    if not (len(xyz) == len(val) == faces.max() + 1):
                        print(f"  [{sub} {hemi} {ses}] MISMATCH: surface {len(xyz)}, texture "
                              f"{len(val)}, reference {faces.max() + 1} vertices -> skipped")
                        n_bad += 1
                        continue
                    if not args.check:
                        sessions.append(ses); coords.append(xyz); values.append(val)
                    else:
                        print(f"  [{sub} {hemi} {ses}] ok ({len(xyz)} vertices)")
                    continue
                if not os.path.isfile(tex):
                    print(f"  [{sub} {hemi} {ses}] no texture at {tex} -> skipped")
                    continue
                # the texture must be defined on the same mesh that was registered
                n_tex, n_mesh = load_metric(tex).shape[0], load_surf(anat)[0].shape[0]
                if n_tex != n_mesh:
                    other = "R" if hemi == "L" else "L"
                    o_anat = f"{in_root}/{sub}/{ses}/{sub}_{ses}_hemi-{other}.anat.surf.gii"
                    n_other = load_surf(o_anat)[0].shape[0] if os.path.isfile(o_anat) else None
                    hint = (f"; matches the hemi-{other} mesh ({n_other}) -> left/right swapped?"
                            if n_tex == n_other else "; matches neither hemisphere -> "
                            "SPANGY was run on a different mesh")
                    print(f"  [{sub} {hemi} {ses}] MISMATCH: texture {n_tex} vertices, "
                          f"registered mesh {n_mesh}{hint} -> skipped")
                    n_bad += 1
                    continue
                if args.check:
                    print(f"  [{sub} {hemi} {ses}] ok ({n_mesh} vertices)")
                    continue
                t_out, s_out = f"{tmp}/{ses}.func.gii", f"{tmp}/{ses}.surf.gii"
                if args.kind == "label":   # keep band numbers whole: no averaging
                    wb("-metric-resample", tex, sph, ref_sphere, "BARYCENTRIC", t_out, "-largest")
                else:
                    wb("-metric-resample", tex, sph, ref_sphere, "ADAP_BARY_AREA", t_out,
                       "-area-surfs", anat, ref_anat)
                wb("-surface-resample", anat, sph, ref_sphere, "BARYCENTRIC", s_out)
                if args.save_gii:
                    gdir = f"{BASE}/spangy_on_ref/{sub}/hemi-{hemi}"
                    os.makedirs(gdir, exist_ok=True)
                    stem = f"{gdir}/{sub}_{ses}_hemi-{hemi}_on_{ref}"
                    struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
                    shutil.copy(t_out, f"{stem}.{tag}.func.gii")
                    shutil.copy(s_out, f"{stem}.anat.surf.gii")
                    wb("-set-structure", f"{stem}.{tag}.func.gii", struct)
                    wb("-set-structure", f"{stem}.anat.surf.gii", struct,
                       "-surface-type", "ANATOMICAL")
                sessions.append(ses)
                values.append(load_metric(t_out))
                coords.append(load_surf(s_out)[0])

        if args.check:
            continue
        if len(sessions) < 2:
            print(f"[{sub} hemi-{hemi}] fewer than 2 sessions with textures, skipped")
            continue
        out = f"{out_dir}/{sub}_hemi-{hemi}_{tag}_on_{ref}.npz"
        np.savez_compressed(out, faces=faces, coords=np.stack(coords), values=np.stack(values),
                            sessions=np.array(sessions), ref=ref, sub=sub, hemi=hemi,
                            kind=args.kind)
        print(f"[{sub} hemi-{hemi}] {len(sessions)} sessions -> {out}")

    if n_bad:
        print(f"---- {n_bad} session(s) skipped because the texture does not match the mesh")


if __name__ == "__main__":
    sys.exit(main())