#!/usr/bin/env python3
"""
resample_spangy_bands.py — move SPANGY results from each session's original mesh
onto its animal's reference mesh WITHOUT interpolating band labels.

SPANGY's dominant-band map is computed (slam.spangy.local_dominance_map) from
continuous values: the filtered mean curvature recomposed in each band
("frecomposed") and the sign of the curvature. This script resamples those
continuous values with area-weighted interpolation and recomputes the dominant
band on the reference mesh with exactly slam's rule. Band borders then fall where
the signals actually cross, and dense meshes are averaged instead of subsampled.

Needs, per session (from spangy_babofet_pymesh.py, run after frecomposed was added):
    <sub>/analysis/spangy_pymesh/textures/spangy_frecomposed_<label>.gii
    <sub>/analysis/spangy_pymesh/curvature/mean/filt_mean_curv_<label>.gii
    <sub>/analysis/spangy_pymesh/textures/spangy_dom_band_<label>.gii   (for the check)
and the chained spheres from chain_to_reference.sh.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`):
    $PYTHON resample_spangy_bands.py                         # sulc registrations
    $PYTHON resample_spangy_bands.py --texture dpfstar
    $PYTHON resample_spangy_bands.py --sub sub-Aziza --roundtrip

Output, per session, in <BASE>/spangy_on_ref/<texture>/<sub>/hemi-<H>/:
    <label>_on_<ref>.dom_band_from_bands.func.gii   dominant band recomputed on the reference mesh
    <label>_on_<ref>.frecomposed.func.gii           per-band components on the reference mesh
    <label>_on_<ref>.filt_mean_curv.func.gii        filtered mean curvature on the reference mesh
    <label>_on_<ref>.principal_curv.func.gii        the two principal curvatures on the reference
                                                    mesh (2 columns, same order as SPANGY's
                                                    principal_curv_<label>.gii), when available
    <label>_on_<ref>.band_margin.func.gii           how clearly the winning band won (0 = tie)
    <label>_on_<ref>.det_band_from_bands.func.gii   determinant band recomputed on the reference mesh
    <label>_on_<ref>.det_band_margin.func.gii       how far the running total stays from 0 after
                                                    the deciding band (0 = on a border)
and a table <BASE>/spangy_on_ref/<texture>/resample_bands_summary.csv.
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile

import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spangy_determinance import determinant_band   # noqa: E402

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"


# --- io --------------------------------------------------------------------------
def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def load_metric(path):
    """(n_vertices, n_columns) array from a GIFTI metric with one or more columns."""
    d = nib.load(path).agg_data()
    d = np.column_stack(d) if isinstance(d, tuple) else np.asarray(d)
    return d.reshape(d.shape[0], -1).astype(np.float64)


def save_metric(values, path, struct):
    values = np.asarray(values, np.float32).reshape(len(values), -1)
    img = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(values[:, k], intent="NIFTI_INTENT_NONE",
                                 datatype="NIFTI_TYPE_FLOAT32") for k in range(values.shape[1])])
    nib.save(img, path)
    if struct:
        wb("-set-structure", path, struct)


def load_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float64),
            np.asarray(g.agg_data("triangle"), np.int64))


def vertex_areas(xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    va = np.zeros(len(xyz))
    for k in range(3):
        np.add.at(va, tri[:, k], fa / 3.0)
    return va


# --- slam's rule -------------------------------------------------------------------
def dominant_band(frecomposed, curvature):
    """
    Same rule as slam.spangy.local_dominance_map:
      diff = [band 1, band 2 - band 1, band 3 - band 2, ...]
      sulcal vertices (curvature <= 0): -(1 + index of the smallest diff)
      gyral vertices  (curvature >  0):  1 + index of the largest diff
    Also returns the margin: how far the winning value is from the runner-up
    (0 = tie, the vertex sits on a band border).
    """
    diff = np.concatenate([frecomposed[:, :1], np.diff(frecomposed, axis=1)], axis=1)
    sulcal = curvature <= 0
    band = np.where(sulcal, -(np.argmin(diff, axis=1) + 1), np.argmax(diff, axis=1) + 1)
    s = np.sort(diff, axis=1)
    margin = np.where(sulcal, s[:, 1] - s[:, 0], s[:, -1] - s[:, -2]) if diff.shape[1] > 1 \
        else np.zeros(len(diff))
    return band.astype(np.int32), margin


def agreement(a, b, w):
    return float(np.sum(w[np.rint(a) == np.rint(b)]) / np.sum(w))


# --- main --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", default="sulc", help="registrations used: sulc | dpfstar")
    p.add_argument("--sub", default="sub-*")
    p.add_argument("--roundtrip", action="store_true",
                   help="also move the result back to the original mesh and measure what "
                        "interpolation lost, for this method and for label resampling")
    args = p.parse_args()
    tex = args.texture
    inputs = f"{BASE}/msm_inputs"
    out_root = f"{BASE}/spangy_on_ref/{tex}"

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for hdir in sorted(glob.glob(f"{BASE}/msm_chained/{tex}/{args.sub}/hemi-*")):
            sub, hemi = os.path.basename(os.path.dirname(hdir)), hdir[-1]
            struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
            spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
            if not spheres:
                continue
            ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
            ref_sphere = f"{inputs}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.sphere.surf.gii"
            ref_anat = f"{inputs}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.anat.surf.gii"
            odir = f"{out_root}/{sub}/hemi-{hemi}"
            os.makedirs(odir, exist_ok=True)
            sp = f"{BASE}/{sub}/analysis/spangy_pymesh"
            print(f"[{sub} hemi-{hemi}] reference {ref}")

            for sph in spheres:
                ses = re.search(r"_(ses-\d+)_in_", sph).group(1)
                label = f"{sub}_{ses}_hemi-{hemi}"
                anat = f"{inputs}/{sub}/{ses}/{label}.anat.surf.gii"
                f_frec = f"{sp}/textures/spangy_frecomposed_{label}.gii"
                f_curv = f"{sp}/curvature/mean/filt_mean_curv_{label}.gii"
                f_princ = f"{sp}/curvature/principal/principal_curv_{label}.gii"
                f_band = f"{sp}/textures/spangy_dom_band_{label}.gii"
                f_det = f"{sp}/textures/spangy_det_band_{label}.gii"
                missing = [f for f in (f_frec, f_curv) if not os.path.isfile(f)]
                if missing:
                    print(f"  {ses}: missing {os.path.basename(missing[0])} -> skipped "
                          f"(rerun spangy_babofet_pymesh.py)")
                    continue

                # 1. load the continuous SPANGY values on the original mesh
                frec, curv = load_metric(f_frec), load_metric(f_curv)[:, 0]
                xyz_o, tri_o = load_surf(anat)
                if not (len(frec) == len(curv) == len(xyz_o)):
                    print(f"  {ses}: vertex counts differ (frecomposed {len(frec)}, curvature "
                          f"{len(curv)}, mesh {len(xyz_o)}) -> skipped")
                    continue
                a_orig = vertex_areas(xyz_o, tri_o)
                row = {"subject": sub, "hemi": hemi, "session": ses, "reference": ref,
                       "n_vertices_session": len(xyz_o), "n_bands": frec.shape[1]}

                # 2. check: slam's rule on the original mesh must give the saved map
                band_orig, _ = dominant_band(frec, curv)
                if os.path.isfile(f_band):
                    saved = load_metric(f_band)[:, 0]
                    row["check_rule_matches_saved_map"] = agreement(band_orig, saved, a_orig)
                    if row["check_rule_matches_saved_map"] < 0.999:
                        print(f"  {ses}: WARNING recomputed band matches the saved map on only "
                              f"{100 * row['check_rule_matches_saved_map']:.1f}% of the surface "
                              f"(slam version with a different rule?)")

                # 2b. same check for the determinant band (cumulative synthesis)
                det_orig = determinant_band(frec)
                if os.path.isfile(f_det):
                    row["check_det_rule_matches_saved_map"] = agreement(
                        det_orig, load_metric(f_det)[:, 0], a_orig)

                # 3. resample bands + curvature together, area-weighted, onto the reference mesh
                stack = f"{tmp}/stack.func.gii"
                moved = f"{tmp}/moved.func.gii"
                save_metric(np.column_stack([frec, curv]), stack, struct)
                wb("-metric-resample", stack, sph, ref_sphere, "ADAP_BARY_AREA", moved,
                   "-area-surfs", anat, ref_anat)
                m = load_metric(moved)
                frec_ref, curv_ref = m[:, :-1], m[:, -1]

                # 4. recompute the dominant band on the reference mesh
                band_ref, margin_ref = dominant_band(frec_ref, curv_ref)
                det_ref, det_margin_ref = determinant_band(frec_ref, return_margin=True)

                # 5. save
                stem = f"{odir}/{label}_on_{ref}"
                save_metric(band_ref, f"{stem}.dom_band_from_bands.func.gii", struct)
                save_metric(frec_ref, f"{stem}.frecomposed.func.gii", struct)
                save_metric(curv_ref, f"{stem}.filt_mean_curv.func.gii", struct)

                # principal curvatures: values only (directions are not resampled),
                # same area-weighted interpolation as the bands
                if os.path.isfile(f_princ):
                    princ = load_metric(f_princ)
                    if len(princ) == len(xyz_o):
                        save_metric(princ, f"{tmp}/princ.func.gii", struct)
                        wb("-metric-resample", f"{tmp}/princ.func.gii", sph, ref_sphere,
                           "ADAP_BARY_AREA", f"{tmp}/princ_moved.func.gii",
                           "-area-surfs", anat, ref_anat)
                        save_metric(load_metric(f"{tmp}/princ_moved.func.gii"),
                                    f"{stem}.principal_curv.func.gii", struct)
                    else:
                        print(f"  {ses}: principal curvature has {len(princ)} values, mesh "
                              f"{len(xyz_o)} -> not resampled")
                save_metric(margin_ref, f"{stem}.band_margin.func.gii", struct)
                save_metric(det_ref, f"{stem}.det_band_from_bands.func.gii", struct)
                save_metric(det_margin_ref, f"{stem}.det_band_margin.func.gii", struct)

                # 6. compare with simple label resampling (the earlier method)
                xyz_r, tri_r = load_surf(ref_anat)
                a_ref = vertex_areas(xyz_r, tri_r)
                lab_moved = f"{tmp}/lab_moved.func.gii"
                save_metric(band_orig, f"{tmp}/lab.func.gii", struct)
                wb("-metric-resample", f"{tmp}/lab.func.gii", sph, ref_sphere, "BARYCENTRIC",
                   lab_moved, "-largest")
                band_largest = load_metric(lab_moved)[:, 0]
                row["agree_with_label_resampling"] = agreement(band_ref, band_largest, a_ref)
                det_moved = f"{tmp}/det_moved.func.gii"
                save_metric(det_orig, f"{tmp}/det.func.gii", struct)
                wb("-metric-resample", f"{tmp}/det.func.gii", sph, ref_sphere, "BARYCENTRIC",
                   det_moved, "-largest")
                row["det_agree_with_label_resampling"] = agreement(
                    det_ref, load_metric(det_moved)[:, 0], a_ref)
                row["share_near_tie"] = float(np.sum(a_ref[margin_ref < 0.05 * np.abs(frec_ref).max()])
                                              / a_ref.sum())

                # 7. optional round trip: back to the original mesh, compare with the original
                if args.roundtrip:
                    back = f"{tmp}/back.func.gii"
                    save_metric(np.column_stack([frec_ref, curv_ref]), f"{tmp}/fwd.func.gii", struct)
                    wb("-metric-resample", f"{tmp}/fwd.func.gii", ref_sphere, sph, "ADAP_BARY_AREA",
                       back, "-area-surfs", ref_anat, anat)
                    mb = load_metric(back)
                    band_back, _ = dominant_band(mb[:, :-1], mb[:, -1])
                    row["roundtrip_bands_method"] = agreement(band_back, band_orig, a_orig)
                    lab_back = f"{tmp}/lab_back.func.gii"
                    wb("-metric-resample", lab_moved, ref_sphere, sph, "BARYCENTRIC",
                       lab_back, "-largest")
                    row["roundtrip_label_method"] = agreement(load_metric(lab_back)[:, 0],
                                                              band_orig, a_orig)
                    row["det_roundtrip_bands_method"] = agreement(
                        determinant_band(mb[:, :-1]), det_orig, a_orig)
                    det_back = f"{tmp}/det_back.func.gii"
                    wb("-metric-resample", det_moved, ref_sphere, sph, "BARYCENTRIC",
                       det_back, "-largest")
                    row["det_roundtrip_label_method"] = agreement(
                        load_metric(det_back)[:, 0], det_orig, a_orig)

                rows.append(row)
                msg = (f"  {ses}: done, agrees with label resampling on "
                       f"{100 * row['agree_with_label_resampling']:.1f}% of the surface")
                if args.roundtrip:
                    msg += (f"; round trip keeps {100 * row['roundtrip_bands_method']:.1f}% "
                            f"(label method {100 * row['roundtrip_label_method']:.1f}%)")
                print(msg)

    if not rows:
        print("Nothing resampled: check that frecomposed textures and chained spheres exist.")
        return
    df = pd.DataFrame(rows)
    os.makedirs(out_root, exist_ok=True)
    out = f"{out_root}/resample_bands_summary.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} sessions -> {out}")
    cols = [c for c in ("check_rule_matches_saved_map", "check_det_rule_matches_saved_map",
                        "agree_with_label_resampling", "det_agree_with_label_resampling",
                        "share_near_tie", "roundtrip_bands_method", "roundtrip_label_method",
                        "det_roundtrip_bands_method", "det_roundtrip_label_method")
            if c in df]
    print((100 * df[cols]).describe().loc[["min", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
