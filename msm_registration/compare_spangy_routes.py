#!/usr/bin/env python3
"""
compare_spangy_routes.py — compare the three ways of getting SPANGY results for
every session of every animal/hemisphere:

  ORIG   SPANGY on the original Pymesh mesh            (analysis/spangy_pymesh/)
  RESTEX ORIG textures resampled onto the reference mesh   (written here, saved as GIFTI)
  ONREF  SPANGY on the surface resampled onto the reference mesh
                                                       (analysis/spangy_onref_<texture>/)

RESTEX and ONREF live on the same vertices, so they are compared vertex by vertex.
ORIG lives on its own mesh, so it is compared through quantities that do not need
shared vertices: the share of surface area dominated by each band, and the
global band powers from SPANGY's per-file JSON.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`):
    $PYTHON compare_spangy_routes.py                       # all animals, sulc registrations
    $PYTHON compare_spangy_routes.py --sub sub-Aziza
    $PYTHON compare_spangy_routes.py --texture dpfstar     # DPF-star registrations

Saved resampled ORIG textures (RESTEX):
    <BASE>/spangy_on_ref/<texture>/<sub>/hemi-<H>/
        <sub>_<ses>_hemi-<H>_on_<ref>.dom_band.func.gii         (labels, nearest weight)
        <sub>_<ses>_hemi-<H>_on_<ref>.filt_mean_curv.func.gii   (area-weighted)
Comparison tables:
    <BASE>/spangy_route_comparison/<texture>/
        per_session.csv        one row per animal/hemisphere/session
        band_area_fractions.csv   area share of each signed band, per route
        band_powers.csv        global band powers, ORIG vs ONREF
"""
import argparse
import glob
import json
import os
import re
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"
BANDS = range(0, 7)


# --- io ------------------------------------------------------------------------
def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True)


def load_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float64),
            np.asarray(g.agg_data("triangle"), np.int64))


def load_metric(path):
    d = nib.load(path).agg_data()
    d = d[0] if isinstance(d, tuple) else d
    return np.asarray(d, np.float64).ravel()


def vertex_areas(xyz, tri):
    """One third of the area of every triangle around each vertex."""
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    va = np.zeros(len(xyz))
    for k in range(3):
        np.add.at(va, tri[:, k], fa / 3.0)
    return va


# --- metrics -------------------------------------------------------------------
def band_fractions(labels, areas):
    """Area share of each signed band label, e.g. {'B4_gyral': 0.21, 'B4_sulcal': 0.18}."""
    lab = np.rint(labels).astype(int)
    tot = areas.sum()
    out = {}
    for b in range(1, 7):                       # slam labels are ±1..±(nlevels-1)
        out[f"B{b}_gyral"] = areas[lab == b].sum() / tot
        out[f"B{b}_sulcal"] = areas[lab == -b].sum() / tot
    return out


def cohen_kappa(x, y, w):
    cats = np.union1d(x, y)
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    np.add.at(m, (np.vectorize(idx.get)(x), np.vectorize(idx.get)(y)), w)
    m /= m.sum()
    po, pe = np.trace(m), (m.sum(0) * m.sum(1)).sum()
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(x, y, w):
    """Area-weighted agreement between two signed band maps on the same vertices."""
    x, y = np.rint(x).astype(int), np.rint(y).astype(int)
    w = w / w.sum()
    return {
        "same_label": float(w[x == y].sum()),                     # same band and same side
        "same_band": float(w[np.abs(x) == np.abs(y)].sum()),      # same band, either side
        "same_side": float(w[np.sign(x) == np.sign(y)].sum()),    # sulcal/gyral agree
        "band_off_by_1": float(w[np.abs(np.abs(x) - np.abs(y)) == 1].sum()),
        "kappa": float(cohen_kappa(x, y, w)),
    }


def json_row(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# --- main ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", default="sulc", help="registrations used: sulc | dpfstar")
    p.add_argument("--sub", default="sub-*")
    args = p.parse_args()

    tex = args.texture
    chain_root = f"{BASE}/msm_chained/{tex}"
    in_root = f"{BASE}/msm_inputs"
    out_dir = f"{BASE}/spangy_route_comparison/{tex}"
    os.makedirs(out_dir, exist_ok=True)

    rows, frac_rows, power_rows = [], [], []
    for hdir in sorted(glob.glob(f"{chain_root}/{args.sub}/hemi-*")):
        sub = os.path.basename(os.path.dirname(hdir))
        hemi = hdir[-1]
        struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
        spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
        if not spheres:
            continue
        ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
        ref_sphere = f"{in_root}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.sphere.surf.gii"
        ref_anat = f"{in_root}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.anat.surf.gii"
        sdir = f"{BASE}/spangy_on_ref/{tex}/{sub}/hemi-{hemi}"
        os.makedirs(sdir, exist_ok=True)
        orig_dir = f"{BASE}/{sub}/analysis/spangy_pymesh"
        onref_dir = f"{BASE}/{sub}/analysis/spangy_onref_{tex}"
        print(f"[{sub} hemi-{hemi}] reference {ref}")

        for sph in spheres:
            ses = re.search(r"_(ses-\d+)_in_", sph).group(1)
            label = f"{sub}_{ses}_hemi-{hemi}"
            anat = f"{in_root}/{sub}/{ses}/{label}.anat.surf.gii"
            onref_surf = f"{BASE}/surfaces_on_ref/{tex}/{sub}/hemi-{hemi}/{label}_on_{ref}.anat.surf.gii"
            f = {
                "orig_band": f"{orig_dir}/textures/spangy_dom_band_{label}.gii",
                "orig_curv": f"{orig_dir}/curvature/mean/filt_mean_curv_{label}.gii",
                "onref_band": f"{onref_dir}/textures/spangy_dom_band_{label}.gii",
                "onref_curv": f"{onref_dir}/curvature/mean/filt_mean_curv_{label}.gii",
            }
            missing = [k for k in ("orig_band", "onref_band") if not os.path.isfile(f[k])]
            if missing or not os.path.isfile(onref_surf):
                print(f"  {ses}: missing {missing or ['surfaces_on_ref']} -> skipped")
                continue

            # ---- RESTEX: resample the ORIG textures onto the reference mesh, keep them
            stem = f"{sdir}/{label}_on_{ref}"
            restex_band = f"{stem}.dom_band.func.gii"
            wb("-metric-resample", f["orig_band"], sph, ref_sphere, "BARYCENTRIC",
               restex_band, "-largest")
            wb("-set-structure", restex_band, struct)
            restex_curv = None
            if os.path.isfile(f["orig_curv"]):
                restex_curv = f"{stem}.filt_mean_curv.func.gii"
                wb("-metric-resample", f["orig_curv"], sph, ref_sphere, "ADAP_BARY_AREA",
                   restex_curv, "-area-surfs", anat, ref_anat)
                wb("-set-structure", restex_curv, struct)

            # ---- load everything
            xyz_o, tri_o = load_surf(anat)
            xyz_r, tri_r = load_surf(onref_surf)
            a_orig, a_ref = vertex_areas(xyz_o, tri_o), vertex_areas(xyz_r, tri_r)
            orig, restex, onref = (load_metric(f["orig_band"]), load_metric(restex_band),
                                   load_metric(f["onref_band"]))
            if len(orig) != len(xyz_o) or len(onref) != len(xyz_r) or len(restex) != len(xyz_r):
                print(f"  {ses}: vertex count mismatch "
                      f"(orig {len(orig)}/{len(xyz_o)}, onref {len(onref)}/{len(xyz_r)}) -> skipped")
                continue

            # ---- vertex-wise: RESTEX vs ONREF (same vertices; weighted by area)
            row = {"subject": sub, "hemi": hemi, "session": ses, "ref": ref,
                   "n_vertices_orig": len(xyz_o), "n_vertices_ref": len(xyz_r)}
            row.update({f"restex_vs_onref_{k}": v
                        for k, v in agreement(restex, onref, a_ref).items()})
            if restex_curv and os.path.isfile(f["onref_curv"]):
                c1, c2 = load_metric(restex_curv), load_metric(f["onref_curv"])
                row["curv_corr_restex_vs_onref"] = float(np.corrcoef(c1, c2)[0, 1])

            # ---- area shares per band: all three routes
            fr = {"ORIG": band_fractions(orig, a_orig),
                  "RESTEX": band_fractions(restex, a_ref),
                  "ONREF": band_fractions(onref, a_ref)}
            for route, d in fr.items():
                frac_rows.append({"subject": sub, "hemi": hemi, "session": ses,
                                  "route": route, **d})
            keys = list(fr["ORIG"])
            row["frac_maxdiff_restex_vs_orig"] = max(abs(fr["RESTEX"][k] - fr["ORIG"][k]) for k in keys)
            row["frac_maxdiff_onref_vs_orig"] = max(abs(fr["ONREF"][k] - fr["ORIG"][k]) for k in keys)
            row["area_ratio_ref_vs_orig"] = a_ref.sum() / a_orig.sum()

            # ---- global band powers: ORIG vs ONREF (RESTEX has none: it is a resampled map)
            jo = json_row(f"{orig_dir}/results/per_file/{label}.json")
            jr = json_row(f"{onref_dir}/results/per_file/{label}.json")
            if jo and jr:
                for b in BANDS:
                    po, pr = jo.get(f"band_power_B{b}"), jr.get(f"band_power_B{b}")
                    if po is None or pr is None:
                        continue
                    power_rows.append({"subject": sub, "hemi": hemi, "session": ses,
                                       "band": f"B{b}", "ORIG": po, "ONREF": pr,
                                       "rel_diff": (pr - po) / po if po else np.nan})
                if jo.get("analyze_folding_power"):
                    row["afp_rel_diff_onref_vs_orig"] = (
                        jr["analyze_folding_power"] - jo["analyze_folding_power"]) / jo["analyze_folding_power"]
            rows.append(row)
            print(f"  {ses}: same label {row['restex_vs_onref_same_label']:.3f}  "
                  f"kappa {row['restex_vs_onref_kappa']:.3f}  "
                  f"band-share diff vs ORIG: restex {row['frac_maxdiff_restex_vs_orig']:.3f}, "
                  f"onref {row['frac_maxdiff_onref_vs_orig']:.3f}")

    if not rows:
        print("Nothing compared: check that both SPANGY runs and surfaces_on_ref exist.")
        return
    pd.DataFrame(rows).to_csv(f"{out_dir}/per_session.csv", index=False)
    pd.DataFrame(frac_rows).to_csv(f"{out_dir}/band_area_fractions.csv", index=False)
    if power_rows:
        pd.DataFrame(power_rows).to_csv(f"{out_dir}/band_powers.csv", index=False)

    df = pd.DataFrame(rows)
    cols = [c for c in ["restex_vs_onref_same_label", "restex_vs_onref_same_band",
                        "restex_vs_onref_same_side", "restex_vs_onref_kappa",
                        "curv_corr_restex_vs_onref", "frac_maxdiff_restex_vs_orig",
                        "frac_maxdiff_onref_vs_orig", "afp_rel_diff_onref_vs_orig",
                        "area_ratio_ref_vs_orig"] if c in df]
    print("\n==== summary over all sessions (median [min, max])")
    for c in cols:
        print(f"  {c:<32} {df[c].median():8.3f}  [{df[c].min():.3f}, {df[c].max():.3f}]")
    if power_rows:
        pw = pd.DataFrame(power_rows)
        print("\n  band power, ONREF vs ORIG, median relative difference per band:")
        print(pw.groupby("band")["rel_diff"].median().round(3).to_string())
    print(f"\nTables in {out_dir}/   resampled textures in {BASE}/spangy_on_ref/{tex}/")


if __name__ == "__main__":
    main()
