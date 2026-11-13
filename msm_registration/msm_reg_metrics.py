#!/usr/bin/env python3
"""
msm_reg_metrics.py — registration metrics for every MSM pair, following the
group's msm_reg_stats.py, adapted to the BaboFet session-to-session layout.

For each pair  <sub> hemi-<H>  ses-A -> ses-B  (input A, reference B):

  Distortion (as msm_reg_stats.py)
    sphere normalisation   registered sphere scaled to the input sphere's max radius
    jacobian_integral      sum over faces of |log2(area_reg / area_in)| * area_in
    jacobian_mean          the same divided by the total input-sphere area
                           (area-weighted mean |log2 J|; 1 = a 2x change)
    jacobian_frac_gt1      share of the sphere area whose area changed more than 2x
    area_diff_pct_*        per-face (area_reg - area_in) / area_in * 100:
                           median, p5, p95, and median of the absolute value
    flipped_faces          triangles folded over in the registered sphere

  Similarity
    corr_<tex>_msm         Pearson r, MSM's transformed_and_reprojected texture vs
                           the reference texture (exactly as msm_reg_stats.py)
    corr_<tex>_before      same texture, spheres only overlaid (no registration)
    corr_curv_before/after mean curvature (msm_inputs *.curv.func.gii), NOT used
                           by either registration: a fair sulc vs DPF-star test
    corr_<other>_after     the texture the registration did NOT use (sulc for a
                           DPF-star registration and vice versa)

  Overlap (Dice, labels moved by nearest neighbour as in overlap())
    dice_sulcal / dice_gyral / agree_sign
                           labels = sign of mean curvature (independent of both
                           registration textures), Dice per class + % agreement
    dice_roi_<name>        with --roi-pattern: painted ROI (e.g. STS) present in
                           both sessions
    dice_B3..B6, agree_band with --bands: SPANGY dominant band (|label|) from
                           spangy_pymesh. Bands change with age, so this is
                           folding-pattern similarity, not a ground truth.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`):
    $PYTHON msm_reg_metrics.py                         # sulc and DPF-star registrations
    $PYTHON msm_reg_metrics.py --texture sulc --sub sub-Aziza
    $PYTHON msm_reg_metrics.py --bands \
        --roi-pattern "<BASE>/{sub}/rois/STS/{sub}_{ses}_hemi-{hemi}_STS.gii"
    $PYTHON msm_reg_metrics.py --save-maps             # per-vertex |log2 J| GIFTI per pair

Output: <BASE>/msm_registered/metrics/msm_reg_metrics.csv (one row per pair and texture)
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"
OTHER = {"sulc": "dpfstar", "dpfstar": "sulc"}


# --- io --------------------------------------------------------------------------
def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)


def load_surf(path):
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float64),
            np.asarray(g.agg_data("triangle"), np.int64))


def load_metric(path):
    d = nib.load(path).agg_data()
    d = d[0] if isinstance(d, tuple) else d
    return np.asarray(d, np.float64).ravel()


def save_metric(values, path, struct):
    img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.asarray(values, np.float32), intent="NIFTI_INTENT_SHAPE",
        datatype="NIFTI_TYPE_FLOAT32")])
    nib.save(img, path)
    wb("-set-structure", path, struct)


# --- metrics (msm_reg_stats.py) ---------------------------------------------------
def face_areas(xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def sphere_normalization(xyz_source, xyz_deform):
    """Scale the deformed sphere to the source sphere's maximum radius."""
    return xyz_deform * (np.linalg.norm(xyz_source, axis=1).max()
                         / np.linalg.norm(xyz_deform, axis=1).max())


def vertex_average_from_faces(n_vertices, faces, face_values, face_areas_):
    """One third of each face's value and area to each of its vertices."""
    vv, va = np.zeros(n_vertices), np.zeros(n_vertices)
    for k in range(3):
        np.add.at(vv, faces[:, k], face_values / 3.0)
        np.add.at(va, faces[:, k], face_areas_ / 3.0)
    return vv, va


def flipped(xyz_in, xyz_reg, tri):
    """Triangles whose orientation on the sphere changed between input and registration."""
    def sign(xyz):
        v0, v1, v2 = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
        return np.sign(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), (v0 + v1 + v2) / 3))
    return int(np.sum(sign(xyz_in) != sign(xyz_reg)))


def overlap(target_xyz, deformed_xyz, parcels):
    """Labels of the deformed (registered) sphere moved onto the target sphere, nearest vertex."""
    _, idx = cKDTree(deformed_xyz).query(target_xyz)
    return parcels[idx]


def dice(source, target, labels):
    """Dice (%) per label, as msm_reg_stats.dice()."""
    out = {}
    for p in labels:
        a, b = source == p, target == p
        n = a.sum() + b.sum()
        out[p] = 200.0 * np.logical_and(a, b).sum() / n if n else np.nan
    return out


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(pearsonr(x[ok], y[ok])[0]) if ok.sum() > 2 else np.nan


# --- main -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", choices=["sulc", "dpfstar", "both"], default="both")
    p.add_argument("--sub", default="sub-*")
    p.add_argument("--bands", action="store_true", help="Dice of SPANGY dominant bands")
    p.add_argument("--roi-pattern", default=None,
                   help="painted ROI per session; placeholders {sub} {ses} {hemi} {hemiword}")
    p.add_argument("--roi-name", default="roi")
    p.add_argument("--save-maps", action="store_true",
                   help="save per-vertex |log2 J| next to each registration")
    args = p.parse_args()
    textures = ["sulc", "dpfstar"] if args.texture == "both" else [args.texture]
    inputs = f"{BASE}/msm_inputs"
    out_dir = f"{BASE}/msm_registered/metrics"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for tex in textures:
            regs = sorted(glob.glob(f"{BASE}/msm_registered/{tex}/{args.sub}/hemi-*/"
                                    "*_to_ses-*.sphere.reg.surf.gii"))
            print(f"[{tex}] {len(regs)} registrations")
            for reg in regs:
                name = os.path.basename(reg).replace(".sphere.reg.surf.gii", "")
                sub = name.split("_hemi-")[0]
                hemi = re.search(r"_hemi-([LR])_", name).group(1)
                ses_a, ses_b = re.findall(r"(ses-\d+)", name)
                struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
                pa = f"{inputs}/{sub}/{ses_a}/{sub}_{ses_a}_hemi-{hemi}"
                pb = f"{inputs}/{sub}/{ses_b}/{sub}_{ses_b}_hemi-{hemi}"
                prefix = reg.replace(".sphere.reg.surf.gii", "")

                xyz_in, tri = load_surf(f"{pa}.sphere.surf.gii")
                xyz_reg, tri_reg = load_surf(reg)
                xyz_ref, _ = load_surf(f"{pb}.sphere.surf.gii")
                if xyz_reg.shape != xyz_in.shape or not np.array_equal(tri, tri_reg):
                    print(f"  {name}: registered sphere does not match the input mesh -> skipped")
                    continue

                # ---- distortion ------------------------------------------------
                xyz_reg = sphere_normalization(xyz_in, xyz_reg)
                a_in, a_reg = face_areas(xyz_in, tri), face_areas(xyz_reg, tri)
                with np.errstate(divide="ignore", invalid="ignore"):
                    log2j = np.log2(a_reg / a_in)
                    pct = (a_reg - a_in) / a_in * 100.0
                ok = np.isfinite(log2j)
                jac_faces = np.abs(log2j[ok]) * a_in[ok]
                row = {"subject": sub, "hemi": hemi, "input": ses_a, "reference": ses_b,
                       "gap": int(ses_b[4:]) - int(ses_a[4:]), "texture": tex,
                       "n_vertices": len(xyz_in),
                       "jacobian_integral": float(jac_faces.sum()),
                       "jacobian_mean": float(jac_faces.sum() / a_in[ok].sum()),
                       "jacobian_frac_gt1": float(a_in[ok][np.abs(log2j[ok]) > 1].sum() / a_in[ok].sum()),
                       "area_diff_pct_median": float(np.nanmedian(pct)),
                       "area_diff_pct_p5": float(np.nanpercentile(pct, 5)),
                       "area_diff_pct_p95": float(np.nanpercentile(pct, 95)),
                       "area_diff_abs_pct_median": float(np.nanmedian(np.abs(pct))),
                       "flipped_faces": flipped(xyz_in, xyz_reg, tri)}
                if args.save_maps:
                    vv, va = vertex_average_from_faces(len(xyz_in), tri[ok], jac_faces, a_in[ok])
                    with np.errstate(invalid="ignore"):
                        save_metric(np.where(va > 0, vv / va, 0), f"{prefix}.jacobian_abslog2.func.gii", struct)

                # ---- similarity ------------------------------------------------
                ref_tex = load_metric(f"{pb}.{tex}.func.gii")
                msm_out = f"{prefix}.transformed_and_reprojected.func.gii"
                if os.path.isfile(msm_out):
                    row[f"corr_tex_msm"] = corr(load_metric(msm_out), ref_tex)

                def moved(metric, sphere):
                    out = f"{tmp}/m.func.gii"
                    wb("-metric-resample", metric, sphere, f"{pb}.sphere.surf.gii",
                       "BARYCENTRIC", out)
                    return load_metric(out)

                row["corr_tex_before"] = corr(moved(f"{pa}.{tex}.func.gii", f"{pa}.sphere.surf.gii"), ref_tex)
                if os.path.isfile(f"{pa}.curv.func.gii") and os.path.isfile(f"{pb}.curv.func.gii"):
                    ref_curv = load_metric(f"{pb}.curv.func.gii")
                    row["corr_curv_before"] = corr(moved(f"{pa}.curv.func.gii", f"{pa}.sphere.surf.gii"), ref_curv)
                    row["corr_curv_after"] = corr(moved(f"{pa}.curv.func.gii", reg), ref_curv)
                other = OTHER[tex]
                if os.path.isfile(f"{pa}.{other}.func.gii") and os.path.isfile(f"{pb}.{other}.func.gii"):
                    row["corr_other_after"] = corr(moved(f"{pa}.{other}.func.gii", reg),
                                                   load_metric(f"{pb}.{other}.func.gii"))

                # ---- overlap (Dice) --------------------------------------------
                if os.path.isfile(f"{pa}.curv.func.gii"):
                    src = np.sign(load_metric(f"{pa}.curv.func.gii")).astype(int)
                    tgt = np.sign(load_metric(f"{pb}.curv.func.gii")).astype(int)
                    moved_lab = overlap(xyz_ref, xyz_reg, src)
                    # wb mean curvature: sign follows the surface normal; the two
                    # classes are named by which one holds the deeper sulc
                    sulc_b = load_metric(f"{pb}.sulc.func.gii")
                    sulcal = 1 if sulc_b[tgt == 1].mean() > sulc_b[tgt == -1].mean() else -1
                    d = dice(tgt, moved_lab, [sulcal, -sulcal])
                    row["dice_sulcal"], row["dice_gyral"] = d[sulcal], d[-sulcal]
                    row["agree_sign"] = 100.0 * float(np.mean(tgt == moved_lab))

                if args.roi_pattern:
                    hw = "left" if hemi == "L" else "right"
                    ra = args.roi_pattern.format(sub=sub, ses=ses_a, hemi=hemi, hemiword=hw)
                    rb = args.roi_pattern.format(sub=sub, ses=ses_b, hemi=hemi, hemiword=hw)
                    if os.path.isfile(ra) and os.path.isfile(rb):
                        la, lb = load_metric(ra) != 0, load_metric(rb) != 0
                        if len(la) == len(xyz_in) and len(lb) == len(xyz_ref):
                            moved_roi = overlap(xyz_ref, xyz_reg, la.astype(int))
                            row[f"dice_roi_{args.roi_name}"] = dice(lb.astype(int), moved_roi, [1])[1]

                if args.bands:
                    sp = f"{BASE}/{sub}/analysis/spangy_pymesh/textures/spangy_dom_band_"
                    fa, fb = f"{sp}{sub}_{ses_a}_hemi-{hemi}.gii", f"{sp}{sub}_{ses_b}_hemi-{hemi}.gii"
                    if os.path.isfile(fa) and os.path.isfile(fb):
                        ba = np.abs(np.rint(load_metric(fa))).astype(int)
                        bb = np.abs(np.rint(load_metric(fb))).astype(int)
                        if len(ba) == len(xyz_in) and len(bb) == len(xyz_ref):
                            moved_b = overlap(xyz_ref, xyz_reg, ba)
                            for b, v in dice(bb, moved_b, [3, 4, 5, 6]).items():
                                row[f"dice_B{b}"] = v
                            row["agree_band"] = 100.0 * float(np.mean(bb == moved_b))

                rows.append(row)
                print(f"  {name}: J_mean {row['jacobian_mean']:.3f}  "
                      f"r_curv {row.get('corr_curv_before', np.nan):.3f}->{row.get('corr_curv_after', np.nan):.3f}  "
                      f"Dice sulcal {row.get('dice_sulcal', np.nan):.1f}")

    if not rows:
        print("No registrations found.")
        return
    df = pd.DataFrame(rows)
    out = f"{out_dir}/msm_reg_metrics.csv"
    if args.sub != "sub-*" or args.texture != "both":
        out = out.replace(".csv", f"_{args.texture}_{args.sub.replace('*', 'all')}.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out}")
    num = [c for c in df.columns if c not in ("subject", "hemi", "input", "reference", "texture")]
    print(df.groupby("texture")[num].median().T.round(3).to_string())


if __name__ == "__main__":
    main()
