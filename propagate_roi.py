#!/usr/bin/env python3
"""
propagate_roi.py — paint a sulcus (or gyrus) ONCE on an animal's reference mesh,
carry it to every session, and read SPANGY's own values inside it on each
session's ORIGINAL mesh.

Workflow
    1. SPANGY on the original meshes         spangy_babofet_pymesh.py
    2. Bands moved onto the reference mesh    resample_spangy_bands.py
    3. Paint the region once, in Anatomist SurfPaint, on any surface of
       <BASE>/surfaces_on_ref/<texture>/<sub>/hemi-<H>/ (they all share the
       reference mesh's vertices; the oldest session shows the folds best).
       Save the texture as
           <BASE>/<sub>/rois/<name>/<sub>_hemi-<H>_<name>.gii
       Several regions can share one texture with different paint values
       (e.g. 1 = STS, 2 = IPS; name them with --labels "1=STS,2=IPS").
    4. This script:
       a. VISUALISATION on the reference mesh: packs each session's anatomy
          (on the reference mesh), its resampled band maps and the painted region
          into one timelapse .npz per region, for spangy_viewer_hub.py.
       b. BACK TO THE ORIGINAL MESHES: moves the painted region from the reference
          mesh to every session's original mesh through the chained MSM spheres
          (reference sphere -> session sphere; area-weighted, so each original
          vertex gets the fraction of its area inside the region, then >= 0.5).
       c. RAW VALUES: reads the SPANGY outputs of the original meshes inside the
          region — no band value is ever interpolated.

The ROI is drawn on the reference mesh, whose vertices are the same for the sulc
and DPF-star chains; --texture only chooses which registrations carry it back.

Usage (cluster, after `export NEED_FREESURFER=0; source env_setup.sh`):
    $PYTHON propagate_roi.py --name STS --texture dpfstar
    $PYTHON propagate_roi.py --name sulci --labels "1=STS,2=IPS" --sub sub-Borgne
    $PYTHON propagate_roi.py --name STS --per-vertex      # also one row per vertex

Inputs per session (original mesh, from spangy_babofet_pymesh.py; each optional):
    <BASE>/<sub>/analysis/spangy_pymesh/textures/spangy_dom_band_<label>.gii
    <BASE>/<sub>/analysis/spangy_pymesh/textures/spangy_det_band_<label>.gii
    <BASE>/<sub>/analysis/spangy_pymesh/textures/spangy_frecomposed_<label>.gii
    <BASE>/<sub>/analysis/spangy_pymesh/curvature/mean/filt_mean_curv_<label>.gii
    <BASE>/<sub>/analysis/spangy_pymesh/curvature/principal/principal_curv_<label>.gii
and for the timelapse (reference mesh):
    <BASE>/surfaces_on_ref/<texture>/<sub>/hemi-<H>/<label>_on_<ref>.anat.surf.gii
    <BASE>/spangy_on_ref/<texture>/<sub>/hemi-<H>/<label>_on_<ref>.dom_band_from_bands.func.gii
    <BASE>/spangy_on_ref/<texture>/<sub>/hemi-<H>/<label>_on_<ref>.det_band_from_bands.func.gii

Outputs
    <BASE>/<sub>/rois/<name>/native_<texture>/
        <label>_<name>.frac.func.gii    per region: share of each vertex's area inside it
        <label>_<name>.label.func.gii   region number per vertex (0 = outside), for Anatomist
        <sub>_hemi-<H>_<name>_values.csv   one row per session and region
        per_vertex/<label>_<region>.csv.gz (with --per-vertex)
    <BASE>/analysis_pooled/roi_<name>_<texture>_native_values.csv   all animals
    <BASE>/spangy_timelapse/<sub>_hemi-<H>_roi_<texture>_<region>_on_<ref>.npz
        channel 0 = dominant band, channel 1 = determinant band (when available)

Band values for one region
    spangy_dom_band_*.gii holds only the winning band (a label). The values come from
    spangy_frecomposed_*.gii: per vertex, the curvature rebuilt band by band. The
    curvature carried by band b alone is f_b = Σ_{k in b} c_k φ_k (column b of
    diff(frecomposed), first column kept — the indexing of the dominant-band rule in
    resample_spangy_bands.py). The eigenvectors are orthonormal for the area-weighted
    product, so SPANGY's band power splits exactly over the surface:
        band power B_b  =  Σ_k∈b c_k²  =  ∫_hemisphere f_b² dA
    and the part of it inside a region is ∫_region f_b² dA. Per region and band:
        band_power_B<b>                ∫_region f_b² dA (same units as SPANGY's band power)
        band_power_share_of_hemi_B<b>  that region's share of the hemisphere's band power
        band_power_rel_B<b>            share of the region's own power (B1..Bn sum to 1)
        band_power_density_B<b>        per mm² of the region (compare regions of any size)
        band_mean_B<b>                 signed mean of f_b (negative: sulcal at that scale)
    check_power_vs_spangy_B<b> compares ∫_hemisphere f_b² dA with the band power in
    SPANGY's JSON: about 1 confirms the band indexing and normalisation.
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile

import numpy as np

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"


# --- io (same conventions as resample_spangy_bands.py) -----------------------------
def wb(*args):
    subprocess.run(["wb_command", *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def load_metric(path):
    import nibabel as nib
    d = nib.load(path).agg_data()
    d = np.column_stack(d) if isinstance(d, tuple) else np.asarray(d)
    return d.reshape(d.shape[0], -1).astype(np.float64)


def save_metric(values, path, struct):
    import nibabel as nib
    values = np.asarray(values, np.float32).reshape(len(values), -1)
    img = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(values[:, k], intent="NIFTI_INTENT_NONE",
                                 datatype="NIFTI_TYPE_FLOAT32") for k in range(values.shape[1])])
    nib.save(img, path)
    if struct:
        wb("-set-structure", path, struct)


def load_surf(path):
    import nibabel as nib
    g = nib.load(path)
    return (np.asarray(g.agg_data("pointset"), np.float64),
            np.asarray(g.agg_data("triangle"), np.int64))


def vertex_areas(xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(tri.ravel(), np.repeat(fa / 3.0, 3), minlength=len(xyz))


def optional_metric(path, n):
    """Metric as (n, C) array, or None if missing or of the wrong size (with a note)."""
    if not os.path.isfile(path):
        return None
    m = load_metric(path)
    if len(m) != n:
        print(f"    {os.path.basename(path)}: {len(m)} values, mesh has {n} -> ignored")
        return None
    return m


# --- regions -------------------------------------------------------------------------
def parse_labels(text):
    """'1=STS,2=IPS' -> {1.0: 'STS', 2.0: 'IPS'}"""
    out = {}
    for part in (text or "").split(","):
        if part.strip():
            k, v = part.split("=", 1)
            out[float(k)] = v.strip()
    return out


def roi_regions(values, labels, name):
    """Painted texture (V,) -> {region name: bool mask}, one per paint value."""
    v = np.round(np.asarray(values, float), 3)
    ids = [x for x in np.unique(v) if x != 0]
    if labels:
        extra = [x for x in ids if x not in labels]
        if extra:
            print(f"  note: paint values {extra} have no name in --labels and are ignored")
        return {lab: v == k for k, lab in labels.items() if np.any(v == k)}
    if len(ids) == 1:
        return {name: v == ids[0]}
    return {f"{name}_{x:g}": v == x for x in ids}


def band_tag(b):
    b = int(b)
    return "B0" if b == 0 else (f"sulc_B{-b}" if b < 0 else f"gyr_B{b}")


def dice(a, b, w):
    s = np.sum(w * a) + np.sum(w * b)
    return float(2 * np.sum(w * (a & b)) / s) if s > 0 else np.nan


def region_stats(w, data, n_bands):
    """Area-weighted summaries of the original-mesh SPANGY values inside one region."""
    tot = float(w.sum())
    row = {"roi_area_mm2": tot, "n_vertices_roi": int(np.count_nonzero(w))}
    if tot == 0:
        return row
    mean = lambda x: float(np.sum(w * x) / tot)                       # noqa: E731
    if data.get("curv") is not None:
        h = data["curv"]
        row.update(mean_H=mean(h), mean_abs_H=mean(np.abs(h)), mean_H2=mean(h ** 2),
                   share_sulcal_area=mean(h <= 0))
    if data.get("princ") is not None and data["princ"].shape[1] >= 2:
        row.update(mean_k1=mean(data["princ"][:, 0]), mean_k2=mean(data["princ"][:, 1]))
    for key in ("dom", "det"):
        if data.get(key) is not None:
            lab = np.rint(data[key]).astype(int)
            for b in range(-n_bands, n_bands + 1):
                if b == 0 and not np.any(lab == 0):
                    continue
                row[f"{key}_share_{band_tag(b)}"] = mean(lab == b)
    if data.get("frec") is not None:
        comp = band_components(data["frec"])
        energy = data["energy"]                      # per-vertex share of each band's power
        hemi_power = data.get("hemi_power")
        inside = w > 0
        power = energy[inside].sum(axis=0)           # ∫_region f_b² dA, exact split
        ptot = float(power.sum())
        for j in range(comp.shape[1]):
            b = j + 1
            row[f"band_power_B{b}"] = float(power[j])
            row[f"band_power_share_of_hemi_B{b}"] = (float(power[j] / hemi_power[j])
                                                     if hemi_power is not None and hemi_power[j] > 0
                                                     else np.nan)
            row[f"band_power_rel_B{b}"] = float(power[j] / ptot) if ptot > 0 else np.nan
            row[f"band_power_density_B{b}"] = float(power[j] / tot)   # per mm² of the region
            row[f"band_mean_B{b}"] = mean(comp[:, j])  # signed: <0 sulcal, >0 gyral at that scale
        row["folding_power_B1_Bn"] = ptot
    return row


def mass_matrix(xyz, tri):
    """Consistent P1 finite-element mass matrix (the inner product SPANGY's eigenvectors use)."""
    import scipy.sparse as sp
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
    rows = np.r_[i0, i1, i2, i0, i1, i1, i2, i2, i0]
    cols = np.r_[i0, i1, i2, i1, i0, i2, i1, i0, i2]
    vals = np.r_[fa / 6, fa / 6, fa / 6, fa / 12, fa / 12, fa / 12, fa / 12, fa / 12, fa / 12]
    n = len(xyz)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def band_energy(comp, M):
    """
    Per-vertex share of each band's power: e_b(i) = f_b(i) · (M f_b)(i).
    Summed over all vertices it is f_bᵀ M f_b = Σ_k∈b c_k², SPANGY's band power,
    so summing it over a region splits that power exactly between regions.
    """
    return comp * (M @ comp)


def band_components(frec):
    """
    frecomposed (cumulative synthesis, one column per band from B1) -> the curvature
    carried by each band alone, f_b = Σ_{k in band b} c_k φ_k (same indexing as the
    dominant-band rule in resample_spangy_bands.py).
    """
    return np.concatenate([frec[:, :1], np.diff(frec, axis=1)], axis=1)


def spangy_band_powers(json_path, n_bands):
    """Whole-hemisphere band powers B1..Bn saved by spangy_babofet_pymesh.py, or None."""
    import json
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as fh:
        d = json.load(fh)
    vals = [d.get(f"band_power_B{b}") for b in range(1, n_bands + 1)]
    return None if any(v is None for v in vals) else np.array(vals, float)


# --- main ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="ROI set name (folder and file name tag)")
    p.add_argument("--texture", default="sulc", help="registrations that carry it: sulc | dpfstar")
    p.add_argument("--sub", default="sub-*")
    p.add_argument("--base", default=BASE)
    p.add_argument("--roi-pattern", default="{base}/{sub}/rois/{name}/{sub}_hemi-{hemi}_{name}.gii",
                   help="painted texture on the reference mesh; fields {base} {sub} {hemi} {name}")
    p.add_argument("--labels", default="", help='paint value names, e.g. "1=STS,2=IPS"')
    p.add_argument("--threshold", type=float, default=0.5,
                   help="share of a vertex's area that must be inside the region")
    p.add_argument("--per-vertex", action="store_true", help="also write one CSV row per vertex")
    p.add_argument("--no-timelapse", action="store_true", help="skip the viewer .npz files")
    args = p.parse_args()
    base, tex, name = args.base, args.texture, args.name
    labels = parse_labels(args.labels)
    inputs = f"{base}/msm_inputs"
    done_csvs = []

    with tempfile.TemporaryDirectory() as tmp:
        for hdir in sorted(glob.glob(f"{base}/msm_chained/{tex}/{args.sub}/hemi-*")):
            sub, hemi = os.path.basename(os.path.dirname(hdir)), hdir[-1]
            struct = "CORTEX_LEFT" if hemi == "L" else "CORTEX_RIGHT"
            spheres = sorted(glob.glob(f"{hdir}/{sub}_hemi-{hemi}_ses-*_in_ses-*.sphere.surf.gii"))
            if not spheres:
                continue
            ref = re.search(r"_in_(ses-\d+)", spheres[0]).group(1)
            roi_path = args.roi_pattern.format(base=base, sub=sub, hemi=hemi, name=name)
            if not os.path.isfile(roi_path):
                print(f"[{sub} hemi-{hemi}] no painted ROI at {roi_path} -> skipped")
                continue
            ref_sphere = f"{inputs}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.sphere.surf.gii"
            ref_anat = f"{inputs}/{sub}/{ref}/{sub}_{ref}_hemi-{hemi}.anat.surf.gii"
            xyz_ref, tri_ref = load_surf(ref_anat)
            roi_vals = load_metric(roi_path)[:, 0]
            if len(roi_vals) != len(xyz_ref):
                print(f"[{sub} hemi-{hemi}] ROI has {len(roi_vals)} values but the reference mesh "
                      f"({ref}) has {len(xyz_ref)} vertices: paint it on a surfaces_on_ref "
                      f"surface or on the reference anatomy -> skipped")
                continue
            regions = roi_regions(roi_vals, labels, name)
            if not regions:
                print(f"[{sub} hemi-{hemi}] ROI texture is empty -> skipped")
                continue
            rnames = list(regions)
            ref_masks = np.column_stack([regions[r] for r in rnames])           # (V_ref, R)
            a_ref = vertex_areas(xyz_ref, tri_ref)
            print(f"[{sub} hemi-{hemi}] reference {ref}; regions: "
                  + ", ".join(f"{r} ({100 * a_ref[regions[r]].sum() / a_ref.sum():.1f}% of area)"
                              for r in rnames))
            f_roi_ref = f"{tmp}/roi_ref.func.gii"
            save_metric(ref_masks.astype(float), f_roi_ref, struct)

            ses_sph = {re.search(r"_(ses-\d+)_in_", s).group(1): s for s in spheres}
            ses_sph.setdefault(ref, None)
            odir = f"{base}/{sub}/rois/{name}/native_{tex}"
            os.makedirs(odir, exist_ok=True)
            sp = f"{base}/{sub}/analysis/spangy_pymesh"
            rows, tl = [], []

            for ses in sorted(ses_sph):
                sph = ses_sph[ses]
                label = f"{sub}_{ses}_hemi-{hemi}"
                anat = f"{inputs}/{sub}/{ses}/{label}.anat.surf.gii"
                if not os.path.isfile(anat):
                    print(f"  {ses}: no original mesh at {anat} -> skipped")
                    continue
                xyz_n, tri_n = load_surf(anat)
                a_n = vertex_areas(xyz_n, tri_n)

                # b. reference -> original mesh
                if ses == ref and len(xyz_n) == len(xyz_ref):
                    frac = ref_masks.astype(float)
                elif sph is None:
                    print(f"  {ses}: no registered sphere -> skipped")
                    continue
                else:
                    f_frac = f"{tmp}/frac.func.gii"
                    wb("-metric-resample", f_roi_ref, ref_sphere, sph, "ADAP_BARY_AREA", f_frac,
                       "-area-surfs", ref_anat, anat)
                    frac = np.clip(load_metric(f_frac), 0, 1)
                masks = frac >= args.threshold
                lab = np.where(masks.any(1), masks.argmax(1) + 1, 0)
                save_metric(frac, f"{odir}/{label}_{name}.frac.func.gii", struct)
                save_metric(lab, f"{odir}/{label}_{name}.label.func.gii", struct)

                # QC: carry the original-mesh mask back and compare with the painting
                if ses == ref and len(xyz_n) == len(xyz_ref):
                    back = ref_masks
                else:
                    f_b, f_bb = f"{tmp}/b.func.gii", f"{tmp}/bb.func.gii"
                    save_metric(masks.astype(float), f_b, struct)
                    wb("-metric-resample", f_b, sph, ref_sphere, "ADAP_BARY_AREA", f_bb,
                       "-area-surfs", anat, ref_anat)
                    back = load_metric(f_bb) >= args.threshold
                surf_onref = (f"{base}/surfaces_on_ref/{tex}/{sub}/hemi-{hemi}/"
                              f"{label}_on_{ref}.anat.surf.gii")
                a_on = None
                if os.path.isfile(surf_onref):
                    xo, to = load_surf(surf_onref)
                    if len(xo) == len(xyz_ref):
                        a_on = vertex_areas(xo, to)

                # c. raw SPANGY values on the original mesh
                n = len(xyz_n)
                data = {
                    "dom": optional_metric(f"{sp}/textures/spangy_dom_band_{label}.gii", n),
                    "det": optional_metric(f"{sp}/textures/spangy_det_band_{label}.gii", n),
                    "frec": optional_metric(f"{sp}/textures/spangy_frecomposed_{label}.gii", n),
                    "curv": optional_metric(f"{sp}/curvature/mean/filt_mean_curv_{label}.gii", n),
                    "princ": optional_metric(
                        f"{sp}/curvature/principal/principal_curv_{label}.gii", n),
                }
                for k in ("dom", "det", "curv"):
                    if data[k] is not None:
                        data[k] = data[k][:, 0]
                n_bands = data["frec"].shape[1] if data["frec"] is not None else 6
                if not any(v is not None for v in data.values()):
                    print(f"  {ses}: no SPANGY outputs on the original mesh (run "
                          f"spangy_babofet_pymesh.py) -> region moved, no values")
                check = {}
                if data["frec"] is not None:
                    # whole-hemisphere power of each band, from the same per-vertex values;
                    # by Parseval it must equal SPANGY's band power (checked against the JSON)
                    comp_all = band_components(data["frec"])
                    data["energy"] = band_energy(comp_all, mass_matrix(xyz_n, tri_n))
                    data["hemi_power"] = data["energy"].sum(axis=0)
                    saved = spangy_band_powers(f"{sp}/results/per_file/{label}.json", n_bands)
                    if saved is not None:
                        ratio = data["hemi_power"] / np.where(saved > 0, saved, np.nan)
                        check = {f"check_power_vs_spangy_B{b + 1}": float(ratio[b])
                                 for b in range(n_bands)}
                        bad = [b + 1 for b in range(n_bands)
                               if not np.isnan(ratio[b]) and abs(ratio[b] - 1) > 0.05]
                        if bad:
                            print(f"    note: per-vertex band power differs from SPANGY's by >5% "
                                  f"for B{bad} (median ratio {np.nanmedian(ratio):.3f}) — see "
                                  f"check_power_vs_spangy_* columns")
                else:
                    print(f"  {ses}: no spangy_frecomposed texture -> no band powers for this "
                          f"session (rerun spangy_babofet_pymesh.py with frecomposed saved)")

                for k, r in enumerate(rnames):
                    m = masks[:, k]
                    row = {"subject": sub, "hemi": hemi, "session": ses, "reference": ref,
                           "region": r, "registration": tex, "n_vertices_session": n,
                           "hemi_area_mm2": float(a_n.sum()),
                           "roi_area_frac_weighted_mm2": float(np.sum(a_n * frac[:, k])),
                           "roundtrip_dice": dice(back[:, k], ref_masks[:, k], a_ref)}
                    if a_on is not None:
                        row["roi_area_on_ref_surface_mm2"] = float(a_on[ref_masks[:, k]].sum())
                    row.update(region_stats(a_n * m, data, n_bands))
                    row.update(check)
                    row["roi_area_share_of_hemi"] = row["roi_area_mm2"] / row["hemi_area_mm2"]
                    if a_on is not None and row["roi_area_on_ref_surface_mm2"] > 0:
                        row["area_ratio_native_vs_on_ref"] = (row["roi_area_mm2"]
                                                              / row["roi_area_on_ref_surface_mm2"])
                    if row["roi_area_mm2"] == 0:
                        print(f"  {ses}: region {r} is empty on the original mesh")
                    rows.append(row)

                    if args.per_vertex and row["roi_area_mm2"] > 0:
                        import pandas as pd
                        idx = np.where(m)[0]
                        pv = {"vertex": idx, "area_mm2": a_n[idx], "frac_inside": frac[idx, k]}
                        for key in ("curv", "dom", "det"):
                            if data[key] is not None:
                                pv[{"curv": "H"}.get(key, key + "_band")] = data[key][idx]
                        if data["princ"] is not None:
                            pv["k1"], pv["k2"] = data["princ"][idx, 0], data["princ"][idx, 1]
                        if data["frec"] is not None:
                            comp = band_components(data["frec"])
                            for j in range(comp.shape[1]):
                                pv[f"f_B{j + 1}"] = comp[idx, j]     # curvature in band j
                                pv[f"power_B{j + 1}"] = data["energy"][idx, j]  # sums to band_power_B
                        os.makedirs(f"{odir}/per_vertex", exist_ok=True)
                        pd.DataFrame(pv).to_csv(f"{odir}/per_vertex/{label}_{r}.csv.gz",
                                                index=False)

                msg = "; ".join(f"{r} {rows[-len(rnames) + k]['roi_area_mm2']:.0f} mm² "
                                f"(Dice back {rows[-len(rnames) + k]['roundtrip_dice']:.3f})"
                                for k, r in enumerate(rnames))
                print(f"  {ses}: {n} vertices; {msg}")

                # a. timelapse inputs on the reference mesh
                if not args.no_timelapse:
                    on = f"{base}/spangy_on_ref/{tex}/{sub}/hemi-{hemi}/{label}_on_{ref}"
                    f_dom, f_det = f"{on}.dom_band_from_bands.func.gii", f"{on}.det_band_from_bands.func.gii"
                    if os.path.isfile(surf_onref) and os.path.isfile(f_dom):
                        xo = load_surf(surf_onref)[0]
                        dom = load_metric(f_dom)[:, 0]
                        det = load_metric(f_det)[:, 0] if os.path.isfile(f_det) else None
                        if len(xo) == len(dom) == len(xyz_ref):
                            tl.append((ses, xo, dom, det))
                    else:
                        print(f"    (timelapse: {os.path.basename(surf_onref)} or "
                              f"{os.path.basename(f_dom)} missing -> session left out)")

            if rows:
                import pandas as pd
                out_csv = f"{odir}/{sub}_hemi-{hemi}_{name}_values.csv"
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                done_csvs.append(out_csv)
                print(f"  -> {out_csv}")

            if not args.no_timelapse and len(tl) >= 2:
                has_det = all(t[3] is not None for t in tl)
                values = np.stack([np.column_stack([t[2], t[3]]) if has_det else t[2][:, None]
                                   for t in tl]).astype(np.float32)
                coords = np.stack([t[1] for t in tl]).astype(np.float32)
                os.makedirs(f"{base}/spangy_timelapse", exist_ok=True)
                for k, r in enumerate(rnames):
                    out = f"{base}/spangy_timelapse/{sub}_hemi-{hemi}_roi_{tex}_{r}_on_{ref}.npz"
                    np.savez_compressed(
                        out, faces=tri_ref.astype(np.int32), coords=coords, values=values,
                        roi=np.repeat(ref_masks[None, :, k], len(tl), axis=0).astype(np.uint8),
                        sessions=np.array([t[0] for t in tl]), ref=ref, sub=sub, hemi=hemi,
                        kind="label", sulcus=r,
                        channels=np.array(["dominant band", "determinant band"][:values.shape[2]]))
                    print(f"  timelapse -> {out}")
            elif not args.no_timelapse:
                print("  timelapse: fewer than 2 sessions with resampled bands (run "
                      "resample_to_ref.py and resample_spangy_bands.py) -> not written")

    # pooled table over every animal processed so far with this name and registration
    import pandas as pd
    allcsv = sorted(glob.glob(f"{base}/sub-*/rois/{name}/native_{tex}/*_hemi-*_{name}_values.csv"))
    if allcsv:
        os.makedirs(f"{base}/analysis_pooled", exist_ok=True)
        pooled = pd.concat([pd.read_csv(f) for f in allcsv], ignore_index=True)
        out = f"{base}/analysis_pooled/roi_{name}_{tex}_native_values.csv"
        pooled.to_csv(out, index=False)
        print(f"\nPooled: {len(pooled)} rows from {pooled['subject'].nunique()} animal(s) -> {out}")
        q = pooled["roundtrip_dice"].describe()
        print(f"Round-trip Dice: min {q['min']:.3f}, median {q['50%']:.3f} "
              "(below ~0.9: check the registration for that session)")
    elif not done_csvs:
        print("Nothing done: check --name, the ROI path and that chained spheres exist.")


if __name__ == "__main__":
    main()
