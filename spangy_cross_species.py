#!/usr/bin/env python3
"""
spangy_cross_species.py
=======================

Run SPANGY on two directories of cortical meshes (one baboon, one human) and
compare the spectral bands across species by SPATIAL WAVELENGTH and FREQUENCY.

Per-mesh steps mirror the BaboFet pipeline:
    load mesh -> align to inertia axes -> mean curvature 0.5*(k1+k2),
    z-score filtered -> spgy.eigenpairs -> spgy.spectrum -> per-band wavelengths.

Wavelength & frequency (Germanaud & Lefevre 2012, Eq. 10):
    frequency   F  = sqrt(lambda) / (2*pi)      [cycles per mm]
    wavelength  WL = 2*pi / sqrt(lambda)         [mm]
Both use the square root of the eigenvalue alone; they are reciprocals.

Comparisons written to CSV:
    RELATIVE  - band B_k baboon vs band B_k human (same step on each brain's own
                ladder; NOT the same physical size, since bands are anchored to
                each brain's fundamental lambda_1).
    PHYSICAL  - bands whose millimetre ranges overlap (usually pairs a baboon
                band with a lower-numbered human band).

Notes:
    * Use the same nb_eig for both species so both get the same band count.
    * Compare composition with normalized power (power_fraction), not raw power.
    * Meshes are assumed to be in millimetres.
"""

import os
import glob
import numpy as np
import pandas as pd

import slam.io as sio
import slam.spangy as spgy
import slam.curvature as scurv
import slam.texture as stex

# ----------------------------------------------------------------------------
# CONFIG  -- edit this block, then run:  python spangy_cross_species.py
# ----------------------------------------------------------------------------
CONFIG = {
    "baboon_dir":  "/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/all_mesh_combined",
    "human_dir":   "/path/to/human_meshes",
    "baboon_glob": "*.surf.gii",
    "human_glob":  "*.surf.gii",

    "nb_eig":              5000,   # identical for both species; clamped to n_vert-2
    "apply_inertia_transform": True,
    "curv_zscore_thresh":  3,      # TextureND.z_score_filtering threshold
    "curvature_sign":      1.0,    # SPANGY expects sulci < 0; flip to -1 if inverted

    "out_dir":     "./spangy_cross_species_out",
}

TINY = 1e-12  # guard for the near-zero constant eigenvalue


# ----------------------------------------------------------------------------
# filename parsing
# ----------------------------------------------------------------------------
def strip_mesh_ext(fn):
    for ext in (".surf.gii", ".gii"):
        if fn.endswith(ext):
            return fn[: -len(ext)]
    return os.path.splitext(fn)[0]


def parse_labels(mesh_path):
    """Return (subject_label, hemisphere) from a filename."""
    fn = os.path.basename(mesh_path)
    stem = strip_mesh_ext(fn)
    if "hemi-L" in fn or "_left" in fn or fn.endswith("lh.surf.gii"):
        hemi = "L"
    elif "hemi-R" in fn or "_right" in fn or fn.endswith("rh.surf.gii"):
        hemi = "R"
    else:
        hemi = "?"
    return stem, hemi


# ----------------------------------------------------------------------------
# mesh + curvature (matches BaboFet convention)
# ----------------------------------------------------------------------------
def load_mesh(mesh_path):
    mesh = sio.load_mesh(mesh_path)
    if CONFIG["apply_inertia_transform"]:
        mesh.apply_transform(mesh.principal_inertia_transform)
    return mesh


def mean_curvature(mesh):
    """0.5*(k1 + k2), then z-score filtered -> the curvature spectrum() decomposes."""
    p_curv, _d1, _d2 = scurv.curvatures_and_derivatives(mesh)   # shape (2, n_vert)
    mc = 0.5 * (p_curv[0, :] + p_curv[1, :])
    tex = stex.TextureND(mc)
    tex.z_score_filtering(z_thresh=CONFIG["curv_zscore_thresh"])   # in-place
    filt = np.asarray(tex.darray, dtype=float).squeeze()
    return CONFIG["curvature_sign"] * filt


# ----------------------------------------------------------------------------
# per-band wavelength / frequency / power table
# ----------------------------------------------------------------------------
def band_table(eig_val, group_indices, coefficients):
    """One row per spectral band. group_indices[b] is band B_b;
    frecomposed[:, b-1] is the same band (B0 constant is skipped there)."""
    total_power = float(np.sum(coefficients ** 2))
    rows = []
    for b in range(group_indices.shape[0]):
        lo, hi = int(group_indices[b, 0]), int(group_indices[b, 1])
        if b == 0:
            rows.append({"band": 0, "note": "constant / mean - no wavelength",
                         "power": float(coefficients[0] ** 2),
                         "power_fraction": float(coefficients[0] ** 2 / total_power)})
            continue

        lam = np.clip(np.asarray(eig_val[lo:hi + 1], dtype=float), TINY, None)
        c2 = np.asarray(coefficients[lo:hi + 1], dtype=float) ** 2
        lam_lo, lam_hi = lam[0], lam[-1]

        # low eigenvalue -> long wavelength; high eigenvalue -> short wavelength
        WL_long = 2 * np.pi / np.sqrt(lam_lo)
        WL_short = 2 * np.pi / np.sqrt(lam_hi)

        # energy-weighted centre: where the folding power actually sits in-band
        if c2.sum() > 0:
            F_centre = float(np.sum(c2 * np.sqrt(lam)) / c2.sum()) / (2 * np.pi)
        else:
            F_centre = (lam_lo * lam_hi) ** 0.25 / (2 * np.pi)   # geometric centre
        WL_centre = 1.0 / F_centre if F_centre > 0 else np.nan

        rows.append({
            "band": b,
            "lambda_lo": float(lam_lo),
            "lambda_hi": float(lam_hi),
            "WL_short_mm": float(WL_short),
            "WL_long_mm": float(WL_long),
            "WL_centre_mm": float(WL_centre),
            "F_low_cpm": float(np.sqrt(lam_lo) / (2 * np.pi)),
            "F_high_cpm": float(np.sqrt(lam_hi) / (2 * np.pi)),
            "F_centre_cpm": float(F_centre),
            "power": float(c2.sum()),
            "power_fraction": float(c2.sum() / total_power),
        })
    return rows


# ----------------------------------------------------------------------------
# run SPANGY over a directory
# ----------------------------------------------------------------------------
def run_one_mesh(mesh_path, species, nb_eig):
    subj, hemi = parse_labels(mesh_path)
    mesh = load_mesh(mesh_path)

    n_vert = len(mesh.vertices)
    ne = min(nb_eig, n_vert - 2)
    if ne < nb_eig:
        print(f"    [warn] {subj}: nb_eig {nb_eig} -> {ne} ({n_vert} vertices)")

    curv = mean_curvature(mesh)
    eig_val, eig_vec, lap_b = spgy.eigenpairs(mesh, ne)
    _grouped, group_indices, coefficients, nlevels = spgy.spectrum(
        curv, lap_b, eig_vec, eig_val
    )

    rows = band_table(eig_val, group_indices, coefficients)
    for r in rows:
        r.update(species=species, subject=subj, hemisphere=hemi, nlevels=int(nlevels))
    return rows


def run_directory(directory, pattern, species, nb_eig):
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No meshes in {directory} matching {pattern}")
    print(f"{species}: {len(paths)} mesh(es)")
    rows = []
    for p in paths:
        print(f"  - {os.path.basename(p)}")
        rows.extend(run_one_mesh(p, species, nb_eig))
    return rows


# ----------------------------------------------------------------------------
# aggregation + comparison
# ----------------------------------------------------------------------------
def species_summary(df):
    num = ["WL_short_mm", "WL_long_mm", "WL_centre_mm", "F_centre_cpm", "power_fraction"]
    g = df[df["band"] > 0].groupby(["species", "band"])[num]
    summ = g.agg(["mean", "std"]).reset_index()
    summ.columns = ["species", "band"] + [f"{c}_{s}" for c in num for s in ("mean", "std")]
    return summ


def relative_comparison(summ):
    keep = ["band", "WL_centre_mm_mean", "power_fraction_mean"]
    bab = summ[summ.species == "baboon"][keep].add_prefix("baboon_").rename(
        columns={"baboon_band": "band"})
    hum = summ[summ.species == "human"][keep].add_prefix("human_").rename(
        columns={"human_band": "band"})
    return pd.merge(bab, hum, on="band", how="outer").sort_values("band")


def overlaps(r1, r2):
    return max(r1[0], r2[0]) < min(r1[1], r2[1])


def physical_comparison(summ):
    bab = summ[summ.species == "baboon"]
    hum = summ[summ.species == "human"]
    out = []
    for _, br in bab.iterrows():
        b_rng = (br["WL_short_mm_mean"], br["WL_long_mm_mean"])
        for _, hr in hum.iterrows():
            h_rng = (hr["WL_short_mm_mean"], hr["WL_long_mm_mean"])
            if overlaps(b_rng, h_rng):
                lo, hi = max(b_rng[0], h_rng[0]), min(b_rng[1], h_rng[1])
                out.append({
                    "baboon_band": int(br["band"]),
                    "baboon_WL_mm": f"{b_rng[0]:.1f}-{b_rng[1]:.1f}",
                    "human_band": int(hr["band"]),
                    "human_WL_mm": f"{h_rng[0]:.1f}-{h_rng[1]:.1f}",
                    "overlap_mm": round(hi - lo, 2),
                })
    return pd.DataFrame(out).sort_values(["baboon_band", "human_band"])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    nb_eig = CONFIG["nb_eig"]

    rows = []
    rows += run_directory(CONFIG["baboon_dir"], CONFIG["baboon_glob"], "baboon", nb_eig)
    rows += run_directory(CONFIG["human_dir"], CONFIG["human_glob"], "human", nb_eig)
    df = pd.DataFrame(rows)

    nb = df.groupby("species")["nlevels"].unique().to_dict()
    print("\nband counts (nlevels) per species:", nb)
    if len({int(x) for v in nb.values() for x in v}) > 1:
        print("  [warn] band counts differ; relative compare only spans shared bands.")

    df.to_csv(os.path.join(CONFIG["out_dir"], "per_subject_bands.csv"), index=False)
    summ = species_summary(df)
    summ.to_csv(os.path.join(CONFIG["out_dir"], "species_band_summary.csv"), index=False)
    rel = relative_comparison(summ)
    rel.to_csv(os.path.join(CONFIG["out_dir"], "comparison_relative.csv"), index=False)
    phys = physical_comparison(summ)
    phys.to_csv(os.path.join(CONFIG["out_dir"], "comparison_physical.csv"), index=False)

    pd.set_option("display.width", 120)
    print("\n=== RELATIVE (B_k baboon vs human; wavelength mm, power normalized) ===")
    print(rel.to_string(index=False))
    print("\n=== PHYSICAL (bands whose millimetre ranges overlap) ===")
    print(phys.to_string(index=False))
    print(f"\nCSV tables written to: {CONFIG['out_dir']}")


if __name__ == "__main__":
    main()