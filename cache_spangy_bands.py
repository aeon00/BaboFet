#!/usr/bin/env python3
"""
recreate_spangy_bands.py
------------------------
Visualise how the SPANGY frequency bands B0..B6 are arranged on a cortical
surface mesh, and recreate the "Eigenfunctions Basis / Band Design" figure.

The script is split in two stages:

  1. COMPUTE (expensive: eigenpairs + curvature + spectrum) runs once and caches
     only the render-ready arrays to a small .npz file. Needs slam.
  2. RENDER (cheap) reloads that cache and draws / saves every figure. Needs only
     pyvista + matplotlib, not slam or the original mesh.

The MODE switch (below) selects which stage runs:
  MODE = "compute"  -> pipeline only, writes the cache        (run on the cluster)
  MODE = "render"   -> figures only, reads the cache          (run locally)
  MODE = "both"     -> compute then render in one process
Typical split: MODE="compute" on the HPC node, transfer the .npz, then
MODE="render" locally. To retune the camera/colours, just rerun MODE="render".

Outputs written into OUT_DIR:
  spectrum_band_design.png   eigenvalue-vs-order curve + x2 boundaries + thumbnails
  eigenfunctions_basis.png   one representative eigenmode per band
  band_recomposition.png     mean-curvature signal recomposed per band B0..B6
  dominant_band_map.png      locally dominant band segmentation
  panels/...                 every individual panel

Dependencies: compute -> slam, numpy, scipy, trimesh ;  render -> pyvista, matplotlib
When rendering on a headless node, set USE_XVFB=True or run under `xvfb-run -a`.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import pyvista as pv

# ============================================================================ #
#  PATHS  — edit these                                                         #
# ============================================================================ #
MESH_PATH = ("/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/"
             "all_mesh_combined/sub-Borgne_ses-01_hemi-L_white.surf.gii")

OUT_DIR   = ("/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/"
             "analysis/spangy/band_figures")

CACHE_PATH = os.path.join(OUT_DIR, "spangy_render_cache.npz")

# ============================================================================ #
#  BEHAVIOUR                                                                   #
# ============================================================================ #
MODE = "compute"     # "compute" -> run the pipeline and write the cache only
                     #              (no rendering; needs slam; use on the cluster)
                     # "render"  -> load the cache and draw the figures only
                     #              (no slam; use locally after transferring the .npz)
                     # "both"    -> compute then render in one go

USE_XVFB  = False    # start a virtual framebuffer for headless GL rendering.
                     # Only matters when MODE renders. Leave False if you render
                     # locally with a display, or launch via `xvfb-run`.
N_EIG     = 4000     # number of eigenpairs (matches your run script)

# ============================================================================ #
#  RENDER SETTINGS                                                             #
# ============================================================================ #
WINDOW    = (720, 640)     # per-panel render size (px)
CMAP_FUNC = "RdBu_r"       # diverging map for signed curvature-like signals
CMAP_DOM  = "coolwarm"     # map for the signed dominant-band segmentation
CLIM_PCT  = 99.0           # percentile for symmetric colour limits

# Camera. After principal_inertia_transform the mesh is axis-aligned but axis
# signs are arbitrary, so tweak these once for a clean lateral view. Applied
# identically to every panel, so all bands stay in register.
CAM_VIEW      = "xz"       # base pyvista view plane: 'xy' | 'xz' | 'yz'
CAM_AZIMUTH   = 0.0        # extra rotation (deg) about the view-up axis
CAM_ELEVATION = 0.0        # extra rotation (deg) about the horizontal axis
CAM_ZOOM      = 1.35
BG_COLOR      = "white"


# ============================================================================ #
#  STAGE 1 — COMPUTE  (heavy; cached)                                          #
# ============================================================================ #
def compute(mesh_path, cache_path):
    """Run the SPANGY pipeline and cache only the arrays needed for rendering."""
    # imported here so the render-only path never needs slam
    import slam.io as sio
    import slam.texture as stex
    import slam.curvature as scurv
    import slam.spangy as spgy

    print(f"[compute] {os.path.basename(mesh_path)}")
    mesh = sio.load_mesh(mesh_path)
    mesh.apply_transform(mesh.principal_inertia_transform)

    print("  eigenpairs ...")
    eig_val, eig_vec, lap_b = spgy.eigenpairs(mesh, N_EIG)

    print("  curvature ...")
    principal_curv, _, _ = scurv.curvatures_and_derivatives(mesh)
    mean_curv = 0.5 * (principal_curv[0, :] + principal_curv[1, :])
    tex_mc = stex.TextureND(mean_curv)
    tex_mc.z_score_filtering(z_thresh=3)
    filt_mean_curv = tex_mc.darray.squeeze()

    print("  spectrum ...")
    grouped_spectrum, group_indices, coefficients, nlevels = \
        spgy.spectrum(filt_mean_curv, lap_b, eig_vec, eig_val)

    print("  band recomposition ...")
    loc_dom_band, frecomposed = spgy.local_dominance_map(
        coefficients, filt_mean_curv, nlevels, group_indices, eig_vec
    )

    # representative eigenmode per band = upper (highest-freq) boundary of band
    rep_idx = np.array(
        [min(int(group_indices[k, 1]) if group_indices[k, 1] > 0 else 1, N_EIG - 1)
         for k in range(nlevels)],
        dtype=int,
    )
    rep_modes = np.asarray(eig_vec[:, rep_idx], dtype=float)          # (V, nlevels)

    # band maps on the mesh: B0 = DC/global, B1..B6 = frecomposed columns
    band_maps = np.column_stack(
        [coefficients[0] * np.asarray(eig_vec[:, 0], dtype=float)] +
        [np.asarray(frecomposed[:, i], dtype=float) for i in range(frecomposed.shape[1])]
    )                                                                # (V, nlevels)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        eig_val=np.asarray(eig_val, dtype=float),
        group_indices=np.asarray(group_indices, dtype=int),
        nlevels=np.int64(nlevels),
        rep_idx=rep_idx,
        rep_modes=rep_modes,
        band_maps=band_maps,
        loc_dom_band=np.asarray(loc_dom_band, dtype=float),
        grouped_spectrum=np.asarray(grouped_spectrum, dtype=float),
    )
    print(f"  cached render arrays -> {cache_path}")
    return load_cache(cache_path)


def load_cache(cache_path):
    z = np.load(cache_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ============================================================================ #
#  RENDER HELPERS                                                              #
# ============================================================================ #
def to_polydata(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    f = np.empty((faces.shape[0], 4), dtype=np.int64)
    f[:, 0] = 3
    f[:, 1:] = faces
    return pv.PolyData(np.asarray(vertices, dtype=float), f.ravel())


def sym_clim(values, pct=CLIM_PCT):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    m = float(np.percentile(np.abs(v), pct))
    if m <= 0:
        m = float(np.max(np.abs(v))) or 1.0
    return (-m, m)


def snapshot(poly, scalars, cmap, clim):
    p = pv.Plotter(off_screen=True, window_size=list(WINDOW))
    p.background_color = BG_COLOR
    p.add_mesh(poly, scalars=np.asarray(scalars, dtype=float).ravel(),
               cmap=cmap, clim=clim, show_scalar_bar=False,
               smooth_shading=True, interpolate_before_map=True)
    getattr(p, f"view_{CAM_VIEW}")()
    if CAM_AZIMUTH:
        p.camera.azimuth = CAM_AZIMUTH
    if CAM_ELEVATION:
        p.camera.elevation = CAM_ELEVATION
    p.camera.zoom(CAM_ZOOM)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def montage(images, labels, out_png, title=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, lab in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}")


def spectrum_band_design(eig_val, group_indices, thumbs, thumb_orders, out_png):
    orders = np.arange(len(eig_val))
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.plot(orders, eig_val, color="0.25", lw=1.6, zorder=1)
    ax.set_xlabel("Eigenvalue order", fontsize=12)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ymax = float(np.max(eig_val))
    for k in range(len(group_indices)):
        lo, hi = int(group_indices[k, 0]), int(group_indices[k, 1])
        ax.axvline(hi, color="0.75", ls="--", lw=0.8, zorder=0)
        ax.text(0.5 * (lo + hi), ymax * 0.015, f"B{k}", ha="center", va="bottom",
                fontsize=11, color="tab:green", fontweight="bold")
    for img, order in zip(thumbs, thumb_orders):
        order = int(min(order, len(eig_val) - 1))
        ab = AnnotationBbox(OffsetImage(img, zoom=0.16),
                            (order, eig_val[order]), frameon=False,
                            box_alignment=(0.5, 0.0),
                            xybox=(0, 42), boxcoords="offset points", zorder=3)
        ax.add_artist(ab)
    ax.set_title("SPANGY band design: eigenvalue spectrum with x2 band boundaries "
                 "and one representative eigenmode per band", fontsize=12)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}")


# ============================================================================ #
#  STAGE 2 — RENDER  (cheap; from cache)                                       #
# ============================================================================ #
def render(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    poly          = to_polydata(data["vertices"], data["faces"])
    eig_val       = data["eig_val"]
    group_indices = data["group_indices"]
    nlevels       = int(data["nlevels"])
    rep_idx       = data["rep_idx"]
    rep_modes     = data["rep_modes"]
    band_maps     = data["band_maps"]
    loc_dom_band  = data["loc_dom_band"]

    # 1) eigenfunctions basis -------------------------------------------------
    eig_imgs, eig_labels = [], []
    for k in range(nlevels):
        scal = rep_modes[:, k]
        img = snapshot(poly, scal, CMAP_FUNC, sym_clim(scal))
        eig_imgs.append(img)
        eig_labels.append(f"B{k}   (mode {int(rep_idx[k])})")
        plt.imsave(os.path.join(panels_dir, f"eigfun_B{k}_mode{int(rep_idx[k])}.png"), img)
    montage(eig_imgs, eig_labels, os.path.join(out_dir, "eigenfunctions_basis.png"),
            title="SPANGY eigenfunctions basis — one representative mode per band")

    # 2) band recomposition on the mesh --------------------------------------
    band_imgs, band_labels = [], []
    for k in range(band_maps.shape[1]):
        bm = band_maps[:, k]
        img = snapshot(poly, bm, CMAP_FUNC, sym_clim(bm))
        band_imgs.append(img)
        band_labels.append(f"B{k}")
        plt.imsave(os.path.join(panels_dir, f"band_B{k}.png"), img)
    montage(band_imgs, band_labels, os.path.join(out_dir, "band_recomposition.png"),
            title="Mean-curvature signal recomposed per SPANGY band (B0-B6)")

    # 3) dominant band segmentation ------------------------------------------
    vmax = float(np.nanmax(np.abs(loc_dom_band))) or 1.0
    dom_img = snapshot(poly, loc_dom_band, CMAP_DOM, (-vmax, vmax))
    plt.imsave(os.path.join(panels_dir, "dominant_band_raw.png"), dom_img)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(dom_img); ax.axis("off")
    ax.set_title("Locally dominant SPANGY band", fontsize=13, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=CMAP_DOM, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04
                 ).set_label("signed dominant band   (- sulci  /  +USE_XVFB  = True    gyri)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dominant_band_map.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, 'dominant_band_map.png')}")

    # 4) reference-style composite -------------------------------------------
    spectrum_band_design(eig_val, group_indices, eig_imgs, rep_idx,
                         os.path.join(out_dir, "spectrum_band_design.png"))


# ============================================================================ #
def main():
    if MODE not in ("compute", "render", "both"):
        raise ValueError(f"MODE must be 'compute', 'render' or 'both' (got {MODE!r})")

    # --- compute stage (cluster): heavy, writes the cache, no rendering ------
    if MODE in ("compute", "both"):
        data = compute(MESH_PATH, CACHE_PATH)

    # --- render stage (local): cheap, reads the cache, draws the figures -----
    if MODE in ("render", "both"):
        if USE_XVFB:
            try:
                pv.start_xvfb()
            except Exception as exc:
                print(f"[xvfb] not started ({exc}); assuming a usable GL context")
        pv.OFF_SCREEN = True

        if MODE == "render":
            if not os.path.exists(CACHE_PATH):
                raise FileNotFoundError(
                    f"cache not found: {CACHE_PATH}\n"
                    "run MODE='compute' first, then transfer the .npz here.")
            print(f"[render] loading cache {CACHE_PATH}")
            data = load_cache(CACHE_PATH)
        render(data, OUT_DIR)

    print("done.")


if __name__ == "__main__":
    main()