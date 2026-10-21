#!/usr/bin/env python3
"""
qc_msm_inputs.py — sanity checks on MSM input files for one hemisphere.

Checks:
  * anat / sphere / metrics all have the same number of vertices
  * sphere is centred at the origin with radius 100 (what MSM expects)
  * number of flipped (inside-out) triangles on the sphere
    -> any flips mean the spherical mapping folded over; MSM will misbehave
  * value ranges of sulc / curvature / DPF-star, NaN check
  * correlation between sulc and DPF-star (sign convention check)
  * areal distortion summary (log2 ratio sphere/anat, from wb_command)

Exit code 1 if a hard failure is found (vertex mismatch, NaNs, flips above
--max-flips, radius off), so it can gate a SLURM pipeline.
"""
import argparse
import sys

import nibabel as nib
import numpy as np


def load_surf(path):
    g = nib.load(path)
    coords = np.asarray(g.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(g.agg_data("triangle"), dtype=np.int64)
    return coords, faces


def load_metric(path):
    data = nib.load(path).agg_data()
    if isinstance(data, tuple):  # multi-column metric -> first column
        data = data[0]
    return np.asarray(data, dtype=np.float64).ravel()


def flipped_triangles(coords, faces):
    """Triangles whose normal points towards the sphere centre."""
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    signs = np.einsum("ij,ij->i", normals, centroids)
    # The majority orientation defines "outward" (handles either winding order)
    majority = np.sign(np.median(signs))
    return int(np.sum(np.sign(signs) == -majority))


def describe(name, x):
    return (f"{name:<12} min={x.min():9.4f}  max={x.max():9.4f}  "
            f"mean={x.mean():9.4f}  std={x.std():9.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anat", required=True)
    p.add_argument("--sphere", required=True)
    p.add_argument("--sulc", required=True)
    p.add_argument("--curv")
    p.add_argument("--distortion")
    p.add_argument("--dpfstar")
    p.add_argument("--max-flips", type=int, default=0)
    args = p.parse_args()

    failures = []

    anat_xyz, anat_tri = load_surf(args.anat)
    sph_xyz, sph_tri = load_surf(args.sphere)
    n = anat_xyz.shape[0]
    print(f"vertices     anat={n}  sphere={sph_xyz.shape[0]}  faces={anat_tri.shape[0]}")

    if sph_xyz.shape[0] != n:
        failures.append("anat and sphere vertex counts differ")
    if anat_tri.shape != sph_tri.shape or not np.array_equal(anat_tri, sph_tri):
        failures.append("anat and sphere triangles differ (vertex order not preserved?)")

    # Euler characteristic: closed genus-0 surface -> 2
    edges = np.sort(np.vstack([anat_tri[:, [0, 1]], anat_tri[:, [1, 2]],
                               anat_tri[:, [2, 0]]]), axis=1)
    n_edges = np.unique(edges, axis=0).shape[0]
    euler = n - n_edges + anat_tri.shape[0]
    print(f"euler char.  {euler}  (2 = closed genus-0)")
    if euler != 2:
        failures.append(f"surface is not genus-0 (Euler = {euler})")

    centre = sph_xyz.mean(axis=0)
    radius = np.linalg.norm(sph_xyz - centre, axis=1)
    print(f"sphere       centre=({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f})  "
          f"radius min={radius.min():.3f} max={radius.max():.3f}")
    if np.linalg.norm(centre) > 1.0 or abs(radius.mean() - 100) > 1.0:
        failures.append("sphere not centred at origin with radius 100")

    n_flip = flipped_triangles(sph_xyz, sph_tri)
    print(f"flipped tri. {n_flip} ({100 * n_flip / sph_tri.shape[0]:.4f} %)")
    if n_flip > args.max_flips:
        failures.append(f"{n_flip} flipped triangles on sphere")

    metrics = {}
    for name, path in (("sulc", args.sulc), ("curv", args.curv),
                       ("dpfstar", args.dpfstar), ("distortion", args.distortion)):
        if not path:
            continue
        x = load_metric(path)
        if x.shape[0] != n:
            failures.append(f"{name} has {x.shape[0]} values, expected {n}")
            continue
        if not np.all(np.isfinite(x)):
            failures.append(f"{name} contains NaN/Inf")
        else:
            metrics[name] = x
        x = x[np.isfinite(x)]
        print(describe(name, x))
        if name == "distortion":
            print(f"{'':<12} |log2 distortion| > 2 at "
                  f"{100 * np.mean(np.abs(x) > 2):.2f} % of vertices")

    # Sign check: both textures measure depth, so they should be strongly
    # correlated. A negative r means opposite sign conventions -- fine for MSM
    # as long as input and reference use the same texture, but worth knowing.
    if "sulc" in metrics and "dpfstar" in metrics:
        r = np.corrcoef(metrics["sulc"], metrics["dpfstar"])[0, 1]
        print(f"corr(sulc, dpfstar) = {r:+.3f}")

    if failures:
        print("\nQC FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nQC PASSED")


if __name__ == "__main__":
    main()
