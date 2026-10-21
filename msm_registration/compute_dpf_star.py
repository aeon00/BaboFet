#!/usr/bin/env python3
"""
compute_dpf_star.py — compute DPF-star sulcal depth on an anatomical surface
and save it as a GIFTI metric (.func.gii) usable as an MSM texture.

Usage:
    compute_dpf_star.py <anat.surf.gii> <out.func.gii> [--alpha 500]
                        [--adaptation volume_hull|volume|surface]

The texture is computed on the exact mesh given (vertex order preserved,
trimesh process=False), so it matches the sphere made from the same surface.
The scale factor lc is written to <out>.json for reference.
"""
import argparse
import importlib
import json
import os
import sys

import nibabel as nib
import numpy as np
import slam.io as sio

# Where to find dpf_star: installed slam first, then the bundled copy.
# Override with DPF_STAR_MODULE=<module.path> if it lives elsewhere.
CANDIDATES = [os.environ.get("DPF_STAR_MODULE"), "slam.sulcal_depth",
              "slam.differential_geometry", "dpf_star_lib"]


def find_dpf_star():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for name in filter(None, CANDIDATES):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "dpf_star"):
            return mod.dpf_star, name
    sys.exit("dpf_star not found in: " + ", ".join(filter(None, CANDIDATES)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("anat")
    p.add_argument("out")
    p.add_argument("--alpha", type=float, default=500.0)
    p.add_argument("--adaptation", default="volume_hull",
                   choices=["volume_hull", "volume", "surface"])
    args = p.parse_args()

    dpf_star, source = find_dpf_star()
    mesh = sio.load_mesh(args.anat)
    print(f"[dpf_star] {os.path.basename(args.anat)}: {len(mesh.vertices)} vertices, "
          f"using {source}.dpf_star (alpha={args.alpha}, adaptation={args.adaptation})")

    dpf, lc = dpf_star(mesh, alphas=[args.alpha], adaptation=args.adaptation)
    values = np.asarray(dpf[0], dtype=np.float32)

    if values.shape[0] != len(mesh.vertices):
        sys.exit("DPF-star length does not match vertex count")
    if not np.all(np.isfinite(values)):
        sys.exit("DPF-star contains NaN/Inf")

    img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        values, intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32")])
    nib.save(img, args.out)

    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"alpha": args.alpha, "adaptation": args.adaptation,
                   "lc": float(lc), "source": source}, f, indent=2)
    print(f"[dpf_star] lc={lc:.3f}  range=[{values.min():.4f}, {values.max():.4f}]")


if __name__ == "__main__":
    main()
