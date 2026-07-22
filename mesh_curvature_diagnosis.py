import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import os
import pandas as pd
import time
import slam.spangy as spgy
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import sys

# ── Output directories ────────────────────────────────────────────────────────
BASE_OUT = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/mean_curv_analysis/'
CURV_PRINC_DIR  = os.path.join(BASE_OUT, 'curvature', 'principal')
CURV_MEAN_DIR   = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/mean_curv_analysis/mean_curvature'

os.makedirs(CURV_PRINC_DIR, exist_ok=True)
os.makedirs(CURV_MEAN_DIR, exist_ok=True)

# ── Mesh source ───────────────────────────────────────────────────────────────
SURFACE_PATH = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/mean_curv_analysis/mesh/'


def get_hull_area(mesh):
    convex_hull = trimesh.convex.convex_hull(mesh)
    return float(convex_hull.area)


def get_gyrification_index(mesh):
    hull_area = get_hull_area(mesh)
    gyrification_index = float(mesh.area) / hull_area
    return gyrification_index, hull_area


def parse_participant_session(filename):
    """
    Derive participant_session label directly from the filename,
    no CSV lookup required.
    """
    hemisphere = 'left' if filename.endswith('left_wm.gii') else 'right'
    parts = filename.split('_')
    # e.g. sub-Borgne_ses-01_left.surf.gii  →  sub-Borgne_ses-01_left
    base = parts[0] + '_' + parts[1] + '_' + parts[2] if len(parts) >= 2 else parts[0]
    return f'{base}_{hemisphere}'


def process_single_file(filename):
    try:
        start_time = time.time()
        print(f"\nProcessing: {filename}")

        participant_session = parse_participant_session(filename)

         # ── Skip if already processed ─────────────────────────────────────────
        curvature_out = os.path.join(CURV_MEAN_DIR, f'filt_mean_curv_{filename}.gii')

        if os.path.exists(curvature_out):
            print(f"  Skipping {participant_session} — already processed.")
            return None

        mesh_file = os.path.join(SURFACE_PATH, filename)
        if not os.path.exists(mesh_file):
            print(f"  Error: file not found: {mesh_file}")
            return None

        mesh = sio.load_mesh(mesh_file)
        mesh.apply_transform(mesh.principal_inertia_transform)
        N = 4000

        # Curvature
        print("  Computing curvature...")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)

        tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
        sio.write_texture(
            tex_PrincipalCurvatures,
            os.path.join(CURV_PRINC_DIR, f'principal_curv_{filename}.gii')
        )

        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        sio.write_texture(
            tex_mean_curv,
            os.path.join(CURV_MEAN_DIR, f'filt_mean_curv_{filename}.gii')
        )

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return None


def main():
    try:
        all_files = [
            f for f in os.listdir(SURFACE_PATH)
            if f.endswith('gii')
        ]
        print(f"Found {len(all_files)} surface files in {SURFACE_PATH}")

        for f in all_files:
            process_single_file(f)

        print("\nDone — all files processed.")

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()