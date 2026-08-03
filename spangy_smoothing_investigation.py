import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
from slam.differential_geometry import laplacian_mesh_smoothing
import os
import pandas as pd
import time
import slam.spangy as spgy
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe on compute nodes
import matplotlib.pyplot as plt
import trimesh
import sys

# ── Output directories ────────────────────────────────────────────────────────
BASE_OUT = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/new_hemi_meshes_10_smoothing_iterations/left/analysis/spangy_smoothing'
PLOTS_DIR       = os.path.join(BASE_OUT, 'plots')
TEXTURES_DIR    = os.path.join(BASE_OUT, 'textures')
CURV_PRINC_DIR  = os.path.join(BASE_OUT, 'curvature', 'principal')
CURV_MEAN_DIR   = os.path.join(BASE_OUT, 'curvature', 'mean')
RESULTS_DIR     = os.path.join(BASE_OUT, 'results')

for d in [PLOTS_DIR, TEXTURES_DIR, CURV_PRINC_DIR, CURV_MEAN_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Mesh source ───────────────────────────────────────────────────────────────
SURFACE_PATH = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/new_hemi_meshes_10_smoothing_iterations/left/chosen_meshes_for_smoothing_analysis'

# ── Smoothing sweep ───────────────────────────────────────────────────────────
SMOOTHING_ITERS = [0, 5, 10, 15, 20]   # nb_iter values to test (0 = raw mesh)
SMOOTHING_DT    = 0.1                   # laplacian smoothing step, held constant
N_EIG           = 4000

# To sweep ONE specific mesh across smoothing levels, put its filename here.
# Leave as None to sweep every mesh in SURFACE_PATH (SLURM-chunked as before).
TARGET_FILE = None
# e.g. TARGET_FILE = 'sub-Borgne_ses-05_hemi-L_white.surf.gii'


def get_hull_area(mesh):
    convex_hull = trimesh.convex.convex_hull(mesh)
    return float(convex_hull.area)


def get_gyrification_index(mesh):
    hull_area = get_hull_area(mesh)
    gyrification_index = float(mesh.area) / hull_area
    return gyrification_index, hull_area


def parse_participant_session(filename):
    """Derive participant_session label directly from the filename."""
    hemisphere = 'left' if filename.endswith('left_wm.gii') else 'right'
    parts = filename.split('_')
    base = parts[0] + '_' + parts[1] if len(parts) >= 2 else parts[0]
    return f'{base}_{hemisphere}'


def process_single_file(filename, nb_iter, dt=SMOOTHING_DT):
    """Run SPANGY on one mesh at one smoothing level. Returns one result dict."""
    try:
        start_time = time.time()
        participant_session = parse_participant_session(filename)
        tag = f'{participant_session}_smooth{nb_iter:02d}'   # unique per smoothing level
        print(f"\nProcessing: {filename}  (nb_iter={nb_iter})")

        # ── Skip if this smoothing level already processed ────────────────────
        texture_out = os.path.join(TEXTURES_DIR, f'spangy_dom_band_{tag}.gii')
        if os.path.exists(texture_out):
            print(f"  Skipping {tag} — already processed.")
            return None

        mesh_file = os.path.join(SURFACE_PATH, filename)
        if not os.path.exists(mesh_file):
            print(f"  Error: file not found: {mesh_file}")
            return None

        # ── Load FRESH each time, then smooth to this nb_iter ─────────────────
        # (smoothing is cumulative, so every level must start from the raw mesh)
        mesh = sio.load_mesh(mesh_file)
        if nb_iter > 0:
            mesh = laplacian_mesh_smoothing(mesh, nb_iter=nb_iter, dt=dt)
        mesh.apply_transform(mesh.principal_inertia_transform)

        # Eigenpairs
        print("  Computing eigenpairs...")
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N_EIG)

        # Curvature
        print("  Computing curvature...")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)

        tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
        sio.write_texture(
            tex_PrincipalCurvatures,
            os.path.join(CURV_PRINC_DIR, f'principal_curv_{tag}.gii')
        )

        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        sio.write_texture(
            tex_mean_curv,
            os.path.join(CURV_MEAN_DIR, f'filt_mean_curv_{tag}.gii')
        )
        filt_mean_curv = tex_mean_curv.darray.squeeze()
        total_mean_curv = float(np.sum(filt_mean_curv))

        # Spectrum
        print("  Computing spectrum...")
        grouped_spectrum, group_indices, coefficients, nlevels = \
            spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
        levels = len(group_indices)
        gs = np.asarray(grouped_spectrum).squeeze()
        n_bands = gs.shape[0]

        # Plot (frequency uses the correct Eq. 10 form: sqrt(lambda)/(2*pi))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        frequency = np.sqrt(eigVal) / (2 * np.pi)
        ax1.scatter(frequency, coefficients, marker='o', s=20, linewidths=0.5)
        ax1.set_xlabel('Frequency (mm⁻¹)'); ax1.set_ylabel('Coefficients')
        ax2.scatter(frequency[1:], coefficients[1:], marker='o', s=20, linewidths=0.5)
        ax2.set_xlabel('Frequency (mm⁻¹)'); ax2.set_ylabel('Coefficients')
        ax3.bar(np.arange(0, levels), gs)
        ax3.set_xlabel('SPANGY Frequency Bands'); ax3.set_ylabel('Power Spectrum')
        ax3.set_title(f'{participant_session}  nb_iter={nb_iter}')
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'{tag}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Whole-brain metrics
        mL_in_MM3, CM2_in_MM2 = 1000, 100
        volume       = mesh.volume
        surface_area = mesh.area
        afp          = float(np.sum(gs[1:]))   # analyze folding power (bands B1+)
        print(f"  Volume={np.floor(volume/mL_in_MM3):.0f} mL  "
              f"Area={np.floor(surface_area/CM2_in_MM2):.0f} cm²  AFP={afp:.4f}")

        # Local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(
            coefficients, filt_mean_curv, levels, group_indices, eigVects
        )
        tmp_tex = stex.TextureND(loc_dom_band)
        sio.write_texture(tmp_tex, texture_out)

        # Gyrification index
        gyrification_index, hull_area = get_gyrification_index(mesh)

        execution_time = time.time() - start_time
        print(f"  Done in {execution_time:.2f}s")

        # ── Assemble result row ───────────────────────────────────────────────
        row = {
            'participant_session':  participant_session,
            'filename':             filename,
            'smoothing_iter':       nb_iter,
            'smoothing_dt':         dt,
            'nlevels':              int(nlevels),
            'total_mean_curvature': total_mean_curv,
            'gyrification_index':   gyrification_index,
            'hull_area':            hull_area,
            'volume_ml':            np.floor(volume / mL_in_MM3),
            'surface_area_cm2':     np.floor(surface_area / CM2_in_MM2),
            'analyze_folding_power': afp,
            'processing_time_s':    execution_time,
        }
        # raw band power B0..B6 (index-safe), plus normalized fraction B1..B6.
        # raw power drops overall as you smooth; the fraction shows how the
        # folding energy REDISTRIBUTES across bands, which is usually what you
        # want when plotting the impact of smoothing on B3-B6.
        for k in range(7):
            row[f'band_power_B{k}'] = float(gs[k]) if k < n_bands else np.nan
        for k in range(1, 7):
            row[f'band_frac_B{k}'] = (float(gs[k] / afp)
                                      if (k < n_bands and afp > 0) else np.nan)
        return row

    except Exception as e:
        print(f"  Error processing {filename} (nb_iter={nb_iter}): {e}")
        return None


def main():
    try:
        if TARGET_FILE is not None:
            all_files = [TARGET_FILE]
        else:
            all_files = [
                f for f in os.listdir(SURFACE_PATH)
                if f.endswith('left_wm.gii') or f.endswith('right_wm.gii')
            ]
        print(f"Found {len(all_files)} surface file(s); "
              f"smoothing levels = {SMOOTHING_ITERS}")

        # SLURM array chunking over FILES (each file still runs the full sweep)
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        chunk_size = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks else 0)
        start_idx  = task_id * chunk_size
        end_idx    = min((task_id + 1) * chunk_size, len(all_files))
        chunk      = all_files[start_idx:end_idx]

        print(f"Task {task_id+1}/{n_tasks} — files {start_idx}–{end_idx-1} "
              f"({len(chunk)} file(s) × {len(SMOOTHING_ITERS)} smoothing levels)")

        results = []
        for f in chunk:
            for nb_iter in SMOOTHING_ITERS:
                r = process_single_file(f, nb_iter, SMOOTHING_DT)
                if r is not None:
                    results.append(r)

        if results:
            out_path = os.path.join(RESULTS_DIR, f'chunk_{task_id}_smoothing_results.csv')
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"\nSaved {len(results)} run(s) → {out_path}")
        else:
            print(f"Warning: no results for task {task_id}")

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()