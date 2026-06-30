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
BASE_OUT = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy'
PLOTS_DIR       = os.path.join(BASE_OUT, 'plots')
TEXTURES_DIR    = os.path.join(BASE_OUT, 'textures')
CURV_PRINC_DIR  = os.path.join(BASE_OUT, 'curvature', 'principal')
CURV_MEAN_DIR   = os.path.join(BASE_OUT, 'curvature', 'mean')
RESULTS_DIR     = os.path.join(BASE_OUT, 'results')

for d in [PLOTS_DIR, TEXTURES_DIR, CURV_PRINC_DIR, CURV_MEAN_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Mesh source ───────────────────────────────────────────────────────────────
SURFACE_PATH = '/envau/work/meca/users/dienye.h/python_files/Babofet/sub-Borgne/all_mesh_combined/'


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
    hemisphere = 'left' if filename.endswith('hemi-L_white.surf.gii') else 'right'
    parts = filename.split('_')
    # e.g. sub-Borgne_ses-01_left.surf.gii  →  sub-Borgne_ses-01_left
    base = parts[0] + '_' + parts[1] if len(parts) >= 2 else parts[0]
    return f'{base}_{hemisphere}'


def process_single_file(filename):
    try:
        start_time = time.time()
        print(f"\nProcessing: {filename}")

        participant_session = parse_participant_session(filename)

        mesh_file = os.path.join(SURFACE_PATH, filename)
        if not os.path.exists(mesh_file):
            print(f"  Error: file not found: {mesh_file}")
            return None

        mesh = sio.load_mesh(mesh_file)
        mesh.apply_transform(mesh.principal_inertia_transform)
        N = 4000

        # Eigenpairs
        print("  Computing eigenpairs...")
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

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
        filt_mean_curv = tex_mean_curv.darray.squeeze()
        total_mean_curv = sum(filt_mean_curv)

        # Spectrum
        print("  Computing spectrum...")
        grouped_spectrum, group_indices, coefficients, nlevels = \
            spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
        levels = len(group_indices)

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        frequency = np.sqrt(eigVal / 2 * np.pi)
        ax1.scatter(frequency, coefficients, marker='o', s=20, linewidths=0.5)
        ax1.set_xlabel('Frequency (m⁻¹)')
        ax1.set_ylabel('Coefficients')
        ax2.scatter(frequency[1:], coefficients[1:], marker='o', s=20, linewidths=0.5)
        ax2.set_xlabel('Frequency (m⁻¹)')
        ax2.set_ylabel('Coefficients')
        ax3.bar(np.arange(0, levels), grouped_spectrum)
        ax3.set_xlabel('Spangy Frequency Bands')
        ax3.set_ylabel('Power Spectrum')
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Whole-brain metrics
        mL_in_MM3, CM2_in_MM2 = 1000, 100
        volume       = mesh.volume
        surface_area = mesh.area
        afp          = np.sum(grouped_spectrum[1:])
        print(f"  Volume={np.floor(volume/mL_in_MM3):.0f} mL  "
              f"Area={np.floor(surface_area/CM2_in_MM2):.0f} cm²  AFP={afp:.4f}")

        # Local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(
            coefficients, filt_mean_curv, levels, group_indices, eigVects
        )
        tmp_tex = stex.TextureND(loc_dom_band)
        sio.write_texture(
            tmp_tex,
            os.path.join(TEXTURES_DIR, f'spangy_dom_band_{participant_session}.gii')
        )

        # Gyrification index
        gyrification_index, hull_area = get_gyrification_index(mesh)

        execution_time = time.time() - start_time
        print(f"  Done in {execution_time:.2f}s")

        return {
            'participant_session':  participant_session,
            'filename':             filename,
            'total_mean_curvature': total_mean_curv,
            'gyrification_index':   gyrification_index,
            'hull_area':            hull_area,
            'band_power_B0':        grouped_spectrum[0],
            'band_power_B1':        grouped_spectrum[1],
            'band_power_B2':        grouped_spectrum[2],
            'band_power_B3':        grouped_spectrum[3],
            'band_power_B4':        grouped_spectrum[4],
            'band_power_B5':        grouped_spectrum[5],
            'band_power_B6':        grouped_spectrum[6],
            'volume_ml':            np.floor(volume / mL_in_MM3),
            'surface_area_cm2':     np.floor(surface_area / CM2_in_MM2),
            'analyze_folding_power': afp,
            'processing_time_s':    execution_time
        }

    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        return None


def main():
    try:
        all_files = [
            f for f in os.listdir(SURFACE_PATH)
            if f.endswith('hemi-L_white.surf.gii') or f.endswith('hemi-R_white.surf.gii')
        ]
        print(f"Found {len(all_files)} surface files in {SURFACE_PATH}")

        # SLURM array chunking (falls back gracefully to single process)
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        chunk_size = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks else 0)
        start_idx  = task_id * chunk_size
        end_idx    = min((task_id + 1) * chunk_size, len(all_files))
        chunk      = all_files[start_idx:end_idx]

        print(f"Task {task_id+1}/{n_tasks} — processing files {start_idx}–{end_idx-1} ({len(chunk)} files)")

        results = [r for f in chunk if (r := process_single_file(f)) is not None]

        if results:
            out_path = os.path.join(RESULTS_DIR, f'chunk_{task_id}_results.csv')
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"\nSaved {len(results)} results → {out_path}")
        else:
            print(f"Warning: no results for task {task_id}")

    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()