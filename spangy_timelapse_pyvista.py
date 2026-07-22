"""
SPANGY dominant-band timelapse viewer / exporter (PyVista).

Walks through sessions ses-01 ... ses-10 for a given subject, loads the
white-matter surface mesh and its corresponding SPANGY dominant-band
texture for both hemispheres.

Texture values:
    Positive (1–6) : gyri   dominant band B1–B6
    Negative (-1 to -6) : sulci  dominant band B1–B6
    0               : no dominant band

Toggle modes:
    'gyri'  — show only positive values (gyri), mask sulci to grey
    'sulci' — show only negative values (sulci), mask gyri to grey
    'both'  — show all values

Usage:
    conda activate babofet
    python spangy_timelapse_pyvista.py                            # interactive viewer (left hemi, sulci)
    python spangy_timelapse_pyvista.py --hemi R --mode gyri       # right hemi, gyri only
    python spangy_timelapse_pyvista.py --export out.mp4 --mode both
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

# ── Configuration ────────────────────────────────────────────────────────────

MESH_DIR    = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/all_mesh_combined/'
TEXTURE_DIR = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy/textures/'

SUBJECT  = 'sub-Borgne'
SESSIONS = [f'ses-{i:02d}' for i in range(1, 11)]
HEMI_TEXTURE_NAME = {'L': 'left', 'R': 'right'}

# Band index → colour (same for gyri and sulci of the same band)
BAND_COLORS = {
    0: '#d3d3d3',   # no dominant band
    1: '#d3d3d3',   # B1 - not of interest
    2: '#d3d3d3',   # B2 - not of interest
    3: '#ffd700',   # B3 - gold
    4: '#4393c3',   # B4 - blue
    5: '#2ca25f',   # B5 - green
    6: '#d73027',   # B6 - red
}
MASKED_COLOR = '#e8e8e8'   # grey for vertices masked by current mode
N_BANDS = 7                # bands 0–6 (slot 7 = masked sentinel)


# ── Discrete colourmap + norm ─────────────────────────────────────────────────

def build_colormap():
    """
    Fully discrete colourmap: one solid colour block per band (0–6) plus
    the masked sentinel (7). Uses BoundaryNorm so each integer value maps
    to exactly one colour with no interpolation.
    """
    colors = [BAND_COLORS[b] for b in range(N_BANDS)] + [MASKED_COLOR]
    cmap   = mcolors.ListedColormap(colors)
    # Boundaries between each integer band value
    bounds = np.arange(0, N_BANDS + 2) - 0.5   # [-0.5, 0.5, 1.5 … 7.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# ── File path helpers ─────────────────────────────────────────────────────────

def mesh_path(session, hemi):
    return os.path.join(MESH_DIR, f'{SUBJECT}_{session}_hemi-{hemi}_white.surf.gii')


def texture_path(session, hemi):
    return os.path.join(TEXTURE_DIR, f'spangy_dom_band_{SUBJECT}_{session}_{HEMI_TEXTURE_NAME[hemi]}.gii')


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_gifti_mesh(path):
    gii      = nib.load(path)
    vertices = gii.darrays[0].data
    faces    = gii.darrays[1].data
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    return pv.PolyData(vertices, faces_pv)


def load_gifti_texture(path):
    return nib.load(path).darrays[0].data.astype(int)


def load_session(session, hemi):
    mp, tp = mesh_path(session, hemi), texture_path(session, hemi)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Mesh not found: {mp}")
    if not os.path.exists(tp):
        raise FileNotFoundError(f"Texture not found: {tp}")
    mesh = load_gifti_mesh(mp)
    band = load_gifti_texture(tp)
    if len(band) != mesh.n_points:
        raise ValueError(
            f"Vertex count mismatch {session}/{hemi}: mesh={mesh.n_points}, texture={len(band)}"
        )
    mesh.point_data['raw_band'] = band
    return mesh


# ── Mode masking ──────────────────────────────────────────────────────────────

def apply_mode(mesh, mode):
    """
    Map signed raw_band (-6…6) to a display index (0–7):
        abs(band) for valid vertices in the selected mode
        7 (sentinel → MASKED_COLOR) for everything else
    """
    raw  = mesh.point_data['raw_band'].copy()
    disp = np.where(raw > 0, raw, np.where(raw < 0, -raw, 0))

    if mode == 'gyri':
        mask = raw <= 0
    elif mode == 'sulci':
        mask = raw >= 0
    else:
        mask = np.zeros(len(raw), dtype=bool)

    disp[mask] = N_BANDS   # sentinel
    mesh.point_data['display_band'] = disp
    return mesh


# ── Scalar bar ────────────────────────────────────────────────────────────────

def scalar_bar_args(mode):
    labels = {
        'both':  'SPANGY band (gyri + sulci)',
        'gyri':  'SPANGY band (gyri only)',
        'sulci': 'SPANGY band (sulci only)',
    }
    return {
        'title':            labels[mode],
        'n_labels':         0,            # suppress auto labels — we add custom ones
        'label_font_size':  11,
        'fmt':              '%.0f',
    }


def add_discrete_scalar_bar(plotter, mode):
    """
    Add a discrete scalar bar with one labelled tick per band of interest.
    Labels are centred on each colour block.
    """
    cmap, norm = build_colormap()

    # Band slots visible in this mode
    if mode == 'gyri':
        active_bands = [3, 4, 5, 6]
    elif mode == 'sulci':
        active_bands = [3, 4, 5, 6]   # abs values — same slots
    else:
        active_bands = [3, 4, 5, 6]

    band_labels = {0: 'B0', 1: 'B1', 2: 'B2', 3: 'B3',
                   4: 'B4', 5: 'B5', 6: 'B6', 7: 'masked'}

    plotter.add_scalar_bar(
        title=scalar_bar_args(mode)['title'],
        mapper=None,
        n_labels=N_BANDS + 1,
        label_font_size=11,
        fmt='%.0f',
    )


# ── Interactive viewer ────────────────────────────────────────────────────────

def interactive_viewer(hemi, mode):
    cmap, norm = build_colormap()

    meshes = {}
    for session in SESSIONS:
        try:
            m = load_session(session, hemi)
            meshes[session] = apply_mode(m, mode)
            print(f"Loaded {session} hemi-{hemi}")
        except FileNotFoundError as e:
            print(f"[warning] Skipping {session}: {e}")

    if not meshes:
        raise RuntimeError(f"No sessions found for hemi-{hemi}")

    available_sessions = list(meshes.keys())
    current_mode       = [mode]
    actor_holder       = {'actor': None}

    plotter = pv.Plotter(window_size=(1100, 850))

    def render(session):
        if actor_holder['actor'] is not None:
            plotter.remove_actor(actor_holder['actor'])

        mesh = meshes[session]
        apply_mode(mesh, current_mode[0])

        actor_holder['actor'] = plotter.add_mesh(
            mesh,
            scalars='display_band',
            cmap=cmap,
            clim=[-0.5, N_BANDS + 0.5],   # centre colormap on integer values
            n_colors=N_BANDS + 1,          # one solid block per band
            show_scalar_bar=True,
            scalar_bar_args={
                'title':           scalar_bar_args(current_mode[0])['title'],
                'n_labels':        N_BANDS + 1,
                'label_font_size': 11,
                'fmt':             '%.0f',
            },
        )
        plotter.add_text(
            f"{SUBJECT}  |  hemi-{hemi}  |  {session}  |  [{current_mode[0]}]",
            position='upper_left', font_size=13, name='session_label'
        )

    def on_slider(value):
        idx     = max(0, min(int(round(value)) - 1, len(available_sessions) - 1))
        session = available_sessions[idx]
        render(session)

    def set_mode_both(flag):
        current_mode[0] = 'both'
        on_slider(plotter.slider_widgets[0].GetRepresentation().GetValue())

    def set_mode_gyri(flag):
        current_mode[0] = 'gyri'
        on_slider(plotter.slider_widgets[0].GetRepresentation().GetValue())

    def set_mode_sulci(flag):
        current_mode[0] = 'sulci'
        on_slider(plotter.slider_widgets[0].GetRepresentation().GetValue())

    render(available_sessions[0])

    plotter.add_slider_widget(
        on_slider,
        rng=[1, len(available_sessions)],
        value=1,
        title='Session',
        fmt='%0.0f',
        pointa=(0.25, 0.92), pointb=(0.75, 0.92),
    )

    plotter.add_checkbox_button_widget(set_mode_both,  position=(10,  10), size=30, value=mode == 'both',  color_on='white',   color_off='grey')
    plotter.add_text('Both',  position=(45,  13), font_size=10, name='lbl_both')
    plotter.add_checkbox_button_widget(set_mode_gyri,  position=(110, 10), size=30, value=mode == 'gyri',  color_on='#4393c3', color_off='grey')
    plotter.add_text('Gyri',  position=(145, 13), font_size=10, name='lbl_gyri')
    plotter.add_checkbox_button_widget(set_mode_sulci, position=(210, 10), size=30, value=mode == 'sulci', color_on='#d73027', color_off='grey')
    plotter.add_text('Sulci', position=(245, 13), font_size=10, name='lbl_sulci')

    plotter.add_axes()
    plotter.show()


# ── MP4 export ────────────────────────────────────────────────────────────────

def export_timelapse(hemi, mode, output_path, fps=1, hold_frames=15):
    cmap, norm = build_colormap()
    plotter    = pv.Plotter(off_screen=True, window_size=(1100, 850))
    plotter.open_movie(output_path, framerate=fps)

    camera_set = False

    for session in SESSIONS:
        try:
            mesh = load_session(session, hemi)
            apply_mode(mesh, mode)
            print(f"Rendering {session} hemi-{hemi} [{mode}]")
        except FileNotFoundError as e:
            print(f"[warning] Skipping {session}: {e}")
            continue

        plotter.clear_actors()
        plotter.add_mesh(
            mesh,
            scalars='display_band',
            cmap=cmap,
            clim=[-0.5, N_BANDS + 0.5],
            n_colors=N_BANDS + 1,
            show_scalar_bar=True,
            scalar_bar_args={
                'title':           scalar_bar_args(mode)['title'],
                'n_labels':        N_BANDS + 1,
                'label_font_size': 11,
                'fmt':             '%.0f',
            },
        )
        plotter.add_text(
            f"{SUBJECT}  |  hemi-{hemi}  |  {session}  |  [{mode}]",
            position='upper_left', font_size=13, name='session_label'
        )

        if not camera_set:
            plotter.camera_position = 'xy'
            camera_set = True

        for _ in range(hold_frames):
            plotter.write_frame()

    plotter.close()
    print(f"Saved timelapse → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPANGY dominant-band timelapse (PyVista)")
    parser.add_argument('--hemi',        choices=['L', 'R'],               default='L')
    parser.add_argument('--mode',        choices=['both', 'gyri', 'sulci'], default='sulci')
    parser.add_argument('--export',      metavar='OUTPUT.mp4',              default=None)
    parser.add_argument('--fps',         type=int,                          default=1)
    parser.add_argument('--hold-frames', type=int,                          default=15)
    args = parser.parse_args()

    if args.export:
        export_timelapse(args.hemi, args.mode, args.export,
                         fps=args.fps, hold_frames=args.hold_frames)
    else:
        interactive_viewer(args.hemi, args.mode)


if __name__ == '__main__':
    main()