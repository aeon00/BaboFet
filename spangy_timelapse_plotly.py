"""
SPANGY dominant-band timelapse viewer / exporter.

Uses facecolor (per-face colour assignment) instead of intensity+colorscale
to avoid Plotly's interpolation artefacts entirely.

Modes (--mode):
    'both'  — show all bands
    'gyri'  — show only positive values (gyri)
    'sulci' — show only negative values (sulci)

Outputs:
    --export-html  : interactive HTML with session slider (open in browser)
    --export-png   : PNG snapshot per session (requires kaleido)
    --export-mp4   : MP4 timelapse (requires ffmpeg + kaleido)

Usage:
    conda activate babofet
    pip install plotly kaleido

    python spangy_timelapse.py --hemi L --mode both --export-html timelapse_L.html
    python spangy_timelapse.py --hemi R --mode sulci --export-mp4 sulci_R.mp4 --fps 2
    python spangy_timelapse.py --hemi L --mode gyri  --export-png ./frames/
"""

import os
import argparse
import subprocess
import numpy as np
import nibabel as nib
import slam.io as sio
import plotly.graph_objs as go
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

MESH_DIR    = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/all_mesh_combined/'
TEXTURE_DIR = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/analysis/spangy/textures/'

SUBJECT  = 'sub-Borgne'
SESSIONS = [f'ses-{i:02d}' for i in range(1, 11)]
HEMI_TEXTURE_NAME = {'L': 'left', 'R': 'right'}

GYRI_BANDS  = {3, 4, 5, 6}
SULCI_BANDS = {-3, -4, -5, -6}
ALL_BANDS   = GYRI_BANDS | SULCI_BANDS

# Per-band colours — same for gyri and sulci of the same band
BAND_COLORS = {
    -6: '#d73027',   # red
    -5: '#2ca25f',   # green
    -4: '#4393c3',   # blue
    -3: '#ffd700',   # gold
    -2: '#d3d3d3',
    -1: '#d3d3d3',
     0: '#f0f0f0',
     1: '#d3d3d3',
     2: '#d3d3d3',
     3: '#ffd700',   # gold
     4: '#4393c3',   # blue
     5: '#2ca25f',   # green
     6: '#d73027',   # red
}
MASKED = '#d3d3d3' # nearly transparent for unselected faces


# ── Per-face colour assignment ────────────────────────────────────────────────

def scalars_to_facecolor(faces, scalars, mode):
    """
    Assign one colour per face. Each face takes the colour of the most common
    valid band value among its 3 vertices. Faces with no valid vertices are
    rendered nearly transparent.

    This bypasses Plotly's vertex-intensity interpolation entirely.
    """
    if mode == 'gyri':
        selected = GYRI_BANDS
    elif mode == 'sulci':
        selected = SULCI_BANDS
    else:
        selected = ALL_BANDS

    rounded     = np.round(scalars).astype(int)
    face_colors = []

    for face in faces:
        vals  = [rounded[v] for v in face]
        valid = [v for v in vals if v in selected]
        if not valid:
            face_colors.append(MASKED)
        else:
            dominant = max(set(valid), key=valid.count)
            face_colors.append(BAND_COLORS[dominant])

    return face_colors


# ── Legend ────────────────────────────────────────────────────────────────────

def build_legend_traces(mode):
    if mode == 'gyri':
        bands = sorted(GYRI_BANDS)
    elif mode == 'sulci':
        bands = sorted(SULCI_BANDS)
    else:
        bands = sorted(ALL_BANDS)

    traces = []
    seen   = set()
    for v in bands:
        k = abs(v)
        if k in seen:
            continue
        seen.add(k)
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=10, color=BAND_COLORS[v]),
            name=f'B{k}', showlegend=True,
        ))
    return traces


# ── Mesh orientation (no transform) ──────────────────────────────────────────

def mesh_orientation(mesh, hemisphere):
    hemisphere = str(hemisphere).lower()
    camera_medial = dict(
        eye=dict(x=1.5,  y=1.0, z=1.0),
        center=dict(x=0, y=0.2, z=0.4),
        up=dict(x=0,     y=0,   z=-1),
    )
    camera_lateral = dict(
        eye=dict(x=-1.5, y=1.0, z=1.0),
        center=dict(x=0, y=0.2, z=0.4),
        up=dict(x=0,     y=0,   z=-1),
    )
    if hemisphere not in ('left', 'right'):
        raise ValueError(f"Invalid hemisphere: {hemisphere}")
    return mesh, camera_medial, camera_lateral


# ── File helpers ──────────────────────────────────────────────────────────────

def mesh_path(session, hemi):
    return os.path.join(MESH_DIR, f'{SUBJECT}_{session}_hemi-{hemi}_white.surf.gii')


def texture_path(session, hemi):
    return os.path.join(TEXTURE_DIR, f'spangy_dom_band_{SUBJECT}_{session}_{HEMI_TEXTURE_NAME[hemi]}.gii')


def load_session(session, hemi):
    mp, tp = mesh_path(session, hemi), texture_path(session, hemi)
    if not os.path.exists(mp):
        raise FileNotFoundError(f"Mesh not found: {mp}")
    if not os.path.exists(tp):
        raise FileNotFoundError(f"Texture not found: {tp}")
    mesh    = sio.load_mesh(mp)
    texture = nib.load(tp).darrays[0].data.astype(float)
    if len(texture) != len(mesh.vertices):
        raise ValueError(
            f"Vertex mismatch {session}/{hemi}: mesh={len(mesh.vertices)}, tex={len(texture)}"
        )
    return mesh, texture


# ── Core figure builder ───────────────────────────────────────────────────────

def plot_mesh_with_legend(vertices, faces, scalars, mode, camera, title=None):
    facecolors = scalars_to_facecolor(faces, scalars, mode)

    mesh3d = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0],    j=faces[:, 1],    k=faces[:, 2],
        facecolor=facecolors,
        showscale=False,
        hovertemplate='<extra></extra>',
    )

    layout = dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera,
        ),
        height=900, width=1100,
        margin=dict(l=10, r=10, b=10, t=50 if title else 10),
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor='rgba(255,255,255,0.8)'),
    )
    if title:
        layout['title'] = dict(text=title, x=0.5, y=0.95,
                               xanchor='center', yanchor='top', font=dict(size=20))

    fig = go.Figure(data=[mesh3d] + build_legend_traces(mode))
    fig.update_layout(**layout)
    return fig


# ── Single frame ──────────────────────────────────────────────────────────────

def make_frame(session, hemi, mode):
    mesh, texture           = load_session(session, hemi)
    mesh, _, camera_lateral = mesh_orientation(mesh, HEMI_TEXTURE_NAME[hemi])
    return plot_mesh_with_legend(
        vertices=mesh.vertices, faces=mesh.faces, scalars=texture,
        mode=mode, camera=camera_lateral,
        title=f'{SUBJECT}  |  hemi-{hemi}  |  {session}  |  [{mode}]',
    )


# ── HTML export ───────────────────────────────────────────────────────────────

def export_html(hemi, mode, output_path):
    all_v, all_f, all_fc, sessions = [], [], [], []

    for session in SESSIONS:
        try:
            mesh, texture           = load_session(session, hemi)
            mesh, _, camera_lateral = mesh_orientation(mesh, HEMI_TEXTURE_NAME[hemi])
            all_v.append(mesh.vertices)
            all_f.append(mesh.faces)
            all_fc.append(scalars_to_facecolor(mesh.faces, texture, mode))
            sessions.append(session)
            print(f"  Loaded {session} hemi-{hemi}")
        except FileNotFoundError as e:
            print(f"  [warning] Skipping {session}: {e}")

    if not sessions:
        raise RuntimeError(f"No sessions found for hemi-{hemi}")

    def make_mesh3d(idx):
        return go.Mesh3d(
            x=all_v[idx][:, 0], y=all_v[idx][:, 1], z=all_v[idx][:, 2],
            i=all_f[idx][:, 0], j=all_f[idx][:, 1], k=all_f[idx][:, 2],
            facecolor=all_fc[idx],
            showscale=False,
        )

    frames = [go.Frame(data=[make_mesh3d(i)], name=s) for i, s in enumerate(sessions)]
    slider_steps = [dict(
        args=[[s], dict(frame=dict(duration=300, redraw=True), mode='immediate')],
        label=s, method='animate',
    ) for s in sessions]

    fig = go.Figure(
        data=[make_mesh3d(0)] + build_legend_traces(mode),
        frames=frames,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=-1.5, y=1.0, z=1.0),
                        center=dict(x=0, y=0.2, z=0.4),
                        up=dict(x=0, y=0, z=-1)),
        ),
        height=900, width=1100,
        margin=dict(l=10, r=10, b=80, t=60),
        title=dict(text=f"{SUBJECT}  |  hemi-{hemi}  |  [{mode}]",
                   x=0.5, y=0.97, xanchor='center', yanchor='top', font=dict(size=18)),
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99,
                    bgcolor='rgba(255,255,255,0.85)'),
        updatemenus=[dict(
            type='buttons', showactive=False, y=0.02, x=0.1, xanchor='right',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=800, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0, steps=slider_steps, x=0.1, y=0.0, len=0.85,
            currentvalue=dict(prefix='Session: ', visible=True, xanchor='center'),
            transition=dict(duration=300),
        )],
    )

    fig.write_html(output_path, config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{SUBJECT}_hemi-{hemi}_{mode}',
            'height': 1800, 'width': 2000, 'scale': 2,
        }
    })
    print(f"Saved interactive HTML → {output_path}")


# ── PNG export ────────────────────────────────────────────────────────────────

def export_png(hemi, mode, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for session in SESSIONS:
        try:
            fig = make_frame(session, hemi, mode)
            out = os.path.join(output_dir, f'{SUBJECT}_{session}_hemi-{hemi}_{mode}.png')
            fig.write_image(out, scale=2)
            print(f"  Saved {out}")
        except FileNotFoundError as e:
            print(f"  [warning] Skipping {session}: {e}")


# ── MP4 export ────────────────────────────────────────────────────────────────

def export_mp4(hemi, mode, output_path, fps=1):
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix='spangy_frames_')
    try:
        export_png(hemi, mode, tmp_dir)
        frame_list = [
            os.path.join(tmp_dir, f'{SUBJECT}_{s}_hemi-{hemi}_{mode}.png')
            for s in SESSIONS
            if os.path.exists(os.path.join(tmp_dir, f'{SUBJECT}_{s}_hemi-{hemi}_{mode}.png'))
        ]
        if not frame_list:
            raise RuntimeError("No frames rendered.")
        list_file = os.path.join(tmp_dir, 'frames.txt')
        with open(list_file, 'w') as f:
            for frame in frame_list:
                f.write(f"file '{frame}'\nduration {1/fps}\n")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path,
        ], check=True)
        print(f"Saved MP4 → {output_path}")
    finally:
        shutil.rmtree(tmp_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPANGY dominant-band timelapse")
    parser.add_argument('--hemi',        choices=['L', 'R'],                default='L')
    parser.add_argument('--mode',        choices=['both', 'gyri', 'sulci'], default='sulci')
    parser.add_argument('--export-html', metavar='OUTPUT.html',             default=None)
    parser.add_argument('--export-png',  metavar='OUTPUT_DIR',              default=None)
    parser.add_argument('--export-mp4',  metavar='OUTPUT.mp4',              default=None)
    parser.add_argument('--fps',         type=int,                          default=1)
    args = parser.parse_args()

    if not any([args.export_html, args.export_png, args.export_mp4]):
        default_out = f'spangy_timelapse_{args.hemi}_{args.mode}.html'
        print(f"No export flag — writing to {default_out}")
        export_html(args.hemi, args.mode, default_out)
    else:
        if args.export_html:
            export_html(args.hemi, args.mode, args.export_html)
        if args.export_png:
            export_png(args.hemi, args.mode, args.export_png)
        if args.export_mp4:
            export_mp4(args.hemi, args.mode, args.export_mp4, fps=args.fps)


if __name__ == '__main__':
    main()