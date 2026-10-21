#!/usr/bin/env python3
"""
Interactive SPANGY dominant-band viewer.

A local Dash app where you can:
  * browse to / type in the mesh, texture and output directories from the UI,
  * pick any mesh from a dropdown,
  * pick any texture from a dropdown (or leave it on "Auto-match"),
  * toggle which bands are shown (B3..B6 and B-3..B-6 by default),
  * switch camera (medial / lateral / anterior / posterior / superior),
  * take a high-resolution screenshot (camera icon in the plot toolbar),
  * export the current view as a standalone interactive HTML file,
  * export the current view as a PNG server-side (needs kaleido).

Usage
-----
    python spangy_band_viewer.py [--mesh-dir ...] [--tex-dir ...] [--out-dir ...]

The directory arguments are only starting points -- all three can be changed
from inside the app. Then open http://127.0.0.1:8050 in a browser.
If you run this on a remote machine, forward the port:
    ssh -L 8050:localhost:8050 user@host

Requires: dash, plotly, nibabel, numpy, slam (sio.load_mesh), optionally kaleido.
"""

import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib
import plotly.graph_objs as go

import dash
from dash import dcc, html, Input, Output, State, no_update

import slam.io as sio


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MESH_DIR = "/home/INT/dienye.h/python_files/rough/temp/mesh"
DEFAULT_TEX_DIR = "/home/INT/dienye.h/python_files/rough/temp/texture"
DEFAULT_OUT_DIR = "/home/INT/dienye.h/python_files/rough/temp/html_outputs"

# Bands of interest. Negative = sulci, positive = gyri.
SULCI_BANDS = [-6, -5, -4, -3]
GYRI_BANDS = [3, 4, 5, 6]
ALL_BANDS = SULCI_BANDS + GYRI_BANDS
DEFAULT_BANDS = ALL_BANDS

# Edit this to change the figure palette. Keys are dominant-band values.
BAND_COLORS = {
    -6: "#FF0000",   # red
    -5: "#00FF00",   # green
    -4: "#0000FF",   # blue
    -3: "#FFD700",   # yellow
    -2: "#d1e5f0",   # pale blue
    -1: "#f7f7f7",   # near white
    0:  "#dcdcdc",   # grey (no dominant band)
    1:  "#fddbc7",   # pale red
    2:  "#f4a582",   # light red
    3:  "#FFD700",   # yellow
    4:  "#0000FF",   # blue
    5:  "#00FF00",   # green
    6:  "#FF0000",   # red
}

BAND_MIN, BAND_MAX = -6, 6
UNSELECTED_COLOR = "#d9d9d9"     # colour for vertices outside the selected bands
CMIN, CMAX = BAND_MIN - 0.5, BAND_MAX + 0.5


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

def list_gii(directory):
    """Return sorted .gii file names in a directory (empty list if invalid)."""
    if not directory:
        return []
    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    try:
        return sorted(p.name for p in d.iterdir()
                      if p.is_file() and p.name.lower().endswith(".gii"))
    except PermissionError:
        return []


def list_subdirs(directory):
    """Return sorted sub-directory names, for the in-app directory browser."""
    if not directory:
        return []
    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    try:
        return sorted(p.name for p in d.iterdir()
                      if p.is_dir() and not p.name.startswith("."))
    except PermissionError:
        return []


@lru_cache(maxsize=32)
def load_mesh_arrays(mesh_path):
    """Load a mesh once and cache its (vertices, faces) arrays.

    The mesh object is never mutated: orientation transforms are applied to a
    copy of the vertex array, so repeated views stay consistent.
    """
    mesh = sio.load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    return verts, faces


@lru_cache(maxsize=64)
def read_gii_file(file_path):
    """Read the first scalar darray from a GIFTI file. Returns None on failure."""
    try:
        gifti_img = nib.load(file_path)
        return np.asarray(gifti_img.darrays[0].data).ravel()
    except Exception as e:  # noqa: BLE001 - surfaced in the UI status line
        print(f"Error loading texture {file_path}: {e}")
        return None


def strip_gii(name):
    """Drop the .gii / .surf.gii extension (used only for output filenames)."""
    s = Path(name).name
    for ext in (".surf.gii", ".gii"):
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s


# --------------------------------------------------------------------------- #
# Mesh <-> texture auto-matching (BIDS entities only)
# --------------------------------------------------------------------------- #

def detect_hemisphere(name):
    """Return 'left', 'right' or None, inferred from a filename."""
    s = name.lower()
    left = bool(re.search(r"(^|[._\-])(left|lh|lhemi|l)([._\-]|$)", s))
    right = bool(re.search(r"(^|[._\-])(right|rh|rhemi|r)([._\-]|$)", s))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def parse_entities(name):
    """Extract (subject, session, hemisphere) from a filename where present."""
    stem = Path(name).name
    sub = re.search(r"(sub-[A-Za-z0-9]+)", stem, flags=re.I)
    ses = re.search(r"(ses-[A-Za-z0-9]+)", stem, flags=re.I)
    return (sub.group(1).lower() if sub else None,
            ses.group(1).lower() if ses else None,
            detect_hemisphere(stem))


def match_score(mesh_name, tex_name):
    """Score in [0, 1] from subject / session / hemisphere agreement.

    Any entity present on both sides but disagreeing vetoes the pair outright.
    Entities missing from either name are simply ignored.
    """
    matched = possible = 0
    for a, b in zip(parse_entities(mesh_name), parse_entities(tex_name)):
        if a is None or b is None:
            continue
        possible += 1
        if a != b:
            return 0.0
        matched += 1
    return matched / possible if possible else 0.0


def auto_match(mesh_name, tex_names):
    """Best texture for a mesh -> (name or None, score, n_tied)."""
    scored = [(t, match_score(mesh_name, t)) for t in tex_names]
    scored = [(t, s) for t, s in scored if s > 0]
    if not scored:
        return None, 0.0, 0
    best_score = max(s for _, s in scored)
    tied = [t for t, s in scored if s == best_score]
    return tied[0], best_score, len(tied)


# --------------------------------------------------------------------------- #
# Geometry / colour helpers
# --------------------------------------------------------------------------- #

ROT180_Y = np.array([[-1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]], dtype=float)


def orientation_transform(rotate180_y=False):
    """4x4 transform applied to the surface before rendering.

    The original script always applied a 180 deg rotation about the y axis
    (x -> -x, z -> -z), which is what turns the brain upside down when the
    camera up-vector is +z. It is now OFF by default and exposed as a checkbox,
    so meshes are rendered in their native coordinates unless you ask otherwise.
    """
    return ROT180_Y if rotate180_y else np.eye(4)


def apply_transform(verts, transform):
    """Apply a 4x4 homogeneous transform to an (N, 3) vertex array (returns a copy)."""
    return verts @ transform[:3, :3].T + transform[:3, 3]


# Base camera positions, expressed in native (RAS-like) coordinates:
# +x = right, +y = anterior, +z = superior.
# "hemi_dependent" views look from the side, so which side depends on the
# hemisphere: a left hemisphere's medial face points to +x, a right one's to -x.
VIEWS = {
    "medial":    dict(eye=(1.8, 0.0, 0.0), up=(0, 0, 1), hemi_dependent=True),
    "lateral":   dict(eye=(-1.8, 0.0, 0.0), up=(0, 0, 1), hemi_dependent=True),
    "anterior":  dict(eye=(0.0, 1.8, 0.0), up=(0, 0, 1), hemi_dependent=False),
    "posterior": dict(eye=(0.0, -1.8, 0.0), up=(0, 0, 1), hemi_dependent=False),
    "superior":  dict(eye=(0.0, 0.0, 1.8), up=(0, 1, 0), hemi_dependent=False),
    "inferior":  dict(eye=(0.0, 0.0, -1.8), up=(0, 1, 0), hemi_dependent=False),
}


def get_camera(name, hemisphere="left", flip_vertical=False, rotate180_y=False):
    """Camera for a named anatomical view.

    The eye is mirrored for right hemispheres on the side views, and both eye
    and up are rotated with the surface if the 180 deg rotation is enabled, so
    the named view keeps showing the same anatomical face either way.
    """
    view = VIEWS.get(name, VIEWS["medial"])
    eye = list(view["eye"])
    up = list(view["up"])

    if view["hemi_dependent"] and hemisphere == "right":
        eye[0] = -eye[0]

    if rotate180_y:                      # follow the surface through the rotation
        eye[0], eye[2] = -eye[0], -eye[2]
        up[0], up[2] = -up[0], -up[2]

    if flip_vertical:
        up = [-u for u in up]

    return dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                center=dict(x=0, y=0, z=0),
                up=dict(x=up[0], y=up[1], z=up[2]))


def discrete_colorscale(selected_bands):
    """Piecewise-constant colorscale so each integer band keeps its exact colour.

    Every band occupies [k - 0.5, k + 0.5] of the intensity range, mapped onto
    [0, 1] between CMIN and CMAX. Unselected bands get a neutral grey.
    """
    span = CMAX - CMIN
    scale = []
    for k in range(BAND_MIN, BAND_MAX + 1):
        color = BAND_COLORS.get(k, UNSELECTED_COLOR) if k in selected_bands \
            else UNSELECTED_COLOR
        scale.append([round(((k - 0.5) - CMIN) / span, 6), color])
        scale.append([round(((k + 0.5) - CMIN) / span, 6), color])
    scale[0][0] = 0.0
    scale[-1][0] = 1.0
    return scale


# --------------------------------------------------------------------------- #
# Figure construction
# --------------------------------------------------------------------------- #

def build_figure(mesh_dir, tex_dir, mesh_name, tex_name, selected_bands,
                 camera_name="medial", hide_unselected=True, flip_vertical=False,
                 rotate180_y=False, title=None):
    """Build the Plotly figure for one mesh/texture pair. Returns (fig, status)."""
    selected_bands = sorted(set(int(b) for b in (selected_bands or [])))
    status = []

    verts, faces = load_mesh_arrays(str(Path(mesh_dir).expanduser() / mesh_name))
    hemi = detect_hemisphere(mesh_name) or "left"
    if rotate180_y:
        verts = apply_transform(verts, orientation_transform(True))

    scalars = None
    if tex_name:
        scalars = read_gii_file(str(Path(tex_dir).expanduser() / tex_name))
        if scalars is None:
            status.append(f"Could not read texture '{tex_name}'.")
        elif scalars.shape[0] != verts.shape[0]:
            status.append(
                f"Vertex count mismatch: mesh has {verts.shape[0]} vertices, "
                f"texture has {scalars.shape[0]}. Texture not applied."
            )
            scalars = None

    legend_traces = []

    if scalars is None:
        mesh_trace = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#cccccc", opacity=1.0, hoverinfo="skip",
        )
    else:
        display = np.asarray(scalars, dtype=float).copy()
        rounded = np.round(display)

        if hide_unselected:
            # NaN vertices are not rendered -> only the bands of interest show.
            display[~np.isin(rounded, selected_bands)] = 0
        display = np.clip(display, BAND_MIN, BAND_MAX)

        mesh_trace = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=display,
            intensitymode="vertex",
            colorscale=discrete_colorscale(selected_bands),
            cmin=CMIN, cmax=CMAX,
            showscale=False,
            hovertemplate="Band: %{intensity:.0f}<extra></extra>",
        )

        # Legend entries, annotated with the share of vertices in each band.
        total = rounded.size
        for b in selected_bands:
            pct = 100.0 * np.count_nonzero(rounded == b) / total if total else 0.0
            legend_traces.append(
                go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="markers",
                    marker=dict(size=10, color=BAND_COLORS.get(b, "#000000")),
                    name=f"B{b}  ({pct:.1f}%)",
                    showlegend=True, hoverinfo="skip",
                )
            )

    fig = go.Figure(data=[mesh_trace] + legend_traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=get_camera(camera_name, hemi, flip_vertical, rotate180_y),
        ),
        height=850,
        margin=dict(l=10, r=10, b=10, t=60 if title else 10),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)"),
        paper_bgcolor="white",
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, y=0.97,
                                     xanchor="center", yanchor="top",
                                     font=dict(size=18)))
    return fig, " ".join(status)


def output_stem(mesh_name, selected_bands, camera_name):
    """Filename stem used for screenshots and exports."""
    bands = "-".join(f"B{b}" for b in sorted(set(selected_bands or [])))
    parts = [strip_gii(mesh_name)] + ([bands] if bands else []) + [camera_name]
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #

LABEL_STYLE = {"fontWeight": "600", "marginTop": "14px", "marginBottom": "4px",
               "display": "block"}
BTN_STYLE = {"marginRight": "6px", "marginTop": "6px", "padding": "6px 10px",
             "cursor": "pointer"}
HINT_STYLE = {"color": "#666", "fontSize": "12px", "marginTop": "4px",
              "wordBreak": "break-all"}


def dir_picker(prefix, label, initial):
    """Path box + up button + sub-folder dropdown for choosing a directory."""
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(id=f"{prefix}-dir", type="text", value=initial, debounce=True,
                  placeholder="/path/to/folder",
                  style={"width": "100%", "boxSizing": "border-box",
                         "fontSize": "12px", "padding": "4px"}),
        html.Div([
            html.Button("\u2191 Up", id=f"{prefix}-up", n_clicks=0, style=BTN_STYLE),
            html.Button("Home", id=f"{prefix}-home", n_clicks=0, style=BTN_STYLE),
        ]),
        dcc.Dropdown(id=f"{prefix}-subdirs", placeholder="Open sub-folder\u2026",
                     options=[], value=None, style={"marginTop": "4px"}),
        html.Div(id=f"{prefix}-count", style=HINT_STYLE),
    ])


def build_app(mesh_dir, tex_dir, out_dir):
    app = dash.Dash(__name__, title="SPANGY band viewer")

    panel_style = {"width": "340px", "padding": "14px", "fontFamily": "sans-serif",
                   "fontSize": "13px", "borderRight": "1px solid #ddd",
                   "height": "100vh", "overflowY": "auto", "boxSizing": "border-box"}

    app.layout = html.Div(
        style={"display": "flex", "margin": "0"},
        children=[
            html.Div(style=panel_style, children=[
                html.H3("SPANGY band viewer", style={"marginTop": "0"}),

                dir_picker("meshdir", "Mesh directory", str(mesh_dir)),
                dir_picker("texdir", "Texture directory", str(tex_dir)),

                html.Label("Mesh", style=LABEL_STYLE),
                dcc.Dropdown(id="mesh-dd", options=[], value=None,
                             clearable=False, optionHeight=48),

                html.Label("Texture", style=LABEL_STYLE),
                dcc.Dropdown(id="tex-dd", options=[], value="__AUTO__",
                             clearable=False, optionHeight=48),
                html.Div(id="match-info", style=HINT_STYLE),

                html.Label("Bands", style=LABEL_STYLE),
                html.Div([
                    html.Button("All", id="btn-all", n_clicks=0, style=BTN_STYLE),
                    html.Button("Sulci", id="btn-sulci", n_clicks=0, style=BTN_STYLE),
                    html.Button("Gyri", id="btn-gyri", n_clicks=0, style=BTN_STYLE),
                ]),
                dcc.Checklist(
                    id="band-check",
                    options=[{"label": f" B{b}", "value": b}
                             for b in sorted(ALL_BANDS, reverse=True)],
                    value=DEFAULT_BANDS,
                    style={"columnCount": 2, "marginTop": "8px"},
                    inputStyle={"marginRight": "4px"},
                ),
                dcc.Checklist(
                    id="hide-rest",
                    options=[{"label": " Hide unselected bands", "value": "hide"}],
                    value=["hide"], style={"marginTop": "8px"},
                    inputStyle={"marginRight": "4px"},
                ),

                html.Label("Camera", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="camera-radio",
                    options=[{"label": f" {c}", "value": c} for c in VIEWS],
                    value="medial", inputStyle={"marginRight": "4px"},
                ),
                dcc.Checklist(
                    id="orient-opts",
                    options=[
                        {"label": " Rotate 180\u00b0 about Y (legacy transform)",
                         "value": "rot180"},
                        {"label": " Flip vertical (up-axis)", "value": "flip"},
                    ],
                    value=[], style={"marginTop": "6px"},
                    inputStyle={"marginRight": "4px"},
                ),

                dir_picker("outdir", "Output directory", str(out_dir)),
                html.Div("Screenshot: camera icon in the plot toolbar "
                         "(2000x1800 @2x PNG, goes to browser downloads).",
                         style=HINT_STYLE),
                html.Div([
                    html.Button("Save HTML", id="btn-html", n_clicks=0, style=BTN_STYLE),
                    html.Button("Save PNG", id="btn-png", n_clicks=0, style=BTN_STYLE),
                ]),
                dcc.Checklist(
                    id="html-standalone",
                    options=[{"label": " Embed plotly.js (offline HTML)", "value": "full"}],
                    value=["full"], style={"marginTop": "6px"},
                    inputStyle={"marginRight": "4px"},
                ),
                html.Div(id="save-status", style={"marginTop": "8px", "fontSize": "12px",
                                                  "color": "#0a6", "wordBreak": "break-all"}),
            ]),

            html.Div(style={"flex": "1", "padding": "10px"}, children=[
                html.Div(id="view-status", style={"fontFamily": "sans-serif",
                                                  "fontSize": "13px", "color": "#b00",
                                                  "minHeight": "18px"}),
                dcc.Graph(id="mesh-graph", style={"height": "88vh"},
                          config={"displaylogo": False}),
            ]),
        ],
    )

    # -- directory pickers ---------------------------------------------------- #
    for prefix, kind in (("meshdir", "mesh"), ("texdir", "texture"), ("outdir", "output")):

        @app.callback(
            Output(f"{prefix}-subdirs", "options"),
            Output(f"{prefix}-subdirs", "value"),
            Output(f"{prefix}-count", "children"),
            Input(f"{prefix}-dir", "value"),
            prevent_initial_call=False,
        )
        def refresh_dir(path, _kind=kind):
            opts = [{"label": f"\U0001F4C1 {d}", "value": d} for d in list_subdirs(path)]
            if not path or not Path(path).expanduser().is_dir():
                note = ("Folder does not exist yet (it will be created on save)."
                        if _kind == "output" else "Folder not found.")
            elif _kind == "output":
                note = "Exports will be written here."
            else:
                note = f"{len(list_gii(path))} .gii file(s) found."
            return opts, None, note

        @app.callback(
            Output(f"{prefix}-dir", "value"),
            Input(f"{prefix}-subdirs", "value"),
            Input(f"{prefix}-up", "n_clicks"),
            Input(f"{prefix}-home", "n_clicks"),
            State(f"{prefix}-dir", "value"),
            prevent_initial_call=True,
        )
        def navigate(subdir, _up, _home, current, _prefix=prefix):
            trigger = dash.ctx.triggered_id
            if trigger == f"{_prefix}-home":
                return str(Path.home())
            base = Path(current).expanduser() if current else Path.home()
            if trigger == f"{_prefix}-up":
                return str(base.parent)
            if subdir:
                return str(base / subdir)
            return no_update

    # -- band preset buttons --------------------------------------------------- #
    @app.callback(
        Output("band-check", "value"),
        Input("btn-all", "n_clicks"),
        Input("btn-sulci", "n_clicks"),
        Input("btn-gyri", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_preset(_a, _s, _g):
        return {"btn-all": ALL_BANDS,
                "btn-sulci": SULCI_BANDS,
                "btn-gyri": GYRI_BANDS}.get(dash.ctx.triggered_id, ALL_BANDS)

    # -- file dropdowns follow the directories --------------------------------- #
    @app.callback(
        Output("mesh-dd", "options"),
        Output("mesh-dd", "value"),
        Input("meshdir-dir", "value"),
        State("mesh-dd", "value"),
    )
    def update_mesh_options(path, current):
        found = list_gii(path)
        value = current if current in found else (found[0] if found else None)
        return [{"label": m, "value": m} for m in found], value

    @app.callback(
        Output("tex-dd", "options"),
        Output("tex-dd", "value"),
        Input("texdir-dir", "value"),
        Input("mesh-dd", "value"),
        State("tex-dd", "value"),
    )
    def update_texture_options(path, _mesh, current):
        found = list_gii(path)
        opts = [{"label": "Auto-match", "value": "__AUTO__"},
                {"label": "None (plain surface)", "value": "__NONE__"}]
        opts += [{"label": t, "value": t} for t in found]
        if dash.ctx.triggered_id == "mesh-dd":
            value = "__AUTO__"
        else:
            value = current if current in ("__AUTO__", "__NONE__") or current in found \
                else "__AUTO__"
        return opts, value

    # -- main figure ------------------------------------------------------------ #
    @app.callback(
        Output("mesh-graph", "figure"),
        Output("mesh-graph", "config"),
        Output("match-info", "children"),
        Output("view-status", "children"),
        Input("mesh-dd", "value"),
        Input("tex-dd", "value"),
        Input("band-check", "value"),
        Input("hide-rest", "value"),
        Input("camera-radio", "value"),
        Input("orient-opts", "value"),
        Input("meshdir-dir", "value"),
        Input("texdir-dir", "value"),
    )
    def update_figure(mesh_name, tex_choice, bands, hide_rest, camera_name,
                      orient, mesh_path, tex_path):
        if not mesh_name:
            return go.Figure(), {"displaylogo": False}, "", \
                "No mesh selected -- check the mesh directory."

        textures_now = list_gii(tex_path)
        if tex_choice == "__AUTO__":
            tex_name, score, tied = auto_match(mesh_name, textures_now)
            if tex_name is None:
                info = "No match on subject/session/hemisphere -- pick a texture manually."
            elif tied > 1:
                info = f"Auto-matched: {tex_name} ({tied} candidates tied, score {score:.2f})"
            else:
                info = f"Auto-matched: {tex_name} (score {score:.2f})"
        elif tex_choice == "__NONE__":
            tex_name, info = None, "No texture applied."
        else:
            tex_name = tex_choice
            info = f"Match score vs. mesh: {match_score(mesh_name, tex_name):.2f}"

        title = f"{strip_gii(mesh_name)} \u2014 SPANGY dominant bands"
        try:
            fig, status = build_figure(
                mesh_path, tex_path, mesh_name, tex_name, bands,
                camera_name=camera_name,
                hide_unselected=("hide" in (hide_rest or [])),
                flip_vertical=("flip" in (orient or [])),
                rotate180_y=("rot180" in (orient or [])),
                title=title,
            )
        except Exception as e:  # noqa: BLE001
            return go.Figure(), {"displaylogo": False}, info, f"Failed to load mesh: {e}"

        config = {
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": output_stem(mesh_name, bands, camera_name),
                "height": 1800, "width": 2000, "scale": 2,
            },
        }
        return fig, config, info, status

    # -- exports ---------------------------------------------------------------- #
    @app.callback(
        Output("save-status", "children"),
        Input("btn-html", "n_clicks"),
        Input("btn-png", "n_clicks"),
        State("mesh-dd", "value"),
        State("tex-dd", "value"),
        State("band-check", "value"),
        State("hide-rest", "value"),
        State("camera-radio", "value"),
        State("orient-opts", "value"),
        State("html-standalone", "value"),
        State("meshdir-dir", "value"),
        State("texdir-dir", "value"),
        State("outdir-dir", "value"),
        prevent_initial_call=True,
    )
    def export(_h, _p, mesh_name, tex_choice, bands, hide_rest, camera_name,
               orient, standalone, mesh_path, tex_path, out_path):
        if not mesh_name:
            return "Nothing to export."
        if not out_path:
            return "Set an output directory first."

        if tex_choice == "__AUTO__":
            tex_name = auto_match(mesh_name, list_gii(tex_path))[0]
        elif tex_choice == "__NONE__":
            tex_name = None
        else:
            tex_name = tex_choice

        out = Path(out_path).expanduser()
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            return f"Cannot create output directory: {e}"

        fig, _ = build_figure(mesh_path, tex_path, mesh_name, tex_name, bands,
                              camera_name=camera_name,
                              hide_unselected=("hide" in (hide_rest or [])),
                              flip_vertical=("flip" in (orient or [])),
                              rotate180_y=("rot180" in (orient or [])),
                              title=f"{strip_gii(mesh_name)} \u2014 SPANGY dominant bands")
        stem = output_stem(mesh_name, bands, camera_name)

        if dash.ctx.triggered_id == "btn-html":
            path = out / f"{stem}.html"
            fig.write_html(
                path,
                include_plotlyjs=True if "full" in (standalone or []) else "cdn",
                config={"toImageButtonOptions": {"format": "png", "filename": stem,
                                                 "height": 1800, "width": 2000,
                                                 "scale": 2}},
            )
            return f"Saved {path}"

        path = out / f"{stem}.png"
        try:
            fig.write_image(path, width=2000, height=1800, scale=2)
            return f"Saved {path}"
        except Exception as e:  # noqa: BLE001
            return (f"PNG export failed ({e}). Install kaleido "
                    f"(pip install -U kaleido) or use the toolbar camera icon.")

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive SPANGY band viewer")
    parser.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR,
                        help="starting mesh directory (changeable in the app)")
    parser.add_argument("--tex-dir", default=DEFAULT_TEX_DIR,
                        help="starting texture directory (changeable in the app)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="starting output directory (changeable in the app)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Meshes   : {args.mesh_dir}  ({len(list_gii(args.mesh_dir))} .gii)")
    print(f"Textures : {args.tex_dir}  ({len(list_gii(args.tex_dir))} .gii)")
    print(f"Outputs  : {args.out_dir}")
    print(f"Open http://{args.host}:{args.port}")

    app = build_app(args.mesh_dir, args.tex_dir, args.out_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()