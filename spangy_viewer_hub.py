#!/usr/bin/env python3
"""
SPANGY viewer hub — one app for global and local SPANGY maps.

Two sources, one viewer:

  * Mesh + texture files   any .gii mesh with any .gii texture: whole-hemisphere
                           dominant bands (spangy_babofet_pymesh.py), sulcus maps
                           (spangy_babofet_sulcus.py, ROI shown automatically from
                           roi_used_*.gii), local spectral GI maps (sGI / wGI /
                           band power from local_spectral_gi.py), curvature...
                           Folders can be searched recursively, so you can step
                           through every native session of an animal.
  * Timelapse (.npz)       files written by build_spangy_timelapse.py (global or
                           --sulcus), one mesh numbering for all sessions.

Features
  - directory pickers with quick-path presets (scanned from --base) and sub-folders
  - Prev / Next / Play to cycle through meshes or sessions
  - Sulci / Gyri / Both toggle + per-band checklist; label or continuous colouring
  - ROI: outside greyed or hidden
  - camera presets, fixed scale across sessions (growth stays visible)
  - screenshots: toolbar camera icon, or Save PNG (PyVista / Plotly-kaleido / matplotlib)
  - Save HTML: current view, or an animated Plotly timelapse (play button + slider)
  - Export timelapse as GIF / MP4 / MOV, optional morphing between sessions
  - Open in PyVista: desktop window with session slider, play button (space),
    arrow keys, 's' = screenshot

Usage
    python spangy_viewer_hub.py [--base ...] [--mesh-dir ...] [--tex-dir ...]
                                [--npz-dir ...] [--out-dir ...] [--preset DIR ...]
    -> open http://127.0.0.1:8050   (remote: ssh -L 8050:localhost:8050 user@host)

    python spangy_viewer_hub.py export  --spec job.spec.json   # re-run a saved export
    python spangy_viewer_hub.py pyvista --spec job.spec.json   # PyVista window

Requires: numpy, matplotlib, nibabel (or slam for meshes), dash>=2.9, plotly.
Optional: pyvista (window + fast export), kaleido (Plotly PNG/movie frames),
imageio + imageio-ffmpeg (MP4/MOV), pillow (GIF/PNG).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"
DEFAULT_MESH_DIR = "/home/INT/dienye.h/python_files/rough/temp/mesh"
DEFAULT_TEX_DIR = "/home/INT/dienye.h/python_files/rough/temp/texture"
DEFAULT_NPZ_DIR = os.path.join(DEFAULT_BASE, "spangy_timelapse")
DEFAULT_OUT_DIR = "/home/INT/dienye.h/python_files/rough/temp/html_outputs"

BAND_MIN, BAND_MAX = -6, 6
ALL_BANDS = [b for b in range(BAND_MIN, BAND_MAX + 1) if b != 0]
DEFAULT_BANDS = [-6, -5, -4, -3, 3, 4, 5, 6]

# Edit to change the palette. Keys are dominant-band values (negative = sulci).
BAND_COLORS = {
    -6: "#FF0000", -5: "#00FF00", -4: "#0000FF", -3: "#FFD700",
    -2: "#d1e5f0", -1: "#f7f7f7", 0: "#dcdcdc",
    1: "#fddbc7", 2: "#f4a582",
    3: "#FFD700", 4: "#0000FF", 5: "#00FF00", 6: "#FF0000",
}
UNSELECTED_COLOR = "#d9d9d9"      # inside the ROI, band not shown
OUTSIDE_COLOR = "#f0f0f0"         # outside the ROI / no value
CMIN, CMAX = BAND_MIN - 0.5, BAND_MAX + 0.5
CMAPS = ["viridis", "magma", "plasma", "cividis", "turbo", "RdBu_r", "coolwarm", "Spectral_r"]
ROI_PREFIX = "roi_used_"
MAX_SEQUENCE = 200                # files mode: most meshes in one sequence/export
AUTO, NONE = "__AUTO__", "__NONE__"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def load_surface(path):
    """(vertices (V,3) float, faces (F,3) int) from a GIFTI surface (nibabel, slam fallback)."""
    try:
        import nibabel as nib
        g = nib.load(path)
        v = np.asarray(g.agg_data("pointset"), float)
        f = np.asarray(g.agg_data("triangle"), np.int64)
        if v.ndim == 2 and v.shape[1] == 3 and f.ndim == 2 and f.shape[1] == 3:
            return v, f
    except Exception:
        pass
    import slam.io as sio
    m = sio.load_mesh(path)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


@lru_cache(maxsize=128)
def load_texture(path):
    """Texture as (V, C) float array (C = 1 for ordinary textures)."""
    import nibabel as nib
    d = nib.load(path).agg_data()
    d = np.column_stack(d) if isinstance(d, tuple) else np.asarray(d)
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    elif d.ndim == 2 and d.shape[0] < d.shape[1] and d.shape[0] <= 16:
        d = d.T                                   # stored as (C, V)
    return d


@lru_cache(maxsize=4)
def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def _match_filter(rel, pattern):
    """Space-separated terms must all appear; '!term' must not appear."""
    s = rel.lower()
    for t in (pattern or "").lower().split():
        if t.startswith("!"):
            if t[1:] and t[1:] in s:
                return False
        elif t not in s:
            return False
    return True


def list_files(directory, exts=(".gii",), recursive=False, pattern="", limit=5000):
    """Sorted paths (relative to directory) of files with these extensions."""
    if not directory:
        return []
    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    out = []
    try:
        it = d.rglob("*") if recursive else d.iterdir()
        for p in it:
            if p.is_file() and p.name.lower().endswith(tuple(exts)):
                rel = str(p.relative_to(d))
                if _match_filter(rel, pattern):
                    out.append(rel)
                    if len(out) >= limit:
                        break
    except (PermissionError, OSError):
        pass
    return sorted(out)


def list_subdirs(directory):
    if not directory:
        return []
    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    try:
        return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
    except PermissionError:
        return []


def discover_presets(base, kind, extra=(), limit=300):
    """Quick-path presets for a directory picker, scanned under the BaboFet base."""
    found = [str(Path(e).expanduser()) for e in extra if Path(e).expanduser().is_dir()]
    b = Path(base).expanduser() if base else None
    if b and b.is_dir():
        globs = {
            "mesh": ["surfaces_on_ref/*/sub-*/hemi-*", "spangy_on_ref/*/sub-*/hemi-*",
                     "msm_inputs/sub-*"],
            "tex": ["sub-*/analysis/*/textures", "sub-*/analysis/*/curvature/*",
                    "sub-*/rois/*", "spangy_on_ref/*/sub-*/hemi-*"],
            "npz": ["spangy_timelapse"],
            "out": ["spangy_timelapse", "figures"],
        }.get(kind, [])
        for g in globs:
            for p in sorted(b.glob(g)):
                if p.is_dir():
                    found.append(str(p))
                if len(found) >= limit:
                    break
        if kind == "mesh":
            pymesh = b.parent.parent / "Pymesh" / "mesh"     # .../dienye.h/Pymesh/mesh
            if pymesh.is_dir():
                found.insert(0, str(pymesh))
    seen, out = set(), []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Mesh <-> texture matching (subject / session / hemisphere)
# --------------------------------------------------------------------------- #
import re  # noqa: E402

_SEP = r"[._\-/]"


def detect_hemisphere(name):
    s = name.lower()
    left = bool(re.search(rf"(^|{_SEP})(left|lh|lhemi|l)({_SEP}|$)", s))
    right = bool(re.search(rf"(^|{_SEP})(right|rh|rhemi|r)({_SEP}|$)", s))
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def parse_entities(name):
    sub = re.search(r"(sub-[A-Za-z0-9]+)", name, flags=re.I)
    ses = re.search(r"(ses-[A-Za-z0-9]+)", name, flags=re.I)
    return (sub.group(1).lower() if sub else None,
            ses.group(1).lower() if ses else None,
            detect_hemisphere(Path(name).name) or detect_hemisphere(name))


def match_score(mesh_name, tex_name):
    matched = possible = 0
    for a, b in zip(parse_entities(mesh_name), parse_entities(tex_name)):
        if a is None or b is None:
            continue
        possible += 1
        if a != b:
            return 0.0
        matched += 1
    return matched / possible if possible else 0.0


def is_roi_name(name):
    return Path(name).name.lower().startswith(ROI_PREFIX)


def auto_match(mesh_name, tex_names, want_roi=False):
    """Best texture (or ROI texture) for a mesh -> (name or None, score, n_tied)."""
    cands = [t for t in tex_names if is_roi_name(t) == want_roi]
    scored = [(t, match_score(mesh_name, t)) for t in cands]
    scored = [(t, s) for t, s in scored if s > 0]
    if not scored:
        return None, 0.0, 0
    best = max(s for _, s in scored)
    tied = [t for t, s in scored if s == best]
    return tied[0], best, len(tied)


def strip_gii(name):
    s = Path(name).name
    for ext in (".surf.gii", ".func.gii", ".shape.gii", ".gii", ".npz"):
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s


# --------------------------------------------------------------------------- #
# Scenes: a list of frames (one per mesh or session), loaded lazily
# --------------------------------------------------------------------------- #
class Scene:
    """frames: dict(name, coords (V,3), faces (F,3), values (V,), roi (V,) bool|None, warning)."""

    def __init__(self, n, loader, title, hemi="left", kind=None, channels=1,
                 coords_all=None, channel_names=None):
        self.n, self._loader, self.title, self.hemi = n, loader, title, hemi
        self.kind, self.channels, self.coords_all = kind, channels, coords_all
        self.channel_names = channel_names
        self._cache = {}

    def __len__(self):
        return self.n

    def frame(self, i):
        i = int(i) % self.n
        if i not in self._cache:
            self._cache[i] = self._loader(i)
        return self._cache[i]

    def bounds(self, indices, rotate180=False):
        if self.coords_all is not None:
            c = self.coords_all[list(indices)].reshape(-1, 3)
            lo, hi = c.min(0), c.max(0)
        else:
            lo = np.min([self.frame(i)["coords"].min(0) for i in indices], axis=0)
            hi = np.max([self.frame(i)["coords"].max(0) for i in indices], axis=0)
        if rotate180:                               # x -> -x, z -> -z
            s = np.array([-1.0, 1.0, -1.0])
            lo, hi = np.minimum(lo * s, hi * s), np.maximum(lo * s, hi * s)
        return lo, hi


def scene_from_npz(path, channel=0):
    z = load_npz(path)
    vals = z["values"]
    if vals.ndim == 2:
        vals = vals[:, :, None]
    c = int(min(max(channel or 0, 0), vals.shape[2] - 1))
    faces = z["faces"].astype(np.int64)
    coords = z["coords"].astype(float)
    roi = z["roi"].astype(bool) if "roi" in z else None
    names = [str(s) for s in z["sessions"]]
    hemi = {"L": "left", "R": "right"}.get(str(z.get("hemi", "L")), "left")
    title = f"{z.get('sub', '')} hemi-{z.get('hemi', '')}"
    if "sulcus" in z:
        title += f" · {z['sulcus']}"

    def loader(i):
        return dict(name=names[i], coords=coords[i], faces=faces, values=vals[i, :, c],
                    roi=None if roi is None else roi[i], warning="")

    kind = str(z["kind"]) if "kind" in z else None
    return Scene(len(names), loader, title.strip(), hemi, kind, vals.shape[2], coords_all=coords)


def scene_from_items(items, names, title, channel=0):
    """items: [(mesh_path, tex_path|None, roi_path|None), ...]"""

    def loader(i):
        mesh, tex, roi = items[i]
        coords, faces = load_surface(mesh)
        warn = []
        values = np.full(len(coords), np.nan)
        if tex:
            t = load_texture(tex)
            if t.shape[0] == len(coords):
                values = t[:, int(min(max(channel or 0, 0), t.shape[1] - 1))]
            else:
                warn.append(f"texture has {t.shape[0]} values, mesh has {len(coords)} "
                            f"vertices: not applied")
        roi_mask = None
        if roi:
            r = load_texture(roi)
            if r.shape[0] == len(coords):
                roi_mask = r[:, 0] > 0.5
            else:
                warn.append(f"ROI has {r.shape[0]} values, mesh has {len(coords)}: ignored")
        return dict(name=names[i], coords=coords, faces=faces, values=values, roi=roi_mask,
                    warning="; ".join(warn))

    hemi = detect_hemisphere(items[0][0]) or "left" if items else "left"
    return Scene(len(items), loader, title, hemi)


def auto_kind(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "label"
    if np.allclose(v, np.round(v)) and v.min() >= BAND_MIN - 0.5 and v.max() <= BAND_MAX + 0.5:
        return "label"
    return "continuous"


def resolve_kind(st, scene, frame):
    if st.get("kind") in ("label", "continuous"):
        return st["kind"]
    return scene.kind if scene.kind in ("label", "continuous") else auto_kind(frame["values"])


def continuous_range(scene, indices, st):
    vmin, vmax = st.get("vmin"), st.get("vmax")
    if vmin is not None and vmax is not None and vmax > vmin:
        return float(vmin), float(vmax)
    chunks = []
    for i in indices:
        f = scene.frame(i)
        v = np.asarray(f["values"], float)
        m = np.isfinite(v) if f["roi"] is None else (np.isfinite(v) & f["roi"])
        chunks.append(v[m])
    allv = np.concatenate(chunks) if chunks else np.array([])
    if allv.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(allv, [2, 98])
    if hi <= lo:
        hi = lo + 1e-9
    lo = float(vmin) if vmin is not None else float(lo)
    hi = float(vmax) if vmax is not None else float(hi)
    return lo, max(hi, lo + 1e-12)


# --------------------------------------------------------------------------- #
# Colours and per-frame render data (shared by Plotly, PyVista and matplotlib)
# --------------------------------------------------------------------------- #
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lighten(h, a=0.5):
    return "#%02x%02x%02x" % tuple(int(round(c + (255 - c) * a)) for c in hex_to_rgb(h))


def band_color(b, gyri_light=False):
    c = BAND_COLORS.get(b, UNSELECTED_COLOR)
    return lighten(c) if gyri_light and b > 0 else c


def colored_bands(st):
    """Bands drawn in colour: show-mode filter, then the checklist if 'only checked'."""
    mode = st.get("show", "both")
    bands = [b for b in ALL_BANDS
             if mode == "both" or (mode == "sulci" and b < 0) or (mode == "gyri" and b > 0)]
    if st.get("only_checked", True):
        checked = set(int(b) for b in (st.get("bands") or []))
        bands = [b for b in bands if b in checked]
    return bands


def discrete_colorscale(bands, gyri_light=False):
    span = CMAX - CMIN
    scale = []
    for k in range(BAND_MIN, BAND_MAX + 1):
        color = band_color(k, gyri_light) if k in bands else UNSELECTED_COLOR
        scale.append([round(((k - 0.5) - CMIN) / span, 6), color])
        scale.append([round(((k + 0.5) - CMIN) / span, 6), color])
    scale[0][0], scale[-1][0] = 0.0, 1.0
    return scale


def mpl_cmap(name):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return matplotlib.colormaps["viridis"]


def mpl_colorscale(name, n=11):
    cmap = mpl_cmap(name)
    return [[i / (n - 1), "rgb(%d,%d,%d)" % tuple(int(255 * c) for c in cmap(i / (n - 1))[:3])]
            for i in range(n)]


def face_majority(vals, faces):
    """Per-face value: the value shared by >= 2 of its vertices (else the first)."""
    a, b, c = vals[faces[:, 0]], vals[faces[:, 1]], vals[faces[:, 2]]
    out = a.copy()
    m = b == c
    out[m] = b[m]
    return out


def face_areas(coords, faces):
    p = coords[faces]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def render_data(frame, st, kind, crange):
    """Everything a renderer needs for one frame."""
    v = np.asarray(frame["values"], float)
    faces = frame["faces"]
    coords = np.asarray(frame["coords"], float)
    if st.get("rotate180"):
        coords = coords * np.array([-1.0, 1.0, -1.0])
    inside_v = np.ones(len(v), bool) if frame["roi"] is None else frame["roi"].copy()
    if kind == "continuous":
        inside_v &= np.isfinite(v)
    fin = inside_v[faces].all(axis=1)
    rd = dict(coords=coords, faces_in=faces[fin], faces_out=faces[~fin], kind=kind,
              name=frame["name"], has_roi=frame["roi"] is not None)
    gl = bool(st.get("gyri_light"))
    if kind == "label":
        bands = colored_bands(st)
        rv = np.clip(np.where(np.isfinite(v), np.round(v), 0), BAND_MIN, BAND_MAX)
        fv = face_majority(rv, faces)[fin]
        shown = np.where(np.isin(fv, bands), fv, 0)
        rd["face_values"] = shown
        rgb = np.tile(np.array(hex_to_rgb(UNSELECTED_COLOR), np.uint8), (len(shown), 1))
        for b in bands:
            rgb[shown == b] = hex_to_rgb(band_color(b, gl))
        rd["face_rgb_in"] = rgb
        area = face_areas(coords, rd["faces_in"])
        tot = float(area.sum())
        rd["legend"] = [(b, band_color(b, gl), 100.0 * float(area[shown == b].sum()) / tot
                         if tot else 0.0) for b in bands]
        rd["colorscale"] = discrete_colorscale(bands, gl)
    else:
        vmin, vmax = crange
        vv = np.where(np.isfinite(v), v, vmin)
        t = np.clip((vv - vmin) / (vmax - vmin), 0, 1)
        vrgb = (mpl_cmap(st.get("cmap", "viridis"))(t)[:, :3] * 255).astype(np.uint8)
        rd["vertex_values"] = vv
        rd["face_rgb_in"] = vrgb[rd["faces_in"]].mean(axis=1).astype(np.uint8)
        rd["colorscale"] = mpl_colorscale(st.get("cmap", "viridis"))
        rd["crange"] = (vmin, vmax)
    return rd


def face_geometry(rd, hide_outside):
    """(faces, rgb uint8) to draw, outside-ROI faces grey or dropped."""
    if hide_outside or len(rd["faces_out"]) == 0:
        return rd["faces_in"], rd["face_rgb_in"]
    grey = np.tile(np.array(hex_to_rgb(OUTSIDE_COLOR), np.uint8), (len(rd["faces_out"]), 1))
    return (np.vstack([rd["faces_in"], rd["faces_out"]]),
            np.vstack([rd["face_rgb_in"], grey]))


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
VIEWS = {
    "medial":    dict(eye=(1.8, 0.0, 0.0), up=(0, 0, 1), hemi_dependent=True),
    "lateral":   dict(eye=(-1.8, 0.0, 0.0), up=(0, 0, 1), hemi_dependent=True),
    "anterior":  dict(eye=(0.0, 1.8, 0.0), up=(0, 0, 1), hemi_dependent=False),
    "posterior": dict(eye=(0.0, -1.8, 0.0), up=(0, 0, 1), hemi_dependent=False),
    "superior":  dict(eye=(0.0, 0.0, 1.8), up=(0, 1, 0), hemi_dependent=False),
    "inferior":  dict(eye=(0.0, 0.0, -1.8), up=(0, 1, 0), hemi_dependent=False),
}


def get_camera(name, hemisphere="left", flip_vertical=False, rotate180_y=False):
    """Plotly camera for a named view (mirrored for right hemispheres)."""
    view = VIEWS.get(name, VIEWS["medial"])
    eye, up = list(view["eye"]), list(view["up"])
    if view["hemi_dependent"] and hemisphere == "right":
        eye[0] = -eye[0]
    if rotate180_y:
        eye[0], eye[2] = -eye[0], -eye[2]
        up[0], up[2] = -up[0], -up[2]
    if flip_vertical:
        up = [-u for u in up]
    return dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), center=dict(x=0, y=0, z=0),
                up=dict(x=up[0], y=up[1], z=up[2]))


def camera_world(cam, bounds, margin=1.15):
    """Plotly camera (scene units) -> world position, focal point, up vector."""
    lo, hi = (np.asarray(b, float) for b in bounds)
    center, size = (lo + hi) / 2, float((hi - lo).max())
    vec = lambda d, dflt: np.array([float(d.get(k, x)) for k, x in zip("xyz", dflt)])  # noqa: E731
    eye = vec(cam.get("eye", {}), (1.8, 0, 0))
    up = vec(cam.get("up", {}), (0, 0, 1))
    focal = center + vec(cam.get("center", {}), (0, 0, 0)) * size
    return focal + eye * size * margin, focal, up


def scene_camera(st, hemi, camera=None):
    return camera or get_camera(st.get("camera", "medial"), hemi,
                                st.get("flip", False), st.get("rotate180", False))


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
def plotly_traces(rd, st, legend=True):
    import plotly.graph_objs as go
    x, y, z = rd["coords"].T
    fi, fo = rd["faces_in"], rd["faces_out"]
    common = dict(x=x, y=y, z=z, flatshading=False,
                  lighting=dict(ambient=0.45, diffuse=0.7, specular=0.1, roughness=0.6),
                  lightposition=dict(x=100, y=200, z=300))
    if rd["kind"] == "label":
        inside = go.Mesh3d(i=fi[:, 0], j=fi[:, 1], k=fi[:, 2], intensity=rd["face_values"],
                           intensitymode="cell", colorscale=rd["colorscale"], cmin=CMIN,
                           cmax=CMAX, showscale=False, name="bands",
                           hovertemplate="Band: %{intensity:.0f}<extra></extra>", **common)
    else:
        vmin, vmax = rd["crange"]
        inside = go.Mesh3d(i=fi[:, 0], j=fi[:, 1], k=fi[:, 2], intensity=rd["vertex_values"],
                           intensitymode="vertex", colorscale=rd["colorscale"], cmin=vmin,
                           cmax=vmax, showscale=True, name="value",
                           colorbar=dict(len=0.6, thickness=18,
                                         title=dict(text=st.get("value_label", ""))),
                           hovertemplate="%{intensity:.4g}<extra></extra>", **common)
    outside = go.Mesh3d(i=fo[:, 0], j=fo[:, 1], k=fo[:, 2], color=OUTSIDE_COLOR,
                        hoverinfo="skip", name="outside",
                        visible=st.get("roi_outside", "grey") != "hide", **common)
    traces = [inside, outside]
    if legend and rd["kind"] == "label":
        for b, color, pct in rd["legend"]:
            traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                                       marker=dict(size=10, color=color),
                                       name=f"B{b}  ({pct:.1f}%)" if pct is not None else f"B{b}",
                                       showlegend=True, hoverinfo="skip"))
    return traces


def plotly_layout(bounds, camera, title=None, uirevision=None, height=850):
    lo, hi = (np.asarray(b, float) for b in bounds)
    pad = 0.02 * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    ar = (hi - lo) / float((hi - lo).max())
    axis = lambda a: dict(visible=False, range=[float(lo[a]), float(hi[a])])  # noqa: E731
    layout = dict(
        scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode="manual",
                   aspectratio=dict(x=float(ar[0]), y=float(ar[1]), z=float(ar[2])),
                   camera=camera, uirevision=uirevision),
        height=height, margin=dict(l=10, r=10, b=10, t=60 if title else 10),
        showlegend=True, uirevision=uirevision,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.8)"),
        paper_bgcolor="white")
    if title:
        layout["title"] = dict(text=title, x=0.5, y=0.97, xanchor="center", yanchor="top",
                               font=dict(size=18))
    return layout


def plotly_animation(scene, st, kind, crange, bounds, camera, ms=500):
    """Figure with one Plotly frame per session/mesh, play button and slider."""
    import plotly.graph_objs as go
    rds = [render_data(scene.frame(i), st, kind, crange) for i in range(len(scene))]
    first = plotly_traces(rds[0], st, legend=False)
    legend = []
    if kind == "label":
        legend = [go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                               marker=dict(size=10, color=c), name=f"B{b}", hoverinfo="skip")
                  for b, c, _ in rds[0]["legend"]]
    frames = [go.Frame(data=plotly_traces(rd, st, legend=False), traces=[0, 1], name=str(i),
                       layout=dict(title=dict(text=f"{scene.title} — {rd['name']}")))
              for i, rd in enumerate(rds)]
    fig = go.Figure(data=first + legend, frames=frames)
    fig.update_layout(**plotly_layout(bounds, camera, f"{scene.title} — {rds[0]['name']}"))
    play = dict(frame=dict(duration=ms, redraw=True), fromcurrent=True,
                transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0.02, y=0.06, xanchor="left", direction="left",
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])],
        sliders=[dict(x=0.15, len=0.8, y=0.02, currentvalue=dict(prefix="Frame: "),
                      steps=[dict(label=rd["name"], method="animate",
                                  args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                                       mode="immediate")])
                             for i, rd in enumerate(rds)])])
    return fig


# --------------------------------------------------------------------------- #
# Offscreen renderers for PNG / GIF / MP4 / MOV
# --------------------------------------------------------------------------- #
class PyVistaRenderer:
    def __init__(self, size, cam_world, off_screen=True):
        import pyvista as pv
        self.pv = pv
        self.pl = pv.Plotter(off_screen=off_screen, window_size=list(size))
        self.pl.set_background("white")
        self.cam = cam_world

    def draw(self, rd, st, title, legend=None, set_camera=True):
        faces, rgb = face_geometry(rd, st.get("roi_outside") == "hide")
        poly = self.pv.PolyData(rd["coords"],
                                np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
        poly.cell_data["rgb"] = rgb
        self.pl.add_mesh(poly, scalars="rgb", rgb=True, name="brain", show_scalar_bar=False,
                         preference="cell", reset_camera=False)
        self.pl.add_text(title, name="title", font_size=11, color="black",
                         position="upper_edge")
        if legend:
            try:
                self.pl.add_legend(labels=legend, bcolor="white", face="rectangle",
                                   name="legend", size=(0.14, 0.2))
            except Exception:
                pass
        if set_camera:
            pos, focal, up = self.cam
            self.pl.camera_position = [tuple(pos), tuple(focal), tuple(up)]
            self.pl.camera.view_angle = 30.0

    def render(self, rd, st, title, legend=None):
        self.draw(rd, st, title, legend)
        return np.asarray(self.pl.screenshot(return_img=True))[..., :3]

    def close(self):
        self.pl.close()


class PlotlyRenderer:
    def __init__(self, size, camera, bounds):
        import plotly.graph_objs as go  # noqa: F401
        self.size, self.camera, self.bounds = size, camera, bounds

    def render(self, rd, st, title, legend=None):
        import io
        import plotly.graph_objs as go
        from PIL import Image
        fig = go.Figure(plotly_traces(rd, st, legend=bool(legend)))
        fig.update_layout(**plotly_layout(self.bounds, self.camera, title, height=self.size[1]))
        png = fig.to_image(format="png", width=self.size[0], height=self.size[1])
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def close(self):
        pass


class MatplotlibRenderer:
    """Fallback: slower, z-up only, but needs nothing beyond matplotlib."""

    def __init__(self, size, cam_world, bounds):
        import matplotlib
        matplotlib.use("Agg")
        self.size, self.cam, self.bounds = size, cam_world, bounds

    def render(self, rd, st, title, legend=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        faces, rgb = face_geometry(rd, st.get("roi_outside") == "hide")
        tri = rd["coords"][faces]
        pos, focal, up = self.cam
        d = (pos - focal) / (np.linalg.norm(pos - focal) + 1e-12)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(n @ d)
        cols = np.clip(rgb / 255.0 * shade[:, None], 0, 1)
        w, h = self.size
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_subplot(projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="none",
                                             linewidths=0))
        lo, hi = self.bounds
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
        ax.set_axis_off()
        elev = np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
        azim = np.degrees(np.arctan2(d[1], d[0]))
        try:
            ax.view_init(elev=elev, azim=azim, roll=180 if up[2] < 0 else 0)
        except TypeError:
            ax.view_init(elev=elev, azim=azim)
        fig.suptitle(title, fontsize=max(8, min(14, int(1.6 * w / max(len(title), 1)))))
        if legend:
            ax.legend(handles=[Patch(color=c, label=l) for l, c in legend], loc="upper right")
        fig.subplots_adjust(0, 0, 1, 0.95)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    def close(self):
        pass


class MovieWriter:
    """Streams frames to PNG (first frame), GIF (pillow) or MP4/MOV (imageio-ffmpeg)."""

    def __init__(self, path, fmt, fps):
        self.path, self.fmt, self.fps = str(path), fmt, max(float(fps), 0.1)
        self.gif, self.writer, self.n = [], None, 0

    @staticmethod
    def _pad16(img):
        h, w = img.shape[:2]
        H, W = -(-h // 16) * 16, -(-w // 16) * 16
        if (H, W) == (h, w):
            return img
        out = np.full((H, W, 3), 255, np.uint8)
        out[:h, :w] = img
        return out

    def add(self, img):
        img = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
        self.n += 1
        if self.fmt == "png":
            if self.n == 1:
                from PIL import Image
                Image.fromarray(img).save(self.path)
        elif self.fmt == "gif":
            from PIL import Image
            self.gif.append(Image.fromarray(img).quantize(
                colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        else:
            if self.writer is None:
                import imageio.v2 as imageio
                self.writer = imageio.get_writer(self.path, format="FFMPEG", mode="I",
                                                 fps=self.fps, codec="libx264",
                                                 pixelformat="yuv420p", quality=8,
                                                 macro_block_size=16)
            self.writer.append_data(self._pad16(img))

    def close(self):
        if self.fmt == "gif" and self.gif:
            self.gif[0].save(self.path, save_all=True, append_images=self.gif[1:],
                             duration=int(round(1000 / self.fps)), loop=0, disposal=1)
        if self.writer is not None:
            self.writer.close()


def iter_render_frames(scene, indices, morph=0, hold=1):
    """Frames to render: each selected frame `hold` times, `morph` in-betweens if meshes match."""
    indices = list(indices)
    for n, i in enumerate(indices):
        f = scene.frame(i)
        for _ in range(max(int(hold), 1)):
            yield f
        if morph and n + 1 < len(indices):
            g = scene.frame(indices[n + 1])
            if f["coords"].shape == g["coords"].shape and (
                    f["faces"] is g["faces"] or np.array_equal(f["faces"], g["faces"])):
                for k in range(1, int(morph) + 1):
                    t = k / (morph + 1)
                    src = f if t < 0.5 else g
                    yield dict(src, coords=(1 - t) * f["coords"] + t * g["coords"],
                               name=f"{f['name']} → {g['name']}")


# --------------------------------------------------------------------------- #
# Specs (what the app hands to the export / PyVista subprocesses)
# --------------------------------------------------------------------------- #
def scene_from_spec(spec):
    src = spec["source"]
    if src["type"] == "npz":
        return scene_from_npz(src["path"], src.get("channel", 0))
    return scene_from_items([tuple(x) for x in src["items"]], src["names"], src["title"],
                            src.get("channel", 0))


def legend_entries(kind, st):
    if kind != "label":
        return None
    gl = bool(st.get("gyri_light"))
    return [[f"B{b}", band_color(b, gl)] for b in colored_bands(st)]


def make_renderer(name, size, camera, bounds, log=print):
    order = [name] + [r for r in ("pyvista", "plotly", "matplotlib") if r != name]
    for r in order:
        try:
            if r == "pyvista":
                return r, PyVistaRenderer(size, camera_world(camera, bounds))
            if r == "plotly":
                import kaleido  # noqa: F401
                return r, PlotlyRenderer(size, camera, bounds)
            if r == "matplotlib":
                return r, MatplotlibRenderer(size, camera_world(camera, bounds), bounds)
        except Exception as e:
            log(f"renderer '{r}' unavailable ({e.__class__.__name__}: {e})")
    raise RuntimeError("no renderer available")


def run_export(spec, log=print):
    st = spec["settings"]
    scene = scene_from_spec(spec)
    indices = spec.get("indices") or list(range(len(scene)))
    first = scene.frame(indices[0])
    kind = resolve_kind(st, scene, first)
    crange = continuous_range(scene, indices, st) if kind == "continuous" else None
    bounds = scene.bounds(indices, st.get("rotate180", False))
    camera = scene_camera(st, scene.hemi, spec.get("camera"))
    size = tuple(int(x) for x in spec.get("size", (1200, 1000)))
    used, renderer = make_renderer(spec.get("renderer", "pyvista"), size, camera, bounds, log)
    log(f"renderer: {used}; {len(indices)} frame(s), morph {spec.get('morph', 0)}, "
        f"hold {spec.get('hold', 1)} -> {spec['output']}")
    legend = legend_entries(kind, st)
    title_of = lambda fr: (fr["name"] if fr["name"].startswith(scene.title)  # noqa: E731
                           else f"{scene.title} — {fr['name']}")
    writer = MovieWriter(spec["output"], spec["format"], spec.get("fps", 4))
    k = 0
    try:
        frames = iter_render_frames(scene, indices, spec.get("morph", 0), spec.get("hold", 1))
        if spec["format"] == "png":
            frames = [next(iter(frames))]
        for fr in frames:
            rd = render_data(fr, st, kind, crange)
            writer.add(renderer.render(rd, st, title_of(fr), legend))
            k += 1
            if k % 10 == 0:
                log(f"  {k} frames rendered")
    finally:
        writer.close()
        renderer.close()
    log(f"Saved {spec['output']} ({k} frame(s))")
    return spec["output"]


def run_pyvista_window(spec):
    """Interactive desktop timelapse: slider, play button / space, arrows, 's' screenshot."""
    import pyvista as pv
    st = spec["settings"]
    scene = scene_from_spec(spec)
    n = len(scene)
    idx = list(range(n))
    kind = resolve_kind(st, scene, scene.frame(0))
    crange = continuous_range(scene, idx, st) if kind == "continuous" else None
    bounds = scene.bounds(idx, st.get("rotate180", False))
    camera = scene_camera(st, scene.hemi, spec.get("camera"))
    fps = float(spec.get("fps", 2))
    out_dir = Path(spec.get("out_dir") or ".").expanduser()
    size = tuple(int(x) for x in spec.get("size", (1200, 1000)))
    r = PyVistaRenderer(size, camera_world(camera, bounds), off_screen=False)
    pl = r.pl
    legend = legend_entries(kind, st)
    state = dict(i=int(spec.get("start", 0)) % n, playing=False, slider=None, button=None,
                 first=True)

    def show(i):
        i = int(i) % n
        state["i"] = i
        rd = render_data(scene.frame(i), st, kind, crange)
        # camera set once; afterwards your own rotation/zoom is kept while stepping
        r.draw(rd, st, f"{scene.title} — {rd['name']}   [{i + 1}/{n}]", legend,
               set_camera=state["first"])
        state["first"] = False
        if kind == "continuous":
            pl.add_text(f"{st.get('cmap')}: {crange[0]:.3g} … {crange[1]:.3g}",
                        name="range", position="lower_right", font_size=9, color="black")
        if state["slider"] is not None:
            state["slider"].GetRepresentation().SetValue(i)
        pl.render()

    def set_playing(flag):
        state["playing"] = bool(flag)
        if state["button"] is not None:
            state["button"].GetRepresentation().SetState(int(state["playing"]))

    def screenshot():
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{strip_gii(spec.get('stem', 'pyvista'))}_{state['i']}_{stamp}.png"
        pl.screenshot(str(path))
        print(f"Saved {path}")

    if n > 1:
        state["slider"] = pl.add_slider_widget(
            lambda v: int(round(v)) != state["i"] and show(int(round(v))),
            rng=[0, n - 1], value=state["i"], title="Session / mesh", fmt="%.0f",
            pointa=(0.25, 0.07), pointb=(0.75, 0.07), style="modern")
        state["button"] = pl.add_checkbox_button_widget(set_playing, value=False,
                                                        position=(10, 10), size=36,
                                                        color_on="green", color_off="grey")
        pl.add_text("play", position=(52, 18), font_size=9, color="black", name="playlabel")
        pl.add_key_event("space", lambda: set_playing(not state["playing"]))
        pl.add_key_event("Right", lambda: show(state["i"] + 1))
        pl.add_key_event("Left", lambda: show(state["i"] - 1))
        try:
            pl.add_timer_event(max_steps=10 ** 9, duration=int(1000 / max(fps, 0.1)),
                               callback=lambda step: state["playing"] and show(state["i"] + 1))
        except Exception as e:
            print(f"Autoplay unavailable in this PyVista version ({e}); use the slider/arrows.")
    pl.add_key_event("s", screenshot)
    show(state["i"])
    pl.show(title=f"SPANGY — {scene.title}")


# --------------------------------------------------------------------------- #
# Dash app
# --------------------------------------------------------------------------- #
JOBS = []          # export subprocesses launched by this app (single local user)

SETTING_INPUTS = [
    ("source", "source-radio", "value"),
    ("mesh_dir", "meshdir-dir", "value"), ("mesh_filter", "mesh-filter", "value"),
    ("mesh_recursive", "mesh-recursive", "value"), ("mesh_name", "mesh-dd", "value"),
    ("mesh_list", "mesh-dd", "options"),
    ("tex_dir", "texdir-dir", "value"), ("tex_filter", "tex-filter", "value"),
    ("tex_recursive", "tex-recursive", "value"), ("tex_choice", "tex-dd", "value"),
    ("roi_choice", "roi-dd", "value"),
    ("npz_dir", "npzdir-dir", "value"), ("npz_name", "npz-dd", "value"),
    ("session", "ses-slider", "value"), ("channel", "channel-dd", "value"),
    ("kind", "kind-radio", "value"), ("show", "show-radio", "value"),
    ("bands", "band-check", "value"), ("display_opts", "display-opts", "value"),
    ("roi_outside", "roi-outside", "value"), ("cmap", "cmap-dd", "value"),
    ("vmin", "vmin-input", "value"), ("vmax", "vmax-input", "value"),
    ("camera", "camera-radio", "value"), ("orient", "orient-opts", "value"),
]


def clean_settings(raw):
    st = dict(raw)
    st["mesh_list"] = [o["value"] if isinstance(o, dict) else o for o in (raw.get("mesh_list") or [])]
    st["mesh_recursive"] = "on" in (raw.get("mesh_recursive") or [])
    st["tex_recursive"] = "on" in (raw.get("tex_recursive") or [])
    opts = raw.get("display_opts") or []
    st["only_checked"] = "only" in opts
    st["gyri_light"] = "light" in opts
    st["fixed_scale"] = "fixed" in opts
    orient = raw.get("orient") or []
    st["rotate180"], st["flip"] = "rot180" in orient, "flip" in orient
    st.pop("display_opts", None)
    st.pop("orient", None)
    for k in ("vmin", "vmax"):
        st[k] = None if st.get(k) in ("", None) else float(st[k])
    return st


def _join(d, name):
    return str(Path(d).expanduser() / name)


def resolve_items(st, sequence=False):
    """Files mode -> (items, names, current index, info string)."""
    mesh_dir, tex_dir = st.get("mesh_dir"), st.get("tex_dir")
    current = st.get("mesh_name")
    meshes = st.get("mesh_list") or ([current] if current else [])
    if not current:
        return [], [], 0, "No mesh selected."
    seq = meshes[:MAX_SEQUENCE] if sequence else [current]
    if current not in seq:
        seq = [current]
    all_tex = list_files(tex_dir, recursive=st.get("tex_recursive"))
    tex_cands = [t for t in list_files(tex_dir, recursive=st.get("tex_recursive"),
                                       pattern=st.get("tex_filter")) if not is_roi_name(t)]
    items, info = [], ""
    for m in seq:
        tc, rc = st.get("tex_choice") or AUTO, st.get("roi_choice") or AUTO
        if tc == NONE:
            tex = None
        elif tc != AUTO and m == current:
            tex = tc
        else:
            tex, score, tied = auto_match(m, tex_cands)
            if m == current:
                info = (f"Texture: {tex} (score {score:.2f}{f', {tied} tied' if tied > 1 else ''})"
                        if tex else "No texture matches subject/session/hemisphere.")
        if rc == NONE:
            roi = None
        elif rc != AUTO and m == current:
            roi = rc
        else:
            roi = auto_match(m, all_tex, want_roi=True)[0]
        if m == current:
            if tc not in (AUTO, NONE):
                info = f"Texture: {tc} (match score {match_score(m, tc):.2f})"
            info += f" · ROI: {roi}" if roi else ""
        items.append((_join(mesh_dir, m), _join(tex_dir, tex) if tex else None,
                      _join(tex_dir, roi) if roi else None))
    return items, [strip_gii(m) for m in seq], seq.index(current), info


def scene_from_settings(st, sequence=False):
    """-> (scene, current index, info, stem) for the app's current state."""
    if st.get("source") == "npz":
        if not st.get("npz_name"):
            return None, 0, "No timelapse file selected.", ""
        path = _join(st["npz_dir"], st["npz_name"])
        sc = scene_from_npz(path, st.get("channel") or 0)
        i = int(st.get("session") or 0) % len(sc)
        info = f"{len(sc)} sessions · {sc.kind or 'auto'} · {sc.channels} channel(s)"
        return sc, i, info, f"{strip_gii(st['npz_name'])}_ch{st.get('channel') or 0}"
    items, names, i, info = resolve_items(st, sequence)
    if not items:
        return None, 0, info, ""
    title = names[i] if not sequence else Path(st["mesh_dir"]).name
    sc = scene_from_items(items, names, title, st.get("channel") or 0)
    stem = names[i] if not sequence else f"sequence_{Path(st['mesh_dir']).name}_{len(names)}"
    return sc, i, info, stem


def spec_source(st, sequence):
    if st.get("source") == "npz":
        return dict(type="npz", path=_join(st["npz_dir"], st["npz_name"]),
                    channel=st.get("channel") or 0)
    items, names, i, _ = resolve_items(st, sequence)
    title = names[i] if not sequence else Path(st["mesh_dir"]).name
    return dict(type="files", items=items, names=names, title=title,
                channel=st.get("channel") or 0), i


def build_app(args):
    import dash
    import plotly.graph_objs as go
    from dash import dcc, html, Input, Output, State, no_update

    label_style = {"fontWeight": "600", "marginTop": "12px", "marginBottom": "4px",
                   "display": "block"}
    btn = {"marginRight": "6px", "marginTop": "6px", "padding": "5px 9px", "cursor": "pointer"}
    hint = {"color": "#666", "fontSize": "12px", "marginTop": "4px", "wordBreak": "break-all"}
    box = {"width": "100%", "boxSizing": "border-box", "fontSize": "12px", "padding": "4px"}
    section = lambda title, *ch, **kw: html.Details(  # noqa: E731
        [html.Summary(title, style={"fontWeight": "700", "cursor": "pointer",
                                    "marginTop": "10px"})] + list(ch), open=True, **kw)
    presets = {k: discover_presets(args.base, k, args.preset) for k in ("mesh", "tex", "npz", "out")}

    def dir_picker(prefix, label, initial):
        return html.Div([
            html.Label(label, style=label_style),
            dcc.Input(id=f"{prefix}-dir", type="text", value=initial, debounce=True,
                      placeholder="/path/to/folder", style=box),
            html.Div([html.Button("↑ Up", id=f"{prefix}-up", n_clicks=0, style=btn),
                      html.Button("Home", id=f"{prefix}-home", n_clicks=0, style=btn)]),
            dcc.Dropdown(id=f"{prefix}-subdirs", placeholder="Quick paths / sub-folders…",
                         options=[], value=None, style={"marginTop": "4px"}, optionHeight=40),
            html.Div(id=f"{prefix}-count", style=hint),
        ])

    app = dash.Dash(__name__, title="SPANGY viewer hub")
    panel = {"width": "360px", "padding": "12px", "fontFamily": "sans-serif",
             "fontSize": "13px", "borderRight": "1px solid #ddd", "height": "100vh",
             "overflowY": "auto", "boxSizing": "border-box"}

    app.layout = html.Div(style={"display": "flex", "margin": "0"}, children=[
        html.Div(style=panel, children=[
            html.H3("SPANGY viewer hub", style={"marginTop": "0"}),
            dcc.RadioItems(id="source-radio", value="files", inline=True,
                           options=[{"label": " Mesh + texture files ", "value": "files"},
                                    {"label": " Timelapse (.npz)", "value": "npz"}],
                           inputStyle={"marginRight": "4px"}),

            html.Div(id="files-section", children=[
                dir_picker("meshdir", "Mesh directory", str(args.mesh_dir)),
                dcc.Input(id="mesh-filter", type="text", value="", debounce=True, style=box,
                          placeholder="filter, e.g. sub-Borgne hemi_left !inflated"),
                dcc.Checklist(id="mesh-recursive", value=[],
                              options=[{"label": " search sub-folders", "value": "on"}]),
                dir_picker("texdir", "Texture directory", str(args.tex_dir)),
                dcc.Input(id="tex-filter", type="text", value="", debounce=True, style=box,
                          placeholder="filter, e.g. dom_band  or  sGI_s4"),
                dcc.Checklist(id="tex-recursive", value=[],
                              options=[{"label": " search sub-folders", "value": "on"}]),
                html.Label("Mesh", style=label_style),
                dcc.Dropdown(id="mesh-dd", options=[], value=None, clearable=False,
                             optionHeight=48),
                html.Label("Texture", style=label_style),
                dcc.Dropdown(id="tex-dd", options=[], value=AUTO, clearable=False,
                             optionHeight=48),
                html.Label("ROI (sulcus mask)", style=label_style),
                dcc.Dropdown(id="roi-dd", options=[], value=AUTO, clearable=False,
                             optionHeight=48),
                html.Div(id="match-info", style=hint),
            ]),
            html.Div(id="npz-section", style={"display": "none"}, children=[
                dir_picker("npzdir", "Timelapse directory", str(args.npz_dir)),
                html.Label("Timelapse file", style=label_style),
                dcc.Dropdown(id="npz-dd", options=[], value=None, clearable=False,
                             optionHeight=48),
                html.Div(id="npz-info", style=hint),
            ]),
            html.Div(id="channel-div", style={"display": "none"}, children=[
                html.Label("Texture column", style=label_style),
                dcc.Dropdown(id="channel-dd", options=[{"label": "0", "value": 0}], value=0,
                             clearable=False)]),

            section("Navigate",
                    html.Div([html.Button("◀ Prev", id="btn-prev", n_clicks=0, style=btn),
                              html.Button("▶ Play", id="btn-play", n_clicks=0, style=btn),
                              html.Button("Next ▶", id="btn-next", n_clicks=0, style=btn),
                              html.Span(" fps "),
                              dcc.Input(id="fps-input", type="number", value=1, min=0.2,
                                        max=10, step=0.2, style={"width": "55px"})]),
                    html.Div(id="ses-div", style={"display": "none", "marginTop": "8px"},
                             children=[dcc.Slider(id="ses-slider", min=0, max=0, step=1,
                                                  value=0, marks={})]),
                    html.Div("Files mode cycles through the mesh list (use the filter and "
                             "'search sub-folders' to build it); timelapse mode through "
                             "sessions. Rendering speed sets the real frame rate.",
                             style=hint)),

            section("Display",
                    dcc.RadioItems(id="kind-radio", value="auto", inline=True,
                                   options=[{"label": " auto ", "value": "auto"},
                                            {"label": " bands ", "value": "label"},
                                            {"label": " continuous", "value": "continuous"}],
                                   inputStyle={"marginRight": "4px"}),
                    html.Label("Show", style=label_style),
                    dcc.RadioItems(id="show-radio", value="both", inline=True,
                                   options=[{"label": " Sulci ", "value": "sulci"},
                                            {"label": " Gyri ", "value": "gyri"},
                                            {"label": " Both", "value": "both"}],
                                   inputStyle={"marginRight": "4px"},
                                   labelStyle={"fontSize": "15px", "marginRight": "10px"}),
                    dcc.Checklist(id="band-check", value=DEFAULT_BANDS,
                                  options=[{"label": f" B{b}", "value": b}
                                           for b in sorted(ALL_BANDS, reverse=True)],
                                  style={"columnCount": 2, "marginTop": "6px"},
                                  inputStyle={"marginRight": "4px"}),
                    dcc.Checklist(id="display-opts", value=["only", "fixed"],
                                  options=[{"label": " only checked bands", "value": "only"},
                                           {"label": " gyri in lighter shades", "value": "light"},
                                           {"label": " fixed scale across sessions",
                                            "value": "fixed"}],
                                  inputStyle={"marginRight": "4px"}),
                    html.Label("Outside the ROI", style=label_style),
                    dcc.RadioItems(id="roi-outside", value="grey", inline=True,
                                   options=[{"label": " grey ", "value": "grey"},
                                            {"label": " hidden", "value": "hide"}],
                                   inputStyle={"marginRight": "4px"}),
                    html.Label("Continuous maps", style=label_style),
                    dcc.Dropdown(id="cmap-dd", value="viridis", clearable=False,
                                 options=[{"label": c, "value": c} for c in CMAPS]),
                    html.Div([dcc.Input(id="vmin-input", type="number", placeholder="min (auto)",
                                        debounce=True, style={"width": "48%"}),
                              dcc.Input(id="vmax-input", type="number", placeholder="max (auto)",
                                        debounce=True, style={"width": "48%",
                                                              "marginLeft": "4%"})],
                             style={"marginTop": "4px"})),

            section("Camera",
                    dcc.RadioItems(id="camera-radio", value="medial",
                                   options=[{"label": f" {c}", "value": c} for c in VIEWS],
                                   inline=True, inputStyle={"marginRight": "4px"}),
                    dcc.Checklist(id="orient-opts", value=[],
                                  options=[{"label": " rotate 180° about Y", "value": "rot180"},
                                           {"label": " flip vertical", "value": "flip"}],
                                  inputStyle={"marginRight": "4px"})),

            section("Export",
                    dir_picker("outdir", "Output directory", str(args.out_dir)),
                    html.Div("Quick screenshot: camera icon in the plot toolbar.", style=hint),
                    html.Div([html.Span("renderer "),
                              dcc.Dropdown(id="renderer-dd", value="pyvista", clearable=False,
                                           options=[{"label": r, "value": r} for r in
                                                    ("pyvista", "plotly", "matplotlib")],
                                           style={"width": "150px", "display": "inline-block",
                                                  "verticalAlign": "middle"})]),
                    html.Div([html.Span("size "),
                              dcc.Input(id="width-input", type="number", value=1200,
                                        style={"width": "70px"}), html.Span(" × "),
                              dcc.Input(id="height-input", type="number", value=1000,
                                        style={"width": "70px"})], style={"marginTop": "4px"}),
                    dcc.Checklist(id="use-view", value=["view"], inputStyle={"marginRight": "4px"},
                                  options=[{"label": " use the current (rotated) view",
                                            "value": "view"}]),
                    html.Div([html.Button("Save PNG", id="btn-png", n_clicks=0, style=btn),
                              html.Button("Save HTML", id="btn-html", n_clicks=0, style=btn)]),
                    dcc.Checklist(id="html-opts", value=["full"], inputStyle={"marginRight": "4px"},
                                  options=[{"label": " HTML: animate all frames", "value": "anim"},
                                           {"label": " HTML: embed plotly.js (offline)",
                                            "value": "full"}]),
                    html.Div([html.Span("timelapse "),
                              dcc.Dropdown(id="fmt-dd", value="mp4", clearable=False,
                                           options=[{"label": f.upper(), "value": f}
                                                    for f in ("gif", "mp4", "mov")],
                                           style={"width": "90px", "display": "inline-block",
                                                  "verticalAlign": "middle"}),
                              html.Span(" fps "),
                              dcc.Input(id="fps-out", type="number", value=4, min=0.2, step=0.5,
                                        style={"width": "55px"})], style={"marginTop": "6px"}),
                    html.Div([html.Span("morph steps "),
                              dcc.Input(id="morph-input", type="number", value=0, min=0, max=30,
                                        style={"width": "50px"}),
                              html.Span(" hold "),
                              dcc.Input(id="hold-input", type="number", value=1, min=1, max=60,
                                        style={"width": "50px"})], style={"marginTop": "4px"}),
                    html.Div([html.Button("Export timelapse", id="btn-movie", n_clicks=0,
                                          style=btn),
                              html.Button("Open in PyVista", id="btn-pyvista", n_clicks=0,
                                          style=btn)]),
                    html.Div(id="export-status", style={"marginTop": "8px", "fontSize": "12px",
                                                        "color": "#0a6",
                                                        "wordBreak": "break-all"}),
                    html.Div(id="job-status", style={"marginTop": "4px", "fontSize": "12px",
                                                     "color": "#555", "whiteSpace": "pre-wrap",
                                                     "wordBreak": "break-all"})),
            dcc.Interval(id="play-interval", interval=1000, disabled=True),
            dcc.Interval(id="job-interval", interval=1500),
            dcc.Store(id="settings-store"),
            dcc.Store(id="camera-store"),
        ]),
        html.Div(style={"flex": "1", "padding": "10px"}, children=[
            html.Div(id="view-status", style={"fontFamily": "sans-serif", "fontSize": "13px",
                                              "color": "#b00", "minHeight": "18px"}),
            dcc.Graph(id="mesh-graph", style={"height": "90vh"}, config={"displaylogo": False}),
        ]),
    ])

    # -- directory pickers ------------------------------------------------------ #
    picker_kinds = {"meshdir": ("mesh", (".gii",)), "texdir": ("tex", (".gii",)),
                    "npzdir": ("npz", (".npz",)), "outdir": ("out", ())}
    for prefix, (kind, exts) in picker_kinds.items():

        @app.callback(Output(f"{prefix}-subdirs", "options"), Output(f"{prefix}-subdirs", "value"),
                      Output(f"{prefix}-count", "children"), Input(f"{prefix}-dir", "value"))
        def refresh_dir(path, _kind=kind, _exts=exts):
            opts = [{"label": f"📌 {p}", "value": p} for p in presets[_kind]]
            opts += [{"label": f"📁 {d}", "value": d} for d in list_subdirs(path)]
            ok = bool(path) and Path(path).expanduser().is_dir()
            if _kind == "out":
                note = "Exports go here." if ok else "Will be created on export."
            elif not ok:
                note = "Folder not found."
            else:
                note = f"{len(list_files(path, _exts))} file(s) here."
            return opts, None, note

        @app.callback(Output(f"{prefix}-dir", "value"), Input(f"{prefix}-subdirs", "value"),
                      Input(f"{prefix}-up", "n_clicks"), Input(f"{prefix}-home", "n_clicks"),
                      State(f"{prefix}-dir", "value"), prevent_initial_call=True)
        def navigate(sub, _u, _h, current, _p=prefix):
            trig = dash.ctx.triggered_id
            if trig == f"{_p}-home":
                return str(Path.home())
            base = Path(current).expanduser() if current else Path.home()
            if trig == f"{_p}-up":
                return str(base.parent)
            if sub:
                return sub if os.path.isabs(sub) else str(base / sub)
            return no_update

    # -- source switch ---------------------------------------------------------- #
    @app.callback(Output("files-section", "style"), Output("npz-section", "style"),
                  Output("ses-div", "style"), Input("source-radio", "value"))
    def switch_source(src):
        show, hide = {"display": "block"}, {"display": "none"}
        return (show, hide, hide) if src == "files" else (hide, show, {"display": "block",
                                                                       "marginTop": "8px"})

    # -- files mode lists ------------------------------------------------------- #
    @app.callback(Output("mesh-dd", "options"), Output("mesh-dd", "value"),
                  Input("meshdir-dir", "value"), Input("mesh-filter", "value"),
                  Input("mesh-recursive", "value"), State("mesh-dd", "value"))
    def mesh_options(path, filt, rec, current):
        found = list_files(path, recursive="on" in (rec or []), pattern=filt)
        value = current if current in found else (found[0] if found else None)
        return [{"label": m, "value": m} for m in found], value

    @app.callback(Output("tex-dd", "options"), Output("tex-dd", "value"),
                  Output("roi-dd", "options"), Output("roi-dd", "value"),
                  Input("texdir-dir", "value"), Input("tex-filter", "value"),
                  Input("tex-recursive", "value"), Input("mesh-dd", "value"),
                  State("tex-dd", "value"), State("roi-dd", "value"))
    def tex_options(path, filt, rec, _mesh, tex_cur, roi_cur):
        rec = "on" in (rec or [])
        allf = list_files(path, recursive=rec)
        found = [t for t in list_files(path, recursive=rec, pattern=filt) if not is_roi_name(t)]
        topts = [{"label": "Auto-match", "value": AUTO},
                 {"label": "None (plain surface)", "value": NONE}] + \
                [{"label": t, "value": t} for t in found]
        ropts = [{"label": "Auto (roi_used_* of this mesh)", "value": AUTO},
                 {"label": "None", "value": NONE}] + \
                [{"label": t, "value": t} for t in allf]
        if dash.ctx.triggered_id == "mesh-dd":
            tval = AUTO
            rval = roi_cur if roi_cur in (AUTO, NONE) else AUTO
        else:
            tval = tex_cur if tex_cur in (AUTO, NONE) or tex_cur in found else AUTO
            rval = roi_cur if roi_cur in (AUTO, NONE) or roi_cur in allf else AUTO
        return topts, tval, ropts, rval

    # -- timelapse mode lists --------------------------------------------------- #
    @app.callback(Output("npz-dd", "options"), Output("npz-dd", "value"),
                  Input("npzdir-dir", "value"), State("npz-dd", "value"))
    def npz_options(path, current):
        found = list_files(path, (".npz",))
        value = current if current in found else (found[0] if found else None)
        return [{"label": f, "value": f} for f in found], value

    @app.callback(Output("ses-slider", "max"), Output("ses-slider", "marks"),
                  Output("ses-slider", "value"), Output("npz-info", "children"),
                  Input("npz-dd", "value"), State("npzdir-dir", "value"))
    def npz_selected(name, d):
        if not name:
            return 0, {}, 0, "No .npz here (build_spangy_timelapse.py writes them)."
        try:
            z = load_npz(_join(d, name))
        except Exception as e:
            return 0, {}, 0, f"Cannot read: {e}"
        ses = [str(s) for s in z["sessions"]]
        step = max(1, len(ses) // 10)
        marks = {i: s.replace("ses-", "") for i, s in enumerate(ses) if i % step == 0}
        roi = " · with sulcus ROI" if "roi" in z else ""
        sulc = f" ({z['sulcus']})" if "sulcus" in z else ""
        return len(ses) - 1, marks, 0, (f"{len(ses)} sessions, {z['values'].shape[1]} vertices, "
                                        f"kind {z.get('kind', '?')}{roi}{sulc}, ref {z.get('ref', '?')}")

    # -- texture columns (e.g. 7 band powers) ----------------------------------- #
    @app.callback(Output("channel-dd", "options"), Output("channel-dd", "value"),
                  Output("channel-div", "style"),
                  Input("source-radio", "value"), Input("npz-dd", "value"),
                  Input("mesh-dd", "value"), Input("tex-dd", "value"),
                  State("npzdir-dir", "value"), State("meshdir-dir", "value"),
                  State("texdir-dir", "value"), State("tex-filter", "value"),
                  State("tex-recursive", "value"), State("channel-dd", "value"))
    def channels(src, npz, mesh, tex, npz_dir, mesh_dir, tex_dir, filt, rec, cur):
        n = 1
        try:
            if src == "npz" and npz:
                v = load_npz(_join(npz_dir, npz))["values"]
                n = v.shape[2] if v.ndim == 3 else 1
            elif src == "files" and mesh and tex != NONE:
                name = tex
                if tex in (AUTO, None):
                    cands = [t for t in list_files(tex_dir, recursive="on" in (rec or []),
                                                   pattern=filt) if not is_roi_name(t)]
                    name = auto_match(mesh, cands)[0]
                if name:
                    n = load_texture(_join(tex_dir, name)).shape[1]
        except Exception:
            n = 1
        labels = [f"B{i}" for i in range(n)] if n == 7 else [str(i) for i in range(n)]
        opts = [{"label": l, "value": i} for i, l in enumerate(labels)]
        value = cur if isinstance(cur, int) and cur < n else 0
        return opts, value, ({"display": "block"} if n > 1 else {"display": "none"})

    # -- play / prev / next ----------------------------------------------------- #
    @app.callback(Output("play-interval", "disabled"), Output("btn-play", "children"),
                  Input("btn-play", "n_clicks"), State("play-interval", "disabled"),
                  prevent_initial_call=True)
    def toggle_play(_n, disabled):
        return (False, "⏸ Pause") if disabled else (True, "▶ Play")

    @app.callback(Output("play-interval", "interval"), Input("fps-input", "value"))
    def set_fps(fps):
        return int(1000 / max(float(fps or 1), 0.1))

    @app.callback(Output("mesh-dd", "value", allow_duplicate=True),
                  Output("ses-slider", "value", allow_duplicate=True),
                  Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks"),
                  Input("play-interval", "n_intervals"),
                  State("source-radio", "value"), State("mesh-dd", "options"),
                  State("mesh-dd", "value"), State("ses-slider", "value"),
                  State("ses-slider", "max"), prevent_initial_call=True)
    def step(_p, _n, _i, src, opts, mesh, ses, ses_max):
        d = -1 if dash.ctx.triggered_id == "btn-prev" else 1
        if src == "npz":
            return no_update, ((ses or 0) + d) % ((ses_max or 0) + 1)
        names = [o["value"] for o in (opts or [])]
        if not names:
            return no_update, no_update
        i = names.index(mesh) if mesh in names else -d
        return names[(i + d) % len(names)], no_update

    # -- settings and camera ---------------------------------------------------- #
    @app.callback(Output("settings-store", "data"),
                  *[Input(cid, prop) for _, cid, prop in SETTING_INPUTS])
    def collect(*vals):
        return clean_settings(dict(zip([k for k, _, _ in SETTING_INPUTS], vals)))

    @app.callback(Output("camera-store", "data"), Input("mesh-graph", "relayoutData"),
                  Input("camera-radio", "value"), Input("orient-opts", "value"),
                  State("camera-store", "data"))
    def remember_camera(relayout, _c, _o, current):
        if dash.ctx.triggered_id in ("camera-radio", "orient-opts"):
            return None
        if relayout and "scene.camera" in relayout:
            return relayout["scene.camera"]
        return current

    # -- main figure -------------------------------------------------------------- #
    def current_figure(st):
        sc, i, info, stem = scene_from_settings(st)
        if sc is None:
            return None, info, "Nothing to show yet.", stem
        frame = sc.frame(i)
        kind = resolve_kind(st, sc, frame)
        idx = list(range(len(sc))) if st.get("source") == "npz" and st.get("fixed_scale") else [i]
        crange = continuous_range(sc, idx, st) if kind == "continuous" else None
        rd = render_data(frame, st, kind, crange)
        bounds = sc.bounds(idx, st.get("rotate180"))
        uirev = "|".join(str(st.get(k)) for k in ("source", "camera", "rotate180", "flip"))
        title = f"{sc.title} — {frame['name']}" if st.get("source") == "npz" else frame["name"]
        fig = go.Figure(plotly_traces(rd, st))
        fig.update_layout(**plotly_layout(bounds, scene_camera(st, sc.hemi), title, uirev))
        warn = frame.get("warning", "")
        if kind == "label" and not colored_bands(st):
            warn = (warn + " · " if warn else "") + "No band selected for this Show mode."
        return fig, info, warn, stem

    @app.callback(Output("mesh-graph", "figure"), Output("mesh-graph", "config"),
                  Output("match-info", "children"), Output("view-status", "children"),
                  Input("settings-store", "data"))
    def update_figure(st):
        if not st:
            return no_update, no_update, no_update, no_update
        try:
            fig, info, warn, stem = current_figure(st)
        except Exception as e:
            return go.Figure(), {"displaylogo": False}, "", f"Error: {e.__class__.__name__}: {e}"
        if fig is None:
            return go.Figure(), {"displaylogo": False}, info, warn
        config = {"displaylogo": False,
                  "toImageButtonOptions": {"format": "png", "filename": stem or "spangy",
                                           "height": 1800, "width": 2000, "scale": 2}}
        return fig, config, info, warn

    # -- exports ------------------------------------------------------------------ #
    def launch(mode, spec, out, stem):
        spec_path = out / f"{stem}.spec.json"
        log_path = out / f"{stem}.log"
        spec_path.write_text(json.dumps(spec, indent=1))
        logf = open(log_path, "w")
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), mode, "--spec",
                                 str(spec_path)], stdout=logf, stderr=subprocess.STDOUT)
        JOBS.append(dict(proc=proc, log=str(log_path), out=spec.get("output", ""), mode=mode,
                         started=datetime.datetime.now().strftime("%H:%M:%S")))
        return spec_path

    @app.callback(Output("export-status", "children"),
                  Input("btn-png", "n_clicks"), Input("btn-html", "n_clicks"),
                  Input("btn-movie", "n_clicks"), Input("btn-pyvista", "n_clicks"),
                  State("settings-store", "data"), State("camera-store", "data"),
                  State("outdir-dir", "value"), State("renderer-dd", "value"),
                  State("width-input", "value"), State("height-input", "value"),
                  State("use-view", "value"), State("html-opts", "value"),
                  State("fmt-dd", "value"), State("fps-out", "value"),
                  State("morph-input", "value"), State("hold-input", "value"),
                  State("fps-input", "value"), prevent_initial_call=True)
    def export(_a, _b, _c, _d, st, cam, out_dir, renderer, w, h, use_view, html_opts,
               fmt, fps, morph, hold, play_fps):
        trig = dash.ctx.triggered_id
        if not st:
            return "Nothing to export."
        if not out_dir:
            return "Set an output directory first."
        out = Path(out_dir).expanduser()
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"Cannot create output directory: {e}"
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        camera = cam if "view" in (use_view or []) else None
        sequence = trig in ("btn-movie", "btn-pyvista") or (
            trig == "btn-html" and "anim" in (html_opts or []))
        try:
            sc, i, _, stem = scene_from_settings(st, sequence=sequence)
            if sc is None:
                return "Nothing to export."
            show = st.get("show", "both")
            stem = f"{stem}_{show}_{st.get('camera')}_{stamp}"
            if trig == "btn-html":
                kind = resolve_kind(st, sc, sc.frame(i))
                idx = list(range(len(sc))) if sequence or st.get("fixed_scale") else [i]
                crange = continuous_range(sc, idx, st) if kind == "continuous" else None
                bounds = sc.bounds(idx, st.get("rotate180"))
                camera = scene_camera(st, sc.hemi, camera)
                if sequence and len(sc) > 1:
                    fig = plotly_animation(sc, st, kind, crange, bounds, camera,
                                           ms=int(1000 / max(float(play_fps or 1), 0.1)))
                else:
                    rd = render_data(sc.frame(i), st, kind, crange)
                    fig = go.Figure(plotly_traces(rd, st))
                    fig.update_layout(**plotly_layout(bounds, camera,
                                                      f"{sc.title} — {rd['name']}"))
                path = out / f"{stem}.html"
                fig.write_html(path, include_plotlyjs=True if "full" in (html_opts or []) else "cdn")
                return f"Saved {path}"

            src = spec_source(st, sequence)
            src, cur = src if isinstance(src, tuple) else (src, i)
            spec = dict(source=src, settings=st, camera=camera, renderer=renderer,
                        size=[int(w or 1200), int(h or 1000)], out_dir=str(out), stem=stem)
            if trig == "btn-pyvista":
                spec.update(start=int(cur), fps=float(play_fps or 1))
                launch("pyvista", spec, out, stem)
                return "PyVista window opening (space = play, ←/→ = step, s = screenshot)."
            if trig == "btn-png":
                spec.update(format="png", indices=[int(cur)], output=str(out / f"{stem}.png"))
            else:
                spec.update(format=fmt, fps=float(fps or 4), morph=int(morph or 0),
                            hold=int(hold or 1), output=str(out / f"{stem}.{fmt}"))
            launch("export", spec, out, stem)
            return f"Export started → {spec['output']}"
        except Exception as e:
            return f"Export failed: {e.__class__.__name__}: {e}"

    @app.callback(Output("job-status", "children"), Input("job-interval", "n_intervals"))
    def poll(_n):
        lines = []
        for job in JOBS[-6:]:
            rc = job["proc"].poll()
            state = "running" if rc is None else ("done" if rc == 0 else f"failed ({rc})")
            last = ""
            try:
                txt = Path(job["log"]).read_text().strip().splitlines()
                last = txt[-1] if txt else ""
            except Exception:
                pass
            lines.append(f"[{job['started']}] {job['mode']} {state}: {last}")
        return "\n".join(lines)

    return app


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("export", "pyvista"):
        p = argparse.ArgumentParser(prog=f"spangy_viewer_hub.py {sys.argv[1]}")
        p.add_argument("--spec", required=True, help="spec JSON written by the app")
        a = p.parse_args(sys.argv[2:])
        spec = json.loads(Path(a.spec).read_text())
        if sys.argv[1] == "export":
            run_export(spec, log=lambda m: print(m, flush=True))
        else:
            run_pyvista_window(spec)
        return

    p = argparse.ArgumentParser(description="SPANGY viewer hub")
    p.add_argument("--base", default=DEFAULT_BASE, help="BaboFet data root (for quick paths)")
    p.add_argument("--mesh-dir", default=DEFAULT_MESH_DIR)
    p.add_argument("--tex-dir", default=DEFAULT_TEX_DIR)
    p.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--preset", nargs="*", default=[], help="extra quick-path folders")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    build_app(args).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
