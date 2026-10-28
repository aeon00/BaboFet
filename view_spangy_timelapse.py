#!/usr/bin/env python3
"""
view_spangy_timelapse.py — run LOCALLY on the .npz made by build_spangy_timelapse.py.
Plays SPANGY bands changing across sessions on one mesh, with no resampling at
view time: everything is already on the reference mesh.

  python view_spangy_timelapse.py sub-Aziza_hemi-L_label_on_ses-07.npz
  python view_spangy_timelapse.py FILE.npz --geometry ref        # fixed reference shape
  python view_spangy_timelapse.py FILE.npz --same-size           # morph shape only, no growth
  python view_spangy_timelapse.py FILE.npz --export aziza_L.mp4  # write a movie, no window

Controls: slider = time (between sessions too), space = play/pause,
          left/right arrows = previous/next session.
Geometry: "morph" (default) shows each session's own anatomy, blended between
sessions, so you see the brain grow and fold while the bands change.
"""
import argparse

import numpy as np

# band number -> colour (B3 gold, B4 blue, B5 green, B6 red, as in the earlier timelapse)
BAND_COLOURS = {3: (0.90, 0.72, 0.00), 4: (0.12, 0.47, 0.71),
                5: (0.17, 0.63, 0.17), 6: (0.84, 0.15, 0.16)}
LOW_BAND = (0.80, 0.80, 0.80)      # bands below B3
SULCAL_SHADE = 0.55                # sulcal vertices (negative labels) drawn darker


def band_rgb(labels, signed=True):
    """SPANGY local dominant band (signed: <0 sulci, >0 gyri) -> RGB per vertex."""
    lab = np.rint(labels).astype(int)
    rgb = np.tile(np.array(LOW_BAND, np.float32), (lab.size, 1))
    for band, colour in BAND_COLOURS.items():
        rgb[np.abs(lab) == band] = colour
    if signed:
        rgb[lab < 0] *= SULCAL_SHADE
    return rgb


def frame_at(t, coords, values, kind, same_size, ref_scale):
    """Geometry and values at continuous time t (0 .. S-1)."""
    s0 = int(np.floor(t)); s1 = min(s0 + 1, len(coords) - 1); f = t - s0
    xyz = (1 - f) * coords[s0] + f * coords[s1]
    xyz = xyz - xyz.mean(axis=0)
    if same_size:
        xyz *= ref_scale / np.sqrt((xyz ** 2).sum(axis=1).mean())
    if kind == "label":
        val = values[s0] if f < 0.5 else values[s1]      # bands switch at the midpoint
    else:
        val = (1 - f) * values[s0] + f * values[s1]
    return xyz.astype(np.float32), val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--geometry", choices=["morph", "ref"], default="morph")
    p.add_argument("--same-size", action="store_true", help="remove growth, keep shape change")
    p.add_argument("--column", type=int, default=0, help="texture column (multi-column maps)")
    p.add_argument("--unsigned", action="store_true", help="ignore sulcal/gyral sign")
    p.add_argument("--speed", type=float, default=0.05, help="sessions per frame when playing")
    p.add_argument("--export", help="write a movie (.mp4 or .gif) instead of opening a window")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    import pyvista as pv

    d = np.load(args.npz, allow_pickle=False)
    faces, sessions, kind = d["faces"], [str(s) for s in d["sessions"]], str(d["kind"])
    values = d["values"][:, :, args.column]
    coords = d["coords"]
    ref_idx = sessions.index(str(d["ref"])) if str(d["ref"]) in sessions else len(sessions) - 1
    if args.geometry == "ref":
        coords = np.repeat(coords[ref_idx][None], len(sessions), axis=0)
    c_ref = coords[ref_idx] - coords[ref_idx].mean(axis=0)
    ref_scale = float(np.sqrt((c_ref ** 2).sum(axis=1).mean()))
    n = len(sessions)
    title = f"{d['sub']} hemi-{d['hemi']}  (on {d['ref']} mesh)"

    xyz, val = frame_at(0.0, coords, values, kind, args.same_size, ref_scale)
    mesh = pv.PolyData(xyz, np.c_[np.full(len(faces), 3), faces].ravel())
    pl = pv.Plotter(off_screen=bool(args.export), window_size=(1200, 900))
    if kind == "label":
        mesh.point_data["rgb"] = band_rgb(val, not args.unsigned)
        pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True)
        legend = [[f"B{b}", c] for b, c in BAND_COLOURS.items()] + [["< B3", LOW_BAND]]
        if not args.unsigned:
            legend.append(["darker = sulcal", (0.35, 0.35, 0.35)])
        pl.add_legend(legend, bcolor="white", size=(0.18, 0.22))
    else:
        lo, hi = np.percentile(values, [2, 98])
        mesh.point_data["value"] = val
        pl.add_mesh(mesh, scalars="value", cmap="viridis", clim=(lo, hi), smooth_shading=True)
    pl.add_text(title, position="upper_left", font_size=12)

    # camera framed on the largest session so nothing leaves the view while growing
    # lateral view: left hemisphere seen from -x, right from +x
    pl.view_vector((-1, 0, 0) if str(d["hemi"]) == "L" else (1, 0, 0), viewup=(0, 0, 1))
    big = np.abs(coords - coords.mean(axis=1, keepdims=True)).max()
    pl.reset_camera(bounds=(-big, big, -big, big, -big, big))

    state = {"t": 0.0, "playing": False}

    def show(t):
        state["t"] = float(np.clip(t, 0, n - 1))
        xyz, val = frame_at(state["t"], coords, values, kind, args.same_size, ref_scale)
        mesh.points = xyz
        if kind == "label":
            mesh.point_data["rgb"][:] = band_rgb(val, not args.unsigned)
        else:
            mesh.point_data["value"][:] = val
        k = int(round(state["t"]))
        pl.add_text(f"{sessions[k]}   ({k + 1}/{n})", position="lower_left",
                    font_size=14, name="session")
        pl.render()

    if args.export:
        if args.export.endswith(".gif"):
            pl.open_gif(args.export, fps=args.fps)
        else:
            pl.open_movie(args.export, framerate=args.fps)
        for t in np.arange(0, n - 1 + 1e-9, args.speed):
            show(t); pl.write_frame()
        for _ in range(args.fps):            # hold the last session for 1 s
            pl.write_frame()
        pl.close()
        print(f"written {args.export}")
        return

    slider = pl.add_slider_widget(show, [0, n - 1], value=0, title="session",
                                  pointa=(0.25, 0.08), pointb=(0.75, 0.08), fmt="%.2f")

    def tick(_step=None):
        if state["playing"]:
            t = state["t"] + args.speed
            if t > n - 1:
                t = 0.0
            slider.GetRepresentation().SetValue(t)
            show(t)

    def toggle():
        state["playing"] = not state["playing"]

    def step(delta):
        t = round(state["t"]) + delta
        slider.GetRepresentation().SetValue(np.clip(t, 0, n - 1))
        show(t)

    pl.add_key_event("space", toggle)
    pl.add_key_event("Right", lambda: step(+1))
    pl.add_key_event("Left", lambda: step(-1))
    pl.add_timer_event(max_steps=10_000_000, duration=50, callback=tick)
    show(0.0)
    pl.show()


if __name__ == "__main__":
    main()
