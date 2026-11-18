#!/usr/bin/env python3
"""
plot_msm_reg_metrics.py — simple figures and a plain-language report from
msm_reg_metrics.csv (written by msm_reg_metrics.py).

Every figure answers one question, written as its title. Bars show the MEDIAN
over registrations; each dot is one registration (one animal, one hemisphere,
one session paired with the next).

Usage:
    python plot_msm_reg_metrics.py
    python plot_msm_reg_metrics.py --csv /path/to/msm_reg_metrics.csv

Writes next to the CSV, in figures/:
    1_which_texture_is_better.png
    2_before_and_after_registration.png
    3_skipped_sessions.png
    4_most_stretched_registrations.png
    5_band_overlap.png            (only if msm_reg_metrics.py was run with --bands)
    6_painted_sulcus_overlap.png  (only if it was run with --roi-pattern)
and report.html with every figure and its explanation.
"""
import argparse
import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"

NAMES = {"sulc": "sulc", "dpfstar": "DPF-star"}
COLOURS = {"sulc": "#2a78d6", "dpfstar": "#eb6834"}
BEFORE = "#b9b7b0"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1b1b1a", "#55534f", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 9, "legend.fontsize": 10, "legend.frameon": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})

# column -> (plain name, unit text, higher is better?)
MEASURES = {
    "corr_curv_after": ("Curvature match after registration",
                        "correlation (0 = none, 1 = perfect)", True),
    "dice_sulcal": ("Sulcus overlap after registration", "overlap in % (100 = perfect)", True),
    "jacobian_mean": ("Stretching of the surface", "average stretching (0 = none, 1 = twice the size)", False),
}


def textures(df):
    return [t for t in ("sulc", "dpfstar") if t in set(df["texture"])]


def jitter(n, seed=0, width=0.12):
    return np.random.default_rng(seed).uniform(-width, width, n)


def fmt(v, unit=""):
    if not np.isfinite(v):
        return ""
    return (f"{v:.0f}{unit}" if abs(v) >= 10 else f"{v:.2f}{unit}")


def bar_with_dots(ax, x, values, colour, seed=0, width=0.6, label_unit=""):
    """One bar at the median, the individual values as dots, the median written above."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    med = np.median(v)
    ax.bar(x, med, width=width, color=colour, alpha=0.85, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.scatter(x + jitter(len(v), seed), v, s=12, color=INK, alpha=0.35, zorder=3, linewidths=0)
    top = max(med, v.max())
    ax.text(x, top, f"median {fmt(med, label_unit)}", ha="center", va="bottom",
            fontsize=9.5, color=INK, zorder=4)


def texture_legend(fig, texs):
    """Legend in its own row under the title, so it never covers data."""
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOURS[t], label=f"registered with {NAMES[t]}")
                        for t in texs], loc="upper left", bbox_to_anchor=(0.005, 0.0), ncol=len(texs))
    fig._has_legend_row = True


def caption(fig, text):
    y = -0.09 if getattr(fig, "_has_legend_row", False) else -0.02
    fig.text(0.01, y, text, ha="left", va="top", fontsize=9.5, color=INK2, wrap=True)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


# --- figures -----------------------------------------------------------------------
def fig_which_texture(df, out):
    cols = [c for c in MEASURES if c in df]
    texs = textures(df)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols), 4.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        name, unit, higher = MEASURES[c]
        for i, t in enumerate(texs):
            bar_with_dots(ax, i, df.loc[df["texture"] == t, c], COLOURS[t], seed=i)
        ax.set_xticks(range(len(texs)), [f"registered\nwith {NAMES[t]}" for t in texs])
        ax.set_title(f"{name}\n({'higher' if higher else 'lower'} is better)", loc="left")
        ax.set_ylabel(unit)
        ax.margins(y=0.15)
    fig.suptitle("Which registration texture works better?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Each dot is one registration (one animal, one hemisphere, one session registered "
                 "to the next). Bars show the median. Curvature was not used by either registration, "
                 "so these are fair comparisons between sulc and DPF-star.")
    fig.tight_layout()
    return save(fig, f"{out}/1_which_texture_is_better.png")


def fig_before_after(df, out):
    if not {"corr_curv_before", "corr_curv_after"} <= set(df):
        return None
    texs = textures(df)
    fig, ax = plt.subplots(figsize=(5.5 + 1.5 * len(texs), 4.4))
    labels, pos = [], 0
    for i, t in enumerate(texs):
        d = df[df["texture"] == t]
        bar_with_dots(ax, pos, d["corr_curv_before"], BEFORE, seed=10 + i)
        bar_with_dots(ax, pos + 1, d["corr_curv_after"], COLOURS[t], seed=20 + i)
        labels += [f"before\n(spheres just overlaid)", f"after registration\nwith {NAMES[t]}"]
        pos += 2.6
    ticks = [p for i in range(len(texs)) for p in (2.6 * i, 2.6 * i + 1)]
    ax.set_xticks(ticks, labels)
    ax.set_ylabel("curvature correlation (0 = none, 1 = perfect)")
    ax.set_title("Curvature match between the two sessions (higher is better)", loc="left")
    ax.margins(y=0.15)
    fig.suptitle("Did registration bring the two sessions into line?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "‘Before’ compares the two sessions with their spheres simply laid on top of each "
                 "other. ‘After’ uses the MSM result. The bigger the rise, the more MSM helped. "
                 "Neither value can reach 1, because the brain changes between sessions.")
    fig.tight_layout()
    return save(fig, f"{out}/2_before_and_after_registration.png")


def fig_skipped(df, out):
    cols = [c for c in ("jacobian_mean", "corr_curv_after") if c in df]
    texs = textures(df)
    df = df.assign(kind=np.where(df["gap"] > 1, "skip", "next"))
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.4))
    axes = np.atleast_1d(axes)
    w = 0.8 / max(len(texs), 1)
    for ax, c in zip(axes, cols):
        name, unit, higher = MEASURES[c]
        for gi, kind in enumerate(("next", "skip")):
            for ti, t in enumerate(texs):
                x = gi + (ti - (len(texs) - 1) / 2) * w
                bar_with_dots(ax, x, df.loc[(df["kind"] == kind) & (df["texture"] == t), c],
                              COLOURS[t], seed=gi * 5 + ti, width=w * 0.9)
        n_next, n_skip = (df["kind"] == "next").sum() // max(len(texs), 1), (df["kind"] == "skip").sum() // max(len(texs), 1)
        ax.set_xticks([0, 1], [f"next session\n({n_next} pairs)", f"a session in between\nwas skipped ({n_skip} pairs)"])
        ax.set_title(f"{name}\n({'higher' if higher else 'lower'} is better)", loc="left")
        ax.set_ylabel(unit)
        ax.margins(y=0.15)
    texture_legend(fig, texs)
    fig.suptitle("Are registrations that skip a session worse?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "A session is skipped when it was excluded (failed quality checks or not scanned). "
                 "The brain changes more over two intervals, so more stretching and a weaker match are "
                 "expected; these pairs deserve a visual check.")
    fig.tight_layout()
    return save(fig, f"{out}/3_skipped_sessions.png")


def fig_most_stretched(df, out, n=12):
    if "jacobian_mean" not in df:
        return None
    texs = textures(df)
    df = df.assign(pair=df["subject"].str.replace("sub-", "") + ", "
                   + df["hemi"].map({"L": "left", "R": "right"}) + ", session "
                   + df["input"].str[4:].str.lstrip("0") + " to " + df["reference"].str[4:].str.lstrip("0"))
    wide = df.pivot_table(index="pair", columns="texture", values="jacobian_mean")
    wide = wide.loc[wide.max(axis=1).sort_values(ascending=False).index[:n]][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(wide) + 1.6))
    h = 0.8 / len(texs)
    y = np.arange(len(wide))
    for ti, t in enumerate(texs):
        if t not in wide:
            continue
        yy = y - (ti - (len(texs) - 1) / 2) * h          # first texture drawn on top
        ax.barh(yy, wide[t], height=h * 0.9, color=COLOURS[t], edgecolor=SURFACE, linewidth=1.5)
        for yi, v in zip(yy, wide[t]):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, wide.index)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("average stretching (0 = none, 1 = twice the size)")
    ax.margins(x=0.12)
    texture_legend(fig, texs)
    fig.suptitle(f"The {len(wide)} registrations with the most stretching", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Start visual checks here. For a per-vertex map of where the stretching happens, "
                 "rerun msm_reg_metrics.py with --save-maps and open the map in wb_view.")
    fig.tight_layout()
    return save(fig, f"{out}/4_most_stretched_registrations.png")


def fig_bands(df, out):
    cols = [f"dice_B{b}" for b in (3, 4, 5, 6) if f"dice_B{b}" in df]
    if not cols:
        return None
    texs = textures(df)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = 0.8 / len(texs)
    for k, c in enumerate(cols):
        for ti, t in enumerate(texs):
            bar_with_dots(ax, k + (ti - (len(texs) - 1) / 2) * w, df.loc[df["texture"] == t, c],
                          COLOURS[t], seed=k * 3 + ti, width=w * 0.9)
    ax.set_xticks(range(len(cols)), [f"Band {c[-1]}" for c in cols])
    ax.set_ylabel("overlap in % (100 = perfect)")
    ax.set_title("Overlap of each SPANGY band after registration (higher is better)", loc="left")
    texture_legend(fig, texs)
    ax.margins(y=0.15)
    fig.suptitle("Do the folding bands line up between sessions?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Bands change as the brain grows, so even a perfect registration cannot reach 100%. "
                 "Read this as how similar the folding pattern is after alignment.")
    fig.tight_layout()
    return save(fig, f"{out}/5_band_overlap.png")


def fig_roi(df, out):
    cols = [c for c in df if c.startswith("dice_roi_")]
    if not cols:
        return None
    texs = textures(df)
    fig, ax = plt.subplots(figsize=(3.5 + 2.2 * len(cols), 4.4))
    w = 0.8 / len(texs)
    for k, c in enumerate(cols):
        for ti, t in enumerate(texs):
            bar_with_dots(ax, k + (ti - (len(texs) - 1) / 2) * w, df.loc[df["texture"] == t, c],
                          COLOURS[t], seed=ti, width=w * 0.9)
    ax.set_xticks(range(len(cols)), [c.replace("dice_roi_", "") for c in cols])
    ax.set_ylabel("overlap in % (100 = perfect)")
    ax.set_title("Overlap of the painted sulcus (higher is better)", loc="left")
    texture_legend(fig, texs)
    ax.margins(y=0.15)
    fig.suptitle("Does the painted sulcus land in the same place?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "The sulcus was painted on both sessions independently of either registration, "
                 "so this is the closest check to a ground truth.")
    fig.tight_layout()
    return save(fig, f"{out}/6_painted_sulcus_overlap.png")


# --- report ------------------------------------------------------------------------
def img(path):
    with open(path, "rb") as fh:
        return f'<img alt="" src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def summary_table(df):
    texs = textures(df)
    rows = []
    plain = [
        ("corr_curv_before", "Curvature match before registration (correlation, higher is better)"),
        ("corr_curv_after", "Curvature match after registration (correlation, higher is better)"),
        ("dice_sulcal", "Sulcus overlap after registration (%, higher is better)"),
        ("dice_gyral", "Gyrus overlap after registration (%, higher is better)"),
        ("jacobian_mean", "Average stretching (0 = none, 1 = twice the size; lower is better)"),
        ("area_diff_abs_pct_median", "Typical change in triangle size (%, lower is better)"),
        ("flipped_faces", "Folded-over triangles (should be 0)"),
        ("agree_band", "Same SPANGY band after registration (% of surface)"),
    ] + [(c, f"Painted sulcus overlap, {c.replace('dice_roi_', '')} (%)") for c in df if c.startswith("dice_roi_")]
    for c, label in plain:
        if c not in df:
            continue
        r = [label]
        for t in texs:
            r.append(fmt(df.loc[df["texture"] == t, c].median()))
        rows.append(r)
    head = "".join(f"<th>Registered with {NAMES[t]} (median)</th>" for t in texs)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><tr><th>Measure</th>{head}</tr>{body}</table>"


def wins(df):
    if len(textures(df)) < 2:
        return ""
    key = ["subject", "hemi", "input", "reference"]
    lines = []
    for c, (name, _, higher) in MEASURES.items():
        if c not in df:
            continue
        w = df.pivot_table(index=key, columns="texture", values=c).dropna()
        if not len(w):
            continue
        better = (w["dpfstar"] > w["sulc"]) if higher else (w["dpfstar"] < w["sulc"])
        lines.append(f"<li><b>{name}:</b> DPF-star did better in {int(better.sum())} of {len(w)} "
                     f"pairs, sulc in {int((~better).sum())}.</li>")
    return "<h2>Pair by pair</h2><p>The same pairs were registered once with each texture.</p><ul>" + "".join(lines) + "</ul>"


EXPLAIN = {
    "1": ("What is shown", "The three main measures, for registrations made with sulc and with DPF-star. "
          "<b>Curvature match</b>: how closely the curvature of the moved session follows the curvature of "
          "the next session (a correlation; 1 would be identical). <b>Sulcus overlap</b>: the share of "
          "sulci of one session that land on sulci of the next. <b>Stretching</b>: how much the registration "
          "had to enlarge or shrink parts of the surface; 1 means parts became twice as big or half as big "
          "on average.",
          "The texture with higher curvature match and sulcus overlap, for similar or less stretching, "
          "aligns the brains better. Curvature was used by neither registration, so the comparison is fair."),
    "2": ("What is shown", "The curvature match before registration (the two spheres simply laid on top of "
          "each other) and after registration.",
          "The rise from the grey bar to the coloured bar is what MSM achieved. A small rise means the "
          "sessions were already well aligned or MSM struggled."),
    "3": ("What is shown", "Stretching and curvature match, separately for pairs of consecutive sessions and "
          "for pairs where a session in between was skipped.",
          "Skipped pairs are expected to be somewhat worse. If they are much worse, check those pairs "
          "before relying on anything mapped through them."),
    "4": ("What is shown", "The registrations that needed the most stretching, for both textures.",
          "These are the first ones to inspect visually."),
    "5": ("What is shown", "For each SPANGY band, the share of the band that lands on the same band in the "
          "next session after registration.",
          "Higher means the folding pattern lines up better. It cannot reach 100%, because the bands "
          "change as the brain grows."),
    "6": ("What is shown", "How much of a sulcus painted on one session lands on the same sulcus painted on "
          "the next session.",
          "This is the most direct check of anatomical correspondence."),
}


def write_report(d, figs, df):
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Registration quality</title>
<style>
body{{background:{SURFACE};color:{INK};font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:18px;margin:32px 0 6px}} p,li{{color:{INK2}}}
img{{max-width:100%;height:auto;margin:10px 0}}
table{{border-collapse:collapse;font-size:14px;margin:8px 0;width:100%}}
th,td{{padding:6px 10px;border-bottom:1px solid {GRID};text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.box{{border-left:3px solid {GRID};padding:2px 12px;margin:6px 0}}
</style></head><body>
<h1>How good are the MSM registrations?</h1>
<p>{df['subject'].nunique()} animals. Each session was registered to the next available session of the same
animal and hemisphere, once using sulc and once using DPF-star to guide the registration.</p>
<h2>Summary</h2>{summary_table(df)}{wins(df)}"""]
    for f in figs:
        k = os.path.basename(f)[0]
        what_t, what, read = EXPLAIN.get(k, ("", "", ""))
        parts.append(f"{img(f)}<div class='box'><p><b>{what_t}.</b> {what}</p>"
                     f"<p><b>How to read it.</b> {read}</p></div>")
    parts.append("</body></html>")
    path = f"{d}/report.html"
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"  {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=f"{BASE}/msm_registered/metrics/msm_reg_metrics.csv")
    args = p.parse_args()
    d = os.path.dirname(os.path.abspath(args.csv))
    out = f"{d}/figures"
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):                    # drop figures from the previous version
        if old.endswith(".png"):
            os.remove(f"{out}/{old}")
    df = pd.read_csv(args.csv)
    print(f"{len(df)} registrations from {args.csv}")
    figs = [fig_which_texture(df, out), fig_before_after(df, out), fig_skipped(df, out),
            fig_most_stretched(df, out), fig_bands(df, out), fig_roi(df, out)]
    write_report(d, [f for f in figs if f], df)


if __name__ == "__main__":
    main()