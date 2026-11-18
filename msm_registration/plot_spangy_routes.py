#!/usr/bin/env python3
"""
plot_spangy_routes.py — simple figures and a plain-language report from the
tables written by compare_spangy_routes.py.

The three versions of every session's SPANGY result:
  original mesh                    SPANGY on the session's own mesh (no interpolation)
  maps moved to the reference      the original maps resampled onto the reference mesh
  SPANGY rerun on the reference    SPANGY run again on the session's surface resampled
                                   onto the reference mesh

Every figure answers one question, written as its title. Bars show the MEDIAN
over sessions; each dot is one session of one animal and hemisphere.

Usage:
    python plot_spangy_routes.py                     # sulc registrations
    python plot_spangy_routes.py --texture dpfstar
    python plot_spangy_routes.py --dir /path/to/spangy_route_comparison/sulc

Writes into <dir>/figures/:
    1_do_the_two_versions_agree.png
    2_band_coverage_vs_original.png
    3_band_strength_change.png
    4_change_with_distance.png
    5_sessions_that_disagree_most.png
and <dir>/report.html with every figure and its explanation.
"""
import argparse
import base64
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"

ROUTE_NAME = {"ORIG": "original mesh",
              "RESTEX": "maps moved to the reference mesh",
              "ONREF": "SPANGY rerun on the reference mesh"}
COLOURS = {"ORIG": "#2a78d6", "RESTEX": "#eb6834", "ONREF": "#1baf7a"}
AGREE_MAIN, AGREE_LIGHT = "#2a78d6", "#9cc3f0"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1b1b1a", "#55534f", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 9, "legend.fontsize": 10, "legend.frameon": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def ses_num(s):
    return int(re.search(r"\d+", str(s)).group())


def gap_label(g):
    return "the reference\nsession itself" if g == 0 else f"{g} session{'s' if g > 1 else ''}\nbefore it"


def jitter(n, seed=0, width=0.12):
    return np.random.default_rng(seed).uniform(-width, width, n)


def fmt(v, unit=""):
    if not np.isfinite(v):
        return ""
    return f"{v:.0f}{unit}" if abs(v) >= 10 else f"{v:.1f}{unit}"


def bar_with_dots(ax, x, values, colour, seed=0, width=0.6, unit="", label_median=True):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    med = np.median(v)
    ax.bar(x, med, width=width, color=colour, alpha=0.85, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.scatter(x + jitter(len(v), seed, width * 0.2), v, s=12, color=INK, alpha=0.35, zorder=3,
               linewidths=0)
    if label_median:
        y = max(med, v.max()) if med >= 0 else min(med, v.min())
        ax.text(x, y, f"median {fmt(med, unit)}", ha="center", va="bottom" if med >= 0 else "top",
                fontsize=9.5, color=INK, zorder=4)


def legend_row(fig, items):
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, c in items],
               loc="upper left", bbox_to_anchor=(0.005, 0.0), ncol=len(items))
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
def fig_agree(ps, out):
    gaps = sorted(ps["gap"].unique())
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(gaps) + 2), 4.6))
    for i, g in enumerate(gaps):
        d = ps[ps["gap"] == g]
        bar_with_dots(ax, i - 0.2, 100 * d["restex_vs_onref_same_label"], AGREE_MAIN, seed=i,
                      width=0.38, unit="%")
        bar_with_dots(ax, i + 0.2, 100 * d["restex_vs_onref_same_side"], AGREE_LIGHT, seed=50 + i,
                      width=0.38, unit="%")
    ax.set_xticks(range(len(gaps)), [gap_label(g) for g in gaps])
    ax.set_ylabel("% of the surface where both versions agree")
    ax.set_ylim(0, 108)
    ax.set_title("Agreement between the two versions on the reference mesh (higher is better)",
                 loc="left")
    legend_row(fig, [("same band and same side (sulcus or gyrus)", AGREE_MAIN),
                     ("same side only (sulcus or gyrus)", AGREE_LIGHT)])
    fig.suptitle("Do the two shared-mesh versions give the same result?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Compares ‘maps moved to the reference mesh’ with ‘SPANGY rerun on the reference mesh’, "
                 "vertex by vertex. Sessions are grouped by how far they are from the reference session "
                 "of their animal. At the reference session itself both should agree almost completely.")
    fig.tight_layout()
    return save(fig, f"{out}/1_do_the_two_versions_agree.png")


def fig_coverage(fr, out):
    long = fr.melt(id_vars=["subject", "hemi", "session", "route"], var_name="band_side",
                   value_name="share")
    long[["band", "side"]] = long["band_side"].str.split("_", expand=True)
    long = long[long["band"].isin(["B3", "B4", "B5", "B6"])]
    wide = long.pivot_table(index=["subject", "hemi", "session", "band", "side"],
                            columns="route", values="share").reset_index()
    if "ORIG" not in wide:
        return None
    bands = ["B3", "B4", "B5", "B6"]
    routes = [r for r in ("RESTEX", "ONREF") if r in wide]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for k, b in enumerate(bands):
        d = wide[wide["band"] == b]
        for ri, r in enumerate(routes):
            # absolute difference from the original, in percentage points of the surface
            diff = 100 * (d[r] - d["ORIG"]).abs()
            bar_with_dots(ax, k + (ri - 0.5) * 0.4, diff, COLOURS[r], seed=k * 3 + ri, width=0.38)
    ax.set_xticks(range(len(bands)), [f"Band {b[1]}" for b in bands])
    ax.set_ylabel("difference from the original\n(percentage points of the surface)")
    ax.set_title("How far each version is from the original mesh result (lower is better)", loc="left")
    ax.margins(y=0.15)
    legend_row(fig, [(ROUTE_NAME[r], COLOURS[r]) for r in routes])
    fig.suptitle("Which version keeps the area covered by each band closest to the original?",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    caption(fig, "For each band, the share of the brain surface that the band dominates, compared with the "
                 "result on the original mesh. Example: 2 means the band covers 2% more or 2% less of the "
                 "surface than in the original. Sulcal and gyral parts are counted separately.")
    fig.tight_layout()
    return save(fig, f"{out}/2_band_coverage_vs_original.png")


def fig_power(pw, out):
    bands = [b for b in ("B1", "B2", "B3", "B4", "B5", "B6") if b in set(pw["band"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for k, b in enumerate(bands):
        bar_with_dots(ax, k, 100 * pw.loc[pw["band"] == b, "rel_diff"], COLOURS["ONREF"],
                      seed=k, unit="%")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(bands)), [f"Band {b[1]}" for b in bands])
    ax.set_ylabel("change compared with the original (%)")
    ax.set_title("Strength of each band when SPANGY is rerun on the reference mesh (0 = no change)",
                 loc="left")
    ax.margins(y=0.2)
    fig.suptitle("Does rerunning SPANGY on the resampled surface change the band strengths?",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    caption(fig, "Band strength is SPANGY's power for that band over the whole hemisphere. Negative values "
                 "mean weaker than on the original mesh. Lower strength in bands 5 and 6 means fine folds "
                 "were smoothed out by resampling the surface.")
    fig.tight_layout()
    return save(fig, f"{out}/3_band_strength_change.png")


def fig_distance(ps, out):
    col = "afp_rel_diff_onref_vs_orig"
    if col not in ps:
        return None
    gaps = sorted(ps["gap"].unique())
    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(gaps) + 2), 4.6))
    for i, g in enumerate(gaps):
        bar_with_dots(ax, i, 100 * ps.loc[ps["gap"] == g, col], COLOURS["ONREF"], seed=i, unit="%")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(gaps)), [gap_label(g) for g in gaps])
    ax.set_ylabel("change compared with the original (%)")
    ax.set_title("Total folding when SPANGY is rerun on the reference mesh (0 = no change)", loc="left")
    ax.margins(y=0.2)
    fig.suptitle("Is the change bigger for sessions far from the reference?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Total folding is SPANGY's folding power (all bands from 1 upwards). If the change grows "
                 "with distance, the earliest sessions are the ones where rerunning SPANGY on the "
                 "reference mesh is least reliable.")
    fig.tight_layout()
    return save(fig, f"{out}/4_change_with_distance.png")


def fig_worst(ps, out, n=12):
    d = ps.assign(name=ps["subject"].str.replace("sub-", "") + ", "
                  + ps["hemi"].map({"L": "left", "R": "right"}) + ", session "
                  + ps["session"].map(ses_num).astype(str) + "  (reference "
                  + ps["ref"].map(ses_num).astype(str) + ")")
    d = d.nsmallest(n, "restex_vs_onref_same_label")[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(d) + 1.6))
    y = np.arange(len(d))
    v = 100 * d["restex_vs_onref_same_label"].values
    ax.barh(y, v, height=0.65, color=AGREE_MAIN, alpha=0.85, edgecolor=SURFACE, linewidth=1.5)
    for yi, vi in zip(y, v):
        ax.text(vi, yi, f" {vi:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, d["name"])
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of the surface with the same band and side in both versions")
    fig.suptitle(f"The {len(d)} sessions where the two versions disagree most", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    caption(fig, "Look at these sessions first in the viewer (prepare_route_viewer.py puts every version "
                 "on inflated surfaces).")
    fig.tight_layout()
    return save(fig, f"{out}/5_sessions_that_disagree_most.png")


# --- report ------------------------------------------------------------------------
def img(path):
    with open(path, "rb") as fh:
        return f'<img alt="" src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def summary(ps, pw):
    rows = [
        ("Same band and same side in both shared-mesh versions", 100 * ps["restex_vs_onref_same_label"], "%"),
        ("Same side (sulcus or gyrus) in both versions", 100 * ps["restex_vs_onref_same_side"], "%"),
    ]
    if "curv_corr_restex_vs_onref" in ps:
        rows.append(("Curvature: moved maps vs recomputed on the reference mesh (correlation)",
                     ps["curv_corr_restex_vs_onref"], ""))
    if "frac_maxdiff_restex_vs_orig" in ps:
        rows.append(("Largest band coverage difference from the original, maps moved (percentage points)",
                     100 * ps["frac_maxdiff_restex_vs_orig"], ""))
        rows.append(("Largest band coverage difference from the original, SPANGY rerun (percentage points)",
                     100 * ps["frac_maxdiff_onref_vs_orig"], ""))
    if "afp_rel_diff_onref_vs_orig" in ps:
        rows.append(("Total folding, SPANGY rerun vs original (% change)", 100 * ps["afp_rel_diff_onref_vs_orig"], "%"))
    if "area_ratio_ref_vs_orig" in ps:
        rows.append(("Surface area of the resampled surface vs the original (%)", 100 * ps["area_ratio_ref_vs_orig"], "%"))
    body = "".join(f"<tr><td>{name}</td><td>{fmt(np.nanmedian(v), u)}</td>"
                   f"<td>{fmt(np.nanmin(v), u)} to {fmt(np.nanmax(v), u)}</td></tr>" for name, v, u in rows)
    return f"<table><tr><th>Measure</th><th>Median</th><th>Range</th></tr>{body}</table>"


EXPLAIN = {
    "1": ("Both shared-mesh versions put every session on the reference session's vertices, so they can be "
          "compared point by point. Dark bars: share of the surface where they give exactly the same band "
          "on the same side (sulcus or gyrus). Light bars: share where they at least agree on sulcus versus "
          "gyrus.",
          "High values mean it hardly matters which version you use. If agreement falls for sessions far "
          "from the reference, the choice matters most for the early sessions."),
    "2": ("For each band, how much of the brain surface it covers in each version, compared with the "
          "original mesh, which involves no interpolation and is the benchmark.",
          "The version with the shorter bars stays closer to the original and is the safer choice for "
          "vertex-by-vertex analyses."),
    "3": ("The strength of each band over the whole hemisphere when SPANGY is rerun on the resampled "
          "surface, as a % change from the original mesh.",
          "Bars near 0 mean resampling the surface did not change the result. Clearly negative bars in "
          "bands 5 and 6 mean fine folds were smoothed out."),
    "4": ("The change in total folding when SPANGY is rerun on the resampled surface, grouped by how far "
          "the session is from the reference.",
          "If the bars grow for sessions far from the reference, avoid the rerun version for those "
          "sessions, or for all of them."),
    "5": ("The sessions where the two shared-mesh versions agree least.",
          "These are the first sessions to look at in the viewer."),
}


def write_report(d, figs, ps, pw, texture):
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SPANGY versions</title>
<style>
body{{background:{SURFACE};color:{INK};font:15px/1.6 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:18px;margin:32px 0 6px}} p,li{{color:{INK2}}}
img{{max-width:100%;height:auto;margin:10px 0}}
table{{border-collapse:collapse;font-size:14px;margin:8px 0;width:100%}}
th,td{{padding:6px 10px;border-bottom:1px solid {GRID};text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.box{{border-left:3px solid {GRID};padding:2px 12px;margin:6px 0}}
</style></head><body>
<h1>Which SPANGY version to use on a shared mesh?</h1>
<p>{ps['subject'].nunique()} animals, {len(ps)} sessions, using the {texture} registrations. Three versions of each
session's SPANGY result are compared:</p>
<ul><li><b>Original mesh</b>: SPANGY on the session's own mesh. No interpolation, so this is the benchmark.</li>
<li><b>Maps moved to the reference mesh</b>: the original maps resampled onto the reference session's vertices.</li>
<li><b>SPANGY rerun on the reference mesh</b>: the session's surface resampled onto the reference session's
vertices, then SPANGY run again on it.</li></ul>
<p>The last two share vertices with every other session of the same animal, which is what vertex-by-vertex
analyses need. The question is which of them stays closer to the original.</p>
<h2>Summary</h2>{summary(ps, pw)}"""]
    for f in figs:
        k = os.path.basename(f)[0]
        what, read = EXPLAIN.get(k, ("", ""))
        parts.append(f"{img(f)}<div class='box'><p><b>What is shown.</b> {what}</p>"
                     f"<p><b>How to read it.</b> {read}</p></div>")
    parts.append("</body></html>")
    path = f"{d}/report.html"
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"  {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texture", default="sulc")
    p.add_argument("--dir", help="comparison folder (default: <BASE>/spangy_route_comparison/<texture>)")
    args = p.parse_args()
    d = args.dir or f"{BASE}/spangy_route_comparison/{args.texture}"
    out = f"{d}/figures"
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):                    # drop figures from the previous version
        if old.endswith(".png"):
            os.remove(f"{out}/{old}")

    ps = pd.read_csv(f"{d}/per_session.csv")
    ps["gap"] = ps["ref"].map(ses_num) - ps["session"].map(ses_num)
    fr = pd.read_csv(f"{d}/band_area_fractions.csv")
    pw_path = f"{d}/band_powers.csv"
    pw = pd.read_csv(pw_path) if os.path.isfile(pw_path) else None
    print(f"{len(ps)} sessions from {d}")

    figs = [fig_agree(ps, out), fig_coverage(fr, out),
            fig_power(pw, out) if pw is not None else None,
            fig_distance(ps, out), fig_worst(ps, out)]
    write_report(d, [f for f in figs if f], ps, pw, args.texture)


if __name__ == "__main__":
    main()