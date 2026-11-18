#!/usr/bin/env python3
"""
plot_spangy_routes.py — figures and an HTML report for the output of
compare_spangy_routes.py (ORIG vs RESTEX vs ONREF).

Usage (cluster or locally, needs pandas + matplotlib):
    python plot_spangy_routes.py                       # sulc registrations
    python plot_spangy_routes.py --texture dpfstar
    python plot_spangy_routes.py --dir /path/to/spangy_route_comparison/sulc   # copied folder

Writes into <dir>/figures/:
    1_agreement_vs_distance.png   RESTEX vs ONREF vertex agreement against sessions from the reference
    2_agreement_heatmap.png       same-label agreement for every animal/hemisphere/session
    3_curvature_corr.png          mean-curvature correlation RESTEX vs ONREF
    4_band_area_shares.png        area share of each band: RESTEX and ONREF against ORIG
    5_band_power_diff.png         global band power, ONREF relative to ORIG, per band
    6_band_power_vs_distance.png  same, against sessions from the reference (B3-B6)
and <dir>/report.html (all figures + summary tables, one self-contained file).
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

# validated categorical palette, slots 1-3 in fixed order (one colour per route)
ROUTE = {"ORIG": "#2a78d6", "RESTEX": "#eb6834", "ONREF": "#1baf7a"}
MARK = {"ORIG": "o", "RESTEX": "s", "ONREF": "^"}     # secondary encoding, not colour alone
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"                                         # one-hue sequential for the heatmap

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.family": "DejaVu Sans",
})


def ses_num(s):
    return int(re.search(r"\d+", str(s)).group())


def jitter(n, width=0.18, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def median_line(ax, x, y, color):
    m = pd.Series(y).groupby(np.asarray(x)).median()
    ax.plot(m.index, m.values, color=color, lw=2, zorder=3)


# --- figures -------------------------------------------------------------------
def fig_agreement_vs_distance(ps, out):
    cols = [("restex_vs_onref_same_label", "Same label (band + side)"),
            ("restex_vs_onref_same_band", "Same band (either side)"),
            ("restex_vs_onref_same_side", "Same side (sulcal/gyral)"),
            ("restex_vs_onref_kappa", "Cohen's kappa")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, (c, title) in zip(axes, cols):
        x = ps["distance"].values
        ax.scatter(x + jitter(len(x)), ps[c], s=22, color=ROUTE["ONREF"], marker="o",
                   edgecolor=SURFACE, linewidth=0.8, alpha=0.85, zorder=2)
        median_line(ax, x, ps[c].values, INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Sessions before the reference")
        ax.set_ylim(min(0.5, ps[c].min() - 0.02), 1.0)
    axes[0].set_ylabel("Area-weighted agreement\nRESTEX vs ONREF")
    axes[-1].plot([], [], color=INK2, label="median")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Vertex-wise agreement between resampled textures and SPANGY on resampled surfaces",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, f"{out}/1_agreement_vs_distance.png")


def fig_heatmap(ps, out):
    ps = ps.assign(row=ps["subject"].str.replace("sub-", "") + " " + ps["hemi"])
    piv = ps.pivot_table(index="row", columns="session", values="restex_vs_onref_same_label")
    piv = piv[sorted(piv.columns, key=ses_num)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * piv.shape[1], 0.9 + 0.36 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap=SEQ, vmin=np.nanmin(piv.values), vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [c.replace("ses-", "") for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                dark = (v - np.nanmin(piv.values)) / (1 - np.nanmin(piv.values) + 1e-9) > 0.6
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if dark else INK)
    ax.set_xlabel("Session")
    ax.set_title("Same-label agreement, RESTEX vs ONREF (blank = not compared)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, f"{out}/2_agreement_heatmap.png")


def fig_curvature(ps, out):
    if "curv_corr_restex_vs_onref" not in ps:
        return False
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x, y = ps["distance"].values, ps["curv_corr_restex_vs_onref"].values
    ax.scatter(x + jitter(len(x)), y, s=22, color=ROUTE["ONREF"], edgecolor=SURFACE,
               linewidth=0.8, alpha=0.85)
    median_line(ax, x, y, INK2)
    ax.set_xlabel("Sessions before the reference")
    ax.set_ylabel("Correlation of mean curvature\nRESTEX vs ONREF")
    ax.set_title("Mean curvature: resampled vs recomputed", loc="left")
    fig.tight_layout()
    save(fig, f"{out}/3_curvature_corr.png")
    return True


def fig_band_shares(fr, out):
    bands = ["B3", "B4", "B5", "B6"]
    long = fr.melt(id_vars=["subject", "hemi", "session", "route"], var_name="band_side",
                   value_name="share")
    long[["band", "side"]] = long["band_side"].str.split("_", expand=True)
    wide = long.pivot_table(index=["subject", "hemi", "session", "band", "side"],
                            columns="route", values="share").reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, b in zip(axes, bands):
        d = wide[wide["band"] == b]
        hi = float(np.nanmax(d[["ORIG", "RESTEX", "ONREF"]].values)) * 1.08 if len(d) else 1
        ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls="--", zorder=1)
        for route in ("RESTEX", "ONREF"):
            for side, fill in (("gyral", True), ("sulcal", False)):
                s = d[d["side"] == side]
                ax.scatter(s["ORIG"], s[route], s=22, marker=MARK[route], zorder=2,
                           facecolor=ROUTE[route] if fill else "none",
                           edgecolor=ROUTE[route], linewidth=1.1, alpha=0.85)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"{b}", loc="left")
        ax.set_xlabel("Area share, ORIG (original mesh)")
    axes[0].set_ylabel("Area share on reference mesh")
    h = [plt.Line2D([], [], ls="", marker=MARK[r], color=ROUTE[r], label=r) for r in ("RESTEX", "ONREF")]
    h += [plt.Line2D([], [], ls="", marker="o", color=INK2, label="gyral (filled)"),
          plt.Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="sulcal (open)"),
          plt.Line2D([], [], color=INK2, lw=1, ls="--", label="identity")]
    fig.suptitle("Share of surface dominated by each band: closer to the identity line = closer to ORIG",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    save(fig, f"{out}/4_band_area_shares.png")
    return wide


def fig_band_power(pw, ps, out):
    pw = pw.merge(ps[["subject", "hemi", "session", "distance"]], on=["subject", "hemi", "session"])
    bands = sorted(pw["band"].unique(), key=lambda b: int(b[1:]))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, b in enumerate(bands):
        y = 100 * pw.loc[pw["band"] == b, "rel_diff"].values
        ax.scatter(k + jitter(len(y)), y, s=18, color=ROUTE["ONREF"], marker=MARK["ONREF"],
                   edgecolor=SURFACE, linewidth=0.6, alpha=0.8)
        ax.plot([k - 0.28, k + 0.28], [np.nanmedian(y)] * 2, color=INK, lw=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(bands)), bands)
    ax.set_ylabel("ONREF − ORIG (% of ORIG)")
    ax.set_title("Global band power after resampling the surface (bar = median)", loc="left")
    fig.tight_layout()
    save(fig, f"{out}/5_band_power_diff.png")

    hb = [b for b in ("B3", "B4", "B5", "B6") if b in bands]
    fig, axes = plt.subplots(1, len(hb), figsize=(3.5 * len(hb), 3.4), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, hb):
        d = pw[pw["band"] == b]
        x, y = d["distance"].values, 100 * d["rel_diff"].values
        ax.scatter(x + jitter(len(x)), y, s=18, color=ROUTE["ONREF"], marker=MARK["ONREF"],
                   edgecolor=SURFACE, linewidth=0.6, alpha=0.8)
        median_line(ax, x, y, INK2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(b, loc="left")
        ax.set_xlabel("Sessions before the reference")
    axes[0].set_ylabel("ONREF − ORIG (% of ORIG)")
    fig.suptitle("Band power difference (ONREF vs ORIG) by distance from the reference (line = median)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, f"{out}/6_band_power_vs_distance.png")
    return pw


# --- report --------------------------------------------------------------------
def img_tag(path):
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def html_table(df, fmt="{:.3f}"):
    return df.to_html(float_format=lambda v: fmt.format(v), border=0, classes="t", na_rep="–")


def write_report(d, figs, ps, pw, wide, texture):
    med = lambda s: f"{s.median():.3f} [{s.min():.3f}, {s.max():.3f}]"
    rows = []
    for c, name in [("restex_vs_onref_same_label", "Same label, RESTEX vs ONREF"),
                    ("restex_vs_onref_same_band", "Same band, RESTEX vs ONREF"),
                    ("restex_vs_onref_same_side", "Same side, RESTEX vs ONREF"),
                    ("restex_vs_onref_kappa", "Cohen's kappa, RESTEX vs ONREF"),
                    ("curv_corr_restex_vs_onref", "Mean-curvature correlation"),
                    ("frac_maxdiff_restex_vs_orig", "Max band-share difference, RESTEX vs ORIG"),
                    ("frac_maxdiff_onref_vs_orig", "Max band-share difference, ONREF vs ORIG"),
                    ("afp_rel_diff_onref_vs_orig", "Folding power, ONREF vs ORIG (relative)"),
                    ("area_ratio_ref_vs_orig", "Surface area, reference-mesh / original")]:
        if c in ps:
            rows.append((name, med(ps[c])))
    summary = pd.DataFrame(rows, columns=["Measure", "Median [min, max]"])

    by_dist = ps.groupby("distance")[[c for c in ("restex_vs_onref_same_label",
                                                  "restex_vs_onref_kappa",
                                                  "curv_corr_restex_vs_onref") if c in ps]].median()
    by_dist.index.name = "sessions before ref"
    by_dist.columns = [c.replace("restex_vs_onref_", "").replace("curv_corr_restex_vs_onref", "curv_corr")
                       for c in by_dist.columns]

    share_err = None
    if wide is not None:
        w = wide[wide["band"].isin(["B3", "B4", "B5", "B6"])]
        share_err = pd.DataFrame({
            "RESTEX |Δ| vs ORIG": (w["RESTEX"] - w["ORIG"]).abs().groupby(w["band"]).median(),
            "ONREF |Δ| vs ORIG": (w["ONREF"] - w["ORIG"]).abs().groupby(w["band"]).median()})
        share_err.index.name = "band (median over sessions and sides)"

    power = None
    if pw is not None:
        power = (100 * pw.groupby("band")["rel_diff"].agg(["median", "min", "max"]))
        power = power.loc[sorted(power.index, key=lambda b: int(b[1:]))]
        power.index.name = "band"
        power.columns = ["median %", "min %", "max %"]

    ref_rows = ps[ps["distance"] == 0]
    ref_note = ""
    if len(ref_rows):
        ref_note = (f"<p class='note'><b>Sanity check (reference sessions, {len(ref_rows)} rows):</b> "
                    f"same-label agreement {ref_rows['restex_vs_onref_same_label'].min():.3f}–"
                    f"{ref_rows['restex_vs_onref_same_label'].max():.3f}. These should be ≈ 1, since the "
                    f"resampled surface is the original mesh there.</p>")

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPANGY route comparison</title>
<style>
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID}}}
body{{background:var(--surface);color:var(--ink);font:14px/1.5 system-ui,sans-serif;
      max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
p,li{{color:var(--ink2)}} img{{max-width:100%;height:auto;margin:8px 0}}
table.t{{border-collapse:collapse;font-size:13px;margin:8px 0}}
table.t th,table.t td{{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:right}}
table.t th:first-child,table.t td:first-child{{text-align:left}}
.note{{border-left:3px solid var(--grid);padding-left:10px}}
</style></head><body>
<h1>SPANGY route comparison — {texture} registrations</h1>
<p>{ps['subject'].nunique()} animals, {len(ps)} animal/hemisphere/session rows.
<b>ORIG</b> = SPANGY on the original mesh · <b>RESTEX</b> = ORIG textures resampled onto the reference mesh ·
<b>ONREF</b> = SPANGY on the surface resampled onto the reference mesh.</p>
<h2>Summary</h2>{html_table(summary)}{ref_note}
<h2>Agreement by distance from the reference</h2>{html_table(by_dist)}"""]
    labels = {
        "1": "RESTEX and ONREF share vertices, so they are compared vertex by vertex (area-weighted). "
             "Agreement is expected to fall for sessions far from the reference.",
        "2": "Every compared session. Low cells point to sessions to inspect visually.",
        "3": "Correlation of the z-filtered mean curvature: how much resampling the geometry "
             "changes the curvature SPANGY analyses.",
        "4": "Area share of each band, computed on each route's own mesh. ORIG involves no interpolation, "
             "so the route closer to the identity line distorts less.",
        "5": "Global SPANGY band powers. A systematic negative shift in B5–B6 means fine folds were "
             "flattened by resampling the surface.",
        "6": "Whether the band-power shift depends on how far the session is from the reference.",
    }
    for f in figs:
        k = os.path.basename(f)[0]
        parts.append(f"<h2>{os.path.basename(f)[2:-4].replace('_', ' ').capitalize()}</h2>"
                     f"<p>{labels.get(k, '')}</p>{img_tag(f)}")
        if k == "4" and share_err is not None:
            parts.append(html_table(share_err))
        if k == "5" and power is not None:
            parts.append(html_table(power, "{:.1f}"))
    parts.append("<p class='note'>Tables: per_session.csv, band_area_fractions.csv, band_powers.csv "
                 "in the same folder.</p></body></html>")
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

    ps = pd.read_csv(f"{d}/per_session.csv")
    ps["distance"] = ps["ref"].map(ses_num) - ps["session"].map(ses_num)
    fr = pd.read_csv(f"{d}/band_area_fractions.csv")
    pw_path = f"{d}/band_powers.csv"
    pw = pd.read_csv(pw_path) if os.path.isfile(pw_path) else None
    print(f"{len(ps)} sessions from {d}")

    fig_agreement_vs_distance(ps, out)
    fig_heatmap(ps, out)
    fig_curvature(ps, out)
    wide = fig_band_shares(fr, out)
    if pw is not None:
        pw = fig_band_power(pw, ps, out)
    figs = sorted(f"{out}/{f}" for f in os.listdir(out) if f.endswith(".png"))
    write_report(d, figs, ps, pw, wide, args.texture)


if __name__ == "__main__":
    main()
