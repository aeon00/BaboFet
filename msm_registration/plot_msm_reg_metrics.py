#!/usr/bin/env python3
"""
plot_msm_reg_metrics.py — figures and an HTML report from msm_reg_metrics.csv.

Usage (cluster or locally; pandas, matplotlib, scipy):
    python plot_msm_reg_metrics.py
    python plot_msm_reg_metrics.py --csv /path/to/msm_reg_metrics.csv

Writes next to the CSV, in figures/:
    1_sulc_vs_dpfstar.png       every pair registered both ways: sulc (x) vs DPF-star (y)
    2_curvature_before_after.png mean-curvature correlation before/after MSM, per texture
    3_distortion_by_gap.png     Jacobian and area change for consecutive vs skipped sessions
    4_heatmap_<metric>.png      per animal/hemisphere and pair (sulc and DPF-star)
    5_band_dice.png             SPANGY band Dice per band (when --bands was used)
    6_roi_dice.png              painted ROI Dice (when --roi-pattern was used)
and report.html (figures + summary tables, one self-contained file).
"""
import argparse
import base64
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = "/envau/work/meca/users/dienye.h/python_files/Babofet"

# validated categorical palette, fixed order: slot 1 = sulc, slot 2 = DPF-star
TEX = {"sulc": "#2a78d6", "dpfstar": "#eb6834"}
TEX_LABEL = {"sulc": "sulc", "dpfstar": "DPF-star"}
TEX_MARK = {"sulc": "o", "dpfstar": "s"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})

# metric -> (label, higher is better?)
METRICS = {
    "jacobian_mean": ("Mean |log2 J| (area-weighted)", False),
    "area_diff_abs_pct_median": ("Median |area change| (%)", False),
    "flipped_faces": ("Folded triangles", False),
    "corr_tex_msm": ("r, own texture (MSM output)", True),
    "corr_curv_after": ("r, mean curvature after", True),
    "corr_other_after": ("r, the other texture after", True),
    "dice_sulcal": ("Dice, sulcal (curvature sign)", True),
    "dice_gyral": ("Dice, gyral (curvature sign)", True),
    "agree_sign": ("Sulcal/gyral agreement (%)", True),
    "agree_band": ("Dominant-band agreement (%)", True),
}


def jitter(n, w=0.16, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def paired(df):
    """One row per pair with <metric>_sulc and <metric>_dpfstar columns."""
    key = ["subject", "hemi", "input", "reference", "gap"]
    if df["texture"].nunique() < 2:
        return None
    w = df.pivot_table(index=key, columns="texture",
                       values=[c for c in METRICS if c in df]).reset_index()
    w.columns = [f"{a}_{b}" if b else a for a, b in w.columns]
    return w


# --- figures -----------------------------------------------------------------------
def fig_paired(w, out):
    cols = [c for c in ("jacobian_mean", "corr_curv_after", "corr_other_after",
                        "dice_sulcal", "agree_sign", "agree_band")
            if f"{c}_sulc" in w and f"{c}_dpfstar" in w]
    if not cols:
        return None
    n = len(cols)
    nc = min(n, 3)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes = axes.ravel()
    for ax, c in zip(axes, cols):
        x, y, g = w[f"{c}_sulc"], w[f"{c}_dpfstar"], w["gap"]
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=1, ls="--")
        for skip, fill in ((False, True), (True, False)):
            s = (g > 1) == skip
            ax.scatter(x[s], y[s], s=22, facecolor=INK2 if fill else "none",
                       edgecolor=INK2, linewidth=1, alpha=0.8)
        better_dpf = (y > x) if METRICS[c][1] else (y < x)
        ax.set_title(METRICS[c][0], loc="left")
        ax.text(0.02, 0.97, f"DPF-star better in {int(better_dpf.sum())}/{int((x.notna() & y.notna()).sum())}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        ax.set_xlabel("sulc registration")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("DPF-star registration")
    h = [plt.Line2D([], [], ls="", marker="o", color=INK2, label="consecutive sessions"),
         plt.Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="session skipped"),
         plt.Line2D([], [], ls="--", color=INK2, lw=1, label="equal")]
    fig.suptitle("Same pairs registered with sulc vs DPF-star", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=3)
    return save(fig, f"{out}/1_sulc_vs_dpfstar.png")


def fig_before_after(df, out):
    if not {"corr_curv_before", "corr_curv_after"} <= set(df):
        return None
    texs = [t for t in TEX if t in set(df["texture"])]
    fig, axes = plt.subplots(1, len(texs), figsize=(4.2 * len(texs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, texs):
        d = df[df["texture"] == t]
        for _, r in d.iterrows():
            ax.plot([0, 1], [r["corr_curv_before"], r["corr_curv_after"]],
                    color=TEX[t], lw=0.8, alpha=0.35)
        for k, c in enumerate(("corr_curv_before", "corr_curv_after")):
            ax.scatter(np.full(len(d), k) + jitter(len(d), 0.04), d[c], s=16,
                       color=TEX[t], marker=TEX_MARK[t], edgecolor=SURFACE, linewidth=0.6)
            ax.plot([k - 0.18, k + 0.18], [d[c].median()] * 2, color=INK, lw=2.5)
        ax.set_xticks([0, 1], ["before MSM", "after MSM"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{TEX_LABEL[t]} registration", loc="left")
    axes[0].set_ylabel("r, mean curvature vs reference\n(feature not used by MSM)")
    fig.suptitle("Alignment of an independent feature (bar = median)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return save(fig, f"{out}/2_curvature_before_after.png")


def fig_gap(df, out):
    cols = [c for c in ("jacobian_mean", "area_diff_abs_pct_median", "corr_curv_after", "dice_sulcal")
            if c in df]
    df = df.assign(gapc=np.where(df["gap"] > 1, "skipped", "consecutive"))
    texs = [t for t in TEX if t in set(df["texture"])]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        for gi, gname in enumerate(("consecutive", "skipped")):
            for ti, t in enumerate(texs):
                d = df[(df["gapc"] == gname) & (df["texture"] == t)][c].dropna()
                xpos = gi + (ti - (len(texs) - 1) / 2) * 0.3
                ax.scatter(xpos + jitter(len(d), 0.07, gi * 7 + ti), d, s=14, color=TEX[t],
                           marker=TEX_MARK[t], edgecolor=SURFACE, linewidth=0.5, alpha=0.85)
                if len(d):
                    ax.plot([xpos - 0.11, xpos + 0.11], [d.median()] * 2, color=INK, lw=2)
        ax.set_xticks([0, 1], ["consecutive", "skipped"])
        ax.set_title(METRICS[c][0], loc="left")
    h = [plt.Line2D([], [], ls="", marker=TEX_MARK[t], color=TEX[t], label=TEX_LABEL[t]) for t in texs]
    h.append(plt.Line2D([], [], color=INK, lw=2, label="median"))
    fig.suptitle("Pairs with a skipped session vs consecutive sessions", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=len(h))
    return save(fig, f"{out}/3_distortion_by_gap.png")


def fig_heatmap(df, metric, out):
    if metric not in df:
        return None
    texs = [t for t in TEX if t in set(df["texture"])]
    df = df.assign(row=df["subject"].str.replace("sub-", "") + " " + df["hemi"],
                   pair=df["input"].str[4:] + "→" + df["reference"].str[4:])
    piv = {t: df[df["texture"] == t].pivot_table(index="row", columns="pair", values=metric)
           for t in texs}
    cols = sorted(set().union(*[p.columns for p in piv.values()]))
    rows = sorted(set().union(*[p.index for p in piv.values()]))
    vals = np.concatenate([p.reindex(index=rows, columns=cols).values.ravel() for p in piv.values()])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    cmap = "Blues_r" if not METRICS[metric][1] else "Blues"   # darker = better
    fig, axes = plt.subplots(len(texs), 1, figsize=(1.4 + 0.55 * len(cols), 0.8 + 0.34 * len(rows) * len(texs) + 0.6 * len(texs)))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, texs):
        m = piv[t].reindex(index=rows, columns=cols).values
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    frac = (m[i, j] - vmin) / (vmax - vmin + 1e-12)
                    dark = frac > 0.6 if METRICS[metric][1] else frac < 0.4
                    ax.text(j, i, f"{m[i, j]:.2f}" if abs(m[i, j]) < 10 else f"{m[i, j]:.0f}",
                            ha="center", va="center", fontsize=6.5, color=SURFACE if dark else INK)
        ax.set_title(f"{METRICS[metric][0]} — {TEX_LABEL[t]} registration (blank = no pair)", loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.outline.set_visible(False)
    fig.tight_layout()
    return save(fig, f"{out}/4_heatmap_{metric}.png")


def fig_band_dice(df, out):
    cols = [f"dice_B{b}" for b in (3, 4, 5, 6) if f"dice_B{b}" in df]
    if not cols:
        return None
    texs = [t for t in TEX if t in set(df["texture"])]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, c in enumerate(cols):
        for ti, t in enumerate(texs):
            d = df[df["texture"] == t][c].dropna()
            xpos = k + (ti - (len(texs) - 1) / 2) * 0.3
            ax.scatter(xpos + jitter(len(d), 0.07, k * 5 + ti), d, s=14, color=TEX[t],
                       marker=TEX_MARK[t], edgecolor=SURFACE, linewidth=0.5, alpha=0.85)
            if len(d):
                ax.plot([xpos - 0.11, xpos + 0.11], [d.median()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(cols)), [c.replace("dice_", "") for c in cols])
    ax.set_ylabel("Dice (%)")
    ax.set_title("SPANGY dominant band moved by MSM vs next session (bar = median)", loc="left")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker=TEX_MARK[t], color=TEX[t], label=TEX_LABEL[t])
                       for t in texs], loc="lower right")
    fig.tight_layout()
    return save(fig, f"{out}/5_band_dice.png")


def fig_roi(df, out):
    cols = [c for c in df if c.startswith("dice_roi_")]
    if not cols:
        return None
    texs = [t for t in TEX if t in set(df["texture"])]
    fig, ax = plt.subplots(figsize=(3 + 1.6 * len(cols), 3.6))
    for k, c in enumerate(cols):
        for ti, t in enumerate(texs):
            d = df[df["texture"] == t][c].dropna()
            xpos = k + (ti - (len(texs) - 1) / 2) * 0.3
            ax.scatter(xpos + jitter(len(d), 0.07, ti), d, s=18, color=TEX[t], marker=TEX_MARK[t],
                       edgecolor=SURFACE, linewidth=0.5)
            if len(d):
                ax.plot([xpos - 0.11, xpos + 0.11], [d.median()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(cols)), [c.replace("dice_roi_", "") for c in cols])
    ax.set_ylabel("Dice (%)")
    ax.set_title("Painted ROI overlap after MSM", loc="left")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker=TEX_MARK[t], color=TEX[t], label=TEX_LABEL[t])
                       for t in texs])
    fig.tight_layout()
    return save(fig, f"{out}/6_roi_dice.png")


# --- report ------------------------------------------------------------------------
def img(path):
    with open(path, "rb") as fh:
        return f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}">'


def table(df, fmt="{:.3f}"):
    return df.to_html(float_format=lambda v: fmt.format(v), border=0, classes="t", na_rep="–")


def write_report(d, figs, df, w):
    cols = [c for c in METRICS if c in df] + [c for c in df if c.startswith("dice_B") or c.startswith("dice_roi_")]
    cols += [c for c in ("corr_tex_before", "corr_curv_before") if c in df]
    med = df.groupby("texture")[cols].median().T
    med.index = [METRICS.get(c, (c,))[0] for c in med.index]
    med.columns = [TEX_LABEL.get(c, c) for c in med.columns]

    test = None
    if w is not None:
        rows = []
        for c in cols:
            a, b = f"{c}_sulc", f"{c}_dpfstar"
            if a not in w or b not in w:
                continue
            x = w[[a, b]].dropna()
            if len(x) < 6:
                continue
            diff = x[b] - x[a]
            try:
                with np.errstate(all="ignore"):
                    pval = wilcoxon(x[a], x[b]).pvalue if (diff != 0).any() else np.nan
            except ValueError:
                pval = np.nan
            better = (diff > 0) if METRICS.get(c, ("", True))[1] else (diff < 0)
            rows.append((METRICS.get(c, (c,))[0], len(x), diff.median(), int(better.sum()), pval))
        if rows:
            test = pd.DataFrame(rows, columns=["metric", "pairs", "median DPF-star − sulc",
                                               "DPF-star better (n)", "Wilcoxon p"]).set_index("metric")

    n_pairs = df.groupby("texture").size().to_dict()
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MSM registration metrics</title>
<style>
body{{background:{SURFACE};color:{INK};font:14px/1.5 system-ui,sans-serif;max-width:1150px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} p,li{{color:{INK2}}}
img{{max-width:100%;height:auto;margin:8px 0}}
table.t{{border-collapse:collapse;font-size:13px;margin:8px 0}}
table.t th,table.t td{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right}}
table.t th:first-child,table.t td:first-child{{text-align:left}}
.note{{border-left:3px solid {GRID};padding-left:10px}}
</style></head><body>
<h1>MSM registration metrics</h1>
<p>{df['subject'].nunique()} animals; registrations per texture: {", ".join(f"{TEX_LABEL.get(k, k)} {v}" for k, v in n_pairs.items())}.
Metrics follow msm_reg_stats.py: Jacobian distortion, per-face area change, Pearson correlation of the
registration texture, and Dice of labels moved by nearest neighbour. Mean curvature and its sign are not
used by either registration, so they compare sulc and DPF-star fairly.</p>
<h2>Medians per registration texture</h2>{table(med)}"""]
    if test is not None:
        parts.append("<h2>Paired comparison, same pairs</h2>" + table(test, "{:.3g}") +
                     "<p class='note'>Pairs come from the same animals and overlap in sessions, so they are not "
                     "independent; treat p-values as descriptive.</p>")
    notes = {
        "1": "Each point is one pair registered both ways. For distortion, lower is better; for correlation and Dice, higher is better.",
        "2": "Mean curvature was not used by either registration: the gain from before to after is the fair test of alignment.",
        "3": "Skipping a session means more growth between the two brains, so more distortion is expected.",
        "4": "Every pair, per animal and hemisphere: spot the pair that stands out.",
        "5": "Bands change with age, so this measures how similar the folding pattern is after alignment, not a ground truth.",
        "6": "Overlap of a sulcus painted on both sessions: the closest thing to a ground-truth landmark.",
    }
    for f in figs:
        k = os.path.basename(f)[0]
        parts.append(f"<h2>{os.path.basename(f)[2:-4].replace('_', ' ').capitalize()}</h2>"
                     f"<p>{notes.get(k, '')}</p>{img(f)}")
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
    df = pd.read_csv(args.csv)
    print(f"{len(df)} rows from {args.csv}")
    w = paired(df)

    figs = [fig_paired(w, out) if w is not None else None,
            fig_before_after(df, out), fig_gap(df, out),
            fig_heatmap(df, "jacobian_mean", out), fig_heatmap(df, "dice_sulcal", out),
            fig_band_dice(df, out), fig_roi(df, out)]
    write_report(d, [f for f in figs if f], df, w)


if __name__ == "__main__":
    main()
