#!/usr/bin/env python3
"""
plot_smoothing.py
=================

For each smoothing-sweep results CSV (one subject per file), produce:
  1. <subject>_features.png      - every numeric feature vs smoothing level
  2. <subject>_band_powers.png   - all band powers B0-B6 on one axes vs smoothing
  3. <subject>_band_powers_B3-B6.png - only B3-B6 vs smoothing

Usage:
  python plot_smoothing.py                 # uses CONFIG["input_dir"]
  python plot_smoothing.py /path/to/csvs   # override input dir
  python plot_smoothing.py /path/to/csvs /path/to/out
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless-safe
import matplotlib.pyplot as plt

CONFIG = {
    "input_dir":  "/home/INT/dienye.h/python_files/Babofet/sub-Borgne (old)/sub-Borgne-seg/sub-Borgne-hemi/new_hemi_meshes_10_smoothing_iterations/left/spangy_smoothing/results",
    "input_glob": "*smoothing_results.csv",
    "out_dir":    "/home/INT/dienye.h/python_files/Babofet/sub-Borgne (old)/sub-Borgne-seg/sub-Borgne-hemi/new_hemi_meshes_10_smoothing_iterations/left/spangy_smoothing/results/smoothing_plots",
    "x_col":      "smoothing_iter",
    "log_y_bands": False,  # plot band powers on a linear axis (raw CSV values)
    "exclude":    {"smoothing_iter", "smoothing_dt"},   # not plotted as features
}

# Fixed per-band colours (matches the BaboFet dominant-band convention).
# B3 yellow, B4 blue, B5 green, B6 red; B0-B2 given distinct colours.
BAND_COLORS = {
    "B0": "#444444",   # dark grey
    "B1": "#8e44ad",   # purple
    "B2": "#e67e22",   # orange
    "B3": "gold",      # yellow
    "B4": "#1f77b4",   # blue
    "B5": "#2ca02c",   # green
    "B6": "#d62728",   # red
    "B7": "#17becf",   # cyan (only if an 8th band ever appears)
}
DEFAULT_COLOR = "#1f4e79"


def band_of(col):
    """'B3' from 'band_power_B3' or 'band_frac_B3'; None otherwise."""
    for prefix in ("band_power_", "band_frac_"):
        if col.startswith(prefix):
            return col[len(prefix):]
    return None


def subject_label(df, csv_path):
    if "participant_session" in df.columns and df["participant_session"].notna().any():
        return str(df["participant_session"].dropna().iloc[0])
    return os.path.splitext(os.path.basename(csv_path))[0]


def numeric_features(df):
    feats = []
    for c in df.columns:
        if c in CONFIG["exclude"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def band_power_cols(df, ks):
    return [f"band_power_B{k}" for k in ks if f"band_power_B{k}" in df.columns]


def plot_features(df, x, subject, out_dir):
    feats = numeric_features(df)
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, feats):
        ax.plot(x, df[feat], marker="o",
                color=BAND_COLORS.get(band_of(feat), DEFAULT_COLOR))
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel(CONFIG["x_col"])
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    for ax in axes[len(feats):]:          # hide unused panels
        ax.axis("off")
    fig.suptitle(f"{subject} — features vs smoothing", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{subject}_features.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bands(df, x, subject, out_dir, cols, fname, title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for col in cols:
        label = col.replace("band_power_", "")
        ax.plot(x, df[col], marker="o", label=label, linewidth=2,
                color=BAND_COLORS.get(label, DEFAULT_COLOR),
                markeredgecolor="#333333", markeredgewidth=0.5)
    ax.set_xlabel("Smoothing iterations")
    ax.set_ylabel("Band power" + ("  (log scale)" if CONFIG["log_y_bands"] else ""))
    ax.set_xticks(x)
    if CONFIG["log_y_bands"]:
        ax.set_yscale("log")
    ax.set_title(f"{subject} — {title}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="Band", frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def process_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path).sort_values(CONFIG["x_col"]).reset_index(drop=True)
    x = df[CONFIG["x_col"]].values
    subject = subject_label(df, csv_path)
    print(f"\n{subject}  ({os.path.basename(csv_path)}, {len(df)} smoothing levels)")

    made = []
    made.append(plot_features(df, x, subject, out_dir))
    made.append(plot_bands(df, x, subject, out_dir,
                           band_power_cols(df, range(0, 7)),
                           f"{subject}_band_powers.png", "all band powers B0-B6"))
    made.append(plot_bands(df, x, subject, out_dir,
                           band_power_cols(df, range(3, 7)),
                           f"{subject}_band_powers_B3-B6.png", "band powers B3-B6"))
    for p in made:
        print(f"  -> {os.path.basename(p)}")
    return made


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else CONFIG["input_dir"]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else CONFIG["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(in_dir, CONFIG["input_glob"])))
    if not csvs:
        print(f"No CSVs matching {CONFIG['input_glob']} in {in_dir}")
        sys.exit(1)
    print(f"Found {len(csvs)} CSV(s) in {in_dir}")

    for csv_path in csvs:
        process_csv(csv_path, out_dir)
    print(f"\nAll plots written to: {out_dir}")


if __name__ == "__main__":
    main()
