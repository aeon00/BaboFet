"""
plot_band_power.py

Scrubs a directory of SPANGY-style "spectrum_results_*.txt" files, extracts
N and the absolute Band Powers (B4, B5, B6) from each file, and produces a
single overlaid line plot showing how each band's power evolves with N.

Edit the CONFIG block below, then just run:
    python plot_band_power.py
"""

import glob
import os
import re

import matplotlib.pyplot as plt

# =====================================================================
# CONFIG - edit these before running
# =====================================================================
INPUT_FOLDER = "/home/INT/dienye.h/python_files/Babofet/sub-Borgne/analysis/n_analysis"                          # folder containing the result text files
FILE_PATTERN = "spectrum_results_*.txt"     # glob pattern to match files
OUTPUT_PATH = "/home/INT/dienye.h/python_files/Babofet/sub-Borgne/analysis/n_analysis/band_power_vs_N.png"         # output PNG path
PLOT_TITLE = "Band Power Evolution with N"

B4_COLOR = "blue"
B5_COLOR = "green"
B6_COLOR = "red"
# =====================================================================

# Regex patterns matching the file format, e.g.:
#   N=1000
#   Band Powers: B4=214.7403242538291, B5=206.24762146789683, B6=0
N_PATTERN = re.compile(r"^\s*N\s*=\s*([-\d.eE+]+)", re.MULTILINE)
BAND_POWER_PATTERN = re.compile(
    r"Band Powers:\s*"
    r"B4\s*=\s*([-\d.eE+]+)\s*,\s*"
    r"B5\s*=\s*([-\d.eE+]+)\s*,\s*"
    r"B6\s*=\s*([-\d.eE+]+)"
)


def parse_file(filepath):
    """Extract (N, B4, B5, B6) from a single result file. Returns None if
    the required fields aren't found."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    n_match = N_PATTERN.search(text)
    bp_match = BAND_POWER_PATTERN.search(text)

    if not n_match or not bp_match:
        print(f"  [skip] Could not find N and/or Band Powers in {filepath}")
        return None

    n = float(n_match.group(1))
    b4, b5, b6 = (float(bp_match.group(i)) for i in (1, 2, 3))
    return n, b4, b5, b6


def scrub_folder(folder, pattern):
    filepaths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not filepaths:
        raise SystemExit(f"No files matching '{pattern}' found in {folder}")

    records = []
    for fp in filepaths:
        result = parse_file(fp)
        if result is not None:
            records.append(result)

    if not records:
        raise SystemExit("No valid records parsed from any file.")

    # Sort by N so the line plot evolves in order
    records.sort(key=lambda r: r[0])
    return records


def make_plot(records, out_path, title):
    n_vals = [r[0] for r in records]
    b4_vals = [r[1] for r in records]
    b5_vals = [r[2] for r in records]
    b6_vals = [r[3] for r in records]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(n_vals, b4_vals, color=B4_COLOR, marker="o", linewidth=2,
             markersize=6, label="B4")
    ax.plot(n_vals, b5_vals, color=B5_COLOR, marker="o", linewidth=2,
             markersize=6, label="B5")
    ax.plot(n_vals, b6_vals, color=B6_COLOR, marker="o", linewidth=2,
             markersize=6, label="B6")

    ax.set_xlabel("N")
    ax.set_ylabel("Band Power")
    ax.set_title(title)
    ax.legend(title="Band")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


def main():
    records = scrub_folder(INPUT_FOLDER, FILE_PATTERN)

    print("\nParsed records (N, B4, B5, B6):")
    for r in records:
        print(f"  N={r[0]:g}  B4={r[1]:.4f}  B5={r[2]:.4f}  B6={r[3]:.4f}")

    make_plot(records, OUTPUT_PATH, PLOT_TITLE)


if __name__ == "__main__":
    main()