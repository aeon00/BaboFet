#!/usr/bin/env python
"""
SPANGY spectral analysis across all BaboFet subjects.

Layout assumed (auto-detected, several fallbacks tried):

    <ROOT>/sub-<NAME>/sub-<NAME>/meshes/*.surf.gii      <- primary
    <ROOT>/sub-<NAME>/meshes/*.surf.gii                 <- fallback
    <ROOT>/sub-<NAME>/all_mesh_combined/*.surf.gii      <- fallback (sub-Borgne)

Outputs, per subject:

    <ROOT>/sub-<NAME>/analysis/spangy/
        plots/                      spectrum figures
        textures/                   spangy_dom_band_*.gii
        curvature/principal/        principal curvature textures
        curvature/mean/             z-filtered mean curvature textures
        results/per_file/*.json     one row per mesh (authoritative)
        results/chunk_<id>.csv      convenience dump per SLURM task
        results/sub-<NAME>_spangy_results.csv   <- written by --merge

Pooled:

    <ROOT>/analysis_pooled/all_subjects_spangy_results.csv

Usage:
    python spangy_babofet_all_subjects.py                 # process (SLURM array aware)
    python spangy_babofet_all_subjects.py --merge         # build per-subject + pooled CSVs
    python spangy_babofet_all_subjects.py --subjects sub-Aziza sub-Borgne
    python spangy_babofet_all_subjects.py --list          # dry run: show work plan
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')          # no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh

import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import slam.spangy as spgy

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT = '/envau/work/meca/users/dienye.h/python_files/Babofet'
POOLED_DIR = os.path.join(ROOT, 'analysis_pooled')
N_EIGENPAIRS = 4000
N_BANDS = 7                     # B0 … B6

# Candidate mesh sub-directories, relative to <ROOT>/<subject>, tried in order.
# '{sub}' is substituted with the subject folder name.
MESH_SUBDIRS = [
    os.path.join('{sub}', 'meshes'),
    'meshes',
    'all_mesh_combined',
    '',
]


# ── Discovery ─────────────────────────────────────────────────────────────────
def find_subjects(root, wanted=None):
    """Return sorted subject folder names (sub-*) under root."""
    subs = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(root, 'sub-*'))
        if os.path.isdir(p)
    )
    if wanted:
        missing = set(wanted) - set(subs)
        if missing:
            print(f"Warning: requested subjects not found under {root}: {sorted(missing)}")
        subs = [s for s in subs if s in set(wanted)]
    return subs


def find_mesh_dir(root, subject):
    """First candidate directory that actually contains .surf.gii files."""
    for pattern in MESH_SUBDIRS:
        candidate = os.path.join(root, subject, pattern.format(sub=subject))
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, '*.surf.gii')):
            return candidate
    return None


def list_meshes(mesh_dir):
    """White-matter surfaces for both hemispheres, deterministic order."""
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(mesh_dir, '*.surf.gii')))
    hemi = [f for f in files if 'hemi-L' in f or 'hemi-R' in f]
    white = [f for f in hemi if 'white' in f]
    # Prefer white surfaces; fall back to any hemisphere-tagged surface, then anything.
    return white or hemi or files


def build_job_list(root, wanted=None):
    """[(subject, mesh_dir, filename), ...] across all subjects."""
    jobs = []
    for subject in find_subjects(root, wanted):
        mesh_dir = find_mesh_dir(root, subject)
        if mesh_dir is None:
            print(f"Warning: no .surf.gii found for {subject} — skipped")
            continue
        meshes = list_meshes(mesh_dir)
        if not meshes:
            print(f"Warning: {subject}: directory found but no usable surfaces — skipped")
            continue
        print(f"  {subject}: {len(meshes)} surfaces in {mesh_dir}")
        jobs.extend((subject, mesh_dir, f) for f in meshes)
    return jobs


# ── Output paths ──────────────────────────────────────────────────────────────
def subject_dirs(root, subject):
    base = os.path.join(root, subject, 'analysis', 'spangy')
    d = {
        'base':       base,
        'plots':      os.path.join(base, 'plots'),
        'textures':   os.path.join(base, 'textures'),
        'curv_princ': os.path.join(base, 'curvature', 'principal'),
        'curv_mean':  os.path.join(base, 'curvature', 'mean'),
        'results':    os.path.join(base, 'results'),
        'per_file':   os.path.join(base, 'results', 'per_file'),
    }
    for path in d.values():
        os.makedirs(path, exist_ok=True)
    return d


def parse_ids(subject, filename):
    """participant_session label, plus session and hemisphere fields."""
    hemisphere = 'left' if 'hemi-L' in filename else ('right' if 'hemi-R' in filename else 'unknown')
    m = re.search(r'(ses-[A-Za-z0-9]+)', filename)
    session = m.group(1) if m else 'ses-NA'
    return f'{subject}_{session}_{hemisphere}', session, hemisphere


# ── Geometry helpers ──────────────────────────────────────────────────────────
def get_hull_area(mesh):
    return float(trimesh.convex.convex_hull(mesh).area)


def get_gyrification_index(mesh):
    hull_area = get_hull_area(mesh)
    return float(mesh.area) / hull_area, hull_area


# ── Core processing ───────────────────────────────────────────────────────────
def process_single_file(subject, mesh_dir, filename, dirs, n_eigen=N_EIGENPAIRS, force=False):
    try:
        start_time = time.time()
        participant_session, session, hemisphere = parse_ids(subject, filename)
        print(f"\nProcessing: {subject} / {filename}  →  {participant_session}")

        # ── Skip if already processed (safe SLURM resubmission) ───────────────
        row_path = os.path.join(dirs['per_file'], f'{participant_session}.json')
        texture_out = os.path.join(dirs['textures'], f'spangy_dom_band_{participant_session}.gii')
        if not force and os.path.exists(row_path) and os.path.exists(texture_out):
            print(f"  Skipping {participant_session} — already processed.")
            with open(row_path) as fh:
                return json.load(fh)

        mesh_file = os.path.join(mesh_dir, filename)
        if not os.path.exists(mesh_file):
            print(f"  Error: file not found: {mesh_file}")
            return None

        mesh = sio.load_mesh(mesh_file)
        mesh.apply_transform(mesh.principal_inertia_transform)

        # Eigenpairs
        print("  Computing eigenpairs...")
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, n_eigen)

        # Curvature
        print("  Computing curvature...")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)

        sio.write_texture(
            stex.TextureND(PrincipalCurvatures),
            os.path.join(dirs['curv_princ'], f'principal_curv_{participant_session}.gii')
        )

        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        sio.write_texture(
            tex_mean_curv,
            os.path.join(dirs['curv_mean'], f'filt_mean_curv_{participant_session}.gii')
        )
        filt_mean_curv = tex_mean_curv.darray.squeeze()
        total_mean_curv = float(np.sum(filt_mean_curv))

        # Spectrum
        print("  Computing spectrum...")
        grouped_spectrum, group_indices, coefficients, nlevels = \
            spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
        levels = len(group_indices)

        # Plot  (note: frequency = sqrt(eigVal / (2*pi)) — parenthesised)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        frequency = np.sqrt(eigVal / (2 * np.pi))
        ax1.scatter(frequency, coefficients, marker='o', s=20, linewidths=0.5)
        ax1.set_xlabel('Frequency (m⁻¹)')
        ax1.set_ylabel('Coefficients')
        ax2.scatter(frequency[1:], coefficients[1:], marker='o', s=20, linewidths=0.5)
        ax2.set_xlabel('Frequency (m⁻¹)')
        ax2.set_ylabel('Coefficients')
        ax3.bar(np.arange(0, levels), grouped_spectrum)
        ax3.set_xlabel('Spangy Frequency Bands')
        ax3.set_ylabel('Power Spectrum')
        fig.suptitle(participant_session)
        plt.tight_layout()
        fig.savefig(os.path.join(dirs['plots'], f'{participant_session}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Whole-brain metrics
        mL_in_MM3, CM2_in_MM2 = 1000.0, 100.0
        volume = float(mesh.volume)
        surface_area = float(mesh.area)
        afp = float(np.sum(grouped_spectrum[1:]))
        print(f"  Volume={volume / mL_in_MM3:.1f} mL  "
              f"Area={surface_area / CM2_in_MM2:.1f} cm²  AFP={afp:.4f}")

        # Local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(
            coefficients, filt_mean_curv, levels, group_indices, eigVects
        )
        sio.write_texture(stex.TextureND(loc_dom_band), texture_out)

        gyrification_index, hull_area = get_gyrification_index(mesh)
        execution_time = time.time() - start_time
        print(f"  Done in {execution_time:.2f}s")

        row = {
            'subject':              subject,
            'session':              session,
            'hemisphere':           hemisphere,
            'participant_session':  participant_session,
            'filename':             filename,
            'mesh_dir':             mesh_dir,
            'n_vertices':           int(mesh.vertices.shape[0]),
            'n_eigenpairs':         int(n_eigen),
            'n_levels':             int(levels),
            'total_mean_curvature': total_mean_curv,
            'gyrification_index':   float(gyrification_index),
            'hull_area':            float(hull_area),
            'volume_mm3':           volume,
            'surface_area_mm2':     surface_area,
            'volume_ml':            volume / mL_in_MM3,
            'surface_area_cm2':     surface_area / CM2_in_MM2,
            'analyze_folding_power': afp,
            'processing_time_s':    execution_time,
        }
        # Band powers, robust to spectra with fewer than N_BANDS levels
        for i in range(N_BANDS):
            row[f'band_power_B{i}'] = float(grouped_spectrum[i]) if i < len(grouped_spectrum) else np.nan

        with open(row_path, 'w') as fh:
            json.dump(row, fh, indent=2)
        return row

    except Exception as e:
        print(f"  Error processing {subject}/{filename}: {e}")
        return None


# ── Merge ─────────────────────────────────────────────────────────────────────
def merge_results(root, wanted=None):
    """Collect per-file JSON rows into per-subject CSVs and one pooled CSV."""
    os.makedirs(POOLED_DIR, exist_ok=True)
    pooled = []

    for subject in find_subjects(root, wanted):
        per_file_dir = os.path.join(root, subject, 'analysis', 'spangy', 'results', 'per_file')
        rows = []
        for path in sorted(glob.glob(os.path.join(per_file_dir, '*.json'))):
            try:
                with open(path) as fh:
                    rows.append(json.load(fh))
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
        if not rows:
            print(f"  {subject}: no results yet")
            continue

        df = pd.DataFrame(rows).sort_values(['session', 'hemisphere'])
        out = os.path.join(root, subject, 'analysis', 'spangy', 'results',
                           f'{subject}_spangy_results.csv')
        df.to_csv(out, index=False)
        print(f"  {subject}: {len(df)} rows → {out}")
        pooled.append(df)

    if not pooled:
        print("No results found to pool.")
        return None

    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df = pooled_df.drop_duplicates(subset='participant_session', keep='last')
    pooled_df = pooled_df.sort_values(['subject', 'session', 'hemisphere'])
    pooled_path = os.path.join(POOLED_DIR, 'all_subjects_spangy_results.csv')
    pooled_df.to_csv(pooled_path, index=False)
    print(f"\nPooled: {len(pooled_df)} rows from {pooled_df['subject'].nunique()} subjects "
          f"→ {pooled_path}")
    return pooled_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT, help='BaboFet root directory')
    ap.add_argument('--subjects', nargs='*', default=None,
                    help='Restrict to these subject folders (default: all sub-*)')
    ap.add_argument('--merge', action='store_true',
                    help='Merge existing per-file results into per-subject + pooled CSVs')
    ap.add_argument('--list', action='store_true',
                    help='Print the work plan and exit (no processing)')
    ap.add_argument('--n-eigen', type=int, default=N_EIGENPAIRS)
    ap.add_argument('--force', action='store_true',
                    help='Recompute even if outputs already exist')
    args = ap.parse_args()

    if args.merge:
        merge_results(args.root, args.subjects)
        return

    print(f"Scanning {args.root} ...")
    jobs = build_job_list(args.root, args.subjects)
    if not jobs:
        print("No meshes found. Check --root / directory layout.")
        sys.exit(1)
    print(f"Total: {len(jobs)} meshes across "
          f"{len({s for s, _, _ in jobs})} subjects")

    # SLURM array chunking (falls back gracefully to a single process)
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    elif 'SLURM_ARRAY_TASK_MIN' in os.environ and 'SLURM_ARRAY_TASK_MAX' in os.environ:
        n_tasks = (int(os.environ['SLURM_ARRAY_TASK_MAX'])
                   - int(os.environ['SLURM_ARRAY_TASK_MIN']) + 1)
    else:
        n_tasks = 1
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))
    idx = task_id - task_min

    # Strided assignment: interleaves subjects/sessions so heavy meshes
    # (e.g. sub-Borgne ses-01) spread across tasks instead of piling on one.
    chunk = jobs[idx::n_tasks]
    print(f"Task {idx + 1}/{n_tasks} — {len(chunk)} meshes assigned")

    if args.list:
        for subject, mesh_dir, filename in chunk:
            print(f"  {subject}\t{filename}")
        return

    dirs_cache = {}
    results = []
    for subject, mesh_dir, filename in chunk:
        if subject not in dirs_cache:
            dirs_cache[subject] = subject_dirs(args.root, subject)
        row = process_single_file(subject, mesh_dir, filename, dirs_cache[subject],
                                  n_eigen=args.n_eigen, force=args.force)
        if row is not None:
            results.append(row)

    # Per-task convenience dump, one file per subject touched by this task
    if results:
        df = pd.DataFrame(results)
        for subject, sub_df in df.groupby('subject'):
            out = os.path.join(dirs_cache[subject]['results'], f'chunk_{task_id}.csv')
            sub_df.to_csv(out, index=False)
            print(f"Saved {len(sub_df)} rows → {out}")
    else:
        print(f"Warning: no results for task {task_id}")


if __name__ == '__main__':
    main()
