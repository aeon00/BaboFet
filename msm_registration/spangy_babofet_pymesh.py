#!/usr/bin/env python
"""
SPANGY spectral analysis across all BaboFet subjects — Pymesh mesh layout.

Runs SPANGY on the SAME white-matter meshes used for the MSM registration, so
every texture it writes can be resampled with the registered spheres.

Code lives in the synced git repo:
    /envau/work/meca/users/dienye.h/BaboFet

Input meshes (read only):

    <MESH_ROOT>/sub-<NAME>/ses-<NN>/anat/
        sub-<NAME>_ses-<NN>_manualseg_extract_surf_hemi_left_wm.gii
        sub-<NAME>_ses-<NN>_manualseg_extract_surf_hemi_right_wm.gii

Outputs, per subject (separate from older SPANGY runs on other meshes):

    <DATA_ROOT>/sub-<NAME>/analysis/spangy_pymesh/
        plots/                      spectrum figures
        textures/                   spangy_dom_band_<sub>_<ses>_hemi-<L|R>.gii
                                    spangy_frecomposed_<sub>_<ses>_hemi-<L|R>.gii
                                      (one column per band B1..B<n>: curvature recomposed
                                       in that band; input of resample_spangy_bands.py)
                                    spangy_det_band_<sub>_<ses>_hemi-<L|R>.gii
                                      (locally determinant band, spangy_determinance.py)
        curvature/principal/        principal curvature textures
        curvature/mean/             z-filtered mean curvature textures
        results/per_file/*.csv      one row per mesh (authoritative): whole-brain metrics,
                                    band_power_B0..B6, band_prop_B1..B6 (share of folding
                                    power), dom_/det_area_pct_B1..B6 (+ _sulcal, _gyral):
                                    % of the surface area where each band is the dominant /
                                    determinant band
        results/chunk_<id>.csv      convenience dump per SLURM task
        results/sub-<NAME>_spangy_results.csv   <- written by --merge

Pooled:

    <DATA_ROOT>/analysis_pooled/all_subjects_spangy_pymesh_results.csv

Usage:
    python spangy_babofet_pymesh.py                 # process (SLURM array aware)
    python spangy_babofet_pymesh.py --merge         # per-subject + pooled CSVs
    python spangy_babofet_pymesh.py --subjects sub-Aziza sub-Borgne
    python spangy_babofet_pymesh.py --list          # dry run: show work plan
    python spangy_babofet_pymesh.py --onref         # run on surfaces_on_ref/sulc/ instead
                                                    # (outputs in analysis/spangy_onref_sulc/)
    python spangy_babofet_pymesh.py --onref --reg-texture dpfstar   # DPF-star registrations
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

from spangy_determinance import determinant_band   # same folder as this script

# --- Configuration -----------------------------------------------------------
MESH_ROOT = '/envau/work/meca/users/dienye.h/Pymesh/mesh'
DATA_ROOT = '/envau/work/meca/users/dienye.h/python_files/Babofet'
OUT_SUBDIR = os.path.join('analysis', 'spangy_pymesh')
POOLED_DIR = os.path.join(DATA_ROOT, 'analysis_pooled')
POOLED_NAME = 'all_subjects_spangy_pymesh_results.csv'
N_EIGENPAIRS = 4000
N_BANDS = 7                     # B0 ... B6

# Mesh files, relative to <MESH_ROOT>/<sub>/
MESH_RELGLOB = os.path.join('ses-*', 'anat', '*_hemi_*_wm.gii')

# --onref: surfaces already resampled onto each animal's reference mesh
# (resample_to_ref.py). Outputs then share one vertex numbering per animal/hemisphere.
# '{reg}' = which registrations were used: sulc or dpfstar (--reg-texture)
ONREF_MESH_ROOT = os.path.join(DATA_ROOT, 'surfaces_on_ref', '{reg}')
ONREF_RELGLOB = os.path.join('hemi-*', '*_on_ses-*.anat.surf.gii')
ONREF_OUT_SUBDIR = os.path.join('analysis', 'spangy_onref_{reg}')
ONREF_POOLED_NAME = 'all_subjects_spangy_onref_{reg}_results.csv'

# Subject folders matching any of these substrings are ignored (case-insensitive).
DEFAULT_EXCLUDE = ['(old)', '_old', '.old', 'backup']

HEMI_MAP = {'left': 'L', 'l': 'L', 'lh': 'L', 'right': 'R', 'r': 'R', 'rh': 'R'}


# --- Filename handling -------------------------------------------------------
def parse_ids(path):
    """
    (label, subject, session, hemisphere) from a Pymesh mesh path.

    sub-Aziza/ses-01/anat/sub-Aziza_ses-01_manualseg_extract_surf_hemi_left_wm.gii
        -> ('sub-Aziza_ses-01_hemi-L', 'sub-Aziza', 'ses-01', 'L')

    The label matches the names used by the MSM pipeline (<sub>_<ses>_hemi-<L|R>).
    """
    name = os.path.basename(path)
    sub = re.search(r'(sub-[^_/]+)', name) or re.search(r'(sub-[^_/]+)', path)
    ses = re.search(r'(ses-[^_/]+)', name) or re.search(r'(ses-[^_/]+)', path)
    side = re.search(r'hemi[_-](left|right|lh|rh|l|r)(?=[_.])', name, re.IGNORECASE)
    subject = sub.group(1) if sub else 'sub-NA'
    session = ses.group(1) if ses else 'ses-NA'
    hemi = HEMI_MAP[side.group(1).lower()] if side else 'unknown'
    return f'{subject}_{session}_hemi-{hemi}', subject, session, hemi


# --- Discovery ---------------------------------------------------------------
def find_subjects(mesh_root, wanted=None, exclude=None):
    """Sorted subject folder names (sub-*) under the mesh root."""
    exclude = [e.lower() for e in (exclude if exclude is not None else DEFAULT_EXCLUDE)]
    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mesh_root, 'sub-*'))
                  if os.path.isdir(p))
    if wanted:
        missing = set(wanted) - set(subs)
        if missing:
            print(f"Warning: requested subjects not found under {mesh_root}: {sorted(missing)}")
        return [s for s in subs if s in set(wanted)]
    kept = []
    for s in subs:
        if exclude and any(e and e in s.lower() for e in exclude):
            print(f"  {s}: excluded by name filter (use --exclude with no values to include)")
        else:
            kept.append(s)
    return kept


def build_job_list(mesh_root, wanted=None, exclude=None):
    """[(subject, mesh_path), ...] across all subjects and sessions."""
    jobs = []
    for subject in find_subjects(mesh_root, wanted, exclude):
        meshes = sorted(glob.glob(os.path.join(mesh_root, subject, MESH_RELGLOB)))
        if not meshes:
            print(f"Warning: no meshes for {subject} matching "
                  f"{os.path.join(mesh_root, subject, MESH_RELGLOB)}")
            continue
        n_ses = len({parse_ids(m)[2] for m in meshes})
        print(f"  {subject}: {len(meshes)} meshes across {n_ses} sessions")
        jobs.extend((subject, m) for m in meshes)
    return jobs


# --- Output paths ------------------------------------------------------------
def subject_dirs(root, subject):
    base = os.path.join(root, subject, OUT_SUBDIR)
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


# --- Geometry helpers --------------------------------------------------------
def get_hull_area(mesh):
    return float(trimesh.convex.convex_hull(mesh).area)


def get_gyrification_index(mesh):
    hull_area = get_hull_area(mesh)
    return float(mesh.area) / hull_area, hull_area


def vertex_areas(mesh):
    """Area represented by each vertex: one third of every triangle around it."""
    va = np.zeros(len(mesh.vertices))
    for k in range(3):
        np.add.at(va, mesh.faces[:, k], mesh.area_faces / 3.0)
    return va


def band_summary(grouped_spectrum, dom_band, det_band, areas):
    """
    Per band k = 0..6:
      band_power_Bk         SPANGY power of the band (grouped spectrum)
      band_prop_Bk          share of the folding power, power_k / sum(power_1..power_6)
                            (B0, the mean curvature offset, is left out as in AFP; k >= 1)
    Per band k = 1..6, for the dominant (dom_) and determinant (det_) maps:
      <map>_area_pct_Bk         % of the hemisphere surface where band k is the band
      <map>_area_pct_Bk_sulcal  % of the surface, sulcal part (negative labels)
      <map>_area_pct_Bk_gyral   % of the surface, gyral part (positive labels)
    Areas are true surface areas on the mesh, not vertex counts.
    """
    out = {}
    gs = np.asarray(grouped_spectrum, dtype=np.float64)
    folding = gs[1:].sum() if len(gs) > 1 else np.nan
    for i in range(N_BANDS):
        p = float(gs[i]) if i < len(gs) else np.nan
        out[f'band_power_B{i}'] = p
        if i >= 1:
            out[f'band_prop_B{i}'] = p / folding if folding else np.nan
    total = areas.sum()
    for name, lab in (('dom', dom_band), ('det', det_band)):
        lab = np.rint(np.asarray(lab, dtype=np.float64)).astype(int)
        for k in range(1, N_BANDS):
            gy = areas[lab == k].sum() / total * 100.0
            su = areas[lab == -k].sum() / total * 100.0
            out[f'{name}_area_pct_B{k}'] = gy + su
            out[f'{name}_area_pct_B{k}_sulcal'] = su
            out[f'{name}_area_pct_B{k}_gyral'] = gy
        out[f'{name}_area_pct_unlabelled'] = areas[lab == 0].sum() / total * 100.0
    return out


ID_COLUMNS = ['subject', 'subject_folder', 'session', 'hemisphere', 'participant_session']


def order_columns(df):
    """Identifiers first, then whole-brain metrics, band powers, proportions, area shares."""
    first = [c for c in ID_COLUMNS if c in df]
    groups = [c for c in df if c not in first and not c.startswith(('band_', 'dom_', 'det_'))]
    for prefix in ('band_power_', 'band_prop_', 'dom_area_pct_', 'det_area_pct_'):
        groups += [c for c in df if c.startswith(prefix)]
    rest = [c for c in df if c not in first + groups]
    return df[first + groups + rest]


def write_row(row, path):
    order_columns(pd.DataFrame([row])).to_csv(path, index=False)


# --- Core processing ---------------------------------------------------------
def process_single_file(subject, mesh_file, dirs, n_eigen=N_EIGENPAIRS, force=False):
    try:
        start_time = time.time()
        label, subject_id, session, hemisphere = parse_ids(mesh_file)
        print(f"\nProcessing: {mesh_file}\n  ->  {label}")
        if hemisphere == 'unknown':
            print(f"  Error: could not read the hemisphere from {os.path.basename(mesh_file)}")
            return None

        # --- Skip if already processed (safe SLURM resubmission) -------------
        row_path = os.path.join(dirs['per_file'], f'{label}.csv')
        texture_out = os.path.join(dirs['textures'], f'spangy_dom_band_{label}.gii')
        frecomposed_out = os.path.join(dirs['textures'], f'spangy_frecomposed_{label}.gii')
        det_band_out = os.path.join(dirs['textures'], f'spangy_det_band_{label}.gii')
        # frecomposed and the determinant band are required too: meshes processed
        # before they were saved are redone
        if (not force and os.path.exists(row_path) and os.path.exists(texture_out)
                and os.path.exists(frecomposed_out) and os.path.exists(det_band_out)):
            print(f"  Skipping {label} - already processed.")
            return pd.read_csv(row_path).iloc[0].to_dict()

        # Vertex order is preserved (process=False), so textures match the
        # mesh used for MSM. The inertia alignment only rotates/translates.
        mesh = sio.load_mesh(mesh_file)
        n_vertices = int(mesh.vertices.shape[0])
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
            os.path.join(dirs['curv_princ'], f'principal_curv_{label}.gii')
        )

        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        sio.write_texture(
            tex_mean_curv,
            os.path.join(dirs['curv_mean'], f'filt_mean_curv_{label}.gii')
        )
        filt_mean_curv = tex_mean_curv.darray.squeeze()
        total_mean_curv = float(np.sum(filt_mean_curv))

        # Spectrum
        print("  Computing spectrum...")
        grouped_spectrum, group_indices, coefficients, nlevels = \
            spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
        levels = len(group_indices)

        # Plot  (frequency = sqrt(eigVal / (2*pi)) - parenthesised)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        frequency = np.sqrt(eigVal / (2 * np.pi))
        ax1.scatter(frequency, coefficients, marker='o', s=20, linewidths=0.5)
        ax1.set_xlabel('Frequency (m^-1)')
        ax1.set_ylabel('Coefficients')
        ax2.scatter(frequency[1:], coefficients[1:], marker='o', s=20, linewidths=0.5)
        ax2.set_xlabel('Frequency (m^-1)')
        ax2.set_ylabel('Coefficients')
        ax3.bar(np.arange(0, levels), grouped_spectrum)
        ax3.set_xlabel('Spangy Frequency Bands')
        ax3.set_ylabel('Power Spectrum')
        fig.suptitle(label)
        plt.tight_layout()
        fig.savefig(os.path.join(dirs['plots'], f'{label}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Whole-brain metrics
        mL_in_MM3, CM2_in_MM2 = 1000.0, 100.0
        volume = float(mesh.volume)
        surface_area = float(mesh.area)
        afp = float(np.sum(grouped_spectrum[1:]))
        print(f"  Volume={volume / mL_in_MM3:.1f} mL  "
              f"Area={surface_area / CM2_in_MM2:.1f} cm2  AFP={afp:.4f}")

        # Local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(
            coefficients, filt_mean_curv, levels, group_indices, eigVects
        )
        if loc_dom_band.shape[0] != n_vertices:
            print(f"  Error: dominance map has {loc_dom_band.shape[0]} values, "
                  f"mesh has {n_vertices} vertices")
            return None
        sio.write_texture(stex.TextureND(loc_dom_band), texture_out)

        # Per-band recomposition of the filtered mean curvature (bands B1 ... B<levels-1>),
        # one column per band. The dominant band above is derived from these values and
        # the curvature sign, so resampling them (resample_spangy_bands.py) and
        # recomputing the band on another mesh avoids interpolating labels.
        frec = np.asarray(frecomposed, dtype=np.float64)          # slam returns dtype=object
        if frec.shape[0] != n_vertices:
            print(f"  Error: frecomposed has {frec.shape[0]} rows, mesh has {n_vertices} vertices")
            return None
        sio.write_texture(stex.TextureND(frec.T), frecomposed_out)

        # Locally determinant band (Germanaud 2012, Appendix C), from the same
        # per-band synthesis, cumulated band by band (spangy_determinance.py).
        loc_det_band = determinant_band(frec)
        sio.write_texture(stex.TextureND(loc_det_band.astype(np.float64)), det_band_out)

        gyrification_index, hull_area = get_gyrification_index(mesh)
        execution_time = time.time() - start_time
        print(f"  Done in {execution_time:.2f}s")

        row = {
            'subject':              subject_id,
            'subject_folder':       subject,
            'session':              session,
            'hemisphere':           hemisphere,
            'participant_session':  label,
            'mesh_file':            mesh_file,
            'n_vertices':           n_vertices,
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
        row.update(band_summary(grouped_spectrum, loc_dom_band, loc_det_band,
                                vertex_areas(mesh)))
        write_row(row, row_path)
        return row

    except Exception as e:
        print(f"  Error processing {mesh_file}: {e}")
        return None


# --- Merge -------------------------------------------------------------------
def merge_results(root, mesh_root, wanted=None, exclude=None):
    """Collect per-file CSV rows (older JSON rows too) into per-subject and pooled CSVs."""
    os.makedirs(POOLED_DIR, exist_ok=True)
    pooled = []

    for subject in find_subjects(mesh_root, wanted, exclude):
        per_file_dir = os.path.join(root, subject, OUT_SUBDIR, 'results', 'per_file')
        rows = []
        csvs = sorted(glob.glob(os.path.join(per_file_dir, '*.csv')))
        have = {os.path.basename(c)[:-4] for c in csvs}
        olds = [j for j in sorted(glob.glob(os.path.join(per_file_dir, '*.json')))
                if os.path.basename(j)[:-5] not in have]
        for path in csvs + olds:
            try:
                if path.endswith('.csv'):
                    rows.append(pd.read_csv(path).iloc[0].to_dict())
                else:
                    with open(path) as fh:
                        rows.append(json.load(fh))
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
        if olds:
            print(f"  {subject}: {len(olds)} older JSON rows without band shares "
                  f"(rerun those meshes to add them)")
        if not rows:
            print(f"  {subject}: no results yet")
            continue

        df = order_columns(pd.DataFrame(rows)).sort_values(['session', 'hemisphere'])
        out = os.path.join(root, subject, OUT_SUBDIR, 'results',
                           f'{subject}_spangy_results.csv')
        df.to_csv(out, index=False)
        print(f"  {subject}: {len(df)} rows -> {out}")
        pooled.append(df)

    if not pooled:
        print("No results found to pool.")
        return None

    pooled_df = order_columns(pd.concat(pooled, ignore_index=True))
    pooled_df = pooled_df.drop_duplicates(subset='participant_session', keep='last')
    pooled_df = pooled_df.sort_values(['subject', 'session', 'hemisphere'])
    pooled_path = os.path.join(POOLED_DIR, POOLED_NAME)
    pooled_df.to_csv(pooled_path, index=False)
    print(f"\nPooled: {len(pooled_df)} rows from {pooled_df['subject'].nunique()} subjects "
          f"-> {pooled_path}")
    return pooled_path


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh-root', default=MESH_ROOT,
                    help='Pymesh root holding sub-*/ses-*/anat/ meshes')
    ap.add_argument('--root', default=DATA_ROOT,
                    help='Root for outputs (<root>/sub-*/analysis/spangy_pymesh)')
    ap.add_argument('--subjects', nargs='*', default=None,
                    help='Restrict to these subject folders (default: all sub-*)')
    ap.add_argument('--exclude', nargs='*', default=None,
                    help=f'Substrings that disqualify a subject folder '
                         f'(default: {DEFAULT_EXCLUDE}); pass with no values to disable')
    ap.add_argument('--merge', action='store_true',
                    help='Merge existing per-file results into per-subject + pooled CSVs')
    ap.add_argument('--list', action='store_true',
                    help='Print the work plan and exit (no processing)')
    ap.add_argument('--n-eigen', type=int, default=N_EIGENPAIRS)
    ap.add_argument('--force', action='store_true',
                    help='Recompute even if outputs already exist')
    ap.add_argument('--onref', action='store_true',
                    help='Run on the surfaces resampled onto each animal\'s reference mesh '
                         '(surfaces_on_ref/<reg>/), writing to analysis/spangy_onref_<reg>/')
    ap.add_argument('--reg-texture', choices=['sulc', 'dpfstar'], default='sulc',
                    help='With --onref: surfaces from the sulc- or DPF-star-driven registrations')
    args = ap.parse_args()

    global MESH_RELGLOB, OUT_SUBDIR, POOLED_NAME
    if args.onref:
        reg = args.reg_texture
        MESH_RELGLOB = ONREF_RELGLOB
        OUT_SUBDIR = ONREF_OUT_SUBDIR.format(reg=reg)
        POOLED_NAME = ONREF_POOLED_NAME.format(reg=reg)
        if args.mesh_root == MESH_ROOT:
            args.mesh_root = ONREF_MESH_ROOT.format(reg=reg)
        print(f"On-reference mode: meshes from {args.mesh_root}, outputs in <sub>/{OUT_SUBDIR}")

    if args.merge:
        merge_results(args.root, args.mesh_root, args.subjects, args.exclude)
        return

    print(f"Scanning {args.mesh_root} ...")
    jobs = build_job_list(args.mesh_root, args.subjects, args.exclude)
    if not jobs:
        print("No meshes found. Check --mesh-root / directory layout.")
        sys.exit(1)
    print(f"Total: {len(jobs)} meshes across {len({s for s, _ in jobs})} subjects")

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
    # spread across tasks instead of piling on one.
    chunk = jobs[idx::n_tasks]
    print(f"Task {idx + 1}/{n_tasks} - {len(chunk)} meshes assigned")

    if args.list:
        for subject, mesh_file in chunk:
            label, subj, session, hemisphere = parse_ids(mesh_file)
            print(f"  {os.path.relpath(mesh_file, args.mesh_root)}\n\t\t-> {label}")
        return

    dirs_cache = {}
    results = []
    for subject, mesh_file in chunk:
        if subject not in dirs_cache:
            dirs_cache[subject] = subject_dirs(args.root, subject)
        row = process_single_file(subject, mesh_file, dirs_cache[subject],
                                  n_eigen=args.n_eigen, force=args.force)
        if row is not None:
            results.append(row)

    # Per-task convenience dump, one file per subject touched by this task
    if results:
        df = pd.DataFrame(results)
        for subject_folder, sub_df in df.groupby('subject_folder'):
            out = os.path.join(dirs_cache[subject_folder]['results'], f'chunk_{task_id}.csv')
            sub_df.to_csv(out, index=False)
            print(f"Saved {len(sub_df)} rows -> {out}")
    else:
        print(f"Warning: no results for task {task_id}")


if __name__ == '__main__':
    main()