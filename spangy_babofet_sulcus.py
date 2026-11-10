#!/usr/bin/env python
"""
Sulcus-restricted SPANGY for BaboFet.

Instead of decomposing the curvature of the whole hemisphere, this script cuts
one painted sulcus out of each white-matter mesh as an open surface patch and
runs the spectral decomposition on that patch only:

    1. curvature is computed (and z-filtered) on the FULL hemisphere, so the
       values at the edge of the sulcus are not corrupted by a mesh border;
    2. the painted vertices (Anatomist SurfPaint texture, non-zero inside) are
       cleaned (optional ring dilation, largest connected piece kept) and the
       faces lying fully inside become a separate open mesh;
    3. Laplace-Beltrami eigenpairs are computed on that patch (slam's FEM
       Laplacian -> free/Neumann boundary, i.e. a cosine-like basis, so the cut
       does not inject spurious high-frequency power the way masking a global
       decomposition would);
    4. the patch curvature is projected on the patch basis, regrouped into
       B0..B6, and slam's local dominance map is computed on the patch and
       written back as a full-hemisphere texture (0 outside the sulcus).

BAND ANCHORING (--band-anchor) — this is the interpretation choice:

    hemi   (default) band edges taken from the first non-zero eigenvalue of the
           whole hemisphere of the same mesh, exactly like whole-hemisphere
           SPANGY. B4 in the sulcus means the same wavelength range as B4 in
           your existing global runs. Long-wavelength bands (B1-B2 usually)
           are empty on a patch smaller than those wavelengths -> NaN.
    patch  slam's own convention applied to the patch: edges relative to the
           patch's first eigenvalue. Scale-free description of the sulcus
           relative to its own size; bands drift in absolute wavelength as the
           sulcus grows.
    fixed  edges anchored to a constant --lambda-ref (mm^-2), identical for
           every session and animal: absolute wavelengths, best for growth
           trajectories in physical units.

ROI TEXTURES

    The ROI texture must have one value per vertex of the mesh being processed.
    The intended route is --onref: all sessions of one animal/hemisphere share
    one vertex numbering, so you paint the sulcus ONCE (e.g. on the last
    session's on-ref surface, where it is clearest) and it follows the sulcus
    back through every session via the MSM registration.

    Looked up in this order (first existing file wins):
        <DATA_ROOT>/<sub>/rois/<sulcus>/<sub>_<ses>_hemi-<L|R>_<sulcus>.gii   (per session)
        <DATA_ROOT>/<sub>/rois/<sulcus>/<sub>_hemi-<L|R>_<sulcus>.gii         (per animal, on-ref)
    or point --roi-pattern at your own layout (fields: {root} {sub} {ses} {hemi} {sulcus}).

OUTPUTS, per subject

    <DATA_ROOT>/<sub>/analysis/spangy_sulcus_<sulcus>[_onref_<reg>]_<anchor>/
        textures/   sulc_dom_band_<label>.gii   dominance map on the full mesh (0 = outside)
                    roi_used_<label>.gii        the cleaned ROI actually analysed (QC)
        plots/      <label>.png
        results/per_file/<label>.json, results/<sub>_spangy_sulcus_results.csv
    <DATA_ROOT>/analysis_pooled/all_subjects_spangy_sulcus_<sulcus>[...]_results.csv

Usage
    python spangy_babofet_sulcus.py --sulcus STS --onref --list
    python spangy_babofet_sulcus.py --sulcus STS --onref                 # SLURM-array aware
    python spangy_babofet_sulcus.py --sulcus STS --onref --merge
    python spangy_babofet_sulcus.py --sulcus STS --onref --band-anchor fixed --lambda-ref 0.002
    python spangy_babofet_sulcus.py --sulcus STS --onref --dilate 2      # add 2 rings around the paint
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# --- Configuration -----------------------------------------------------------
MESH_ROOT = '/envau/work/meca/users/dienye.h/Pymesh/mesh'
DATA_ROOT = '/envau/work/meca/users/dienye.h/python_files/Babofet'
POOLED_DIR = os.path.join(DATA_ROOT, 'analysis_pooled')
N_EIGEN_HEMI = 4000            # what the whole-hemisphere runs use
N_BANDS = 7                    # B0 ... B6
MIN_PATCH_VERTICES = 200

MESH_RELGLOB = os.path.join('ses-*', 'anat', '*_hemi_*_wm.gii')
ONREF_MESH_ROOT = os.path.join(DATA_ROOT, 'surfaces_on_ref', '{reg}')
ONREF_RELGLOB = os.path.join('hemi-*', '*_on_ses-*.anat.surf.gii')

ROI_PATTERNS = [
    os.path.join('{root}', '{sub}', 'rois', '{sulcus}', '{sub}_{ses}_hemi-{hemi}_{sulcus}.gii'),
    os.path.join('{root}', '{sub}', 'rois', '{sulcus}', '{sub}_hemi-{hemi}_{sulcus}.gii'),
]

DEFAULT_EXCLUDE = ['(old)', '_old', '.old', 'backup']
HEMI_MAP = {'left': 'L', 'l': 'L', 'lh': 'L', 'right': 'R', 'r': 'R', 'rh': 'R'}


# --- Filename handling / discovery (unchanged from spangy_babofet_pymesh.py) --
def parse_ids(path):
    name = os.path.basename(path)
    sub = re.search(r'(sub-[^_/]+)', name) or re.search(r'(sub-[^_/]+)', path)
    ses = re.search(r'(ses-[^_/.]+)', name) or re.search(r'(ses-[^_/.]+)', path)
    side = re.search(r'hemi[_-](left|right|lh|rh|l|r)(?=[_./])', name, re.IGNORECASE) \
        or re.search(r'hemi[_-](left|right|lh|rh|l|r)(?=[_./])', path, re.IGNORECASE)
    subject = sub.group(1) if sub else 'sub-NA'
    session = ses.group(1) if ses else 'ses-NA'
    hemi = HEMI_MAP[side.group(1).lower()] if side else 'unknown'
    return f'{subject}_{session}_hemi-{hemi}', subject, session, hemi


def find_subjects(mesh_root, wanted=None, exclude=None):
    exclude = [e.lower() for e in (exclude if exclude is not None else DEFAULT_EXCLUDE)]
    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mesh_root, 'sub-*'))
                  if os.path.isdir(p))
    if wanted:
        missing = set(wanted) - set(subs)
        if missing:
            print(f"Warning: requested subjects not found under {mesh_root}: {sorted(missing)}")
        return [s for s in subs if s in set(wanted)]
    return [s for s in subs if not any(e and e in s.lower() for e in exclude)]


def build_job_list(mesh_root, relglob, wanted=None, exclude=None):
    jobs = []
    for subject in find_subjects(mesh_root, wanted, exclude):
        meshes = sorted(glob.glob(os.path.join(mesh_root, subject, relglob)))
        if not meshes:
            print(f"Warning: no meshes for {subject}")
            continue
        print(f"  {subject}: {len(meshes)} meshes")
        jobs.extend((subject, m) for m in meshes)
    return jobs


def find_roi(root, subject, session, hemi, sulcus, patterns):
    for pat in patterns:
        path = pat.format(root=root, sub=subject, ses=session, hemi=hemi, sulcus=sulcus)
        if os.path.exists(path):
            return path
    return None


# --- Mesh-patch helpers (numpy/scipy only, testable without slam) ------------
def vertex_adjacency(faces, n_vertices):
    """Symmetric sparse vertex adjacency from triangle faces."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_vertices, n_vertices))
    a = (a + a.T).tocsr()
    a.data[:] = 1
    return a


def roi_mask_from_texture(values, roi_value=None, atol=1e-3):
    """Boolean mask: vertices painted with roi_value (or any non-zero value if None)."""
    values = np.asarray(values, dtype=float)
    if roi_value is None:
        return np.abs(values) > atol
    return np.isclose(values, roi_value, atol=atol)


def dilate_mask(mask, adjacency, n_rings):
    m = mask.copy()
    for _ in range(n_rings):
        m = m | (adjacency @ m.astype(float) > 0)
    return m


def extract_patch(vertices, faces, mask):
    """
    Open surface made of the faces whose three vertices are all in mask,
    keeping only the largest connected piece.

    Returns patch_vertices, patch_faces, patch_to_full (index of each patch
    vertex in the full mesh), info dict.
    """
    keep_f = mask[faces].all(axis=1)
    f_in = faces[keep_f]
    if len(f_in) == 0:
        raise ValueError('no face lies fully inside the ROI')

    used = np.unique(f_in)
    remap = -np.ones(len(vertices), dtype=int)
    remap[used] = np.arange(len(used))
    f_local = remap[f_in]

    adj = vertex_adjacency(f_local, len(used))
    n_comp, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp)
    biggest = int(np.argmax(sizes))

    v_keep = np.where(comp == biggest)[0]
    remap2 = -np.ones(len(used), dtype=int)
    remap2[v_keep] = np.arange(len(v_keep))
    f_keep = f_local[(comp[f_local] == biggest).all(axis=1)]
    patch_faces = remap2[f_keep]
    patch_to_full = used[v_keep]

    # Topology: a clean painted sulcus is a disk (Euler characteristic 1,
    # one boundary loop). Holes in the paint add internal boundaries.
    edges = np.sort(np.vstack([patch_faces[:, [0, 1]], patch_faces[:, [1, 2]],
                               patch_faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    n_edges = len(uniq)
    euler = len(patch_to_full) - n_edges + len(patch_faces)
    bnd = uniq[counts == 1]
    if len(bnd):
        badj = vertex_adjacency(np.c_[bnd, bnd[:, 1]], len(patch_to_full))
        bverts = np.unique(bnd)
        sub = badj[bverts][:, bverts]
        n_loops = connected_components(sub, directed=False)[0]
    else:
        n_loops = 0

    info = {
        'roi_n_painted': int(mask.sum()),
        'roi_n_components': int(n_comp),
        'roi_dropped_vertices': int(len(used) - len(v_keep)),
        'patch_euler_characteristic': int(euler),
        'patch_n_boundary_loops': int(n_loops),
    }
    return vertices[patch_to_full], patch_faces, patch_to_full, info


def group_bands(eig_val, lambda_ref, n_bands=N_BANDS):
    """
    slam-style group_indices ([first, last] eigen-index per band, inclusive) with
    dyadic frequency bands anchored at lambda_ref:

        B0 = index 0 (constant mode)
        Bk = lambda in [lambda_ref * 4**(k-1), lambda_ref * 4**k]   (k = 1 .. n_bands-2)
        B(n_bands-1) = everything above, up to the last computed eigenvalue

    (factor 4 in eigenvalue = factor 2 in spatial frequency). Empty bands get
    first = last + 1 so that np.arange(first, last + 1) is empty.
    """
    eig_val = np.asarray(eig_val).ravel()
    n = len(eig_val)
    gi = np.zeros((n_bands, 2), dtype=int)
    gi[0] = [0, 0]
    for k in range(1, n_bands):
        lo = lambda_ref * 4.0 ** (k - 1)
        hi = lambda_ref * 4.0 ** k if k < n_bands - 1 else np.inf
        idx = np.where((eig_val >= lo) & (eig_val <= hi))[0]
        idx = idx[idx > 0]
        if k > 1:
            idx = idx[idx > gi[k - 1, 1]]          # no index in two bands
        if len(idx):
            gi[k] = [idx[0], idx[-1]]
        else:
            start = max(gi[k - 1, 1] + 1, 1)
            gi[k] = [start, start - 1]
    return gi


def band_powers(coefficients, group_indices):
    c2 = np.asarray(coefficients).ravel() ** 2
    out = np.full(len(group_indices), np.nan)
    for k, (a, b) in enumerate(group_indices):
        if b >= a:
            out[k] = c2[a:b + 1].sum()
    return out


def n_eigen_for_patch(patch_area, hemi_area, n_patch_vertices,
                      n_eigen_hemi=N_EIGEN_HEMI, margin=1.15, minimum=50):
    """
    Weyl's law: N(lambda) ~ area * lambda / (4 pi). Reaching the same highest
    frequency as the whole-hemisphere run therefore needs
    n_eigen_hemi * patch_area / hemi_area eigenpairs (plus a margin).
    """
    k = int(np.ceil(n_eigen_hemi * patch_area / hemi_area * margin))
    return int(max(min(k, n_patch_vertices - 2), min(minimum, n_patch_vertices - 2)))


# --- Core processing ---------------------------------------------------------
def process_single_file(subject, mesh_file, dirs, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    import slam.io as sio
    import slam.texture as stex
    import slam.curvature as scurv
    import slam.spangy as spgy

    try:
        t0 = time.time()
        label, subject_id, session, hemi = parse_ids(mesh_file)
        print(f"\nProcessing: {mesh_file}\n  ->  {label}")
        if hemi == 'unknown':
            print("  Error: hemisphere not found in file name")
            return None

        row_path = os.path.join(dirs['per_file'], f'{label}.json')
        tex_out = os.path.join(dirs['textures'], f'sulc_dom_band_{label}.gii')
        if not args.force and os.path.exists(row_path) and os.path.exists(tex_out):
            print("  Skipping - already processed.")
            with open(row_path) as fh:
                return json.load(fh)

        roi_path = find_roi(args.root, subject, session, hemi, args.sulcus, args.roi_patterns)
        if roi_path is None:
            print(f"  Skipping - no ROI texture for {args.sulcus} "
                  f"({args.roi_patterns[0].format(root=args.root, sub=subject, ses=session, hemi=hemi, sulcus=args.sulcus)})")
            return None

        mesh = sio.load_mesh(mesh_file)
        n_vertices = int(mesh.vertices.shape[0])
        roi_vals = np.asarray(sio.load_texture(roi_path).darray)
        if roi_vals.ndim > 1:                      # (n_times, n_vertices) -> first time step
            roi_vals = roi_vals[0] if roi_vals.shape[-1] == n_vertices else roi_vals[:, 0]
        if roi_vals.shape[0] != n_vertices:
            print(f"  Error: ROI {roi_path} has {roi_vals.shape[0]} values but the mesh has "
                  f"{n_vertices} vertices (paint on the same mesh, or use --onref)")
            return None

        # 1. curvature on the full hemisphere (same filtering as the global runs)
        print("  Computing curvature on the full hemisphere...")
        pcurv, _, _ = scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (pcurv[0, :] + pcurv[1, :])
        tex = stex.TextureND(mean_curv)
        tex.z_score_filtering(z_thresh=3)
        filt_mean_curv = np.asarray(tex.darray).squeeze()

        # 2. ROI -> open patch
        mask = roi_mask_from_texture(roi_vals, args.roi_value)
        if args.dilate:
            mask = dilate_mask(mask, vertex_adjacency(np.asarray(mesh.faces), n_vertices),
                               args.dilate)
        pv, pf, p2f, info = extract_patch(np.asarray(mesh.vertices), np.asarray(mesh.faces), mask)
        if info['roi_n_components'] > 1:
            print(f"  Warning: ROI has {info['roi_n_components']} pieces; kept the largest, "
                  f"dropped {info['roi_dropped_vertices']} vertices")
        if info['patch_n_boundary_loops'] > 1:
            print(f"  Warning: patch has {info['patch_n_boundary_loops']} boundary loops "
                  f"(holes in the paint?) - holes act as extra boundaries in the spectrum")
        if len(p2f) < MIN_PATCH_VERTICES:
            print(f"  Error: patch has only {len(p2f)} vertices")
            return None
        patch = trimesh.Trimesh(vertices=pv, faces=pf, process=False)
        f_patch = filt_mean_curv[p2f]

        roi_used = np.zeros(n_vertices)
        roi_used[p2f] = 1
        sio.write_texture(stex.TextureND(roi_used),
                          os.path.join(dirs['textures'], f'roi_used_{label}.gii'))

        # 3. eigenpairs on the patch
        hemi_area, patch_area = float(mesh.area), float(patch.area)
        n_eigen = args.n_eigen or n_eigen_for_patch(patch_area, hemi_area, len(p2f),
                                                    args.n_eigen_hemi)
        n_eigen = min(n_eigen, len(p2f) - 2)
        print(f"  Patch: {len(p2f)} vertices, {patch_area:.1f} mm2 "
              f"({100 * patch_area / hemi_area:.1f}% of hemisphere) -> {n_eigen} eigenpairs")
        eig_val, eig_vec, lap_b = spgy.eigenpairs(patch, n_eigen)
        eig_val = np.asarray(eig_val).ravel()
        if np.any(np.diff(eig_val) < -1e-9):
            print("  Warning: eigenvalues are not in ascending order")

        # slam's own spectrum on the patch (patch-relative bands) - also gives the coefficients
        gs_slam, gi_slam, coefficients, _ = spgy.spectrum(f_patch, lap_b, eig_vec, eig_val)
        coefficients = np.asarray(coefficients).ravel()
        gi_slam = np.asarray(gi_slam, dtype=int)

        # Self-check: our grouping with the patch anchor must reproduce slam's
        gi_check = group_bands(eig_val, eig_val[1], n_bands=len(gi_slam))
        if not np.array_equal(gi_check[:-1], gi_slam[:-1]):
            print("  Warning: band grouping differs from slam's spectrum() - check "
                  "group_bands() against your slam version:\n"
                  f"    slam: {gi_slam.tolist()}\n    ours: {gi_check.tolist()}")

        # 4. bands with the chosen anchor
        if args.band_anchor == 'patch':
            lambda_ref = float(eig_val[1])
        elif args.band_anchor == 'hemi':
            ev_h, _, _ = spgy.eigenpairs(mesh, 3)
            lambda_ref = float(np.sort(np.asarray(ev_h).ravel())[1])
        else:
            lambda_ref = float(args.lambda_ref)
        group_indices = group_bands(eig_val, lambda_ref, N_BANDS)
        powers = band_powers(coefficients, group_indices)
        levels = len(group_indices)

        top_edge = lambda_ref * 4.0 ** (N_BANDS - 2)
        if eig_val[-1] < top_edge:
            print(f"  Note: highest eigenvalue {eig_val[-1]:.4g} < B{N_BANDS - 1} lower edge "
                  f"{top_edge:.4g}; B{N_BANDS - 1} is empty - raise --n-eigen if you need it")

        loc_dom, _ = spgy.local_dominance_map(coefficients, f_patch, levels,
                                              group_indices, eig_vec)
        loc_dom = np.asarray(loc_dom).ravel()
        dom_full = np.zeros(n_vertices)
        dom_full[p2f] = loc_dom
        sio.write_texture(stex.TextureND(dom_full), tex_out)

        # area fraction of the sulcus dominated by each band
        vertex_area = np.asarray(lap_b.sum(axis=1)).ravel()
        dom_abs = np.abs(np.rint(loc_dom)).astype(int)

        # plot
        freq = np.sqrt(eig_val / (2 * np.pi))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.scatter(freq[1:], coefficients[1:], s=12)
        for k in range(1, N_BANDS):
            ax1.axvline(np.sqrt(lambda_ref * 4.0 ** (k - 1) / (2 * np.pi)), color='0.8', lw=0.8)
        ax1.set_xlabel('Frequency (mm^-1)')
        ax1.set_ylabel('Coefficient')
        ax2.bar(np.arange(N_BANDS), np.nan_to_num(powers))
        ax2.set_xticks(np.arange(N_BANDS), [f'B{k}' for k in range(N_BANDS)])
        ax2.set_ylabel('Power')
        fig.suptitle(f'{label} - {args.sulcus} ({args.band_anchor} bands)')
        fig.tight_layout()
        fig.savefig(os.path.join(dirs['plots'], f'{label}.png'), dpi=200)
        plt.close(fig)

        non_dc = np.nansum(powers[1:])
        row = {
            'subject': subject_id, 'subject_folder': subject, 'session': session,
            'hemisphere': hemi, 'participant_session': label, 'sulcus': args.sulcus,
            'mesh_file': mesh_file, 'roi_file': roi_path, 'dilate_rings': int(args.dilate),
            'band_anchor': args.band_anchor, 'lambda_ref': lambda_ref,
            'lambda1_patch': float(eig_val[1]), 'lambda_max': float(eig_val[-1]),
            'n_eigenpairs': int(n_eigen),
            'n_vertices_hemi': n_vertices, 'n_vertices_patch': int(len(p2f)),
            'hemi_area_mm2': hemi_area, 'patch_area_mm2': patch_area,
            'patch_area_fraction': patch_area / hemi_area,
            'patch_mean_curvature': float(np.sum(vertex_area * f_patch) / vertex_area.sum()),
            'patch_folding_power': float(non_dc),
            'processing_time_s': time.time() - t0,
            **info,
        }
        for k in range(N_BANDS):
            row[f'band_power_B{k}'] = float(powers[k])
            row[f'band_power_rel_B{k}'] = float(powers[k] / non_dc) if k and non_dc > 0 else np.nan
            row[f'band_power_per_mm2_B{k}'] = float(powers[k] / patch_area)
            row[f'dom_area_frac_B{k}'] = float(vertex_area[dom_abs == k].sum() / vertex_area.sum())

        with open(row_path, 'w') as fh:
            json.dump(row, fh, indent=2)
        print(f"  Done in {row['processing_time_s']:.1f}s")
        return row

    except Exception as e:
        print(f"  Error processing {mesh_file}: {e}")
        return None


# --- Output paths / merge ----------------------------------------------------
def subject_dirs(root, subject, out_subdir):
    base = os.path.join(root, subject, out_subdir)
    d = {k: os.path.join(base, *p) for k, p in
         {'base': (), 'plots': ('plots',), 'textures': ('textures',),
          'results': ('results',), 'per_file': ('results', 'per_file')}.items()}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def merge_results(root, mesh_root, out_subdir, pooled_name, wanted=None, exclude=None):
    import pandas as pd
    os.makedirs(POOLED_DIR, exist_ok=True)
    pooled = []
    for subject in find_subjects(mesh_root, wanted, exclude):
        rows = []
        for path in sorted(glob.glob(os.path.join(root, subject, out_subdir,
                                                  'results', 'per_file', '*.json'))):
            with open(path) as fh:
                rows.append(json.load(fh))
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(['session', 'hemisphere'])
        out = os.path.join(root, subject, out_subdir, 'results',
                           f'{subject}_spangy_sulcus_results.csv')
        df.to_csv(out, index=False)
        print(f"  {subject}: {len(df)} rows -> {out}")
        pooled.append(df)
    if not pooled:
        print("No results found to pool.")
        return
    df = pd.concat(pooled, ignore_index=True).drop_duplicates('participant_session', keep='last')
    path = os.path.join(POOLED_DIR, pooled_name)
    df.sort_values(['subject', 'session', 'hemisphere']).to_csv(path, index=False)
    print(f"\nPooled: {len(df)} rows -> {path}")


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sulcus', required=True, help='Sulcus name used in ROI file names and outputs')
    ap.add_argument('--mesh-root', default=MESH_ROOT)
    ap.add_argument('--root', default=DATA_ROOT)
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--exclude', nargs='*', default=None)
    ap.add_argument('--onref', action='store_true',
                    help='Use surfaces resampled on each animal\'s reference mesh (paint once per animal/hemi)')
    ap.add_argument('--reg-texture', choices=['sulc', 'dpfstar'], default='sulc')
    ap.add_argument('--roi-pattern', action='append', default=None,
                    help='Custom ROI path pattern(s) with {root} {sub} {ses} {hemi} {sulcus}')
    ap.add_argument('--roi-value', type=float, default=None,
                    help='SurfPaint value marking the sulcus (default: any non-zero value)')
    ap.add_argument('--dilate', type=int, default=0, help='Grow the ROI by N vertex rings')
    ap.add_argument('--band-anchor', choices=['hemi', 'patch', 'fixed'], default='hemi')
    ap.add_argument('--lambda-ref', type=float, default=None,
                    help='Fixed B1 lower edge in mm^-2 (with --band-anchor fixed)')
    ap.add_argument('--n-eigen', type=int, default=None,
                    help='Eigenpairs on the patch (default: scaled from --n-eigen-hemi by area)')
    ap.add_argument('--n-eigen-hemi', type=int, default=N_EIGEN_HEMI)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if args.band_anchor == 'fixed' and args.lambda_ref is None:
        ap.error('--band-anchor fixed needs --lambda-ref')
    args.roi_patterns = args.roi_pattern or ROI_PATTERNS

    relglob = MESH_RELGLOB
    tag = f'spangy_sulcus_{args.sulcus}'
    if args.onref:
        relglob = ONREF_RELGLOB
        tag += f'_onref_{args.reg_texture}'
        if args.mesh_root == MESH_ROOT:
            args.mesh_root = ONREF_MESH_ROOT.format(reg=args.reg_texture)
    tag += f'_{args.band_anchor}' + (f'_dil{args.dilate}' if args.dilate else '')
    out_subdir = os.path.join('analysis', tag)
    pooled_name = f'all_subjects_{tag}_results.csv'
    print(f"Meshes: {args.mesh_root}\nOutputs: <sub>/{out_subdir}")

    if args.merge:
        merge_results(args.root, args.mesh_root, out_subdir, pooled_name,
                      args.subjects, args.exclude)
        return

    jobs = build_job_list(args.mesh_root, relglob, args.subjects, args.exclude)
    if not jobs:
        print("No meshes found.")
        sys.exit(1)

    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if 'SLURM_ARRAY_TASK_COUNT' in os.environ:
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    elif 'SLURM_ARRAY_TASK_MIN' in os.environ and 'SLURM_ARRAY_TASK_MAX' in os.environ:
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_MAX']) - int(os.environ['SLURM_ARRAY_TASK_MIN']) + 1
    else:
        n_tasks = 1
    idx = task_id - int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))
    chunk = jobs[idx::n_tasks]
    print(f"Task {idx + 1}/{n_tasks} - {len(chunk)} meshes")

    if args.list:
        for subject, mesh_file in chunk:
            label, _, ses, hemi = parse_ids(mesh_file)
            roi = find_roi(args.root, subject, ses, hemi, args.sulcus, args.roi_patterns)
            print(f"  {label:32s} ROI: {roi or 'MISSING'}")
        return

    dirs_cache = {}
    for subject, mesh_file in chunk:
        if subject not in dirs_cache:
            dirs_cache[subject] = subject_dirs(args.root, subject, out_subdir)
        process_single_file(subject, mesh_file, dirs_cache[subject], args)


if __name__ == '__main__':
    main()
