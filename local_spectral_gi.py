#!/usr/bin/env python
"""
Local spectral gyrification indices (Rabiei, Richard, Coulon & Lefèvre)
=======================================================================

Python implementation of the graph windowed Fourier transform (GWFT) analysis of
mean curvature from

    H. Rabiei, F. Richard, O. Coulon, J. Lefèvre. Local Spectral Analysis of the
    Cerebral Cortex: New Gyrification Indices. IEEE TMI 36(3):838-848, 2017.
    (method first presented as "The graph windowed Fourier transform: a tool to
    quantify the gyrification of the cerebral cortex", Insight Journal, 2015)

applied to BaboFet white-matter meshes (native, non-registered surfaces by default).

METHOD
------
f = mean curvature. {lambda_l, phi_l} = Laplace-Beltrami eigenpairs.

    window          g_hat(l) = exp(-tau * lambda_l),  ||g_hat||_2 = 1
    translation     (T_i g)(n)   = c * sum_l g_hat(l) phi_l(i) phi_l(n)
    modulation      (M_k h)(n)   = c * h(n) phi_k(n)
    atom            g_ik = M_k T_i g
    spectrogram     S f(i, k) = < f, g_ik >          (vertex i, frequency k)

    sGI(i) = sum_k |S f(i,k)|^2                          ("total power", TP)
    wGI(i) = sum_k (lambda_k / nu)^2 |S f(i,k)|^2        ("total weighted power", TWP)

With w_i = sum_l g_hat(l) phi_l(i) phi_l (the window centred at i),
S f(i,k) = c^2 <f . w_i, phi_k>: row i of the spectrogram is the Fourier
transform of the curvature multiplied by the window at i. The code computes
these coefficients in vertex batches over local neighbourhoods, never storing
the N x K spectrogram.

sGI is the local energy of the curvature seen through a window centred on i;
wGI weights each frequency by lambda_k^2, i.e. favours fine, tight folding
(it is the local energy of the Laplacian of the windowed curvature). Subject-level
indices = mean over the surface; the paper further divides by the group mean.

Frequency sums run over the first --n-eigen eigenpairs (default 4000, as in your
SPANGY runs; the paper used 5000). Keep it fixed across meshes: the cut-off then
sits at about the same frequency relative to lambda_1 in every mesh (Weyl's law),
so wGI does not drift with mesh resolution. The window itself may need more
eigenpairs (small sigma); they are computed but not added to the sums.

DISCRETISATION CHOICES (differences from the paper are deliberate, see notes)
-----------------------------------------------------------------------------
* Laplacian: linear FEM (cotangent stiffness K, consistent mass M), generalised
  problem K phi = lambda M phi, M-orthonormal eigenvectors. Same operator as
  slam's 'fem' Laplacian used by SPANGY (the paper used Meyer et al.'s cotangent
  / Voronoi-area operator, the same operator with a lumped mass matrix).
  Inner products are area-weighted (<a, b> = a^T M b), so results do not depend
  on mesh density.
* Constants. In the paper c = sqrt(N) (N vertices), so the raw indices carry a
  factor N^2 and the TWP weights are divided by nu = ||(lambda_1..lambda_N)||_2,
  which also grows with N: both depend on mesh resolution, which the paper
  absorbs by dividing each subject's index by the group mean (meshes of ~equal
  size). On native BaboFet meshes vertex counts differ several-fold between
  sessions, so here the continuous analogue is used: c = sqrt(A) (A = surface
  area), the spectrogram energy is divided by its frame bound A, and nu = lambda_1
  (first non-zero eigenvalue, the SPANGY anchor):

      sGI(i) = (1/A) sum_k |S f(i,k)|^2
      wGI(i) = (1/A) sum_k (lambda_k/lambda_1)^2 |S f(i,k)|^2

  sGI(i) is then the window-weighted local mean of H^2 (mm^-2): unchanged by
  remeshing or by uniform growth of the brain. wGI has the same units and
  dimensionless frequency weights. The group-normalised indices (sGI_norm,
  wGI_norm in the pooled CSV) follow the paper's last step.
* Window size: g_hat = exp(-tau*lambda) is the heat kernel; the window w_i is
  approximately Gaussian with standard deviation sigma = sqrt(2*tau) (mm, mesh units). Give
  --sigma-mm (tau = sigma^2/2) or --tau directly; several values may be given.
  The paper used tau = 20, 200, 1000 on adult human brains; fetal baboon
  hemispheres need much smaller windows.
* Optional (--band-power): local power in the SPANGY bands B0..B6 (anchored at
  lambda_1 of the hemisphere, as in whole-hemisphere SPANGY):
  P_b(i) = sum_{k in b} |S f(i,k)|^2. This is a local, windowed version of the
  SPANGY spectrum (summed over the computed eigenpairs only).
* Optional (--sulcus NAME): area-weighted means of every map inside a painted
  sulcus (same ROI textures as spangy_babofet_sulcus.py). Because the window
  is local, averaging these maps inside a sulcus is legitimate on native meshes.

Requires slam (I/O, curvature) on the cluster; the numerical core needs only
numpy/scipy. Shares discovery helpers with spangy_babofet_sulcus.py (same repo).

USAGE
    python local_spectral_gi.py --list
    python local_spectral_gi.py --sigma-mm 2 4 8                    # SLURM-array aware
    python local_spectral_gi.py --sigma-mm 4 --band-power --sulcus STS
    python local_spectral_gi.py --sigma-mm 2 4 8 --merge
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

N_EIGEN = 4000
N_BANDS = 7
WINDOW_TOL = 1e-6          # window coefficients below this (relative) are dropped
RADIUS_FACTOR = 4.0        # neighbourhood radius = RADIUS_FACTOR * sigma


# =============================================================================
# Numerical core (numpy / scipy only)
# =============================================================================
def fem_matrices(vertices, faces):
    """Cotangent stiffness K (positive semi-definite) and consistent mass M (P1 FEM)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    n = len(v)
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    p0, p1, p2 = v[i0], v[i1], v[i2]
    dbl = np.maximum(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1), 1e-12)
    cot0 = np.einsum('ij,ij->i', p1 - p0, p2 - p0) / dbl      # angle at vertex 0
    cot1 = np.einsum('ij,ij->i', p2 - p1, p0 - p1) / dbl
    cot2 = np.einsum('ij,ij->i', p0 - p2, p1 - p2) / dbl
    rows = np.r_[i1, i2, i2, i0, i0, i1]
    cols = np.r_[i2, i1, i0, i2, i1, i0]
    w = 0.5 * np.r_[cot0, cot0, cot1, cot1, cot2, cot2]      # edge opposite each angle
    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    K = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W

    a = dbl / 2.0
    rows = np.r_[i0, i1, i2, i0, i1, i1, i2, i2, i0]
    cols = np.r_[i0, i1, i2, i1, i0, i2, i1, i0, i2]
    vals = np.r_[a / 6, a / 6, a / 6, a / 12, a / 12, a / 12, a / 12, a / 12, a / 12]
    M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return K.tocsr(), M


def lb_eigenpairs(K, M, k):
    """Smallest k eigenpairs of K phi = lambda M phi, ascending, M-orthonormal."""
    k = int(min(k, K.shape[0] - 2))
    vals, vecs = eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=-1e-8, which='LM')
    order = np.argsort(vals)
    vals = np.clip(vals[order], 0, None)
    vecs = vecs[:, order]
    vals[0] = 0.0
    return vals, vecs


def heat_window(eig_val, tau):
    g = np.exp(-tau * eig_val)
    return g / np.linalg.norm(g)


def tau_from_sigma(sigma_mm):
    return 0.5 * sigma_mm ** 2


def eigenpairs_needed(area, tau, tol=WINDOW_TOL):
    """Weyl's law estimate of how many eigenpairs the window needs: N(lambda) ~ A lambda / 4pi."""
    return int(np.ceil(area * np.log(1.0 / tol) / tau / (4 * np.pi)))


def morton_order(points, bits=10):
    """Vertex order along a Z-curve, so consecutive vertices are spatial neighbours."""
    p = np.asarray(points, dtype=float)
    q = ((p - p.min(0)) / (np.ptp(p, axis=0).max() + 1e-12) * (2 ** bits - 1)).astype(np.uint64)
    code = np.zeros(len(p), dtype=np.uint64)
    for b in range(bits):
        for d in range(3):
            code |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + d)
    return np.argsort(code, kind='stable')


def band_groups(eig_val, lambda_ref, n_bands=N_BANDS):
    """SPANGY dyadic bands [first, last] (inclusive) anchored at lambda_ref."""
    try:
        from spangy_babofet_sulcus import group_bands
        return group_bands(eig_val, lambda_ref, n_bands)
    except ImportError:
        gi = np.zeros((n_bands, 2), dtype=int)
        for k in range(1, n_bands):
            lo, hi = lambda_ref * 4.0 ** (k - 1), (lambda_ref * 4.0 ** k if k < n_bands - 1 else np.inf)
            idx = np.where((eig_val >= lo) & (eig_val <= hi) & (np.arange(len(eig_val)) > gi[k - 1, 1]))[0]
            gi[k] = [idx[0], idx[-1]] if len(idx) else [gi[k - 1, 1] + 1, gi[k - 1, 1]]
        return gi


def local_spectral_indices(vertices, M, f, eig_val, eig_vec, tau, n_cut=None,
                           bands=None, batch=256,
                           radius_factor=RADIUS_FACTOR, tol=WINDOW_TOL, verbose=True):
    """
    sGI and wGI maps (and optional local band powers) for one window size tau.

    The window uses as many eigenpairs as it needs (all of eig_vec if necessary).
    The frequency sums of sGI / wGI / band powers run over the first n_cut
    eigenpairs (default: all) - the paper truncated at 5000; keep n_cut fixed
    across meshes so the cut-off sits at the same relative frequency.
    'sGI_full' is the exact all-frequency sGI (Parseval), for reference.

    bands: None or slam-style group_indices (n_bands x 2) -> local power per band.
    """
    vertices = np.asarray(vertices, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    n = len(vertices)
    M = M.tocsr()
    m = np.asarray(M.sum(axis=1)).ravel()                    # lumped vertex areas
    area = float(m.sum())
    n_cut = len(eig_val) if n_cut is None else int(min(n_cut, len(eig_val)))
    phi_c = eig_vec[:, :n_cut]
    wts = (eig_val[:n_cut] / eig_val[1]) ** 2

    g = heat_window(eig_val, tau)
    kw = int(np.searchsorted(-g, -tol * g.max()))            # g is decreasing
    kw = max(kw, 2)
    window_truncated = bool(g[-1] > tol * g.max())
    if window_truncated and verbose:
        print(f"    Warning: window not converged at the last eigenpair "
              f"(g_hat = {g[-1] / g.max():.1e} of max). Need ~{eigenpairs_needed(area, tau)} "
              f"eigenpairs, have {len(eig_val)}: raise --max-eigen or use a larger sigma.")
    phi_w = eig_vec[:, :kw]
    g_w = g[:kw]

    c2 = area                                     # frame-bound scaling, see module docstring

    radius = radius_factor * np.sqrt(2 * tau)
    tree = cKDTree(vertices)
    order = morton_order(vertices)

    sgi = np.empty(n)
    sgi_full = np.empty(n)
    wgi = np.empty(n)
    bp = np.empty((len(bands), n)) if bands is not None else None
    t0 = time.time()
    for s in range(0, n, batch):
        idx = order[s:s + batch]
        nbrs = tree.query_ball_point(vertices[idx], r=radius)
        U = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in nbrs] + [idx]))
        w = phi_w[U] @ (g_w[:, None] * phi_w[idx].T)        # windows centred at idx, on U
        h = f[U][:, None] * w                                 # f . T_i g   (without c)
        Mh = M[U][:, U] @ h
        sgi_full[idx] = c2 * np.einsum('ij,ij->j', h, Mh)    # all frequencies (Parseval)
        coef2 = c2 * (phi_c[U].T @ Mh) ** 2                   # |S f(i,k)|^2 / A, k < n_cut
        sgi[idx] = coef2.sum(axis=0)
        wgi[idx] = wts @ coef2
        if bands is not None:
            for b, (lo, hi) in enumerate(bands):
                bp[b, idx] = coef2[lo:hi + 1].sum(axis=0) if hi >= lo else np.nan
        if verbose and (s // batch) % 50 == 0:
            print(f"    {min(s + batch, n)}/{n} vertices  ({time.time() - t0:.0f}s, "
                  f"neighbourhood {len(U)} vertices)")

    out = {'sGI': sgi, 'wGI': wgi, 'sGI_full': sgi_full,
           'tau': float(tau), 'sigma_mm': float(np.sqrt(2 * tau)), 'n_cut': n_cut,
           'lambda_cut': float(eig_val[n_cut - 1]),
           'n_window_eigenpairs': kw, 'window_truncated': window_truncated,
           'radius_mm': float(radius), 'area': area}
    if bp is not None:
        out['band_power'] = bp
    return out


def area_mean(values, m, mask=None):
    m = np.asarray(m)
    if mask is not None:
        values, m = values[..., mask], m[mask]
    return float(np.nansum(values * m) / m.sum()) if values.ndim == 1 else \
        (np.nansum(values * m, axis=-1) / m.sum())


# =============================================================================
# BaboFet driver (needs slam)
# =============================================================================
def process_single_file(subject, mesh_file, out_dirs, args):
    import slam.io as sio
    import slam.texture as stex
    import slam.curvature as scurv
    from spangy_babofet_sulcus import parse_ids, find_roi, roi_mask_from_texture

    t0 = time.time()
    label, subject_id, session, hemi = parse_ids(mesh_file)
    print(f"\nProcessing: {mesh_file}\n  ->  {label}")
    row_path = os.path.join(out_dirs['per_file'], f'{label}.json')
    if not args.force and os.path.exists(row_path):
        print("  Skipping - already processed.")
        return
    try:
        mesh = sio.load_mesh(mesh_file)
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        n = len(V)

        print("  Curvature...")
        pcurv, _, _ = scurv.curvatures_and_derivatives(mesh)
        H = 0.5 * (pcurv[0, :] + pcurv[1, :])
        if not args.no_zfilter:
            tex = stex.TextureND(H)
            tex.z_score_filtering(z_thresh=3)
            H = np.asarray(tex.darray).squeeze()

        print("  FEM Laplacian + eigenpairs...")
        K, M = fem_matrices(V, F)
        m = np.asarray(M.sum(axis=1)).ravel()
        area = float(m.sum())
        need = max(eigenpairs_needed(area, t) for t in args.taus)
        n_cut = min(args.n_eigen, n - 2)
        n_eigen = min(max(n_cut, need), n - 2, args.max_eigen)
        print(f"  {n} vertices, {area:.0f} mm2; sums over {n_cut} eigenpairs, "
              f"windows need ~{need}; computing {n_eigen}")
        eig_val, eig_vec = lb_eigenpairs(K, M, n_eigen)
        del K

        bands = band_groups(eig_val[:n_cut], eig_val[1]) if args.band_power else None

        roi_mask = None
        if args.sulcus:
            roi_path = find_roi(args.root, subject, session, hemi, args.sulcus, args.roi_patterns)
            if roi_path:
                vals = np.asarray(sio.load_texture(roi_path).darray)
                vals = vals[0] if vals.ndim > 1 and vals.shape[-1] == n else vals.ravel()
                if len(vals) == n:
                    roi_mask = roi_mask_from_texture(vals, args.roi_value)
                else:
                    print(f"  Warning: ROI {roi_path} has {len(vals)} values, mesh has {n}; ignored")
            else:
                print(f"  Note: no {args.sulcus} ROI for {label}")

        row = {'subject': subject_id, 'subject_folder': subject, 'session': session,
               'hemisphere': hemi, 'participant_session': label, 'mesh_file': mesh_file,
               'n_vertices': n, 'surface_area_mm2': area, 'n_eigenpairs': int(len(eig_val)),
               'n_cut': int(n_cut), 'lambda1': float(eig_val[1]),
               'lambda_cut': float(eig_val[n_cut - 1]),
               'lambda_cut_over_lambda1': float(eig_val[n_cut - 1] / eig_val[1]),
               'zfilter': not args.no_zfilter,
               'mean_H2': area_mean(H ** 2, m)}
        if roi_mask is not None:
            row.update({'sulcus': args.sulcus, 'roi_area_mm2': float(m[roi_mask].sum())})

        for tau in args.taus:
            res = local_spectral_indices(V, M, H, eig_val, eig_vec, tau, n_cut=n_cut,
                                         bands=bands, batch=args.batch)
            tag = f"s{res['sigma_mm']:g}"
            sio.write_texture(stex.TextureND(res['sGI']),
                              os.path.join(out_dirs['textures'], f'sGI_{tag}_{label}.gii'))
            sio.write_texture(stex.TextureND(res['wGI']),
                              os.path.join(out_dirs['textures'], f'wGI_{tag}_{label}.gii'))
            row.update({
                f'tau_{tag}': res['tau'],
                f'window_truncated_{tag}': res['window_truncated'],
                f'sGI_{tag}': area_mean(res['sGI'], m),             # area-weighted subject index
                f'wGI_{tag}': area_mean(res['wGI'], m),
                f'sGI_vertexmean_{tag}': float(res['sGI'].mean()),  # paper's (1/N) sum_i
                f'wGI_vertexmean_{tag}': float(res['wGI'].mean()),
                # share of the local curvature energy below the cut-off (QC)
                f'sGI_captured_{tag}': area_mean(res['sGI'], m) / area_mean(res['sGI_full'], m),
            })
            if roi_mask is not None:
                row[f'sGI_roi_{tag}'] = area_mean(res['sGI'], m, roi_mask)
                row[f'wGI_roi_{tag}'] = area_mean(res['wGI'], m, roi_mask)
            if bands is not None:
                bp = res['band_power']
                sio.write_texture(stex.TextureND(bp),
                                  os.path.join(out_dirs['textures'], f'bandpower_{tag}_{label}.gii'))
                # band holding the most local power (B1..B6). Not SPANGY's signed
                # local dominance map: this one comes from the windowed spectrum.
                dom = np.nanargmax(np.nan_to_num(bp[1:], nan=-1), axis=0) + 1
                sio.write_texture(stex.TextureND(dom.astype(float)),
                                  os.path.join(out_dirs['textures'], f'maxpower_band_{tag}_{label}.gii'))
                bmean = area_mean(bp, m)
                for b in range(len(bands)):
                    row[f'band_power_B{b}_{tag}'] = float(bmean[b])
                    if roi_mask is not None:
                        row[f'band_power_roi_B{b}_{tag}'] = float(area_mean(bp[b], m, roi_mask))

        row['processing_time_s'] = time.time() - t0
        with open(row_path, 'w') as fh:
            json.dump(row, fh, indent=2)
        print(f"  Done in {row['processing_time_s']:.0f}s")
    except Exception as e:
        print(f"  Error processing {mesh_file}: {e}")


def merge(args, out_subdir, pooled_name):
    import glob
    import pandas as pd
    from spangy_babofet_sulcus import find_subjects, POOLED_DIR
    rows = []
    for subject in find_subjects(args.mesh_root, args.subjects, args.exclude):
        for p in sorted(glob.glob(os.path.join(args.root, subject, out_subdir,
                                               'results', 'per_file', '*.json'))):
            with open(p) as fh:
                rows.append(json.load(fh))
    if not rows:
        print("No results found.")
        return
    df = pd.DataFrame(rows).sort_values(['subject', 'session', 'hemisphere'])
    # Paper's group normalisation: index / mean of the index over all rows pooled here.
    # Pooling mixes gestational ages - prefer normalising within a reference group yourself.
    for col in [c for c in df.columns if c.startswith(('sGI_s', 'wGI_s'))]:
        df[col.replace('GI_', 'GI_norm_', 1)] = df[col] / df[col].mean()
    os.makedirs(POOLED_DIR, exist_ok=True)
    path = os.path.join(POOLED_DIR, pooled_name)
    df.to_csv(path, index=False)
    print(f"Pooled: {len(df)} rows -> {path}")


def main():
    from spangy_babofet_sulcus import (MESH_ROOT, DATA_ROOT, MESH_RELGLOB, ROI_PATTERNS,
                                       build_job_list, parse_ids)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh-root', default=MESH_ROOT, help='Native (non-registered) meshes')
    ap.add_argument('--mesh-glob', default=MESH_RELGLOB)
    ap.add_argument('--root', default=DATA_ROOT)
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--exclude', nargs='*', default=None)
    win = ap.add_mutually_exclusive_group()
    win.add_argument('--sigma-mm', type=float, nargs='+', help='Window std in mm (tau = sigma^2/2)')
    win.add_argument('--tau', type=float, nargs='+', help='Window parameter tau (mm^2)')
    ap.add_argument('--n-eigen', type=int, default=N_EIGEN,
                    help='Eigenpairs the frequency sums run over (fixed across meshes; default 4000)')
    ap.add_argument('--max-eigen', type=int, default=12000,
                    help='Upper limit on eigenpairs computed for small windows (memory guard)')
    ap.add_argument('--band-power', action='store_true', help='Also local power in SPANGY bands B0..B6')
    ap.add_argument('--sulcus', default=None, help='Average maps inside this painted sulcus')
    ap.add_argument('--roi-value', type=float, default=None)
    ap.add_argument('--no-zfilter', action='store_true', help='Skip the z>3 curvature filtering')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    sigmas = args.sigma_mm or ([np.sqrt(2 * t) for t in args.tau] if args.tau else [4.0])
    args.taus = [tau_from_sigma(s) for s in sigmas]
    args.roi_patterns = ROI_PATTERNS
    tag = 'local_spectral_gi_' + '_'.join(f's{s:g}' for s in sigmas)
    out_subdir = os.path.join('analysis', tag)
    print(f"Windows: sigma = {sigmas} mm (tau = {[round(t, 3) for t in args.taus]})\n"
          f"Meshes: {args.mesh_root}\nOutputs: <sub>/{out_subdir}")

    if args.merge:
        merge(args, out_subdir, f'all_subjects_{tag}_results.csv')
        return

    jobs = build_job_list(args.mesh_root, args.mesh_glob, args.subjects, args.exclude)
    if not jobs:
        sys.exit("No meshes found.")
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))
    n_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT',
                                 int(os.environ.get('SLURM_ARRAY_TASK_MAX', task_min)) - task_min + 1))
    chunk = jobs[task_id - task_min::n_tasks]
    print(f"Task {task_id - task_min + 1}/{n_tasks} - {len(chunk)} meshes")
    if args.list:
        for _, mf in chunk:
            print('  ', parse_ids(mf)[0])
        return

    for subject, mesh_file in chunk:
        base = os.path.join(args.root, subject, out_subdir)
        dirs = {'textures': os.path.join(base, 'textures'),
                'per_file': os.path.join(base, 'results', 'per_file')}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        process_single_file(subject, mesh_file, dirs, args)


if __name__ == '__main__':
    main()
