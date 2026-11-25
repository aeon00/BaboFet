"""
spangy_determinance.py — locally DETERMINANT band of SPANGY
(Germanaud et al., NeuroImage 63 (2012) 1257–1272, Appendix C, formulas C.2–C.4).

Dominant band (slam.spangy.local_dominance_map): NON-cumulative synthesis. Each
vertex gets the band contributing most to its curvature.

Determinant band (this module): CUMULATIVE synthesis. Add the bands one by one
(band 1, then 1+2, then 1+2+3, ...). After each step the vertex is gyral if the
running total is > 0 and sulcal otherwise (the paper's CFP map). The paper's
differential map is

    SM_k(p) = [cum_k(p) > 0] - [cum_(k-1)(p) > 0]          (C.2 – C.4)

i.e. +1 where band k turns the vertex gyral, -1 where it turns it sulcal, 0 where
nothing changes. Labels are given from the HIGHEST band down: a vertex takes the
highest band at which its CFP label changed, i.e. the band after which it never
changes again. That band "determines whether it belongs to the sulcal or the
gyral pattern". The label is signed like the dominant band: positive = gyral,
negative = sulcal (the vertex's final sign).

One addition to the paper: the paper starts from an empty sum, counted as "not
gyral", so a vertex that is sulcal from band 1 onwards never changes and gets no
label. Here it gets -1 (determined by band 1), symmetric with a vertex gyral from
band 1 onwards, which gets +1 in both. Use paper_strict=True for the literal rule
(those vertices stay 0). The paper's figures only relabel bands 4–6 and leave the
rest at their lower-band label; to reproduce that, merge |label| <= 3 afterwards.

Bands are numbered as in slam: column i of frecomposed is band i+1 (band 0, the
mean, is not part of the synthesis, as in the paper's sums starting at i = 1).
"""
import numpy as np


def recompose_bands(coefficients, nlevels, group_indices, eig_vec):
    """Per-band (non-cumulative) synthesis, exactly as slam.spangy.local_dominance_map."""
    N = np.size(coefficients)
    eig_vec = np.flip(eig_vec, 1)
    frecomposed = np.zeros((eig_vec.shape[0], nlevels - 1))
    for i in range(nlevels - 1):
        levels_i = np.arange(group_indices[i + 1, 0], group_indices[i + 1, 1] + 1)
        frecomposed[:, i] = np.dot(eig_vec[:, N - levels_i - 1], coefficients[levels_i].T)
    return frecomposed


def determinant_band(frecomposed, paper_strict=False, return_margin=False):
    """
    Determinant band from the per-band synthesis (n_vertices x n_bands, band 1 first).

    Returns signed labels (+k gyral, -k sulcal) and, if asked, a margin: the
    smallest |running total| from the deciding band onwards (how close the vertex
    came to changing side again; 0 = on a border).
    """
    frec = np.asarray(frecomposed, dtype=np.float64)
    n, K = frec.shape
    cum = np.cumsum(frec, axis=1)                 # cumulative synthesis, bands 1..k
    gyral = cum > 0                               # CFP map after each band
    final = gyral[:, -1]

    # last band whose CFP label differs from the final one; the deciding band is the next
    differs = gyral != final[:, None]
    changed = differs.any(axis=1)
    last_diff = K - 1 - np.argmax(differs[:, ::-1], axis=1)
    k = np.where(changed, last_diff + 2, 1)       # 1-based band number

    label = np.where(final, k, -k).astype(np.int32)
    if paper_strict:
        label[~changed & ~final] = 0              # sulcal from band 1: no change, no label

    if not return_margin:
        return label
    cols = np.arange(K)[None, :]
    masked = np.where(cols >= (k - 1)[:, None], np.abs(cum), np.inf)
    return label, masked.min(axis=1)


def local_determinance_map(coefficients, f2analyse, nlevels, group_indices, eig_vec,
                           paper_strict=False):
    """
    Same inputs and outputs as slam.spangy.local_dominance_map, for the
    determinant band. f2analyse is only used for its length (kept for the same
    signature); the side comes from the cumulative synthesis itself.

    Returns
    -------
    loc_det_band : int array, signed determinant band per vertex
    frecomposed : (n_vertices, nlevels - 1) array, per-band synthesis
    """
    frecomposed = recompose_bands(coefficients, nlevels, group_indices, eig_vec)
    if frecomposed.shape[0] != len(f2analyse):
        raise ValueError("eigenvectors and f2analyse have different numbers of vertices")
    return determinant_band(frecomposed, paper_strict=paper_strict), frecomposed
