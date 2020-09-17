from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def get_aligned_pi_distance(Pi_true, Pi_est, means_true, means_est):
    """
    Aligns true and estimated view clusters by sequentially finding nearest pairs of means. Zero-pads both Pi matrices to make them the same shape.
    Computes normalized L1 norm of results.
    """
    n_views = Pi_true.ndim

    view_alignments = [align_centers(means_true[v], means_est[v])
                       for v in range(n_views)]

    Pi_true_, Pi_est_ = make_same_shape(Pi_true, Pi_est)
    for v in range(n_views):
        Pi_true_ = reorder(arr=Pi_true_, idxs=view_alignments[v][:, 0], axis=v)
        Pi_est_ = reorder(arr=Pi_est_, idxs=view_alignments[v][:, 1], axis=v)

    return abs(Pi_true_ - Pi_est_).mean()


def make_same_shape(A, B):
    n_rows = max(A.shape[0], B.shape[0])
    n_cols = max(A.shape[1], B.shape[1])

    A_ = np.zeros((n_rows, n_cols))
    A_[0:A.shape[0], 0:A.shape[1]] = A

    B_ = np.zeros((n_rows, n_cols))
    B_[0:B.shape[0], 0:B.shape[1]] = B

    return A_, B_


def drop_row_col(arr, row_idx, col_idx):
    return np.delete(arr=np.delete(arr=arr, obj=row_idx, axis=0),
                     obj=col_idx, axis=1)


def align_centers(A, B):
    """
    Finds alignment of centers from two arrays.

    Parameteres
    -----------
    A: array-like, (n_centers_A, n_features)

    B: array-like, (n_centers_B, n_features)

    Output
    -----
    idxs: (min(n_centers_A, n_centers_B), 2)

    """

    dists = euclidean_distances(A, B)
    dists_del = deepcopy(dists)
    idxs = []
    while min(dists_del.shape) > 0:
        min_val = np.min(dists_del)

        # delete row/column
        idx_del = np.argwhere(dists_del == min_val)[0]
        dists_del = drop_row_col(dists_del, *idx_del)

        # index in original array
        idx_orig = np.argwhere(dists == min_val)[0]
        idxs.append(idx_orig)

    idxs = np.array(idxs)

    return idxs


def reorder(arr, idxs, axis):

    if axis == 0:
        idxs_ = np.arange(arr.shape[0])
    elif axis == 1:
        idxs_ = np.arange(arr.shape[1])
    len(idxs)
    len(idxs_)
    idxs_[0:len(idxs)] = idxs

    if axis == 0:
        return arr[idxs_, :]
    else:
        return arr[:, idxs_]
