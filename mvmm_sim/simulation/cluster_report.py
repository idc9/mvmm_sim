from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, \
    completeness_score, homogeneity_score,\
    mutual_info_score, normalized_mutual_info_score

# from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.special import comb
import numpy as np


def cluster_report(labels_true, labels_pred, additional=None):
    """

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate

    additional: None, callable(labels_true, labels_pred) --> float

    Output
    ------
    report: dict
    """

    report = {}
    report['ars'] = adjusted_rand_score(labels_true, labels_pred)
    # report['ri'] = rand_index(labels_true, labels_pred)

    report['mi'] = mutual_info_score(labels_true, labels_pred)
    report['nmi'] = normalized_mutual_info_score(labels_true, labels_pred,
                                                 average_method='arithmetic')

    # report['v'] = v_measure_score(labels_true, labels_pred, beta=1.0)
    report['hs'] = homogeneity_score(labels_true, labels_pred)
    report['compl_score'] = completeness_score(labels_true, labels_pred)
    report['amis'] = adjusted_mutual_info_score(labels_true, labels_pred,
                                                average_method='arithmetic')

    if additional is not None:
        for name, func in additional:
            report[name] = func(labels_true, labels_pred)

    return report


def rand_index(labels_true, labels_pred):
    """
    from https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    Returns
    -------
    ri : float
       Similarity score between 0 and 1.0.
    """

    # labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    tp_plus_fp = comb(np.bincount(labels_true), 2).sum()
    tp_plus_fn = comb(np.bincount(labels_pred), 2).sum()
    A = np.c_[(labels_true, labels_pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(labels_true))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
