import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd


def get_super_means(resp, super_data, stand=False):
    """

    Parameters
    ----------
    model:
        A mixture model fit on fit_data

    fit_data:
        The data the mixture model was fit on.

    super_data:
        The data fit_data were derived from.

    """

    # if v is None:
    # resp = model.predict_proba(fit_data)
    # else:
    #     resp = get_view_resp(model, fit_data, v)

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    super_means = np.dot(resp.T, super_data) / nk[:, np.newaxis]

    if stand:
        processor = StandardScaler(with_mean=True, with_std=True)
        processor.fit(super_data)
        super_means = processor.transform(super_means)

    return super_means


# def get_view_resp(model, X, v):
#     # TODO: check this is what I want

#     resp = model.predict_proba(X)

#     n_samples, n_overall = resp.shape
#     n_comp_view = model.n_view_components[v]

#     # overall index to view index map
#     idxmap = np.zeros(model.n_view_components).astype(int)
#     for k in range(n_overall):
#         k0, k1 = model._get_view_clust_idx(k)
#         idxmap[k0, k1] = k

#     new_resp = np.zeros((n_samples, n_comp_view))
#     for view_k in range(n_comp_view):

#         if v == 0:
#             overall_k = idxmap[view_k, :]
#         elif v == 1:
#             overall_k = idxmap[:, view_k]

#         new_resp[:, view_k] = resp[:, overall_k].sum(axis=1)

#     return new_resp


def get_super_means_for_datasets(model, fit_data, super_data_dict,
                                 stand=False):
    raise NotImplementedError

    cluster_means = {}
    overall_means = {}
    overall_stds = {}

    for dataset_name in super_data_dict.keys():
        super_data = super_data_dict[dataset_name].values

        overall_means[dataset_name] = super_data.mean(axis=0)
        overall_stds[dataset_name] = super_data.std(axis=0)

        cluster_means[dataset_name] = get_super_means(model=model,
                                                      fit_data=fit_data,
                                                      super_data=super_data,
                                                      stand=stand)

    y_pred = model.predict(fit_data)
    y_cnts = pd.Series(0, index=range(model.n_components))
    for cl_idx, cnt in Counter(y_pred).items():
        y_cnts[cl_idx] = cnt

    return overall_means, overall_stds, cluster_means, y_cnts
