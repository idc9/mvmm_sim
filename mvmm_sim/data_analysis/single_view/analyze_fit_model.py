import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from lifelines.statistics import multivariate_logrank_test

from mvmm_sim.data_analysis.utils import argmax, drop_small_classes
from mvmm_sim.data_analysis.super_means import get_super_means

from explore.BlockBlock import BlockBlock


def get_interpret_data(model, X,
                       super_data=None,
                       vars2compare=None,
                       survival_df=None,
                       stub=None,
                       n_top_samples=5,
                       clust_size_min=0):

    """
    Parameters
    ----------
    model:

    X: pd.DataFrame

    vars2compare: None, pd.DataFrame

    survival_df: None, pd.DataFrame

    stub: None, str

    n_top_samples: int

    clust_size_min: int
        Ignore clusters with too few samples for metadata comparisons.

    """

    assert isinstance(X, pd.DataFrame)
    sample_names = X.index.values
    feat_names = X.columns.values

    X = X.values

    if vars2compare is not None:
        vars2compare = vars2compare.loc[sample_names]
    if survival_df is not None:
        survival_df = survival_df.loc[sample_names]

    n_comp = model.n_components
    if stub is None:
        stub = 'cl'
    cluster_labels = np.array(['{}__{}'.format(stub, k + 1)
                               for k in range(n_comp)])

    ##########################
    # summarized predictions #
    ##########################
    out = {}

    out['summary'] = summarize_mm_clusters(y_pred=model.predict(X),
                                           weights=model.weights_,
                                           covs=model.covariances_,
                                           cluster_labels=cluster_labels)
    # summarizes the prediction
    # out['summary'] = summarize_mm_clusters(gmm=model, X=X,
    #                                        cluster_labels=cluster_labels)

    # model predictions
    y_pred = model.predict(X)
    y_pred = cluster_labels[y_pred]
    y_pred = pd.Series(y_pred, index=sample_names, name='cluster')

    out['y_pred'] = y_pred

    # ignore small classes for below comparisons!
    y_pred = drop_small_classes(y=y_pred, clust_size_min=clust_size_min)

    ################################
    # top samples for each cluster #
    ################################
    clust_probs = model.predict_proba(X)
    clust_best_sample_idxs = []
    for cl_idx in range(n_comp):

        p = clust_probs[:, cl_idx]
        best_idxs = argmax(values=p, n=n_top_samples)
        # clust_best_samples.append(sample_names[best_idxs])
        clust_best_sample_idxs.append(best_idxs)

    # format to pandas
    clust_best_samples = [sample_names[best_idxs]
                          for best_idxs in clust_best_sample_idxs]
    col_names = ['top_' + str(k + 1) for k in range(n_top_samples)]
    clust_best_samples = pd.DataFrame(clust_best_samples,
                                      index=cluster_labels,
                                      columns=col_names)

    out['clust_best_samples'] = clust_best_samples

    # format cluster probs
    clust_probs = pd.DataFrame(clust_probs,
                               index=sample_names,
                               columns=cluster_labels)

    out['clust_probs'] = clust_probs

    ##################################
    # get standardized cluster means #
    ##################################

    # transform cluster means using overall means/variances
    cl_means = model.means_
    processor = StandardScaler(with_mean=True, with_std=True).fit(X)
    stand_cl_means = processor.transform(cl_means)

    out['cl_means'] = pd.DataFrame(cl_means,
                                   index=cluster_labels,
                                   columns=feat_names)
    out['stand_cl_means'] = pd.DataFrame(stand_cl_means,
                                         index=cluster_labels,
                                         columns=feat_names)

    # get super data means
    if super_data is not None:

        super_feat_names = super_data.columns.values

        resp = model.predict_proba(X)
        cl_super_means = get_super_means(resp=resp,
                                         super_data=super_data,
                                         stand=False)

        stand_cl_super_means = get_super_means(resp=resp,
                                               super_data=super_data,
                                               stand=True)

        out['cl_super_means'] = pd.DataFrame(cl_super_means,
                                             index=cluster_labels,
                                             columns=super_feat_names)

        out['stand_cl_super_means'] = pd.DataFrame(stand_cl_super_means,
                                                   index=cluster_labels,
                                                   columns=super_feat_names)

    #################################################
    # Compare cluster predictions to known metadata #
    #################################################

    if vars2compare is not None:

        vars2compare = vars2compare.loc[y_pred.index, :]

        comparison = BlockBlock(alpha=0.05, cat_test='auc',
                                corr='pearson', multi_cat='ovo',
                                multi_test='fdr_bh', nan_how='drop')

        comparison.fit(y_pred, vars2compare).correct_multi_tests()

        out['comparison'] = comparison

    ############
    # Survival #
    ############
    if survival_df is not None:
        survival_df = survival_df.loc[y_pred.index, :]
        out['survival'] = get_survival(survival_df=survival_df,
                                       y=y_pred)

    info = {}
    info['n_components'] = n_comp
    info['data_shape'] = X.shape
    info['cluster_labels'] = cluster_labels
    out['info'] = info

    return out


def summarize_mm_clusters(y_pred, weights, covs=None, cluster_labels=None):

    n_comp = len(weights)
    df = pd.DataFrame(index=range(n_comp))

    df['weights'] = weights

    pred_counts = Counter(y_pred)
    pred_counts = pd.Series(pred_counts).sort_index()
    df['pred_props'] = pred_counts / pred_counts.sum()
    df['pred_counts'] = pred_counts

    if covs is not None:
        # TODO: this should probably depend on the covariance type
        df['cov_norms'] = pd.Series([np.linalg.norm(covs[k])
                                    for k in range(n_comp)])

    if cluster_labels is not None:
        df.index.name = 'cl_index'
        df = df.reset_index(drop=True)
        df.index = cluster_labels

    df.index.name = 'cluster'

    df = df.sort_values('pred_counts', ascending=False)

    return df


def get_survival(survival_df, y, cat_name='cluster'):

    assert np.alltrue(y.index.values == survival_df.index.values)

    _survival_df = survival_df.copy()
    _survival_df[cat_name] = y

    # drop observations with missing survivals
    _survival_df = _survival_df.dropna(axis=0)
    if _survival_df.shape[0] < survival_df.shape[0]:
        print("After dropping observaings missing survival, n_samples = {}".
              format(_survival_df.shape[0]))

    # TODO: multiple testing!
    survival_out = \
        multivariate_logrank_test(event_durations=_survival_df['duration'],
                                  groups=_survival_df[cat_name],
                                  event_observed=_survival_df['event_obs'])
    pval = survival_out.p_value

    return {'out': survival_out,
            'pval': pval,
            'df': _survival_df}

# def summarize_mm(gmm, X, cluster_labels=None):
#     n_comp = gmm.n_components
#     df = pd.DataFrame(index=range(n_comp))

#     df['weights'] = gmm.weights_
#     y_pred = gmm.predict(X)
#     pred_counts = Counter(y_pred)
#     pred_counts = pd.Series(pred_counts).sort_index()
#     df['pred_props'] = pred_counts / pred_counts.sum()
#     df['pred_counts'] = pred_counts

#     df['cov_norms'] = pd.Series([np.linalg.norm(gmm.covariances_[k])
#                                 for k in range(n_comp)])

#     if cluster_labels is not None:
#         df.index.name = 'cl_index'
#         df = df.reset_index(drop=True)
#         df.index = cluster_labels

#     df.index.name = 'cluster'

#     df = df.sort_values('pred_counts', ascending=False)

#     return df
