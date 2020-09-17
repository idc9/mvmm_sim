import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from mvmm_sim.viz_utils import simpleaxis
from mvmm_sim.data_analysis.super_means import get_super_means


def plot_ephys_curve(values, overall_mean=None, overall_std=None, color=None):
    """
    Plots an ephys curve. Optinoally also plots the overall sample mean and std.

    """
    if overall_mean is not None:

        plt.plot(np.array(overall_mean),
                 marker='x', label='mean', color='grey', alpha=.5)

    if overall_mean is not None and overall_std is not None:

        plt.fill_between(x=np.arange(len(overall_mean)),
                         y1=overall_mean - overall_std,
                         y2=overall_mean + overall_std,
                         color='grey', alpha=.2)

    plt.plot(np.array(values),
             marker='.', color=color)
    plt.xlim(0)
    simpleaxis(plt.gca())


def plot_cluster_ephys_curve(values, overall_means=None, overall_stds=None,
                             color=None, grid=None, row_idx=None,
                             y_label=None):
    """
    Plots ephys curves for each data type.
    """
    assert isinstance(values, dict)

    n_datasets = len(values)
    if grid is None:
        grid = plt.GridSpec(nrows=1, ncols=n_datasets,
                            wspace=0.4)  #, hspace=0.4)
        row_idx = 0
    else:
        assert row_idx is not None

    m = None
    s = None
    for data_idx, data_name in enumerate(values.keys()):
        if overall_means is not None:
            m = overall_means[data_name]
        if overall_stds is not None:
            s = overall_stds[data_name]

        plt.subplot(grid[row_idx, data_idx])

        plot_ephys_curve(values[data_name], overall_mean=m, overall_std=s,
                         color=color)
        plt.title(data_name)
        plt.ylabel(y_label)


def plot_top_clust_ephys_curves(cluster_super_means, y_cnts,
                                overall_means=None, overall_stds=None,
                                clust_labels=None, n_to_show=10, inches=5):

    """
    Plots ephys curves for top clusters.
    """

    dataset_names = list(cluster_super_means.keys())
    n_datasets = len(dataset_names)

    # setup plottin grid
    n_clusters = cluster_super_means[dataset_names[0]].shape[0]
    n_to_show = min(n_clusters, n_to_show)
    grid = plt.GridSpec(nrows=n_to_show,
                        ncols=n_datasets,
                        wspace=0.4, hspace=0.4)

    if inches is not None:
        plt.figure(figsize=(2 * n_datasets * inches, n_to_show * inches))

    # setup cluster labels and colors
    cluster_colors = sns.color_palette("Set2", n_to_show)
    if clust_labels is None:
        clust_labels = np.arange(1, n_clusters + 1)  # 1 indexing

    # get top clusters by counts
    y_cnts = y_cnts.sort_values(ascending=False)
    top_idxs = y_cnts[0:n_to_show].index.values

    for row_idx, cl_idx in enumerate(top_idxs):
        label = clust_labels[cl_idx]
        color = cluster_colors[row_idx]

        values = {}
        for name in dataset_names:
            values[name] = cluster_super_means[name][cl_idx]

        plot_cluster_ephys_curve(values,
                                 overall_means=overall_means,
                                 overall_stds=overall_stds,
                                 color=color, grid=grid, row_idx=row_idx,
                                 y_label=label)


def get_ephys_super_data(model, fit_data, ephys_raw):
    """
    Gets data for ephys curve plotting.
    """
    # cl_labels = np.arange(1, model.n_components + 1)
    # cl_labels = ['cluster_{}'.format(cl_idx + 1)
    #              for cl_idx in range(model.n_components)]

    # get cluster prediction counts
    y_pred = model.predict(fit_data)
    # y_cnts = pd.Series(0, index=range(model.n_components))
    y_cnts = np.zeros(model.n_components, dtype=int)
    for cl_idx, cnt in Counter(y_pred).items():
        y_cnts[cl_idx] = cnt
    y_cnts = pd.Series(y_cnts)

    resp = model.predict_proba(fit_data)

    cluster_super_means = {}
    super_data_means = {}
    super_data_stds = {}
    for k in ephys_raw.keys():

        super_data = ephys_raw[k]
        super_data_means[k] = super_data.mean(axis=0)
        super_data_stds[k] = super_data.std(axis=0)

        cluster_super_means[k] = get_super_means(resp=resp,
                                                 super_data=super_data,
                                                 stand=False)

    return cluster_super_means, super_data_means, super_data_stds, y_cnts
