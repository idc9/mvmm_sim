import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import pairwise_distances

from explore.viz.jitter import jitter

from mvmm.linalg_utils import pca
from mvmm_sim.viz_utils import set_xaxis_int_ticks
from mvmm.clustering_measures import MEASURE_MIN_GOOD

from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.viz_utils import draw_ellipse


def plot_model_selection(mm_gs, save_dir=None, name_stub=None,
                         title='', inches=8):

    n_comps = mm_gs.param_grid['n_components']
    # best_n_comp = mm_gs.best_params_['n_components']

    select_metric = mm_gs.select_metric

    # for sel_metric in mm_gs.model_sel_scores_.columns:

    scores = mm_gs.model_sel_scores_[select_metric]

    if MEASURE_MIN_GOOD[select_metric]:
        sel_n_comp = n_comps[scores.idxmin()]
    else:
        sel_n_comp = n_comps[scores.idxmax()]

    # get selected number of components
    # mm_gs.select_metric = sel_metric
    # sel_n_comp = mm_gs.best_params_['n_components']
    # mm_gs.select_metric = orig_sel_metric

    plt.figure(figsize=(inches, inches))
    plt.plot(n_comps, scores, marker='.')  # , label='bic')
    plt.axvline(sel_n_comp,
                color='black', ls='--',
                label='Est. number of components = {}'.format(sel_n_comp))
    set_xaxis_int_ticks()
    plt.legend()
    plt.title(title)

    plt.ylabel(select_metric)
    plt.xlabel("Number of components")

    if save_dir is not None:
        fname = '{}_model_selection.png'.format(select_metric)
        if name_stub is not None:
            fname = name_stub + '_' + fname

        save_fig(os.path.join(save_dir, fname))


def plot_gmm_pcs(gmm, X):

    n_comp = gmm.n_components
    colors = sns.color_palette("Set2", n_comp)

    # get firt two PCs of data
    U, D, V, m = pca(X, rank=2)

    # project data and means onto PCs
    proj_data = X @ V  # U * D
    proj_means = gmm.means_ @ V  # (gmm.means_ - m) @ V

    y_pred = gmm.predict(X)
    data_colors = np.array(colors)[y_pred]
    # data_colors = 'black'

    # plt.figure(figsize=(8, 8))
    plt.scatter(proj_data[:, 0], proj_data[:, 1], color=data_colors, s=10)

    for k in range(n_comp):
        plt.scatter(proj_means[k, 0], proj_means[k, 1],
                    marker='x', s=200, lw=5,
                    color=colors[k])

        # get class covariance
        if gmm.covariance_type == 'diag':
            cov = np.diag(gmm.covariances_[k])

        # project covariance matrix onto PCs
        proj_cov = (V.T @ cov @ V)  # TODO: does this make sense

        draw_ellipse(position=proj_means[k, :], covariance=proj_cov,
                     facecolor='none',
                     edgecolor=colors[k], lw=1)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')


def plot_means_to_data(X, means):
    dists = pairwise_distances(X, means)

    plt.subplot(1, 2, 1)
    sns.heatmap(dists, vmin=0)
    plt.ylabel("samples")
    plt.xlabel("cluster means")
    plt.xticks(np.arange(means.shape[0]) + .5,
               np.arange(1, means.shape[0] + 1))

    plt.subplot(1, 2, 2)
    plt.hist(dists.ravel(), bins=10)
    jitter(dists.ravel())
    plt.title('min: {:1.5f}, max: {:1.5f}'.format(dists.min(), dists.max()))
