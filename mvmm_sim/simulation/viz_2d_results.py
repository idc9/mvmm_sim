import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from itertools import product
import numpy as np

from mvmm.multi_view.MVMM import MVMM
from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.LogPenMVMM import LogPenMVMM
from mvmm.multi_view.TwoStage import TwoStage
from mvmm_sim.viz_utils import draw_ellipse


def plot_means(means, Pi_nonzero_mask=None,
               means_alpha=None, covs=None,
               color='red', label=None, **kws):

    n_view_components = (means[0].shape[0], means[1].shape[0])
    # n_view_components = Pi_nonzero_mask.shape
    # assert means[0].shape[0] == n_view_components[0]
    # assert means[1].shape[0] == n_view_components[1]

    first_mean_plotted = True
    for k0, k1 in product(range(n_view_components[0]),
                          range(n_view_components[1])):

        if Pi_nonzero_mask is not None and not Pi_nonzero_mask[k0, k1]:
            continue

        x = means[0][k0]
        y = means[1][k1]

        if means_alpha is not None:
            kws['alpha'] = means_alpha[k0, k1]

        if first_mean_plotted:  # only add label once
            label = label
            first_mean_plotted = False
        else:
            label = None

        plt.scatter(x, y, marker='x', label=label, color=color, **kws)

        if covs is not None:
            # cov = np.diag([covs[0][k0][0, 0], covs[1][k1][0, 0]])
            cov = np.diag([covs[0][k0], covs[1][k1]])

            draw_ellipse(position=[x, y], covariance=cov,
                         facecolor='none',
                         edgecolor=color, lw=1)

    plt.xlabel("View 1")
    plt.ylabel("View 2")
    plt.axis('equal')


def plot_pi(Pi, means=None, Pi_nonzero_mask=None,
            sort_pi='mean_order'):

    assert not sort_pi or sort_pi in ['mean_order', 'blocks']
    if sort_pi == 'blocks':
        raise NotImplementedError

    # put first view on cols, second view on rows
    Pi = deepcopy(Pi).T
    if Pi_nonzero_mask is not None:
        Pi_nonzero_mask = deepcopy(Pi_nonzero_mask).T

    if means is not None and sort_pi == 'mean_order':
        view_1_ordering = np.argsort(means[1].reshape(-1))[::-1]
        view_0_ordering = np.argsort(means[0].reshape(-1))

        Pi = Pi[view_1_ordering, :][:, view_0_ordering]

        if Pi_nonzero_mask is not None:
            Pi_nonzero_mask = Pi_nonzero_mask[view_1_ordering, :][:, view_0_ordering]

    if Pi_nonzero_mask is not None:
        mask = ~Pi_nonzero_mask
        n_comp = Pi_nonzero_mask.sum()
    else:
        mask = None
        n_comp = (Pi > 0).sum()

    sns.heatmap(Pi, cmap='Blues', annot=True, fmt='.3f', mask=mask)
    plt.title('Num non-zero: {}'.format(n_comp))
    plt.xlabel("View 1 clusters")
    plt.ylabel("View 2 clusters")


def plot_true(view_params, Pi_true, X_tr, show_cov=True):

    means = [view_params[v]['means'] for v in range(2)]

    covs = [view_params[v]['covs'].reshape(-1) for v in range(2)]
    if not show_cov:
        covs = None

    Pi_nonzero_mask = Pi_true > 0

    plt.subplot(1, 2, 1)
    plt.scatter(X_tr[0], X_tr[1], color='black', s=10, alpha=.2)
    plot_means(means=means, Pi_nonzero_mask=Pi_nonzero_mask,
               covs=covs,
               label='True', s=200, lw=5)

    plt.subplot(1, 2, 2)
    plot_pi(Pi=Pi_true, means=means)


def plot_full_mvmm_means(mvmm, means_alpha=True, show_cov=True, **kws):

    means = [mvmm.view_models_[v].means_ for v in range(2)]

    if means_alpha:
        Pi = mvmm.weights_mat_
        means_alpha = Pi / Pi.max()
    else:
        means_alpha = None

    covs = [mvmm.view_models_[v].covariances_.reshape(-1) for v in range(2)]
    if not show_cov:
        covs = None

    plot_means(means=means, means_alpha=means_alpha,
               covs=covs, **kws)


def plot_full_mvmm_pi(mvmm, sort_pi='mean_order'):
    Pi_full_est = mvmm.weights_mat_
    full_est_means = [mvmm.view_models_[v].means_ for v in range(2)]
    plot_pi(Pi_full_est, means=full_est_means, sort_pi=sort_pi)


def plot_full_mvmm(mvmm, X_tr, sort_pi='mean_order', show_cov=True):

    plt.subplot(1, 2, 1)
    plt.scatter(X_tr[0], X_tr[1], color='black', s=10, alpha=.2)

    plot_full_mvmm_means(mvmm,
                         color='blue', label='full_mvmm',
                         s=200, lw=5)

    # plot_means(means_true, Pi_true_nonzero_mask,
    #            color='red', label='true',
    #            s=200, lw=5)

    # plt.legend()

    plt.subplot(1, 2, 2)
    plot_full_mvmm_pi(mvmm, sort_pi=sort_pi)


def plot_log_pen_mvmm_means(mvmm, means_alpha=True, show_cov=True, **kws):
    means = [mvmm.view_models_[v].means_ for v in range(2)]

    Pi_nonzero_mask = ~mvmm.zero_mask_

    if means_alpha:
        Pi = mvmm.weights_mat_
        means_alpha = Pi / Pi.max()
    else:
        means_alpha = None

    covs = [mvmm.view_models_[v].covariances_.reshape(-1) for v in range(2)]
    if not show_cov:
        covs = None

    plot_means(means=means,
               means_alpha=means_alpha,
               Pi_nonzero_mask=Pi_nonzero_mask,
               covs=covs,
               **kws)


def plot_log_pen_pi(mvmm, sort_pi='mean_order'):
    Pi_est = mvmm.weights_mat_
    means_est = [mvmm.view_models_[v].means_ for v in range(2)]
    Pi_non_zero_mask_est = ~mvmm.zero_mask_

    plot_pi(Pi=Pi_est, means=means_est, Pi_nonzero_mask=Pi_non_zero_mask_est,
            sort_pi=sort_pi)


def plot_log_pen_mmvm(mvmm, X_tr, sort_pi='mean_order', show_cov=True):

    plt.subplot(1, 2, 1)
    plt.scatter(X_tr[0], X_tr[1], color='black', s=10, alpha=.2)

    plot_log_pen_mvmm_means(mvmm,
                            color='blue', label='log_pen_mvmm',
                            s=200, lw=5)

    # plot_means(means_true, Pi_true_nonzero_mask,
    #            color='red', label='true',
    #            s=200, lw=5)

    # plt.legend()

    plt.subplot(1, 2, 2)
    plot_log_pen_pi(mvmm, sort_pi=sort_pi)


# def plot_log_pen_seq(mvmm, X_tr, inches=8, save_dir=None):

#     best_idx = mvmm.best_idx_

#     for tune_idx, est in enumerate(mvmm.estimators_):

#         plt.figure(figsize=(2 * inches, inches))
#         plot_log_pen_mmvm(est.final_, X_tr)

#         n_comp_est = (~est.final_.zero_mask_).sum()
#         pen_val = mvmm.best_estimator_.final_.pen

#         if tune_idx == best_idx:
#             title = 'Log pen MVMM best estimator\n pen={:1.3f}' \
#                     '(n_comp_est={})'.format(pen_val, n_comp_est)
#         else:
#             title = 'Log pen MVMM\n pen={:1.3f} (n_comp_est={})'.\
#                 format(pen_val, n_comp_est)

#         a = plt.gcf()
#         a.axes[1].set_title(title)

#         if save_dir is not None:
#             save_fig(os.path.join(save_dir,
#                                   'tune_idx_{}.png'.format(tune_idx)))


def plot_bd_mvmm_means(mvmm, means_alpha=True, show_cov=True, **kws):
    means = [mvmm.view_models_[v].means_ for v in range(2)]

    # D = mvmm.bd_weights_
    Pi_nonzero_mask = mvmm.bd_weights_ > mvmm.zero_thresh

    if means_alpha:
        Pi = mvmm.weights_mat_
        means_alpha = Pi / Pi.max()
    else:
        means_alpha = None

    covs = [mvmm.view_models_[v].covariances_.reshape(-1) for v in range(2)]
    if not show_cov:
        covs = None

    plot_means(means,
               means_alpha=means_alpha,
               Pi_nonzero_mask=Pi_nonzero_mask,
               covs=covs,
               **kws)


def plot_bd_mvmm_pi(mvmm, sort_pi='mean_order'):
    Pi_est = mvmm.bd_weights_
    Pi_non_zero_mask_est = mvmm.bd_weights_ > mvmm.zero_thresh
    means_est = [mvmm.view_models_[v].means_ for v in range(2)]

    plot_pi(Pi=Pi_est, means=means_est, Pi_nonzero_mask=Pi_non_zero_mask_est,
            sort_pi=sort_pi)


def plot_bd_mvmm(mvmm, X_tr, sort_pi='mean_order', show_cov=True):

    plt.subplot(1, 2, 1)

    plt.scatter(X_tr[0], X_tr[1], color='black', s=10, alpha=.2)
    plot_bd_mvmm_means(mvmm, color='blue', label='bd_mvmm',
                       s=200, lw=5)

    # plot_means(means_true, Pi_true_nonzero_mask,
    #            color='red', label='true',
    #            s=200, lw=5)

    # plt.legend()

    plt.subplot(1, 2, 2)
    plot_bd_mvmm_pi(mvmm, sort_pi=sort_pi)


def plot_estimator(mvmm, X_tr, show_cov=True, sort_pi='mean_order'):
    if isinstance(mvmm, TwoStage):
        mvmm = mvmm.final_

    if type(mvmm) == MVMM:
        plot_full_mvmm(mvmm=mvmm, X_tr=X_tr, sort_pi=sort_pi,
                       show_cov=show_cov)

    elif type(mvmm) == BlockDiagMVMM:
        plot_bd_mvmm(mvmm=mvmm, X_tr=X_tr, sort_pi=sort_pi,
                     show_cov=show_cov)

    elif type(mvmm) == LogPenMVMM:
        plot_log_pen_mmvm(mvmm=mvmm, X_tr=X_tr, sort_pi=sort_pi,
                          show_cov=show_cov)

# def plot_tune_seq(estimators, X_tr, save_dir=None, inches=8):

#     for idx in range(len(estimators)):

#         plt.figure(figsize=(2 * inches, inches))

#         plot_estimator(mvmm)

#         else:
#             raise ValueError()

#         if save_dir is not None:
#             save_fig(os.path.join(save_dir,
#                                   'tune_idx_{}.png'.format(idx)))
