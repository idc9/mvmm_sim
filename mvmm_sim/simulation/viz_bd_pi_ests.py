import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mvmm.multi_view.block_diag.graph.bipt_community import community_summary
from mvmm.multi_view.MVMM import MVMM


def plot_pi_ests_bd_mvmm(fit_models, zero_thresh, inches=10):
    n_samples_tr_seq = np.sort(list(fit_models.keys()))

    n_blocks_seq = fit_models[n_samples_tr_seq[0]]['bd_mvmm'].param_grid_

    ncols = len(n_samples_tr_seq)
    nrows = len(n_blocks_seq)

    plt.figure(figsize=(ncols * inches, nrows * inches))
    grid = plt.GridSpec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.2)
    for c, n_samples in enumerate(n_samples_tr_seq):

        mvmm_gs = fit_models[n_samples]['bd_mvmm']

        for r, est in enumerate(mvmm_gs.estimators_):

            Pi_est_blk = est.final_.bd_weights_
            summary, Pi_comm = community_summary(Pi_est_blk,
                                                 zero_thresh=zero_thresh)
            n_blocks = est.final_.n_blocks
            n_blocks_est = summary['n_communities']

            title = 'n_samples={}, n_blocks={} (est={})\n{}'.\
                format(n_samples, n_blocks, n_blocks_est,
                       summary['comm_shapes'])

            plt.subplot(grid[r, c])
            sns.heatmap(Pi_comm.T, cmap='Blues',
                        square=True, cbar=False, vmin=0)
            plt.title(title)
            plt.xlabel("View 1 clusters")
            plt.xlabel("View 2 clusters")


def plot_pi_ests_log_pen(fit_models, zero_thresh, nrows=5, inches=10):

    select_metric = 'bic'  # TODO: if we want other ones need to change
    # code below

    n_samples_tr_seq = np.sort(list(fit_models.keys()))

    ncols = len(n_samples_tr_seq)

    plt.figure(figsize=(ncols * inches, nrows * inches))
    grid = plt.GridSpec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.2)
    for c, n_samples in enumerate(n_samples_tr_seq):

        gs = fit_models[n_samples]['log_pen_mvmm']

        meow = [{'tune_idx': i,
                 'n_comp': est.n_components,
                 'bic': gs.model_sel_scores_.iloc[i][select_metric]}
                for i, est in enumerate(gs.estimators_)]

        meow = pd.DataFrame(meow)

        tune_idxs = []
        for _, df in meow.groupby('n_comp'):
            idx_best = df['bic'].idxmin()
            tune_idx = int(df.loc[idx_best]['tune_idx'])
            tune_idxs.append(tune_idx)

        r = 0
        for tune_idx in tune_idxs:
            est = gs.estimators_[tune_idx].final_

            n_comp = est.n_components

            Pi_est = est.weights_mat_
            summary, Pi_comm = community_summary(Pi_est,
                                                 zero_thresh=zero_thresh)
            n_blocks_est = summary['n_communities']

            title = 'n_samples={}, n_components={} (n_blocks={})\n{}'.\
                format(n_samples, n_comp, n_blocks_est, summary['comm_shapes'])

            plt.subplot(grid[r, c])
            sns.heatmap(Pi_comm.T,
                        cmap='Blues', square=True, cbar=False, vmin=0)
            plt.title(title)
            plt.xlabel("View 1 clusters")
            plt.xlabel("View 2 clusters")
            if r == nrows - 1:
                continue
            else:
                r += 1


def plot_pi_ests_sp_mvmm(fit_models, zero_thresh, inches=10, nrows=5):
    n_samples_tr_seq = np.sort(list(fit_models.keys()))
    ncols = len(n_samples_tr_seq)

    plt.figure(figsize=(ncols * inches, nrows * inches))
    grid = plt.GridSpec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.2)

    for c, n_samples in enumerate(n_samples_tr_seq):

        estimators = fit_models[n_samples]['sp_mvmm'].estimators_

        for r in range(nrows):

            if r >= len(estimators):
                continue
            else:
                est = estimators[r]

            if type(est) == MVMM:
                Pi = est.weights_mat_
            else:
                Pi = est.bd_weights_

            summary, Pi_comm = community_summary(Pi,
                                                 zero_thresh=zero_thresh)
            n_blocks_est = summary['n_communities']

            title = 'n_samples={}, est_n_blocks={}\n{}'.\
                format(n_samples, n_blocks_est, summary['comm_shapes'])

            plt.subplot(grid[r, c])
            sns.heatmap(Pi_comm.T,
                        cmap='Blues', square=True, cbar=False, vmin=0)
            plt.title(title)
            plt.xlabel("View 1 clusters")
            plt.xlabel("View 2 clusters")
