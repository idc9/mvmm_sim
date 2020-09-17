import matplotlib.pyplot as plt
import os
import seaborn as sns

from mvmm.multi_view.block_diag.graph.linalg import eigh_sym_laplacian_bp
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary
from mvmm_sim.simulation.gif_utils import get_frame_iter, make_gif
from mvmm_sim.simulation.opt_viz import plot_loss_history


def get_plot_pi_from_idx(sp_mvmm):

    def plot_pi_from_idx(idx):
        estimator = sp_mvmm.estimators_[idx]

        if idx == 0:
            Pi = estimator.weights_mat_
            title = 'MVMM'
            zero_thresh = 0

        else:
            Pi = estimator.bd_weights_
            title = 'Penalty = {:1.2f}'.format(estimator.eval_pen_base)
            zero_thresh = estimator.zero_thresh

        evals, _ = eigh_sym_laplacian_bp(Pi)

        comm_summary, Pi_comm = community_summary(Pi, zero_thresh=zero_thresh)
        n_blocks = comm_summary['n_communities']

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        sns.heatmap(Pi, cmap='Blues', square=True, cbar=False, vmin=0)
        plt.title(title)

        plt.subplot(1, 3, 2)
        sns.heatmap(Pi_comm, cmap='Blues', square=True, cbar=False, vmin=0)
        plt.title('n_blocks = {}'.format(n_blocks))

        plt.subplot(1, 3, 3)
        plt.plot(evals, marker='.')

    return plot_pi_from_idx


def get_plot_loss_history_from_idx(sp_mvmm):
    def plot_loss_history_from_idx(idx):
        estimator = sp_mvmm.estimators_[idx]

        if idx == 0:
            loss_vals = estimator.opt_data_['history']['loss_val']
            # title = 'MVMM'

        else:
            loss_vals = estimator.opt_data_['adpt_opt_data']['history']['loss_val']
            # title = 'Penalty = {:1.2f}'.format(estimator.eval_pen_base)

        plot_loss_history(loss_vals)

    return plot_loss_history_from_idx


def get_sp_estimator_idx_iter(sp_mvmm):

    def sp_estimator_idx_iter():
        # for idx in sp_mvmm.estimators_.keys():
        for idx in range(sp_mvmm.n_pen_vals_):
            yield {'idx': idx}

    return sp_estimator_idx_iter


def make_gifs(sp_mvmm, save_dir):

    plot_pi_from_idx = get_plot_pi_from_idx(sp_mvmm)

    sp_estimator_idx_iter = get_sp_estimator_idx_iter(sp_mvmm)

    make_gif(get_frame_iter(plot_func=plot_pi_from_idx,
                            kwarg_iter=sp_estimator_idx_iter()),
             fpath=os.path.join(save_dir, 'spec_pen_Pi.gif'),
             duration=200)

    sp_estimator_idx_iter = get_sp_estimator_idx_iter(sp_mvmm)
    plot_loss_history_from_idx = get_plot_loss_history_from_idx(sp_mvmm)

    make_gif(get_frame_iter(plot_func=plot_loss_history_from_idx,
                            kwarg_iter=sp_estimator_idx_iter()),
             fpath=os.path.join(save_dir, 'spec_pen_loss_history.gif'),
             duration=200)
