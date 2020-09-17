from os.path import join
from joblib import load
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mvmm.multi_view.block_diag.graph.bipt_community import community_summary
from mvmm.multi_view.block_diag.graph.linalg import eigh_sym_laplacian_bp
from mvmm.clustering_measures import MEASURE_MIN_GOOD
from mvmm.linalg_utils import pca

from mvmm_sim.simulation.opt_viz import plot_loss_history
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.viz_utils import draw_ellipse


def plot_mvmm_model_selection(model_sel_df, group_var,
                              group_var_label=None, select_metric='bic',
                              cmap="Set2"):

    all_view_comp_idxs = np.unique(model_sel_df['view_comp_idx'])
    colors = sns.color_palette(cmap, len(all_view_comp_idxs))

    if group_var_label is None:
        group_var_label = group_var

    for i, view_comp_idx in enumerate(all_view_comp_idxs):
        df = model_sel_df.query("view_comp_idx == @view_comp_idx")
        df = df.sort_values(group_var)

        color = colors[i]

        plt.plot(df[group_var], df[select_metric],
                 marker='.', color=color, alpha=.5)

        plt.xlabel(group_var_label)
        plt.ylabel(select_metric)

        # label n view comp curves
        x_coord = max(df[group_var])
        y_coord = df[select_metric].values[-1]
        text = df['n_view_comp'].values[0]
        plt.text(x=x_coord, y=y_coord, s=text, color=color)

    if MEASURE_MIN_GOOD[select_metric]:
        sel_idx = model_sel_df[select_metric].idxmin()
    else:
        sel_idx = model_sel_df[select_metric].idxmax()

    sel_row = model_sel_df.loc[sel_idx]
    plt.title('{} selected {}\n n_blocks {}, n_comp {}'.
              format(select_metric,
                     sel_row['n_view_comp'],
                     sel_row['n_blocks_est'],
                     sel_row['n_comp_est']))


def plot_Pi(Pi, mask=None, cmap="Blues", cbar=True, square=True,
            force_annot_off=False, linewidths=0):
    """
    Plots estimated Pi matrix.
    Transposes so the first view is on the columns and the second view
    is on the rows.
    """
    # TODO: allow for labels on each axis

    annot = max(Pi.shape) <= 10

    if force_annot_off:
        annot = False

    if mask is not None:
        mask = mask.T

    sns.heatmap(Pi.T, square=square, cbar=cbar, vmin=0, cmap=cmap,
                annot=annot, fmt='.3f', mask=mask,
                linewidths=linewidths,
                xticklabels=True, yticklabels=True)


def plot_mvmm(mvmm, inches=8, save_dir=None):
    """
    Plots loss history and estimated Pi matrix.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # TODO: maybe add Pi start
    loss_vals = mvmm.opt_data_['history']['loss_val']

    ################
    # Loss history #
    ################
    plot_loss_history(loss_vals,
                      loss_name='Observed data negative log-likelihood')

    if save_dir is not None:
        fpath = join(save_dir, 'loss_history.png')
        save_fig(fpath)

    ###############
    # Pi estimate
    ################
    plt.figure(figsize=(inches, inches))
    plot_Pi(mvmm.weights_mat_)
    plt.title("Estimated Pi")

    if save_dir is not None:
        fpath = join(save_dir, 'Pi_est.png')
        save_fig(fpath)

#######################
# Block diagonal MVMM #
#######################


def plot_bd_mvmm(mvmm, inches=8, save_dir=None):
    """
    Initial BD weights, Estimated BD weights, spectrums of both
    Number of steps in each adaptive stage
    Evals of entire path
    Loss history for each segment
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    info = get_bd_mvmm_info(mvmm)
    if save_dir is not None:
        # TODO: save this
        save_dir
    else:
        print(info)

    # BD weight estimate
    bd_weights = mvmm.bd_weights_
    zero_thresh = mvmm.zero_thresh
    summary, Pi_comm = community_summary(bd_weights, zero_thresh=zero_thresh)
    bd_weights_symlap_spec = eigh_sym_laplacian_bp(bd_weights)[0]

    # initial BD weights
    bd_weights_init = mvmm.opt_data_['adpt_opt_data']['adapt_pen_history']['opt_data'][0]['history']['init_params']['bd_weights']
    bd_weights_init_symlap_spec = eigh_sym_laplacian_bp(bd_weights_init)[0]

    # optimization history
    adpt_history = mvmm.opt_data_['adpt_opt_data']['adapt_pen_history']['opt_data']

    n_steps = [len(adpt_history[i]['history']['raw_eval_sum'])
               for i in range(len(adpt_history))]
    n_steps_cumsum = np.cumsum(n_steps)

    raw_eval_sum = np.concatenate([adpt_history[i]['history']['raw_eval_sum']
                                  for i in range(len(adpt_history))])

    obs_nll = np.concatenate([adpt_history[i]['history']['obs_nll']
                              for i in range(len(adpt_history))])

    if mvmm.opt_data_['ft_opt_data'] is not None:
        fine_tune_obs_nll = mvmm.opt_data_['ft_opt_data']['history']['obs_nll']
    else:
        fine_tune_obs_nll = None

    ######################
    # Initial BD weights #
    ######################
    plt.figure(figsize=(inches, inches))
    plot_Pi(bd_weights_init)
    plt.title('BD weights initial value')

    if save_dir is not None:
        fpath = join(save_dir, 'BD_weights_init.png')
        save_fig(fpath)

    ########################
    # Estimated BD weights #
    ########################
    plt.figure(figsize=(2 * inches, inches))
    plt.subplot(1, 2, 1)
    plot_Pi(bd_weights)
    plt.title('BD weights estimate, n_blocks={}'.
              format(summary['n_communities']))

    plt.subplot(1, 2, 2)
    plot_Pi(bd_weights, mask=Pi_comm < zero_thresh)
    plt.title('BD weights estimate, block diagonal perm')

    if save_dir is not None:
        fpath = join(save_dir, 'BD_weights_est.png')
        save_fig(fpath)

    ##########################
    # Spectrum of BD weights #
    ##########################
    plt.figure(figsize=(inches, inches))
    idxs = np.arange(1, len(bd_weights_symlap_spec) + 1)
    plt.plot(idxs, bd_weights_symlap_spec, marker='.', label='Estimate')
    plt.plot(idxs, bd_weights_init_symlap_spec, marker='.', label="Initial")
    plt.title('BD weights estimate spectrum')
    plt.ylim(0)
    plt.legend()

    if save_dir is not None:
        fpath = join(save_dir, 'BD_weights_spectrum.png')
        save_fig(fpath)

    ##################################
    # Number of steps for each stage #
    ##################################
    plt.figure(figsize=(inches, inches))
    idxs = np.arange(1, len(n_steps) + 1)
    plt.plot(idxs, n_steps, marker='.')
    plt.ylim(0)
    plt.ylabel("Number of steps")
    plt.xlabel("Adaptive stage")

    if save_dir is not None:
        fpath = join(save_dir, 'n_steps.png')
        save_fig(fpath)

    ###########################
    # Obs NLL for entire path #
    ###########################
    plt.figure(figsize=[inches, inches])
    plot_loss_history(obs_nll, loss_name="Obs NLL (entire path)")

    if save_dir is not None:
        fpath = join(save_dir, 'path_obs_nll.png')
        save_fig(fpath)

    #########################
    # Evals for entire path #
    #########################
    plt.figure(figsize=[inches, inches])
    plt.plot(np.log10(raw_eval_sum), marker='.')
    plt.ylabel('log10(sum smallest evals)')
    plt.xlabel('step')
    plt.title('Eigenvalue history (entire path)')
    for s in n_steps_cumsum:
        plt.axvline(s - 1, color='grey')

    if save_dir is not None:
        fpath = join(save_dir, 'path_evals.png')
        save_fig(fpath)

    ###########################
    # Losses for each segment #
    ###########################
    if save_dir is not None:
        segment_dir = join(save_dir, 'segments')
        os.makedirs(segment_dir, exist_ok=True)

    for i in range(len(adpt_history)):
        loss_vals = adpt_history[i]['history']['loss_val']
        plot_loss_history(loss_vals, 'loss val, adapt segment {}'.
                          format(i + 1))

        if save_dir is not None:
            fpath = join(segment_dir, 'loss_history_{}.png'.format(i + 1))
            save_fig(fpath)

    ##########################
    # fine tune loss history #
    ##########################
    if fine_tune_obs_nll is not None:
        plot_loss_history(fine_tune_obs_nll, 'fine tune obs NLL')

        if save_dir is not None:
            fpath = join(segment_dir, 'fine_tune_loss_history.png')
            save_fig(fpath)


def get_bd_mvmm_info(mvmm):
    info = {"sucess": mvmm.opt_data_['success'],
            "n_blocks_req": mvmm.n_blocks,
            "n_blocks_est": mvmm.opt_data_['n_blocks_est'],
            "adpat_opt_runtime": mvmm.opt_data_['adpt_opt_data']["runtime"]}

    if mvmm.opt_data_['ft_opt_data'] is not None:
        info["fine_tune_runtime"] = mvmm.opt_data_['ft_opt_data']["runtime"]

    info['eval_pen_inits'] = mvmm.opt_data_['adpt_opt_data']['eval_pen_init']
    return info


def plot_log_pen_mvmm(mvmm, inches=8, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # info = get_log_pen_mvmm_info(mvmm)
    # if save_dir is not None:
    #     # TODO: save this
    #     save_dir
    # else:
    #     print(info)

    Pi = mvmm.weights_mat_
    zero_thresh = 1e-10  # not sure if we need this

    summary, Pi_comm = community_summary(Pi, zero_thresh=zero_thresh)
    Pi_symlap_spec = eigh_sym_laplacian_bp(Pi)[0]

    if 'init_params' in mvmm.opt_data_['history']:
        Pi_init = mvmm.opt_data_['history']['weights'].reshape(Pi.shape)  # TODO: check this
        Pi_init_symlap_spec = eigh_sym_laplacian_bp(Pi_init)[0]
    else:
        Pi_init = None

    obs_nll = mvmm.opt_data_['history']['obs_nll']
    loss_vals = mvmm.opt_data_['history']['loss_val']

    ####################
    # Initial weights #
    ###################
    if Pi_init is not None:

        plt.figure(figsize=(inches, inches))
        plot_Pi(Pi_init)
        plt.title('weights initial value')

        if save_dir is not None:
            fpath = join(save_dir, 'weights_init.png')
            save_fig(fpath)

    ######################
    # Estimated  weights #
    ######################
    plt.figure(figsize=(2 * inches, inches))
    plt.subplot(1, 2, 1)
    plot_Pi(Pi)
    plt.title('weights estimate, n_blocks={}'.
              format(summary['n_communities']))

    plt.subplot(1, 2, 2)
    plot_Pi(Pi, mask=Pi_comm < zero_thresh)
    plt.title('weights estimate, block diagonal perm')

    if save_dir is not None:
        fpath = join(save_dir, 'weights_est.png')
        save_fig(fpath)

    ##########################
    # Spectrum of BD weights #
    ##########################
    plt.figure(figsize=(inches, inches))
    idxs = np.arange(1, len(Pi_symlap_spec) + 1)
    plt.plot(idxs, Pi_symlap_spec, marker='.', label='Estimate')
    if Pi_init is not None:
        plt.plot(idxs, Pi_init_symlap_spec, marker='.', label="Initial")
    plt.title('weights estimate spectrum')
    plt.ylim(0)
    plt.legend()

    if save_dir is not None:
        fpath = join(save_dir, 'weights_spectrum.png')
        save_fig(fpath)

    ###########################
    # Obs NLL for entire path #
    ###########################
    plt.figure(figsize=[inches, inches])
    plot_loss_history(obs_nll,
                      loss_name="Obs NLL")

    if save_dir is not None:
        fpath = join(save_dir, 'obs_nll.png')
        save_fig(fpath)

    plt.figure(figsize=[inches, inches])
    plot_loss_history(loss_vals,
                      loss_name="log penalized obs nll")

    if save_dir is not None:
        fpath = join(save_dir, 'loss_vals.png')
        save_fig(fpath)


def plot_mvmm_pcs(mvmm, X, dataset_names=None):

    if dataset_names is None:
        dataset_names = ['view 1', 'view 2']

    y_pred = mvmm.predict(X)
    n_comp = mvmm.n_components
    n_view_comps = mvmm.n_view_components
    overall_clust_colors = sns.color_palette("Set2", n_comp)
    data_colors = np.array(overall_clust_colors)[y_pred]

    view_clust_colors = [[None for _ in range(n_view_comps[0])],
                         [None for _ in range(n_view_comps[1])]]

    for k in range(n_comp):
        col = overall_clust_colors[k]
        k0, k1 = mvmm._get_view_clust_idx(k)
        view_clust_colors[0][k0] = col
        view_clust_colors[1][k1] = col

    for v in range(len(X)):
        # get firt two PCs of data
        # TODO: should we standarize data first or something
        U, D, V, m = pca(X[v], rank=2)

        # project data and means onto PCs
        proj_data = X[v] @ V  # U * D

        gmm = mvmm.view_models_[v]

        proj_means = gmm.means_ @ V  # (gmm.means_ - m) @ V

        # data_colors = 'black'

        # plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, v + 1)
        plt.scatter(proj_data[:, 0], proj_data[:, 1], color=data_colors, s=10)

        for j in range(n_view_comps[v]):

            plt.scatter(proj_means[j, 0], proj_means[j, 1],
                        marker='x', s=200, lw=5,
                        color=view_clust_colors[v][j])

            # get class covariance
            if gmm.covariance_type == 'diag':
                cov = np.diag(gmm.covariances_[j])

            # project covariance matrix onto PCs
            proj_cov = (V.T @ cov @ V)  # TODO: does this make sense

            draw_ellipse(position=proj_means[j, :], covariance=proj_cov,
                         facecolor='none',
                         edgecolor=view_clust_colors[v][j], lw=1)

        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
