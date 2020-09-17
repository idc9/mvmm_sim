#!/usr/bin/python
import matplotlib.pyplot as plt
import os
from os.path import join
import seaborn as sns
import numpy as np
import matplotlib as mpl
from itertools import product
from copy import deepcopy
from matplotlib.ticker import StrMethodFormatter

import argparse

from mvmm_sim.simulation.Paths import Paths
# from mvmm.simulation.utils import safe_drop
from mvmm_sim.viz_utils import axhline_with_tick, \
    set_yaxis_int_ticks, axvline_with_tick  # set_xaxis_int_ticks
from mvmm_sim.simulation.sim_viz import plot_grouped_var_per_n, \
    plot_n_samples_vs_best_tune_metric, value_per_n_samples_hline, \
    plot_clf_results, save_fig, plot_grouped_x_vs_y, \
    plot_sp_mvmm_pen_path, plot_sp_opt_history, plot_model_fit_times
from mvmm_sim.simulation.viz_bd_pi_ests import plot_pi_ests_bd_mvmm, \
    plot_pi_ests_log_pen, plot_pi_ests_sp_mvmm
from mvmm_sim.simulation.utils import safe_drop_list
from mvmm_sim.simulation.load_results4viz import load_results, get_timing_data
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.simulation.viz_2d_results import plot_estimator, plot_true
from mvmm_sim.viz_utils import simpleaxis
from mvmm_sim.data_analysis.multi_view.viz_resuls import plot_Pi

parser = argparse.\
    ArgumentParser(description='Make simulation results visualization.')
parser.add_argument('--sim_name', type=str,
                    help='Which simulation to run.')
args = parser.parse_args()

sim_name = args.sim_name
# sim_name = 'calum_new'  # 'sparse_pi', 'block_diag_pi, 'mini'

results, model_dfs, model_dfs_at_truth, \
    fit_models, pi_true_summary, \
    n_samples_tr_seq, zero_thresh, data = load_results(sim_name)

timing_data = get_timing_data(results, fit_models)

n_comp_tot_true = pi_true_summary['n_comp_tot_true']
n_blocks_true = pi_true_summary['n_blocks_true']

models2exclude = []
for model_name in ['log_pen_mvmm', 'bd_mvmm', 'sp_mvmm']:
    if model_dfs[model_name].shape[0] == 0:
        models2exclude.append(model_name)

# where to save simulation results
results_save_dir = make_and_get_dir(Paths().results_dir, sim_name)
# results_save_dir = join(Paths().results_dir, sim_name)
# os.makedirs(results_save_dir, exist_ok=True)

##################
# Set parameters #
##################

# model names
model_names = {'log_pen_mvmm': 'log penalized MVMM',
               'bd_mvmm': 'block diagonal MVMM',
               'cat_gmm': 'Mixture model on concatenated data',
               'full_mvmm': 'MVMM',
               'sp_mvmm': 'spectral penalized MVMM',
               'marginal_view_0': 'Mixture model on view 1 marginal data',
               'marginal_view_1': 'Mixture model on view 2 marginal data'}

for model in model_names.keys():
    os.makedirs(join(results_save_dir, model), exist_ok=True)

# # add view resuls  models
# for v in range(2):
#     for k in ['bd_mvmm', 'log_pen_mvmm']:
#         new_key = k + '_view_' + str(v)
#         model_names[new_key] = model_names[k] + \
#             ', on view {} marginal data'.format(v)

# model colors
color_pal = sns.color_palette("Set2", len(model_names))
model_colors = {k: color_pal[i] for (i, k) in enumerate(model_names.keys())}

# model markers
markers = ['s', 'o', 'P', 'x', '8', '3', 'p']
model_markers = {k: markers[i] for (i, k) in enumerate(model_names.keys())}


model2tune_param = {'bd_mvmm': 'n_blocks_est',
                    'log_pen_mvmm': 'tune__pen',
                    'cat_gmm': 'n_comp_est',
                    'sp_mvmm': 'tune__sp_pen'}

models_by_n_comp_est = ['log_pen_mvmm']  # 'bd_mvmm', 'sp_mvmm'
models_by_n_block_est = ['log_pen_mvmm', 'sp_mvmm']

tune_param_labels = {'n_blocks_est': 'Number of blocks',
                     'n_comp_est': 'Number of components',
                     'tune__sp_pen': 'Spectral penalty value',
                     'tune__pen': 'Log penalty weight'}


# measure names
measure_labels = {'test_overall_ars': 'Cluster adjusted Rand index',
                  # 'test_overall_nmi': 'Cluster label NMI',
                  # 'test_overall_hs': 'Cluster label HS',
                  # 'test_overall_amis': 'Cluster label AMI',
                  # 'test_overall_mi': 'Cluster label MI',

                  'test_view_0_ars': 'View 1 cluster label ARS',
                  'test_view_1_ars': 'View 2 cluster label ARS',

                  # 'pi_graph_acc_norm': 'Graph accuracy (random walk)',
                  'pi_aligned_dist': 'Aligned Pi error (mean absolute error)',

                  'test_community_ars': 'Block level adjusted Rand index',

                  'test_community_restr_ars': 'Block level adjusted Rand index',

                  'n_blocks_est': 'Estimated number of blocks',

                  'n_comp_est': 'Number of components',

                  'bic': 'BIC',

                  'aic': 'AIC',

                  'fit_time': 'Fit time'}

measure2models = {'test_overall_ars': ['cat_gmm', 'full_mvmm', 'log_pen_mvmm',
                                       'bd_mvmm', 'sp_mvmm'],

                  'test_community_ars': ['full_mvmm', 'log_pen_mvmm',
                                         'bd_mvmm', 'sp_mvmm'],

                  'test_community_restr_ars': ['full_mvmm', 'log_pen_mvmm',
                                               'bd_mvmm', 'sp_mvmm'],

                  'pi_aligned_dist': ['full_mvmm', 'log_pen_mvmm',
                                      'bd_mvmm', 'sp_mvmm'],

                  'n_blocks_est': ['log_pen_mvmm', 'sp_mvmm', 'bd_mvmm'],

                  'n_comp_est': ['cat_gmm', 'log_pen_mvmm', 'bd_mvmm',
                                 'sp_mvmm'],

                  'bic': ['cat_gmm', 'full_mvmm', 'log_pen_mvmm',
                          'bd_mvmm', 'sp_mvmm'],

                  'aic': ['cat_gmm', 'full_mvmm', 'log_pen_mvmm',
                          'bd_mvmm', 'sp_mvmm'],

                  'fit_time': ['cat_gmm', 'full_mvmm', 'log_pen_mvmm',
                               'bd_mvmm', 'sp_mvmm']}


def measure_model_tune_iter(measures, models2exclude):
    for measure in measures:
        for model in measure2models[measure]:
            if model == 'full_mvmm' or model in models2exclude:
                continue
            else:
                yield measure, model


plot_interval = 'std'
include_full = True

# visualization opetions
inches = 10
dpi = 200
mpl.rcParams['font.size'] = 25
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['legend.fontsize'] = 25
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


ylims = {'test_overall_ars': None,  # [0, None],  # [0, 1]
         'test_view_0_ars': [0, None],  # [0, 1]
         'test_view_1_ars': [0, None],  # [0, 1]
         'pi_graph_acc_norm': [0, None],
         'pi_aligned_dist': [0, None],  # [0, 1]
         'test_community_ars': None,  # [0, None],  # [0, None],  # [0, 1]
         'test_community_restr_ars': None,  # [0, None],
         'n_blocks_est': [0, None],
         'fit_time': [0, None],
         'n_comp_est': None,
         'bic': None,
         'aic': None,
         'silhouette': None,
         'calinski_harabasz': None,
         'davies_bouldin': None,
         'dunn': None}


yticks = {'test_overall_ars': None,  # np.linspace(0, 1, 11),
          'test_view_0_ars': None,  # np.linspace(0, 1, 11),
          'test_view_1_ars': None,  # np.linspace(0, 1, 11),
          'pi_graph_acc_norm': None,  # np.linspace(0, 1, 11),
          'pi_aligned_dist': None,  # np.linspace(0, 1, 11),
          'test_community_ars': None,  # np.linspace(0, 1, 11),
          'test_community_restr_ars': None,
          'n_blocks_est': None,
          'fit_time': None,
          'n_comp_est': None,
          'bic': None,
          'aic': None,
          'silhouette': None,
          'calinski_harabasz': None,
          'davies_bouldin': None,
          'dunn': None}

###########################
# classification accuracy #
###########################
plt.figure(figsize=[inches, inches])
plot_clf_results(results)
save_fig(join(results_save_dir, 'clf_acc'))

###########
# Pi True #
###########
Pi_true = pi_true_summary['Pi']
Pi_true_mask = Pi_true == 0

# non-paper version of Pi_true
plt.figure(figsize=[inches, inches])
sns.heatmap(Pi_true.T,
            cmap='Blues', square=True, cbar=False, vmin=0)
plt.xlabel('View 1 clusters')
plt.ylabel('View 2 clusters')
title = 'Pi True\nNumber of blocks={}\nBlock shapes: {}'.\
    format(pi_true_summary['summary']['n_communities'],
           pi_true_summary['summary']['comm_shapes'])
plt.title(title)
save_fig(join(results_save_dir, 'Pi_true_detailed.png'))


# paper verison of mask
plt.figure(figsize=(inches, inches))
plot_Pi(Pi_true, force_annot_off=True, cbar=False, mask=Pi_true_mask,
        linewidths=.5)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
save_fig(join(results_save_dir, 'pi_true.png'), dpi=dpi)


##################################
# n samples vs. measure at truth #
##################################

for measure in ['test_overall_ars', 'test_community_ars', 'pi_aligned_dist',
                'n_blocks_est', 'test_community_restr_ars']:  # n_comp_est

    to_show = measure2models[measure]
    to_show = safe_drop_list(to_show, models2exclude)
    # to_show = safe_drop(to_show, 'sp_mvmm')  # TODO: maybe implement this

    dfs_at_truth = [model_dfs_at_truth[k] for k in to_show]
    labels = [model_names[k] for k in to_show]
    colors = [model_colors[k] for k in to_show]
    markers = [model_markers[k] for k in to_show]

    # adjust MVMM label for block measures
    if measure in ['test_community_ars', 'test_community_restr_ars'] and\
            'MVMM' in labels:
        labels[labels == 'MVMM'] = 'MVMM (bipartite spectral clustering)'

    plt.figure(figsize=[inches, inches])
    plot_grouped_x_vs_y(dfs_at_truth,
                        group_var='n_samples',
                        value_var=measure,
                        labels=labels,
                        plot_interval='std',
                        colors=colors,
                        markers=markers)
    plt.legend()
    plt.xlabel("Number of samples")

    plt.xlim(0)
    # plt.ylim(0)
    plt.ylim(ylims[measure])
    plt.yticks(yticks[measure], yticks[measure])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylabel(measure_labels[measure])
    simpleaxis(plt.gca())

    save_fig(join(results_save_dir,
                  'n_samples_vs_{}_at_truth.png'.format(measure)),
             dpi=dpi)


############################################
# n_samples vs. measure at model selection #
############################################
# select_metrics = ['bic', 'aic', 'silhouette',
#                   'calinski_harabasz', 'davies_bouldin', 'dunn']

for select_metric, measure in \
        product(['bic'],  # select_metrics,  # ['bic', 'aic'],
                ['test_overall_ars',
                'test_community_ars',
                 'test_community_restr_ars',
                 'pi_aligned_dist', 'n_blocks_est', 'n_comp_est']):

    to_show = measure2models[measure]
    to_show = safe_drop_list(to_show, models2exclude)
    to_show = safe_drop_list(to_show, ['cat_gmm'])
    dfs = [model_dfs[k] for k in to_show]
    labels = [model_names[k] for k in to_show]
    colors = [model_colors[k] for k in to_show]
    markers = [model_markers[k] for k in to_show]

    # adjust MVMM label for block measures
    if measure in ['test_community_ars', 'test_community_restr_ars'] and\
            'MVMM' in labels:
        labels[labels == 'MVMM'] = 'MVMM (bipartite spectral clustering)'

    plt.figure(figsize=[inches, inches])
    plot_n_samples_vs_best_tune_metric(dfs=dfs,
                                       measure=measure,
                                       labels=labels,
                                       select_metric=select_metric,
                                       plot_interval='std',
                                       colors=colors,
                                       markers=markers)
    plt.xlim(0)
    # plt.ylim(0)
    plt.ylim(ylims[measure])
    plt.yticks(yticks[measure], yticks[measure])

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    plt.ylabel(measure_labels[measure])

    if measure == 'n_comp_est':
        axhline_with_tick(n_comp_tot_true, bold=True, color='black')
        set_yaxis_int_ticks()

    elif measure == 'n_blocks_est':
        axhline_with_tick(n_blocks_true, bold=True, color='black')
        set_yaxis_int_ticks()

    plt.legend()

    simpleaxis(plt.gca())

    save_fig(join(results_save_dir,
                  'n_samples_vs_{}_at_{}_selected.png'.
                  format(measure, select_metric)),
             dpi=dpi)

#######################
# tune parameter grid #
#######################

tune_measures = ['test_overall_ars',
                 'test_community_ars', 'test_community_restr_ars',
                 'pi_aligned_dist',
                 'n_blocks_est', 'n_comp_est', 'fit_time']

for (measure, model), group_var in \
        product(measure_model_tune_iter(tune_measures, models2exclude),
                ['EXTRA__n_comp_est', 'EXTRA__n_blocks_est', 'TUNE']):

    dfs = model_dfs[model]

    if group_var == 'TUNE':
        group_var = model2tune_param[model]
        # if model in ['log_pen_mvmm', 'sp_mvmm']:
        #     dfs = model_dfs_all_tune_vals[model]

        # if using sp by block then we don't have continuous spectral penalty
        if model == 'sp_mvmm' and 'tune__sp_pen' not in dfs.columns:
            continue

    elif group_var == 'EXTRA__n_comp_est':
        group_var = 'n_comp_est'
        if model not in models_by_n_comp_est:
            continue

    elif group_var == 'EXTRA__n_blocks_est':
        group_var = 'n_blocks_est'
        if model not in models_by_n_block_est:
            continue

    if group_var == measure:
        continue

    if model not in measure2models[measure]:
        raise ValueError('Oops')

    plt.figure(figsize=[inches, inches])

    plot_grouped_var_per_n(dfs,
                           group_var=group_var,
                           value_var=measure,
                           plot_scatter=False,
                           plot_interval='std',
                           force_x_ticks=False,
                           plot_summary=True,
                           s=200)

    # plt.xlim(0)
    # plt.ylim(0)
    plt.ylim(ylims[measure])
    # plt.yticks(yticks[measure], yticks[measure])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylabel(measure_labels[measure])

    # set_xaxis_int_ticks()
    # axvline_with_tick(true_summary['n_communities'], color='black')
    plt.xlabel(tune_param_labels[group_var])

    if group_var == 'n_blocks_est':
        axvline_with_tick(n_blocks_true, color='black')

    elif group_var == 'n_comp_est':
        axvline_with_tick(n_comp_tot_true, color='black')

    # this is occasionally causing some memory issues
    # add full results
    # if include_full and \
    #         measure not in ['fit_time', 'n_comp_est', 'n_blocks_est']:
    #     value_per_n_samples_hline(model_dfs['full_mvmm'],
    #                               value_var=measure,
    #                               plot_interval=False)
    simpleaxis(plt.gca())

    save_fig(join(results_save_dir, model,
                  'tune_{}_vs_{}.png'.format(group_var, measure)))

#########################
# Plot model sel curves #
#########################
for select_measure, model in product(['bic', 'aic'], ['bd_mvmm',
                                                      'log_pen_mvmm',
                                                      'cat_gmm']):

    if model in models2exclude:
        continue

    elif model == 'bd_mvmm':
        tune_param = 'n_blocks_est'
        true_val = n_blocks_true

    elif model in ['log_pen_mvmm', 'cat_gmm']:
        tune_param = 'n_comp_est'
        true_val = n_comp_tot_true

    df = model_dfs[model]

    all_mc_idxs = set(df['mc_index'])
    all_n_samples = set(df['n_samples'])

    for n_samples in all_n_samples:
        plt.figure(figsize=[inches, inches])
        for mc_index in all_mc_idxs:

            exper_df = df.query("n_samples == @n_samples").\
                query("mc_index == @mc_index")
            plt.plot(exper_df['n_blocks_est'], exper_df['bic'],
                     marker='.', color='black')
            plt.ylabel(measure_labels[select_measure])
            plt.xlabel(tune_param_labels[tune_param])

            best_idx = exper_df['bic_best_tune_idx'].values[0]
            best_val = exper_df.\
                query("tune_idx == @best_idx")[tune_param].values[0]
            plt.axvline(best_val, color='black')

        plt.axvline(true_val, label="Truth {}".format(true_val),
                    color='red', ls='--')
        plt.legend()

        save_fig(join(results_save_dir, model,
                      '{}_model_selection_n={}.png'.format(select_measure,
                                                           n_samples)))

1 / 0
##############################
# tune grid, start vs. final #
##############################

for measure, model in measure_model_tune_iter(['test_overall_ars',
                                               'pi_aligned_dist'],
                                              models2exclude):

    group_var = model2tune_param[model]

    if model in ['sp_mvmm', 'cat_gmm']:
        continue

    # if model in ['log_pen_mvmm']:
    #     df = model_dfs_all_tune_vals[model]
    # else:
    df = model_dfs[model]

    df_final = df[['n_samples', group_var, measure]]
    df_start = df[['n_samples', group_var, 'start_' + measure]]
    df_start = df_start.rename(columns={'start_' + measure: measure})
    diff = deepcopy(df_final)
    diff[measure] = df_final[measure] - df_start[measure]

    plt.figure(figsize=[inches, inches])
    plot_grouped_var_per_n(diff,
                           group_var=group_var,
                           value_var=measure,
                           plot_scatter=False,
                           plot_interval=False)

    # plt.xlim(0)
    # plt.ylim(0)
    # plt.ylim(ylims[measure])
    # plt.yticks(yticks[measure], yticks[measure])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylabel(measure_labels[measure])

    plt.title('Difference final - start')
    plt.axhline(0, color='black')

    save_fig(join(results_save_dir, model,
                  'start_vs_final_{}_vs_{}.png'.format(group_var, measure)))


################
# SP MVMM path #
################

if 'sp_mvmm' not in models2exclude:
    sp_n_block_est_seqs = model_dfs['sp_mvmm'].\
        groupby(['n_samples', 'mc_index'])['n_blocks_est'].\
        agg(lambda x: list(set(x))).\
        reset_index()

    sp_n_block_est_seqs.to_csv(join(results_save_dir, 'sp_mvmm',
                               'n_block_est_seqs.csv'))

    # penalty path summary
    for n_samples in n_samples_tr_seq:
        init_fit_data = fit_models[n_samples]['sp_mvmm'].init_fit_data_

        plot_sp_mvmm_pen_path(init_fit_data)

        save_fig(join(results_save_dir, 'sp_mvmm',
                 'n_{}__pen_path.png'.format(n_samples)))

    # loss history
    for n_samples in fit_models.keys():
        sp_mvmm_gs = fit_models[n_samples]['sp_mvmm']
        save_dir = join(results_save_dir,
                        'sp_mvmm', 'opt', 'n_{}'.format(n_samples))

    plot_sp_opt_history(sp_mvmm_gs, save_dir)

    # estimated Pi matrix
    plot_pi_ests_sp_mvmm(fit_models, zero_thresh=zero_thresh,
                         inches=inches, nrows=6)

    save_fig(join(results_save_dir, 'sp_mvmm',
                  'sp_mvmm__est_Pi_mats.png'))

#########################
# Estimated Pi matrices #
#########################

if 'log_pen_mvmm' not in models2exclude:
    plot_pi_ests_log_pen(fit_models, zero_thresh=zero_thresh,
                         nrows=5, inches=inches)
    save_fig(join(results_save_dir, 'log_pen_mvmm',
                  'log_pen_mvmm__est_Pi_mats.png'))

if 'bd_mvmm' not in models2exclude:
    plot_pi_ests_bd_mvmm(fit_models, zero_thresh=zero_thresh, inches=inches)
    save_fig(join(results_save_dir, 'bd_mvmm',
                  'bd_mvmm__est_Pi_mats.png'))


############
# Runtimes #
############
time_dir = make_and_get_dir(results_save_dir, 'runtimes')

# total simulation runtimes
plt.figure(figsize=(inches, inches))
plt.title("Average total runtime: {:1.2f} hours".
          format(timing_data['tot_runtime'].mean()))
plt.scatter(timing_data['n_samples_tr'], timing_data['tot_runtime'])
plt.xlabel('Number of samples')
plt.ylabel('Total simulation runtime (hours)')
save_fig(join(time_dir, 'simulation_total_runtime.png'))

# model fitting runtimes
to_show = ['bd_mvmm', 'sp_mvmm', 'log_pen_mvmm', 'full_mvmm', 'cat_gmm']
to_show = safe_drop_list(to_show, models2exclude)
for model_name in to_show:

    plt.figure(figsize=(3.5 * inches, inches))
    plot_model_fit_times(timing_data, model_name,
                         name_mapping=model_names[model_name])

    save_fig(join(time_dir, '{}_runtime.png'.format(model_name)))


##########################
# 2D data visualizations #
##########################
is_2d_data = (data['X_tr'][n_samples_tr_seq[0]][0].shape[1] == 1) and \
    (data['X_tr'][n_samples_tr_seq[0]][1].shape[1] == 1)

if is_2d_data:

    view_params = data['view_params']
    Pi_true = data['Pi_true']

    for n_samples in n_samples_tr_seq:
        X_tr = data['X_tr'][n_samples]
        models = fit_models[n_samples]

        two_d_plots_dir = make_and_get_dir(results_save_dir,
                                           'two_d_plots',
                                           'n={}'.format(n_samples))

        # plot true data
        plt.figure(figsize=(2 * inches, inches))
        plot_true(view_params, Pi_true, X_tr)
        save_fig(join(two_d_plots_dir, 'true.png'))

        # plot full MVMM
        plt.figure(figsize=(2 * inches, inches))
        plot_estimator(mvmm=models['full_mvmm'], X_tr=X_tr)
        save_fig(join(two_d_plots_dir, 'full_mvmm.png'))

        if 'log_pen_mvmm' not in models2exclude:
            # plot log pen mvmm
            plt.figure(figsize=(2 * inches, inches))
            plot_estimator(mvmm=models['log_pen_mvmm'].best_estimator_,
                           X_tr=X_tr)
            save_fig(join(two_d_plots_dir,
                     'log_pen_mvmm_best_est.png'))

            # log pen mvm tuning sequence
            log_pen2d_dir = make_and_get_dir(two_d_plots_dir, 'log_pen_mvmm')
            for tune_idx, mvmm in\
                    enumerate(models['log_pen_mvmm'].estimators_):
                plt.figure(figsize=(2 * inches, inches))
                plot_estimator(mvmm, X_tr=X_tr)
                save_fig(join(log_pen2d_dir, 'tune_{}.png'.format(tune_idx)))

        if 'bd_mvmm' not in models2exclude:
            # bd mvmm
            plt.figure(figsize=(2 * inches, inches))
            plot_estimator(mvmm=models['bd_mvmm'].best_estimator_, X_tr=X_tr)
            save_fig(join(two_d_plots_dir, 'bd_mvmm.png'))

            # bd mvmm tuning sequence
            bd_2d_dir = make_and_get_dir(two_d_plots_dir, 'bd_mvmm')
            for tune_idx, mvmm in enumerate(models['bd_mvmm'].estimators_):
                plt.figure(figsize=(2 * inches, inches))
                plot_estimator(mvmm, X_tr=X_tr)
                save_fig(join(bd_2d_dir, 'tune_{}.png'.format(tune_idx)))

        if 'sp_mvmm' not in models2exclude:

            # sp mvmm best estimator
            plt.figure(figsize=(2 * inches, inches))
            plot_estimator(mvmm=models['sp_mvmm'].best_estimator_, X_tr=X_tr)
            save_fig(join(two_d_plots_dir, 'sp_mvmm_best_estimator.png'))

            # sp mvmm tuning sequence
            sp_mvmm2d_dir = make_and_get_dir(two_d_plots_dir, 'sp_mvmm')
            for tune_idx, mvmm in enumerate(models['sp_mvmm'].estimators_):
                plt.figure(figsize=(2 * inches, inches))
                plot_estimator(mvmm, X_tr=X_tr)
                save_fig(join(sp_mvmm2d_dir, 'tune_{}.png'.
                         format(tune_idx)))
