#!/usr/bin/python
import numpy as np
from time import time
from joblib import dump
from copy import deepcopy
import os
import argparse
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_distances

from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.utils import sample_seed
from mvmm_sim.simulation.utils import format_mini_experiment
from mvmm_sim.simulation.from_args import add_parsers, \
    rand_means_data_dist_parser, grid_means_data_dist_parser,\
    pi_parser, \
    general_opt_parser, base_gmm_parser, bd_mvmm_parser, \
    log_pen_mvmm_parser, \
    pi_from_args, \
    rand_means_data_dist_from_args, grid_means_data_dist_from_args, \
    gmm_from_args, full_mvmm_from_args, \
    ts_log_pen_mvmm_from_args, ts_bd_mvmm_from_args, \
    spect_pen_parser, spect_pen_from_args
from mvmm_sim.simulation.load_results4viz import add_model_selection_by_measures
from mvmm_sim.simulation.utils import clf_fit_and_score
# from mvmm.multi_view.toy_data import setup_rand_view_params, sample_gmm, \
#     setup_grid_mean_view_params
from mvmm_sim.simulation.oracle_mvmm import format_view_params, \
    set_mvmm_from_params
from mvmm_sim.simulation.data_dist_from_config import get_data_dist

from mvmm.multi_view.utils import view_labs_to_overall
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary
# from mvmm.simulation.utils import get_empirical_pi
# from mvmm.multi_view.SpectralPenSearchMVMM import linspace_zero_to
from mvmm_sim.simulation.cluster_report import cluster_report
from mvmm_sim.simulation.opt_viz import summarize_bd
from mvmm_sim.simulation.run_sim import add_gs_results
# from mvmm.simulation.data_dist_from_config import get_pi
from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.simulation.models_from_config import get_mvmms, \
    get_single_view_models
# from mvmm.simulation.spect_pen_gif_utils import make_gifs
from mvmm_sim.simulation.utils import extract_tuning_param_vals
from mvmm_sim.simulation.opt_viz import plot_loss_history
from mvmm_sim.simulation.load_results4viz import get_best_expers_at_truthish
from mvmm_sim.simulation.sim_viz import plot_grouped_var_per_n, \
    value_per_n_samples_hline, plot_sp_mvmm_pen_path, save_fig

from mvmm_sim.simulation.viz_2d_results import plot_estimator, plot_true
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm.single_view.gaussian_mixture import default_cov_regularization


parser = argparse.ArgumentParser(description='Simulation with sparse Pi.')
parser.add_argument('--mini', action='store_true', default=False,
                    help='Run a mini simulation for debugging.')

parser.add_argument('--sim_name', default=None,
                    help='Name of the siulation.')

parser.add_argument('--metaseed', type=int, default=89921,
                    help='Seed that sets the seeds.')

parser.add_argument('--n_jobs', default=None,
                    help='n_jobs for grid search.')

parser.add_argument('--exclude_bd_mvmm', action='store_true', default=False,
                    help='Do not include block diagonal models.')

# parser.add_argument('--exclude_sp_mvmm', action='store_true', default=False,
#                     help='Do not include spectral penalty models.')

parser.add_argument('--exclude_log_pen_mvmm', action='store_true',
                    default=False,
                    help='Do no include log pen models.')

parser.add_argument('--grid_means', action='store_true', default=False,
                    help='Put cluster means on a grid.')
# parser.add_argument('--sp_user_seq', action='store_true', default=False,
#                     help='User specficed sp pen seqence.')
# grid = True
# if grid:
#     parser = grid_means_data_dist_parser(parser)

# else:
#     parser = rand_means_data_dist_parser(parser)

parser = add_parsers(parser,
                     to_add=[pi_parser,  # rand_means_data_dist_parser,
                             grid_means_data_dist_parser,
                             rand_means_data_dist_parser,
                             general_opt_parser, base_gmm_parser,
                             log_pen_mvmm_parser, bd_mvmm_parser,
                             spect_pen_parser])

parser = bayes_parser(parser)
args = parser.parse_args()
args = format_mini_experiment(args)
args.job_name = args.sim_name

bayes_submit(args)

if args.sim_name is None:
    args.sim_name = 'meow'

save_dir = make_and_get_dir(Paths().results_dir, 'single', args.sim_name)


res_writer = ResultsWriter(os.path.join(save_dir, 'results.txt'),
                           delete_if_exists=True)

res_writer.write('\n\n\n Input args')
res_writer.write(args)


rng = check_random_state(args.metaseed)

to_exclude = []
# if args.exclude_sp_mvmm:
to_exclude.append('sp_mvmm')
if args.exclude_bd_mvmm:
    to_exclude.append('bd_mvmm')
if args.exclude_log_pen_mvmm:
    to_exclude.append('log_pen_mvmm')

inches = 8

##############
# Data setup #
##############

pi_config = pi_from_args(args)
if args.grid_means:
    clust_param_config = grid_means_data_dist_from_args(args)
else:
    clust_param_config = rand_means_data_dist_from_args(args)

clust_param_config['random_state'] = sample_seed(rng)

# n_samples_tr = 800

n_samples_tst = 2000


##########
# Models #
##########

base_gmm_config = gmm_from_args(args)

view_gmm_config = deepcopy(base_gmm_config)
view_gmm_config['random_state'] = sample_seed(rng)

cat_gmm_config = deepcopy(base_gmm_config)
cat_gmm_config['random_state'] = sample_seed(rng)

full_mvmm_config = full_mvmm_from_args(args)
full_mvmm_config['random_state'] = sample_seed(rng)

log_pen_config = ts_log_pen_mvmm_from_args(args)
log_pen_config['two_stage']['random_state'] = sample_seed(rng)

# mult_values = np.concatenate(
#     [10 ** -np.linspace(start=2, stop=0, num=50, endpoint=False),
#      np.logspace(start=-15, stop=-2, num=20, endpoint=False)])
# mult_values = np.sort(mult_values)
# log_pen_config['mult_values'] = mult_values

bd_config = ts_bd_mvmm_from_args(args)
bd_config['two_stage']['random_state'] = sample_seed(rng)
bd_config['final']['history_tracking'] = 1

spect_pen_config = spect_pen_from_args(args)
spect_pen_config['search']['random_state'] = sample_seed(rng)

# if args.sp_user_seq:
#     sp_pen_vals = np.concatenate([linspace_zero_to(stop=20, num=200),
#                                   20 + linspace_zero_to(stop=80, num=160)])
#     # sp_pen_vals = np.concatenate([custom_linspace(stop=20, num=100),
#     #                               20 + custom_linspace(stop=80, num=160)])

#     spect_pen_config['search']['user_pen_vals'] = sp_pen_vals


if args.n_jobs is not None:
    n_jobs_tune = int(args.n_jobs)
else:
    n_jobs_tune = None

# model_config = {'cat_gmm_config': cat_gmm_config,
#                 'view_gmm_config': view_gmm_config,
#                 'base_gmm_config': base_gmm_config,
#                 'full_mvmm_config': full_mvmm_config,
#                 'log_pen_config': log_pen_config,
#                 'bd_config': bd_config,
#                 'spect_pen_config': spect_pen_config,
#                 'sp_by_block': not args.sp_not_by_block,
#                 'do_tuning': False,
#                 'select_metric': args.select_metric,
#                 'n_jobs_tune': n_jobs_tune}


single_view_config = {'cat_gmm_config': cat_gmm_config,
                      'view_gmm_config': view_gmm_config,
                      'n_jobs_tune': n_jobs_tune,
                      'select_metric': args.select_metric}

mvmm_config = {'base_gmm_config': base_gmm_config,
               'full_mvmm_config': full_mvmm_config,
               'log_pen_config': log_pen_config,
               'bd_config': bd_config,
               'spect_pen_config': spect_pen_config,
               # 'n_blocks': 'default',
               'sp_by_block': True,
               'select_metric': args.select_metric,
               'n_jobs_tune': n_jobs_tune}


#################################
# sample data and create models #
#################################

data_dist, Pi_true, view_params = \
    get_data_dist(clust_param_config=clust_param_config,
                  grid_means=args.grid_means,
                  pi_dist=pi_config['dist'],
                  pi_config=pi_config['config'])

n_view_components = Pi_true.shape
n_comp_true = (Pi_true > 0).sum()
n_blocks_true = community_summary(Pi_true)[0]['n_communities']


n_samples_tr = 100 * n_comp_true

X_tr, Y_tr = data_dist(n_samples=n_samples_tr,
                       random_state=sample_seed(rng))

X_tst, Y_tst = data_dist(n_samples=n_samples_tst,
                         random_state=sample_seed(rng))


res_writer.write('\n\n\n data paramters')
res_writer.write(clust_param_config['custom_view_kws'])
res_writer.write('n_samples_tr {}'.format(n_samples_tr))
res_writer.write('Pi_dist {}'.format(args.pi_name))

# update config with true values
# cat gmm, view gmms and block diag mvmm are all fit at true values
single_view_config['cat_n_comp'] = n_comp_true
single_view_config['view_n_comp'] = n_view_components
mvmm_config['n_blocks'] = n_blocks_true


# models = models_from_config(n_view_components=n_view_components,
#                             n_comp_tot=n_comp_true,
#                             n_blocks=n_blocks_true,
#                             **model_config)

# do tuning for log pen mvmm
# __model_config = deepcopy(model_config)
# __model_config['do_tuning'] = True
# models['log_pen_mvmm'] = \
#     models_from_config(n_view_components=n_view_components,
#                        n_comp_tot=n_comp_true,
#                        n_blocks=n_blocks_true,
#                        **__model_config)['log_pen_mvmm']
models = {**get_single_view_models(**single_view_config),

          **get_mvmms(n_view_components=n_view_components,
                      **mvmm_config)}

_view_params = format_view_params(view_params,
                                  covariance_type='full')

models['oracle'] = set_mvmm_from_params(view_params=_view_params,
                                        Pi=Pi_true,
                                        covariance_type='full')
#############################
# covariance regularization #
#############################
n_views = len(X_tr)
reg_covar = {}

# set cov reg for each view
for v in range(n_views):
    reg = default_cov_regularization(X=X_tr[v], mult=args.reg_covar_mult)

    models['full_mvmm'].base_view_models[v].set_params(reg_covar=reg)

    models['bd_mvmm'].base_start.base_view_models[v].\
        set_params(reg_covar=reg)
    models['bd_mvmm'].base_final.base_view_models[v].\
        set_params(reg_covar=reg)

    models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
        set_params(reg_covar=reg)
    models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
        set_params(reg_covar=reg)

    models['sp_mvmm'].base_mvmm_0.base_view_models[v].set_params(reg_covar=reg)
    models['sp_mvmm'].base_wbd_mvmm.base_view_models[v].\
        set_params(reg_covar=reg)

    # print and save
    reg_covar[v] = reg
    res_writer.write("\nCovarinace regularization for view {} is {}".
                     format(v, reg_covar[v]))
    stds = X_tr[v].std(axis=0)
    res_writer.write("Smallest variance: {}".format(stds.min() ** 2))
    res_writer.write("Largest variance: {}".format(stds.max() ** 2))

# for cat GMM
reg = default_cov_regularization(X=np.hstack(X_tr), mult=args.reg_covar_mult)
models['cat_gmm'].set_params(reg_covar=reg)
reg_covar['cat_gmm'] = reg

##############
# fit models #
##############

zero_thresh = .01 / (n_view_components[0] * n_view_components[1])

X_tr_precomp_dists = pairwise_distances(X=np.hstack(X_tr))

kws = {'sim_stub': {'mc_index': 0, 'n_samples': n_samples_tr},
       'X_tr': X_tr, 'Y_tr': Y_tr,
       'X_tst': X_tst, 'Y_tst': Y_tst,
       'Pi_true': Pi_true, 'view_params_true': view_params,
       'zero_thresh': zero_thresh,
       'X_tr_precomp_dists': X_tr_precomp_dists}


res_writer.write('\n\n\n')

results = {}
runtimes = {}

###########
# Cat GMM #
###########
start_time = time()
models['cat_gmm'].fit(np.hstack(X_tr))
runtimes['cat_gmm'] = time() - start_time
res_writer.write('\n\n\n cat gmm runtime {:1.2f} seconds'.
                 format(runtimes['cat_gmm']))


results['cat_gmm'] = add_gs_results(pd.DataFrame(),
                                    model=models['cat_gmm'],
                                    model_name='cat_gmm',
                                    dataset='full', view='both',
                                    **kws)

#############
# Full MVMM #
#############
start_time = time()
models['full_mvmm'].fit(X_tr)
runtimes['full_mvmm'] = time() - start_time
res_writer.write('\n\n\nfull mvmm runtime {:1.2f} seconds'.
                 format(runtimes['full_mvmm']))

results['full_mvmm'] = add_gs_results(pd.DataFrame(),
                                      model=models['full_mvmm'],
                                      model_name='full_mvmm',
                                      run_biptsp_on_full=True,
                                      dataset='full', view='both',
                                      **kws)


################
# log_pen MVMM #
################

if 'log_pen_mvmm' not in to_exclude:
    start_time = time()
    models['log_pen_mvmm'].fit(X_tr)
    runtimes['log_pen_mvmm'] = time() - start_time
    res_writer.write('\n\n\nlog_pen_mvmm runtime {:1.2f} seconds'.
                     format(runtimes['log_pen_mvmm']))

    results['log_pen_mvmm'] = add_gs_results(pd.DataFrame(),
                                             model=models['log_pen_mvmm'],
                                             model_name='log_pen_mvmm',
                                             dataset='full', view='both',
                                             **kws)

#######################
# Block diagonal MVMM #
#######################

if 'bd_mvmm' not in to_exclude:
    start_time = time()
    models['bd_mvmm'].fit(X_tr)
    runtimes['bd_mvmm'] = time() - start_time
    res_writer.write('\n\n\nBD runtime {:1.2f} seconds'.
                     format(runtimes['bd_mvmm']))

    results['bd_mvmm'] = add_gs_results(pd.DataFrame(),
                                        model=models['bd_mvmm'],
                                        model_name='bd_mvmm',
                                        dataset='full', view='both',
                                        **kws)

####################
# Spectral penalty #
####################

if 'sp_mvmm' not in to_exclude:

    start_time = time()
    models['sp_mvmm'].fit(X_tr)
    runtimes['sp_mvmm'] = time() - start_time
    res_writer.write('\n\n\nSpectPen runtime {:1.2f} seconds'.
                     format(runtimes['sp_mvmm']))

    results['sp_mvmm'] = add_gs_results(pd.DataFrame(),
                                        model=models['sp_mvmm'],
                                        model_name='sp_mvmm',
                                        dataset='full', view='both',
                                        **kws)


#################
# Oracle model #
#################


results['oracle'] = add_gs_results(pd.DataFrame(),
                                   model=models['oracle'],
                                   model_name='oracle',
                                   run_biptsp_on_full=False,  # or true?
                                   dataset='full', view='both',
                                   **kws)
##################
# classification #
##################

clf_results = clf_fit_and_score(LinearDiscriminantAnalysis(),
                                X_tr=np.hstack(X_tr),
                                y_tr=view_labs_to_overall(Y_tr),
                                X_tst=np.hstack(X_tst),
                                y_tst=view_labs_to_overall(Y_tst))


################
# save results #
################

dump({'results': results, 'models': models, 'runtimes': runtimes,
      'data': {'X_tr': X_tr, 'Y_tr': Y_tr, 'X_tst': X_tst, 'Y_tst': Y_tst,
               'Pi_true': Pi_true}},
     os.path.join(save_dir, 'results_data'))

####################
# display results  #
####################

# measures = ['test_overall_ars', 'test_community_ars']
# to_print = pd.DataFrame(index=results.keys(), columns=measures)
# for model_name in results.keys():
#     for measure in measures:

#         if model_name in ['sp_mvmm', 'log_pen_mvmm']:
#             best_idx = results[model_name]['best_tuning_idx'].iloc[0]
#             res = results[model_name].\
#                 query("tune_idx == @best_idx").reset_index()

#             res_writer.write('Best {} {}: {}'.
#                              format(model_name,
#                                     measure,
#                                     max(results[model_name][measure])))
#         else:
#             res = results[model_name]

#         if measure in res.columns:
#             val = res[measure][0]
#             to_print.loc[model_name, measure] = val

###################
# process results #
###################

results_model_sel = {}
results_at_truth = {}

model_sel_measures = ['bic', 'aic']

for model_name in results.keys():

    if model_name not in ['sp_mvmm', 'log_pen_mvmm']:
        continue

    results_model_sel[model_name] = {}

    df = results[model_name]
    df = add_model_selection_by_measures(df)

    for sel_metric in model_sel_measures:
        tune_idx = df['{}_best_tune_idx'.format(sel_metric)].values[0]
        results_model_sel[model_name][sel_metric] = \
            df.query("tune_idx == @tune_idx")
        assert results_model_sel[model_name][sel_metric].shape[0] == 1

    # # model selection
    # bic_tune_idx = df['bic_best_tune_idx'].values[0]
    # results_bic_sel[model_name] = df.query("tune_idx == @bic_tune_idx")
    # assert results_bic_sel[model_name].shape[0] == 1

    # aic_tune_idx = df['aic_best_tune_idx'].values[0]
    # results_aic_sel[model_name] = df.query("tune_idx == @aic_tune_idx")
    # assert results_aic_sel[model_name].shape[0] == 1

    # model_sel_tune_idx = df['best_tuning_idx'].values[0]
    # results_sel[model_name] = df.query("tune_idx == @model_sel_tune_idx")
    # assert results_sel[model_name].shape[0] == 1

    # at truth
    if model_name == 'log_pen_mvmm':
        group_var = 'n_comp_est'
        true_val = n_comp_true
    else:
        group_var = 'n_blocks_est'
        true_val = n_blocks_true

    results_at_truth[model_name] = \
        get_best_expers_at_truthish(df,
                                    group_var=group_var,
                                    true_val=true_val)

    assert results_at_truth[model_name].shape[0] == 1

to_print = pd.DataFrame()
measures = ['test_overall_ars', 'test_community_ars',
            'n_blocks_est', 'n_comp_est']
for model_name, measure in product(results.keys(), measures):
    if measure not in results[model_name].columns:
        continue

    if model_name in ['log_pen_mvmm', 'sp_mvmm']:

        # model selection results
        for sel_metric in model_sel_measures:
            # val = results_bic_sel[model_name][measure].values[0]
            # to_print.loc[model_name + '__at_bic_sel', measure] = val
            val = results_model_sel[model_name][sel_metric][measure].values[0]
            to_print.loc['{}__at_{}_sel'.format(model_name, sel_metric),
                         measure] = val

        # at truth results
        val = results_at_truth[model_name][measure].values[0]
        to_print.loc[model_name + '__at_truth', measure] = val

        # best possible results
        if measure in ['test_overall_ars', 'test_community_ars']:
            best_idx = results[model_name][measure].idxmax()
            best_df = results[model_name].loc[best_idx]
            to_print.loc[model_name + '__best_possible__' + measure, :] = \
                best_df[measures]

    else:
        assert results[model_name][measure].shape[0] == 1
        to_print.loc[model_name,
                     measure] = results[model_name][measure].values[0]

to_print = to_print[measures]
to_print.to_csv(os.path.join(save_dir, 'results.csv'))

res_writer.write('\n\n\n RESULTS\n\n\n')
res_writer.write(to_print)


pd.Series(runtimes).to_csv(os.path.join(save_dir, 'runtimes.csv'))
res_writer.write('Clf accuracy {}'.format(clf_results['tst']['acc']))

######################
# additional results #
######################

n_blocks_true = community_summary(Pi_true)[0]['n_communities']
n_comp_true = (Pi_true > 0).sum()

###########
# Pi true #
###########

plt.figure()
sns.heatmap(Pi_true, cmap='Blues', square=True, cbar=True, vmin=0)
plt.xlabel('View 1 clusters')
plt.ylabel('View 2 clusters')
plt.title("Pi True")
save_fig(os.path.join(save_dir, 'Pi_true.png'))

if 'bd_mvmm' not in to_exclude:

    #########################
    # Block diagonal Pi Viz #
    #########################
    bd_mvmm = models['bd_mvmm'].final_

    print('\n\n\n BD summary\n\n\n')
    summarize_bd(D=bd_mvmm.bd_weights_,
                 n_blocks=n_blocks_true,
                 zero_thresh=zero_thresh)
    plt.title('BD estimate')
    save_fig(os.path.join(save_dir, 'Pi_est_bd_mvmm.png'))

    y_pred = bd_mvmm.predict(X_tr)
    print(cluster_report(view_labs_to_overall(Y_tr), y_pred))


# if 'sp_mvmm' not in to_exclude:
#

###########
# log pen #
###########

if 'log_pen_mvmm' in results.keys():
    log_pen_mvmm_df = results['log_pen_mvmm']
    vals, param_name = extract_tuning_param_vals(log_pen_mvmm_df)
    param_name = 'tune__' + param_name
    log_pen_mvmm_df[param_name] = vals

    best_tune_idx = log_pen_mvmm_df['best_tuning_idx'][0]
    bic_best_exper = log_pen_mvmm_df.query("tune_idx == @best_tune_idx")
    os.makedirs(os.path.join(save_dir, 'log_pen_mvmm'), exist_ok=True)

    ################################
    # log pen tune grid vs measure #
    ################################
    for measure in ['test_community_ars', 'test_overall_ars', 'bic',
                    'n_blocks_est', 'n_comp_est']:

        plt.figure(figsize=(5, 5))
        plt.plot(log_pen_mvmm_df[param_name],
                 log_pen_mvmm_df[measure], marker='.')
        plt.xlabel(param_name)
        plt.ylabel(measure)

        plt.axhline(bic_best_exper[measure].item(), color='red',
                    label='BIC selected')

        if measure == 'n_comp_est':
            plt.axhline(n_comp_true, color='black',
                        label='true n comp {}'.format(n_comp_true))
            plt.axvline(bic_best_exper[param_name].item(),
                        label='BIC selected pen')

        elif measure == 'n_blocks_est':
            plt.axhline(n_blocks_true, color='black',
                        label='true n blocks {}'.format(n_blocks_true))

        plt.legend()

        save_fig(os.path.join(save_dir, 'log_pen_mvmm',
                 '{}_vs_{}.png'.format(param_name, measure)))

    ################
    # loss history #
    ################
    log_pen_mvmm_gs = models['log_pen_mvmm']

    # mvmm = log_pen_mvmm_gs.estimators_[idx]
    mvmm = log_pen_mvmm_gs.best_estimator_
    start_loss_val = mvmm.start_.opt_data_['history']['obs_nll']
    final_loss_val = mvmm.final_.opt_data_['history']['loss_val']

    plot_loss_history(start_loss_val)
    save_fig(os.path.join(save_dir, 'log_pen_mvmm', 'loss_history_start.png'))

    plot_loss_history(final_loss_val)
    save_fig(os.path.join(save_dir, 'log_pen_mvmm', 'loss_history_final.png'))


###########
# SP MVMM #
###########

if 'sp_mvmm' not in to_exclude:
    os.makedirs(os.path.join(save_dir, 'sp_mvmm'), exist_ok=True)

    models['sp_mvmm'].init_fit_data_.\
        to_csv(os.path.join(save_dir, 'sp_mvmm',
                            'init_fit_data_.csv'))

    # make_gifs(models['sp_mvmm'], os.path.join(save_dir, 'sp_mvmm'))

    init_fit_data = models['sp_mvmm'].init_fit_data_
    plot_sp_mvmm_pen_path(init_fit_data)
    save_fig(os.path.join(save_dir, 'sp_mvmm', 'sp_mvmm_pen_path.png'))

    for measure in ['test_community_ars', 'test_overall_ars', 'bic',
                    'n_comp_est']:

        plt.figure(figsize=(5, 5))
        plot_grouped_var_per_n(results['sp_mvmm'],
                               group_var='n_blocks_est',
                               value_var=measure,
                               plot_scatter=True,
                               plot_interval='std',
                               force_x_ticks=False,
                               plot_summary=True, s=200)

        if measure in ['test_community_ars', 'test_overall_ars']:
            value_per_n_samples_hline(results['full_mvmm'],
                                      value_var=measure,
                                      plot_interval=False)

        plt.title('Spectral Penalized MVMM tuning grid')

        save_fig(os.path.join(save_dir, 'sp_mvmm',
                 'n_blocks_vs_{}.png'.format(measure)))


############
# 2d plots #
############

# if both view are one dimensional
if X_tr[0].shape[1] == 1 and X_tr[1].shape[1] == 1:

    two_d_plots_dir = make_and_get_dir(save_dir, 'two_d_plots')

    # plot true data
    plt.figure(figsize=(2 * inches, inches))
    plot_true(view_params, Pi_true, X_tr)
    save_fig(os.path.join(two_d_plots_dir, 'true.png'))

    # plot full MVMM
    plt.figure(figsize=(2 * inches, inches))
    plot_estimator(mvmm=models['full_mvmm'], X_tr=X_tr)
    save_fig(os.path.join(two_d_plots_dir, 'full_mvmm.png'))

    if 'log_pen_mvmm' not in to_exclude:
        # plot log pen mvmm
        plt.figure(figsize=(2 * inches, inches))
        plot_estimator(mvmm=models['log_pen_mvmm'].best_estimator_, X_tr=X_tr)
        save_fig(os.path.join(two_d_plots_dir, 'log_pen_mvmm_best_est.png'))

        # log pen mvm tuning sequence
        log_pen2d_dir = make_and_get_dir(two_d_plots_dir, 'log_pen_mvmm')
        for tune_idx, mvmm in enumerate(models['log_pen_mvmm'].estimators_):
            plt.figure(figsize=(2 * inches, inches))
            plot_estimator(mvmm, X_tr=X_tr)
            save_fig(os.path.join(log_pen2d_dir, 'tune_{}.png'.
                                  format(tune_idx)))

    if 'bd_mvmm' not in to_exclude:
        # bd mvmm
        plt.figure(figsize=(2 * inches, inches))
        plot_estimator(mvmm=models['bd_mvmm'], X_tr=X_tr)
        save_fig(os.path.join(two_d_plots_dir, 'bd_mvmm.png'))

    if 'sp_mvmm' not in to_exclude:

        # sp mvmm best estimator
        plt.figure(figsize=(2 * inches, inches))
        plot_estimator(mvmm=models['sp_mvmm'].best_estimator_, X_tr=X_tr)
        save_fig(os.path.join(two_d_plots_dir, 'sp_mvmm_best_estimator.png'))

        # sp mvmm tuning sequence
        sp_mvmm2d_dir = make_and_get_dir(two_d_plots_dir, 'sp_mvmm')
        for tune_idx, mvmm in enumerate(models['sp_mvmm'].estimators_):
            plt.figure(figsize=(2 * inches, inches))
            plot_estimator(mvmm, X_tr=X_tr)
            save_fig(os.path.join(sp_mvmm2d_dir, 'tune_{}.png'.
                     format(tune_idx)))

# TODO:
# sp_mvmm = models['sp_mvmm'].final_

# print('\n\n\n SP summary\n\n\n')
# summarize_bd(D=bd_mvmm.bd_weights_,
#              n_blocks=n_blocks_true,
#              zero_thresh=zero_thresh)
# plt.title('SP estimate')
# save_fig(os.path.join(save_dir, 'Pi_est_sp_mvmm.png'))

# y_pred = sp_mvmm.predict(X_tr)
# print(cluster_report(view_labs_to_overall(Y_tr), y_pred))


# ###################
# # Visualize setup #
# ###################
# print('n_comp true', n_comp_true)
# print('n_blocks true', n_blocks_true)

# summary, Pi_comm = community_summary(Pi_true)
# print('found {} communities of sizes {}'.format(summary['n_communities'],
#       summary['comm_shapes']))


# # plt.figure()
# # pi_empir = get_empirical_pi(Y_tr, Pi_true.shape, scale='prob')
# # sns.heatmap(pi_empir, cmap='Blues', square=True, cbar=True, vmin=0)
# # plt.title('empirical from Y obs')


# ########################
# # optimization history #
# ########################
# loss_val = bd_mvmm.final_.opt_data_['adpt_opt_data']['history']['loss_val']
# obs_nll = bd_mvmm.final_.opt_data_['adpt_opt_data']['history']['obs_nll']
# eval_sum = bd_mvmm.final_.opt_data_['adpt_opt_data']['history']['raw_eval_sum']


# plt.figure(figsize=(16, 4))

# plt.subplot(1, 3, 1)
# plt.plot(loss_val, marker='.')
# plt.ylabel('loss value')
# plt.xlabel('EM step')
# plt.title('Adaptive Lap pen phase')

# plt.subplot(1, 3, 2)
# plt.plot(obs_nll, marker='.')
# plt.ylabel('Obs NLL')
# plt.xlabel('EM step')

# plt.subplot(1, 3, 3)
# plt.plot(eval_sum, marker='.')
# plt.ylabel('eval sum')
# plt.xlabel('EM step')


# plt.figure()
# if bd_mvmm.final_.opt_data_['ft_opt_data'] is not None:
#     loss_val = bd_mvmm.final_.opt_data_['ft_opt_data']['history']['loss_val']
#     plt.plot(loss_val, marker='.')
#     plt.ylabel('loss value')
#     plt.xlabel('EM step')
#     plt.title('Fixed zeros phase')
