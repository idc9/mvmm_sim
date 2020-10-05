import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from joblib import dump
import os
from sklearn.utils import check_random_state
import argparse

from mvmm.single_view.gaussian_mixture import default_cov_regularization

from mvmm_sim.data_analysis.utils import load_data
from mvmm_sim.simulation.models_from_config import get_mvmms
from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.utils import sample_seed
from mvmm_sim.simulation.utils import make_and_get_dir
# from mvmm.simulation.opt_viz import plot_loss_history
# from mvmm.viz_utils import set_xaxis_int_ticks
from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.simulation.utils import format_mini_experiment
from mvmm_sim.simulation.from_args import add_parsers, \
    general_opt_parser, base_gmm_parser, bd_mvmm_parser, \
    gmm_from_args, full_mvmm_from_args, \
    ts_bd_mvmm_from_args, ts_log_pen_mvmm_from_args, log_pen_mvmm_parser


parser = argparse.ArgumentParser(description='Fits MVMMs for a given'
                                             'number of view components.')

parser.add_argument('--mini', action='store_true', default=False,
                    help='Run a mini simulation for debugging.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')
# parser.add_argument('--sim_name', default=None,
#                     help='Name of the siulation.')

parser.add_argument('--metaseed', type=int, default=89921,
                    help='Seed that sets the seeds.')

parser.add_argument('--n_jobs', default=None,
                    help='n_jobs for grid search.')

parser.add_argument('--fpaths', nargs='+',
                    help='Paths to data sets.')

parser.add_argument('--n_view_comps', nargs='+', type=int,
                    help='Number of components fore each view.')

parser.add_argument('--exclude_bd_mvmm', action='store_true',
                    default=False,
                    help='Do not include block diagonal models.')

parser.add_argument('--exclude_log_pen_mvmm', action='store_true',
                    default=False,
                    help='Do no include log pen models.')

parser = add_parsers(parser,
                     to_add=[general_opt_parser, base_gmm_parser,
                             bd_mvmm_parser, log_pen_mvmm_parser])


parser = bayes_parser(parser)
args = parser.parse_args()
bayes_submit(args)
args = format_mini_experiment(args)

# stub = 'mvmm_fitting_{}_{}'.format(args.n_comp_v0, args.n_comp_v1)
stub = 'mvmm_fitting'
for nc in args.n_view_comps:
    stub += '_{}'.format(nc)

results_dir = args.results_dir
# results_dir = make_and_get_dir(args.top_dir, args.sim_name)
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting')
model_sel_dir = make_and_get_dir(results_dir, 'model_selection')
opt_diag_dir = make_and_get_dir(results_dir, 'opt_diagnostics')

res_writer = ResultsWriter(os.path.join(log_dir, '{}.txt'.format(stub)),
                           delete_if_exists=True)

res_writer.write(args)

run_start_time = time()

to_exclude = []
# if args.exclude_sp_mvmm:
#     to_exclude.append('sp_mvmm')
if args.exclude_bd_mvmm:
    to_exclude.append('bd_mvmm')
if args.exclude_log_pen_mvmm:
    to_exclude.append('log_pen_mvmm')

#############
# load data #
#############
n_views = len(args.fpaths)

view_data, dataset_names, sample_names, feat_names = \
    load_data(*args.fpaths)


for v in range(n_views):
    res_writer.write('{} (view {}) shape : {}'.
                     format(dataset_names[v], v, view_data[v].shape))

################
# setup models #
################


rng = check_random_state(args.metaseed)

base_gmm_config = gmm_from_args(args)

view_gmm_config = deepcopy(base_gmm_config)
view_gmm_config['random_state'] = sample_seed(rng)

# cat_gmm_config = deepcopy(base_gmm_config)
# cat_gmm_config['random_state'] = sample_seed(rng)

full_mvmm_config = full_mvmm_from_args(args)
full_mvmm_config['random_state'] = sample_seed(rng)

log_pen_config = ts_log_pen_mvmm_from_args(args)
log_pen_config['two_stage']['random_state'] = sample_seed(rng)

bd_config = ts_bd_mvmm_from_args(args)
bd_config['two_stage']['random_state'] = sample_seed(rng)
bd_config['final']['history_tracking'] = 1

# spect_pen_config = spect_pen_from_args(args)
# spect_pen_config['search']['random_state'] = sample_seed(rng)

if args.n_jobs is not None:
    n_jobs_tune = int(args.n_jobs)
else:
    n_jobs_tune = None

if args.n_blocks_seq == 'default':
    n_blocks = 'default'

else:
    max_n_blocks = int(args.n_blocks_seq)
    max_n_blocks = min(min(args.n_view_comps), max_n_blocks)
    n_blocks = np.arange(1, max_n_blocks + 1)

mvmm_view_config = {'base_gmm_config': base_gmm_config,
                    'full_mvmm_config': full_mvmm_config,
                    'log_pen_config': log_pen_config,
                    'bd_config': bd_config,
                    # 'spect_pen_config': spect_pen_config,
                    'n_blocks': n_blocks,
                    'sp_by_block': True,
                    'select_metric': args.select_metric,
                    'metrics2compute': ['aic', 'bic', 'silhouette',
                                        'calinski_harabasz',
                                        'davies_bouldin', 'dunn'],
                    'n_jobs_tune': n_jobs_tune}

models = get_mvmms(n_view_components=args.n_view_comps,
                   **mvmm_view_config)


#############################
# covariance regularization #
#############################

reg_covar = {}

# set cov reg for each view
for v in range(n_views):
    reg = default_cov_regularization(X=view_data[v], mult=args.reg_covar_mult)

    # models['full_mvmm'].base_view_models[v].set_params(reg_covar=reg)

    models['bd_mvmm'].base_estimator.base_start.base_view_models[v].\
        set_params(reg_covar=reg)
    models['bd_mvmm'].base_estimator.base_final.base_view_models[v].\
        set_params(reg_covar=reg)

    models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
        set_params(reg_covar=reg)
    models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
        set_params(reg_covar=reg)

    # print and save
    reg_covar[dataset_names[v]] = reg
    res_writer.write("\nCovarinace regularization for {} is {}".
                     format(dataset_names[v], reg))
    stds = view_data[v].std(axis=0)
    res_writer.write("Smallest variance: {}".format(stds.min() ** 2))
    res_writer.write("Largest variance: {}".format(stds.max() ** 2))


#########################
# fit multi-view models #
#########################

runtimes = {}

del models['full_mvmm']
# fit model
# start_time = time()
# models['full_mvmm'].fit(view_data)

# full_runtime = time() - start_time
# res_writer.write('fitting full mvmm took {:1.2f} seconds'.
#                  format(full_runtime))

# runtimes['full_mvmm'] = full_runtime

# full_model_sel_scores = \
#     unsupervised_cluster_scores(X=view_data, estimator=models['full_mvmm'],
#                                 measures=mvmm_view_config['metrics2compute'])

if 'bd_mvmm' not in to_exclude:
    # TODO: do we want this
    # models['bd_mvmm'].backend = 'multiprocessing'
    models['bd_mvmm'].verbose = 50

    start_time = time()
    models['bd_mvmm'].fit(view_data)

    bd_runtime = time() - start_time
    res_writer.write('fitting bd mvmm took {:1.2f} seconds'.
                     format(bd_runtime))

    runtimes['bd_mvmm'] = bd_runtime

else:
    del models['bd_mvmm']

if 'log_pen_mvmm' not in to_exclude:
    start_time = time()
    models['log_pen_mvmm'].fit(view_data)
    log_pen_mvmm_runtime = time() - start_time
    res_writer.write('fitting log pen mvmm took {:1.2f} seconds'.
                     format(log_pen_mvmm_runtime))

    runtimes['log_pen_mvmm'] = log_pen_mvmm_runtime
else:
    del models['log_pen_mvmm']


dump({'models': models,
      # 'full_model_sel_scores': full_model_sel_scores,
      'runtimes': runtimes,
      'n_view_components': args.n_view_comps,
      'args': args,
      'reg_covar': reg_covar,
      'dataset_names': dataset_names,
      'sample_names': sample_names},
     os.path.join(fitting_dir, stub))

res_writer.write('Fitting models took {:1.2f} seconds'.
                 format(time() - run_start_time))


##############################
# run model selection script #
##############################
# py_fpath = os.path.join(Paths().sim_scripts_dir, 'data_analysis',
#                         'mvmm_model_selection.py')
# py_command = 'python {} --results_dir {}'.format(py_fpath, args.results_dir)
# os.system(py_command)
