import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from joblib import dump
import os
from os.path import join
from sklearn.utils import check_random_state
import argparse

from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.simulation.models_from_config import \
    get_single_view_models
from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.utils import sample_seed
from mvmm_sim.simulation.utils import make_and_get_dir
# from mvmm.simulation.opt_viz import plot_loss_history
# from mvmm.viz_utils import set_xaxis_int_ticks
from mvmm_sim.single_view.gaussian_mixture import default_cov_regularization

from mvmm_sim.data_analysis.utils import load_data
from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.simulation.utils import format_mini_experiment
from mvmm_sim.simulation.from_args import add_parsers, \
    general_opt_parser, base_gmm_parser, \
    gmm_from_args

parser = argparse.ArgumentParser(description='Fit single view mixture models.')

parser.add_argument('--mini', action='store_true', default=False,
                    help='Run a mini simulation for debugging.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

parser.add_argument('--metaseed', type=int, default=89921,
                    help='Seed that sets the seeds.')

parser.add_argument('--n_jobs', default=None,
                    help='n_jobs for grid search.')

parser.add_argument('--fpaths', nargs='+',
                    help='Paths to data sets.')

parser.add_argument('--min_ncomps', nargs='+', type=int,
                    help='Minimum number of components for each dataset.')

parser.add_argument('--max_ncomps', nargs='+', type=int,
                    help='Maximum number of components for each dataset.')

parser.add_argument('--min_ncomp_cat', default=1, type=int,
                    help='Min number of components for concatenated data.')

parser.add_argument('--max_ncomp_cat', default=20, type=int,
                    help='Maximum number of components for concatenated data.')

parser.add_argument('--exclude_cat_gmm', action='store_true', default=False,
                    help='Dont run cat gmm.')

parser = add_parsers(parser,
                     to_add=[general_opt_parser, base_gmm_parser])


parser = bayes_parser(parser)
args = parser.parse_args()
bayes_submit(args)
args = format_mini_experiment(args)


results_dir = args.results_dir
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting', 'single_view')

res_writer = ResultsWriter(join(log_dir, 'single_view_fitting.txt'),
                           delete_if_exists=True)

res_writer.write(args)

run_start_time = time()


n_views = len(args.fpaths)

#############
# load data #
#############

view_data, dataset_names, sample_names, feat_names = \
    load_data(*args.fpaths)

for v in range(n_views):

    res_writer.write('{} (view {}) shape : {}'.
                     format(dataset_names[v], v, view_data[v].shape))

################
# setup models #
################

cat_gmm_n_comp_seq = np.arange(args.min_ncomp_cat, args.max_ncomp_cat + 1)
# view_gmm_n_comp_seq = [np.arange(args.min_ncomp_cat, args.max_ncomp_v0 + 1),
#                        np.arange(args.min_ncomp_cat, args.max_ncomp_v1 + 1)]

view_gmm_n_comp_seq = [np.arange(args.min_ncomps[v], args.max_ncomps[v] + 1)
                       for v in range(n_views)]


rng = check_random_state(args.metaseed)

base_gmm_config = gmm_from_args(args)

view_gmm_config = deepcopy(base_gmm_config)
view_gmm_config['random_state'] = sample_seed(rng)

cat_gmm_config = deepcopy(base_gmm_config)
cat_gmm_config['random_state'] = sample_seed(rng)

if args.n_jobs is not None:
    n_jobs_tune = int(args.n_jobs)
else:
    n_jobs_tune = None


single_view_config = {'cat_gmm_config': cat_gmm_config,
                      'view_gmm_config': view_gmm_config,
                      'cat_n_comp': cat_gmm_n_comp_seq,
                      'view_n_comp': view_gmm_n_comp_seq,
                      'n_jobs_tune': n_jobs_tune,
                      'select_metric': args.select_metric,
                      'metrics2compute': ['aic', 'bic', 'silhouette',
                                          'calinski_harabasz',
                                          'davies_bouldin', 'dunn']}

single_view_models = get_single_view_models(**single_view_config)


if args.mini:
    cat_gmm_n_comp_seq = np.arange(1, 3)
    view_gmm_n_comp_seq = [np.arange(1, 3) for v in range(n_views)]

#############################
# covariance regularization #
#############################

reg_covar = {}

# set cov reg for each view
for v in range(n_views):
    # set covariance regularization
    reg = default_cov_regularization(X=view_data[v], mult=args.reg_covar_mult)
    single_view_models['view_gmms'][v].base_estimator.set_params(reg_covar=reg)

    # print and save
    reg_covar[dataset_names[v]] = reg
    res_writer.write("\nCovarinace regularization for {} is {}".
                     format(dataset_names[v], reg))
    stds = view_data[v].std(axis=0)
    res_writer.write("Smallest variance: {}".format(stds.min() ** 2))
    res_writer.write("Largest variance: {}".format(stds.max() ** 2))


# for cat GMM
reg = default_cov_regularization(X=np.hstack(view_data),
                                 mult=args.reg_covar_mult)
single_view_models['cat_gmm'].base_estimator.set_params(reg_covar=reg)
reg_covar['cat_gmm'] = reg

###############
# fit  models #
###############

runtimes = {}

if not args.exclude_cat_gmm:

    start_time = time()
    single_view_models['cat_gmm'].fit(np.hstack(view_data))

    runtimes['cat_gmm'] = time() - start_time
    res_writer.write('fitting cat gmm took {:1.2f} seconds'.
                     format(runtimes['cat_gmm']))

    # save and delete for the sake of memory
    dump(single_view_models['cat_gmm'], join(fitting_dir, 'fitted_cat_gmms'))
    single_view_models['cat_gmm'] = None


else:
    del single_view_models['cat_gmm']


for v in range(n_views):

    start_time = time()
    single_view_models['view_gmms'][v].fit(view_data[v])

    runtimes['view_' + str(v)] = time() - start_time
    res_writer.write('fitting view {} gmm took {:1.2f} seconds'.
                     format(v, runtimes['view_' + str(v)]))

    # save and delete for the sake of memory
    dump(single_view_models['view_gmms'][v],
         join(fitting_dir, 'fitted_view_{}_{}'.format(v, dataset_names[v])))

    single_view_models['view_gmms'][v]

dump({'runtimes': runtimes,  # 'models': single_view_models,
      'cat_gmm_n_comp_seq': cat_gmm_n_comp_seq,
      'view_gmm_n_comp_seq': view_gmm_n_comp_seq,
      'args': args,
      'reg_covar': reg_covar,
      'dataset_names': dataset_names,
      'sample_names': sample_names},
     os.path.join(fitting_dir, 'single_view_fit_metadata'))

res_writer.write('Fitting models took {:1.2f} seconds'.
                 format(time() - run_start_time))


##############################
# run model selection script #
##############################
py_fpath = os.path.join(Paths().sim_scripts_dir, 'data_analysis',
                        'single_view_model_selection.py')
py_command = 'python {} --results_dir {}'.format(py_fpath, args.results_dir)
os.system(py_command)
