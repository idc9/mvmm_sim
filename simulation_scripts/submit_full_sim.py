#!/usr/bin/python
"""
This script setups up and runs one simulation e.g. from Section 5 of (Carmichael, 2020). This script creates the arguments for the simulation then calls functions from mvmm_sim.simulations.run_sim which actually run the simulation. This script allows you to parallelize each individual experiment (a fixed sample size for one Monte-Carlo seed) across different cluster nodes.
"""
import numpy as np
from time import time
from tqdm import tqdm
from joblib import dump
from copy import deepcopy
import os
import argparse
import warnings
from sklearn.utils import check_random_state

# from mvmm.multi_view.SpectralPenSearchMVMM import linspace_zero_to
from mvmm.utils import get_seeds, sample_seed

from mvmm_sim.simulation.utils import format_mini_experiment
from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser, \
    get_bayes_command
from mvmm_sim.simulation.run_sim import run_sim_from_configs
from mvmm_sim.simulation.Paths import Paths, which_computer
from mvmm_sim.simulation.sim_naming import get_new_name
from mvmm_sim.simulation.from_args import add_parsers, \
    grid_means_data_dist_parser, rand_means_data_dist_parser, pi_parser, \
    general_opt_parser, base_gmm_parser, bd_mvmm_parser, \
    log_pen_mvmm_parser, pi_from_args, \
    gmm_from_args, full_mvmm_from_args, \
    ts_log_pen_mvmm_from_args, ts_bd_mvmm_from_args, \
    spect_pen_parser, spect_pen_from_args, \
    grid_means_data_dist_from_args, rand_means_data_dist_from_args

parser = argparse.ArgumentParser(description='Simulation with sparse Pi.')
parser.add_argument('--mini', action='store_true', default=False,
                    help='Run a mini simulation for debugging.')

parser.add_argument('--sim_name', default=None,
                    help='Name of the siulation.')

parser.add_argument('--metaseed', type=int, default=89921,
                    help='Seed that sets the seeds.')

parser.add_argument('--submit_sep_sims', action='store_true', default=False,
                    help='Submit each simulation separately on'
                         'different cluster nodes.')

parser.add_argument('--n_jobs', default=None,
                    help='n_jobs for grid search.')

parser.add_argument('--n_mc_reps', type=int, default=10,
                    help='Number of Monte-Carlo repitition.')

parser.add_argument('--n_sims_to_skip', default=0, type=int,
                    help='Skip the first several simulations.')

parser.add_argument('--n_tr_samples_seq', type=str, default='lin',
                    choices=['exp', 'lin'],
                    help='N training samples sequence.')

parser.add_argument('--fix_cluster_means', action='store_true', default=False,
                    help='Fix cluster means across all MC repitions.')

parser.add_argument('--exclude_bd_mvmm', action='store_true', default=False,
                    help='Do not include block diagonal models.')

# parser.add_argument('--exclude_sp_mvmm', action='store_true', default=False,
#                     help='Do not include spectral penalty models.')

parser.add_argument('--exclude_log_pen_mvmm', action='store_true',
                    default=False,
                    help='Do no include log pen models.')

parser.add_argument('--grid_means', action='store_true', default=False,
                    help='Put cluster means on a grid.'
                         'Otherwise cluster meansare sampled from an'
                         'isotropic gaussian.')


parser = add_parsers(parser,
                     to_add=[pi_parser,
                             grid_means_data_dist_parser,
                             rand_means_data_dist_parser,
                             general_opt_parser, base_gmm_parser,
                             log_pen_mvmm_parser, bd_mvmm_parser,
                             spect_pen_parser])

parser = bayes_parser(parser)
args = parser.parse_args()
args = format_mini_experiment(args)
args.job_name = args.sim_name


# args.submit_sep_sims will parallelize across cluster nodes
# you may have to change the code a little to make this work on your cluster!
if args.submit_sep_sims:

    print("If this below code breaks you may have to modify"
          "some code for your cluster!")


# useful for the cluster Iain used to run the experiment
if not args.submit_sep_sims:
    bayes_submit(args)

rng = check_random_state(args.metaseed)

###########################
# Setup data distribution #
###########################

pi_config = pi_from_args(args)
# clust_param_config = data_dist_from_args(args)
if args.grid_means:
    clust_param_config = grid_means_data_dist_from_args(args)
else:
    clust_param_config = rand_means_data_dist_from_args(args)

################
# Setup models #
################

base_gmm_config = gmm_from_args(args)

view_gmm_config = deepcopy(base_gmm_config)
view_gmm_config['random_state'] = sample_seed(rng)

cat_gmm_config = deepcopy(base_gmm_config)
cat_gmm_config['random_state'] = sample_seed(rng)

full_mvmm_config = full_mvmm_from_args(args)
full_mvmm_config['random_state'] = sample_seed(rng)

log_pen_config = ts_log_pen_mvmm_from_args(args)
log_pen_config['two_stage']['random_state'] = sample_seed(rng)

bd_config = ts_bd_mvmm_from_args(args)
bd_config['two_stage']['random_state'] = sample_seed(rng)

bd_config['final']['history_tracking'] = 1

spect_pen_config = spect_pen_from_args(args)
spect_pen_config['search']['random_state'] = sample_seed(rng)

###################
# Tuning settings #
###################

# controls the grid to search over e.g. plus minuts 15 components from the true
# number of components

gmm_pm = 15
n_blocks_pm = 3

#######################
# Simulation settings #
#######################
# e.g. training  sample sizes, seeds

if args.n_tr_samples_seq == 'exp':  # exponentially increasing i.e. fewer
    n_samples_tr_seq = [200, 400, 800, 1600, 2400, 3200]

if args.n_tr_samples_seq == 'lin':
    n_samples_tr_seq = [200, 500, 1000, 1500, 2000,
                        2500, 3000, 3500, 4000]

n_samples_tst = 5000

mc_metaseed = sample_seed(rng)

if args.fix_cluster_means:
    gmm_params_seed = sample_seed(rng)

# parallelization
if args.n_jobs is not None:
    n_jobs_tune = int(args.n_jobs)
else:
    n_jobs_tune = None

if n_jobs_tune is not None:
    full_mvmm_config['n_jobs'] = n_jobs_tune

########
# mini #
########
# If we want to run a mini simulation for debuggin purposes
if args.mini:

    args.n_mc_reps = 2
    n_samples_tr_seq = [100, 150]

    n_samples_tst = 500
    gmm_plus_minus = 2

    log_pen_config['mult_values'] = [.1, .9]
    n_blocks_pm = 1

    bd_config['final']['max_n_steps'] = 5
    log_pen_config['final']['max_n_steps'] = 5

    spect_pen_config['wbd']['max_n_steps'] = 5
    spect_pen_config['search']['n_pen_seq'] = 5


##################
# run simulation #
##################

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

# don't run some models
to_exclude = []
if args.exclude_bd_mvmm:
    to_exclude.append('bd_mvmm')
if args.exclude_log_pen_mvmm:
    to_exclude.append('log_pen_mvmm')
to_exclude.append('sp_mvmm')  # just don't include this

# every simulation needs a name!
if args.sim_name is not None:
    folder_name = args.sim_name
else:
    # randomly generate a name if you were too lazy to provide one
    folder_name = get_new_name(Paths().out_data_dir)


# where to save simulation data
save_dir = os.path.join(Paths().out_data_dir, folder_name)
os.makedirs(save_dir, exist_ok=True)

print(args)

# setup monte-carlo seeds
mc_seeds = get_seeds(random_state=mc_metaseed, n_seeds=args.n_mc_reps)

start_time = time()
sim_idx = 0  # simulations are 1 indexed not zero indexed
for mc_index in tqdm(range(args.n_mc_reps)):
    for n_samples_tr in n_samples_tr_seq:

        sim_idx += 1
        print('running n_samples_tr = {}, mc_index = {}, sim_idx = {}'.
              format(n_samples_tr, mc_index, sim_idx))

        if args.fix_cluster_means:
            data_seed = mc_seeds[mc_index]
        else:
            # get seeds for GMM parameters and observed data
            gmm_params_seed, data_seed = \
                get_seeds(random_state=mc_seeds[mc_index], n_seeds=2)

        clust_param_config['random_state'] = gmm_params_seed

        config = {'clust_param_config': clust_param_config,
                  'grid_means': args.grid_means,
                  'pi_dist': pi_config['dist'],
                  'pi_config': pi_config['config'],
                  'n_samples_tr': n_samples_tr,
                  'n_samples_tst': n_samples_tst,
                  'single_view_config': single_view_config,
                  'mvmm_config': mvmm_config,
                  'gmm_pm': gmm_pm,
                  'n_blocks_pm': n_blocks_pm,
                  'reg_covar_mult': args.reg_covar_mult,
                  'to_exclude': to_exclude,
                  'args': args,
                  }

        kwargs = deepcopy(config)
        kwargs['n_samples_tr'] = n_samples_tr
        kwargs['mc_index'] = mc_index
        kwargs['data_seed'] = data_seed

        kwargs['save_fpath'] = os.path.join(save_dir,
                                            'sim_res_{}'.format(sim_idx))

        # possibly skip first several simulations
        if sim_idx <= args.n_sims_to_skip:
            print("skipping simulation {}".format(sim_idx))
            continue

        # some of this code assumes you are working on the cluster
        # Iain used to run the experiments.

        if args.submit_sep_sims:
            # parallelize the individual experiments across different nodes
            # You will have to modify some of this code if you want to use
            # your own cluster ;(

            # save configuration arguments
            config_fpath = os.path.join(save_dir, 'config_{}'.format(sim_idx))
            dump(kwargs, config_fpath)

            # setup python parameters
            py_fpath = os.path.join(Paths().sim_scripts_dir,
                                    'run_a_sim.py')
            py_command = '{} --config_fpath {}'.\
                format(py_fpath, config_fpath)

            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            if which_computer() == 'bayes':
                submit_command = get_bayes_command(py_command, args)
                print('sumitting:')
                print(submit_command)
                print()

                # Actually submit the simulation!!!!
                os.system(submit_command)
            else:
                # print('running python', py_command)
                # Actually submit the simulation!!!!
                os.system('python ' + py_command)
        else:
            # Just run all the simulations sequentially

            # Actually submit the simulation!!!!
            run_sim_from_configs(**kwargs)


#######################
# simulation metadata #
#######################
runtime = time() - start_time
print('Simulation took {:1.1f} seconds'.format(runtime))

sim_metadata = {'n_samples_tr_seq': n_samples_tr_seq,
                'n_mc_reps': args.n_mc_reps,
                'n_sims': mc_index + 1,
                'metaseed': args.metaseed,
                'mc_metaseed': mc_metaseed,
                'runtime': runtime}

dump(sim_metadata, os.path.join(save_dir, 'sim_metadata'))

##################################
# run results aggregation script #
##################################
if not args.submit_sep_sims:
    py_fpath = os.path.join(Paths().sim_scripts_dir, 'agg_sim_results.py')
    py_command = 'python {} --sim_name {}'.format(py_fpath, args.sim_name)
    # if delete_files:
    #     py_command += ' --delete_files'
    print(py_command)
    os.system(py_command)
