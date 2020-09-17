import sys
import os
from itertools import product
import argparse
import numpy as np
from copy import deepcopy

from mvmm_sim.simulation.submit.bayes import bayes_parser
from mvmm_sim.simulation.from_args import add_parsers, \
    general_opt_parser, base_gmm_parser, bd_mvmm_parser, \
    log_pen_mvmm_parser


parser = argparse.ArgumentParser(description='Submits MVMMs for a range'
                                             'of number of view components')

parser.add_argument('--mini', action='store_true', default=False,
                    help='Run a mini simulation for debugging.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

parser.add_argument('--fpaths', nargs='+',
                    help='Paths to data sets.')

parser.add_argument('--metaseed', type=int, default=89921,
                    help='Seed that sets the seeds.')

parser.add_argument('--n_jobs', default=None,
                    help='n_jobs for grid search.')

parser.add_argument('--min_n_view_comps', nargs='+', type=int,
                    help='Minimum number of components for each dataset.')

parser.add_argument('--max_n_view_comps', nargs='+', type=int,
                    help='Maximum number of components for each dataset.')

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

n_views = len(args.fpaths)
if args.mini:
    args.min_n_view_comps = [2 for v in range(n_views)]
    args.max_n_view_comps = [2 for v in range(n_views)]


n_comp_range = [range(args.min_n_view_comps[v], args.max_n_view_comps[v] + 1)
                for v in range(n_views)]

for n_view_components in product(*n_comp_range):
    print("submitting", n_view_components)

    fpath = os.path.join(os.getcwd(), 'mvmm_fit_models.py')

    # process node to string
    argv = deepcopy(sys.argv[1:])
    if '--node' in argv:
        idx = np.where(np.array(argv) == '--node')[0].item()
        argv[idx + 1] = "'{}'".format(argv[idx + 1])

    # remove min_n_view_comps/max_n_view_comps from args
    idxs2remove = []
    for i, a in enumerate(argv):
        if a in ['--min_n_view_comps', '--max_n_view_comps']:
            for idx in range(i, i + n_views + 1):
                idxs2remove.append(idx)
    argv = np.delete(argv, idxs2remove)

    new_args = " ".join(argv)

    # add view components to args
    new_args += ' --n_view_comps'
    for nc in n_view_components:
        new_args += ' {}'.format(nc)

    command = "python {} {}".format(fpath, new_args)
    os.system(command)
    # print(command)
