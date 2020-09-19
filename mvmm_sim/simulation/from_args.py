
def rand_means_data_dist_parser(parser):
    parser.add_argument('--clust_mean_std_v1', type=float, default=1.0,
                        help='How far apart the cluster means are'
                             ' for the first view.')

    parser.add_argument('--clust_mean_std_v2', type=float, default=1.0,
                        help='How far apart the cluster means are'
                             ' for the second view.')

    return parser


def rand_means_data_dist_from_args(args):

    custom_view_kws = [{'clust_mean_std': args.clust_mean_std_v1},
                       {'clust_mean_std': args.clust_mean_std_v2}]

    return {'n_features': [args.n_feats, args.n_feats],
            'custom_view_kws': custom_view_kws,
            'cluster_std': 1.0}


def grid_means_data_dist_parser(parser):
    parser.add_argument('--cluster_std_v1', type=float, default=1.0,
                        help='How noisey the clusters are in the first view.')

    parser.add_argument('--cluster_std_v2', type=float, default=1.0,
                        help='How noisey the clusters are in the second view.')

    parser.add_argument('--n_feats', type=int, default=10,
                        help='Number of features in each view')

    return parser


def grid_means_data_dist_from_args(args):

    custom_view_kws = [{'cluster_std': args.cluster_std_v1},
                       {'cluster_std': args.cluster_std_v2}]

    return {'n_features': [args.n_feats, args.n_feats],
            'custom_view_kws': custom_view_kws}


def pi_parser(parser):
    parser.add_argument('--pi_name', default='motivating_ex',
                        help='Name of the Pi distribution.')

    return parser


def pi_from_args(args):

    config = {}

    if args.pi_name == 'sparse_pi':
        config['dist'] = 'sparse'
        config['config'] = {'n_rows_base': 10,
                            'n_cols_base': 10,
                            'density': .25,
                            'random_state': 849}

    elif 'diagonal' in args.pi_name:

        config['dist'] = 'block_diag'
        n_blocks = int(args.pi_name.split('_')[1])
        block_shapes = [(1, 1)] * n_blocks
        config['config'] = {'block_shapes': block_shapes}

    elif 'indep' in args.pi_name:
        config['dist'] = 'indep'
        n_comp = int(args.pi_name.split('_')[1])
        config['config'] = {'n_comp': n_comp}

    elif 'lollipop' in args.pi_name:
        config['dist'] = 'block_diag'
        _, stick, bulb = args.pi_name.split('_')
        stick = int(stick)
        bulb = int(bulb)

        block_shapes = [(1, 1)] * stick + [(bulb, bulb)]
        block_weights = [1] * stick + [1 / bulb]
        config['config'] = {'block_shapes': block_shapes,
                            'block_weights': block_weights}

    elif 'beads' in args.pi_name:
        config['dist'] = 'block_diag'
        _, size, n_beads = args.pi_name.split('_')

        block_shapes = [(int(size), int(size))] * int(n_beads)
        config['config'] = {'block_shapes': block_shapes}

    elif args.pi_name == 'motivating_ex':
        config['dist'] = 'motivating_ex'
        config['config'] = {}

    else:
        raise ValueError('bad arguent for pi_name, {}'.format(args.pi_name))

    return config


def general_opt_parser(parser):

    parser.add_argument('--n_init', type=int, default=20,
                        help='Number of random initalizations.')

    parser.add_argument('--abs_tol', type=float, default=1e-8,
                        help='Abs tolerance for optimization convergence.')

    parser.add_argument('--max_n_steps', type=int, default=200,
                        help='Maximum number of optimization steps.')

    parser.add_argument('--select_metric', type=str, default='bic',
                        choices=['bic', 'aic'],
                        help='Model selection criterion.')
    return parser


def gmm_parser(parser):
    return parser


def gmm_from_args(args):
    return {'reg_covar': args.reg_covar,
            'covariance_type': args.covariance_type,
            'init_params_method': args.gmm_init_params,
            'n_init': args.n_init,
            'max_n_steps': args.max_n_steps,
            'abs_tol': args.abs_tol}


def base_gmm_parser(parser):
    parser.add_argument('--reg_covar', default=1e-4, type=float,
                        help='Covariance regularization.')

    parser.add_argument('--covariance_type', default='diag', type=str,
                        help='Type of covariance matrix.')

    parser.add_argument('--gmm_init_params', default='kmeans',
                        choices=['rand_pts', 'kmeans'],
                        help='How to initialize GMMs.')

    parser.add_argument('--reg_covar_mult', default=1e-2, type=float,
                        help='Mult value for default covariance'
                             'regularization.')
    return parser


def full_mvmm_from_args(args):

    return {'n_init': args.n_init,
            'max_n_steps': args.max_n_steps,
            'abs_tol': args.abs_tol,
            'init_params_method': 'init',  # fit
            'init_weights_method': 'uniform'}


def two_stage_parser(parser, base=None):

    if base is None:

        start_steps = '--start_max_n_steps'
        final_steps = '--final_max_n_steps'
        start_tol = '--start_abs_tol'
        final_tol = '--final_abs_tol'

        help_start_steps = 'Maximum number of optimization steps'\
                           'for start of two stage  model.'

        help_final_steps = 'Maximum number of optimization steps'\
                           'for final of two stage model.'

        help_start_tol = 'Absolute tolerance for convergence for '\
                         'start of two stage model'

        help_final_tol = 'Absolute tolerance for convergence for '\
                         'final of two stage model'

    else:
        start_steps = '--{}_start_max_n_steps'.format(base)
        final_steps = '--{}_final_max_n_steps'.format(base)
        start_tol = '--{}_start_abs_tol'.format(base)
        final_tol = '--{}_final_abs_tol'.format(base)

        help_start_steps = 'Maximum number of optimization steps'\
                           'for start of two stage {} model.'.format(base)

        help_final_steps = 'Maximum number of optimization steps'\
                           'for final of two stage {} model.'.format(base)

        help_start_tol = 'Absolute tolerance for convergence for '\
                         'start of two stage {} model'.format(base)

        help_final_tol = 'Absolute tolerance for convergence for '\
                         'final of two stage {} model'.format(base)

    parser.add_argument(start_steps, type=int, default=200,
                        help=help_start_steps)

    parser.add_argument(final_steps, type=int, default=200,
                        help=help_final_steps)

    parser.add_argument(start_tol, type=float, default=1e-8,
                        help=help_start_tol)

    parser.add_argument(final_tol, type=float, default=1e-8,
                        help=help_final_tol)

    return parser


def bd_mvmm_parser(parser):
    parser.add_argument('--bd_rel_epsilon', default=1e-2, type=float,
                        help='Epsilon value for block diagonal MVMM.')

    parser.add_argument('--bd_eval_pen_base',
                        default='guess_from_init',
                        help='Eval base penalty. Either a number'
                             ' or guess_from_init.')

    parser.add_argument('--bd_eval_pen_incr', type=float, default=2,
                        help='Eval pen increase.')

    parser.add_argument('--bd_n_pen_tries', type=int, default=30,
                        help='N eval pen tries.')

    parser.add_argument('--bd_init_pen_c', type=float, default=1e-2,
                        help='Constant for eval pen initialization.')

    parser.add_argument('--bd_init_pen_use_bipt_sp',
                        action='store_true', default=False,
                        help='Use spectral bi-partitioning for'
                             'eval pen init guess.')

    parser.add_argument('--bd_init_pen_K',
                        action='store_true', default='default',
                        help='Number of communities to estimate for'
                             'spectral bi-partitioning guess of init'
                             'eval pen.')

    parser.add_argument('--bd_exclude_vdv_constr',
                        action='store_true', default=False,
                        help='Excluded VDV constraint.')

    parser.add_argument('--n_blocks_seq', default='default',
                        help='Blocks tuning sequence for BD MVMM.')

    parser = two_stage_parser(parser, base='bd')

    return parser


def ts_bd_mvmm_from_args(args):

    start_config = {'init_params_method': 'init',
                    'init_weights_method': 'uniform',
                    'n_init': 1,
                    'abs_tol': args.bd_start_abs_tol,
                    'max_n_steps': args.bd_start_max_n_steps}

    if args.bd_eval_pen_base == 'guess_from_init':
        eval_pen_base = args.bd_eval_pen_base
    else:
        eval_pen_base = float(args.bd_eval_pen_base)

    final_config = {'init_params_method': 'user',
                    'init_weights_method': 'user',
                    'n_init': 1,

                    'abs_tol': args.bd_final_abs_tol,
                    'max_n_steps': args.bd_final_max_n_steps,

                    'eval_pen_base': eval_pen_base,
                    'eval_pen_incr': args.bd_eval_pen_incr,
                    'n_pen_tries': args.bd_n_pen_tries,

                    'init_pen_c': args.bd_init_pen_c,
                    'init_pen_use_bipt_sp': args.bd_init_pen_use_bipt_sp,
                    'init_pen_K': args.bd_init_pen_K,

                    'exclude_vdv_constr': args.bd_exclude_vdv_constr}

    return {'start': start_config,
            'final': final_config,
            'two_stage': {'n_init': args.n_init}}


def log_pen_mvmm_parser(parser):

    parser = two_stage_parser(parser, base='log_pen')

    parser.add_argument('--log_pen_n_pen_seq', type=int, default=50,
                        help='Number of penalty values for log'
                             'penalized MVMM penalty.')

    parser.add_argument('--log_pen_seq_spacing', type=str, default='quad',
                        # choices=['lin', 'quad'],
                        help='Spacing for log pen penalty values.')

    return parser


def ts_log_pen_mvmm_from_args(args):

    start_config = {'init_params_method': 'init',
                    'init_weights_method': 'uniform',
                    'n_init': 1,
                    'abs_tol': args.log_pen_start_abs_tol,
                    'max_n_steps': args.log_pen_start_max_n_steps}

    final_config = {'init_params_method': 'user',
                    'init_weights_method': 'user',
                    'n_init': 1,
                    'abs_tol': args.log_pen_final_abs_tol,
                    'max_n_steps': args.log_pen_final_max_n_steps,
                    'delta': 1e-6}

    two_stage_config = {'n_init': args.n_init}

    search_config = {'num': args.log_pen_n_pen_seq,
                     'spacing': args.log_pen_seq_spacing}

    return {'start': start_config,
            'final': final_config,
            'two_stage': two_stage_config,
            'search': search_config}


def spect_pen_parser(parser):

    parser.add_argument('--sp_max_n_steps', default=200,
                        help='Maximum n steps for sp mvmm.')

    parser.add_argument('--sp_n_pen_seq', type=int, default=100,
                        help='Number of penalty values for spectral penalty.')

    # parser.add_argument('--spect_pen_min', type=float, default=.1,
    #                     help='Minimum spectral penalty value.')

    parser.add_argument('--sp_max_pen_val', default='default',
                        help='Maximum spectral penalty value.')

    # parser.add_argument('--spect_pen_spacing', type=str, default='lin',
    #                     choices=['lin', 'log'],
    #                     help='Spacing for spectral penalty.')

    # parser.add_argument('--sp_max_n_steps', type=int, default=200,
    #                     help='Maximum number of steps for spectral penalty.')

    parser.add_argument('--sp_abs_tol', type=float, default=1e-7,
                        help='Tolerance for spectral penalty stopping.')

    parser.add_argument('--sp_fine_tune_n_steps', default=None,
                        help='Number of fine tuning steps for sp mvmm.')

    parser.add_argument('--sp_eval_weights', type=str, default='adapt',
                        choices=['lin', 'quad', 'exp', 'adapt'],
                        help='Eigenvalue weights for spectral penalty.')

    parser.add_argument('--sp_pen_seq_spacing', type=str, default='lin',
                        choices=['lin', 'quad', 'exp'],
                        help='Spacing for spentral penalty values.')

    # parser.add_argument('--sp_not_by_block', action='store_true',
    #                     default=False,
    #                     help='Do not use sp_mvmm by block.')

    parser.add_argument('--sp_adapt_expon', type=float, default=2,
                        help='Exponent for adaptive weights .')

    parser.add_argument('--sp_default_c', type=float, default=1000,
                        help='Default multiplier for eval pen guess.')

    return parser


def spect_pen_from_args(args):

    # full_config = full_mvmm_from_args(args)
    # full_config['n_init'] = 1  # initializations controled elsewhere

    if args.sp_fine_tune_n_steps is None:
        fine_tune_n_steps = None
    else:
        fine_tune_n_steps = int(args.sp_fine_tune_n_steps)

    wbd_config = {'abs_tol': args.sp_abs_tol,
                  'max_n_steps': args.sp_max_n_steps,

                  'n_blocks': None,
                  'init_params_method': 'user',
                  'init_weights_method': 'user',
                  'n_init': 1,
                  'n_pen_tries': 1,
                  'fine_tune_n_steps': fine_tune_n_steps}

    if args.sp_max_pen_val == 'default':
        pen_max = 'default'
    else:
        pen_max = float(args.sp_max_pen_val)

    search_config = {'n_init': args.n_init,

                     'eval_weights': args.sp_eval_weights,
                     'adapt_expon': args.sp_adapt_expon,
                     'max_n_blocks': 'default',

                     'n_pen_seq': args.sp_n_pen_seq,
                     'pen_max': pen_max,
                     'pen_seq_spacing': args.sp_pen_seq_spacing,
                     'default_c': args.sp_default_c
                     }

    return {'wbd': wbd_config,
            'search': search_config}


def add_parsers(parser, to_add=[]):
    for add_func in to_add:
        parser = add_func(parser)
    return parser
