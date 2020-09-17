import numpy as np
from time import time
from collections import Counter
import networkx as nx
# import gmatch4py as gm
from grakel import graph_from_networkx, RandomWalk
import pandas as pd
import os
from copy import deepcopy

from mvmm.multi_view.block_diag.graph.linalg import get_adjmat_bp
from mvmm.multi_view.base import MultiViewMixtureModelMixin
from mvmm.multi_view.MVMMGridSearch import MVMMGridSearch
from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.TwoStage import TwoStage
# from mvmm.multi_view.SpectralPenSearchMVMM import SpectralPenSearchMVMM
from mvmm.multi_view.SpectralPenSearchByBlockMVMM import \
    SpectralPenSearchByBlockMVMM


def is_mvmm(estimator):
    """
    Returns True iff estimator is a multi-view mixture model.
    """
    if isinstance(estimator, MultiViewMixtureModelMixin) or \
        isinstance(estimator, MVMMGridSearch) or \
            isinstance(estimator, TwoStage) or \
            isinstance(estimator, SpectralPenSearchByBlockMVMM):
        #  isinstance(estimator, SpectralPenSearchMVMM) or \

        return True
    else:
        return False


def is_block_diag_mvmm(estimator):
    if isinstance(estimator, MVMMGridSearch):
        return is_block_diag_mvmm(estimator.base_estimator)

    if isinstance(estimator, TwoStage):
        return is_block_diag_mvmm(estimator.base_final)

    if isinstance(estimator, BlockDiagMVMM):
        return True
    else:
        return False


def clf_fit_and_score(clf, X_tr, y_tr, X_tst, y_tst):
    """
    Fits a classification model and scores the results for the training
    and test set.

    Parameters
    ----------
    clf:
        A sklearn compatible classifier..

    X_tr, y_tr: training data and true labels.

    X_tst, y_tst: test data and true labels.

    Output
    ------
    results: dict
        Train and test set results.
    """
    start_time = time()

    def get_metrics(y_true, y_pred):
        """
        Measures of classification accuracy.
        """
        return {'acc': np.mean(y_true == y_pred)}

    clf.fit(X_tr, y_tr)

    y_hat_tr = clf.predict(X_tr)
    y_hat_tst = clf.predict(X_tst)

    results = {'tr': get_metrics(y_tr, y_hat_tr),
               'tst': get_metrics(y_tst, y_hat_tst),
               'runtime': time() - start_time}

    return results


def get_pi_acc(Pi_est, Pi_true, method='random_walk', **kwargs):
    """
    Computes the graph edit distance between the sparsity graphs.
    """
    A_est = get_adjmat_bp(Pi_est > 0)
    A_true = get_adjmat_bp(Pi_true > 0)

    G_est = nx.from_numpy_array(A_est)
    G_true = nx.from_numpy_array(A_true)

    sim = graph_similarity(G_est, G_true, method=method, **kwargs)
    return sim


def graph_similarity(G, H, method='random_walk', **kwargs):
    """
    Parameters
    ----------
    G, H: nx.Graph

    """
    assert method in ['random_walk']
    if method == 'random_walk':
        kernel = RandomWalk(**kwargs)

    return kernel.fit_transform(graph_from_networkx([G, H]))[0, 1]


def get_n_comp_seq(true_n_components, pm):
    return np.arange(max(1, true_n_components - pm), true_n_components + pm)


def get_empirical_pi(Y, shape, scale='prob'):
    assert scale in ['prob', 'counts']
    pi_empir = np.zeros(shape)
    pairs = Counter(tuple(Y[i, :]) for i in range(Y.shape[0]))

    for k in pairs.keys():
        pi_empir[k[0], k[1]] = pairs[k]

    if scale == 'prob':
        pi_empir = pi_empir / pi_empir.sum()
    return pi_empir


def extract_tuning_param_vals(df):

    vals = []
    for tune_param in df['tuning_param_values']:
        assert len(list(tune_param.keys())) == 1
        param_name = list(tune_param.keys())[0]
        vals.append(tune_param[param_name])

    vals = pd.Series(vals, index=df.index, name=param_name)

    return vals, param_name


def add_pi_config(pi_name='motivating_ex', config={}):

    if pi_name == 'sparse_pi':
        config['pi_dist'] = 'sparse'
        config['pi_config'] = {'n_rows_base': 5,
                               'n_cols_base': 8,
                               'density': .6,
                               'random_state': 78923}

    elif pi_name == 'sparse_pi_2':
        config['pi_dist'] = 'sparse'
        config['pi_config'] = {'n_rows_base': 10,
                               'n_cols_base': 10,
                               'density': .6,
                               'random_state': 94009}

    elif 'diagonal' in pi_name:

        config['pi_dist'] = 'block_diag'
        n_blocks = int(pi_name.split('_')[1])
        block_shapes = [(1, 1)] * n_blocks
        config['pi_config'] = {'block_shapes': block_shapes}

    elif 'indep' in pi_name:
        config['pi_dist'] = 'indep'
        n_comp = int(pi_name.split('_')[1])
        config['pi_config'] = {'n_comp': n_comp}

    elif 'lollipop' in pi_name:
        config['pi_dist'] = 'block_diag'
        _, stick, bulb = pi_name.split('_')
        stick = int(stick)
        bulb = int(bulb)

        block_shapes = [(1, 1)] * stick + [(bulb, bulb)]
        block_weights = [1] * stick + [1 / bulb]
        config['pi_config'] = {'block_shapes': block_shapes,
                               'block_weights': block_weights}

    elif 'beads' in pi_name:
        config['pi_dist'] = 'block_diag'
        _, size, n_beads = pi_name.split('_')

        block_shapes = [(int(size), int(size))] * int(n_beads)
        config['pi_config'] = {'block_shapes': block_shapes}

    elif pi_name == 'motivating_ex':
        config['pi_dist'] = 'motivating_ex'
        config['pi_config'] = {}

    else:
        raise ValueError('bad arguent for pi_name, {}'.format(pi_name))

    return config


def safe_drop(x, val):
    if type(x) == list:
        ret = 'list'
    elif type(x) == np.array:
        ret = 'np'
    else:
        raise ValueError('bad x input')

    new_x = list(set(x).difference([val]))

    if ret == 'list':
        return new_x

    else:
        return np.array(new_x)


def safe_drop_list(x, to_drop):
    _x = deepcopy(x)
    for val in to_drop:
        _x = safe_drop(x=_x, val=val)
    return _x


def format_mini_experiment(args):
    if not args.mini:
        return args

    args.sim_name = 'mini'

    # max n steps
    for k in args.__dict__:
        if 'max_n_steps' in k:
            args.__dict__[k] = 5

    # number of initilalizations
    args.n_init = 1

    args.bd_n_pen_tries = 1
    args.sp_n_pen_seq = 2
    args.log_pen_n_pen_seq = 2

    return args


def make_and_get_dir(*args):
    d = os.path.join(*args)
    os.makedirs(d, exist_ok=True)
    return d
