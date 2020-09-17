import pandas as pd
from itertools import product
import numpy as np
from glob import glob
from joblib import load
import os
from collections import Counter


def get_mismatches(a, b):
    A = set(a)
    B = set(b)

    in_a_not_b = A.difference(B)
    in_b_not_a = B.difference(A)
    return in_a_not_b, in_b_not_a


def intersection(*args):
    s = set(args[0])
    for a in args[1:]:
        s = s.intersection(a)
    return list(s)


def apply_pd_df(func, x):
    index = x.index
    columns = x.columns
    return pd.DataFrame(func(x), index=index, columns=columns)


def get_n_view_comp_seq(n_comp_range):
    n_view_comp_seq = []
    for nc0, nc1 in product(range(n_comp_range[0][0], n_comp_range[0][1] + 1),
                            range(n_comp_range[1][0], n_comp_range[1][1] + 1)):

        n_view_comp_seq += [[nc0, nc1]]

    return n_view_comp_seq


def argmax(values, n=1):
    return (-np.array(values)).argsort()[:n]


def argmin(values, n=1):
    return (np.array(values)).argsort()[:n]


def load_fitted_mvmms(save_dir):
    fpaths = glob(save_dir + '/mvmm_*')
    fit_data = [load(fpath) for fpath in fpaths]

    n_view_components = [res['n_view_components'] for res in fit_data]

    return {'fit_data': fit_data,
            'n_view_components': n_view_components,
            'dataset_names': fit_data[0]['dataset_names']}


def load_data(*fpaths):
    view_data = []
    dataset_names = []
    feat_names = []
    sample_names = None
    for v in range(len(fpaths)):
        name = os.path.basename(fpaths[v])
        if '.' in name:
            name = name.split('.')[0]
        dataset_names.append(name)

        x = pd.read_csv(fpaths[v], index_col=0)

        feat_names.append(x.columns.values)

        if sample_names is None:
            sample_names = x.index.values
        else:
            assert np.alltrue(sample_names == x.index.values)

        view_data.append(x.values)

    return view_data, dataset_names, sample_names, feat_names


def apply_pd(func, df):
    index = df.index.values
    columns = df.columns.values

    df = func(df.values)
    df = pd.DataFrame(df, index=index, columns=columns)
    return df


def drop_small_classes(y, clust_size_min=0):
    cl_counts = pd.Series(Counter(y))
    small_cls = cl_counts[cl_counts < clust_size_min].index.values

    return y[~pd.Series(y).isin(small_cls)]
