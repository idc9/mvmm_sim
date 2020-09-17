import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loading(v, abs_sorted=True, show_var_names=True,
                 significant_vars=None,
                 colors=None, vert=True, show_top=None):
    """
    Plots a single loadings component.

    Parameters
    ----------
    v: array-like
        The loadings component.

    abs_sorted: bool
        Whether or not to sort components by their absolute values.


    significant_vars: {array-like, None}
        Indicated which features are significant in this component.

    show_top: None, int
        Will only display this number of top loadings components when
        sorting by absolute value.

    colors: None, array-like
        Colors for each loading. If None, will use sign.

    vert: bool
        Make plot vertical or horizontal
    """
    if hasattr(v, 'name'):
        xlab = v.name
    else:
        xlab = ''

    if show_top is not None:
        show_top = min(len(v), show_top)

    if type(v) != pd.Series:
        v = pd.Series(v, index=['feature {}'.format(i) for i in range(len(v))])
        if significant_vars is not None:
            significant_vars = v.index.iloc[significant_vars]
    else:
        if colors is not None:
            colors = colors.loc[v.index]

    if abs_sorted:
        v_abs_sorted = np.abs(v).sort_values()
        v = v[v_abs_sorted.index]

        if show_top is not None:
            v = v[-show_top:]

            if significant_vars is not None:
                significant_vars = significant_vars[-show_top:]

    if show_top is not None:
        v = v[0:int(show_top)]

    inds = np.arange(len(v))

    signs = v.copy()
    signs[v > 0] = 'pos'
    signs[v < 0] = 'neg'
    if significant_vars is not None:
        signs[v.index.difference(significant_vars)] = 'zero'
    else:
        signs[v == 0] = 'zero'
    s2c = {'pos': 'red', 'neg': 'blue', 'zero': 'grey'}

    if colors is None:
        colors = signs.apply(lambda x: s2c[x])

    if vert:
        plt.scatter(v, inds, color=colors)
        plt.axvline(x=0, alpha=.5, color='black')
        plt.xlabel(xlab)
        if show_var_names:
            plt.yticks(inds, v.index)
    else:
        v = v[::-1]
        colors = colors[::-1]
        plt.scatter(inds, v, color=colors)
        plt.axhline(y=0, alpha=.5, color='black')
        plt.ylabel(xlab)
        if show_var_names:
            plt.xticks(inds, v.index)

    max_abs = np.abs(v).max()
    xmin = -1.2 * max_abs
    xmax = 1.2 * max_abs
    if np.mean(signs == 'pos') == 1:
        xmin = 0
    elif np.mean(signs == 'neg') == 1:
        xmax = 0
    elif np.mean(signs == 'zero') == 1:
        xmin = 1
        xmax = 1

    if vert:
        plt.xlim(xmin, xmax)
    else:
        plt.ylim(xmin, xmax)

    if vert:
        ticklabs = plt.gca().get_yticklabels()
    else:
        ticklabs = plt.gca().get_xticklabels()

    for t, c in zip(ticklabs, colors):
        t.set_color(c)
        if c != 'grey':
            t.set_fontweight('bold')

    if not vert:
        plt.gca().xaxis.set_tick_params(rotation=55)


def plot_loadings_scatter(v, show_top=None):
    s2c = {1: 'red', -1: 'blue', 0: 'grey'}
    colors = np.sign(v).apply(lambda x: s2c[x])

    if show_top is not None:
        show_top = min(show_top, len(v))

        v_abs_sorted = np.abs(v).sort_values(ascending=False)
        pos_cutoff = v_abs_sorted[show_top - 1]
        neg_cutoff = - pos_cutoff

    else:
        pos_cutoff = None
        neg_cutoff = None
    if (v > 0).sum() == 0:
        # all_neg = True
        ylim = [None, 0]
        pos_cutoff = None

    elif (v < 0).sum() == 0:
        # all_pos = True
        ylim = [0, None]
        neg_cutoff = None

    else:
        ylim = [None, None]

    # plt.figure(figsize=(8, 8))
    idxs = np.arange(1, len(v) +1)
    plt.scatter(idxs, v, c=colors)
    # plt.scatter(idxs, v_abs_sorted, c=colors)
    # plt.scatter(idxs, np.sort(v), c=colors)

    plt.axhline(0, color='black')
    # plt.xlabel('Variable index')
    plt.ylim(*ylim)
    if pos_cutoff is not None:
        plt.axhline(pos_cutoff, color='red')

    if neg_cutoff is not None:
        plt.axhline(neg_cutoff, color='blue')

    if hasattr(v, 'name'):
        plt.ylabel(v.name)
