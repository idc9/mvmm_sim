import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from numbers import Number
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd


def draw_ellipse(position, covariance, ax=None, n_sig=3, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    if isinstance(n_sig, Number):
        n_sig = [n_sig]
    # Draw the Ellipse
    for ns in n_sig:
        ax.add_patch(Ellipse(position, ns * width, ns * height,
                             angle, **kwargs))


def savefig(fpath, dpi=100, close=True):
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)
    if close:
        plt.close()


def safe_heatmap(X, **kws):
    """
    Seaborn heatmap without cutting top/bottom off.
    """
    f, ax = plt.subplots()
    sns.heatmap(X, ax=ax, **kws)
    ax.set_ylim(X.shape[1] + .5, 0)
    ax.set_xlim(0, X.shape[0] + .5)


def set_xaxis_int_ticks():
    """
    Sets integer x ticks
    """
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def set_yaxis_int_ticks():
    """
    Sets integer y ticks
    """
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def mc_curve_summary(values, axis=0):
    """
    Gets summary curves for monte-carlo simulations for a range of settings.

    Parameters
    ----------
    values: array-like, (n_mc_samples, n_settings)
        The values of each simulation.

    axis: int
        If 0, the MC samples are the rows.
    """
    # TODO: add options for +/- std and mean
    values = pd.DataFrame(values)

    center = values.median(axis=axis)
    lower = values.quantile(q=.05, axis=axis)
    upper = values.quantile(q=.95, axis=axis)

    return center, lower, upper


def plot_mc_curve(values, x_values=None, color='black', alpha=.4,
                  label=None, include_points=True, axis=0):
    """
    Plots the results of a monte-carlo simulation for a range of settings.

    Parameters
    ----------
    values: array-like, (n_mc_samples, n_settings)
        The values of each simulation.

    x_values: array-like, (n_mc_samples, ) or None
        The values for the x-axis

    color: str
        Color of the curve.

    alpha: float
        Alpha for the uncertainty curves.

    label: None, str
        Optional label.

    include_points: bool
        Whether or not to include the raw MC points.

    axis: int
        If 0, the MC samples are the rows.

    """

    if x_values is None:
        x_values = np.arange(values.shape[axis])

    center, lower, upper = mc_curve_summary(values, axis=axis)

    plt.plot(x_values, center, marker='.', label=label, color=color)
    plt.fill_between(x_values, lower, upper,
                     color=color, alpha=alpha, edgecolor=None)

    # plot the raw points
    if include_points:
        for r in range(values.shape[0]):
            plt.scatter(x_values, np.array(values)[r, :],
                        color=color, alpha=alpha, s=10)
    if label is not None:
        plt.legend()


def axvline_with_tick(x=0, bold=False, **kwargs):
    """
    plt.axvline but atomatically adds tick to x axis

    Parameters
    ----------
    x, **kwargs: see plt.axvline arguments

    bold: bool
        Whether or not to bold the added tick.
    """
    plt.axvline(x=x, **kwargs)
    plt.xticks(list(plt.xticks()[0]) + [x])

    if bold:
        ax = plt.gca()
        ax.get_xticklabels()[-1].set_weight("bold")


def axhline_with_tick(y=0, bold=False, **kwargs):
    """
    plt.axhline but atomatically adds tick to y axis

    Parameters
    ----------
    y, **kwargs: see plt.axhline arguments

    bold: bool
        Whether or not to bold the added tick.
    """
    plt.axhline(y=y, **kwargs)
    plt.yticks(list(plt.yticks()[0]) + [y])

    if bold:
        ax = plt.gca()
        ax.get_xticklabels()[-1].set_weight("bold")


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
