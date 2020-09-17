import matplotlib.pyplot as plt
import seaborn as sns

from mvmm_sim.viz_utils import draw_ellipse


def plot_cluster_params_2d(means, covariances=None,
                           kws={'marker': 'x', 'lw': 4, 's': 200}):
    """
    Plots GMM parameters for a 2d example.

    Parameters
    ----------
    means: (n_components, n_features)
        Cluster means.

    covariances: None, (n_components, n_features, n_features)
        Optional. Covariance matrix of each cluster.
    """
    assert means.shape[1] == 2

    n_components = means.shape[0]
    clust_colors = sns.color_palette("Set2", n_components)

    for k in range(n_components):
        col = clust_colors[k]
        plt.scatter(means[k, 0], means[k, 1],
                    **kws, color=col)

        if covariances is not None:
            draw_ellipse(position=means[k, :],
                         covariance=covariances[k, :, :],
                         color=col, zorder=0, alpha=1, fill=False)


def plot_data_2d(X, y, **kwargs):
    """
    Plots 2d data conditional on cluster assignments.

    Parameters
    ----------
    X: (n_samples, 2)
        Observations.

    y: (n_samples, )
        Cluster assignments.
    """
    n_components = len(set(y))
    colors = sns.color_palette("Set2", n_components)

    for k in range(n_components):
        mask = y == k
        plt.scatter(X[mask, 0], X[mask, 1],
                    color=colors[k],
                    label='cluster {}'.format(k),
                    **kwargs)
