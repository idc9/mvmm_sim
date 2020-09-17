from mvmm.multi_view.toy_data import setup_rand_view_params, sample_gmm, \
    setup_grid_mean_view_params
from mvmm.multi_view.toy_data_pi import sample_sparse_Pi, \
    sample_sparse_Pi_force_full_rowcols, view_indep_pi, motivating_ex
from mvmm.multi_view.block_diag.toy_data import block_diag_pi
from mvmm.multi_view.utils import get_n_comp


def get_data_dist(clust_param_config,
                  grid_means=False,
                  pi_dist='sparse',
                  pi_config={'n_rows_base': 5,
                             'n_cols_base': 8,
                             'density': .6,
                             'random_state': None}):
    """
    Multi-view GMM with sparse Pi whose parameters have been randomly generated.

    Output
    ------
    data_dist, Pi, view_params

    data_dist: callable(n_samples, seed)

    Pi: array-like

    view_params:
        Cluster parameters for each view.
    """

    # sample random Pi matrix
    Pi = get_pi(dist=pi_dist, **pi_config)

    # get number of view components
    _, n_view_components = get_n_comp(Pi)

    # sample GMM parameters
    # view_params = setup_rand_view_params(n_view_components,
    #                                      **clust_param_config)

    if grid_means:
        view_params = setup_grid_mean_view_params(n_view_components,
                                                  **clust_param_config)
    else:
        view_params = setup_rand_view_params(n_view_components,
                                             **clust_param_config)

    def data_dist(n_samples, random_state=None):
        """
        Parameters
        ----------
        n_samples: int
            Number of samples to draw.

        random_state: int, None
            Seed for generating samples.

        Output
        ------
        data_dist, Pi, view_params

        view_data: list of data matrices of shape (n_samples, n_features_v)
            Data for each view.

        Y: (n_samples, n_views)
            Y[i, v] = cluster index of ith obesrvation for the vth view
        """
        return sample_gmm(view_params, Pi, n_samples=n_samples,
                          random_state=random_state)

    return data_dist, Pi, view_params


def get_pi(dist='motivating_ex', *args, **kwargs):
    if dist == 'motivating_ex':
        return motivating_ex()
    elif dist == 'sparse':
        return sample_sparse_Pi(*args, **kwargs)
    elif dist == 'sparse_force_full_rowcols':
        return sample_sparse_Pi_force_full_rowcols(*args, **kwargs)
    elif dist == 'block_diag':
        return block_diag_pi(*args, **kwargs)
    elif dist == 'indep':
        return view_indep_pi(*args, **kwargs)
    else:
        raise ValueError('dist = {} is invalid argument'.format(dist))
