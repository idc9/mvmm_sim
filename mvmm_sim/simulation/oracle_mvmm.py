from copy import deepcopy

from mvmm.multi_view.MVMM import MVMM
from mvmm.single_view.gaussian_mixture import GaussianMixture


def format_view_params(view_params, covariance_type='diag'):
    # if covariance_type != 'diag':
    #     raise NotImplementedError
    new = []
    for v in range(len(view_params)):

        p = {'means': view_params[v]['means']}

        c = view_params[v]['covs']
        print(c.shape)
        if covariance_type == 'diag':
            n_feats = p['means'].shape[1]
            c = c.reshape(-1, n_feats)

        p['covariances'] = c
        new.append(p)

    return new


def set_mvmm_from_params(view_params, Pi, covariance_type='diag'):
    n_views = len(view_params)

    if isinstance(covariance_type, str):
        covariance_type = [covariance_type] * n_views

    base_view_models = []
    for v in range(n_views):
        params = deepcopy(view_params[v])

        # if covariance_type[v] != 'diag':
        #     raise NotImplementedError

        gmm = GaussianMixture(covariance_type=covariance_type[v])
        gmm._set_parameters(params)
        gmm.n_components = params['means'].shape[0]

        base_view_models.append(gmm)

    mvmm = MVMM(base_view_models=base_view_models)
    mvmm.view_models_ = [mvmm.base_view_models[v] for v in range(n_views)]
    mvmm._set_parameters({'weights': Pi.reshape(-1)})

    return mvmm
