import numpy as np
from numpy import size
from numbers import Number

from copy import deepcopy

from mvmm.multi_view.LogPenMVMM import LogPenMVMM
from mvmm.multi_view.MVMMGridSearch import MVMMGridSearch
from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.MVMM import MVMM
from mvmm.multi_view.TwoStage import TwoStage
# from mvmm.multi_view.SpectralPenSearchMVMM import SpectralPenSearchMVMM
from mvmm.multi_view.SpectralPenSearchByBlockMVMM import \
    SpectralPenSearchByBlockMVMM
from mvmm.single_view.gaussian_mixture import GaussianMixture
from mvmm.single_view.MMGridSearch import MMGridSearch
from mvmm.multi_view.utils import unit_intval_linspace, \
    unit_intval_polyspace, unit_intval_logspace


def get_single_view_models(cat_gmm_config=None, view_gmm_config=None,
                           cat_n_comp=None, view_n_comp=None,
                           select_metric='bic',
                           metrics2compute=['aic', 'bic', 'silhouette',
                                            'calinski_harabasz',
                                            'davies_bouldin', 'dunn'],
                           n_jobs_tune=None):

    models = {}

    ###########
    # cat GMM #
    ###########
    if cat_gmm_config is not None:

        cat_gmm = GaussianMixture(**cat_gmm_config)

        # tuning
        if isinstance(cat_n_comp, Number):
            cat_gmm.set_params(n_components=int(cat_n_comp))

        elif cat_n_comp is not None:
            assert size(cat_n_comp) >= 2

            cat_gmm = _get_gmm_gs(cat_gmm,
                                  n_comp_seq=cat_n_comp,
                                  select_metric=select_metric,
                                  metrics2compute=metrics2compute,
                                  n_jobs=n_jobs_tune)

        models['cat_gmm'] = cat_gmm

    ############
    # view GMM #
    ############

    if view_gmm_config is not None:

        view_gmms = []
        for v in range(len(view_n_comp)):
            view_gmms.append(GaussianMixture(**view_gmm_config))

            # tuning
            if isinstance(view_n_comp[v], Number):
                view_gmms[v].set_params(n_components=int(view_n_comp[v]))

            elif view_n_comp[v] is not None:
                assert size(view_n_comp[v]) >= 2

                view_gmms[v] = _get_gmm_gs(view_gmms[v],
                                           n_comp_seq=view_n_comp[v],
                                           select_metric=select_metric,
                                           metrics2compute=metrics2compute,
                                           n_jobs=n_jobs_tune)

        models['view_gmms'] = view_gmms

    return models


def get_mvmms(n_view_components,
              base_gmm_config, full_mvmm_config=None,
              log_pen_config=None, bd_config=None,
              spect_pen_config=None,
              n_blocks='default',
              sp_by_block=False,
              select_metric='bic',
              metrics2compute=['aic', 'bic', 'silhouette',
                               'calinski_harabasz', 'davies_bouldin', 'dunn'],
              n_jobs_tune=None):

    n_views = len(n_view_components)
    models = {}
    #############
    # Full MVMM #
    #############

    if full_mvmm_config is not None:
        base_view_models = \
            [GaussianMixture(n_components=n_view_components[v],
                             **base_gmm_config) for v in range(n_views)]

        full_mvmm = MVMM(base_view_models=base_view_models,
                         **full_mvmm_config)

        models['full_mvmm'] = full_mvmm

    ################
    # log pen MVMM #
    ################

    if log_pen_config is not None:

        base_start = MVMM(base_view_models=base_view_models,
                          **log_pen_config['start'])
        base_start.n_init = 1

        base_final = LogPenMVMM(base_view_models=deepcopy(base_view_models),
                                **log_pen_config['final'])

        log_pen_mvmm = TwoStage(base_start=base_start,
                                base_final=base_final,
                                **log_pen_config['two_stage'])

        mult_vals = get_log_pen_mult_vals(**log_pen_config['search'])
        param_grid = {'pen':
                      np.array(mult_vals) / np.product(n_view_components)}
        log_pen_mvmm = MVMMGridSearch(base_estimator=log_pen_mvmm,
                                      param_grid=param_grid,
                                      select_metric=select_metric,
                                      metrics2compute=metrics2compute,
                                      n_jobs=n_jobs_tune)

        models['log_pen_mvmm'] = log_pen_mvmm

    ###################
    # block diag MVMM #
    ###################

    if bd_config is not None:

        base_start = MVMM(base_view_models=base_view_models,
                          **bd_config['start'])

        base_start.n_init = 1

        base_final = BlockDiagMVMM(base_view_models=base_view_models,
                                   **bd_config['final'])

        bd_mvmm = TwoStage(base_start=base_start,
                           base_final=base_final,
                           **bd_config['two_stage'])

        # setup tuning
        if isinstance(n_blocks, Number):
            # no tuning
            bd_mvmm.base_final.set_params(n_blocks=n_blocks)

        else:

            if type(n_blocks) == str and n_blocks == 'default':
                n_blocks = np.arange(1, min(n_view_components) + 1)

            bd_mvmm = MVMMGridSearch(base_estimator=bd_mvmm,
                                     param_grid={'n_blocks': n_blocks},
                                     select_metric=select_metric,
                                     metrics2compute=metrics2compute,
                                     n_jobs=n_jobs_tune)

        models['bd_mvmm'] = bd_mvmm

    #########################
    # Spectral penalty MVMM #
    #########################

    if spect_pen_config is not None:
        assert full_mvmm is not None  # need to setup full_mvmm for sp_mvmm

        full_mvmm_sp = deepcopy(full_mvmm)
        full_mvmm_sp.n_init = 1

        wbd_mvmm = BlockDiagMVMM(base_view_models=deepcopy(base_view_models),
                                 verbosity=0,
                                 **spect_pen_config['wbd'])

        if not sp_by_block:
            raise NotImplementedError
            # sp_mvmm = SpectralPenSearchMVMM(base_mvmm_0=full_mvmm_sp,
            #                                 base_wbd_mvmm=wbd_mvmm,
            #                                 select_metric=select_metric,
            #                                 metrics2compute=metrics2compute,
            #                                 verbosity=1,
            #                                 **spect_pen_config['search'])

        else:
            sp_mvmm = \
                SpectralPenSearchByBlockMVMM(base_mvmm_0=full_mvmm_sp,
                                             base_wbd_mvmm=wbd_mvmm,
                                             select_metric=select_metric,
                                             metrics2compute=metrics2compute,
                                             verbosity=1,
                                             **spect_pen_config['search'])

        models['sp_mvmm'] = sp_mvmm

    return models


def get_log_pen_mult_vals(spacing, num):
    if spacing == 'lin':
        return unit_intval_linspace(num=num)

    elif spacing == 'quad':
        return unit_intval_polyspace(num=num, deg=2)

    elif spacing == 'sqrt':
        return unit_intval_polyspace(num=num, deg=.5)

    elif 'log' in spacing:
        stop = - float(spacing.split('_')[1])
        return unit_intval_logspace(num=num, stop=stop)


def _get_gmm_gs(gmm, n_comp_seq, select_metric, metrics2compute, n_jobs=None):

    param_grid = {'n_components': n_comp_seq}
    gs = MMGridSearch(base_estimator=gmm,
                      param_grid=param_grid,
                      select_metric=select_metric,
                      metrics2compute=metrics2compute,
                      n_jobs=n_jobs)
    return gs
