import numpy as np
import pandas as pd
from joblib import dump
import os

# from sklearn.metrics import adjusted_rand_score
from sklearn.base import clone
from time import time
from copy import deepcopy
from datetime import datetime
from warnings import simplefilter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mvmm.multi_view.utils import view_labs_to_overall,\
    get_n_comp
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary,\
    get_comm_mat

from mvmm.utils import get_seeds
from mvmm.multi_view.TwoStage import TwoStage
from mvmm.BaseGridSearch import BaseGridSearch
from mvmm.multi_view.block_diag.graph.bipt_spect_partitioning import \
    run_bipt_spect_partitioning
# from mvmm.multi_view.SpectralPenSearchMVMM import SpectralPenSearchMVMM
from mvmm.multi_view.SpectralPenSearchByBlockMVMM import \
    SpectralPenSearchByBlockMVMM
from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.LogPenMVMM import LogPenMVMM
from mvmm.multi_view.MVMM import MVMM
from mvmm_sim.simulation.oracle_mvmm import format_view_params, \
    set_mvmm_from_params
from mvmm.clustering_measures import unsupervised_cluster_scores, \
    multi_view_safe_pairwise_distances
from mvmm.single_view.gaussian_mixture import default_cov_regularization

from mvmm_sim.simulation.utils import get_empirical_pi, get_n_comp_seq
from mvmm_sim.simulation.community_results import get_comm_pred_summary
from mvmm_sim.simulation.utils import is_mvmm, get_pi_acc, \
    clf_fit_and_score, is_block_diag_mvmm
from mvmm_sim.simulation.data_dist_from_config import get_data_dist
from mvmm_sim.simulation.models_from_config import get_mvmms, \
    get_single_view_models
from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.simulation.aligned_pi_dist import get_aligned_pi_distance
from mvmm_sim.simulation.cluster_report import cluster_report

from sklearn.exceptions import ConvergenceWarning


def run_sim_from_configs(clust_param_config, grid_means, pi_dist, pi_config,
                         single_view_config, mvmm_config,
                         n_samples_tr, n_samples_tst,
                         gmm_pm, n_blocks_pm,
                         reg_covar_mult=1e-2,
                         to_exclude=None,
                         data_seed=None, mc_index=None, args=None,
                         save_fpath=None):

    input_config = locals()

    print('Simulation starting at',
          datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    print(args)

    start_time = time()

    # cluster parameters are different for each MC iteration
    data_dist, Pi, view_params = \
        get_data_dist(clust_param_config=clust_param_config,
                      grid_means=grid_means,
                      pi_dist=pi_dist, pi_config=pi_config)

    n_comp_tot, n_view_components = get_n_comp(Pi)
    n_blocks_true = community_summary(Pi)[0]['n_communities']

    # cat and view GMMs sequences to search over
    single_view_config['cat_n_comp'] = get_n_comp_seq(n_comp_tot,
                                                      gmm_pm)

    single_view_config['view_n_comp'] = \
        [get_n_comp_seq(n_view_components[v], gmm_pm) for v in range(2)]

    # n block sequence to search over
    lbd = max(2, n_blocks_true - n_blocks_pm)
    ubd = min(n_blocks_true + n_blocks_pm, min(n_view_components))
    nb_seq = np.arange(lbd, ubd + 1)

    mvmm_config['n_blocks'] = nb_seq  # 'default'

    # get models
    # models = models_from_config(n_view_components=n_view_components,
    #                             n_comp_tot=n_comp_tot,
    #                             n_blocks=n_blocks_true,
    #                             **model_config)

    models = {**get_single_view_models(**single_view_config),

              **get_mvmms(n_view_components=n_view_components,
                          **mvmm_config),

              'clf': LinearDiscriminantAnalysis()}

    # set oracle model
    _view_params = format_view_params(view_params,
                                      covariance_type='full')

    models['oracle'] = set_mvmm_from_params(view_params=_view_params,
                                            Pi=Pi,
                                            covariance_type='full')
    zero_thresh = .01 / (n_view_components[0] * n_view_components[1])

    # log_dir = os.path.dirname(save_fpath)
    # log_dir = os.path.join(save_fpath, 'log')
    # os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(save_fpath), 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_fname = os.path.basename(save_fpath) + '_simulation_progress.txt'
    log_fpath = os.path.join(log_dir, log_fname)

    # run simulation
    clust_results, clf_results, fit_models,\
        bd_summary, Pi_empirical, tr_data, runtimes = \
        run_sim(models=models,
                data_dist=data_dist, Pi=Pi, view_params=view_params,
                n_samples_tr=n_samples_tr, data_seed=data_seed,
                mc_index=mc_index, n_samples_tst=n_samples_tst,
                zero_thresh=zero_thresh,
                reg_covar_mult=reg_covar_mult,
                to_exclude=to_exclude,
                log_fpath=log_fpath)

    log_pen_param_grid = fit_models['log_pen_mvmm'].param_grid_
    bd_param_grid = fit_models['bd_mvmm'].param_grid_
    sp_param_grid = fit_models['sp_mvmm'].param_grid_

    metadata = {'n_samples_tr': n_samples_tr,
                'n_comp_tot': n_comp_tot,
                'mc_index': mc_index,
                'n_view_components': n_view_components,
                'config': input_config,
                'args': args,
                'log_pen_param_grid': log_pen_param_grid,
                'bd_param_grid': bd_param_grid,
                'sp_param_grid': sp_param_grid,
                'zero_thresh': zero_thresh,
                'tot_runtime': time() - start_time,
                'fit_runtimes': runtimes}

    print('Simulation finished at {} and took {} seconds'.
          format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                 metadata['tot_runtime']))

    if save_fpath is not None:
        print('saving file at {}'.format(save_fpath))

        dump({'clust_results': clust_results,
              'clf_results': clf_results,
              'bd_summary': bd_summary,
              'metadata': metadata,
              'config': input_config}, save_fpath)

        # save some extra data for one MC repition
        if mc_index == 0:
            save_dir = os.path.dirname(save_fpath)
            fpath = os.path.join(save_dir,
                                 'extra_data_mc_0__n_samples_{}'.
                                 format(n_samples_tr))
            dump({'Pi': Pi,
                  'view_params': view_params,
                  'tr_data': tr_data,
                  'fit_models': fit_models,
                  'Pi_empirical': Pi_empirical}, fpath)

    return {'clust_results': clust_results,
            'clf_results': clf_results,
            'bd_summary': bd_summary,
            'metadata': metadata,
            'fit_models': fit_models,
            'Pi': Pi,
            'view_params': view_params}


def run_sim(models, data_dist, Pi, view_params, n_samples_tr, data_seed,
            n_samples_tst=2000, zero_thresh=0, reg_covar_mult=1e-2,
            mc_index=None, to_exclude=None, log_fpath=None):

    """

    Parameters
    ----------
    models

    data_dist: callable(n_samples, seed)
        Function to generate data.

    n_samples_tr: int
        Number of training samples.

    data_seed: int
        Seed for sampling train/test observations.

    n_samples_tst: int
        Number of samples to get for test data.

    """

    res_writer = ResultsWriter(log_fpath, delete_if_exists=True)
    res_writer.write("Beginning simulation at {}".format(get_current_time))
    overall_start_time = time()

    seeds = get_seeds(random_state=data_seed, n_seeds=2)

    # sample data
    X_tr, Y_tr = data_dist(n_samples=n_samples_tr,
                           random_state=seeds[0])
    X_tst, Y_tst = data_dist(n_samples=n_samples_tst,
                             random_state=seeds[1])
    n_views = len(X_tr)

    Pi_empirical = get_empirical_pi(Y_tr, Pi.shape, scale='counts')

    runtimes = {}

    if to_exclude is None:
        to_exclude = []
    for m in to_exclude:
        assert m in ['bd_mvmm', 'sp_mvmm', 'log_pen_mvmm']

    #############################
    # covariance regularization #
    #############################
    n_views = len(X_tr)
    reg_covar = {}

    # set cov reg for each view
    for v in range(n_views):
        reg = default_cov_regularization(X=X_tr[v], mult=reg_covar_mult)

        models['view_gmms'][v].base_estimator.set_params(reg_covar=reg)

        models['full_mvmm'].base_view_models[v].set_params(reg_covar=reg)

        models['bd_mvmm'].base_estimator.base_start.base_view_models[v].\
            set_params(reg_covar=reg)
        models['bd_mvmm'].base_estimator.base_final.base_view_models[v].\
            set_params(reg_covar=reg)

        models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
            set_params(reg_covar=reg)
        models['log_pen_mvmm'].base_estimator.base_start.base_view_models[v].\
            set_params(reg_covar=reg)

        models['sp_mvmm'].base_mvmm_0.base_view_models[v].\
            set_params(reg_covar=reg)
        models['sp_mvmm'].base_wbd_mvmm.base_view_models[v].\
            set_params(reg_covar=reg)

        # print and save
        reg_covar[v] = reg
        res_writer.write("\nCovarinace regularization for view {} is {}".
                         format(v, reg))
        stds = X_tr[v].std(axis=0)
        res_writer.write("Smallest variance: {}".format(stds.min() ** 2))
        res_writer.write("Largest variance: {}".format(stds.max() ** 2))

    # for cat GMM
    reg = default_cov_regularization(X=np.hstack(X_tr), mult=reg_covar_mult)
    models['cat_gmm'].base_estimator.set_params(reg_covar=reg)
    reg_covar['cat_gmm'] = reg

    ##############
    # fit models #
    ##############

    # get classification resuls
    clf_results = {}
    start_time = time()
    clf_results['cat'] = clf_fit_and_score(clone(models['clf']),
                                           X_tr=np.hstack(X_tr),
                                           y_tr=view_labs_to_overall(Y_tr),
                                           X_tst=np.hstack(X_tst),
                                           y_tst=view_labs_to_overall(Y_tst))

    runtimes['cat'] = time() - start_time

    for v in range(n_views):
        start_time = time()
        clf_results['view_{}'.format(v)] =\
            clf_fit_and_score(clone(models['clf']),
                              X_tr=X_tr[v],
                              y_tr=Y_tr[:, v],
                              X_tst=X_tst[v],
                              y_tst=Y_tst[:, v])

        runtimes['clf_view_{}'.format(v)] = time() - start_time

    # fit clustering
    simplefilter('ignore', ConvergenceWarning)

    results_df = pd.DataFrame()

    sim_stub = {'mc_index': mc_index, 'n_samples': n_samples_tr}

    dists_cat = pairwise_distances(X=np.hstack(X_tr))

    dists_views = [pairwise_distances(X=X_tr[v]) for v in range(n_views)]

    kws = {'sim_stub': sim_stub,
           'X_tr': X_tr, 'Y_tr': Y_tr,
           'X_tst': X_tst, 'Y_tst': Y_tst,
           'Pi_true': Pi, 'view_params_true': view_params,
           'zero_thresh': zero_thresh,
           }

    ###########
    # cat-GMM #
    ###########

    # print('start fitting cat-GMM at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    start_time = time()
    models['cat_gmm'].fit(np.hstack(X_tr))

    runtimes['cat_gmm'] = time() - start_time
    res_writer.write('fitting grid search cat-GMM took {:1.2f} seconds'.
                     format(runtimes['cat_gmm']))

    results_df = add_gs_results(results_df=results_df,
                                model=models['cat_gmm'],
                                model_name='gmm_cat',
                                dataset='full', view='both',
                                X_tr_precomp_dists=dists_cat,
                                **kws)

    #############
    # View GMMs #
    #############

    # print('start fitting view marginal GMMs at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for v in range(n_views):
        start_time = time()
        models['view_gmms'][v].fit(X_tr[v])

        runtimes['gmm_view_{}'.format(v)] = time() - start_time
        res_writer.write('fitting marginal view {} GMM took {:1.2f} seconds'.
                         format(v, runtimes['gmm_view_{}'.format(v)]))

        # gmm fit on this view
        results_df = add_gs_results(results_df=results_df,
                                    model=models['view_gmms'][v],
                                    model_name='marginal_view_{}'.format(v),
                                    dataset='view', view=v,
                                    X_tr_precomp_dists=dists_views[v],
                                    **kws)
    #############
    # Full MVMM #
    #############
    # print('start fitting full MVMM at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    start_time = time()
    models['full_mvmm'].fit(X_tr)

    runtimes['full_mvmm'] = time() - start_time
    res_writer.write('fitting full mvmm took {:1.2f} seconds'.
                     format(runtimes['full_mvmm']))

    results_df = add_gs_results(results_df=results_df,
                                model=models['full_mvmm'],
                                model_name='full_mvmm',
                                run_biptsp_on_full=True,
                                dataset='full', view='both',
                                X_tr_precomp_dists=dists_cat,
                                **kws)

    for v in range(n_views):
        # add MVMM results for this view
        results_df = add_gs_results(results_df=results_df,
                                    model=models['full_mvmm'],
                                    model_name='full_mvmm',
                                    dataset='view', view=v,
                                    X_tr_precomp_dists=dists_views[v],  # TODO is this what we want
                                    **kws)
    ################
    # log pen MVMM #
    ################

    if 'log_pen_mvmm' not in to_exclude:
        # print('start fitting log pen grid search MVMM at {}'.
        #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        start_time = time()
        models['log_pen_mvmm'].fit(X_tr)

        runtimes['log_pen_mvmm'] = time() - start_time
        res_writer.write('fitting grid search for log pen'
                         'mvmm took {:1.2f} seconds'.
                         format(runtimes['log_pen_mvmm']))

        results_df = add_gs_results(results_df=results_df,
                                    model=models['log_pen_mvmm'],
                                    model_name='log_pen_mvmm',
                                    dataset='full', view='both',
                                    X_tr_precomp_dists=dists_cat,
                                    **kws)

        for v in range(n_views):

            # add log pen MVMM results for this view
            results_df = add_gs_results(results_df=results_df,
                                        model=models['log_pen_mvmm'],
                                        model_name='log_pen_mvmm',
                                        dataset='view', view=v,
                                        X_tr_precomp_dists=dists_views[v],  # TODO: is this what we want
                                        **kws)
    #######################
    # block diagonal MVMM #
    #######################

    if 'bd_mvmm' not in to_exclude:
        # print('start fitting block diag grid search MVMM at {}'.
        #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        start_time = time()
        models['bd_mvmm'].fit(X_tr)

        runtimes['bd_mvmm'] = time() - start_time
        res_writer.write('fitting grid search for block'
                         'diag mvmm took {:1.2f} seconds'.
                         format(runtimes['bd_mvmm']))

        results_df = add_gs_results(results_df=results_df,
                                    model=models['bd_mvmm'],
                                    model_name='bd_mvmm',
                                    dataset='full', view='both',
                                    X_tr_precomp_dists=dists_cat,
                                    **kws)

        for v in range(n_views):
            # add bd MVMM results for this view
            results_df = add_gs_results(results_df=results_df,
                                        model=models['bd_mvmm'],
                                        model_name='bd_mvmm',
                                        dataset='view', view=v,
                                        X_tr_precomp_dists=dists_views[v],  # TODO: is this what we want
                                        **kws)

    #########################
    # spectral penalty MVMM #
    #########################
    if 'sp_mvmm' not in to_exclude:
        # print('start fitting spectral penalty MVMM at {}'.
        #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        start_time = time()
        models['sp_mvmm'].fit(X_tr)

        runtimes['sp_mvmm'] = time() - start_time
        res_writer.write('fitting grid search for spect pen'
                         'mvmm took {:1.2f} seconds'.
                         format(runtimes['sp_mvmm']))

        results_df = add_gs_results(results_df=results_df,
                                    model=models['sp_mvmm'],
                                    model_name='sp_mvmm',
                                    dataset='full', view='both',
                                    X_tr_precomp_dists=dists_cat,
                                    **kws)

        for v in range(n_views):
            # add sp MVMM results for this view
            results_df = add_gs_results(results_df=results_df,
                                        model=models['sp_mvmm'],
                                        model_name='sp_mvmm',
                                        dataset='view', view=v,
                                        X_tr_precomp_dists=dists_views[v],  # TODO: is this what we want
                                        **kws)

    ##########
    # oracle #
    ##########

    results_df = add_gs_results(results_df,
                                model=models['oracle'],
                                model_name='oracle',
                                run_biptsp_on_full=False,  # or true?
                                dataset='full', view='both',
                                **kws)

    # Some formatting
    # ensure these columns are saved as integers
    int_cols = ['mc_index', 'best_tuning_idx',
                'n_comp_est', 'n_comp_resid', 'n_comp_tot_est',
                'n_samples', 'tune_idx']
    results_df[int_cols] = results_df[int_cols].astype(int)

    # block diagonal summary of Pi estimates
    bd_summary = {}

    if 'log_pen_mvmm' not in to_exclude:
        _sim_stub = deepcopy(sim_stub)
        _sim_stub.update({'model_name': 'log_pen_mvmm'})
        bd_summary['log_pen_mvmm'] = \
            get_bd_summary_for_gs(_sim_stub, models['log_pen_mvmm'],
                                  zero_thresh=zero_thresh)

    if 'bd_mvmm' not in to_exclude:
        _sim_stub = deepcopy(sim_stub)
        _sim_stub.update({'model_name': 'bd_mvmm'})
        bd_summary['bd_mvmm'] = \
            get_bd_summary_for_gs(_sim_stub, models['bd_mvmm'],
                                  zero_thresh=zero_thresh)

    if 'sp_mvmm' not in to_exclude:
        _sim_stub = deepcopy(sim_stub)
        _sim_stub.update({'model_name': 'sp_mvmm'})
        bd_summary['sp_mvmm'] = \
            get_bd_summary_for_gs(_sim_stub, models['sp_mvmm'],
                                  zero_thresh=zero_thresh)

    res_writer.write("Entire simulation took {:1.2f} seconds".
                     format(time() - overall_start_time))

    tr_data = {'X_tr': X_tr, 'Y_tr': Y_tr}

    return results_df, clf_results, models, bd_summary, Pi_empirical, tr_data,\
        runtimes


def add_gs_results(results_df, sim_stub, model, model_name, dataset, view,
                   X_tr, Y_tr, X_tst, Y_tst,
                   Pi_true, view_params_true, zero_thresh=0,
                   run_biptsp_on_full=False, X_tr_precomp_dists=None):
    """
    Extracts results from after fitting model.

    Parameters
    ----------
    results_df: pd.DataFrame
        The dataframe containing the results

    sim_stub: dict
        Dict contatining simulation information.

    model:
        Grid search clustering model.

    model_name: str, [
        Name of model.

    dataset: str, ['full', 'view']
        Which dataset are we looking at -- the full dataset or only a single view.

    view: int, str
        Either 'both' or which view we are looking at.


    X_tr, y_tr, X_tst, y_tst

    """
    # idetifying information for results
    identif_stub = {'dataset': dataset, 'view': view, 'model': model_name}
    res = {}
    res.update(sim_stub)
    res.update(identif_stub)

    assert dataset in ['full', 'view']
    assert type(view) == int or view == 'both'
    # assert model_name in ['ts_gs_mvmm', 'gmm_cat'] or \
    #     'marginal_view' in model_name
    n_views = len(X_tr)
    n_samples = Y_tr.shape[0]

    # TODO
    is_mvmm_model = is_mvmm(model)
    is_gs_model = isinstance(model, BaseGridSearch) or \
        isinstance(model, SpectralPenSearchByBlockMVMM)
    # isinstance(model, SpectralPenSearchMVMM) or \

    # format X data
    if not is_mvmm_model and dataset == 'full':
        X_tr = np.hstack(X_tr)
        X_tst = np.hstack(X_tst)

    elif dataset == 'view':
        X_tr = X_tr[view]
        X_tst = X_tst[view]

    X_tr_precomp_dists = multi_view_safe_pairwise_distances(X=X_tr)

    model_sel_measures = ['aic', 'bic', 'silhouette',
                          'calinski_harabasz', 'davies_bouldin', 'dunn']
    # format y data
    y_tr_overall = view_labs_to_overall(Y_tr)
    y_tst_overall = view_labs_to_overall(Y_tst)
    # if dataset == 'full':
    #     y_tr = view_labs_to_overall(Y_tr)
    #     y_tst = view_labs_to_overall(Y_tst)
    # else:
    #     y_tr = Y_tr[:, view]
    #     y_tst = Y_tst[:, view]

    # get true number of components
    n_comp_tot_true, n_comp_views_true = get_n_comp(Pi_true)

    # true communities
    # comm_mat_true[np.isnan(comm_mat_true)] = -1
    comm_mat_true = get_comm_mat(Pi_true > 0)
    # res['n_blocks_true'] = int(np.nanmax(comm_mat_true) + 1)
    res['n_blocks_true'] = get_n_blocks(comm_mat_true)
    # comm_true_tr = [comm_mat_true[Y_tr[i, 0], Y_tr[i, 1]]
    #                 for i in range(Y_tr.shape[0])]
    # comm_true_tst = [comm_mat_true[Y_tst[i, 0], Y_tst[i, 1]]
    #                  for i in range(Y_tst.shape[0])]

    # add tuning parameter data
    if is_gs_model:
        n_tune_values = len(model.param_grid_)
        res['best_tuning_idx'] = model.best_idx_
        all_estimators = [model.estimators_[tune_idx]
                          for tune_idx in range(n_tune_values)]
    else:
        n_tune_values = 1
        res['best_tuning_idx'] = 0
        all_estimators = [model]

    # TODO-DEBUG
    # if res['best_tuning_idx'] is None:
    #     1/0

    # get results for every estimator
    for tune_idx in range(n_tune_values):

        #################
        # get model out #
        #################

        estimator = all_estimators[tune_idx]

        # get final est for two stage
        if isinstance(estimator, TwoStage):
            is_two_stage = True
            estimator = estimator.final_
            start_est = all_estimators[tune_idx].start_

        else:
            is_two_stage = False
            start_est = None

        # if it's a MVMM model and a marginal view dataset
        # get the view marginal model
        if is_mvmm_model and dataset == 'view':
            # for view datasets use view marginal part of MVMM
            estimator = estimator.view_models_[view]

            if is_two_stage:
                start_est = start_est.view_models_[view]

        ###############
        # Get results #
        ###############

        # bureaucracy

        res['tune_idx'] = int(tune_idx)

        # if grid search get the tuing param valeus
        if is_gs_model:
            res['tuning_param_values'] = model.param_grid_[tune_idx]
        else:
            res['tuning_param_values'] = None

        # total number of mixture components
        if isinstance(estimator, BlockDiagMVMM):
            n_comp_est = (estimator.bd_weights_ > estimator.zero_thresh).sum()
        else:
            n_comp_est = estimator.n_components

        res['n_comp_est'] = n_comp_est

        if dataset == 'full':
            n_comp_true = n_comp_tot_true
        else:
            n_comp_true = n_comp_views_true[view]
        res['n_comp_resid'] = n_comp_true - n_comp_est

        if is_mvmm_model and dataset == 'full':
            res['n_comp_tot_est'] = estimator.n_components
            res['n_comp_views_est'] = estimator.n_view_components

        else:
            if dataset == 'full':
                res['n_comp_tot_est'] = n_comp_est
                res['n_comp_views_est'] = [None for _ in range(n_views)]
            else:
                res['n_comp_tot_est'] = n_comp_est
                res['n_comp_views_est'] = [None for _ in range(n_views)]
                res['n_comp_views_est'][view] = n_comp_est

        # fit time
        if hasattr(estimator, 'metadata_'):
            res['fit_time'] = estimator.metadata_['fit_time']

        ###################
        # Fitting results #
        ###################

        # model fitting measures
        # res['bic'] = estimator.bic(X_tr)
        # res['aic'] = estimator.aic(X_tr)

        model_sel_scores = \
            unsupervised_cluster_scores(X=X_tr, estimator=estimator,
                                        measures=model_sel_measures,
                                        metric='precomputed',
                                        precomp_dists=X_tr_precomp_dists,
                                        dunn_kws={'diameter_method':
                                                  'farthest',
                                                  'cdist_method': 'nearest'})

        for k in model_sel_measures:
            res[k] = model_sel_scores[k]

        res['log_lik'] = estimator.log_likelihood(X_tr)
        res['n_params'] = estimator._n_parameters()

        # compute metrics on training data
        y_tr_pred = estimator.predict(X_tr)
        y_tst_pred = estimator.predict(X_tst)

        # compare to overall labeling
        res = add_to_res(res, cluster_report(y_tr_overall, y_tr_pred),
                         stub='train_overall')
        res = add_to_res(res, cluster_report(y_tst_overall, y_tst_pred),
                         stub='test_overall')

        # TODO remove these
        # compare overall labels to commmunity labels
        # TODO: not sure if this is valuable
        # res = add_cluster_report(res, y_tr_pred, comm_true_tr,
        #                          stub='train_overall_vs_community')
        # res = add_cluster_report(res, y_tst_pred, comm_true_tst,
        #                          stub='test_overall_vs_community')
        # res['train_ars_overall'] = adjusted_rand_score(y_tr_overall,
        #                                                y_tr_pred)

        # res['test_ars_overall'] = adjusted_rand_score(y_tst_overall,
        #                                               y_tst_pred)

        # compare to view specific labels
        for v in range(n_views):
            res = add_to_res(res, cluster_report(Y_tr[:, v], y_tr_pred),
                             stub='train_view_{}'.format(v))

            res = add_to_res(res, cluster_report(Y_tst[:, v], y_tst_pred),
                             stub='test_view_{}'.format(v))

            # res['train_ars_view_{}'.format(v)] = \
            #     adjusted_rand_score(Y_tr[:, v], y_tr_pred)

            # res['test_ars_view_{}'.format(v)] = \
            #     adjusted_rand_score(Y_tst[:, v], y_tst_pred)

        ############
        # TwoStage #
        ############
        if is_two_stage:

            y_tr_pred_start = start_est.predict(X_tr)
            y_tst_pred_start = start_est.predict(X_tst)
            res = add_to_res(res,
                             cluster_report(y_tr_overall, y_tr_pred_start),
                             stub='start_train_overall')
            res = add_to_res(res,
                             cluster_report(y_tst_overall, y_tst_pred_start),
                             stub='start_test_overall')

        # if MVMM on full data, get pi accuracy
        if is_mvmm_model and dataset == 'full':

            Pi_est = estimator.weights_mat_

            # get estimated community structure
            if is_block_diag_mvmm(estimator):
                # for bd_mvmm the communities come from D
                D = estimator.bd_weights_
                comm_mat_est = get_comm_mat(D > zero_thresh)
                res['n_blocks_est'] = get_n_blocks(comm_mat_est)

            elif type(model) == MVMM and run_biptsp_on_full:
                # for full MVMM run spectral bipartite partitioning
                # for true number of blocks
                n_blocks = res['n_blocks_true']
                comm_mat_est = run_bipt_spect_partitioning(Pi_est, n_blocks)

                res['n_blocks_est'] = np.nan

            else:
                # otherwise the communities come from just Pi_est
                comm_mat_est = get_comm_mat(Pi_est > zero_thresh)
                res['n_blocks_est'] = get_n_blocks(comm_mat_est)
                # int(np.nanmax(comm_mat_est) + 1)

            #####################
            # Pi graph accuracy #
            #####################
            # TODO: we might want to use D or a normalized D for these
            if is_block_diag_mvmm(estimator):
                # for bd_weights use a normalzied version of D
                # for graph accuracy
                Pi_est_tilde = D / D.sum()
            else:
                Pi_est_tilde = Pi_est

            res['pi_graph_acc_norm'] = \
                get_pi_acc(Pi_est_tilde, Pi_true,
                           method='random_walk',
                           normalize=True,
                           method_type='fast',
                           kernel_type='exponential')

            res['pi_graph_acc_unnorm'] = \
                get_pi_acc(Pi_est_tilde, Pi_true,
                           method='random_walk',
                           normalize=False,
                           method_type='fast',
                           kernel_type='exponential')

            #######################
            # Pi aligned accuracy #
            #######################
            means_true = [view_params_true[v]['means']
                          for v in range(n_views)]
            means_est = [estimator.view_models_[v].means_
                         for v in range(n_views)]

            res['pi_aligned_dist'] = \
                get_aligned_pi_distance(Pi_true, Pi_est, means_true, means_est)

            ######################
            # community accuracy #
            ######################

            # predicted communities
            # comm_mat_est = get_comm_mat(Pi_est > zero_thresh)
            # comm_mat_est[np.isnan(comm_mat_est)] = -1
            # res['n_blocks_est'] = int(np.nanmax(comm_mat_est) + 1)

            cl_report_no_out_tr, cl_report_restr_tr, n_out_tr = \
                get_comm_pred_summary(Pi_true=Pi_true, Y_true=Y_tr,
                                      mvmm=estimator, view_data=X_tr,
                                      comm_mat_est=comm_mat_est)

            cl_report_no_out_tst, cl_report_restr_tst, n_out_tst = \
                get_comm_pred_summary(Pi_true=Pi_true, Y_true=Y_tst,
                                      mvmm=estimator, view_data=X_tst,
                                      comm_mat_est=comm_mat_est)

            res = add_to_res(res, cl_report_no_out_tr,
                             stub='train_community')

            res = add_to_res(res, cl_report_no_out_tst,
                             stub='test_community')

            res = add_to_res(res, cl_report_restr_tst,
                             stub='test_community_restr')

            res['comm_est_train_n_out'] = n_out_tr
            res['comm_est_test_n_out'] = n_out_tst

            # comm_est_tr = []
            # comm_est_tst = []

            # for i in range(len(y_tr_pred)):
            #     y0, y1 = estimator._get_view_clust_idx(y_tr_pred[i])
            #     comm_est_tr.append(comm_mat_est[y0, y1])

            # for i in range(len(y_tst_pred)):
            #     y0, y1 = estimator._get_view_clust_idx(y_tst_pred[i])
            #     comm_est_tst.append(comm_mat_est[y0, y1])

            # res = add_cluster_report(res, comm_true_tr, comm_est_tr,
            #                          stub='train_community')
            # res = add_cluster_report(res, comm_true_tst, comm_est_tst,
            #                          stub='test_community')
            # res['train_ars_comm'] = adjusted_rand_score(comm_true_tr,
            #                                             comm_est_tr)
            # res['test_ars_comm'] = adjusted_rand_score(comm_true_tst,
            #                                            comm_est_tst)

            ############
            # TwoStage #
            ############
            if is_two_stage:

                Pi_est = start_est.weights_mat_

                # aligned accuracy
                means_true = [view_params_true[v]['means']
                              for v in range(n_views)]
                means_est = [start_est.view_models_[v].means_
                             for v in range(n_views)]
                res['start_pi_aligned_dist'] = \
                    get_aligned_pi_distance(Pi_true, Pi_est,
                                            means_true, means_est)

                # n_blocks
                comm_mat_est = get_comm_mat(Pi_est > zero_thresh)
                comm_mat_est[np.isnan(comm_mat_est)] = -1
                # res['start_n_blocks_est'] = int(np.nanmax(comm_mat_est) + 1)
                res['start_n_blocks_est'] = get_n_blocks(comm_mat_est)

        # to_add.append()
        results_df = results_df.append(res, ignore_index=True)

    # # do model selection for each measures
    # for sel_measure in model_sel_measures:
    #     if MEASURE_MIN_GOOD[sel_measure]:
    #         best_pd_idx = results_df[sel_measure].idxmin()
    #     else:
    #         best_pd_idx = results_df[sel_measure].idxmax()

    #     best_tune_idx = results_df.loc[best_pd_idx]['tune_idx']
    #     results_df['{}_best_tune_idx'.format(sel_measure)] = best_tune_idx

    # bic_best_pd_idx = results_df['bic'].idxmin()
    # bic_best_tune_idx = results_df.loc[bic_best_pd_idx]['tune_idx']
    # results_df['bic_best_tune_idx'] = bic_best_tune_idx

    return results_df


def get_pi_ests(model):
    n_tune_values = len(model.param_grid_)

    Pi_ests = []
    for tune_idx in range(n_tune_values):
        Pi_ests.append(model.estimators_[tune_idx].weights_mat_)
    return Pi_ests


def get_bd_summary_for_gs(sim_stub, gs_mvmm, zero_thresh=0):
    """
    Summary statistics of community structure of Pi for each
    estimator in a grid search
    """

    # is_spect_pen_cts = isinstance(gs_mvmm, SpectralPenSearchMVMM)
    is_spect_pen_cts = False  # TODO: just remove this
    is_spect_pen_block = isinstance(gs_mvmm, SpectralPenSearchByBlockMVMM)

    results_df = pd.DataFrame()
    for tune_idx, estimator in enumerate(gs_mvmm.estimators_):

        if isinstance(estimator, TwoStage):
            estimator = deepcopy(estimator.final_)

        res = deepcopy(sim_stub)
        res['tune_idx'] = tune_idx

        if is_spect_pen_cts:
            if isinstance(estimator, BlockDiagMVMM):
                res['tune__sp_pen'] = estimator.eval_pen_base
                Pi = estimator.bd_weights_
            else:
                res['tune__sp_pen'] = 0
                Pi = estimator.weights_mat_

        elif is_spect_pen_block:
            res['tune__n_blocks'] = gs_mvmm.est_n_blocks_[tune_idx]

            if isinstance(estimator, BlockDiagMVMM):
                Pi = estimator.bd_weights_
            else:
                Pi = estimator.weights_mat_

        elif isinstance(estimator, BlockDiagMVMM):
            res['tune__n_blocks'] = estimator.n_blocks
            Pi = estimator.bd_weights_

        elif isinstance(estimator, LogPenMVMM):
            res['tune__pen'] = estimator.pen
            # res['n_components'] = estimator.n_components
            Pi = estimator.weights_mat_

        # Pi = estimator.weights_mat_

        summary, _ = community_summary(Pi, zero_thresh=zero_thresh)
        res.update(summary)

        results_df = results_df.append(res, ignore_index=True)

    return results_df


def add_to_res(res, d, stub=None):
    if stub is not None:
        stub += '_'
    else:
        stub = ''

    for name in d.keys():
        res['{}{}'.format(stub, name)] = d[name]

    return res


# TODO: double check this!
def get_n_blocks(comm_mat):
    n_blocks = int(np.nanmax(comm_mat) + 1)
    if np.isnan(n_blocks):
        n_blocks = 1
    return n_blocks


def get_current_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
# def add_cluster_report(res, labels_true, labels_pred, stub=None):

#     if stub is not None:
#         stub += '_'
#     else:
#         stub = ''

#     report = cluster_report(labels_true, labels_pred)

#     for name in report.keys():
#         res['{}{}'.format(stub, name)] = report[name]

#     return res
