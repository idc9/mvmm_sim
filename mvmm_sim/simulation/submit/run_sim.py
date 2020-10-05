import numpy as np
import pandas as pd
from joblib import dump
import os

# from sklearn.metrics import adjusted_rand_score
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
from copy import deepcopy
from datetime import datetime
from warnings import simplefilter

from mvmm.multi_view.utils import view_labs_to_overall,\
    get_n_comp
from mvmm.multi_view.block_diag.graph.bipt_community import community_summary,\
    get_block_mat
from mvmm.utils import get_seeds
from mvmm.multi_view.TwoStage import TwoStage
from mvmm.BaseGridSearch import BaseGridSearch
from mvmm.multi_view.block_diag.graph.bipt_spect_partitioning import \
    run_bipt_spect_partitioning

from mvmm_sim.simulation.basic_sim_models import get_data_dist,\
    get_mvmm_log_pen_gs, get_gmm_gs, get_mvmm_block_diag_gs, get_full_mvmm
from mvmm_sim.simulation.aligned_pi_dist import get_aligned_pi_distance
from mvmm_sim.simulation.cluster_report import cluster_report
from mvmm_sim.simulation.utils import get_empirical_pi
from mvmm_sim.simulation.community_results import get_comm_pred_summary
from mvmm_sim.simulation.utils import is_mvmm, get_pi_acc, \
    clf_fit_and_score, get_n_comp_seq, is_block_diag_mvmm

from sklearn.exceptions import ConvergenceWarning


def run_sim_from_configs(clust_param_config, pi_dist, pi_config,
                         n_samples_tr, n_samples_tst,
                         view_gmm_config, cat_gmm_config, gmm_plus_minus,
                         mvmm_model, full_mvmm_config, base_gmm_config,
                         tune_config,
                         start_config, final_config, two_stage_config,
                         n_jobs=None,
                         data_seed=None, mc_index=None,
                         save_fpath=None):

    input_config = locals()

    print('Simulation starting at',
          datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    print(input_config)

    start_time = time()

    # cluster parameters are different for each MC iteration
    data_dist, Pi, view_params = \
        get_data_dist(clust_param_config=clust_param_config,
                      pi_dist=pi_dist, pi_config=pi_config)

    n_comp_tot, n_view_components = get_n_comp(Pi)
    n_blocks_true = community_summary(Pi)[0]['n_communities']

    ########
    # MVMM #
    ########

    # number of view components for MVMM
    mvmm_n_view_components = tune_config['n_view_components']
    if mvmm_n_view_components == 'true':
        mvmm_n_view_components = n_view_components

    # Full MVMM
    full_mvmm = get_full_mvmm(mvmm_n_view_components,
                              gmm_config=base_gmm_config,
                              config=full_mvmm_config)

    # Two stage estimators
    if mvmm_model == 'log_pen':

        ts_gs_mvmm = \
            get_mvmm_log_pen_gs(n_view_components=mvmm_n_view_components,
                                gmm_config=base_gmm_config,
                                full_config=start_config,
                                log_pen_config=final_config,
                                two_stage_config=two_stage_config,
                                mult_values=tune_config['mult_values'],
                                n_jobs=n_jobs)

    elif mvmm_model == 'block_diag':
        pm = int(tune_config['n_blocks_pm'])

        n_blocks_tune = np.arange(max(2, n_blocks_true - pm),
                                  n_blocks_true + pm + 1)
        ts_gs_mvmm = \
            get_mvmm_block_diag_gs(n_view_components=mvmm_n_view_components,
                                   gmm_config=base_gmm_config,
                                   full_config=start_config,
                                   bd_config=final_config,
                                   two_stage_config=two_stage_config,
                                   n_blocks=n_blocks_tune,
                                   n_jobs=n_jobs)

    #############
    # cat GMM #
    #############
    n_components_seq = get_n_comp_seq(n_comp_tot, gmm_plus_minus)
    cat_gmm = get_gmm_gs(n_components_seq,
                         gmm_config=cat_gmm_config,
                         n_jobs=n_jobs)

    #############
    # view GMMs #
    #############
    view_gmms = []
    for v in range(len(n_view_components)):
        n_components_seq = get_n_comp_seq(n_view_components[v], gmm_plus_minus)
        view_gmms.append(get_gmm_gs(n_components_seq,
                                    gmm_config=view_gmm_config,
                                    n_jobs=n_jobs))



    # classifier
    clf = LinearDiscriminantAnalysis()

    zero_thresh = .01 / (n_view_components[0] * n_view_components[1])

    # run simulation
    clust_results, clf_results, fit_models,\
        bd_results, Pi_empirical, runtimes = \
        run_sim(full_mvmm=full_mvmm, ts_gs_mvmm=ts_gs_mvmm, cat_gmm=cat_gmm,
                view_gmms=view_gmms, clf=clf,
                data_dist=data_dist, Pi=Pi, view_params=view_params,
                n_samples_tr=n_samples_tr, data_seed=data_seed,
                mc_index=mc_index, n_samples_tst=n_samples_tst,
                zero_thresh=zero_thresh)

    mvmm_param_grid = fit_models['ts_gs_mvmm'].param_grid_

    metadata = {'n_samples_tr': n_samples_tr,
                'n_comp_tot': n_comp_tot,
                'n_view_components': n_view_components,
                'config': input_config,
                'mvmm_param_grid': mvmm_param_grid,
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
              'bd_results': bd_results,
              'metadata': metadata,
              'config': input_config}, save_fpath)

        # save some extra data for one MC repition
        if mc_index == 0:
            save_dir = os.path.dirname(save_fpath)
            fpath = os.path.join(save_dir,
                                 'extra_data_mc_0__n_samples_{}'.
                                 format(n_samples_tr))
            dump({'Pi': Pi, 'fit_models': fit_models,
                  'Pi_empirical': Pi_empirical}, fpath)

    return {'clust_results': clust_results,
            'clf_results': clf_results,
            'bd_results': bd_results,
            'metadata': metadata,
            'fit_models': fit_models,
            'Pi': Pi,
            'view_params': view_params}


def run_sim(full_mvmm, ts_gs_mvmm, cat_gmm, view_gmms, clf,
            data_dist, Pi, view_params, n_samples_tr, data_seed,
            mc_index=None,
            n_samples_tst=2000, zero_thresh=0):

    """

    Parameters
    ----------
    ts_gs_mvmm:
        Mutli-view mixture model grid search.

    cat_gmm:
        GMM grid serach for concatonated data.

    view_gmms: list
        GMMs for each view.

    clf:
        Classifier for concatenated data.

    data_dist: callable(n_samples, seed)
        Function to generate data.

    n_samples_tr: int
        Number of training samples.

    data_seed: int
        Seed for sampling train/test observations.

    n_samples_tst: int
        Number of samples to get for test data.

    """

    seeds = get_seeds(random_state=data_seed, n_seeds=2)

    # sample data
    X_tr, Y_tr = data_dist(n_samples=n_samples_tr,
                           random_state=seeds[0])
    X_tst, Y_tst = data_dist(n_samples=n_samples_tst,
                             random_state=seeds[1])
    n_views = len(X_tr)

    Pi_empirical = get_empirical_pi(Y_tr, Pi.shape, scale='counts')

    runtimes = {}

    # get classification resuls
    clf_results = {}
    start_time = time()
    clf_results['cat'] = clf_fit_and_score(clone(clf),
                                           X_tr=np.hstack(X_tr),
                                           y_tr=view_labs_to_overall(Y_tr),
                                           X_tst=np.hstack(X_tst),
                                           y_tst=view_labs_to_overall(Y_tst))

    runtimes['clf_cat'] = time() - start_time

    for v in range(n_views):
        start_time = time()
        clf_results['view_{}'.format(v)] =\
            clf_fit_and_score(clone(clf),
                              X_tr=X_tr[v],
                              y_tr=Y_tr[:, v],
                              X_tst=X_tst[v],
                              y_tst=Y_tst[:, v])

        runtimes['clf_view_{}'.format(v)] = time() - start_time

    # fit clustering
    simplefilter('ignore', ConvergenceWarning)

    # print('start fitting full MVMM at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    start_time = time()
    full_mvmm.fit(X_tr)
    runtimes['full_mvmm'] = time() - start_time
    print('fitting full mvmm took {:1.2f} seconds'.
          format(runtimes['full_mvmm']))

    # print('start fitting Two stage grid search MVMM at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    start_time = time()
    ts_gs_mvmm.fit(X_tr)
    runtimes['ts_gs_mvmm'] = time() - start_time
    print('fitting grid search mvmm took {:1.2f} seconds'.
          format(runtimes['ts_gs_mvmm']))

    # print('start fitting cat-GMM at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    start_time = time()
    cat_gmm.fit(np.hstack(X_tr))
    runtimes['cat_gmm'] = time() - start_time
    print('fitting grid search cat-GMM took {:1.2f} seconds'.
          format(runtimes['cat_gmm']))

    # print('start fitting view marginal GMMs at {}'.
    #       format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    for v in range(n_views):
        start_time = time()
        view_gmms[v].fit(X_tr[v])
        runtimes['gmm_view_{}'.format(v)] = time() - start_time
        print('fitting marginal view {} GMM took {:1.2f} seconds'.
              format(v, runtimes['gmm_view_{}'.format(v)]))

    fit_models = {'full_mvmm': full_mvmm,
                  'ts_gs_mvmm': ts_gs_mvmm,
                  'cat_gmm': cat_gmm,
                  'view_gmms': view_gmms}

    results_df = pd.DataFrame()

    sim_stub = {'mc_index': mc_index, 'n_samples': n_samples_tr}

    kws = {'sim_stub': sim_stub,
           'X_tr': X_tr, 'Y_tr': Y_tr,
           'X_tst': X_tst, 'Y_tst': Y_tst,
           'Pi_true': Pi, 'view_params_true': view_params,
           'zero_thresh': zero_thresh}

    start_time = time()

    # add MVMM
    results_df = add_gs_results(results_df=results_df,
                                model=full_mvmm, model_name='full_mvmm',
                                dataset='full', view='both',
                                **kws)

    results_df = add_gs_results(results_df=results_df,
                                model=ts_gs_mvmm, model_name='ts_gs_mvmm',
                                dataset='full', view='both',
                                **kws)

    # pi_ests = get_pi_ests(mvmm)

    # add GMM on concat data
    results_df = add_gs_results(results_df=results_df,
                                model=cat_gmm, model_name='gmm_cat',
                                dataset='full', view='both',
                                **kws)

    for v in range(n_views):
        # add MVMM results for this view
        results_df = add_gs_results(results_df=results_df,
                                    model=full_mvmm, model_name='full_mvmm',
                                    dataset='view', view=v,
                                    **kws)

        # add MVMM results for this view
        results_df = add_gs_results(results_df=results_df,
                                    model=ts_gs_mvmm, model_name='ts_gs_mvmm',
                                    dataset='view', view=v,
                                    **kws)

        # gmm fit on this view
        results_df = add_gs_results(results_df=results_df,
                                    model=view_gmms[v],
                                    model_name='marginal_view_{}'.format(v),
                                    dataset='view', view=v,
                                    **kws)

    # ensure these columns are saved as integers
    int_cols = ['mc_index', 'best_tuning_idx',
                'n_comp_est', 'n_comp_resid', 'n_comp_tot_est',
                'n_samples', 'tune_idx']
    results_df[int_cols] = results_df[int_cols].astype(int)

    if is_block_diag_mvmm(ts_gs_mvmm):
        bd_results = get_bd_results(sim_stub, ts_gs_mvmm,
                                    zero_thresh=zero_thresh)
    else:
        bd_results = None

    print('getting the results took {:1.2f} seconds'.
          format(time() - start_time))

    return results_df, clf_results, fit_models, bd_results, Pi_empirical,\
        runtimes


def add_gs_results(results_df, sim_stub, model, model_name, dataset, view,
                   X_tr, Y_tr, X_tst, Y_tst,
                   Pi_true, view_params_true, zero_thresh=0):
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

    model_name: str, ['mvmm', 'gmm']
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
    is_gs_model = isinstance(model, BaseGridSearch)

    # format X data
    if not is_mvmm_model and dataset == 'full':
        X_tr = np.hstack(X_tr)
        X_tst = np.hstack(X_tst)

    elif dataset == 'view':
        X_tr = X_tr[view]
        X_tst = X_tst[view]

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
    comm_mat_true = get_block_mat(Pi_true > 0)
    res['true_n_communities'] = int(np.nanmax(comm_mat_true) + 1)
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

        res['tune_idx'] = tune_idx

        # if grid search get the tuing param valeus
        if is_gs_model:
            res['tuning_param_values'] = model.param_grid_[tune_idx]
        else:
            res['tuning_param_values'] = None

        # number of components this model is trying to estimate
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
        res['bic'] = estimator.bic(X_tr)
        res['aic'] = estimator.aic(X_tr)

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

            # get estimated Pi matrix
            if is_block_diag_mvmm(estimator):
                Pi_est = estimator.bd_weights_
                Pi_est /= Pi_est.sum()  # normalize
            else:
                Pi_est = estimator.weights_mat_

            # get community matrix for block diagonal matrix
            if is_two_stage:
                comm_mat_est = get_block_mat(Pi_est > zero_thresh)

                res['est_n_communities'] = int(np.nanmax(comm_mat_est) + 1)

            else:  # full MVMM
                # for full MVMM run spectral bipartite partitioning
                # for true number of blocks
                n_blocks = res['true_n_communities']
                comm_mat_est = run_bipt_spect_partitioning(Pi_est, n_blocks)

                res['est_n_communities'] = np.nan

            #####################
            # Pi graph accuracy #
            #####################
            res['pi_graph_acc_norm'] = \
                get_pi_acc(Pi_est, Pi_true,
                           method='random_walk',
                           normalize=True,
                           method_type='fast',
                           kernel_type='exponential')

            res['pi_graph_acc_unnorm'] = \
                get_pi_acc(Pi_est, Pi_true,
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
            # comm_mat_est = get_block_mat(Pi_est > zero_thresh)
            # comm_mat_est[np.isnan(comm_mat_est)] = -1
            # res['est_n_communities'] = int(np.nanmax(comm_mat_est) + 1)

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

                # n_communities
                comm_mat_est = get_block_mat(Pi_est > zero_thresh)
                comm_mat_est[np.isnan(comm_mat_est)] = -1
                res['start_est_n_communities'] = int(np.nanmax(comm_mat_est) + 1)

        results_df = results_df.append(res, ignore_index=True)

    return results_df


def get_pi_ests(model):
    n_tune_values = len(model.param_grid_)

    Pi_ests = []
    for tune_idx in range(n_tune_values):
        Pi_ests.append(model.estimators_[tune_idx].weights_mat_)
    return Pi_ests


def get_bd_results(sim_stub, mvmm_gs, zero_thresh=0):
    """
    Summary statistics of community structure of Pi
    """

    results_df = pd.DataFrame()
    for tune_idx, estimator in enumerate(mvmm_gs.estimators_):

        if isinstance(estimator, TwoStage):
            estimator = deepcopy(estimator.final_)

        res = deepcopy(sim_stub)
        res['tune_idx'] = tune_idx
        res['n_blocks'] = estimator.n_blocks

        # Pi = estimator.weights_mat_
        Pi = estimator.bd_weights_
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


# def add_cluster_report(res, labels_true, labels_pred, stub=None):

#     if stub is not None:
#         stub += '_'
#     else:
#         stub = ''

#     report = cluster_report(labels_true, labels_pred)

#     for name in report.keys():
#         res['{}{}'.format(stub, name)] = report[name]

#     return res
