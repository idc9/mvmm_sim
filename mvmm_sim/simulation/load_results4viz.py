from joblib import load
import os
import pandas as pd
import numpy as np

from mvmm.multi_view.block_diag.graph.bipt_community import community_summary

from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.simulation.sim_viz import get_best_tune_expers
from mvmm_sim.simulation.utils import extract_tuning_param_vals
from mvmm.clustering_measures import MEASURE_MIN_GOOD

# TODO: change est_n_communities to n_blocks_est


def load_results(sim_name, select_metric='bic'):
    """
    Parameters
    -----------

    select_metric: str
        Used to pick the best model if there are ties for log_pen_mvmm_at_truth.
    """

    results = load(os.path.join(Paths().out_data_dir, sim_name,
                   'simulation_results'))

    extra_data = load(os.path.join(Paths().out_data_dir, sim_name,
                      'extra_data_mc_0'))

    # clustering results
    clust_results = results['clust_results']
    clust_results = clust_results.reset_index(drop=True)

    # TODO: drop this
    # clust_results = clust_results.\
    #     rename(columns={'est_n_communities': 'n_blocks_est',
    #                     'true_n_communities': 'n_blocks_true'})

    int_cols = ['mc_index', 'best_tuning_idx', 'n_comp_est',
                'n_comp_resid', 'n_comp_tot_est', 'n_samples', 'tune_idx']
    clust_results[int_cols] = clust_results[int_cols].astype(int)

    # need to do these below  because of nans
    bd_int_cols = ['n_blocks_est', 'n_blocks_true']

    # block diagonal summary
    bd_summary = results['bd_summary']
    int_cols = ['n_nonzero_entries', 'n_communities', 'tune_idx',
                'n_samples', 'mc_index']
# 'n_connected_rows', 'n_connected_cols']

    bd_summary[int_cols] = bd_summary[int_cols].astype(int)

    log_pen_bd_summary = bd_summary.\
        query("model_name == 'log_pen_mvmm'").\
        set_index(['n_samples', 'mc_index', 'tune_idx'])

    bd_mvmm_bd_summary = bd_summary.\
        query("model_name == 'bd_mvmm'").\
        set_index(['n_samples', 'mc_index', 'tune_idx'])

    # TODO: uncomment after rerunning
    sp_mvmm_bd_summary = bd_summary.\
        query("model_name == 'sp_mvmm'").\
        set_index(['n_samples', 'mc_index', 'tune_idx'])

    n_samples_tr_seq = results['sim_metadata']['n_samples_tr_seq']

    # # TODO: get this from results when implemented
    # zero_thresh = .1 / (Pi_true.shape[0] * Pi_true.shape[1])
    zero_thresh = results['metadata'][0]['zero_thresh']

    # true number of components
    n_comp_tot_true = results['metadata'][0]['n_comp_tot']
    n_comp_views_true = results['metadata'][0]['n_view_components']

    fit_models = extra_data['fit_models']

    data = {}
    data['X_tr'] = extra_data['X_tr']
    data['Y_tr'] = extra_data['Y_tr']
    data['view_params'] = extra_data['view_params']
    data['Pi_true'] = extra_data['Pi']

    Pi_true = extra_data['Pi']
    true_summary, _ = community_summary(Pi_true)
    n_blocks_true = true_summary['n_communities']

    pi_true_summary = {'n_comp_tot_true': n_comp_tot_true,
                       'n_comp_views_true': n_comp_views_true,
                       'n_blocks_true': n_blocks_true,
                       'Pi': Pi_true,
                       'summary': true_summary}

    # sim_summary = get_sim_summary(results)

    models2exclude = []

    ###########
    # Log pen #
    ###########

    log_pen_mvmm_df = clust_results.\
        query("model == 'log_pen_mvmm' & dataset == 'full' & view == 'both'")

    if log_pen_mvmm_df.shape[0] == 0:
        models2exclude.append('log_pen_mvmm')

    if 'log_pen_mvmm' not in models2exclude:

        log_pen_mvmm_df = add_model_selection_by_measures(log_pen_mvmm_df)

        log_pen_mvmm_df[bd_int_cols] = log_pen_mvmm_df[bd_int_cols].astype(int)

        log_pen_mvmm_df = log_pen_mvmm_df.set_index(['n_samples',
                                                     'mc_index', 'tune_idx'])

        log_pen_mvmm_df = pd.concat([log_pen_mvmm_df,
                                     log_pen_bd_summary], axis=1).reset_index()

        vals, param_name = extract_tuning_param_vals(log_pen_mvmm_df)
        log_pen_mvmm_df['tune__' + param_name] = vals

        # log_pen_mvmm_df_all = log_pen_mvmm_df.copy()

        # TODO: do we want this?
        # for lambd values which give the same n_est_comp,
        # keep only the best one
        # log_pen_mvmm_df = get_best_tune_expers(log_pen_mvmm_df,
        #                                        by='n_comp_est',
        #                                        measure=select_metric,
        #                                        min_good=True)

    ###########
    # BD MVMM #
    ###########

    bd_mvmm_df = clust_results.\
        query("model == 'bd_mvmm' & dataset == 'full' & view == 'both'")

    if bd_mvmm_df.shape[0] == 0:
        models2exclude.append('bd_mvmm')

    if 'bd_mvmm' not in models2exclude:

        bd_mvmm_df = add_model_selection_by_measures(bd_mvmm_df)

        bd_mvmm_df[bd_int_cols] = bd_mvmm_df[bd_int_cols].astype(int)

        bd_mvmm_df['n_blocks_req'] = \
            bd_mvmm_df.loc[:, 'tuning_param_values'].\
            apply(lambda x: x['n_blocks'])

        bd_mvmm_df['n_blocks_req'] = bd_mvmm_df['n_blocks_req'].astype(int)

        bd_mvmm_df = bd_mvmm_df.set_index(['n_samples', 'mc_index',
                                           'tune_idx'])

        bd_mvmm_df = pd.concat([bd_mvmm_df,
                                bd_mvmm_bd_summary], axis=1).reset_index()

        # TODO: this is a hack -- come up with a better solution
        # bd_mvmm_df['n_comp_est'] = bd_mvmm_df['n_nonzero_entries']

        vals, param_name = extract_tuning_param_vals(bd_mvmm_df)
        bd_mvmm_df['tune__' + param_name] = vals

    ####################
    # spetral pen MVMM #
    ####################
    sp_mvmm_df = clust_results.\
        query("model == 'sp_mvmm' & dataset == 'full' & view == 'both'")

    if sp_mvmm_df.shape[0] == 0:
        models2exclude.append('sp_mvmm')

    if 'sp_mvmm' not in models2exclude:

        sp_mvmm_df = add_model_selection_by_measures(sp_mvmm_df)

        sp_mvmm_df[bd_int_cols] = sp_mvmm_df[bd_int_cols].astype(int)

        sp_mvmm_df = sp_mvmm_df.set_index(['n_samples', 'mc_index',
                                           'tune_idx'])

        sp_mvmm_df = pd.concat([sp_mvmm_df,
                                sp_mvmm_bd_summary], axis=1).reset_index()

        # sp_mvmm_df['n_comp_est'] = sp_mvmm_df['n_nonzero_entries'].astype(int)

        vals, param_name = extract_tuning_param_vals(sp_mvmm_df)
        sp_mvmm_df['tune__' + param_name] = vals

        # sp_mvmm_df_all = sp_mvmm_df.copy()

        # TODO: I don't think we want this
        # sp_mvmm_df = get_best_tune_expers(sp_mvmm_df, by='n_blocks_est',
        #                                   measure=select_metric,
        #                                   min_good=True)

    #################
    # others models #
    #################
    full_df = clust_results.\
        query("model == 'full_mvmm' & dataset == 'full' & view == 'both'")

    full_df = add_model_selection_by_measures(full_df)

    cat_gmm_df = clust_results.\
        query("model == 'gmm_cat' & dataset == 'full' & view == 'both'")

    cat_gmm_df = add_model_selection_by_measures(cat_gmm_df)

    view_0_gmm_df = clust_results.\
        query("model == 'marginal_view_0' & dataset == 'view' & view == 0")

    view_1_gmm_df = clust_results.\
        query("model == 'marginal_view_1' & dataset == 'view' & view == 1")

    log_pen_view_0_df = clust_results.\
        query("model == 'log_pen_mvmm' & dataset == 'view' & view == 0")

    log_pen_view_1_df = clust_results.\
        query("model == 'log_pen_mvmm' & dataset == 'view' & view == 1")

    bd_mvmm_view_0_df = clust_results.\
        query("model == 'bd_mvmm' & dataset == 'view' & view == 0")

    bd_mvmm_view_1_df = clust_results.\
        query("model == 'bd_mvmm' & dataset == 'view' & view == 1")

    sp_mvmm_view_0_df = clust_results.\
        query("model == 'sp_mvmm' & dataset == 'view' & view == 0")

    sp_mvmm_view_1_df = clust_results.\
        query("model == 'sp_mvmm' & dataset == 'view' & view == 1")

    #######################################
    # results at true parameter settings #
    ######################################
    cat_gmm_df_at_truth = cat_gmm_df. \
        query("n_comp_tot_est == {}".format(n_comp_tot_true))

    log_pen_mvmm_df_at_truth = log_pen_mvmm_df.\
        query("n_comp_est == {}".format(n_comp_tot_true))

    log_pen_mvmm_df_at_truth = get_best_tune_expers(log_pen_mvmm_df_at_truth,
                                                    by='n_comp_est',
                                                    measure=select_metric)

    bd_mvmm_df_at_truth = bd_mvmm_df.\
        query("n_blocks_est == {}".format(n_blocks_true))

    sp_mvmm_df_at_truth = get_best_expers_at_truthish(sp_mvmm_df,
                                                      group_var='n_blocks_est',
                                                      true_val=n_blocks_true)

    # break ties with BIC best
    # this is not needed for sp by block
    # sp_mvmm_df_at_truth = get_best_tune_expers(sp_mvmm_df_at_truth,
    #                                            by='n_blocks_est',
    #                                            measure=select_metric)

    # TODO: subset out to best BIC of these
    # TODO: what if no one gives truth?
    # sp_mvmm_df_at_truth = sp_mvmm_df.\
    #     query("n_blocks_est == {}".format(n_blocks_true))

    model_dfs = {'cat_gmm': cat_gmm_df,
                 'full_mvmm': full_df,
                 'log_pen_mvmm': log_pen_mvmm_df,
                 'bd_mvmm': bd_mvmm_df,
                 'sp_mvmm': sp_mvmm_df,
                 'view_0_gmm': view_0_gmm_df,
                 'view_1_gmm': view_1_gmm_df,
                 'log_pen_view_0': log_pen_view_0_df,
                 'log_pen_view_1': log_pen_view_1_df,
                 'bd_mvmm_view_0': bd_mvmm_view_0_df,
                 'bd_mvmm_view_1': bd_mvmm_view_1_df,
                 'sp_mvmm_view_0_df': sp_mvmm_view_0_df,
                 'sp_mvmm_view_1_df': sp_mvmm_view_1_df}

    # TODO: get rid of this
    # model_dfs_all_tune_vals = {'log_pen_mvmm': log_pen_mvmm_df_all,
    #                            'sp_mvmm': sp_mvmm_df_all}

    model_dfs_at_truth = {'full_mvmm': full_df,
                          'cat_gmm': cat_gmm_df_at_truth,
                          'log_pen_mvmm': log_pen_mvmm_df_at_truth,
                          'bd_mvmm': bd_mvmm_df_at_truth,
                          'sp_mvmm': sp_mvmm_df_at_truth}  # TODO

    return results, model_dfs, model_dfs_at_truth, \
        fit_models, pi_true_summary, \
        n_samples_tr_seq, zero_thresh, data


def get_best_expers_at_truthish(df, group_var, true_val):
    trueish_df = []
    for _, exper_df in df.groupby(['n_samples', 'mc_index']):

        all_vals = np.array(list(set(exper_df[group_var])))

        if true_val in all_vals:
            df_subset = exper_df.query("{} == @true_val".format(group_var))

        else:
            diffs = abs(all_vals - true_val)
            best_diff = min(diffs)
            vals_to_get = all_vals[diffs == best_diff]
            df_subset = exper_df.query("{} in @vals_to_get".format(group_var))

        best_idx = df_subset['bic'].idxmin()
        true_ish_exper = df.loc[best_idx].to_dict()
        trueish_df.append(true_ish_exper)

    trueish_df = pd.DataFrame(trueish_df)
    return trueish_df


def get_timing_data(results, fit_models):
    # model_names = list(results['metadata'][0]['fit_runtimes'].keys())

    n_sims = len(results['metadata'])
    data = []

    for sim_idx in range(n_sims):
        res = results['metadata'][sim_idx]

        dat = {}
        dat['sim_idx'] = sim_idx

        dat['tot_runtime'] = res['tot_runtime'] / (60 * 60)
        dat['n_samples_tr'] = res['n_samples_tr']
        # res['mc_index'] = results['metadata'][sim_idx]['mc_index']

        # runtime for each model
        for model, time in res['fit_runtimes'].items():
            dat[model + '__runtime'] = time / (60 * 60)

        data.append(dat)

    data = pd.DataFrame(data)
    data['n_init'] = results['metadata'][sim_idx]['args'].n_init

    # n fit models
    # fm = extra_data['fit_models'][2000]
    fm = list(fit_models.values())[0]
    data['cat_gmm__n_fit_models'] = len(fm['cat_gmm'].param_grid_)
    data['full_mvmm__n_fit_models'] = 1
    data['gmm_view_0__n_fit_models'] = 1
    data['gmm_view_1__n_fit_models'] = 1
    data['log_pen_mvmm__n_fit_models'] = len(fm['log_pen_mvmm'].param_grid_)
    data['bd_mvmm__n_fit_models'] = len(fm['bd_mvmm'].param_grid_)
    data['sp_mvmm__n_fit_models'] = fm['sp_mvmm'].n_pen_seq

    return data


# def add_bic_aic_sel(model_df):
#     for _, df in model_df.groupby(['n_samples', 'mc_index']):
#         for ic in ['bic', 'aic']:

#             best_pd_idx = df[ic].idxmin()
#             best_tune_idx = df.loc[best_pd_idx]['tune_idx']
#             model_df.loc[df.index, ic + '_best_tune_idx'] = best_tune_idx

#     return model_df


def add_model_selection_by_measures(df):
    # make sure only one experiment is in this df
    assert len(np.unique(df['model'])) == 1
    assert len(np.unique(df['dataset'])) == 1
    assert len(np.unique(df['view'])) == 1

    # get the select metric measures in this df
    all_model_sel_measures = ['aic', 'bic', 'silhouette',
                              'calinski_harabasz', 'davies_bouldin', 'dunn']
    model_sel_measures = []
    for select_metric in all_model_sel_measures:
        if select_metric in df.columns:
            model_sel_measures.append(select_metric)

    # do model selection for each measure
    for sel_measure in model_sel_measures:
        if MEASURE_MIN_GOOD[sel_measure]:
            best_pd_idx = df[sel_measure].idxmin()
        else:
            best_pd_idx = df[sel_measure].idxmax()

        best_tune_idx = df.loc[best_pd_idx]['tune_idx']
        df['{}_best_tune_idx'.format(sel_measure)] = best_tune_idx

    return df
