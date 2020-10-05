import pandas as pd

from mvmm.multi_view.block_diag.graph.bipt_community import get_block_mat
from mvmm.clustering_measures import MEASURE_MIN_GOOD

from mvmm_sim.simulation.run_sim import get_n_blocks


def get_bd_mvmm_model_sel(mvmm_results, select_metric='bic',
                          user_best_idx=None):
    """
    Parameters
    ----------
    mvmm_results
    select_metric: str
        Which metric to use for model selection.

    user_best_idx: None, int
        User provided index for which model to select. May be used to overwrite
        best BIC selected index.

    """
    fit_data = mvmm_results['fit_data']
    n_view_comp_seq = mvmm_results['n_view_components']

    # n_view_comp_seq = mvmm_results['n_view_components']
    n_view_comp_settings = len(fit_data)  # len(n_view_comp_seq)

    model_sel_df = []
    for view_comp_idx in range(n_view_comp_settings):

        data = fit_data[view_comp_idx]
        bd_models = data['models']['bd_mvmm']

        if 'full_mvmm' in data['models'].keys():
            full_model = data['models']['full_mvmm']
        else:
            full_model = None

        model_scores_measures = bd_models.model_sel_scores_.columns.values

        #############
        # BD models #
        #############

        # bd_scores = bd_models.scores_
        # for bd_idx, row in  bd_scores.iterrows():
        for bd_idx in range(len(bd_models.estimators_)):
            est = bd_models.estimators_[bd_idx]
            model_sel_scores = bd_models.model_sel_scores_.iloc[bd_idx]

            # est n_blocks
            D_est = est.final_.bd_weights_
            zero_thresh = est.final_.zero_thresh
            comm_mat_est = get_block_mat(D_est > zero_thresh)
            n_blocks_est = get_n_blocks(comm_mat_est)

            n_comp_est = (D_est > zero_thresh).sum()

            # requested n blocks
            n_blocks_req = est.final_.n_blocks

            res = {'model': 'bd_' + str(bd_idx),
                   'view_comp_idx': view_comp_idx,
                   # 'bic': scores['bic'],
                   # 'aic': scores['aic'],
                   'n_blocks_est': n_blocks_est,
                   'n_blocks_req': n_blocks_req,
                   'n_comp_est': n_comp_est,
                   'n_view_comp': n_view_comp_seq[view_comp_idx]}

            for measure in model_scores_measures:
                res[measure] = model_sel_scores[measure]

            model_sel_df.append(res)

        #############
        # Full mvmm #
        #############

        if full_model is not None:
            comm_mat_est = get_block_mat(full_model.weights_mat_ > zero_thresh)
            n_blocks_est = get_n_blocks(comm_mat_est)

            n_comp_est = (full_model.weights_mat_ > zero_thresh).sum()

            res = {'model': 'full',
                   'view_comp_idx': view_comp_idx,
                   # 'bic': data['full_model_sel_scores']['bic'],
                   'n_blocks_req': 1,
                   'n_blocks_est': n_blocks_est,
                   'n_comp_est': n_comp_est,
                   'n_view_comp': n_view_comp_seq[view_comp_idx]
                   }

            for measure in model_scores_measures:
                res[measure] = data['full_model_sel_scores'][measure]

            model_sel_df.append(res)

    model_sel_df = pd.DataFrame(model_sel_df)

    # get best model
    if user_best_idx is None:
        # select with BIC/AIC
        if MEASURE_MIN_GOOD[measure]:
            best_idx = model_sel_df[select_metric].idxmin()
        else:
            best_idx = model_sel_df[select_metric].idxmax()
    else:
        # user provided
        best_idx = int(user_best_idx)

    best_model_name = model_sel_df.loc[best_idx]['model']
    best_view_comp_idx = model_sel_df.loc[best_idx]['view_comp_idx']

    best_vc_models = fit_data[best_view_comp_idx]['models']
    if 'bd' in best_model_name:
        bd_idx = int(best_model_name.split('_')[1])
        best_model = best_vc_models['bd_mvmm'].estimators_[bd_idx]
    else:
        best_model = best_vc_models['full_mvmm']

    # if isinstance(best_model, TwoStage):
    #     best_model = best_model.final_

    sel_metadata = {'idx': best_idx,
                    'model_name': best_model_name,
                    'best_view_comp_idx': view_comp_idx}

    return best_model, model_sel_df, sel_metadata


def get_log_pen_mvmm_model_sel(mvmm_results, select_metric='bic',
                               user_best_idx=None):
    """
    Parameters
    ----------
    mvmm_results
    select_metric: str
        Which metric to use for model selection.

    user_best_idx: None, int
        User provided index for which model to select. May be used to overwrite
        best BIC selected index.

    """
    fit_data = mvmm_results['fit_data']
    n_view_comp_seq = mvmm_results['n_view_components']

    # n_view_comp_seq = mvmm_results['n_view_components']
    n_view_comp_settings = len(fit_data)  # len(n_view_comp_seq)

    model_sel_df = []
    for view_comp_idx in range(n_view_comp_settings):

        data = fit_data[view_comp_idx]
        log_pen_models = data['models']['log_pen_mvmm']

        if 'full_mvmm' in data['models'].keys():
            full_model = data['models']['full_mvmm']

        model_scores_measures = log_pen_models.model_sel_scores_.columns.values

        #############
        # BD models #
        #############

        # bd_scores = bd_models.scores_
        # for bd_idx, row in  bd_scores.iterrows():
        for tune_idx in range(len(log_pen_models.estimators_)):
            est = log_pen_models.estimators_[tune_idx]
            model_sel_scores = log_pen_models.model_sel_scores_.iloc[tune_idx]

            Pi_est = est.final_.weights_mat_
            zero_thresh = 0
            comm_mat_est = get_block_mat(Pi_est > zero_thresh)
            n_blocks_est = get_n_blocks(comm_mat_est)

            n_comp_est = (Pi_est > zero_thresh).sum()

            res = {'model': 'logpen_' + str(tune_idx),
                   'view_comp_idx': view_comp_idx,
                   # 'bic': scores['bic'],
                   # 'aic': scores['aic'],
                   'n_blocks_est': n_blocks_est,
                   # 'n_blocks_req': n_blocks_req,
                   'n_comp_est': n_comp_est,
                   'n_view_comp': n_view_comp_seq[view_comp_idx],
                   'n_view_comp_est': list(Pi_est.shape)}

            for measure in model_scores_measures:
                res[measure] = model_sel_scores[measure]

            model_sel_df.append(res)

        #############
        # Full mvmm #
        #############

        if full_model is not None:
            comm_mat_est = get_block_mat(full_model.weights_mat_ > zero_thresh)
            n_blocks_est = get_n_blocks(comm_mat_est)

            n_comp_est = (full_model.weights_mat_ > zero_thresh).sum()

            res = {'model': 'full',
                   'view_comp_idx': view_comp_idx,
                   # 'bic': data['full_model_sel_scores']['bic'],
                   'n_blocks_est': n_blocks_est,
                   'n_comp_est': n_comp_est,
                   'n_view_comp': n_view_comp_seq[view_comp_idx]
                   }

            for measure in model_scores_measures:
                res[measure] = data['full_model_sel_scores'][measure]

            model_sel_df.append(res)

    model_sel_df = pd.DataFrame(model_sel_df)

    # get best model
    if user_best_idx is None:
        # select with BIC/AIC
        if MEASURE_MIN_GOOD[measure]:
            best_idx = model_sel_df[select_metric].idxmin()
        else:
            best_idx = model_sel_df[select_metric].idxmax()
    else:
        # user provided
        best_idx = int(user_best_idx)

    best_model_name = model_sel_df.loc[best_idx]['model']
    best_view_comp_idx = model_sel_df.loc[best_idx]['view_comp_idx']

    best_vc_models = fit_data[best_view_comp_idx]['models']
    if 'logpen' in best_model_name:
        bd_idx = int(best_model_name.split('_')[1])
        best_model = best_vc_models['log_pen_mvmm'].estimators_[bd_idx]
    else:
        best_model = best_vc_models['full_mvmm']

    # if isinstance(best_model, TwoStage):
    #     best_model = best_model.final_

    sel_metadata = {'idx': best_idx,
                    'model_name': best_model_name,
                    'best_view_comp_idx': view_comp_idx}

    return best_model, model_sel_df, sel_metadata
