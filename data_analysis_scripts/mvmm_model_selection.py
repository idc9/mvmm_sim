import matplotlib.pyplot as plt
from joblib import load, dump
import os
import argparse
from os.path import join

from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.data_analysis.multi_view.model_selection import \
    get_bd_mvmm_model_sel, get_log_pen_mvmm_model_sel
from mvmm_sim.data_analysis.multi_view.viz_resuls import plot_mvmm, \
    plot_bd_mvmm, plot_log_pen_mvmm, plot_mvmm_model_selection

# from mvmm.multi_view.TwoStage import TwoStage

from mvmm_sim.data_analysis.utils import load_fitted_mvmms
# from mvmm.simulation.opt_viz import plot_loss_history
from mvmm_sim.simulation.utils import make_and_get_dir

inches = 8

parser = argparse.\
    ArgumentParser(description='Model selection results.')
# parser.add_argument('--sim_name', type=str,
#                     help='Which simulation to run.')
parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

parser.add_argument('--user_bd_mvmm_best_idx', default=None,
                    help='Optional user provided index for'
                         'best bd MVMM model.')

parser.add_argument('--user_log_pen_mvmm_best_idx', default=None,
                    help='Optional user provided index for'
                         'best log pen MVMM model.')

parser.add_argument('--select_metric', type=str, default='bic',
                    help='Model selection criterion.')
args = parser.parse_args()


results_dir = args.results_dir
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting')
mv_fitting_dir = make_and_get_dir(fitting_dir, 'multi_view')
model_sel_dir = make_and_get_dir(results_dir, 'model_selection')
# bd_sel_dir = make_and_get_dir(model_sel_dir, 'bd_mvmm')
# log_sel_dir = make_and_get_dir(model_sel_dir, 'log_pen_mvmm')
bd_sel_dir = make_and_get_dir(model_sel_dir)
log_sel_dir = make_and_get_dir(model_sel_dir)
opt_diag_dir = make_and_get_dir(results_dir, 'opt_diagnostics')


res_writer = ResultsWriter(os.path.join(log_dir, 'mvmm_model_selection.txt'),
                           delete_if_exists=True)


res_writer.write('user_bd_mvmm_best_idx: {}'.
                 format(args.user_bd_mvmm_best_idx))
res_writer.write('user_log_pen_mvmm_best_idx: {}'.
                 format(args.user_log_pen_mvmm_best_idx))
res_writer.write('select_metric: {}'.format(args.select_metric))

# data = load(os.path.join(save_dir, 'multi_view_fit_data'))

#############
# load data #
#############

mvmm_results = load_fitted_mvmms(fitting_dir)
dataset_names = mvmm_results['dataset_names']

# possilby load existing selected models
sel_models_fpath = os.path.join(fitting_dir, 'selected_models')
sel_models = {}


include_bd_mvmm = 'bd_mvmm' in mvmm_results['fit_data'][0]['models'].keys()
include_lp_mvmm = 'log_pen_mgmm' in mvmm_results['fit_data'][0]['models'].keys()

####################
# bd mvmm  results #
####################
# TODO new MVMM loading from mv_fitting_dir

if include_bd_mvmm:
    estimator, mvmm_sel_df, sel_metadata = \
        get_bd_mvmm_model_sel(mvmm_results, select_metric=args.select_metric,
                              user_best_idx=args.user_bd_mvmm_best_idx)

    sel_models['bd_mvmm'] = estimator

    # model selection
    best_row = mvmm_sel_df.loc[sel_metadata['idx']]

    res_writer.write(dataset_names)
    res_writer.write("MVMM estimate: n_blocks requested {} (estimated {}), "
                     "n_view_components: {}".
                     format(best_row['n_blocks_req'],
                            best_row['n_blocks_est'],
                            best_row['n_view_comp']))

    mvmm_sel_df.to_csv(os.path.join(bd_sel_dir, 'bd_mvmm_model_sel.csv'))

    # metrics2compute = mvmm_results['fit_data'][0]['models']['bd_mvmm'].metrics2compute
    metrics2compute = [args.select_metric]

    for metric in metrics2compute:
        plt.figure(figsize=(inches, inches))
        # plot_bd_mvmm_model_sel(mvmm_sel_df, select_metric=metric)
        plot_mvmm_model_selection(mvmm_sel_df, group_var='n_blocks_est',
                                  group_var_label="Number of blocks",
                                  select_metric=metric)
        # plt.title('Multiview models')
        save_fig(join(bd_sel_dir, 'bd_mvmm_model_sel_{}.png'.format(metric)))

    # fitting history
    # if isinstance(estimator, TwoStage):

    if estimator.start_.max_n_steps > 0:
        bd_start_save_dir = make_and_get_dir(opt_diag_dir,
                                             'bd_mvmm', 'start')
        bd_final_save_dir = make_and_get_dir(opt_diag_dir,
                                             'bd_mvmm', 'final')

        plot_mvmm(estimator.start_, inches=inches,
                  save_dir=bd_start_save_dir)

    else:
        bd_final_save_dir = make_and_get_dir(opt_diag_dir, 'bd_mvmm')

    plot_bd_mvmm(estimator.final_, inches=inches,
                 save_dir=bd_final_save_dir)
    # plot_bd_mvmm_opt_history(estimator=estimator, save_dir=opt_diag_dir)

    # else:
    #     bd_save_dir = make_and_get_dir(opt_diag_dir, 'bd_mvmm')
    #     plot_mvmm(estimator, inches=inches, save_dir=bd_save_dir)

#########################
# log pen mvmm  results #
#########################

if include_lp_mvmm:

    estimator, mvmm_sel_df, sel_metadata = \
        get_log_pen_mvmm_model_sel(mvmm_results,
                                   select_metric=args.select_metric,
                                   user_best_idx=args.user_log_pen_mvmm_best_idx)

    sel_models['log_pen_mvmm'] = estimator

    # model selection
    best_row = mvmm_sel_df.loc[sel_metadata['idx']]

    res_writer.write(dataset_names)
    res_writer.write("Log pen estimate: n_comp {}, n_blocks estimated {}, "
                     "n_view_components: {}".
                     format(best_row['n_comp_est'],
                            best_row['n_blocks_est'],
                            best_row['n_view_comp']))

    mvmm_sel_df.to_csv(join(log_sel_dir, 'log_pen_mvmm_model_sel.csv'))

    # metrics2compute = mvmm_results['fit_data'][0]['models']['log_pen_mvmm'].metrics2compute

    metrics2compute = [args.select_metric]

    for metric in metrics2compute:
        plt.figure(figsize=(inches, inches))
        # plot_log_pen_mvmm_model_sel(mvmm_sel_df, select_metric=metric)
        plot_mvmm_model_selection(mvmm_sel_df, group_var='n_comp_est',
                                  group_var_label="Number of components",
                                  select_metric=metric)
        # plt.title('Multiview models')
        save_fig(join(log_sel_dir, 'log_pen_model_sel_{}.png'.format(metric)))

    # fitting history
    # if isinstance(estimator, TwoStage):

    if estimator.start_.max_n_steps > 0:
        lp_start_save_dir = make_and_get_dir(opt_diag_dir,
                                             'log_pen_mvmm', 'start')
        lp_final_save_dir = make_and_get_dir(opt_diag_dir,
                                             'log_pen_mvmm', 'final')

        plot_mvmm(estimator.start_, inches=inches,
                  save_dir=lp_start_save_dir)

    else:
        lp_final_save_dir = make_and_get_dir(opt_diag_dir, 'log_pen_mvmm')

    plot_log_pen_mvmm(estimator.final_, inches=inches,
                      save_dir=lp_final_save_dir)
    # plot_bd_mvmm_opt_history(estimator=estimator, save_dir=opt_diag_dir)

    # else:
    #     lp_save_dir = make_and_get_dir(opt_diag_dir, 'log_pen_mvmm')
    #     plot_mvmm(estimator, inches=inches, save_dir=lp_save_dir)


# save selected models
if os.path.exists(sel_models_fpath):
    existing_sel_models = load(sel_models_fpath)
    for k in existing_sel_models.keys():
        if k not in sel_models.keys():
            sel_models[k] = existing_sel_models[k]

dump(sel_models, sel_models_fpath)
