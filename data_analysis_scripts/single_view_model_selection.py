from joblib import load, dump
import os
import argparse
import matplotlib.pyplot as plt
from os.path import join
from glob import glob

from mvmm_sim.simulation.ResultsWriter import ResultsWriter
from mvmm_sim.data_analysis.single_view.viz_results import plot_model_selection
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.simulation.opt_viz import plot_loss_history
from mvmm_sim.simulation.utils import make_and_get_dir

inches = 8

parser = argparse.\
    ArgumentParser(description='Model selection results.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

args = parser.parse_args()

# sim_name = args.sim_name

results_dir = args.results_dir
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting')
sv_fitting_dir = make_and_get_dir(fitting_dir, 'single_view')
model_sel_dir = make_and_get_dir(results_dir, 'model_selection')
opt_diag_dir = make_and_get_dir(results_dir, 'opt_diagnostics')

res_writer = ResultsWriter(os.path.join(log_dir,
                           'single_view_model_selection.txt'),
                           delete_if_exists=True)

# data = load(os.path.join(save_dir, 'multi_view_fit_data'))

#############
# load data #
#############

metadata = load(os.path.join(sv_fitting_dir, 'single_view_fit_metadata'))
dataset_names = metadata['dataset_names']

# load models
fpaths = glob(join(sv_fitting_dir, 'fitted*'))
n_view_models = sum(('fitted_view' in os.path.basename(fpath))
                    for fpath in fpaths)
view_models = [None for _ in range(n_view_models)]
cat_model = None
for fpath in fpaths:
    fname = os.path.basename(fpath)

    if 'fitted_cat' in fname:
        cat_model = load(fpath)

    elif 'fitted_view' in fname:
        v = int(fname.split('_')[2])
        view_models[v] = load(fpath)

sel_models_fpath = os.path.join(fitting_dir, 'selected_models')
sel_models = {}

###########
# cat gmm #
###########
# if 'cat_gmm' in results['models'].keys():
if cat_model is not None:
    if cat_model.check_fit():

        estimator = cat_model.best_estimator_
        sel_models['cat_gmm'] = estimator

        # model selection
        est_n_comp = estimator.n_components
        res_writer.write("Cat data GMM estimated number of components: {}".
                         format(est_n_comp))

        # _model_sel_dir = make_and_get_dir(model_sel_dir, 'cat_gmm')
        # plt.figure(figsize=(inches, inches))
        plot_model_selection(cat_model, save_dir=model_sel_dir,
                             name_stub='cat_gmm',
                             title='GMM on concatenated data',
                             inches=inches)

        # optimization history
        loss_history = estimator.opt_data_['history']['obs_nll']
        plot_loss_history(loss_history,
                          loss_name='Observed data negative log-likelihood')
        save_fig(os.path.join(opt_diag_dir, 'cat_best_model_opt_history.png'))

    else:
        print("cat isnt fit...")


########################
# view marginal models #
########################

n_views = len(view_models)
for v in range(n_views):

    estimator = view_models[v].best_estimator_
    sel_models['view_' + str(v)] = estimator

    # model selection
    est_n_comp = estimator.n_components
    res_writer.write("{} GMM estimated number of components: {}".
                     format(dataset_names[v], est_n_comp))

    # _model_sel_dir = make_and_get_dir(model_sel_dir, dataset_names[v])
    # plt.figure(figsize=(inches, inches))
    # plot_mix_model_selc(view_models[v])
    # plt.title(dataset_names[v])
    # save_fig(os.path.join(model_sel_dir,
    #                       '{}_gmm_model_sel.png'.format(dataset_names[v])))
    plot_model_selection(view_models[v], save_dir=model_sel_dir,
                         name_stub=dataset_names[v],
                         title=dataset_names[v],
                         inches=inches)

    loss_history = estimator.opt_data_['history']['obs_nll']
    plot_loss_history(loss_history,
                      loss_name='Observed data negative log-likelihood')
    save_fig(os.path.join(opt_diag_dir,
                          '{}_best_model_opt_history.png'.
                          format(dataset_names[v])))

# possibly load existing files
if os.path.exists(sel_models_fpath):
    existing_sel_models = load(sel_models_fpath)
    for k in existing_sel_models.keys():
        if k not in sel_models.keys():
            sel_models[k] = existing_sel_models[k]

dump(sel_models, sel_models_fpath)
