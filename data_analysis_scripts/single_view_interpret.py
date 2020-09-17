from joblib import load, dump
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from mvmm_sim.data_analysis.utils import load_data
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.data_analysis.single_view.analyze_fit_model import \
    get_interpret_data
from mvmm_sim.data_analysis.single_view.viz_results import plot_gmm_pcs
from mvmm_sim.data_analysis.viz_loadings import plot_loading, \
    plot_loadings_scatter
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.data_analysis.survival import plot_survival

# vizualization settings
inches = 8
n_top_samples = 5
max_var_to_show = 50

parser = argparse.\
    ArgumentParser(description='Cluster interpretation.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

parser.add_argument('--fpaths', nargs='+',
                    help='Paths to data sets.')

parser.add_argument('--super_fpaths', nargs='+', default=None,
                    help='Paths to super data sets.')

parser.add_argument('--vars2compare_fpath', default=None,
                    help='(optinoal) file path to external metadata'
                         'to compare to.')

parser.add_argument('--survival_fpath', default=None,
                    help='(optinoal) file path to survival data.')

parser.add_argument('--duration_col',
                    help='Column name of the survival duration data.')

parser.add_argument('--event_col',
                    help='Column name of the survival event data.')


args = parser.parse_args()

results_dir = args.results_dir
fpaths = args.fpaths
vars2compare_fpath = args.vars2compare_fpath
super_fpaths = args.super_fpaths
survival_fpath = args.survival_fpath

print(args)

# setup directories
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting')
model_sel_dir = make_and_get_dir(results_dir, 'model_selection')
opt_diag_dir = make_and_get_dir(results_dir, 'opt_diagnostics')
clust_interpret_dir = make_and_get_dir(results_dir, 'interpret')

# load models and data
n_views = len(fpaths)
models = load(join(fitting_dir, 'selected_models'))
view_data, dataset_names, sample_names, view_feat_names = \
    load_data(*fpaths)

view_data = [pd.DataFrame(view_data[v],
                          index=sample_names,
                          columns=view_feat_names[v])
             for v in range(n_views)]

# only single view models
view_model_keys = [k for k in models.keys() if 'view' in k]
assert len(view_model_keys) == n_views


# possibly load metadata for comparison
if vars2compare_fpath is not None:
    vars2compare = pd.read_csv(vars2compare_fpath, index_col=0)
    vars2compare = vars2compare.loc[sample_names, :]
else:
    vars2compare = None

# possibly load super data
super_data = [None for _ in range(n_views)]
if super_fpaths is not None:
    assert len(super_fpaths) == len(fpaths)
    for v, fpath in enumerate(super_fpaths):
        # if this path is empty this view does have super data
        if len(fpath) == 0:
            continue
        super_data[v] = pd.read_csv(fpath, index_col=0)

# possibly load survival data
if survival_fpath is not None:
    metadata = pd.read_csv(survival_fpath, index_col=0)

    n_samples = metadata.shape[0]
    print('n sample = {}'.format(n_samples))

    survival_df = pd.DataFrame()
    survival_df['duration'] = metadata[args.duration_col]
    survival_df['event_obs'] = metadata[args.event_col]

    survival_df = survival_df.loc[sample_names]

else:
    survival_df = None

for model_key in view_model_keys:
    v = int(model_key.split('view_')[1])

    model = models[model_key]
    dataset_name = dataset_names[v]
    X = view_data[v]
    # feat_names = view_feat_names[v]
    # X = pd.DataFrame(X, index=sample_names, columns=feat_names)

    save_dir = make_and_get_dir(clust_interpret_dir, dataset_name)

    out = get_interpret_data(model, X=X,
                             super_data=super_data[v],
                             vars2compare=vars2compare,
                             survival_df=survival_df,
                             stub=dataset_names[v],
                             n_top_samples=5,
                             clust_size_min=5)

    # save
    for k in ['y_pred', 'clust_best_samples', 'clust_probs',
              'summary', 'cl_means', 'stand_cl_means']:
        pd.DataFrame(out[k]).to_csv(join(save_dir, '{}.csv'.format(k)))

    if 'comparison' in out.keys():
        dump(out['comparison'], join(save_dir, 'variable_comparison'))

    if 'cl_super_means' in out.keys():
        out['cl_super_means'].to_csv(join(save_dir, 'cl_super_means.csv'))
        out['stand_cl_super_means'].\
            to_csv(join(save_dir, 'stand_cl_super_means.csv'))

    ############
    # Plot PCA #
    ############
    plt.figure(figsize=(inches, inches))
    plot_gmm_pcs(gmm=model, X=X.values)
    save_fig(join(save_dir, 'pca_projections.png'))

    ######################
    # Plot cluster means #
    ######################
    mean_save_dir = make_and_get_dir(save_dir, 'cl_means')
    scatter_save_dir = make_and_get_dir(mean_save_dir, 'scatter')

    if 'stand_cl_super_means' in out.keys():
        stand_cl_means = out['stand_cl_super_means']
    else:
        stand_cl_means = out['stand_cl_means']

    cluster_labels = out['info']['cluster_labels']

    for cl_idx in range(stand_cl_means.shape[0]):
        cl_name = cluster_labels[cl_idx]
        plt.figure(figsize=(5, 10))

        v = stand_cl_means.iloc[cl_idx, :]

        # visualize loadings
        # only shows top variables
        plot_loading(v, show_top=max_var_to_show)
        plt.title('{} standardized mean'.format(cl_name))
        plt.xlabel('')
        save_fig(join(mean_save_dir, '{}.png'.format(cl_name)))

        # scatter plots of all variables in loadings
        plt.figure(figsize=(inches, inches))
        plot_loadings_scatter(v=v, show_top=max_var_to_show)
        plt.title('{} standardized mean'.format(cl_name))
        save_fig(join(scatter_save_dir, '{}_scatter.png'.format(cl_name)))

    ########################
    # metadata comparisons #
    ########################
    if 'comparison' in out.keys():
        plt.figure(figsize=(inches, vars2compare.shape[1] * inches))
        out['comparison'].plot(wspace=.5)
        save_fig(join(save_dir, 'metadata_comparison.png'))

    ############
    # survival #
    ############

    if 'survival' in out.keys():
        pval = out['survival']['pval']

        plt.figure(figsize=(inches, inches))
        plot_survival(df=out['survival']['df'], cat_col='cluster')
        plt.xlabel("Time (days)")
        plt.ylabel("Survival")
        plt.title('{} vs. {}, p={:1.3f}'.format(args.event_col, dataset_name,
                                                pval))
        save_fig(join(save_dir, 'survival.png'))
        # TODO: label survival variable!
