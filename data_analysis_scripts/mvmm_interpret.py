from joblib import load, dump
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from mvmm.multi_view.TwoStage import TwoStage

from mvmm_sim.data_analysis.utils import load_data
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.data_analysis.viz_loadings import plot_loading, \
    plot_loadings_scatter
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.data_analysis.multi_view.analyze_fit_mvmm import \
    get_mvmm_interpret_data
from mvmm_sim.data_analysis.multi_view.viz_resuls import plot_Pi
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

duration_col = args.duration_col
event_col = args.event_col

# setup directories
log_dir = make_and_get_dir(results_dir, 'log')
fitting_dir = make_and_get_dir(results_dir, 'model_fitting')
model_sel_dir = make_and_get_dir(results_dir, 'model_selection')
opt_diag_dir = make_and_get_dir(results_dir, 'opt_diagnostics')
clust_interpret_dir = make_and_get_dir(results_dir, 'interpret')

# load models and data
models = load(join(fitting_dir, 'selected_models'))
view_data, dataset_names, sample_names, view_feat_names = \
    load_data(*fpaths)

n_views = len(fpaths)
view_data = [pd.DataFrame(view_data[v],
                          index=sample_names,
                          columns=view_feat_names[v])
             for v in range(n_views)]

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
    survival_df['duration'] = metadata[duration_col]
    survival_df['event_obs'] = metadata[event_col]

    survival_df = survival_df.loc[sample_names]

else:
    survival_df = None


mvmm_model_keys = [k for k in models.keys()
                   if k in ['bd_mvmm', 'log_pen_mvmm']]

for model_key in mvmm_model_keys:
    n_views = len(view_data)

    model = models[model_key]
    if isinstance(model, TwoStage):
        start_model = model.start_
        model = model.final_

    else:
        start_model = None

    save_dir = make_and_get_dir(clust_interpret_dir, model_key)

    # compute all the results we need
    out = get_mvmm_interpret_data(model, view_data=view_data,
                                  super_data=super_data,
                                  vars2compare=vars2compare,
                                  survival_df=survival_df,
                                  dataset_names=dataset_names,
                                  stub=model_key,
                                  n_top_samples=5,
                                  clust_size_min=0)

    ###############
    # joint level #
    ###############

    # cluster prediction data
    for k in ['joint_summary', 'y_pred_joint', 'joint_clust_best_samples']:
        out[k].to_csv(join(save_dir, '{}.csv'.format(k)))

    # metadata comparisons
    if 'joint_comparison' in out.keys():
        dump(out['joint_comparison'], join(save_dir, 'metadata_comparison'))

        plt.figure(figsize=(2 * inches, 2 * inches))
        out['joint_comparison'].plot()
        save_fig(join(save_dir, 'metadata_comparison.png'))

    # survival
    if 'joint_survival' in out.keys():
        pval = out['joint_survival']['pval']
        plt.figure(figsize=(inches, inches))
        plot_survival(df=out['joint_survival']['df'], cat_col='cluster')
        plt.xlabel("Time (days)")
        plt.ylabel("Survival")
        plt.title('{} vs. joint label, p={:1.3f}'.format(args.event_col, pval))
        save_fig(join(save_dir, 'joint_survival.png'))
        # TODO: label survival variable!

    ##############
    # View level #
    ##############

    for v in range(n_views):
        name = dataset_names[v]
        view_save_dir = make_and_get_dir(save_dir, name)

        # save cluster prediction data
        for k in ['view_clust_best_samples', 'view_summaries', 'y_pred_view']:
            pd.DataFrame(out[k][name]).\
                to_csv(join(view_save_dir, '{}.csv'.format(k)))

        # metadata comparisions
        if 'view_comparisons' in out.keys():
            dump(out['view_comparisons'][name],
                 join(view_save_dir, 'metadata_comparison_{}'.format(name)))

            plt.figure(figsize=(2 * inches, 2 * inches))
            out['view_comparisons'][name].plot()
            save_fig(join(view_save_dir, 'metadata_comparison_{}.png'.
                                         format(name)))

        if 'view_survival' in out.keys():
            pval = out['view_survival'][name]['pval']
            plt.figure(figsize=(inches, inches))
            plot_survival(df=out['view_survival'][name]['df'],
                          cat_col='cluster')
            plt.xlabel("Time (days)")
            plt.ylabel("Survival")
            plt.title('{} vs. {} marginal, p={:1.3f}'.
                      format(args.event_col, name, pval))
            save_fig(join(view_save_dir, 'survival.png'))
            # TODO: label survival variable!

        # plot pca
        # TODO: do PCAs

        # plot cluster means
        if name in out['view_stand_cl_super_means'].keys():
            stand_cl_means = out['view_stand_cl_super_means'][name]
        else:
            stand_cl_means = out['view_cl_means'][name]

        mean_save_dir = make_and_get_dir(view_save_dir, 'cluster_means')
        scatter_save_dir = make_and_get_dir(mean_save_dir, 'scatter')
        top_txt_save_dir = make_and_get_dir(mean_save_dir, 'top')

        for cl_idx in range(stand_cl_means.shape[0]):

            cl_name = stand_cl_means.index[cl_idx]
            v = stand_cl_means.iloc[cl_idx, :]

            # visualize loadings
            # only shows top variables
            plt.figure(figsize=(5, 10))
            plot_loading(v, show_top=max_var_to_show)
            plt.title('{} standardized mean'.format(cl_name))
            plt.xlabel('')
            save_fig(join(mean_save_dir, '{}.png'.format(cl_name)))

            # scatter plots of all variables in loadings
            plt.figure(figsize=(inches, inches))
            plot_loadings_scatter(v=v, show_top=max_var_to_show)
            plt.title('{} standardized mean'.format(cl_name))
            save_fig(join(scatter_save_dir, '{}_scatter.png'.format(cl_name)))

            # save text file with top variables
            _max_var_to_show = min(len(v), max_var_to_show)
            top_vars = abs(v).sort_values(ascending=False).\
                index.values[0:_max_var_to_show]
            v_top = v.loc[top_vars]
            df = pd.DataFrame(columns=['value', 'sign'])
            df['value'] = v_top
            df['sign'] = 'positive'
            df['sign'][v_top < 0] = 'negative'
            df.to_csv(join(top_txt_save_dir, '{}.csv'.format(cl_name)),
                      index=True)

    ###############
    # block level #
    ###############
    block_save_dir = make_and_get_dir(save_dir, 'block')

    # block comparisons
    for k in ['y_pred_block', 'block_summary']:
        if k in out.keys():
            out[k].to_csv(join(block_save_dir, '{}.csv'.format(k)))

    # metadata comparisons
    if 'block_comparisons' in out.keys():
        dump(out['block_comparisons'],
             join(block_save_dir, 'metadata_comparisons_block'))

        plt.figure(figsize=(2 * inches, 2 * inches))
        out['block_comparisons'].plot()
        save_fig(join(block_save_dir, 'metadata_comparison_block.png'))

    # survival
    if 'block_survival' in out.keys():
        pval = out['block_survival']['pval']

        plt.figure(figsize=(inches, inches))
        plot_survival(df=out['block_survival']['df'], cat_col='cluster')
        plt.xlabel("Time (days)")
        plt.ylabel("Survival")
        plt.title('{} vs. blocks, p={:1.3f}'.format(args.event_col, pval))
        save_fig(join(block_save_dir, 'block_survival.png'))
        # TODO: label survival variable!

    # save survival
    dump({'joint_survival': out['joint_survival'],
          'block': out['block_survival'],
          'view_survival': out['view_survival']},
         filename=join(save_dir, 'survival'))

    ###########
    # Plot Pi #
    ###########

    # raw Pi
    plt.figure(figsize=(inches, inches))
    plot_Pi(out['Pi'])
    save_fig(join(save_dir, 'Pi_est.png'))
    plt.title(model_key)

    # cluster permuted Pi
    plt.figure(figsize=(inches, inches))
    plot_Pi(out['Pi_block_perm'], mask=out['Pi_block_perm_zero_mask'])
    plt.title('{}, n_blocks = {}'.format(model_key, out['info']['n_blocks']))
    save_fig(join(save_dir, 'Pi_est_bd_perm.png'))

    dump({'Pi': out['Pi'],
          'Pi_block_perm': out['Pi_block_perm'],
          'Pi_block_perm_zero_mask': out['Pi_block_perm_zero_mask']},
         filename=join(save_dir, 'pi_data'))
