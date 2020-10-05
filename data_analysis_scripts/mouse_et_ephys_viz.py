from joblib import load
from os.path import join
import argparse
import numpy as np
import matplotlib.pyplot as plt


from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.data_analysis.utils import load_data
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.mouse_et.MouseETPaths import MouseETPaths
from mvmm_sim.mouse_et.raw_ephys_loading import load_raw_ephys
from mvmm_sim.mouse_et.ephys_viz import get_ephys_super_data,\
    plot_top_clust_ephys_curves, plot_cluster_ephys_curve

parser = argparse.\
    ArgumentParser(description='Cluster interpretation.')

parser.add_argument('--results_dir', default=None,
                    help='Directory to save results.')

parser.add_argument('--fpaths', nargs='+',
                    help='Paths to data sets.')
args = parser.parse_args()

inches = 8
n_top_clust = 10

results_dir = args.results_dir
fpaths = args.fpaths

fitting_dir = join(results_dir, 'model_fitting')
ephys_viz_dir = join(results_dir, 'interpret', 'bd_mvmm', 'ephys_pca_feats')


# load models and data
models = load(join(fitting_dir, 'selected_models'))
view_data, dataset_names, sample_names, view_feat_names = load_data(*fpaths)


# load raw ephys data
orig_data_dir = join(MouseETPaths().raw_data_dir, 'inh_patchseq_spca_files',
                     'orig_data_csv')
ephys_raw = load_raw_ephys(orig_data_dir, concat=False)
for k in ephys_raw.keys():
    ephys_raw[k] = ephys_raw[k].loc[sample_names]
    print(k, ephys_raw[k].shape)
n_datasets = len(ephys_raw)


# get data for plotting
v = 1
cluster_super_means, super_data_means, super_data_stds, y_cnts = \
    get_ephys_super_data(model=models['bd_mvmm'].final_.view_models_[v],
                         fit_data=view_data[v],
                         ephys_raw=ephys_raw)

clust_labels = ['cluster_{}'.format(cl_idx + 1)
                for cl_idx in range(len(y_cnts))]


# plot top several clusters
plot_top_clust_ephys_curves(cluster_super_means,
                            y_cnts=y_cnts,
                            overall_means=super_data_means,
                            overall_stds=super_data_stds,
                            clust_labels=clust_labels,
                            n_to_show=n_top_clust,
                            inches=inches)

save_fig(join(ephys_viz_dir, 'ephys_curves_top_clust.png'))

# plot each (non-trival) cluster
# non_trivial_clusters = y_cnts[y_cnts >= 5].index.values
non_trivial_clusters = y_cnts[y_cnts >= 0].index.values
save_dir = make_and_get_dir(ephys_viz_dir, 'cluster_curves')
for cl_idx in non_trivial_clusters:

    label = clust_labels[cl_idx]

    values = {}
    for name in cluster_super_means.keys():
        values[name] = cluster_super_means[name][cl_idx]

    plt.figure(figsize=(2 * n_datasets * inches, inches))

    plot_cluster_ephys_curve(values,
                             overall_means=super_data_means,
                             overall_stds=super_data_stds,
                             y_label=label)

    save_fig(join(save_dir, '{}_ephys_curve.png'.format(label)))
