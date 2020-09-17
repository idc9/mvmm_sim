from os.path import join
import pandas as pd
# import os
import matplotlib.pyplot as plt
from joblib import dump
# import numpy as np
from time import time
# import matplotlib.pyplot as plt

from pca.PCA import PCA
from pca.viz import scree_plot
from pca.rank_selection.noise_estimates import estimate_noise_mp_quantile
from pca.rank_selection.rmt_threshold import marcenko_pastur_edge_threshold, \
    donoho_gavish_threshold

from mvmm_sim.tcga.TCGAPaths import TCGAPaths
from mvmm_sim.simulation.sim_viz import save_fig
from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.simulation.ResultsWriter import ResultsWriter

import argparse

# cancer_type = 'LUAD'  # 'GBM'  # 'BRCA'
parser = argparse.ArgumentParser(description='Process TCGA data.')

parser.add_argument('--cancer_type', default='BRCA',
                    help='Which cancer type e.g. LUAD, BCRA, OV.')

parser.add_argument('--feat_list', default='icluster',
                    help='Which feature list to use e.g. icluster,'
                    'all, top_2000 by variace.')

parser = bayes_parser(parser)
args = parser.parse_args()
bayes_submit(args)

cancer_type = args.cancer_type
feat_list = args.feat_list

print(cancer_type)

raw_data_dir = TCGAPaths().raw_data_dir

pca_dir = make_and_get_dir(TCGAPaths().top_dir, 'pca', cancer_type, feat_list)
diagnostics_dir = make_and_get_dir(pca_dir, 'diagnostics')
feat_save_dir = join(TCGAPaths().pro_data_dir, cancer_type, feat_list)

datasets = ['rna', 'mi_rna', 'dna_meth', 'cp']
data = {}
# save processed data
for k in datasets:
    fpath = join(feat_save_dir, '{}.csv'.format(k))
    data[k] = pd.read_csv(fpath, index_col=0)


########################
# extract PCA features #
########################

# if cancer_type == 'pan':
#     rank_sel_kws = {'n_components': 'bi_cv',
#                     # 'rank_sel_kws': {'rotate': True},
#                     'max_rank': 500}

# else:
#     rank_sel_kws = {'n_components': 'rmt_threshold',
#                     'rank_sel_kws': {'thresh_method': 'dg'}}

# rank_sel_kws = {'n_components': 'bai_ng_bic',
#                 'rank_sel_kws': {'who': 1},
#                 'max_rank': 400
#                 }


for k in data.keys():
    res_writer = ResultsWriter(fpath=join(diagnostics_dir,
                                          '{}_log.txt'.format(k)),
                               delete_if_exists=True)

    res_writer.write(k)
    res_writer.write('shape {}'.format(data[k].shape))

    rank_sel_kws = {'n_components': 'rmt_threshold',
                    'rank_sel_kws': {'thresh_method': 'dg'}}

    pca = PCA(**rank_sel_kws)

    start_time = time()
    pca.fit(data[k].values)
    runtime = time() - start_time
    res_writer.write("computed pca took {:1.2f} seconds".format(runtime))
    res_writer.write('Estimated n components {}'.format(pca.n_components_))

    ###################
    # save fitted PCA #
    ###################
    dump(pca, join(pca_dir, '{}_pca'.format(k)))

    comp_names = ['pc_{}'.format(k + 1) for k in range(pca.n_components_)]

    # save scores
    UD = pca.unnorm_scores_
    UD = pd.DataFrame(UD, index=data[k].index,
                      columns=comp_names)
    UD.index.name = data[k].index.name
    UD.name = k
    UD.to_csv(join(feat_save_dir, '{}_pca_feats.csv'.format(k)))

    V = pd.DataFrame(pca.loadings_,
                     index=data[k].columns,
                     columns=comp_names)
    V.name = k
    V.to_csv(join(pca_dir, '{}_pca_loadings.csv'.format(k)))

    ##############################
    # Rank selection diagnostics #
    ##############################
    svals = pca.all_svals_
    shape = data[k].shape

    res_writer.write('largest sval: {}, smallest sval: {}'.
                     format(max(svals), min(svals)))

    stds = data[k].std(axis=0)
    res_writer.write('largest variable std: {}, smallest variable std: {}'.
                     format(max(stds), min(stds)))

    noise_est_mpq = estimate_noise_mp_quantile(svals=svals, shape=shape)
    res_writer.write('marcenko_pastur noise estimate {}'.format(noise_est_mpq))

    # max_rank = 500
    # noise_est_naive = estimate_noise_naive_rank_based(X=df.values, rank=max_rank, UDVt=pca.UDVt_)
    # res_writer.write('Naise noise estimate with rank {}: {}'.format(max_rank, noise_est_mpq))

    thresh_mpe = marcenko_pastur_edge_threshold(shape=shape,
                                                sigma=noise_est_mpq)
    rank_est_mpe = sum(svals > thresh_mpe)

    thresh_dg = donoho_gavish_threshold(shape=shape,
                                        sigma=noise_est_mpq)
    rank_est_dg = sum(svals > thresh_dg)

    res_writer.write('marcenko pastur edge threshold rank estimate {}'.
                     format(rank_est_mpe))
    res_writer.write('donoho gavish edge threshold rank estimate {}'.
                     format(rank_est_dg))

    # scree plot
    plt.figure(figsize=(20, 20))
    scree_plot(svals, color='black')
    plt.ylim(0)
    plt.ylabel("Singular value")

    plt.axhline(thresh_dg, label='DG theshold ({:1.2f})'.format(thresh_dg),
                color='blue', ls='--')
    plt.axhline(thresh_mpe, label='MPE theshold ({:1.2f})'.format(thresh_mpe),
                color='orange', ls='--')

    plt.axvline(rank_est_dg,
                label='DG rank estimate ({})'.format(rank_est_dg),
                color='blue')
    plt.axvline(rank_est_mpe,
                label='MPE rank estimate ({})'.format(rank_est_mpe),
                color='orange')

    plt.legend()
    plt.title(k)
    save_fig(join(diagnostics_dir, '{}_scree_plot.png'.format(k)))

    # # save scree plot
    # plt.figure(figsize=(10, 10))
    # scree_plot(pca.all_svals_, color='black')
    # plt.axvline(pca.n_components_,
    #             label='{}'.format(pca.n_components_),
    #             color='red')
    # plt.legend()
    # plt.ylim(0)
    # save_fig(join(diagnostics_dir, '{}_rank_selection.png'.format(k)))

    # # for bi cross validation save MSE curve
    # if pca.n_components == 'bi_cv':
    #     plt.figure(figsize=(10, 10))
    #     mse = pca.rank_sel_out_['errors']
    #     plt.plot(mse.index, mse, marker='.')
    #     plt.title(k)
    #     plt.ylabel("CV-MSE(k) / CV-MSE(0)")
    #     plt.xlabel("Rank")
    #     plt.axvline(pca.n_components_, label=pca.n_components_,
    #                 color='red')
    #     plt.legend()
    #     plt.ylim(0)
    #     save_fig(join(diagnostics_dir, '{}_bi_cv_mse.png'.format(k)))
