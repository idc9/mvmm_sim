import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler

from mvmm.multi_view.BlockDiagMVMM import BlockDiagMVMM
from mvmm.multi_view.LogPenMVMM import LogPenMVMM
from mvmm.multi_view.block_diag.graph.bipt_community import \
    community_summary, get_block_mat

from mvmm_sim.simulation.community_results import get_y_comm_pred_out_comm, \
    get_y_comm_pred_restrict_comm
from mvmm_sim.data_analysis.utils import argmax, drop_small_classes
from mvmm_sim.data_analysis.single_view.analyze_fit_model import \
    summarize_mm_clusters
from mvmm_sim.data_analysis.super_means import get_super_means
from mvmm_sim.data_analysis.single_view.analyze_fit_model import get_survival

from explore.BlockBlock import BlockBlock


def get_mvmm_interpret_data(model, view_data,
                            super_data=None,
                            vars2compare=None,
                            survival_df=None,
                            stub=None,
                            dataset_names=None,
                            n_top_samples=5,
                            clust_size_min=0):

    n_views = len(view_data)
    n_samples = view_data[0].shape[0]
    # dims = [view_data[v].shape[1] for v in range(n_views)]

    for v in range(n_views):
        assert isinstance(view_data[v], pd.DataFrame)

    sample_names = view_data[0].index.values
    view_feat_names = [view_data[v].columns.values for v in range(n_views)]

    view_data = [view_data[v].values for v in range(n_views)]

    if vars2compare is not None:
        vars2compare = vars2compare.loc[sample_names, :]
    if survival_df is not None:
        survival_df = survival_df.loc[sample_names, :]

    if super_data is None:
        super_data = [None] * n_views

    out = {}

    n_joint_comp = model.n_components
    n_view_comp = model.n_view_components

    if dataset_names is None:
        dataset_names = ['view_{}'.format(v + 1) for v in range(n_views)]

    if stub is None:
        stub = 'joint'

    # create labels for clusters
    # joint_cluster_labels = np.array(['{}__{}'.format(stub, k + 1)
    #                                  for k in range(n_joint_comp)])

    # view_cluster_labels = [np.array(['{}__{}'.format(dataset_names[v], k + 1)
    #                                  for k in range(n_view_comp[v])])
    #                        for v in range(n_views)]
    joint_cluster_labels = (1 + np.arange(n_joint_comp)).astype(str)

    view_cluster_labels = [(1 + np.arange(n_view_comp[v])).astype(str)
                           for v in range(n_views)]
    #######################
    # cluster predictions #
    #######################

    # map joint clusters to the view marginal clusters
    view_cl_idxs = np.array([model._get_view_clust_idx(k)
                             for k in range(model.n_components)])

    view_cl_labels = view_cl_idxs.astype(str)
    for k, v in product(range(n_joint_comp), range(n_views)):
        view_cl_labels[k, v] = view_cluster_labels[v][view_cl_idxs[k, v]]
    view_cl_labels = pd.DataFrame(view_cl_labels,
                                  index=joint_cluster_labels,
                                  columns=dataset_names)

    top_col_names = ['top_' + str(k + 1) for k in range(n_top_samples)]

    ################
    # joint labels #
    ################

    # predictions
    y_pred_joint = model.predict(view_data)
    joint_summary = summarize_mm_clusters(y_pred=y_pred_joint,
                                          weights=model.weights_,
                                          cluster_labels=joint_cluster_labels)
    joint_summary = joint_summary.join(view_cl_labels)
    out['joint_summary'] = joint_summary

    # convert to cluster labels
    y_pred_joint = [joint_cluster_labels[y] for y in y_pred_joint]
    y_pred_joint = pd.Series(y_pred_joint, index=sample_names,
                             name='joint_clusters')
    out['y_pred_joint'] = y_pred_joint

    # ignore small classes for below comparisons!
    y_pred_joint = drop_small_classes(y=y_pred_joint,
                                      clust_size_min=clust_size_min)

    # top samples
    cl_prob_joint = model.predict_proba(view_data)
    joint_clust_best_samples = [argmax(cl_prob_joint[:, k], n=n_top_samples)
                                for k in range(n_joint_comp)]
    joint_clust_best_samples = \
        np.array([sample_names[best_idxs]
                 for best_idxs in joint_clust_best_samples])

    joint_clust_best_samples = pd.DataFrame(joint_clust_best_samples,
                                            index=joint_cluster_labels,
                                            columns=top_col_names)
    out['joint_clust_best_samples'] = joint_clust_best_samples

    # metadata comparisons
    if vars2compare is not None:
        vars2compare = vars2compare.loc[y_pred_joint.index, :]

        compare_kws = {'alpha': 0.05, 'cat_test': 'auc',
                       'corr': 'pearson', 'multi_cat': 'ovo',
                       'multi_test': 'fdr_bh', 'nan_how': 'drop'}

        joint_comparison = BlockBlock(**compare_kws)
        joint_comparison.fit(y_pred_joint, vars2compare).correct_multi_tests()

        out['joint_comparison'] = joint_comparison

    # survival
    if survival_df is not None:
        _survival_df = survival_df.loc[y_pred_joint.index, :]

        out['joint_survival'] = get_survival(survival_df=_survival_df,
                                             y=y_pred_joint)

    ##############
    # view level #
    ##############

    # predictions
    y_pred_view = model.predict_view_labels(view_data)

    view_summaries = {}
    for v in range(n_views):
        name = dataset_names[v]

        c = model.view_models_[v].covariances_
        w = model.view_models_[v].weights_
        y = y_pred_view[:, v]

        view_summaries[name] = \
            summarize_mm_clusters(y_pred=y, weights=w, covs=c,
                                  cluster_labels=view_cluster_labels[v])

    out['view_summaries'] = view_summaries

    # convert to cluster labels
    # y_pred_view = _y_pred_view.copy()  # .astype(str)
    # for i, v in product(range(n_samples), range(n_views)):
    #     y_pred_view[i, v] = view_cluster_labels[v][_y_pred_view[i, v]]
    y_pred_view = pd.DataFrame(y_pred_view + 1,  # convert to 1 indexing
                               index=sample_names,
                               columns=dataset_names).astype(str)

    out['y_pred_view'] = y_pred_view

    # top samples
    view_cl_prob = model.predict_view_marginal_probas(view_data)
    view_clust_best_samples = {}
    for v in range(n_views):
        name = dataset_names[v]

        vc_best_samples = [argmax(view_cl_prob[v][:, k], n=n_top_samples)
                           for k in range(n_view_comp[v])]

        vc_best_samples = np.array([sample_names[best_idxs]
                                    for best_idxs in vc_best_samples])

        view_clust_best_samples[name] = \
            pd.DataFrame(vc_best_samples,
                         index=view_cluster_labels[v],
                         columns=top_col_names)

    out['view_clust_best_samples'] = view_clust_best_samples

    # metadata comparisons
    if vars2compare is not None:
        view_comparisons = {}
        for v in range(n_views):
            name = dataset_names[v]

            y = y_pred_view.iloc[:, v]
            # ignore small classes
            y = drop_small_classes(y=y,
                                   clust_size_min=clust_size_min)

            _vars2compare = vars2compare.loc[y.index, :]

            c = BlockBlock(**compare_kws)
            c.fit(y, _vars2compare).correct_multi_tests()
            view_comparisons[name] = c

        out['view_comparisons'] = view_comparisons

    # survival
    if survival_df is not None:
        view_survival = {}
        for v in range(n_views):
            name = dataset_names[v]

            y = y_pred_view.iloc[:, v]
            # ignore small classes
            y = drop_small_classes(y=y,
                                   clust_size_min=clust_size_min)

            _survival_df = survival_df.loc[y.index, :]

            view_survival[name] = get_survival(survival_df=_survival_df,
                                               y=y)

        out['view_survival'] = view_survival

    # get cluster means for each view
    out['view_cl_means'] = {}
    out['view_stand_cl_means'] = {}
    out['view_cl_super_means'] = {}
    out['view_stand_cl_super_means'] = {}
    for v in range(n_views):
        name = dataset_names[v]

        cl_means = model.view_models_[v].means_
        processor = StandardScaler(with_mean=True,
                                   with_std=True).fit(view_data[v])
        stand_cl_means = processor.transform(cl_means)

        cl_means = pd.DataFrame(cl_means,
                                index=view_cluster_labels[v],
                                columns=view_feat_names[v])

        stand_cl_means = pd.DataFrame(stand_cl_means,
                                      index=view_cluster_labels[v],
                                      columns=view_feat_names[v])

        out['view_cl_means'][name] = cl_means
        out['view_stand_cl_means'][name] = stand_cl_means

        if super_data[v] is not None:
            super_feat_names = super_data[v].columns.values
            resp = view_cl_prob[v]

            cl_super_means = get_super_means(resp=resp,
                                             super_data=super_data[v],
                                             stand=False)

            stand_cl_super_means = get_super_means(resp=resp,
                                                   super_data=super_data[v],
                                                   stand=True)

            cl_super_means = pd.DataFrame(cl_super_means,
                                          index=view_cluster_labels[v],
                                          columns=super_feat_names)

            stand_cl_super_means = pd.DataFrame(stand_cl_super_means,
                                                index=view_cluster_labels[v],
                                                columns=super_feat_names)

            out['view_cl_super_means'][name] = cl_super_means
            out['view_stand_cl_super_means'][name] = stand_cl_super_means

    ###############
    # Block level #
    ###############
    if isinstance(model, BlockDiagMVMM):
        Pi = model.bd_weights_
        zero_thresh = model.zero_thresh

    elif isinstance(model, LogPenMVMM):
        Pi = model.weights_mat_
        zero_thresh = 0

    else:
        Pi = model.weights_mat_
        zero_thresh = 0

    # get block s of the matrix
    block_mat = get_block_mat(Pi > zero_thresh)
    block_summary, Pi_block = community_summary(Pi, zero_thresh=zero_thresh)
    n_blocks_est = block_summary['n_communities']
    # block_labels = ['block_{}'.format(b + 1) for b in range(n_blocks_est)]
    block_labels = (1 + np.arange(n_blocks_est)).astype(str)
    out['block_mat'] = block_mat

    if n_views > 2:
        raise NotImplementedError

    # TODO: check this
    # add cluster names to Pi
    row_names = view_cluster_labels[0]
    col_names = view_cluster_labels[1]
    Pi = pd.DataFrame(Pi, index=row_names, columns=col_names)
    Pi.index.name = dataset_names[0]
    Pi.columns.name = dataset_names[1]
    out['Pi'] = Pi

    # get map from blocks to view clusters
    block2view_clusts = {block_labels[b]: {} for b in range(n_blocks_est)}
    for b in range(n_blocks_est):
        row_mask = block_summary['row_memberships'] == b
        row_view_clust_labels = Pi.index.values[row_mask]
        # row_view_clust_labels = [l.split('_')[-1]  # just get the number
        #                          for l in row_view_clust_labels]

        block2view_clusts[block_labels[b]][dataset_names[0]] = \
            row_view_clust_labels

        col_mask = block_summary['col_memberships'] == b
        col_view_clust_labels = Pi.columns.values[col_mask]
        # col_view_clust_labels = [l.split('_')[-1]  # just get the number
        #                          for l in col_view_clust_labels]
        block2view_clusts[block_labels[b]][dataset_names[1]] = \
            col_view_clust_labels

    # add cluster names to block permuted Pi
    row_names_perm = row_names[np.argsort(block_summary['row_memberships'])]
    col_names_perm = col_names[np.argsort(block_summary['col_memberships'])]
    Pi_block_perm = pd.DataFrame(Pi_block,
                                 index=row_names_perm, columns=col_names_perm)
    Pi_block_perm.index.name = dataset_names[0]
    Pi_block_perm.columns.name = dataset_names[1]
    out['Pi_block_perm'] = Pi_block_perm
    out['Pi_block_perm_zero_mask'] = Pi_block_perm.values < zero_thresh

    # add block labels to summary data frame
    block_label_info = []
    for k, joint_label in enumerate(joint_cluster_labels):
        view_idxs = model._get_view_clust_idx(k)
        b = block_mat[view_idxs[0], view_idxs[1]]
        if np.isnan(b):
            block_lab = ''
        else:
            block_lab = block_labels[int(b)]
        block_label_info.append({'cluster': joint_label,
                                 'block': block_lab})

    block_label_info = pd.DataFrame(block_label_info).set_index('cluster')
    out['joint_summary']['block'] = block_label_info['block']

    if n_blocks_est > 1:

        # TODO: get block weights
        block_weights = []
        for b in range(n_blocks_est):
            mask = out['block_mat'] == b
            block_weights.append(model.weights_mat_[mask].sum())

        block_weights = pd.Series(block_weights, index=block_labels)
        out['block_weights'] = block_weights

        # get block level predictions
        y_block_pred_no_restr, n_out_of_comm = \
            get_y_comm_pred_out_comm(model, view_data, block_mat)
        y_block_pred_restr = \
            get_y_comm_pred_restrict_comm(model, view_data, block_mat)
        print('n_out_of_comm', n_out_of_comm)
        # print(cluster_report(y_block_pred_restr, y_block_pred)['ars'])
        y_pred_block = y_block_pred_restr

        out['block_summary'] = \
            summarize_mm_clusters(y_pred=y_pred_block,
                                  weights=block_weights,
                                  cluster_labels=block_labels)
        # add block2view cluster labels
        for dn in dataset_names:
            out['block_summary'][dn + '__clusters'] = ''
            for b in range(n_blocks_est):
                bl = block_labels[b]
                x = ' '.join(block2view_clusts[bl][dn])
                out['block_summary'].loc[bl, dn + '__clusters'] = x

        # format for pandas
        y_pred_block = [block_labels[y] for y in y_pred_block]
        y_pred_block = pd.Series(y_pred_block, index=sample_names,
                                 name='block')

        out['y_pred_block'] = y_pred_block

        # metadata comparisons
        if vars2compare is not None:
            _vars2compare = vars2compare.loc[y_pred_block.index, :]
            block_comparison = BlockBlock(**compare_kws)
            block_comparison.fit(y_pred_block,
                                 _vars2compare).correct_multi_tests()

            out['block_comparisons'] = block_comparison

        # survival
        if survival_df is not None:
            out['block_survival'] = get_survival(survival_df=survival_df,
                                                 y=y_pred_block)

    ##########################
    # process data to return #
    ##########################
    info = {}
    info['n_samples'] = n_samples
    info['n_views'] = n_views
    info['n_joint_comp'] = n_joint_comp
    info['n_view_comp'] = n_view_comp
    info['n_blocks'] = n_blocks_est
    info['joint_cluster_labels'] = joint_cluster_labels
    info['view_cluster_labels'] = view_cluster_labels
    info['view_cl_idxs'] = view_cl_idxs
    info['view_cl_labels'] = view_cl_labels
    info['block_labels'] = block_labels
    info['zero_thresh'] = zero_thresh
    info['block_summary'] = block_summary
    out['info'] = info

    return out
