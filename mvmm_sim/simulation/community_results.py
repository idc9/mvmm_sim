import numpy as np
import pandas as pd

from mvmm.multi_view.block_diag.graph.bipt_community import get_comm_mat
from mvmm_sim.simulation.cluster_report import cluster_report


def get_y_comm_pred_out_comm(mvmm, view_data, comm_mat):

    y_pred = mvmm.predict(view_data)
    row_idxs_pred, col_idxs_pred = mvmm._get_view_clust_idx(y_pred)

    y_comm_pred = np.zeros_like(y_pred)
    base = int(np.nanmax(comm_mat)) + 1
    n_out_of_comm = 0
    for i in range(len(y_pred)):
        c = comm_mat[row_idxs_pred[i], col_idxs_pred[i]]

        if np.isnan(c):

            c = base + n_out_of_comm
            n_out_of_comm += 1

        y_comm_pred[i] = c

    return y_comm_pred.astype(int), n_out_of_comm


def get_y_comm_pred_restrict_comm(mvmm, view_data, comm_mat):

    # get the overall cluster indices corresponding to the community blocks
    cl_idxs2keep = []
    for k in range(mvmm.n_components):
        row_idx, col_idx = mvmm._get_view_clust_idx(k)
        spect_c = comm_mat[row_idx, col_idx]
        if not np.isnan(spect_c):
            cl_idxs2keep.append(k)

    # get the probability predictions for all clusters
    prob_pred = mvmm.predict_proba(view_data)
    prob_pred = pd.DataFrame(prob_pred, columns=range(prob_pred.shape[1]))
    prob_pred = prob_pred.loc[:, cl_idxs2keep]

    # get the most likely cluster index of the cluster indices that are
    # in blocks
    y_pred = np.array([prob_pred.iloc[i, :].idxmax()
                       for i in range(prob_pred.shape[0])])
    row_idxs_pred, col_idxs_pred = mvmm._get_view_clust_idx(y_pred)

    # get predicted communities
    y_comm_pred = np.zeros(prob_pred.shape[0])
    for i in range(prob_pred.shape[0]):
        y_comm_pred[i] = comm_mat[row_idxs_pred[i], col_idxs_pred[i]]

    return y_comm_pred.astype(int)


def get_comm_pred_summary(Pi_true, Y_true, mvmm, view_data, comm_mat_est):

    # extract true communities
    comm_mat_true = get_comm_mat(Pi_true > 0)
    # comm_mat_true[np.isnan(comm_mat_true)] = -1
    y_comm_true = [comm_mat_true[Y_true[i, 0], Y_true[i, 1]]
                   for i in range(Y_true.shape[0])]

    y_comm_pred_no_out, n_out = get_y_comm_pred_out_comm(mvmm=mvmm,
                                                         view_data=view_data,
                                                         comm_mat=comm_mat_est)

    y_comm_pred_restr = get_y_comm_pred_restrict_comm(mvmm=mvmm,
                                                      view_data=view_data,
                                                      comm_mat=comm_mat_est)

    cl_report_no_out = cluster_report(y_comm_true, y_comm_pred_no_out)
    cl_report_restr = cluster_report(y_comm_true, y_comm_pred_restr)

    return cl_report_no_out, cl_report_restr, n_out
