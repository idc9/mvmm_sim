import os
from joblib import load
from glob import glob
from tqdm import tqdm
import pandas as pd


def aggregate_sims(save_dir, delete_files=False,
                   include_extra_data=False):
    """
    Aggregates data from each monte-carlo iteration
    and returns a dict of results.

    Parameters
    ----------
    save_dir: str
        Folder containing the MC results

    delete_files: bool
        Delete each MC results file.
    """

    _results_fpaths = glob('{}/sim_res_*'.format(save_dir))

    # filter log files
    # TODO: can remove this
    results_fpaths = []
    for fpath in _results_fpaths:
        if '.txt' not in fpath:
            results_fpaths.append(fpath)

    # results_fpaths = glob('{}/sim_res_*'.format(save_dir))
    if len(results_fpaths) == 0:
        raise ValueError('No simulation results found in {}'.format(save_dir))

    # load results from each job
    clust_results_agg = []
    clf_results_agg = []
    metadata_agg = []
    bd_summary_agg = []
    configs_agg = []
    for fpath in tqdm(results_fpaths):
        sim_res = load(fpath)

        clust_results_agg.append(sim_res['clust_results'])
        clf_results_agg.append(sim_res['clf_results'])
        for k in sim_res['bd_summary'].keys():
            bd_summary_agg.append(pd.DataFrame(sim_res['bd_summary'][k]))
        metadata_agg.append(sim_res['metadata'])
        configs_agg.append(sim_res['config'])

        if delete_files:
            os.remove(fpath)

    clust_results_agg = pd.concat(clust_results_agg).reset_index(drop=True)

    if len(bd_summary_agg) > 0:
        bd_summary_agg = pd.concat(bd_summary_agg).reset_index(drop=True)

    else:
        bd_summary_agg = None

    results = {'clust_results': clust_results_agg,
               'clf_results': clf_results_agg,
               'bd_summary': bd_summary_agg,
               'metadata': metadata_agg,
               'config': configs_agg}

    fpath = os.path.join(save_dir, 'sim_metadata')
    sim_metadata = load(fpath)
    results['sim_metadata'] = sim_metadata
    if delete_files:
        os.remove(fpath)

    # extra data
    extra_mc_0_fpaths = glob('{}/extra_data_mc_0_*'.format(save_dir))
    extra_data_mc_0 = {'fit_models': {}, 'X_tr': {}, 'Y_tr': {}}
    for extra_fpath in extra_mc_0_fpaths:
        n_samples = os.path.basename(extra_fpath).split('n_samples_')[1]
        n_samples = int(n_samples)
        exdat = load(extra_fpath)

        extra_data_mc_0['Pi'] = exdat['Pi']
        extra_data_mc_0['view_params'] = exdat['view_params']
        extra_data_mc_0['fit_models'][n_samples] = exdat['fit_models']
        extra_data_mc_0['X_tr'][n_samples] = exdat['tr_data']['X_tr']
        extra_data_mc_0['Y_tr'][n_samples] = exdat['tr_data']['Y_tr']

        if delete_files:
            os.remove(extra_fpath)

    return results, extra_data_mc_0
