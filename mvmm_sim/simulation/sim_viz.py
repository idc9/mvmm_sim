import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# from mvmm.viz_utils import set_xaxis_int_ticks
from mvmm.single_view.opt_diagnostics import plot_opt_hist
from mvmm.multi_view.MVMM import MVMM

from mvmm_sim.simulation.opt_viz import plot_loss_history
from mvmm.clustering_measures import MEASURE_MIN_GOOD

_mod_pals = ['Blues', 'Reds', 'Greens',
             'Purples', 'Greys', 'Oranges']  # TODO: add more

# _mod_line_styles = ['solid', 'dashed', 'dotted', 'dashdot',
#                     'losely dashed', 'losely dotted']


def get_best_tune_expers(df, by='n_comp_est', measure='bic'):

    assert measure in MEASURE_MIN_GOOD.keys()
    idxs_to_keep = []
    # for each monte-carlo/n_samples combination
    for _, exper_df in df.groupby(['mc_index', 'n_samples']):

        # fits that have the same by value
        for val, n_samp_df in exper_df.groupby(by):
            if MEASURE_MIN_GOOD[measure]:
                idx_to_keep = n_samp_df[measure].idxmin()
            else:
                idx_to_keep = n_samp_df[measure].idxmax()

            idxs_to_keep.append(idx_to_keep)

    idxs_to_drop = set(df.index).difference(idxs_to_keep)

# def get_best_tune_expers(df, by='n_comp_est', measure='bic', min_good=True):
#     idxs_to_keep = []
#     # for each monte-carlo/n_samples combination
#     for _, exper_df in df.groupby(['mc_index', 'n_samples']):

#         # fits that have the same by value
#         for val, n_samp_df in exper_df.groupby(by):

#             best_tuning_idx = n_samp_df['best_tuning_idx'].values[0]

#             # if the best tuning index has this value, we should keep
#             # the best tuning index
#             if best_tuning_idx in n_samp_df['tune_idx'].values:
#                 b = n_samp_df[n_samp_df['tune_idx'] == best_tuning_idx]
#                 assert b.shape[0] == 1
#                 idx_to_keep = b.index[0]
#             else:

#                 if min_good:
#                     idx_to_keep = n_samp_df[measure].idxmin()
#                 else:
#                     idx_to_keep = n_samp_df[measure].idxmax()

#             idxs_to_keep.append(idx_to_keep)

#     idxs_to_drop = set(df.index).difference(idxs_to_keep)

    return df.drop(index=idxs_to_drop).reset_index(drop=True)

# def thin_mvmm_gs_clust_results(clust_results):
#     """
#     Multiple lambda values for the MVMM grid search can have the same
#     number of estimated components. Pick the lambda value with the best
#     BIC.
#     """

#     mvmm_mask = clust_results['model'] == 'mvmm'
#     mvmm_results = clust_results[mvmm_mask]

#     idxs_to_keep = []
#     for _, exper_df in \
#             mvmm_results.groupby(['mc_index', 'n_samples', 'view']):
#         for n_comp_est, df in exper_df.groupby('n_comp_est'):
#             # idx_to_keep = df['bic'].idxmin()

#             best_tuning_idx = df['best_tuning_idx'].values[0]
#             if best_tuning_idx in df['tune_idx'].values:
#                 b = df[df['tune_idx'] == best_tuning_idx]
#                 assert b.shape[0] == 1
#                 idx_to_keep = b.index[0]

#             else:
#                 idx_to_keep = df['bic'].idxmin()

#             idxs_to_keep.append(idx_to_keep)

#     idxs_to_drop = set(mvmm_results.index).difference(idxs_to_keep)

#     return clust_results.drop(index=idxs_to_drop).reset_index(drop=True)


def plot_grouped_x_vs_y(dfs, group_var, value_var,
                        labels=None, plot_interval=False, force_x_ticks=True,
                        colors=sns.color_palette("Set2"),
                        # ls=_mod_line_styles,
                        markers='.'):

    assert not plot_interval or plot_interval in ['quant', 'std']

    n_df = len(dfs)

    # if type(ls) == str:
    #     ls = [ls] * n_df

    if type(markers) == str:
        markers = [markers] * n_df
    assert len(markers) == n_df

    # assert n_df <= 4  # figure out >= 5 customization e.g. linestyles
    if labels is not None:
        assert len(labels) == n_df

    if labels is None:
        labels = [None for d in range(n_df)]

    all_x_ticks = set()

    for i, df in enumerate(dfs):

        color = colors[i]

        summary = get_grouped_summary(df, group_var=group_var,
                                      value_var=value_var)
        plt.plot(summary.index, summary['mean'],
                 marker=markers[i], color=color,  # ls=ls[i],
                 label=labels[i])

        if plot_interval:

            if plot_interval == 'std':
                lower = summary['lower_m_sqn_std']
                upper = summary['upper_m_sqn_std']

            elif plot_interval == 'quant':
                lower = summary['lower_quant']
                upper = summary['upper_quant']

            plt.fill_between(x=summary.index.values,
                             y1=lower, y2=upper,
                             color=color, alpha=.4, edgecolor=None)

        all_x_ticks = all_x_ticks.union(summary.index)

    if force_x_ticks:
        all_x_ticks = np.sort(list(all_x_ticks)).astype(int)
        plt.xticks(all_x_ticks, all_x_ticks)

    plt.xlabel(group_var)
    plt.ylabel(value_var)

##################################
# n_comp vs. clustering accuracy #
##################################


def plot_grouped_var_per_n(dfs, group_var, value_var,
                           labels=None, x_label=None, y_label=None,
                           plot_summary=True, plot_scatter=False,
                           plot_interval=False,
                           highlight_selected=False,
                           force_x_ticks=True,
                           pals=_mod_pals,
                           select_metric='bic',
                           lw=2, s=50):

    assert not plot_interval or plot_interval in ['quant', 'std']

    # check input
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    if isinstance(pals, str):
        pals = [pals]

    n_df = len(dfs)
    assert n_df <= 4  # figure out >= 5 customization e.g. linestyles
    n_samples_seq = sorted(set(dfs[0]['n_samples'].values))
    if labels is not None:
        assert len(labels) == n_df
    assert len(pals) >= n_df

    n_samp_colors = [sns.color_palette(pals[d], len(n_samples_seq))
                     for d in range(n_df)]

    if highlight_selected:
        # measure and n_comp_est at BIC selcted number of components
        sel_measure_ests = [get_best_tune_metrics(dfs[d],
                                                  [group_var, value_var],
                                                  select_metric=select_metric)
                            for d in range(n_df)]
        sel_measure_ests = [df.groupby(['n_samples']).mean()
                            for df in sel_measure_ests]

    all_x_ticks = set()

    # plot results for each value of number of samples
    # ret_measure_summary = {}  # return this
    for n_idx, n_samples in enumerate(n_samples_seq):

        # get data for this number of smaples
        dfs_fixed_n = [dfs[d].query('n_samples == {}'.format(n_samples))
                       for d in range(n_df)]

        measure_summary = [get_grouped_summary(dfs_fixed_n[d],
                                               group_var=group_var,
                                               value_var=value_var)
                           for d in range(n_df)]

        # ret_measure_summary[n_samples] = measure_summary

        # label the darkset color
        if n_idx == len(n_samples_seq) - 1 and labels is not None:
            _labels = labels
        else:
            _labels = [None for d in range(n_df)]

        # plot results for each model
        for d in range(n_df):
            color = n_samp_colors[d][n_idx]

            # plot mean
            if plot_summary:
                plt.plot(measure_summary[d].index.values,
                         measure_summary[d]['mean'],
                         marker='x',
                         lw=lw,  # ls=_mod_line_styles[d],
                         color=color, label=_labels[d])

            # label number of samples
            add_label(measure_summary[d], text=n_samples,
                      color=color, use_max=True)

            # scatter plot of all points
            if plot_scatter:
                plt.scatter(dfs_fixed_n[d][group_var],
                            dfs_fixed_n[d][value_var],
                            marker='.',
                            color=color, s=s)

            # plot upper/lower quantiles
            if plot_interval:

                if plot_interval == 'std':
                    lower = measure_summary[d]['lower_m_sqn_std']
                    upper = measure_summary[d]['upper_m_sqn_std']

                elif plot_interval == 'quant':
                    lower = measure_summary[d]['lower_quant']
                    upper = measure_summary[d]['upper_quant']

                plt.fill_between(x=measure_summary[d].index.values,
                                 y1=lower, y2=upper,
                                 color=color, alpha=.4, edgecolor=None)

            # plot measure at selected value
            if highlight_selected:
                sel_ncomp = sel_measure_ests[d].loc[n_samples, group_var]
                measure_at_sel_ncomp = sel_measure_ests[d].\
                    loc[n_samples, value_var]
                plt.scatter(sel_ncomp, measure_at_sel_ncomp,
                            color=color,
                            marker='*', lw=1, s=300, zorder=1000)

            all_x_ticks = all_x_ticks.union(measure_summary[d].index.values)

    if force_x_ticks:
        all_x_ticks = np.sort(list(all_x_ticks)).astype(int)
        plt.xticks(all_x_ticks, all_x_ticks)

    if x_label is None:
        x_label = group_var
    if y_label is None:
        y_label = value_var

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # return ret_measure_summary


def get_grouped_summary(df, group_var, value_var, q=0.95):
    """
    Computes mean, lower and upper quantiles a given measure.
    """
    def mean_sqrtn_std(x, n_std=1):
        """
        returns mean + n_std * std / sqrt(n)
        """

        n = len(x)
        m = np.mean(x)
        s = np.std(x)

        return m + n_std * s / np.sqrt(n)

    return df.groupby(group_var)[value_var].\
        agg(mean=np.mean,
            std=np.std,
            median=np.median,
            upper_m_sqn_std=lambda x: mean_sqrtn_std(x, n_std=1),
            lower_m_sqn_std=lambda x: mean_sqrtn_std(x, n_std=-1),
            upper_quant=lambda x: np.quantile(x, q=q),
            lower_quant=lambda x: np.quantile(x, q=1 - q)).\
        sort_index()


def add_label(res, text, color=None, use_max=True):
    """
    Adds sample size label
    """

    if use_max:
        lab_pt_idx = np.argmax(np.array(res.index))
        # res['n_comp_est'].astype(int).idxmax()
    else:
        lab_pt_idx = np.argmin(np.array(res.index))

    x_coord = res.index[lab_pt_idx]  # res.iloc[lab_pt_idx]['n_comp_est']
    y_coord = res.iloc[lab_pt_idx, :]['mean']  # res[x_coord]  # ['ars']
    plt.text(x=x_coord, y=y_coord, s=text, color=color)


def value_per_n_samples_hline(df, value_var, lambd=.9, plot_interval=False):
    """
    Pools value by number of samples, adds horiztonal lines to plot
    """
    assert not plot_interval or plot_interval in ['quant', 'std']
    n_samples_seq = sorted(set(df['n_samples'].values))
    n_samp_colors = sns.color_palette('Greys', len(n_samples_seq))

    summary = get_grouped_summary(df, group_var='n_samples',
                                  value_var=value_var)

    for i, n_samples in enumerate(n_samples_seq):
        color = n_samp_colors[i]
        val = summary.loc[n_samples, 'mean']

        plt.axhline(val, color=color)

        if plot_interval:

            if plot_interval == 'std':
                lower = summary.loc[n_samples, 'lower_m_sqn_std']
                upper = summary.loc[n_samples, 'upper_m_sqn_std']

            elif plot_interval == 'quant':
                lower = summary.loc[n_samples, 'lower_quant']
                upper = summary.loc[n_samples, 'upper_quant']

            plt.axhspan(lower, upper, alpha=.5, color=color)

        text = '{}'.format(n_samples)

        x_min, x_max = plt.gca().get_xlim()

        x_coord = lambd * x_max + (1 - lambd) * x_min
        plt.text(x=x_coord, y=val, s=text, color=color)


##################################
# n_samples vs. estimated n_comp #
##################################
def plot_n_samples_vs_best_tune_metric(dfs, measure,
                                       labels=None, plot_interval=False,
                                       colors=sns.color_palette("Set2"),
                                       # ls=_mod_line_styles,
                                       select_metric='bic',
                                       markers='.',
                                       force_x_ticks=True):

    assert not plot_interval or plot_interval in ['quant', 'std']

    n_df = len(dfs)

    # if type(ls) == str:
    #     ls = [ls] * n_df

    if type(markers) == str:
        markers = [markers] * n_df
    assert len(markers) == n_df

    if labels is not None:
        assert len(labels) == n_df

    if labels is None:
        labels = [None for d in range(n_df)]

    best_measure_ests = [get_best_tune_metrics(dfs[d], [measure],
                                               select_metric=select_metric)
                         for d in range(n_df)]

    # def get_summary(df):
    #     df = df.loc[:, ['n_samples', measure]]
    #     df[measure] = df[measure].astype(float)
    #     return df.groupby('n_samples')[measure].\
    #         agg(mean=np.mean,
    #             std=np.std,
    #             median=np.median,
    #             upper_quant=lambda x: np.quantile(x, q=0.95),
    #             lower_quant=lambda x: np.quantile(x, q=0.05)).\
    #         sort_index()
    # summaries = [get_summary(best_measure_ests[d]) for d in range(n_df)]

    summaries = [get_grouped_summary(df=best_measure_ests[d],
                                     group_var='n_samples',
                                     value_var=measure)
                 for d in range(n_df)]

    all_x_ticks = set()

    for d in range(n_df):
        color = colors[d]

        plt.plot(summaries[d].index.values,
                 summaries[d]['mean'],
                 color=color, marker=markers[d],  # ls=ls[d],
                 label=labels[d])

        if plot_interval:

            if plot_interval == 'std':
                lower = summaries[d]['lower_m_sqn_std']
                upper = summaries[d]['upper_m_sqn_std']

            elif plot_interval == 'quant':
                lower = summaries[d]['lower_quant']
                upper = summaries[d]['upper_quant']

            plt.fill_between(x=summaries[d].index.values,
                             y1=lower,
                             y2=upper,
                             color=color, alpha=.4, edgecolor=None)

        all_x_ticks = all_x_ticks.union(summaries[d].index.values)

    if force_x_ticks:
        all_x_ticks = np.sort(list(all_x_ticks)).astype(int)
        plt.xticks(all_x_ticks, all_x_ticks)

    plt.xlabel('Number of samples')
    plt.ylabel(measure)


def get_best_tune_metrics(df, cols, select_metric='bic'):
    """
    Gets measures for best tuning parameter.
    """
    def getter(df):
        """

        """

        best_tuning_idx = df.iloc[0, ]['{}_best_tune_idx'.
                                       format(select_metric)]

        # best_tuning_idx = df.iloc[0, ]['best_tuning_idx']
        best_run = df[df['tune_idx'] == best_tuning_idx]
        assert best_run.shape[0] == 1
        return pd.Series({c: best_run[c].values[0] for c in cols})

    return df.groupby(['mc_index', 'n_samples']).apply(getter).reset_index()


# def get_est_n_comp(df):
#     """

#     """
#     best_tuning_idx = df.iloc[0, ]['best_tuning_idx']
#     best_run = df[df['tune_idx'] == best_tuning_idx]
#     assert best_run.shape[0] == 1
#     return best_run['n_comp_est'].values[0]


# def get_best_tune_n_comp_est(df):
#     blah = df[['mc_index', 'n_samples',
#                'best_tuning_idx', 'tune_idx', 'n_comp_est']]
#     blah = blah.groupby(['mc_index',
#                          'n_samples']).apply(get_est_n_comp).reset_index()
#     blah = blah.rename(columns={0: 'n_comp_est'})
#     return blah


##########################
# classification results #
##########################

def plot_clf_results(results):
    """
    Plots classification test set error vs. number of samples.
    """

    # extract data
    clf_results = results['clf_results']

    clf_tst_acc = [clf_results[i]['cat']['tst']['acc']
                   for i in range(len(clf_results))]

    n_samples_tr = [results['metadata'][i]['n_samples_tr']
                    for i in range(len(results['metadata']))]

    clf_df = pd.DataFrame.from_dict({'n_samples_tr': n_samples_tr,
                                     'clf_tst_acc': clf_tst_acc})

    clf_summary = clf_df.\
        groupby('n_samples_tr')['clf_tst_acc'].\
        agg(mean=np.mean,
            upper=lambda x: np.quantile(x, q=0.95),
            lower=lambda x: np.quantile(x, q=0.05)).\
        sort_index()

    # Make plots
    plt.plot(clf_summary.index.values, clf_summary['mean'],
             marker='.', color='black')

    plt.fill_between(clf_summary.index.values,
                     y1=clf_summary['lower'],
                     y2=clf_summary['upper'],
                     color='black', alpha=.2)
    plt.ylim([0, 1])
    plt.yticks(np.linspace(start=0.0, stop=1, num=11))
    plt.xticks(clf_summary.index.values)
    plt.ylabel('LDA test accuracy')
    plt.xlabel('Number of training samples')


def save_fig(fpath, dpi=100, bbox_inches='tight', close=True):
    plt.savefig(fpath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def get_sim_summary(results):

    metadata = results['metadata']
    n_sims = len(metadata)

    sim_summary = {}
    sim_summary = {k: [metadata[s][k] for s in range(n_sims)]
                   for k in ['n_samples_tr', 'tot_runtime']}
    sim_summary.update({'n_tune_params': [len(metadata[s]['mvmm_param_grid'])
                        for s in range(n_sims)]})

    return pd.DataFrame(sim_summary)

###########
# runtime #
###########


def get_fit_runtimes_overall(results):
    fit_runtimes = []

    for sim_idx, res in enumerate(results['metadata']):
        n_samples_tr = res['n_samples_tr']

        for model_name, time in res['fit_runtimes'].items():
            fit_runtimes.append({'n_samples': n_samples_tr,
                                 # 'sim_idx': sim_idx,
                                 'time': time,
                                 'model_name': model_name})

    fit_runtimes = pd.DataFrame.from_dict(fit_runtimes)

    # convert to minutes
    fit_runtimes['time'] = fit_runtimes['time'] / 60
    fit_runtimes['log_time'] = np.log10(fit_runtimes['time'])

    return fit_runtimes


def plot_fit_runtimes(fit_runtimes, log=True):

    if log:
        value_var = 'log_time'
        y_lab = 'log10(Minutes)'
    else:
        value_var = 'time'
        y_lab = 'Minutes'

    for model_name, df in fit_runtimes.groupby('model_name'):

        summary = get_grouped_summary(df, group_var='n_samples',
                                      value_var=value_var)

        p = plt.plot(summary.index, summary['mean'],
                     marker='.', label=model_name)
        color = p[0].get_color()

        plt.scatter(df['n_samples'], df[value_var], color=color)

        x_coord = max(summary.index)
        y_coord = summary.loc[x_coord]['mean']
        text = model_name
        plt.text(x=x_coord, y=y_coord, s=text, color=color)

    plt.legend()

    plt.xlabel('Number of samples')
    plt.ylabel(y_lab)


def plot_grid_search_param_vs_fit_time_by_n(extra_data):
    df = []
    n_samples_seq = list(extra_data['fit_models'].keys())
    for n_samples in n_samples_seq:
        ts_gs_mvmm = extra_data['fit_models'][n_samples]['ts_mvmm']
        param_name = list(ts_gs_mvmm.param_grid.keys())[0]
        param_vals = ts_gs_mvmm.param_grid[param_name]

        for i, val in enumerate(param_vals):
            fit_time = ts_gs_mvmm.metadata_['fits'][i]['fit_time']

            df.append({param_name: val,
                       'n_samples': n_samples,
                       'fit_time': fit_time})

    df = pd.DataFrame(df)
    plot_grouped_var_per_n(df, group_var=param_name, value_var='fit_time')

    plt.ylabel('Seconds')


########################
# Optimization history #
########################

def plot_model_opt_history(extra_data, model_name, n_samples,
                           save_folder=None, inches=10):

    model = extra_data['fit_models'][n_samples][model_name]

    stub = '{}_{}_'.format(model_name, n_samples)

    def save(name):
        if save_folder is not None:
            fpath_stub = os.path.join(save_folder, stub)
            save_fig(fpath=fpath_stub + name + '.png')

    if model_name == 'view_gmm':
        None

    elif model_name == 'full_mvmm':

        plot_opt_hist(loss_vals=model.opt_data_['history']['loss_val'],
                      init_loss_vals=model.opt_data_['init_loss_vals'],
                      loss_name='Obs neg log-likelihood',
                      title='MVMM, n = {}'.format(n_samples),
                      inches=inches)

        save('loss_history')

    elif model_name == 'ts_mvmm' and hasattr(model.best_estimator_.final_,
                                             'bd_weights_'):

        ########################
        # block diagonal model #
        ########################

        estimator = model.best_estimator_

        ##############
        # start MVMM #
        ##############
        loss_vals = estimator.start_.opt_data_['history']['loss_val']
        plot_opt_hist(loss_vals=loss_vals,
                      loss_name='Obs neg log-likelihood',
                      title='start, MVMM, n = {}'.format(n_samples),
                      inches=inches)

        save('start_loss_history')

        ###########################
        # final, adaptive history #
        ###########################
        adpt_history = estimator.\
            final_.opt_data_['adpt_opt_data']['adapt_pen_history']['opt_data']

        n_steps = np.cumsum([len(adpt_history[i]['history']['raw_eval_sum'])
                             for i in range(len(adpt_history))])
        # n_steps = n_steps[0:len(n_steps) -1] # ignore last step

        raw_eval_sum = np.concatenate([
            adpt_history[i]['history']['raw_eval_sum']
            for i in range(len(adpt_history))])

        loss_vals = np.concatenate([
            adpt_history[i]['history']['loss_val']
            for i in range(len(adpt_history))])

        # loss history
        plot_opt_hist(loss_vals=loss_vals,
                      loss_name='loss',
                      title='final, adaptive, MVMM, n = {}'.format(n_samples),
                      step_vals=n_steps,
                      inches=inches)

        save('final_adaptive_loss_history')

        # eigen values
        plt.figure(figsize=[inches, inches])
        plt.plot(np.log10(raw_eval_sum), marker='.')
        plt.ylabel('log10(sum smallest evals)')
        plt.xlabel('step')
        plt.title('Eigenvalue history')
        for s in n_steps:
            plt.axvline(s - 1, color='grey')

        save('final_adaptive_eval_history')

        #######################
        # final, fine tuning #
        ######################

        init_loss_vals = estimator.fit_data_['init_loss_vals']
        loss_vals = estimator.\
            final_.opt_data_['ft_opt_data']['history']['obs_nll']

        plot_opt_hist(loss_vals=loss_vals,
                      init_loss_vals=init_loss_vals,
                      loss_name='Obs neg log-likelihood',
                      title='fine tune block diagonal MVMM, n = {}'.format(n_samples),
                      inches=inches)

        save('final_fine_tune_loss_history')

    elif model_name == 'ts_mvmm' and hasattr(model.best_estimator_.final_,
                                             'pen'):

        ######################
        # Log penalized MVMM #
        ######################
        estimator = model.best_estimator_

        # start estimator
        loss_vals = estimator.start_.opt_data_['history']['loss_val']
        plot_opt_hist(loss_vals=loss_vals,
                      loss_name='Obs neg log-likelihood',
                      title='start, MVMM, n = {}'.format(n_samples),
                      inches=inches)

        save('start_loss_history')

        # final estimator
        loss_vals = estimator.final_.opt_data_['history']['loss_val']
        plot_opt_hist(loss_vals=loss_vals,
                      loss_name='Log penalized neg-likelihood',
                      title='final, log-pen MVMM, n = {}'.format(n_samples),
                      inches=inches)

        save('final_loss_history')

    elif model_name == 'cat_gmm':

        ############################
        # GMM on concatenated data #
        ############################

        estimator = model.best_estimator_

        plot_opt_hist(loss_vals=estimator.opt_data_['history']['loss_val'],
                      init_loss_vals=estimator.opt_data_['init_loss_vals'],
                      loss_name='Obs neg log-likelihood',
                      title='cat-GMM, start, n = {}'.format(n_samples),
                      inches=inches)

        save('loss_history')


def plot_sp_mvmm_pen_path(init_fit_data):

    n_inits = len(np.unique(init_fit_data['init']))
    colors = sns.color_palette("Set2", n_inits)
    max_n_blocks = init_fit_data['n_blocks'].max()

    plt.figure(figsize=(10, 10))
    for init in range(n_inits):
        init_df = init_fit_data.query("init == @init")
        color = colors[init]

        plt.subplot(2, 2, 1)
        plt.plot(init_df['pen_val'], init_df['n_blocks'],
                 marker='.', alpha=.5, color=color)
        plt.xlabel('Penalty Value')
        plt.ylabel('Number of blocks')

        plt.subplot(2, 2, 2)
        plt.plot(init_df['pen_val'], init_df['obs_nll'],
                 marker='.', alpha=.5, color=color)
        plt.xlabel('Penalty value')
        plt.ylabel('Obs NLL')

        plt.subplot(2, 2, 3)
        plt.plot(init_df['pen_idx'], init_df['n_blocks'],
                 marker='.', alpha=.5, color=color)
        plt.xlabel('Penalty index')
        plt.ylabel('Number of blocks')

        init_df_at_max = init_df.query("n_blocks == @max_n_blocks")
        if init_df_at_max.shape[0] > 0:
            idx_max = init_df_at_max['pen_idx'].min()

            # axvline_with_tick(idx_max, color=color, ls='--', alpha=.5)
            plt.axvline(idx_max, color=color, ls='--', alpha=.5)

        plt.subplot(2, 2, 4)
        plt.plot(init_df['pen_idx'], init_df['pen_val'],
                 marker='.', color=color)
        plt.xlabel('Penalty index')
        plt.ylabel('Penaly value')


def plot_sp_opt_history(sp_mvmm_gs, save_dir):
    for idx, est in enumerate(sp_mvmm_gs.estimators_):
        n_blocks = sp_mvmm_gs.est_n_blocks_[idx]

        if type(est) == MVMM:
            loss_vals = est.opt_data_['history']['obs_nll']
            loss_name = 'Obs NLL'

        else:
            loss_vals = est.opt_data_['adpt_opt_data']['history']['loss_val']
            loss_name = "eval pen loss"

        plot_loss_history(loss_vals, loss_name=loss_name)
        plt.title('{} steps'.format(len(loss_vals)))

        os.makedirs(save_dir, exist_ok=True)
        save_fig(os.path.join(save_dir,
                 'opt_history__n_blocks_{}.png'.format(n_blocks)))


def plot_model_fit_times(timing_data, model_name,
                         name_mapping=None):

    model_runtimes = timing_data[model_name + '__runtime']
    n_fits = timing_data[model_name + '__n_fit_models'].values[0]
    n_init = timing_data['n_init'].values[0]
    n_fit_init = n_fits * n_init

    # plt.figure(figsize=(15, 5))

    if name_mapping is not None:
        model_name = name_mapping

    plt.subplot(1, 3, 1)
    plt.scatter(timing_data['n_samples_tr'], model_runtimes)
    plt.title("{} \navg simulation runtime: {:1.3f} hours".
              format(model_name, model_runtimes.mean()))
    plt.xlabel('Number of samples')
    plt.ylabel('Total runtime (hours)')
    plt.ylim(0)

    plt.subplot(1, 3, 2)
    plt.scatter(timing_data['n_samples_tr'], 60 * model_runtimes / n_init)
    plt.title("avg init runtime: {:1.2f} minutes \nn_init {}".
              format(model_runtimes.mean() / n_init * 60, n_init))
    plt.xlabel('Number of samples')
    plt.ylabel('Init runtime (minutes)')
    plt.ylim(0)

    plt.subplot(1, 3, 3)
    plt.scatter(timing_data['n_samples_tr'],
                60 * 60 * model_runtimes / n_fit_init)
    plt.title('avg fit init runtime: {:1.2f} seconds \nn fit'
              ' models: {}, n_init {}, n fit init {}'.
              format(model_runtimes.mean() / n_fit_init * 60 * 60,
                     n_fits, n_init, n_fit_init))
    plt.xlabel('Number of samples')
    plt.ylabel('Fit init runtime (seconds)')
    plt.ylim(0)
