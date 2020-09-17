import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from collections import Counter


def plot_survival(df, cat_col, dur_col='duration',
                  event_obs_col='event_obs'):

    kmf = KaplanMeierFitter()

    cnts = dict(Counter(df[cat_col]))
    for cat in np.unique(df[cat_col]):
        cat_df = df.query("{} == @cat".format(cat_col))

        label = '{} (n={})'.format(cat, cnts[cat])

        kmf.fit(durations=cat_df[dur_col],
                event_observed=cat_df[event_obs_col],
                label=label)

        kmf.plot()

    plt.ylim(0, 1)
    # plt.ylabel("Survival probability")
    plt.xlabel(dur_col)


# def cat_survival_ovo(df, cat_col, dur_col='duration',
#                      event_obs_col='event_obs',
#                      alpha=0.05,
#                      multi_test_method='fdr_bh'):

#     # one vs rest
#     results = []
#     for cat_A, cat_B in combinations(np.unique(df[cat_col]), 2):
#         df_A = df.query("{} == @cat_A".format(cat_col))
#         df_B = df.query("{} == @cat_B".format(cat_col))

#         test = logrank_test(durations_A=df_A[dur_col],
#                             durations_B=df_B[dur_col],
#                             event_observed_A=df_A[event_obs_col],
#                             event_observed_B=df_B[event_obs_col])

#         results.append([cat_A, cat_B, test.p_value])

#     results = pd.DataFrame(results)
#     results.columns = ["cat_A", "cat_B", "raw_pval"]

#     rej_corr, pval_corr, _, __ = \
#         multipletests(pvals=results['raw_pval'],
#                       alpha=alpha,
#                       method=multi_test_method)

#     results['corr_pval'] = pval_corr
#     results['reject'] = rej_corr

#     return results


# def cat_survival_ovr(df, cat_col, dur_col='duration',
#                      event_obs_col='event_obs',
#                      alpha=0.05,
#                      multi_test_method='fdr_bh'):

#     # one vs rest
#     results = []
#     for cat in np.unique(df[cat_col]):
#         df_A = df.query("{} == @cat".format(cat_col))
#         df_B = df.query("{} != @cat".format(cat_col))

#         test = logrank_test(durations_A=df_A[dur_col],
#                             durations_B=df_B[dur_col],
#                             event_observed_A=df_A[event_obs_col],
#                             event_observed_B=df_B[event_obs_col])

#         results.append([cat, test.p_value])

#     results = pd.DataFrame(results)
#     results.columns = ["cat", "raw_pval"]

#     rej_corr, pval_corr, _, __ = \
#         multipletests(pvals=results['raw_pval'],
#                       alpha=alpha,
#                       method=multi_test_method)

#     results['corr_pval'] = pval_corr
#     results['reject'] = rej_corr

#     return results
