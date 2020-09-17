import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from mvmm.multi_view.block_diag.graph.bipt_community import community_summary
from mvmm.multi_view.block_diag.graph.linalg import eigh_sym_laplacian_bp, \
    get_unnorm_laplacian_bp
from mvmm.linalg_utils import eigh_wrapper


def summarize_bd(D, n_blocks, zero_thresh=None, lap='sym'):

    assert lap in ['sym', 'un']
    comm_summary, Pi_comm = community_summary(D, zero_thresh=zero_thresh)

    print(comm_summary)

    plt.figure(figsize=(8, 4))
    if lap == 'sym':
        evals = eigh_sym_laplacian_bp(D)[0]
    else:
        Lun = get_unnorm_laplacian_bp(D)
        evals = eigh_wrapper(Lun)[0]

    plt.subplot(1, 2, 1)
    plt.plot(evals, marker='.')
    plt.title('all evals of L_{}'.format(lap))
    plt.subplot(1, 2, 2)
    plt.plot(evals[-n_blocks:], marker='.')
    plt.title('smallest {} evals'.format(n_blocks))
    print('evals', evals)

    # print('found {} communities of sizes {}'.format(summary['n_communities'], summary['comm_shapes']))

    plt.figure()
    sns.heatmap(Pi_comm, cmap='Blues', square=True, cbar=False, vmin=0)
    plt.xlabel('View 1 clusters')
    plt.ylabel('View 2 clusters')


def plot_loss_history(loss_vals, loss_name='loss val', title=None):

    steps = np.arange(1, len(loss_vals) + 1)

    diffs = np.diff(loss_vals)
    sgns = np.sign(diffs)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, loss_vals, marker='.', color='black')
    plt.xlabel('step')
    plt.ylabel(loss_name)
    if title is not None:
        plt.title(title)

    plt.subplot(1, 2, 2)
    plt.plot(steps[:-1], np.log10(np.abs(diffs)), marker='.', color='black')
    s2c = {-1: 'red', 1: 'green', 0: 'grey'}
    colors = np.array([s2c[s] for s in sgns])
    plt.scatter(steps[:-1], np.log10(np.abs(diffs)), color=colors, s=100)

    plt.xlabel('step')
    plt.ylabel('log10(diff)')
