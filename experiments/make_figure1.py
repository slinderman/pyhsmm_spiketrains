"""
Make the first figure. We show the following subfigures
1. Example of synthetic data
2. True transition matrix
3. True firing rates
"""
import os
import numpy as np

import matplotlib
from matplotlib.colorbar import Colorbar
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors
from hips.plotting.layout import *

from experiment_helper import load_synth_data

def make_figure1(model, S, Z, A, lmbdas, figdir='.'):
    """
    Plot the spike train matrix and the vector of states
    :param S:
    :param Z:
    :return:
    """
    N_used = len(model.used_states)
    A = A[:N_used, :N_used]
    lmbdas = lmbdas[:N_used, :]
    T,C = S.shape
    fig = create_figure((5,3.5))

    # S = S[:1000]
    # Z = Z[:1000]

    # Plot the spike train
    ax = create_axis_at_location(fig, 1,2.4,2.75,1)
    plt.figtext(.4/5, 3.3/3.5, "A")
    px = (10,1)
    im = ax.imshow(np.kron(S.T, np.ones(px)),
              cmap='Greys')

    # Set the tick labels
    yticks = np.arange(C+1, step=10)
    ax.set_yticks(yticks * px[0] + px[0]/2)
    ax.set_yticklabels(map(lambda y: '%d' % y, yticks))
    ax.set_xticklabels([])
    ax.set_ylabel('Neuron')
    ax.set_title('Spike train')

    # Add colorbar
    S_max = np.amax(S)
    cbax = create_axis_at_location(fig, 3.9, 2.55, 0.1,.7)
    Colorbar(cbax, im, ticks=np.arange(0,16,step=5))
    cbax.set_ylabel('# Spikes')

    # Add the latent state below
    zax = create_axis_at_location(fig, 1,2.,2.75,.3)
    plt.figtext(.4/5, 2.3/3.5, "B")
    px = (50,1)
    zim = zax.imshow(np.kron(Z[None,:], np.ones(px)),
              cmap='YlOrRd', vmin=0, vmax=N_used)
    zax.set_xlabel('Time index')
    zax.set_yticks([])
    zax.set_title('Latent state')

    cbax = create_axis_at_location(fig, 3.9, 2.075, 0.1,.15)
    Colorbar(cbax, zim, ticks=[0,N_used-1])
    cbax.set_yticklabels([1, N_used])

    # Plot the true transition matrix
    left = 0.5
    bottom = 0.5
    height = 1.0
    width = 1.0
    px = 10

    ax = create_axis_at_location(fig, left,bottom,width,height)
    plt.figtext(.075/5, 1.5/3.5, "C")
    im = _plot_transition_matrix(ax, A)
    ax.set_title('Transition Matrix')

    cbax = create_axis_at_location(fig, left+width+0.05, bottom, 0.1, height)
    Colorbar(cbax, im, ticks=np.arange(0,1.1,step=0.25))
    # cbax.set_ylabel('')

    # Plot the firing rate matrix
    scale = C/float(N_used)
    ax = create_axis_at_location(fig, 3, 0.5, scale*1.0, 1.0)
    plt.figtext(2.475/5, 1.5/3.5, "D")
    im = ax.imshow(np.kron(lmbdas, np.ones((px,px))), cmap='Greys', interpolation="none")
    ax.set_xlabel('Neuron')
    ax.set_xticks(np.arange(px/2, C*px, step=10*px))
    ax.set_xticklabels(np.arange(0, C, step=10))
    ax.set_ylabel('State')
    ax.set_yticks(np.arange(px/2,N_used*px, step=5*px))
    ax.set_yticklabels(np.arange(0,K, step=5))
    ax.set_title('Firing rates')

    # Add colorbar for transition matrix
    # from matplotlib.colorbar import Colorbar
    cbax = create_axis_at_location(fig, 3+scale+0.05, 0.5, 0.1, 1.0)
    Colorbar(cbax, im)
    cbax.set_ylabel('Spikes/bin')

    plt.savefig(os.path.join(figdir, "figure1.pdf"))
    plt.savefig(os.path.join(figdir, "figure1.png"))
    plt.show()

def _plot_transition_matrix(ax, A):

    px = 10
    K = A.shape[0]
    im = ax.imshow(np.kron(A, np.ones((px,px))), cmap='Greys',
                   vmin=0, vmax=1.0, interpolation="none")
    ax.set_xlabel('$S_{t+1}$')
    ax.set_xticks(np.arange(px/2,K*px, step=5*px))
    ax.set_xticklabels(np.arange(0,K, step=5))
    ax.set_ylabel('$S_{t}$')
    ax.set_yticks(np.arange(px/2,K*px, step=5*px))
    ax.set_yticklabels(np.arange(0,K, step=5))

    return im

if __name__ == "__main__":
    # Load synthetic dataset
    T = 2000
    K = 100
    N = 50
    T_test = 200
    version = 1
    modeltype = "hdp-hmm"
    runnum = 1
    results_dir = os.path.join("results",
                               "synth_%s_T%d_K%d_N%d_v%d" % (modeltype, T, K, N, version),
                               "run%03d" % runnum)

    model, S_train, _, S_test, _ = \
        load_synth_data(T, K, N, T_test=T_test,
                        model=modeltype, version=version)

    model.relabel_by_usage()
    Z_train = model.stateseqs[0]
    A = model.A
    lmbdas = model.rates

    make_figure1(model, S_train, Z_train, A, lmbdas, figdir=results_dir)

