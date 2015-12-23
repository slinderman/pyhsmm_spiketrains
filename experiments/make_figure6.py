"""
Plot the MCMC samples for the hippocampal dataset.
"""
import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase, Colorbar

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from hips.plotting.layout import *

from pyhsmm_spiketrains.internals.utils import split_train_test

from experiment_helper import load_hipp_data

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

def plot_results(results,
                 data, burnin=5,
                 figdir='.'):
    """
    Plot the true and inferred transition matrices using a variety
    of inference algorithms.

    :param hmm:
    :return:
    """
    N_samples = np.array(results.loglikes).size

    model = results.samples
    model.relabel_by_usage()
    N_used = results.N_used[-1]
    A = model.A[:N_used,:N_used]
    lmbdas = model.rates[:N_used,:].T
    C,K = lmbdas.shape

    px = 10
    stepK = 25
    stepC = 10

    # Plot the log likelihood as a function of iteration
    fig = create_figure((5,5))

    # Num States vs Iteration
    # ax = fig.add_subplot(gs[0,:M])
    ax = create_axis_at_location(fig, 0.5, 3.75, 1.75, 1.)
    ax.plot(np.arange(burnin, N_samples), results.N_used[burnin:],
            color=allcolors[1])
    ax.set_xlabel('Iteration $(\\times 10^3)$')
    ax.set_xlim([burnin, N_samples])
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('Number of States')
    plt.figtext(.05/5, 4.8/5, "A")

    # LL vs Iteration
    lls = np.array(results.loglikes[burnin:])
    # ax = fig.add_subplot(gs[0,M:])
    ax = create_axis_at_location(fig, 3., 3.75, 1.75, 1.)
    ax.plot(np.arange(burnin, N_samples), lls / 1000.,
            color=allcolors[1])
    ax.set_xlabel('Iteration $(\\times 10^3)$')
    ax.set_xlim([burnin, N_samples])
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('Log Lkhd. $(\\times 10^3)$')
    plt.figtext(2.5/5, 4.8/5, "B")

    # alpha vs Iteration
    # ax = fig.add_subplot(gs[1,:M])
    ax = create_axis_at_location(fig, .5, 2.25, 1.75, 1.)
    ax.plot(np.arange(burnin, N_samples), results.alphas[burnin:],
            color=allcolors[1])
    ax.set_xlabel('Iteration $(\\times 10^3)$')
    ax.set_xlim([burnin, N_samples])
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('$\\alpha_0$')
    plt.figtext(.05/5, 3.3/5, "C")

    # gamma vs Iteration
    # ax = fig.add_subplot(gs[1,M:])
    ax = create_axis_at_location(fig, 3., 2.25, 1.75, 1.)
    ax.plot(np.arange(burnin, N_samples), results.gammas[burnin:],
            color=allcolors[1])
    ax.set_xlabel('Iteration $(\\times 10^3)$')
    ax.set_xlim([burnin, N_samples])
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('$\\gamma$')
    plt.figtext(2.5/5, 3.3/5, "D")

    # Plot the transition matrix (From the last sample)
    ax = create_axis_at_location(fig, .5, .5, 1., 1.)
    im = ax.imshow(np.kron(A, np.ones((px,px))), cmap='Greys')
    ax.set_xlabel('$S_{t+1}$')
    ax.set_xticks(np.arange(px/2,A.shape[1]*px, step=stepK*px))
    ax.set_xticklabels(np.arange(0,A.shape[1], step=stepK))
    ax.set_ylabel('$S_{t}$')
    ax.set_yticks(np.arange(px/2,A.shape[0]*px, step=stepK*px))
    ax.set_yticklabels(np.arange(0,A.shape[0], step=stepK))
    ax.set_title('$\\mathbf{P}$')
    plt.figtext(.05/5, 1.55/5, "E")

    # Add colorbar for transition matrix
    # cbax = fig.add_subplot(gs[2, M-1])
    cbax = create_axis_at_location(fig, 1.65, .5, .1, 1.)
    cbar = ColorbarBase(cbax, cmap='Greys',
                        values=np.linspace(0, 1),
                        boundaries=np.linspace(0, 1),
                        ticks=np.linspace(0,1,6))

    # Plot the firing rates (From the last sample)
    px = 10
    # ax = fig.add_subplot(gs[2,M:-1])
    ax = create_axis_at_location(fig, 3, .5, 1.3, 1.)
    im = ax.imshow(np.kron(lmbdas, np.ones((px,px))), cmap='Greys')
    ax.set_xlabel('State')
    ax.set_xticks(np.arange(px/2,K*px, step=stepK*px))
    ax.set_xticklabels(np.arange(0,K, step=stepK))
    ax.set_ylabel('Cell')
    ax.set_yticks(np.arange(px/2,C*px, step=stepC*px))
    ax.set_yticklabels(np.arange(0,C, step=stepC))
    ax.set_title('$\\mathbf{\Lambda}$')
    plt.figtext(2.5/5, 1.55/5, "F")

    # Add colorbar for firing rate matrix
    # cbax = fig.add_subplot(gs[2,-1])
    cbax = create_axis_at_location(fig, 4.4, .5, .1, 1.)
    cbar = Colorbar(cbax, im, ticks=np.arange(0,16.1, step=1), label="spikes/bin")

    fig.savefig(os.path.join(figdir, 'figure6.pdf'))
    fig.savefig(os.path.join(figdir, 'figure6.png'))

    print "Plots can be found at %s*.pdf" % os.path.join(figdir, 'figure6')

# Figure 7: Hippocampal inference trajectories
dataset = "hipp_2dtrack_a"
N, S_train, pos_train, S_test, pos_test, center, radius = \
    load_hipp_data(dataname=dataset)

# Load results
runnum = 1
results_dir = os.path.join("results", dataset, "run%03d" % runnum)
results_type = "hdphmm_scale"
results_file = os.path.join(results_dir, results_type + ".pkl.gz")
with gzip.open(results_file, "r") as f:
    results = cPickle.load(f)

plot_results(results,
             S_train,
             figdir=results_dir)
