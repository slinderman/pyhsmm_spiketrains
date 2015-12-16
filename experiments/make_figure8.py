"""
Plot true and inferred place fields for the hippocampal data
"""

import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from pyhsmm_spiketrains.internals.utils import split_train_test

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

from hips.plotting.layout import *
from hips.plotting.colormaps import white_to_color_cmap
from hips.distributions.circular_distribution import CircularDistribution

from experiment_helper import load_hipp_data

def plot_place_fields(results, pos, center, radius, data,
                      figdir='.'):
    """
    Plot the observation vector associated with a latent state
    """
    model = results.samples
    model.relabel_by_usage()
    N_used = results.N_used[-1]
    lmbdas = model.rates[:N_used,:]
    stateseq = model.stateseqs[0]
    occupancy = model.state_usages

    # Plot a figure for each latent state
    N_colors = 9
    colors = brewer2mpl.get_map('Set1', 'qualitative', N_colors).mpl_colors

    # State distributions
    dists = []
    for s in xrange(N_used):
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos[stateseq==s,0], pos[stateseq==s,1])
        dists.append(cd)

    # Plot the log likelihood as a function of iteration
    fig = create_figure((5,2.8))

    for i,c in enumerate([0, 13, 28, 38]):
        left = 1.25 * i + 0.05
        print "Plotting cell ", c
        color = colors[np.mod(c,N_colors)]
        cmap = white_to_color_cmap(color)

        # Compute the inferred place field
        inf_place_field = dists[0] * lmbdas[0,c] * occupancy[0]
        for s in range(1,N_used):
            inf_place_field += dists[s] * lmbdas[s,c] * occupancy[s]

        # inf_place_field = sum([d*(l*o) for d,l,o in zip(dists, lmbdas[c,:], occupancy)])
        spks = np.array(data[:,c] > 0).ravel()
        true_place_field = CircularDistribution(center, radius)
        true_place_field.fit_xy(pos[spks, 0], pos[spks, 1])

        # Plot the locations of this state
        ax = create_axis_at_location(fig, left, 1.45, 1.15, 1.15, transparent=True)
        remove_plot_labels(ax)
        # Plot the empirical location distribution
        inf_place_field.plot(ax=ax, cmap=cmap, plot_data=True, plot_colorbar=False)

        ax.set_title('Inf. Place Field %d' % (c+1),
                      fontdict={'fontsize' : 9})

        # Now plot the true place field
        ax = create_axis_at_location(fig, left, 0.05, 1.15, 1.15, transparent=True)
        remove_plot_labels(ax)

        true_place_field.plot(ax=ax, cmap=cmap, plot_data=True, plot_colorbar=False)
        ax.set_title('True Place Field %d' % (c+1),
                      fontdict={'fontsize' : 9})

    # fig.savefig(os.path.join(figdir,'figure8.pdf'))
    fig.savefig(os.path.join(figdir,'figure8.png'))

if __name__ == "__main__":
    dataset = "hipp_2dtrack_a"
    N, S_train, pos_train, S_test, pos_test, center, radius = \
        load_hipp_data(dataname=dataset)

    # Load results
    runnum = 2
    results_dir = os.path.join("results", dataset, "run%03d" % runnum)
    results_type = "hdphmm_scale"
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    plot_place_fields(results, pos_train, center, radius, S_train,
                      figdir=results_dir)
