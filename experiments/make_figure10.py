"""
Make figure 6, which includes
1. A plot of the centerpoints of all states
2. A plot of the top three latent state maps
3. A plot of the true and reconstructed locations

"""
import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np
from scipy.io import loadmat

import matplotlib
import matplotlib.patches
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from pyhsmm_spiketrains.internals.utils import split_train_test, convert_polar_to_xy

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

from hips.plotting.layout import *
from hips.plotting.colormaps import white_to_color_cmap

from hips.distributions.circular_distribution import CircularDistribution


def make_figure(results, S_train, pos_train, S_test, pos_test, center, radius, figdir="."):

    model = results.samples
    model.relabel_by_usage()
    N_used = results.N_used[-1]
    stateseq = model.stateseqs[0]
    occupancy = model.state_usages
    T_test = S_test.shape[0]
    t_test = np.arange(T_test) * 0.25

    fig = create_figure(figsize=(5,3))

    # Plot the centers of the latent states
    ax = create_axis_at_location(fig, .1, 1.7, 1., 1., transparent=True)
    # plt.figtext(0.05/5, 2.8/3, "A")
    remove_plot_labels(ax)

    circle = matplotlib.patches.Circle(xy=[0,0],
                                       radius= radius,
                                       linewidth=1,
                                       edgecolor="k",
                                       facecolor="white")
    ax.add_patch(circle)

    to_plot = np.array([0, 1, 2, 3, 25, 29, 31])

    # plt.figtext(1.2/5, 2.8/3, "B")
    # for k in xrange(N_used-1,-1,-1):
    for k in xrange(N_used):
        relocc = occupancy[k] / np.float(np.amax(occupancy))
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos_train[stateseq==k,0], pos_train[stateseq==k,1])
        # import pdb; pdb.set_trace()
        rm, thm = cd.mean
        xm,ym = convert_polar_to_xy(np.array([[rm, thm]]), [0,0])

        # Figure out color
        if k in to_plot:
            k_ind = np.where(to_plot==k)[0][0]
            color = allcolors[k_ind]
        else:
            color = 'k'

        ax.plot(xm,ym,'o',
                 markersize=3+relocc*4,
                 markerfacecolor=color,
                 # markeredgecolor=color,
                 markeredgecolor='k',
                 markeredgewidth=1)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    ax.set_title('All states', fontdict={'fontsize' : 9})

    # Plot a figure for each latent state
    print np.row_stack((np.arange(N_used),
                        np.array([dd.r for dd in model.dur_distns[:N_used]])))
    dur = np.arange(1,16)
    yticks = [0, 0.2, 0.4]
    for k in xrange(3):
        left = 1.45 + 1.1*k + 0.1
        color = allcolors[k]

        # Plot the locations of this state
        ax = create_axis_at_location(fig, left, 1.8, 1., .9, transparent=True)
        # remove_plot_labels(ax)
        # # Plot the empirical location distribution
        # cd = CircularDistribution(center, radius)
        # cd.fit_xy(pos_train[stateseq==k,0], pos_train[stateseq==k,1])
        # cd.plot(ax=ax, cmap=cmap, plot_data=True, plot_colorbar=False)
        dur_distn = model.dur_distns[to_plot[k]]
        ax.bar(dur, np.exp(dur_distn.log_pmf(dur)), width=1, color=color)

        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks(yticks)
        if k > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Duration Prob.", labelpad=0)
        ax.set_xlim(1,16)
        ax.set_xlabel("Duration", labelpad=0)


        ax.set_title('State %d (%.1f%%)' % (to_plot[k]+1, 100.*occupancy[to_plot[k]]),
                     fontdict={'fontsize' : 9})

    # Bottom row
    for k in xrange(3,7):
        left = .35 + 1.1*(k-3) + 0.1
        color = allcolors[k]

        # Plot the locations of this state
        ax = create_axis_at_location(fig, left, .4, 1., .9, transparent=True)
        # remove_plot_labels(ax)
        # # Plot the empirical location distribution
        # cd = CircularDistribution(center, radius)
        # cd.fit_xy(pos_train[stateseq==k,0], pos_train[stateseq==k,1])
        # cd.plot(ax=ax, cmap=cmap, plot_data=True, plot_colorbar=False)
        dur_distn = model.dur_distns[to_plot[k]]
        ax.bar(dur, np.exp(dur_distn.log_pmf(dur)), width=1, color=color)

        ax.set_xticks([1, 5, 10, 15])
        ax.set_yticks(yticks)
        if k > 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Duration Prob.", labelpad=0)
        ax.set_xlim(1,16)
        ax.set_xlabel("Duration", labelpad=0)


        ax.set_title('State %d (%.1f%%)' % (to_plot[k]+1, 100.*occupancy[to_plot[k]]),
                     fontdict={'fontsize' : 9})

    fig.savefig(os.path.join(figdir, 'figure10.pdf'))
    fig.savefig(os.path.join(figdir, 'figure10.png'), dpi=300)

    plt.show()

def load_hipp_data(dataname="hipp_2dtrack_a", trainfrac=0.8):
    raw_data = loadmat("data/%s.mat" % dataname)
    S = raw_data['S'].astype(np.int).copy("C")

    # Get the time stamps
    T,N = S.shape
    dt = 0.25
    ts = np.arange(T) * dt

    # Get the corresponding position
    pos = raw_data['pos']
    S_train, pos_train, S_test, pos_test = split_train_test(S, pos, trainfrac=trainfrac)

    if "cp" in raw_data and "r" in raw_data:
        center = raw_data['cp'].ravel()
        radius = np.float(raw_data['r'])
    else:
        center = radius = None

    return N, S_train, pos_train, S_test, pos_test, center, radius

if __name__ == "__main__":
    # Load the data
    dataset = "hipp_2dtrack_a"
    N, S_train, pos_train, S_test, pos_test, center, radius = \
        load_hipp_data(dataname=dataset)

    # Load results
    runnum = 1
    results_dir = os.path.join("results", dataset, "run%03d" % runnum)
    results_type = "hdphsmm_scale"
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    make_figure(results,
                S_train, pos_train,
                S_test, pos_test,
                center, radius,
                figdir=results_dir)
