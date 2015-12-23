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


def make_figure6(results, S_train, pos_train, S_test, pos_test, center, radius, figdir="."):

    model = results.samples
    model.relabel_by_usage()
    N_used = results.N_used[-1]
    stateseq = model.stateseqs[0]
    occupancy = model.state_usages
    T_test = S_test.shape[0]
    t_test = np.arange(T_test) * 0.25

    fig = create_figure(figsize=(5,3))

    # Plot the centers of the latent states
    ax = create_axis_at_location(fig, .05, 1.55, 1.15, 1.15, transparent=True)
    plt.figtext(0.05/5, 2.8/3, "A")
    remove_plot_labels(ax)

    circle = matplotlib.patches.Circle(xy=[0,0],
                                       radius= radius,
                                       linewidth=1,
                                       edgecolor="k",
                                       facecolor="white")
    ax.add_patch(circle)

    plt.figtext(1.2/5, 2.8/3, "B")
    for k in xrange(N_used):
        relocc = occupancy[k] / np.float(np.amax(occupancy))
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos_train[stateseq==k,0], pos_train[stateseq==k,1])
        # import pdb; pdb.set_trace()
        rm, thm = cd.mean
        xm,ym = convert_polar_to_xy(np.array([[rm, thm]]), [0,0])
        ax.plot(xm,ym,'o',
                 markersize=relocc*6,
                 markerfacecolor='k',
                 markeredgecolor='k',
                 markeredgewidth=1)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    ax.set_title('All states', fontdict={'fontsize' : 9})

    # Plot a figure for each latent state
    for k in xrange(3):
        left = 1.25 * (k+1) + 0.05
        color = allcolors[k]
        cmap = white_to_color_cmap(color)

        # Plot the locations of this state
        ax = create_axis_at_location(fig, left, 1.55, 1.15, 1.15, transparent=True)
        remove_plot_labels(ax)
        # Plot the empirical location distribution
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos_train[stateseq==k,0], pos_train[stateseq==k,1])
        cd.plot(ax=ax, cmap=cmap, plot_data=True, plot_colorbar=False)

        ax.set_title('State %d (%.1f%%)' % (k+1, 100.*occupancy[k]),
                     fontdict={'fontsize' : 9})


    # Bottom: Plot the true and predicted locations for heldout data
    plt.figtext(0.05/5, 1.55/3, "C")
    epdf = estimate_pos(model,
                        S_train, pos_train,
                        S_test, pos_test,
                        center, radius)

    # Compute the mean trajectory
    mean_location = np.zeros_like(pos_test)
    for t in range(T_test):
        cd = CircularDistribution(center, radius, pdf=epdf[t,:])
        mean_location[t,:] = convert_polar_to_xy(np.atleast_2d(cd.mean), center)

    # Convert estimates to x,y and compute mean squared error
    sqerr = np.sqrt((mean_location - pos_test)**2).mean(axis=1)
    mse = sqerr.mean(axis=0)
    stdse = sqerr.std(axis=0)
    print "MSE: %f \pm %f" % (mse, stdse)

    ax_y = create_axis_at_location(fig, 0.6, 0.4, 3.8, 0.5, box=True, ticks=True)
    ax_y.plot(t_test, pos_test[:,1] - center[1], '-k', lw=1)
    ax_y.plot(t_test, mean_location[:,1] - center[1], '-', color=allcolors[1])

    ax_y.set_ylabel('$y(t)$ [cm]', fontsize=9)
    ax_y.set_ylim([-radius,radius])
    ax_y.set_xlabel('$t$ [s]', fontsize=9)
    ax_y.set_xlim(0,T_test*0.25)
    ax_y.tick_params(axis='both', which='major', labelsize=9)

    ax_x = create_axis_at_location(fig, 0.6, 1., 3.8, 0.5, box=True, ticks=True)
    ax_x.plot(t_test, pos_test[:,0] - center[0], '-k', lw=1)
    ax_x.plot(t_test, mean_location[:,0] - center[0], '-', color=allcolors[1])
    ax_x.set_ylabel('$x(t)$ [cm]', fontsize=9)
    ax_x.set_ylim([-radius,radius])
    ax_x.set_xticks(ax_y.get_xticks())
    ax_x.set_xticklabels([])
    ax_x.tick_params(axis='both', which='major', labelsize=9)
    ax_x.set_xlim(0,T_test*0.25)

    fig.savefig(os.path.join(figdir, 'figure5.pdf'))
    fig.savefig(os.path.join(figdir, 'figure5.png'))

    plt.show()


def estimate_pos(model, S_train, pos_train, S_test, pos_test, center, radius):
    # Compute the marginal distribution over states at each of the test time steps
    expected_states_test = model.heldout_state_marginals(S_test)

    assert expected_states_test.shape[1] == model.num_states
    T,K = expected_states_test.shape

    # Compute the empirical location distributions for each of the latent states
    cds = []
    states_obj = model.states_list[0]
    state_seq = states_obj.stateseq

    for k in range(K):
        cd = CircularDistribution(center, radius)
        cd.fit_xy(pos_train[state_seq == k, 0], pos_train[state_seq == k, 1])
        cds.append(cd)

    epdf = np.zeros((T,) + cds[0].pdf.shape)
    for t in range(T):
        # Get the average of the location distributions at time t
        assert np.allclose(expected_states_test[t,:].sum(), 1.0)
        epdf[t,:,:] = np.array([expected_states_test[t,k]*cds[k].pdf for k in range(K)]).sum(0)

    return epdf

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
    results_type = "hdphmm_scale"
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    make_figure6(results,
                 S_train, pos_train,
                 S_test, pos_test,
                 center, radius,
                 figdir=results_dir)
