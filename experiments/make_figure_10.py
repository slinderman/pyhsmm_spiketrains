"""
Compute the mutual information of a latent state sequence
and a sequence of positions.
"""
import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from hips.plotting.layout import *
from hips.distributions.circular_distribution import CircularDistribution
from experiment_helper import load_hipp_data

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

def discrete_mutual_info(z, x):
    """
    Compute the  mutual information between discrete random variables Z and X
    given sequences of observations z and x
    """
    N = float(len(z))
    assert len(x) == len(z), "z and x must be the same shape!"
    I = 0

    # Sum over values of z
    z_min = np.amin(z)
    z_max = np.amax(z)
    x_min = np.amin(x)
    x_max = np.amax(x)
    for vz in np.arange(z_min, z_max+1):
        # Find where z == vz
        wvz = z == vz
        nvz = np.sum(wvz)
        p_vz = nvz / N

        if nvz == 0:
            continue

        # Sum over values of x
        for vx in np.arange(x_min, x_max+1):
            wvx = x == vx
            nvx = np.sum(wvx)
            p_vx = nvx / N

            # Compute the conditional probability of z given x
            # p_vz_given_vx = np.sum(np.bitwise_and(z==vz, x==vx)) / p_vx
            p_vz_given_vx = np.sum(wvz * wvx) / float(nvx)

            # assert np.allclose(p_vz_given_vx, p_vz_given_vx2), "ERROR calculating conditional prob"

            # Add to the mutual information
            if p_vz_given_vx > 0:
                I += p_vx * p_vz_given_vx * np.log2(p_vz_given_vx/p_vz)

    return I

# def mutual_info_per_location(results, pos, center, radius):
#     """
#     Compute the mutual information that a state sequence conveys about
#     whether or not the rat is in a particular location.
#     """
#     # Parse the smpls
#     z = results.samples.stateseqs[0]
#     N_states = results.N_used[-1]
#
#     # Fit a circular distribution to pos and get discrete locations
#     cd = CircularDistribution(center, radius)
#     cd.fit_xy(pos[:,0], pos[:,1])
#     xs = cd.bin_xy(pos[:,0], pos[:,1])
#
#     # Compute the mutual information about each bin
#     Is = np.zeros_like(cd.pdf)
#     for r in xrange(cd.rbins):
#         for th in xrange(cd.thbins):
#             # Create a binary r.v. to indicate whether rat is in bin i,j
#             x = (xs[:,0] == r) * (xs[:,1] == th)
#
#             # Compute mutual info of last state sequence with this position
#             Is[r,th] = discrete_mutual_info(z, x)
#
#             print "Mutual info (%d,%d): %.3f" % (r,th,Is[r,th])
#
#     cd_I = CircularDistribution(center, radius, pdf=Is)
#
#     # Plot the results
#     # plt.figure()
#     ax, _ = cd_I.plot()
#     # Plot the locations of this state
#     fig = plt.figure(figsize=(2., 1.5))
#     ax = create_axis_at_location(fig, .05, 0.05, 1.25, 1.25)
#     remove_plot_labels(ax)
#     # Plot the empirical location distribution
#     _,cs = cd_I.plot(ax=ax, cmap='YlOrRd', plot_data=False, plot_colorbar=False)
#
#     ax.set_title('Mutual Information',
#                   fontdict={'fontsize' : 9})
#
#     cbax = create_axis_at_location(fig, 1.35, 0.05, 0.1, 1.25)
#     cbar = Colorbar(cbax, cs, ticks=np.arange(0,0.061, step=0.01))
#     cbax.tick_params(labelsize=9)
#     cbax.set_ylabel('bits', fontdict={'fontsize' : 9})
#     plt.savefig('mutual_info_per_loc.pdf')
#     plt.close(fig)


def mutual_info_ss_pos(results, pos, center, radius):
    """
    Compute the mutual information of a state sequence with a set of positions
    We bin the positions into L equal sized bins and treat the instantaneous
    position as a discrete random variable.
    """
    # Parse the smpls
    zs = results.samples.stateseqs[0]

    # Fit a circular distribution to pos and get discrete locations
    cd = CircularDistribution(center, radius)
    cd.fit_xy(pos[:,0], pos[:,1])
    xs = cd.bin_xy(pos[:,0], pos[:,1])

    # Ravel the 2D bins and the CD PDF
    xs = np.ravel_multi_index((xs[:,0], xs[:,1]), (cd.rbins, cd.thbins), order='C')

    # Compute the mutual information as a function of samples
    N_samples = 1
    thin = 100
    Is = []
    for i in xrange(0, N_samples, thin):
        Is.append(discrete_mutual_info(zs[i], xs))
        print "Mutual info for sample %d: %.3f" % (i, Is[-1])

    Is = np.asarray(Is)

    # Plot the results
    colors = brewer2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors
    color = colors[1]

    fig = create_figure((2.4,2))
    ax = create_axis_at_location(fig, 0.5,0.5,1.5,1.)
    ax.plot(np.arange(0, N_samples, thin), Is, color=color)
    ax.set_xlabel('Iteration')
    ax.set_xticks(np.arange(0,10001,step=2500))
    ax.set_ylabel('Mutual Information (bits)')
    fig.savefig(os.path.join('.', 'figure_10a.pdf'))
    plt.close(fig)

def mutual_info_per_state(results, pos, center, radius, figdir='.'):
    """
    Compute the mutual information that a state sequence conveys about
    whether or not the rat is in a particular location.
    """
    # Parse the smpls
    z = results.samples.stateseqs[0]
    N_states = results.N_used[-1]

    # Sort the states by occupancy
    occupancy = np.zeros(np.amax(z))
    for n in np.arange(np.amax(z)):
        occupancy[n] = np.sum(z==n)
    perm = np.argsort(occupancy)[::-1]

    # Only keep the used states
    perm = perm[:N_states]

    # Fit a circular distribution to pos and get discrete locations
    cd = CircularDistribution(center, radius)
    cd.fit_xy(pos[:,0], pos[:,1])
    xs = cd.bin_xy(pos[:,0], pos[:,1])

    # Ravel the 2D bins and the CD PDF
    xs = np.ravel_multi_index((xs[:,0], xs[:,1]), (cd.rbins, cd.thbins), order='C')

    # Compute the mutual information per state
    Is = np.zeros(N_states)
    for i,n in enumerate(perm):
        Is[i] = discrete_mutual_info(z==n, xs)
        print "Mutual info per state %d: %.3f" % (i, Is[i])

    print "Total MI: %.3f" % np.sum(Is)

    # Plot the results
    colors = brewer2mpl.get_map('Set1', 'Qualitative', 3).mpl_colors
    color = colors[1]

    fig = create_figure((2.4,2))
    ax = create_axis_at_location(fig, 0.5,0.5,1.5,1.)
    ax.plot(np.arange(0, N_states), Is, color=color)
    ax.set_xlabel('Latent state')
    ax.set_xlim([0,N_states])
    ax.set_ylabel("bits")
    ax.set_title('Mutual Information')
    fig.savefig(os.path.join(figdir, 'figure10.pdf'))
    plt.close(fig)


# Load the data
dataset = "hipp_2dtrack_a"
N, S_train, pos_train, S_test, pos_test, center, radius = \
    load_hipp_data(dataname=dataset)

# Load results
runnum = 2
results_dir = os.path.join("results", dataset, "run%03d" % runnum)
results_type = "hdphmm_scale_alpha_obs2.0"
results_file = os.path.join(results_dir, results_type + ".pkl.gz")
print "Loading ", results_file
with gzip.open(results_file, "r") as f:
    results = cPickle.load(f)


# Compute the mutual info
# mutual_info_ss_pos(results, pos_train, center, radius)
# mutual_info_per_location(smpls, pos_train, center, radius)
mutual_info_per_state(results, pos_train, center, radius,
                      figdir=results_dir)
