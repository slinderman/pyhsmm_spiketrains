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

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from hips.plotting.layout import create_axis_at_location, create_figure

from experiment_helper import load_hipp_data

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

def plot_results(alpha_a_0s, Ks_alpha_a_0,
                 gamma_a_0s, Ks_gamma_a_0,
                 figdir="."):

    # Plot the number of inferred states as a function of params
    fig = create_figure((5,1.5))

    ax = create_axis_at_location(fig, 0.6, 0.5, 1.7, .8, transparent=True)
    plt.figtext(0.05/5, 1.25/1.5, "A")
    ax.boxplot(Ks_alpha_a_0, positions=np.arange(1,1+len(alpha_a_0s)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(alpha_a_0s)
    plt.xlim(0.5,4.5)
    plt.ylim(40,90)
    # plt.yticks(np.arange(0,101,20))
    ax.set_xlabel("$a_{\\alpha_0}$")
    ax.set_ylabel("Number of States")

    ax = create_axis_at_location(fig, 3.1, 0.5, 1.7, .8, transparent=True)
    plt.figtext(2.55/5, 1.25/1.5, "B")
    ax.boxplot(Ks_gamma_a_0, positions=np.arange(1,1+len(gamma_a_0s)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(gamma_a_0s)
    plt.xlim(0.5,4.5)
    plt.ylim(40,90)
    # plt.yticks(np.arange(0,101,20))
    ax.set_xlabel("$a_{\\gamma}$")
    ax.set_ylabel("Number of States")

    plt.savefig(os.path.join(figdir, "figure7.pdf"))
    plt.savefig(os.path.join(figdir, "figure7.png"))


# Figure 7: Hippocampal inference trajectories
dataset = "hipp_2dtrack_a"
N, S_train, pos_train, S_test, pos_test, center, radius = \
    load_hipp_data(dataname=dataset)

# Load results
runnum = 1
results_dir = os.path.join("results", dataset, "run%03d" % runnum)

# Load alpha_a_0 results
alpha_a_0s = [1.0, 5.0, 10.0, 100.0]
alpha_a_0_results = []
for alpha_a_0 in alpha_a_0s:
    results_type = "hdphmm_scale_alpha_a_0%.1f" % alpha_a_0
    print "Loading ", results_type
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    alpha_a_0_results.append(results.N_used[-2000:])

gamma_a_0s = [1.0, 5.0, 10.0, 100.0]
gamma_a_0_results = []
for gamma_a_0 in gamma_a_0s:
    results_type = "hdphmm_scale_gamma_a_0%.1f" % gamma_a_0
    print "Loading ", results_type
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    gamma_a_0_results.append(results.N_used[-2000:])

# alpha_obss = [0.1, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
# alpha_obs_results = []
# for alpha_obs in alpha_obss:
#     results_type = "hdphmm_scale_alpha_obs%.1f" % alpha_obs
#     print "Loading ", results_type
#     results_file = os.path.join(results_dir, results_type + ".pkl.gz")
#     with gzip.open(results_file, "r") as f:
#         results = cPickle.load(f)
#
#     alpha_obs_results.append(results.N_used[-2000:])

plot_results(alpha_a_0s, alpha_a_0_results,
             gamma_a_0s, gamma_a_0_results,
             figdir=results_dir)
