"""
Measure the number of inferred states as a function of:
    - number of observed neurons
    - time bin size
    - length of recording
    - firing rate
"""
import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from pyhsmm.util.text import progprint_xrange
from pyhsmm_spiketrains.models import PoissonHDPHMM

import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from hips.plotting.layout import *

from experiment_helper import load_synth_data

# Set the random seed for reproducibility
np.random.seed(0)
T = 2000
N = 50
K = 100
alpha = 12.0
gamma = 12.0
alpha_obs = 1.0
beta_obs = 1.0
N_iters = 1000

def K_used(model):
    return (model.state_usages > 0).sum()

def fit_model(data, N_iters, args={}):
    # Now fit the model with a model using all the data

    default_args = dict(N=N,
                        K_max=K,
                        alpha=alpha,
                        gamma=gamma,
                        alpha_obs=alpha_obs,
                        beta_obs=beta_obs,
                        init_state_concentration=1.0)
    default_args.update(args)
    model = PoissonHDPHMM(**default_args)
    model.add_data(data)

    def _evaluate(model):
        ll = model.log_likelihood()
        return ll, K_used(model)

    def _step(model):
        model.resample_model()
        return _evaluate(model)

    results = [_step(model) for _ in progprint_xrange(N_iters)]
    lls = np.array([r[0] for r in results])
    Ks = np.array([r[1] for r in results])
    return lls, Ks

def test_all(data):
    return fit_model(data, N_iters)

def test_N(data, N_test):
    """
    :param test_frac: Fraction of all neurons to use for fitting
    :return:
    """
    # Downsample the data
    test_neurons = np.random.permutation(N)[:N_test]
    test_data = data[:,test_neurons].copy('C')
    assert test_data.shape[1] == N_test

    return fit_model(test_data, N_iters, args={"N": N_test})

def test_T(data, T_test):
    """
    :param test_frac: Fraction of all neurons to use for fitting
    :return:
    """
    # Downsample the data
    test_data = data[:T_test,:].copy('C')
    return fit_model(test_data, N_iters)

def test_dt(data, freq):
    """
    :param freq: Number of time bins to aggregate
    :return:
    """
    # Aggregate time bins
    test_data = data.reshape((T//freq, freq, N)).sum(1).copy('C')
    assert np.all(test_data[0,:] == data[:freq,:].sum(0))
    return fit_model(test_data, N_iters)

def test_fr(true_model, scale):
    # Get the true rate, scale it, and resample the data
    true_rate = true_model.states_list[0].rate
    test_rate = scale * true_rate
    assert np.all(test_rate >= 0)
    test_data = np.random.poisson(test_rate)
    return fit_model(test_data, N_iters)

def fit_with_subsets_of_data(true_model, data, results_dir,
                             N_repeats=10):
    # Generate synth data
    K_true = K_used(true_model)

    # Experiments with subsets of neurons
    results_file = os.path.join(results_dir, "Ks_vs_N.pkl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            Ns_test, Ks_Ns, _ = cPickle.load(f)
    else:
        Ns_test = (np.array([0.1, 0.2, 0.5, 0.8, 1.0]) * N).astype(np.int)
        Ks_Ns = []
        for N_test in Ns_test:
            Ks_N = []
            for rpt in xrange(N_repeats):
                print "N_test: ", N_test, ". Repeat: ", rpt
                _, Ks = test_N(data, N_test)
                Ks_N.append(Ks[-1])
            Ks_Ns.append(Ks_N)

        with open(results_file, "w") as f:
            cPickle.dump((Ns_test, Ks_Ns, K_true), f, protocol=-1)

    # Experiments with subsets of time bins
    results_file = os.path.join(results_dir, "Ks_vs_T.pkl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            Ts_test, Ks_Ts, _ = cPickle.load(f)
    else:
        Ts_test = (np.array([0.1, 0.2, 0.5, 0.8, 1.0]) * T).astype(np.int)
        Ks_Ts = []
        for T_test in Ts_test:
            Ks_T = []
            for rpt in xrange(N_repeats):
                print "T_test: ", T_test, ". Repeat: ", rpt
                _, Ks = test_T(data, T_test)
                Ks_T.append(Ks[-1])
            Ks_Ts.append(Ks_T)

        with open(results_file, "w") as f:
            cPickle.dump((Ts_test, Ks_Ts, K_true), f, protocol=-1)

    # Experiments with varying firing rates
    results_file = os.path.join(results_dir, "Ks_vs_fr.pkl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            frs_test, Ks_frs, _ = cPickle.load(f)
    else:
        frs_test = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        Ks_frs = []
        for fr_test in frs_test:
            Ks_fr = []
            for rpt in xrange(N_repeats):
                print "fr_test: ", fr_test, ". Repeat: ", rpt
                _, Ks = test_fr(true_model, fr_test)
                Ks_fr.append(Ks[-1])
            Ks_frs.append(Ks_fr)

        with open(results_file, "w") as f:
            cPickle.dump((frs_test, Ks_frs, K_true), f, protocol=-1)

    # Experiments with varying time bin size
    results_file = os.path.join(results_dir, "Ks_vs_dt.pkl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            dts_test, Ks_dts, _ = cPickle.load(f)
    else:

        dts_test = np.array([1,2,4,5,10])
        Ks_dts = []
        for dt_test in dts_test:
            Ks_dt = []
            for rpt in xrange(N_repeats):
                print "dt_test: ", dt_test, ". Repeat: ", rpt
                _, Ks = test_dt(data, dt_test)
                Ks_dt.append(Ks[-1])
            Ks_dts.append(Ks_dt)

        with open(results_file, "w") as f:
            cPickle.dump((dts_test, Ks_dts, K_true), f, protocol=-1)

    return K_true, \
           Ns_test, Ks_Ns, \
           Ts_test, Ks_Ts, \
           frs_test, Ks_frs, \
           dts_test, Ks_dts

def plot_results(K_true,
                 Ns_test, Ks_Ns,
                 Ts_test, Ks_Ts,
                 frs_test, Ks_frs,
                 dts_test, Ks_dts,
                 figdir="."):

    # Plot the number of inferred states as a function of params
    fig = create_figure((5,3))

    # K vs num neurons
    ax = create_axis_at_location(fig, 0.6, 2., 1.7, .8, transparent=True)
    ax.boxplot(Ks_Ns, positions=np.arange(1,1+len(Ns_test)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(Ns_test)
    ax.plot([0,6], [K_true, K_true], ':k')
    plt.xlim(0.5,5.5)
    plt.ylim(0,100)
    ax.set_xlabel("$C$")
    ax.set_ylabel("Number of States", labelpad=-0.1)
    plt.figtext(0.05/5, 2.8/3, "A")

    # K vs time
    ax = create_axis_at_location(fig, 3.1, 2., 1.7, .8, transparent=True)
    ax.boxplot(Ks_Ts, positions=np.arange(1,1+len(Ts_test)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(Ts_test)
    ax.plot([0,6], [K_true, K_true], ':k')
    plt.xlim(0.5,5.5)
    plt.ylim(0,100)
    ax.set_xlabel("$T$")
    ax.set_ylabel("Number of States", labelpad=-0.1)
    plt.figtext(2.55/5, 2.8/3, "B")


    ax = create_axis_at_location(fig, .6, .5, 1.7, .8, transparent=True)
    ax.boxplot(Ks_frs, positions=np.arange(1,1+len(frs_test)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(frs_test)
    ax.plot([0,6], [K_true, K_true], ':k')
    plt.xlim(0.5,5.5)
    plt.ylim(0,100)
    ax.set_xlabel("$\lambda$ scale")
    ax.set_ylabel("Number of States", labelpad=-0.1)
    plt.figtext(0.05/5, 1.3/3, "C")

    ax = create_axis_at_location(fig, 3.1, .5, 1.7, .8, transparent=True)
    ax.boxplot(Ks_dts, positions=np.arange(1, 1+len(dts_test)),
               boxprops=dict(color=allcolors[1]),
               whiskerprops=dict(color=allcolors[0]),
               flierprops=dict(color=allcolors[1]))
    ax.set_xticklabels(dts_test)
    ax.plot([0,6], [K_true, K_true], ':k')
    plt.xlim(0.5,5.5)
    plt.ylim(0,100)
    ax.set_xlabel("$\Delta t$ scale")
    ax.set_ylabel("Number of States", labelpad=-0.1)
    plt.figtext(2.55/5, 1.3/3, "D")

    plt.savefig(os.path.join(figdir, "figure2.pdf"))
    plt.savefig(os.path.join(figdir, "figure2.png"))


if __name__ == "__main__":
    # Load the data
    N_repeats = 10
    modelname = "hdp-hmm"
    T = 2000
    T_test = 200
    K = 100
    N = 50
    version = 1
    runnum = 1
    dataset = "synth_%s_T%d_K%d_N%d_v%d" % (modelname, T, K, N, version)
    results_dir = os.path.join("results", dataset, "run%03d" % runnum)

    true_model, data, _, _, _ = \
        load_synth_data(T, K, N, T_test=T_test,
                        model=modelname, version=version)

    res = fit_with_subsets_of_data(true_model, data, results_dir, N_repeats)
    plot_results(*res, figdir=results_dir)


