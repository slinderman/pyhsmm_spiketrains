"""
Fit a sequence of models to the rat hippocampal recordings
"""
import os
import time
import gzip
import cPickle

import numpy as np
from scipy.io import loadmat
from collections import namedtuple

from pybasicbayes.util.text import progprint_xrange

import matplotlib.pyplot as plt
import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

import pyhsmm_spiketrains.models
reload(pyhsmm_spiketrains.models)

from pyhsmm_spiketrains.internals.utils import \
    log_expected_pll, split_train_test

# Set the seed
# seed = np.random.randint(0, 2**16)
seed = 0
print "setting seed to ", seed
np.random.seed(seed)

def load_hipp_data(dataname="hipp_2dtrack_b", trainfrac=0.8):
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


Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])

def fit(name, model, test_data, N_iter=1000, init_state_seq=None):
    def evaluate(model):
        ll = model.log_likelihood()
        pll = model.log_likelihood(test_data)
        N_used = len(model.used_states)
        trans = model.trans_distn
        alpha = trans.alpha
        gamma = trans.gamma if hasattr(trans, "gamma") else None
        rates = model.rates.copy()
        obs_hypers = model.obs_hypers
        # print 'N_states: {}, \tPLL:{}\n'.format(len(model.used_states), pll),
        return ll, pll, N_used, alpha, gamma, rates, obs_hypers

    def sample(model):
        tic = time.time()
        model.resample_model()
        timestep = time.time() - tic
        return evaluate(model), timestep

    # Initialize with given state seq
    if init_state_seq is not None:
        model.states_list[0].stateseq = init_state_seq
        for _ in xrange(100):
            model.resample_obs_distns()

    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(N_iter)])

    lls, plls, N_used, alphas, gammas, rates, obs_hypers = \
        zip(*((init_val,) + vals))
    timestamps = np.cumsum((0.,) + timesteps)

    return Results(name, lls, plls, N_used, alphas, gammas,
                   rates, obs_hypers,
                   model.copy_sample(), timestamps)

def plot_predictive_log_likelihoods(results, colors, burnin=50, baseline=0):
    plt.figure()
    plt.subplot(121)
    for res, color in zip(results, colors):
        plt.plot(res.timestamps, res.predictive_lls, color=color, label=res.name)

    plt.xlabel("time (s)")
    plt.ylabel("predictive log lkhd")
    # plt.legend(loc="lower right")

    plt.subplot(122)
    min_pll = np.inf
    max_pll = -np.inf
    for i, (res, color) in enumerate(zip(results, colors)):
        plls = np.array(res.predictive_lls[burnin:])
        # y = plls.mean() - baseline
        # yerr = plls.std()
        y = log_expected_pll(plls) - baseline
        yerr = 0
        plt.bar(i, y,
                yerr=yerr,
                width=0.9, color=color, ecolor='k',
                label=res.name)

        min_pll = min(min_pll, y)
        max_pll = max(max_pll, y)

    # plt.legend(loc="lower right")
    plt.xlabel("model")
    plt.xticks([])
    plt.ylabel("predictive log lkhd")
    plt.ylim(min_pll - 0.1 * (max_pll-min_pll),
             max_pll + 0.1 * (max_pll-min_pll))
    plt.show()

# def save_git_info(output_dir):
#     import git
#     repo = git.Repo(".")
#     commit_id = str(repo.head.commit)
#
#     with open(os.path.join(output_dir, "readme.md"), "w") as f:
#         f.writelines(["git commit: ", commit_id])

def make_hmm_models(N, S_train, Ks=np.arange(5,25, step=5), **kwargs):
    # Define a sequence of models
    names_list = []
    fnames_list = []
    hmm_list = []
    color_list = []

    for K in Ks:
        names_list.append("HMM (K=%d)" % K)
        fnames_list.append("hmm_K%d" % K)
        color_list.append(allcolors[0])
        hmm = \
            pyhsmm_spiketrains.models.PoissonHMM(
                N=N, K=K, alpha_a_0=5.0, alpha_b_0=1.0,
                init_state_concentration=1.0,
                **kwargs)
        hmm.add_data(S_train)
        hmm_list.append(hmm)

    return names_list, fnames_list, color_list, hmm_list

def make_hdphmm_models(N, S_train, K_max=100, **kwargs):
    # Define a sequence of models
    names_list = []
    fnames_list = []
    hmm_list = []
    color_list = []

    # Standard HDP-HMM (Scale resampling)
    names_list.append("HDP-HMM (Scale)")
    fnames_list.append("hdphmm_scale")
    color_list.append(allcolors[1])
    hmm = \
        pyhsmm_spiketrains.models.PoissonHDPHMM(
            N=N, K_max=K_max,
            alpha_a_0=5.0, alpha_b_0=1.0,
            gamma_a_0=8.0, gamma_b_0=1.0,
            init_state_concentration=1.0,
            **kwargs)
    hmm.add_data(S_train)
    hmm_list.append(hmm)

    # HDP-HMM with HMC for hyperparameters
    names_list.append("HDP-HMM (HMC)")
    fnames_list.append("hdphmm_hmc")
    color_list.append(allcolors[1])
    hmm = \
        pyhsmm_spiketrains.models.PoissonHDPHMM(
            N=N, K_max=K_max,
            alpha_a_0=5.0, alpha_b_0=1.0,
            gamma_a_0=8.0, gamma_b_0=1.0,
            init_state_concentration=1.0,
            **kwargs)
    hmm.add_data(S_train)
    hmm._resample_obs_method = "resample_obs_hypers_hmc"
    hmm_list.append(hmm)

    # HDP-HMM with hypers set by empirical bayes
    names_list.append("HDP-HMM (EB)")
    fnames_list.append("hdphmm_eb")
    color_list.append(allcolors[1])
    hmm = \
        pyhsmm_spiketrains.models.PoissonHDPHMM(
            N=N, K_max=K_max,
            alpha_a_0=5.0, alpha_b_0=1.0,
            gamma_a_0=8.0, gamma_b_0=1.0,
            init_state_concentration=1.0,
            **kwargs)
    hmm.add_data(S_train)
    hmm.init_obs_hypers_via_empirical_bayes()
    hmm._resample_obs_method = "resample_obs_hypers_null"
    hmm_list.append(hmm)

    return names_list, fnames_list, color_list, hmm_list


def run_experiment():
    # Set output parameters
    dataname = "hipp_2dtrack_a"
    runnum = 2
    output_dir = os.path.join("results", dataname, "run%03d" % runnum)
    assert os.path.exists(output_dir)

    # Load the data
    N, S_train, pos_train, S_test, pos_test, center, radius = \
        load_hipp_data(dataname)

    print "Running Experiment"
    print "Dataset:\t", dataname
    print "N:\t\t", N
    print "T_train:\t", S_train.shape[0]
    print "T_test:\t", S_test.shape[0]

    # Fit the baseline model
    static_model = pyhsmm_spiketrains.models.PoissonStatic(N)
    static_model.add_data(S_train)
    static_model.max_likelihood()
    static_ll = static_model.log_likelihood(S_test)

    # Define a set of HMMs
    names_list = []
    fnames_list = []
    color_list = []
    model_list = []

    # Add parametric HMMs
    nl, fnl, cl, ml = \
        make_hmm_models(N, S_train, Ks=np.arange(5,90,step=10),
                        alpha_obs=1.0, beta_obs=1.0)
    names_list.extend(nl)
    fnames_list.extend(fnl)
    color_list.extend(cl)
    model_list.extend(ml)

    # Add HDP_HMMs
    nl, fnl, cl, ml = \
        make_hdphmm_models(N, S_train, K_max=100,
                           alpha_obs=1.0, beta_obs=1.0)
    names_list.extend(nl)
    fnames_list.extend(fnl)
    color_list.extend(cl)
    model_list.extend(ml)

    # Fit the models with Gibbs sampling
    N_iter = 5000
    results_list = []
    for model_name, model_fname, model in \
            zip(names_list, fnames_list, model_list):
        print "Model: ", model_name
        print "File:  ", model_fname
        print ""
        output_file = os.path.join(output_dir, model_fname + ".pkl.gz")

        # Check for existing results
        if os.path.exists(output_file):
            print "Loading results from: ", output_file
            with gzip.open(output_file, "r") as f:
                res = cPickle.load(f)

        else:
            res = fit(model_name, model, S_test, N_iter=N_iter)

            # Save results
            with gzip.open(output_file, "w") as f:
                print "Saving results to: ", output_file
                cPickle.dump(res, f, protocol=-1)

        results_list.append(res)

    # Plot
    plot_predictive_log_likelihoods(results_list, color_list, baseline=static_ll)


if __name__ == "__main__":
    run_experiment()
