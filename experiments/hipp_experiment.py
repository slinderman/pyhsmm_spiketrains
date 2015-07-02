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

import pyhsmm
import pyhsmm_spiketrains.models

# Set the seed
seed = np.random.randint(0, 2**16)
print "setting seed to ", seed
np.random.seed(seed)

# Specify the rat data
data_file = os.path.join('data','positional_data.mat')

def split_train_test(S, pos, trainfrac):
    T,N = S.shape

    T_split = int(trainfrac * T)
    S_train = S[:T_split,:]
    S_test = S[T_split:, :]

    if pos is not None:
        pos_train = pos[:T_split, :]
        pos_test = pos[T_split:, :]

        return S_train, pos_train, S_test, pos_test

    else:
        return S_train, S_test

def load_rat_data(trainfrac=0.8):
    raw_data = loadmat(data_file)
    data = raw_data['AllSpikeData'].astype(np.int)

    # Transpose so that time is the first axis
    S = data.T
    S = np.array(S.todense())

    # Get the time stamps
    T,N = S.shape
    dt = 0.25
    ts = np.arange(T) * dt

    # Get the corresponding position
    pos = raw_data['rat_pos']
    center = raw_data['cp'].ravel()
    radius = np.float(raw_data['r'])

    S_train, pos_train, S_test, pos_test = split_train_test(S, pos, trainfrac=trainfrac)

    return N, S_train, pos_train, S_test, pos_test


Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls", "samples", "timestamps"])

def fit(name, model, test_data, N_iter=1000):
    def evaluate(model):
        ll, pll = \
            model.log_likelihood(), \
            model.log_likelihood(test_data) \
            # model.heldout_log_likelihood(test_data)
        # print '{} '.format(pll),
        return ll, pll

    def sample(model):
        tic = time.time()
        model.resample_model()
        timestep = time.time() - tic
        return evaluate(model), timestep

    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(N_iter)])

    lls, plls = zip(*((init_val,) + vals))
    timestamps = np.cumsum((0.,) + timesteps)

    return Results(name, lls, plls, model.copy_sample(), timestamps)

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
        y = plls.mean() - baseline
        yerr = plls.std()
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

def save_git_info(output_dir):
    import git
    repo = git.Repo(".")
    commit_id = str(repo.head.commit)

    with open(os.path.join(output_dir, "readme.md"), "w") as f:
        f.writelines(["git commit: ", commit_id])



if __name__ == "__main__":
    # Set output parameters
    runnum = 2
    output_dir = os.path.join("results", "hipp", "run%03d" % runnum)
    assert os.path.exists(output_dir)

    # Before running the experiment, save a file to specify the Git commit id
    save_git_info(output_dir)

    # Load the data
    N, S_train, pos_train, S_test, pos_test = load_rat_data()

    # Model dimensions
    Ks = np.arange(5, 55, step=5)

    # Gibbs iterations
    N_iter = 100

    # Define a sequence of models
    names_list = []
    fnames_list = []
    class_list = []
    args_list  = []
    color_list = []

    # Homogeneous Poisson baseline
    from scipy.special import gammaln
    homog_lmbda = S_train.sum(0) / float(S_train.shape[0])
    homog_ll = -gammaln(S_test+1).sum() \
               - (homog_lmbda * S_test.shape[0]).sum() \
               + (S_test * np.log(homog_lmbda)).sum()

    # Mixture Models
    for K in Ks:
        names_list.append("Mixture (K=%d)" % K)
        fnames_list.append("mixture_K%d" % K)
        class_list.append(pyhsmm_spiketrains.models.PoissonMixture)
        args_list.append({"K": K, "alpha_0": 10.0})
        color_list.append(allcolors[0])

    # DP Mixture
    # names_list.append("DP-Mixture")
    # class_list.append(pyhsmm_spiketrains.models.PoissonCRPMixture)
    # args_list.append({"alpha_0": 10})

    # HMMs
    for K in Ks:
        names_list.append("HMM (K=%d)" % K)
        fnames_list.append("hmm_K%d" % K)
        class_list.append(pyhsmm_spiketrains.models.PoissonHMM)
        args_list.append({"K": K, "alpha": 10.0, "init_state_concentration": 1.})
        color_list.append(allcolors[1])

    # HDP-HMM
    names_list.append("HDP-HMM")
    fnames_list.append("hdp_hmm")
    class_list.append(pyhsmm_spiketrains.models.PoissonHDPHMM)
    args_list.append({"K_max": 50,
                      "alpha_a_0": 10.0, "alpha_b_0": 1.0,
                      # "alpha": 10.0,
                      "gamma_a_0": 8.0, "gamma_b_0": 1.0,
                      # "gamma": 8.0,
                      "init_state_concentration": 1.})
    color_list.append(allcolors[2])

    # HSMMs
    # for K in Ks:
    #     avg_dur = 1./0.25
    #     dur_hypers = {"alpha_0": 2*avg_dur, "beta_0": 2.}
    #     dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypers) for _ in range(K)]
    #
    #     names_list.append("HSMM (K=%d)" % K)
    #     fnames_list.append("hsmm_K%d" % K)
    #     class_list.append(pyhsmm_spiketrains.models.PoissonHSMM)
    #     args_list.append({"K": K, "alpha": 10.0, "init_state_concentration": 1.,
    #                       "dur_distns": dur_distns})
    #     color_list.append(allcolors[3])
    #
    # # HDP-HSMM
    # names_list.append("HDP-HSMM")
    # fnames_list.append("hdp_hsmm")
    # class_list.append(pyhsmm_spiketrains.models.PoissonHDPHSMM)
    # avg_dur = 1./0.25
    # dur_hypers = {"alpha_0": 2*avg_dur, "beta_0": 2.}
    # dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypers) for _ in range(100)]
    #
    # args_list.append({"K_max": 100,
    #                   # "alpha_a_0": 1.0, "alpha_b_0": 1.0,
    #                   "alpha": 10.0,
    #                   # "gamma_a_0": 8.0, "gamma_b_0": 1.0,
    #                   "gamma": 8.0,
    #                   "init_state_concentration": 1.,
    #                   "dur_distns": dur_distns})
    # color_list.append(allcolors[4])

    results_list = []
    for model_name, model_fname, model_class, model_args in \
            zip(names_list, fnames_list, class_list, args_list):
        print "Model: ", model_name
        print "File:  ", model_fname
        print "Class: ", model_class
        print "Args:  "
        print model_args
        print ""
        output_file = os.path.join(output_dir, model_fname + ".pkl.gz")

        # Check for existing results
        if os.path.exists(output_file):
            print "Loading results from: ", output_file
            with gzip.open(output_file, "r") as f:
                res = cPickle.load(f)

        else:
            model = model_class(N, alpha_obs=1.0, beta_obs=1.0, **model_args)
            model.add_data(S_train)
            res = fit(model_name, model, S_test, N_iter=N_iter)

            # Save results
            with gzip.open(output_file, "w") as f:
                print "Saving results to: ", output_file
                cPickle.dump(res, f, protocol=-1)

        results_list.append(res)

    # Plot
    plot_predictive_log_likelihoods(results_list, color_list, baseline=homog_ll)
