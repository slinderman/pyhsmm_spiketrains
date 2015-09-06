"""
Fit a sequence of models to the rat hippocampal recordings
"""
import os
import time
import gzip
import cPickle

import numpy as np
from scipy.io import loadmat
from scipy.misc import logsumexp
from collections import namedtuple

from pybasicbayes.util.text import progprint_xrange

import matplotlib.pyplot as plt
import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

import pyhsmm_spiketrains.models
reload(pyhsmm_spiketrains.models)

# Set the seed
seed = np.random.randint(0, 2**16)
print "setting seed to ", seed
np.random.seed(seed)

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

    return N, S_train, pos_train, S_test, pos_test


Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls", "samples", "timestamps"])

def fit(name, model, test_data, N_iter=1000, init_state_seq=None):
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

    # Initialize with given state seq
    if init_state_seq is not None:
        model.states_list[0].stateseq = init_state_seq
        for _ in xrange(100):
            model.resample_obs_distns()

    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(N_iter)])

    lls, plls = zip(*((init_val,) + vals))
    timestamps = np.cumsum((0.,) + timesteps)

    return Results(name, lls, plls, model.copy_sample(), timestamps)

def log_expected_pll(plls):
    return -np.log(len(plls)) + logsumexp(plls)

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

def save_git_info(output_dir):
    import git
    repo = git.Repo(".")
    commit_id = str(repo.head.commit)

    with open(os.path.join(output_dir, "readme.md"), "w") as f:
        f.writelines(["git commit: ", commit_id])

def make_model_list(Ks=np.arange(5,25,step=5)):
    # Define a sequence of models
    names_list = []
    fnames_list = []
    class_list = []
    args_list  = []
    color_list = []

    # Mixture Models
    # for K in Ks:
    #     names_list.append("Mixture (K=%d)" % K)
    #     fnames_list.append("mixture_K%d" % K)
    #     class_list.append(pyhsmm_spiketrains.models.PoissonMixture)
    #     args_list.append({"K": K, "alpha_0": 10.0})
    #     color_list.append(allcolors[0])

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
    args_list.append({"K_max": 100,
                      # "alpha_a_0": 10.0, "alpha_b_0": 1.0,
                      "alpha": 10.0,
                      # "gamma_a_0": 100.0, "gamma_b_0": 1.0,
                      "gamma": 500.0,
                      "init_state_concentration": 1.})
    color_list.append(allcolors[2])

    # Negative Binomial Duration HSMMs
    avg_dur = 1./0.25
    for K in Ks:
        names_list.append("HSMM (K=%d)" % K)
        fnames_list.append("intnegbin_hsmm_K%d" % K)
        class_list.append(pyhsmm_spiketrains.models.PoissonHSMMIntNegBinDuration)
        args_list.append({"K": K, "alpha": 10.0, "init_state_concentration": 1.,
                          "r_max": 1, "alpha_dur": 1.0, "beta_dur": 1.})

        color_list.append(allcolors[3])


    # Poisson Duration HSMMs
    # for K in Ks:
    #     names_list.append("HSMM (K=%d)" % K)
    #     fnames_list.append("poisson_hsmm_K%d" % K)
    #     class_list.append(pyhsmm_spiketrains.models.PoissonHSMMPoissonDuration)
    #     args_list.append({"K": K, "alpha": 10.0, "init_state_concentration": 1.,
    #                       "alpha_dur": 2*avg_dur, "beta_dur": 2.})
    #
    #     color_list.append(allcolors[4])


    # HDP-HSMM
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

    return names_list, fnames_list, class_list, args_list, color_list

def run_experiment():
    # Set output parameters
    dataname = "hipp_1dtrack"
    runnum = 2
    output_dir = os.path.join("results", dataname, "run%03d" % runnum)
    assert os.path.exists(output_dir)


    # Load the data
    N, S_train, pos_train, S_test, pos_test = load_hipp_data(dataname)

    # Define a set of models
    Ks = np.arange(5, 50, step=5)
    names_list, fnames_list, class_list, args_list, color_list = \
        make_model_list(Ks)

    # Fit the baseline model
    from scipy.special import gammaln
    homog_lmbda = S_train.sum(0) / float(S_train.shape[0])
    homog_ll = -gammaln(S_test+1).sum() \
               - (homog_lmbda * S_test.shape[0]).sum() \
               + (S_test * np.log(homog_lmbda)).sum()

    # Fit the models with Gibbs sampling
    N_iter = 1000
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


def test_hmm_vs_hsmm():
    # Set output parameters
    dataname = "hipp_1dtrack"
    runnum = 2
    output_dir = os.path.join("results", dataname, "run%03d" % runnum)
    assert os.path.exists(output_dir)

    # Before running the experiment, save a file to specify the Git commit id
    save_git_info(output_dir)

    # Load the data
    N, S_train, pos_train, S_test, pos_test = load_hipp_data(dataname)

    # Set observation hypers
    S_mean = S_train.mean(0)
    alpha_obs = 2.
    beta_obs = 2. / S_mean

    # Fit the baseline model
    from scipy.special import gammaln
    homog_lmbda = S_train.sum(0) / float(S_train.shape[0])
    homog_ll = -gammaln(S_test+1).sum() \
               - (homog_lmbda * S_test.shape[0]).sum() \
               + (S_test * np.log(homog_lmbda)).sum()


    results_list = []
    # Fit an HMM
    K = 50
    hmm_model = pyhsmm_spiketrains.models.PoissonHMM(
        N, alpha_obs=alpha_obs, beta_obs=beta_obs, K=K, alpha=10.0, init_state_concentration=1.)
    hmm_model.add_data(S_train)
    results_list.append(fit("HMM (50)", hmm_model, S_test, N_iter=1000))

    # Fit an HSMM initialized with the HMM state sequence
    hsmm_model = pyhsmm_spiketrains.models.PoissonHSMMIntNegBinDuration(
        N, alpha_obs=alpha_obs, beta_obs=beta_obs, K=K,
        **{"alpha": 10.0, "init_state_concentration": 1.,
           "r_max": 1, "alpha_dur": 1.0, "beta_dur": 1.})
    hsmm_model.add_data(S_train)
    results_list.append(fit("HMM (50)", hsmm_model, S_test, N_iter=1000,
                            init_state_seq=hmm_model.states_list[0].stateseq))

    # Plot
    plot_predictive_log_likelihoods(results_list, ["r", "b"], baseline=homog_ll, burnin=200)


# def run_experiment_with_init():
if __name__ == "__main__":
    # Set output parameters
    dataname = "hipp_2dtrack_b"
    runnum = 2
    output_dir = os.path.join("results", dataname, "run%03d" % runnum)
    assert os.path.exists(output_dir)


    # Load the data
    N, S_train, pos_train, S_test, pos_test = load_hipp_data(dataname)

    # Set observation hypers
    S_mean = S_train.mean(0)
    alpha_obs = 2.
    beta_obs = 2. / S_mean

    # Define a set of model sizes
    Ks = np.arange(5, 50, step=5)

    # Fit the baseline model
    from scipy.special import gammaln
    homog_lmbda = S_train.sum(0) / float(S_train.shape[0])
    homog_ll = -gammaln(S_test+1).sum() \
               - (homog_lmbda * S_test.shape[0]).sum() \
               + (S_test * np.log(homog_lmbda)).sum()

    # Fit the models with Gibbs sampling
    N_iter = 1000
    results_list = []
    names_list = []
    color_list = []
    for K in Ks:
        # Mixture Models
        model_name = "Mixture (K=%d)" % K
        model_fname = os.path.join(output_dir, "mixture_K%d.pkl" % K)
        if os.path.exists(model_fname):
            print "Loading results from: ", model_fname
            with gzip.open(model_fname, "r") as f:
                res = cPickle.load(f)

        else:
            mixture_model = pyhsmm_spiketrains.models.PoissonMixture(
                N, K=K, alpha_obs=alpha_obs, beta_obs=beta_obs, alpha_0=10.0)
            mixture_model.add_data(S_train)
            res = fit(model_name, mixture_model, S_test, N_iter=N_iter)

            # Save results
            with gzip.open(model_fname, "w") as f:
                print "Saving results to: ", model_fname
                cPickle.dump(res, f, protocol=-1)

        names_list.append(model_name)
        results_list.append(res)
        color_list.append(allcolors[0])

        # HMM
        model_name = "HMM (K=%d)" % K
        model_fname = os.path.join(output_dir, "hmm_K%d.pkl" % K)
        if os.path.exists(model_fname):
            print "Loading results from: ", model_fname
            with gzip.open(model_fname, "r") as f:
                res = cPickle.load(f)
                hmm_model = res.samples

        else:
            hmm_model = pyhsmm_spiketrains.models.PoissonHMM(
            N, alpha_obs=alpha_obs, beta_obs=beta_obs, K=K, alpha=10.0, init_state_concentration=1.)
            hmm_model.add_data(S_train)
            res  = fit(model_name, hmm_model, S_test, N_iter=1000)

            # Save results
            with gzip.open(model_fname, "w") as f:
                print "Saving results to: ", model_fname
                cPickle.dump(res, f, protocol=-1)

        names_list.append(model_name)
        results_list.append(res)
        color_list.append(allcolors[1])


        # Negative Binomial Duration HSMM
        model_name = "NB HSMM (K=%d)" % K
        model_fname = os.path.join(output_dir, "intnegbin_hsmm_K%d.pkl" % K)
        if os.path.exists(model_fname):
            print "Loading results from: ", model_fname
            with gzip.open(model_fname, "r") as f:
                res = cPickle.load(f)
        else:
            hsmm_model = pyhsmm_spiketrains.models.PoissonHSMMIntNegBinDuration(
            N, alpha_obs=alpha_obs, beta_obs=beta_obs, K=K,
            **{"alpha": 10.0, "init_state_concentration": 1.,
               "r_max": 10, "alpha_dur": 1.0, "beta_dur": 1.})
            hsmm_model.add_data(S_train)
            res = fit(model_name, hsmm_model, S_test, N_iter=1000,
                                    init_state_seq=hmm_model.states_list[0].stateseq)

            # Save results
            with gzip.open(model_fname, "w") as f:
                print "Saving results to: ", model_fname
                cPickle.dump(res, f, protocol=-1)

        names_list.append(model_name)
        results_list.append(res)
        color_list.append(allcolors[2])


        # Poisson Duration HSMM
        model_name = "Poisson HSMM (K=%d)" % K
        model_fname = os.path.join(output_dir, "poisson_hsmm_K%d.pkl" % K)
        if os.path.exists(model_fname):
            print "Loading results from: ", model_fname
            with gzip.open(model_fname, "r") as f:
                res = cPickle.load(f)
        else:
            phsmm_model = pyhsmm_spiketrains.models.PoissonHSMMPoissonDuration(
            N, alpha_obs=alpha_obs, beta_obs=beta_obs, K=K,
            **{"alpha": 10.0, "init_state_concentration": 1.,
               "alpha_dur": 2.0 * 4.0, "beta_dur": 2.})
            phsmm_model.add_data(S_train)

            res = fit(model_name, phsmm_model, S_test, N_iter=1000,
                                    init_state_seq=hmm_model.states_list[0].stateseq)

            # Save results
            with gzip.open(model_fname, "w") as f:
                print "Saving results to: ", model_fname
                cPickle.dump(res, f, protocol=-1)

        names_list.append(model_name)
        results_list.append(res)
        color_list.append(allcolors[3])


    # Plot
    plot_predictive_log_likelihoods(results_list, color_list, baseline=homog_ll)
