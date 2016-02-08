
import os
import time
import gzip
import cPickle

import numpy as np
from collections import namedtuple

from pybasicbayes.util.text import progprint_xrange

import matplotlib.pyplot as plt
import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

import pyhsmm_spiketrains.models
reload(pyhsmm_spiketrains.models)

from pyhsmm_spiketrains.internals.utils import \
    log_expected_pll, split_train_test

from experiment_helper import load_synth_data

# Set the seed
# seed = np.random.randint(0, 2**16)
seed = 0
print "setting seed to ", seed
np.random.seed(seed)

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

def fit_vb(name, model, test_data, N_iter=1000, init_state_seq=None):
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
        model.meanfield_coordinate_descent_step()
        timestep = time.time() - tic

        # Resample from mean field posterior
        model._resample_from_mf()

        return evaluate(model), timestep

    # Initialize with given state seq
    if init_state_seq is not None:
        model.states_list[0].stateseq = init_state_seq
        for _ in xrange(100):
            model.resample_obs_distns()

    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(200)])

    lls, plls, N_used, alphas, gammas, rates, obs_hypers = \
        zip(*((init_val,) + vals))
    timestamps = np.cumsum((0.,) + timesteps)

    return Results(name, lls, plls, N_used, alphas, gammas,
                   rates, obs_hypers,
                   model.copy_sample(), timestamps)

def make_hmm_models(N, S_train, Ks=np.arange(5,25, step=5), **kwargs):
    # Define a sequence of models
    names_list = []
    fnames_list = []
    hmm_list = []
    color_list = []
    method_list = []

    assert len(Ks) == 1
    for K in Ks:
        names_list.append("HMM (K=%d)" % K)
        fnames_list.append("hmm")
        color_list.append(allcolors[0])
        hmm = \
            pyhsmm_spiketrains.models.PoissonHMM(
                N=N, K=K, alpha_a_0=5.0, alpha_b_0=1.0,
                init_state_concentration=1.0,
                **kwargs)
        hmm.add_data(S_train)
        hmm_list.append(hmm)
        method_list.append(fit)

        names_list.append("HMM VB (K=%d)" % K)
        fnames_list.append("hmm_vb")
        color_list.append(allcolors[0])
        hmm = \
            pyhsmm_spiketrains.models.PoissonHMM(
                N=N, K=K, alpha=12.0,
                init_state_concentration=1.0,
                **kwargs)
        hmm.add_data(S_train)
        hmm_list.append(hmm)
        method_list.append(fit_vb)

    return names_list, fnames_list, color_list, hmm_list, method_list

def make_hdphmm_models(N, S_train, K_max=100, **kwargs):
    # Define a sequence of models
    names_list = []
    fnames_list = []
    hmm_list = []
    method_list = []
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
    method_list.append(fit)

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
    method_list.append(fit)

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
    method_list.append(fit)

    # HDP-HMM with hypers set by mean field
    # TODO: Support concentration priors!
    names_list.append("HDP-HMM (VB)")
    fnames_list.append("hdphmm_vb")
    color_list.append(allcolors[1])
    hmm = \
        pyhsmm_spiketrains.models.PoissonDATruncHDPHMM(
            N=N, K_max=K_max,
            alpha=12.0,
            gamma=12.0,
            init_state_concentration=1.0,
            **kwargs)
    hmm.add_data(S_train, stateseq=np.random.randint(50, size=(S_train.shape[0],)))
    hmm.init_obs_hypers_via_empirical_bayes()
    hmm_list.append(hmm)
    method_list.append(fit_vb)

    return names_list, fnames_list, color_list, hmm_list, method_list


def run_experiment(T, K, N, T_test, modelname, version, runnum):
    data_name = "synth_%s_T%d_K%d_N%d_v%d" % (modelname, T, K, N, version)
    results_dir = os.path.join("results",
                               data_name,
                               "run%03d" % runnum)

    hmm, S_train, _, S_test, _ = \
        load_synth_data(T, K, N, T_test=T_test,
                        model=modelname, version=version)

    S_train = S_train.reshape((-1, N))
    S_test = S_test.reshape((-1, N))

    # Set output parameters
    N = hmm.N
    K_true = len(hmm.used_states)

    print "Running Synthetic Experiment"
    print "Dataset:\t", data_name
    print "N:\t\t", N
    print "N_used:\t", len(hmm.used_states)
    print "T_train:\t", S_train.shape[0]
    print "T_test:\t", S_test.shape[0]

    # Define a set of HMMs
    names_list = []
    fnames_list = []
    color_list = []
    model_list = []
    method_list = []

    # Add parametric HMMs
    # nl, fnl, cl, ml, mtdl = \
    #     make_hmm_models(N, S_train, Ks=np.array([K_true]),
    #                     alpha_obs=1.0, beta_obs=1.0)
    # names_list.extend(nl)
    # fnames_list.extend(fnl)
    # color_list.extend(cl)
    # model_list.extend(ml)
    # method_list.extend(mtdl)

    # Add HDP_HMMs
    nl, fnl, cl, ml, mtdl = \
        make_hdphmm_models(N, S_train, K_max=100,
                           alpha_obs=1.0, beta_obs=1.0)
    names_list.extend(nl)
    fnames_list.extend(fnl)
    color_list.extend(cl)
    model_list.extend(ml)
    method_list.extend(mtdl)

    # Fit the models with Gibbs sampling
    N_iter = 5000
    # results_list = []
    for model_name, model_fname, model, method in \
            zip(names_list, fnames_list, model_list, method_list):
        print "Model: ", model_name
        print "File:  ", model_fname
        print ""
        output_file = os.path.join(results_dir, model_fname + ".pkl.gz")

        # Check for existing results
        if os.path.exists(output_file):
            print "Found results at: ", output_file

        else:
            res = method(model_name, model, S_test, N_iter=N_iter)

            # Save results
            with gzip.open(output_file, "w") as f:
                print "Saving results to: ", output_file
                cPickle.dump(res, f, protocol=-1)

            del res


if __name__ == "__main__":
    modelname = "hdp-hmm"
    T = 2000
    T_test = 1000
    K = 100
    N = 1
    version = 1
    runnum = 1

    for version in xrange(1,11):
        run_experiment(T, K, N, T_test, modelname, version, runnum)