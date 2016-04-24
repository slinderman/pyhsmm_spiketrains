import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from experiment_helper import load_hipp_data

from hips.distributions.circular_distribution import CircularDistribution

from pyhsmm_spiketrains.models import PoissonStatic
from pyhsmm_spiketrains.internals.utils import expected_ll, convert_polar_to_xy

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])


def estimate_pos(model, pos_train, S_test, pos_test, center, radius):
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


def compute_mse(results_list, pos_train, S_test, pos_test):

    for results in results_list:
        model = results.samples
        epdf = estimate_pos(model, pos_train, S_test, pos_test, center, radius)

        # Compute the mean trajectory
        T_test = S_test.shape[0]
        mean_location = np.zeros_like(pos_test)
        for t in range(T_test):
            cd = CircularDistribution(center, radius, pdf=epdf[t,:])
            mean_location[t,:] = convert_polar_to_xy(np.atleast_2d(cd.mean), center)

        # Convert estimates to x,y and compute mean squared error
        sqerr = np.sqrt((mean_location - pos_test)**2).mean(axis=1)
        mse = sqerr.mean(axis=0)
        stdse = sqerr.std(axis=0)

        print "MSE: %f \pm %f" % (mse, stdse)

def compute_predictive_log_likelihood(S_train, S_test,
                                      results_list,
                                      N_samples=2000):

    # Fit the baseline model
    static_model = PoissonStatic(N)
    static_model.add_data(S_train)
    static_model.max_likelihood()
    static_ll = static_model.log_likelihood(S_test)


    for results in results_list:
        # Get mean and standard deviation
        m,s = expected_ll(np.array(results.predictive_lls[-N_samples:]))

        # Standardize
        m -= static_ll
        m /= np.sum(S_test)
        s /= np.sum(S_test)
        print "PLL: %.3f +- %.3f" % (m,s)

# Table 2: Pred LL and MSE for hippocampal data
dataset = "hipp_2dtrack_a"
N, S_train, pos_train, S_test, pos_test, center, radius = \
    load_hipp_data(dataname=dataset)

# Load results
runnum = 1
results_dir = os.path.join("results", dataset, "run%03d" % runnum)
results_types = ["hmm_K25", "hmm_K45", "hmm_K65",
                 "hdphmm_scale",
                 "hdphmm_hmc",
                 "hdphmm_eb",
                 "hdphmm_vb",
                 "hdphsmm_scale"]
                 # "hdphmm_scale_alpha_obs0.1",
                 # "hdphmm_scale_alpha_obs0.5",
                 # "hdphmm_scale_alpha_obs1.0",
                 # "hdphmm_scale_alpha_obs2.0",
                 # "hdphmm_scale_alpha_obs2.5",
                 # "hdphmm_scale_alpha_obs5.0",
                 # "hdphmm_scale_alpha_obs10.0"]

for results_type in results_types:
    results_file = os.path.join(results_dir, results_type + ".pkl.gz")
    # print "Loading ", results_file
    print "Model: ", results_type
    with gzip.open(results_file, "r") as f:
        results = cPickle.load(f)

    # print results.name
    compute_predictive_log_likelihood(S_train, S_test, [results])
    compute_mse([results], pos_train, S_test, pos_test)
    print ""

    del results