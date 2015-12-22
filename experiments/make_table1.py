import os
import cPickle
import gzip
from collections import namedtuple

import numpy as np

import brewer2mpl
allcolors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors

from pyhsmm.util.general import hamming_error

from experiment_helper import load_synth_data

from pyhsmm_spiketrains.models import PoissonStatic
from pyhsmm_spiketrains.internals.utils import expected_ll, permute_stateseq
from pyhsmm_spiketrains.internals.hungarian import linear_sum_assignment

Results = namedtuple(
    "Results", ["name", "loglikes", "predictive_lls",
                "N_used", "alphas", "gammas",
                "rates", "obs_hypers",
                "samples", "timestamps"])


def compute_predictive_log_likelihood(S_train, S_test,
                                      results,
                                      N_samples=2000):

    # Fit the baseline model
    static_model = PoissonStatic(N)
    static_model.add_data(S_train)
    static_model.max_likelihood()
    static_ll = static_model.log_likelihood(S_test)

    # Get mean and standard deviation
    m,s = expected_ll(np.array(results.predictive_lls[-N_samples:]))

    # Standardize
    m -= static_ll
    m /= np.sum(S_test)
    s /= np.sum(S_test)
    print "PLL: %.3f +- %.3f" % (m,s)
    return m

def compute_state_overlap(Z_true, Z_inf,K):
    """
    Compute the overlap between true and inferred state sequences
    using the Hungarian algorithm.
    :param Z_true:
    :param Z_inf:
    :return:
    """
    assert Z_true.ndim == Z_inf.ndim == 1
    assert Z_true.size == Z_inf.size
    assert Z_true.max() <= K
    assert Z_inf.max() <= K

    # Make the cost matrix
    C = np.zeros((K,K))
    for k1 in xrange(K):
        for k2 in xrange(K):
            C[k1,k2] = np.sum(np.bitwise_and(Z_true==k1, Z_inf==k2))

    # Find a permutation that maximizes overlap
    true_perm, inf_perm = linear_sum_assignment(-C)

    Z_true_hat = permute_stateseq(true_perm, Z_true)
    Z_inf_hat = permute_stateseq(inf_perm, Z_inf)
    assert np.allclose(Z_true, Z_true_hat)

    # Compute the Hamming error
    herr = hamming_error(Z_true_hat, Z_inf_hat)

    print "Hamming Error: ", herr
    return herr

def make_table(table, row_headers, col_headers, fmt="{0.0f}"):
    rows,cols = table.shape
    tex = " "
    for col_header in col_headers:
        tex += " & " + str(col_header)
    tex += " \\\\"

    for i in xrange(rows):
        tex += str(row_headers[i])
        for j in xrange(cols):
            tex += " & " + fmt.format(table[i,j])
        tex += " \\\\ "

    return tex

# Table 1: Pred LL and Hamming Error for synthetic data
if __name__ == "__main__":
    # Load synthetic dataset
    true_model_type = "hdp-hmm"
    T = 2000
    K = 100
    N = 50
    T_test = 1000
    runnum = 1
    versions = xrange(1,6)
    results_types = ["hmm", "hmm_vb", "hdphmm_scale", "hdphmm_hmc", "hdphmm_eb", "hdphmm_vb"]
    # results_types = ["hdphmm_vb"]
    pll_table = np.zeros((len(versions), len(results_types)))
    err_table = np.zeros((len(versions), len(results_types)))

    for i,version in enumerate(versions):
        results_dir = os.path.join("results",
                                   "synth_%s_T%d_K%d_N%d_v%d" % (true_model_type, T, K, N, version),
                                   "run%03d" % runnum)

        model, S_train, Z_train, S_test, _ = \
            load_synth_data(T, K, N, T_test=T_test,
                            model=true_model_type, version=version)

        # Load results
        for j,results_type in enumerate(results_types):
            results_file = os.path.join(results_dir, results_type + ".pkl.gz")
            print "Loading ", results_file
            with gzip.open(results_file, "r") as f:
                results = cPickle.load(f)
                # print results.N_used[-1]

            pll_table[i,j] = compute_predictive_log_likelihood(S_train, S_test, results)

            # Set the stateseq for vb
            # if results_type == "hdphmm_vb":
            #     # results.samples.states_list[0].Viterbi()
            #     statesobj = results.samples.states_list[0]
            #     statesobj.resample()


            err_table[i,j] = compute_state_overlap(Z_train, results.samples.stateseqs[0], K)

            print ""

            del results

    pll_tex = make_table(pll_table.T, results_types, versions, fmt="{0:.3f}")
    err_tex = make_table(err_table.T, results_types, versions, fmt="{0:.0f}")