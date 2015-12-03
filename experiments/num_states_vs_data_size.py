"""
Measure the number of inferred states as a function of:
    - number of observed neurons
    - time bin size
    - length of recording
    - firing rate
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from pyhsmm.util.text import progprint_xrange

from pyhsmm_spiketrains.models import PoissonHDPHMM

# Set the random seed for reproducibility
np.random.seed(0)
T = 10000
K = 100
N = 50
alpha = 25.0
gamma = 25.0
alpha_obs = 1.0
beta_obs = 1.0
N_iters = 100

def generate_synth_data():
    model = PoissonHDPHMM(N,
                          K_max=K,
                          alpha=alpha,
                          gamma=alpha,
                          alpha_obs=alpha_obs,
                          beta_obs=beta_obs,
                          init_state_concentration=1.)
    data, _ = model.generate(T, keep=True)

    # Permute the hmm_model by state usage
    model.relabel_by_usage()
    # plt.bar(np.arange(K), model.state_usages)

    return model, data

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

if __name__ == "__main__":
    # Generate synth data
    true_model, data = generate_synth_data()
    K_true = K_used(true_model)

    N_repeats = 10

    # Experiments with all the data
    Ks_all = []
    for rpt in xrange(N_repeats):
        # Fit
        print "All. Repeat: ", rpt
        _, Ks = test_all(data)
        Ks_all.append(Ks[-1])

    # Experiments with subsets of neurons
    Ns_test = (np.array([0.1, 0.2, 0.5, 0.8, 1.0]) * N).astype(np.int)
    Ks_Ns = []
    for N_test in Ns_test:
        Ks_N = []
        for rpt in xrange(N_repeats):
            print "N_test: ", N_test, ". Repeat: ", rpt
            _, Ks = test_N(data, N_test)
            Ks_N.append(Ks[-1])
        Ks_Ns.append(Ks_N)

    # Experiments with subsets of time bins
    Ts_test = (np.array([0.1, 0.2, 0.5, 0.8, 1.0]) * T).astype(np.int)
    Ks_Ts = []
    for T_test in Ts_test:
        Ks_T = []
        for rpt in xrange(N_repeats):
            print "T_test: ", T_test, ". Repeat: ", rpt
            _, Ks = test_T(data, T_test)
            Ks_T.append(Ks[-1])
        Ks_Ts.append(Ks_T)


    # Experiments with varying firing rates
    frs_test = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    Ks_frs = []
    for fr_test in frs_test:
        Ks_fr = []
        for rpt in xrange(N_repeats):
            print "fr_test: ", fr_test, ". Repeat: ", rpt
            _, Ks = test_fr(true_model, fr_test)
            Ks_fr.append(Ks[-1])
        Ks_frs.append(Ks_fr)


    # plt.figure()
    # plt.plot([0, N_iters], K_true * np.ones(2), '--k')