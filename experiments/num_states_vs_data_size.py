"""
Measure the number of inferred states as a function of:
    - number of observed neurons
    - time bin size
    - length of recording

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
alpha_obs = 1.0
beta_obs = 1.0
N_iters = 500

def generate_synth_data():
    model = PoissonHDPHMM(N,
                          K_max=K,
                          alpha=25.0,
                          gamma=25.0,
                          alpha_obs=alpha_obs,
                          beta_obs=beta_obs,
                          init_state_concentration=1.)
    data, _ = model.generate(T, keep=True)

    # Permute the hmm_model by state usage
    model.relabel_by_usage()
    plt.bar(np.arange(K), model.state_usages)

    return model, data

def N_used(model):
    return (model.state_usages > 0).sum()

def fit_model(model, N_iters):
    def _evaluate(model):
        ll = model.log_likelihood()
        return ll, N_used(model)

    def _step(model):
        model.resample_model()
        return _evaluate(model)

    results = [_step(model) for _ in progprint_xrange(N_iters)]
    lls = np.array([r[0] for r in results])
    Ns = np.array([r[1] for r in results])
    return lls, Ns

if __name__ == "__main__":
    # Generate synth data
    true_model, data = generate_synth_data()

    # Now fit the model with a model using all the data
    test_model = PoissonHDPHMM(N,
                               K_max=K,
                               alpha=50.0,
                               gamma=50.0,
                               alpha_obs=alpha_obs,
                               beta_obs=beta_obs,
                               init_state_concentration=1.
                               )

    test_model.add_data(data)
    lls, Ns = fit_model(test_model, N_iters)

    plt.figure()
    plt.plot(lls)

    plt.figure()
    plt.plot(Ns)
    plt.plot([0, N_iters], N_used(true_model) * np.ones(2))
    plt.ylim(0,K)
