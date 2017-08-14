from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

import pyhsmm_spiketrains.models

# Set the seed
seed = 0
print("setting seed to ", seed)
np.random.seed(seed)

# Generate a synthetic dataset
N = 25          # Number of neurons
K = 20          # Number of states
N_trials = 10   # Number of trials
T = 100         # Number of time bins to simulate per trial
N_iter = 500    # Number of iterations of Gibbs sampling

# Simulate from an HMM with a known transition matrix
true_A = np.eye(K) + 0.25*np.random.rand(K,K)
true_A /= true_A.sum(axis=1)[:,None]
true_hmm = pyhsmm_spiketrains.models.PoissonHMM(N=N, K=K, trans_matrix=true_A)

# Generate test spike trains (S) and their underlying state sequences (Z).
# Then mask off a fraction of the data.
trials = []
for _ in range(N_trials):
    S, _ = true_hmm.generate(T, keep=False)
    mask = np.random.rand(T, N) <= 0.9
    trials.append((S, mask))
    true_hmm.add_data(S, mask=mask)

true_ll = true_hmm.log_likelihood()
true_hll = true_hmm.heldout_log_likelihood()

# Relabel the states in order of usage
true_hmm.relabel_by_usage()
Zs = true_hmm.stateseqs
N_used = len(true_hmm.used_states)
print("Number of used states: ", N_used)

# Create a test model with the same parameters, and add the data
test_hmm = pyhsmm_spiketrains.models.PoissonHDPHMM(N=N, K_max=100)
for S, mask in trials:
    test_hmm.add_data(S, mask=mask)

# Uncomment these lines if you want to *fix* the observation hyperparameters
# test_hmm._resample_obs_method = "resample_obs_hypers_null"
# for true_obs, test_obs in zip(true_hmm.obs_distns, test_hmm.obs_distns):
#     test_obs.hypers = true_obs.hypers

# Fit the test model with Gibbs sampling
lls = []
hlls = []
for itr in progprint_xrange(N_iter):
    test_hmm.resample_model()

    # Collect the log likelihood and predictive log likelihood
    lls.append(test_hmm.log_likelihood())
    hlls.append(test_hmm.heldout_log_likelihood())

# Get the inferred state sequence
test_hmm.relabel_by_usage()
Z_train_inf = test_hmm.stateseqs[0]
N_used_inf = len(test_hmm.used_states)

# Plot the log likelihood over time
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(lls, 'b')
plt.plot([0,N_iter], true_ll * np.ones(2), ':k')
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.subplot(122)
plt.plot(hlls, 'b')
plt.plot([0,N_iter], true_hll * np.ones(2), ':k')
plt.xlabel("Iteration")
plt.ylabel("Heldout Log Likelihood")
plt.tight_layout()


# Visualize the data and the true and inferred state sequences
plt.figure()
plt.subplot(311)
plt.imshow(trials[0][0][:100].T, interpolation="none", cmap="Greys", vmin=0, vmax=S.max())
plt.title("Spike train")
plt.subplot(312)
plt.imshow(Zs[0].reshape((1, -1))[:, :100], aspect=10.0, cmap="YlOrRd", interpolation="none", vmin=0, vmax=N_used)
plt.title("True states")
plt.subplot(313)
plt.title("Inferred states")
plt.imshow(Z_train_inf.reshape((1,-1))[:,:100], aspect=10.0, cmap="YlOrRd", interpolation="none", vmin=0, vmax=N_used)

# Visualize the true and inferred transition matrices
plt.figure()
plt.subplot(121)
plt.imshow(true_hmm.A[:N_used, :N_used], interpolation="none", cmap="Greys", vmin=0, vmax=1)
plt.title("True Transitions")
plt.subplot(122)
plt.imshow(test_hmm.A[:N_used_inf, :N_used_inf], interpolation="none", cmap="Greys", vmin=0, vmax=1)
plt.title("Inf. Transitions")

# Visualize the true and inferred firing rates
plt.figure()
plt.subplot(121)
plt.imshow(true_hmm.rates[:N_used, :], interpolation="none", cmap="Greys", vmin=0, vmax=1)
plt.title("True Firing Rates")
plt.subplot(122)
plt.imshow(test_hmm.rates[:N_used_inf, :], interpolation="none", cmap="Greys", vmin=0, vmax=1)
plt.title("Inf. Firing Rates")

plt.show()
