from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

import pyhsmm_spiketrains.models

# Set the seed
seed = 0
print("setting seed to ", seed)
np.random.seed(seed)

# Generate a synthetic dataset with two populations
N1 = 25         # Number of neurons in pop. 1
N2 = 25         # Number of neurons in pop. 2
K = 20          # Number of states
T_train = 1000  # Number of time bins to simulate per trial
T_test = 100    # Number of time bins to simulate per trial
N_iter = 500    # Number of iterations of Gibbs sampling

# Simulate from an HMM with a known transition matrix
true_A = np.eye(K) + 0.25*np.random.rand(K,K)
true_A /= true_A.sum(axis=1)[:,None]
true_hmm = pyhsmm_spiketrains.models.PoissonHMM(N=N1+N2, K=K, trans_matrix=true_A)

# Generate training data and testing data
S_train, _ = true_hmm.generate(T_train, keep=True)
S1_train, S2_train = S_train[:, :N1].copy('C'), S_train[N1:].copy('C')
S_test, _ = true_hmm.generate(T_test, keep=True)
S1_test, S2_test = S_test[:, :N1].copy('C'), S_test[N1:].copy('C')

# Mask off the activity from population 2
M1_train = np.hstack([np.ones((T_train, N1), dtype=bool), np.zeros((T_train, N2), dtype=bool)])
M1_test = np.hstack([np.ones((T_test, N1), dtype=bool), np.zeros((T_test, N2), dtype=bool)])

# Compute true likelihoods
ll_S1tr_true = true_hmm.log_likelihood(S_train, mask=M1_train)
ll_S1te_true = true_hmm.log_likelihood(S_test, mask=M1_test)
ll_S2te_given_S1te_true = true_hmm.predictive_log_likelihood(S_test, mask=M1_test)

print("Fitting naive model on population 1 alone")
naive_hmm = pyhsmm_spiketrains.models.PoissonHDPHMM(N=N1+N2, K_max=100)
naive_hmm.add_data(S1_train)
for itr in progprint_xrange(N_iter):
    naive_hmm.resample_model()

# Evaluate the marginal likelihood of population 1 under the naive model
ll_S1tr_naive = naive_hmm.log_likelihood(S1_train)
ll_S1te_naive = naive_hmm.log_likelihood(S1_test)

print("Fitting augmented model on populations 1 and 2")
aug_hmm = pyhsmm_spiketrains.models.PoissonHDPHMM(N=N1+N2, K_max=100)
aug_hmm.add_data(S_train)
for itr in progprint_xrange(N_iter):
    aug_hmm.resample_model()

# Evaluate the likelihood of population 1 under the full model
ll_S1tr_aug = aug_hmm.log_likelihood(S_train, mask=M1_train)
ll_S1te_aug = aug_hmm.log_likelihood(S_test, mask=M1_test)

# We can also compute the likelihood of the entire dataset
ll_Str = aug_hmm.log_likelihood(S_train)
ll_Ste = aug_hmm.log_likelihood(S_test)

# And we can compute the predictive likelihood using masks
ll_S2te_given_S1te_aug = aug_hmm.predictive_log_likelihood(S_test, mask=M1_test)


print("Let theta_naive denote the parameters learned by naively "
      "fitting an HMM to population 1 only. Also, let theta_aug "
      "denote the parameters learned by fitting an HMM to "
      "populations 1 and 2 jointly.")
print("")
print("p(Y1_train | theta_true):            ", ll_S1tr_true)
print("p(Y1_train | theta_naive):           ", ll_S1tr_naive)
print("p(Y1_train | theta_aug):             ", ll_S1tr_aug)
print("")
print("p(Y1_test  | theta_true):            ", ll_S1te_true)
print("p(Y1_test |  theta_naive):           ", ll_S1te_naive)
print("p(Y1_test | theta_aug):              ", ll_S1te_aug)
print("")
print("We can compute the predictive likelihood of one population "
      "given another using a full model. Note that this is not possible "
      "with the 'naive' model. ")
print("")
print("p(Y2_test | Y1_test, theta_true):    ", ll_S2te_given_S1te_true)
print("p(Y2_test | Y1_test, theta_aug):     ", ll_S2te_given_S1te_aug)
