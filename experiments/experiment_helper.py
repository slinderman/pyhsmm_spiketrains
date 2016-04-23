
import os
import gzip
import cPickle

import numpy as np
from scipy.io import loadmat

import pyhsmm_spiketrains.models
reload(pyhsmm_spiketrains.models)
from pyhsmm_spiketrains.internals.utils import split_train_test

def load_synth_data(T, K, N, T_test=1000,
                    model='hdp-hmm',
                    version=1,
                    alpha_obs=1.0, beta_obs=1.0):
    """
    Make a synthetic HMM dataset
    :param T: Number of time bins
    :param K: Number of latent states (or max number in the HDP case
    :param N: Number of neurons
    :return:
    """
    data_dir = "data"
    file_name = 'synth_%s_K%d_T%d_N%d_v%d.pkl' % (model, K, T, N, version)
    data_file = os.path.join(data_dir, file_name)
    if os.path.exists(data_file):
        with gzip.open(data_file, "r") as f:
            res = cPickle.load(f)

    else:
        if model == 'hmm':
            hmm = pyhsmm_spiketrains.models.PoissonHMM(
                    N=N, K=K,
                    alpha_obs=alpha_obs, beta_obs=beta_obs,
                    alpha=12.0, gamma=12.0,
                    init_state_concentration=1.0
                    )

        elif model == 'hdp-hmm':
            hmm = pyhsmm_spiketrains.models.PoissonHDPHMM(
                    N=N, K_max=K,
                    alpha_obs=alpha_obs, beta_obs=beta_obs,
                    alpha=12.0, gamma=12.0,
                    init_state_concentration=1.0,
                    )
        else:
            raise Exception('Unrecognized model')


        S_train, Z_train = hmm.generate(T)
        S_test, Z_test = hmm.generate(T_test)
        print "Num used states: ", len(np.unique(Z_train))
        res = hmm, S_train, Z_train, S_test, Z_test

        with gzip.open(data_file, "w") as f:
            cPickle.dump(res, f, protocol=-1)

    return res

def load_hipp_data(dataname="hipp_2dtrack_a", trainfrac=0.8):
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