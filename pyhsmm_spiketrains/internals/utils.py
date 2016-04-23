import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.misc import logsumexp

def log_expected_pll(plls):
    return -np.log(len(plls)) + logsumexp(plls)


def expected_ll(plls, subsamples=20, subsmplfrac=0.75):
    """
    Compute the expected pred ll and its standard deviation by
    using random subsamples of the pred ll samples.
    """
    N = len(plls)
    subplls = np.zeros(subsamples)
    for s in range(subsamples):
        insubset = np.random.rand(N) < subsmplfrac
        subset = plls[insubset]
        subplls[s] = logsumexp(subset) - np.log(len(subset))

    mean_pll = subplls.mean()
    std_pll = subplls.std()

    return mean_pll, std_pll


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


def fit_nbinom(kin, rmin=1e-1):
    """
    Fit the parameters of a negative binomial to a set of spike count observations
    :param kin:
    :return:
    """
    k = np.array(kin, dtype=np.float)
    N = len(k)
    sumx = np.sum(k)
    xbar = sumx / N

    def nll(r):
        return -np.sum(gammaln(r+k)) + N*gammaln(r) \
               - N*r*np.log(r/(xbar+r)) - sumx*np.log(xbar/(xbar+r))

    r = minimize(nll,1.0, bounds=[(rmin, None)], method='L-BFGS-B')
    r = r.x
    q = r/(r + np.sum(k/N))  # Note: this `p` = 1 - `p` from Wikipedia
    p = 1-q
    return r, p


def convert_xy_to_polar(pos, center, radius=np.Inf):
    # Convert true position to polar
    pos_c = (pos - center)
    pos_r = np.sqrt((pos_c**2).sum(axis=1))
    pos_r = np.clip(pos_r, 0, radius)
    pos_th = np.arctan2(pos_c[:,1], pos_c[:,0])

    return pos_r, pos_th

def convert_polar_to_xy(pos, center):
    # Convert true position to polar
    pos_x = center[0] + pos[:,0] * np.cos(pos[:,1])
    pos_y = center[1] + pos[:,0] * np.sin(pos[:,1])

    return pos_x, pos_y

def permute_stateseq(perm, stateseq):
    ranks = np.argsort(perm)
    perm_stateseq = np.nan*np.ones(stateseq.size)
    good = ~np.isnan(stateseq)
    perm_stateseq[good] = ranks[stateseq[good].astype('int32')]
    return perm_stateseq