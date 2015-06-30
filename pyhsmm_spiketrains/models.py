from __future__ import division
import numpy as np
from scipy.special import gammaln, psi

import pyhsmm
import pybasicbayes

from hips.inference.hmc import hmc

from pyhsmm_spiketrains.internals.poisson_observations \
    import PoissonVector
from pyhsmm_spiketrains.internals.poisson_statistics \
    import fast_likelihoods, fast_statistics

### Special case the Poisson states to use Cython-ized stat calculations
class PoissonHMMStates(pyhsmm.models.WeakLimitHDPHMM._states_class):
    ### speeding up likelihood calculation
    @property
    def aBl(self):
        if self._aBl is None:
            gammalns = self.gammalns
            lmbdas = np.array([o.lmbdas for o in self.obs_distns])
            data = np.array(self.data,copy=False) # self.data is not a plain ndarray

            self._aBl = np.zeros((self.T,self.num_states))
            fast_likelihoods(gammalns,lmbdas,data,self._aBl)

        return self._aBl

    @property
    def slow_aBl(self):
        return super(PoissonHMMStates,self).aBl

    @property
    def gammalns(self):
        if not hasattr(self,'_gammalns'):
            self._gammalns = gammaln(np.array(self.data,copy=False) + 1)
        return self._gammalns

    ### speeding up obs resampling
    @property
    def obs_stats(self):
        ns = np.zeros(self.num_states,dtype='int')
        tots = np.zeros((self.num_states,self.data.shape[1]),dtype='int')
        fast_statistics(self.stateseq,self.data,ns,tots)
        return (ns,tots)

### Special case resampling Poisson observation distributions
class _PoissonMixin(pyhsmm.models._HMMGibbsSampling):
    _states_class = PoissonHMMStates

    ### speeding up obs resampling
    def resample_obs_distns(self):
        if len(self.states_list) > 0:
            ns, totss = map(sum,zip(*[s.obs_stats for s in self.states_list]))

            for o, n, tots in zip(self.obs_distns,ns,totss):
                o.resample(n=n,tots=tots)
            self._clear_caches()
        else:
            super(_PoissonMixin,self).resample_obs_distns()

        # Resample hypers
        self.resample_obs_hypers()

    def slow_resample_obs_distns(self):
        super(_PoissonMixin,self).resample_obs_distns()

        # Resample hypers
        self.resample_obs_hypers()

    ### resampling observation hypers
    def resample_obs_hypers(self):
        """
        Sample the parameters of a gamma prior given firing rates L

        log p(L[:,c] | a_0[c], b_0[c]) =
            \sum_k a_0[c] log b_0[c] - gammaln(a_0[c]) + (a_0[c]-1) log(L[k,c]) - b_0[c] L[k,c]

        We place a improper uniform prior over log a_0[c] and log b_0[c],
        which effectively introduces a prior of the form p(a_0) = const/a_0

        Since a_0 and b_0 are required to be positive, we work in log space

        :param a_0:
        :param b_0:
        :param L: a K x C matrix of firing rates for each cell
        :return:
        """
        a_0, b_0 = self.obs_distns[0].hypers
        L = np.array([o.lmbdas for o in self.obs_distns])

        # Use a gamma(aa,bb) prior over a_0 and b_0
        aa = 3.
        bb = 3.

        # Define helpers for log prob and its gradient
        def nlpc(x, c):
            lna = x[0]
            lnb = x[1]

            a = np.exp(lna)
            b = np.exp(lnb)
            ll =  (a * np.log(b) - gammaln(a) + (a-1) * np.log(L[:,c]) - b * L[:,c]).sum()

            # Prior is constant with respect to log a and log b (i.e. x)
            # lprior = 0
            lprior = (aa) * np.log(a) - bb*a
            lprior += (aa) * np.log(b) - bb*b

            lp = ll + lprior
            return -lp

        def gnlpc(x, c):

            # import pdb; pdb.set_trace()
            lna = x[0]
            lnb = x[1]

            a = np.exp(lna)
            b = np.exp(lnb)
            dll_da =  (np.log(b) - psi(a) + np.log(L[:,c])).sum()
            dll_db =  (a/b  - L[:,c]).sum()

            # Prior is constant with respect to log a and log b (i.e. x)
            # dlprior_da = 0
            # dlprior_db = 0

            dlprior_da = aa/a - bb
            dlprior_db = aa/b - bb

            dlp_da = dll_da + dlprior_da
            dlp_db = dll_db + dlprior_db

            da_dlna = a
            db_dlnb = b

            dlp_dlna = dlp_da * da_dlna
            dlp_dlnb = dlp_db * db_dlnb

            return np.array([-dlp_dlna, -dlp_dlnb])

        n_steps = 10
        step_sz = 0.01

        # Update the hypers for each cell
        for n,o in enumerate(self.obs_distns):
            nlp = lambda x: nlpc(x,n)
            gnlp = lambda x: gnlpc(x,n)

            x0 = np.array([np.log(a_0[n]), np.log(b_0[n])])
            xc = hmc(nlp, gnlp, step_sz, n_steps, x0)

            # Set the hypers
            o.hypers = np.exp(xc)


### Now create Poisson versions of the Mixture, DP-Mixture, HMM, HDP-HMM, HSMM, and HDP-HSMM
def _make_obs_distns(K, N, alpha_obs, beta_obs):
    obs_distns = []
    for k in range(K):
        obs_distns.append(PoissonVector(N, alpha_0=alpha_obs, beta_0=beta_obs))
    return obs_distns


class PoissonMixture(pybasicbayes.models.Mixture):
    def __init__(self, K, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonMixture, self).__init__(
            components=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)

        # TODO: Implement component hyperparameter resampling


class PoissonCRPMixture(pybasicbayes.models.CRPMixture):
    def __init__(self, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonCRPMixture, self).__init__(
            obs_distn=PoissonVector(N, alpha_0=alpha_obs, beta_0=beta_obs), **kwargs)

        # TODO: Implement component hyperparameter resampling - need to sample lambdas


class PoissonHMM(_PoissonMixin, pyhsmm.models.HMM):
    def __init__(self, K, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


class PoissonHDPHMM(_PoissonMixin, pyhsmm.models.WeakLimitHDPHMM):
    def __init__(self, K, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHDPHMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


class PoissonHSMM(_PoissonMixin, pyhsmm.models.HSMM):
    def __init__(self, K, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHSMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


class PoissonHDPHSMM(_PoissonMixin, pyhsmm.models.WeakLimitHDPHSMM):
    def __init__(self, K, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHDPHSMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


