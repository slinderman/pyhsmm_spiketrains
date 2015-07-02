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

### Special case the Poisson Mixture model
class PoissonLabels(pybasicbayes.internals.labels.Labels):
    def _compute_scores(self):
        data, K = self.data, len(self.components)
        scores = np.zeros((data.shape[0],K))
        lmbdas = np.array([o.lmbdas for o in self.components])
        fast_likelihoods(self.gammalns, lmbdas, data, scores)

        # scores2 = np.zeros_like(scores)
        # for idx, c in enumerate(self.components):
        #     scores2[:,idx] = c.log_likelihood(data)
        # assert np.allclose(scores, scores2)

        scores += self.weights.log_likelihood(np.arange(K))
        scores[np.isnan(data).any(1)] = 0. # missing data

        return scores

    @property
    def gammalns(self):
        if not hasattr(self,'_gammalns'):
            self._gammalns = gammaln(np.array(self.data, copy=False) + 1)
        return self._gammalns

    ### speeding up obs resampling
    @property
    def obs_stats(self):
        ns = np.zeros(self.N, dtype='int')
        tots = np.zeros((self.N, self.data.shape[1]), dtype='int')
        fast_statistics(self.z, self.data, ns, tots)
        return ns, tots

class _PoissonMixtureMixin(pybasicbayes.models.Mixture):
    _labels_class = PoissonLabels

    def resample_components(self, num_procs=None):
        if len(self.labels_list) > 0:
            ns, totss = map(sum,zip(*[s.obs_stats for s in self.labels_list]))

            for c, n, tots in zip(self.components, ns, totss):
                c.resample(n=n, tots=tots)
            self._clear_caches()

### Special case the Poisson states to use Cython-ized stat calculations
class PoissonStates(pyhsmm.internals.hmm_states._StatesBase):
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

class PoissonHMMStates(PoissonStates, pyhsmm.models.HMM._states_class):
    pass

class PoissonHSMMStates(PoissonStates, pyhsmm.models.HSMM._states_class):
    pass

### Special case resampling Poisson observation distributions
class _PoissonMixin(pyhsmm.models._HMMGibbsSampling):

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
        # self.resample_obs_hypers()
        self.resample_obs_scale()

    def slow_resample_obs_distns(self):
        super(_PoissonMixin,self).resample_obs_distns()

        # Resample hypers
        # self.resample_obs_hypers()
        self.resample_obs_scale()

    ### resampling observation hypers
    def resample_obs_hypers_hmc(self):
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
        a_f = a_0.copy()
        b_f = b_0.copy()
        for n in xrange(self.obs_distns[0].N):
            nlp = lambda x: nlpc(x,n)
            gnlp = lambda x: gnlpc(x,n)

            x0 = np.array([np.log(a_0[n]), np.log(b_0[n])])
            xc = hmc(nlp, gnlp, step_sz, n_steps, x0)

            # Set the hypers
            a_f[n], b_f[n] = np.exp(xc)
            # o.hypers = np.exp(xc)

        for o in self.obs_distns:
            o.hypers = (a_f,b_f)

    def resample_obs_scale(self):
        """
        Resample the scale of the gamma prior on firing rates
        p(\lam | a, b) = Gamma(\lam | a, b)
                       = b^a / G(a) \lam^{a-1} e^{-b\lam}

        p(b | c, d) = Gamma(b | c, d)
                    = d^c/ G(c) b^{c-1} e^{-db}

        p( b | lam, a, c, d)
                    = b^a e^{-b \lam} b^{c-1} e^{-db}
                    = b^{a + c - 1} e^{-b * (d+lam)}
        :return:
        """
        assert all(map(lambda o: isinstance(o, PoissonVector),
                       self.obs_distns))
        N = self.obs_distns[0].N
        a, b = self.obs_distns[0].hypers
        c, d = 1., 1.
        L = np.array([o.lmbdas for o in self.obs_distns])
        used = self.state_usages > 0
        for n in xrange(N):
            # Rates of neuron n over all states
            Ln = L[:,n]

            # Posterior parameters (only consider states that are used)
            c_post = c + used.sum()*a[n]
            d_post = d + (Ln * used).sum()

            # Sample and set
            bn = np.random.gamma(c_post, 1./d_post)
            for o in self.obs_distns:
                o.beta_0[n] = bn


### Special case resampling Poisson observation distributions
class _PoissonHMMMixin(_PoissonMixin):
    _states_class = PoissonHMMStates

class _PoissonHSMMMixin(_PoissonMixin):
    _states_class = PoissonHSMMStates

### Now create Poisson versions of the Mixture, DP-Mixture, HMM, HDP-HMM, HSMM, and HDP-HSMM
def _make_obs_distns(K, N, alpha_obs, beta_obs):
    obs_distns = []
    for k in range(K):
        obs_distns.append(PoissonVector(N, alpha_0=alpha_obs, beta_0=beta_obs))
    return obs_distns

class PoissonMixture(_PoissonMixtureMixin, pybasicbayes.models.Mixture):
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonMixture, self).__init__(
            components=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)

        # TODO: Implement component hyperparameter resampling

    def heldout_log_likelihood(self, test_data):
        return self.log_likelihood(test_data)


class PoissonCRPMixture(pybasicbayes.models.CRPMixture):
    def __init__(self, N, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonCRPMixture, self).__init__(
            obs_distn=PoissonVector(N, alpha_0=alpha_obs, beta_0=beta_obs), **kwargs)

        # TODO: Implement component hyperparameter resampling - need to sample lambdas

    def heldout_log_likelihood(self, test_data):
        return self.log_likelihood(test_data)


class PoissonHMM(_PoissonHMMMixin, pyhsmm.models.HMM):
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


class PoissonHDPHMM(_PoissonHMMMixin, pyhsmm.models.WeakLimitHDPHMM):
    def __init__(self, N, K_max, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHDPHMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs), **kwargs)


class PoissonHSMM(_PoissonHSMMMixin, pyhsmm.models.HSMM):
    # TODO: Override PoissonHMMMixin to use HSMMStates
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHSMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)


class PoissonHDPHSMM(_PoissonHSMMMixin, pyhsmm.models.WeakLimitHDPHSMM):
    def __init__(self, N, K_max, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonHDPHSMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs), **kwargs)


