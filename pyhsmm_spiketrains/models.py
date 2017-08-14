from __future__ import division

import operator
import numpy as np
from scipy.special import gammaln, psi

import pyhsmm
import pybasicbayes

from hips.inference.hmc import hmc

from pyhsmm_spiketrains.internals.poisson_observations \
    import PoissonVector
from pyhsmm_spiketrains.internals.poisson_statistics \
    import fast_likelihoods, fast_statistics
from pyhsmm_spiketrains.internals.utils \
    import fit_nbinom

### Special case the Poisson Mixture model
class PoissonLabels(pybasicbayes.models.Labels):
    def __init__(self, model, data=None, mask=None, N=None, z=None,
                 initialize_from_prior=True):

        super(PoissonLabels, self).__init__(
            model, data=data, N=N, z=z,
            initialize_from_prior=initialize_from_prior)

        if mask is None:
            self.mask = np.ones(data.shape, dtype=np.int32)
        else:
            assert mask.shape == data.shape and mask.dtype == np.int32
            self.mask = mask

    def _compute_scores(self):
        data, mask, K = self.data, self.mask, len(self.components)
        scores = np.zeros((data.shape[0],K))
        lmbdas = np.array([o.lmbdas for o in self.components])
        fast_likelihoods(self.gammalns, lmbdas, data, mask, scores)

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
        fast_statistics(self.z, self.data, self.mask, ns, tots)
        return ns, tots

### Special case the Poisson states to use Cython-ized stat calculations
class _PoissonStatesMixin(object):
    def __init__(self, model, T=None, data=None, mask=None, stateseq=None,
                 generate=True, initialize_from_prior=True, fixed_stateseq=False):

        super(_PoissonStatesMixin, self).__init__(
            model, T=T, data=data, stateseq=stateseq, generate=generate,
            initialize_from_prior=initialize_from_prior,
            fixed_stateseq=fixed_stateseq)

        if mask is None:
            self.mask = np.ones((self.T, self.N), dtype=np.int32)
        else:
            assert mask.shape == (self.T, self.N) and mask.dtype == bool
            self.mask = mask.astype(np.int32)

    @property
    def N(self):
        return self.obs_distns[0].N

    ### speeding up likelihood calculation
    @property
    def aBl(self):
        if self._aBl is None:
            gammalns = self.gammalns
            lmbdas = np.array([o.lmbdas for o in self.obs_distns])
            data = np.array(self.data, copy=False) # self.data is not a plain ndarray

            self._aBl = np.zeros((self.T,self.num_states))
            fast_likelihoods(gammalns, lmbdas, data, self.mask, self._aBl)

        return self._aBl

    @property
    def slow_aBl(self):
        return super(_PoissonStatesMixin, self).aBl

    @property
    def gammalns(self):
        if not hasattr(self,'_gammalns'):
            self._gammalns = gammaln(np.array(self.data,copy=False) + 1)
        return self._gammalns

    ### speeding up obs resampling
    @property
    def obs_stats(self):
        ns = np.zeros(self.num_states, dtype='int')
        tots = np.zeros((self.num_states, self.data.shape[1]), dtype='int')
        data = np.array(self.data, copy=False)  # self.data is not a plain ndarray
        fast_statistics(self.stateseq, data, self.mask, ns, tots)
        return ns, tots

    @property
    def rate(self):
        lmbdas = np.array([o.lmbdas for o in self.obs_distns])
        rate = lmbdas[self.stateseq,:]
        assert rate.shape == (self.T, self.obs_distns[0].N)
        return rate

    # Heldout log likelihoods
    def heldout_log_likelihood(self):
        gammalns = self.gammalns
        lmbdas = np.array([o.lmbdas for o in self.obs_distns])
        data = np.array(self.data, copy=False)  # self.data is not a plain ndarray

        # Compute expected states given observed data
        self.E_step()

        # Compute the heldout log likelihood for each latent state
        heldout_aBl = np.zeros((self.T, self.num_states))
        fast_likelihoods(gammalns, lmbdas, data, 1 - self.mask, heldout_aBl)
        return np.sum(heldout_aBl * self.expected_states)


class PoissonHMMStates(_PoissonStatesMixin, pyhsmm.models.HMM._states_class):
    pass

class PoissonHSMMStates(_PoissonStatesMixin, pyhsmm.models.HSMM._states_class):
    pass

class PoissonIntNegBinHSMMStates(_PoissonStatesMixin, pyhsmm.models.HSMMIntNegBin._states_class):
    pass

### Special case resampling Poisson observation distributions
class _PoissonMixin(pyhsmm.models._HMMGibbsSampling):

    _resample_obs_method = "resample_obs_scale"
    @property
    def N(self):
        return self.obs_distns[0].N

    @property
    def rates(self):
        return np.array([o.lmbdas for o in self.obs_distns])

    @property
    def obs_hypers(self):
        return self.obs_distns[0].hypers

    @property
    def A(self):
        return self.trans_distn.trans_matrix

    # Likelihoods
    def heldout_log_likelihood(self):
        return sum(s.heldout_log_likelihood() for s in self.states_list)

    # Helper function
    def relabel_by_usage(self):
        """
        Relabel the state sequence by usage.
        Permute the transition matrix accordingly.
        """
        stateseqs = self.stateseqs
        N = self.num_states
        usages = sum(np.bincount(l[~np.isnan(l)].astype('int32'), minlength=N)
                     for l in stateseqs)
        perm  = np.argsort(usages)[::-1]
        ranks = np.argsort(perm)

        # Permute the stateseqs
        for stateseq,statesobj in zip(stateseqs,self.states_list):
            perm_stateseq = np.empty_like(stateseq)
            good = ~np.isnan(stateseq)
            perm_stateseq[good] = ranks[stateseq[good].astype('int32')]
            if np.isnan(stateseq).any():
                perm_stateseq[~good] = np.nan

            statesobj.stateseq = perm_stateseq

        # Permute the observations
        perm_obs_distns = [self.obs_distns[i] for i in perm]
        self.obs_distns = perm_obs_distns

        if hasattr(self, "dur_distns"):
            perm_dur_distns = [self.dur_distns[i] for i in perm]
            self.dur_distns = perm_dur_distns

        # Permute the transition matrix
        perm_A = self.trans_distn.trans_matrix[np.ix_(perm, perm)]
        self.trans_distn.trans_matrix = perm_A


    ### speeding up obs resampling
    def resample_obs_distns(self):
        if len(self.states_list) > 0:
            ns, totss = map(sum,zip(*[s.obs_stats for s in self.states_list]))

            for o, n, tots in zip(self.obs_distns, ns, totss):
                o.resample(n=n, tots=tots)
            self._clear_caches()
        else:
            super(_PoissonMixin,self).resample_obs_distns()

        # Resample hypers
        operator.methodcaller(self._resample_obs_method)(self)

    def slow_resample_obs_distns(self):
        super(_PoissonMixin,self).resample_obs_distns()

        # Resample hypers
        operator.methodcaller(self._resample_obs_method)(self)

    ### resampling observation hypers
    def resample_obs_hypers_null(self):
        # Dummy for skipping obs resampling
        pass

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
        step_sz = 0.001

        # Update the hypers for each cell
        a_f = a_0.copy()
        b_f = b_0.copy()
        for n in range(self.obs_distns[0].N):
            nlp = lambda x: nlpc(x,n)
            gnlp = lambda x: gnlpc(x,n)

            x0 = np.array([np.log(a_0[n]), np.log(b_0[n])])
            xc = hmc(nlp, gnlp, step_sz, n_steps, x0)

            # Set the hypers
            a_f[n], b_f[n] = np.exp(xc)

        # Truncate to make sure > 0
        a_f = np.clip(a_f, 1e-4, np.inf)
        b_f = np.clip(b_f, 1e-4, np.inf)

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
        for n in range(N):
            # Rates of neuron n over all states
            Ln = L[:,n]

            # Posterior parameters (only consider states that are used)
            c_post = c + used.sum()*a[n]
            d_post = d + (Ln * used).sum()

            # Sample and set
            bn = np.random.gamma(c_post, 1./d_post)
            for o in self.obs_distns:
                o.poissons[n].beta_0 = bn

    def init_obs_hypers_via_empirical_bayes(self):
        """
        Initialize the firing rate hyperparameters with EB
        """
        assert all(map(lambda o: isinstance(o, PoissonVector),
                       self.obs_distns))

        # Get the data
        alldata = np.vstack([s.data for s in self.states_list])
        assert alldata.size > 0

        # Initialize spike rate hypers with the maximum likelihood estimate
        N = self.obs_distns[0].N
        alpha = np.zeros(N)
        beta = np.zeros(N)

        for n in range(N):
            rc, pc = fit_nbinom(alldata[:,n])
            alpha[n] = rc
            beta[n] = 1.0/pc - 1

        # Observation is a product of independent Poisson distributions
        for o in self.obs_distns:
            o.hypers = alpha, beta


### Special case resampling Poisson observation distributions
class _PoissonHMMMixin(_PoissonMixin):
    _states_class = PoissonHMMStates

class _PoissonHSMMMixin(_PoissonMixin):
    _states_class = PoissonHSMMStates

class _PoissonIntNegBinHSMMMixin(_PoissonHSMMMixin):
    _states_class = PoissonIntNegBinHSMMStates

### Now create Poisson versions of the Mixture, DP-Mixture, HMM, HDP-HMM, HSMM, and HDP-HSMM
def _make_obs_distns(K, N, alpha_obs, beta_obs):
    obs_distns = []
    for k in range(K):
        obs_distns.append(PoissonVector(N, alpha_0=alpha_obs, beta_0=beta_obs))
    return obs_distns

class PoissonStatic(object):
    """
    Simple base class for Poisson neurons with homogeneous rates
    """
    def __init__(self, N):
        self.N = N
        self.data_list = []
        self.mask_list = []
        self.lmbda = np.zeros(N)

    def add_data(self, S, mask=None):
        assert S.ndim == 2 and S.shape[1] == self.N

        if mask is None:
            mask = np.ones_like(S, dtype=bool)
        else:
            assert mask.shape == S.shape and mask.dtype == bool

        self.data_list.append(S)
        self.mask_list.append(mask)

    def max_likelihood(self):
        S = 0.0
        T = 0.0
        for data, mask in zip(self.data_list, self.mask_list):
            S += (data * mask).sum(0)
            T += mask.sum(0)
        self.lmbda = S / float(T)

    def log_likelihood(self, S, mask=None):
        if mask is None:
            mask = np.ones_like(S, dtype=bool)
        else:
            assert mask.shape == S.shape and mask.dtype == bool

        lmbda = self.lmbda
        homog_ll = (-gammaln(S + 1) * mask).sum() \
                   - (lmbda * mask).sum() \
                   + (S * np.log(lmbda) * mask).sum()
        return homog_ll

class PoissonMixture(pybasicbayes.models.Mixture):

    _labels_class = PoissonLabels

    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonMixture, self).__init__(
            components=_make_obs_distns(K, N, alpha_obs, beta_obs), **kwargs)

        # TODO: Implement component hyperparameter resampling

    def resample_components(self, num_procs=None):
        if len(self.labels_list) > 0:
            ns, totss = map(sum, zip(*[s.obs_stats for s in self.labels_list]))

            for c, n, tots in zip(self.components, ns, totss):
                c.resample(n=n, tots=tots)
            self._clear_caches()

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
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0,
                 alpha_a_0=10.0, alpha_b_0=1.0,
                 init_state_concentration=1.0,
                 **kwargs):
        super(PoissonHMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs),
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)


class PoissonHDPHMM(_PoissonHMMMixin, pyhsmm.models.WeakLimitHDPHMM):
    def __init__(self, N, K_max,
                 alpha_obs=1.0, beta_obs=1.0,
                 alpha_a_0=10.0, alpha_b_0=1.0,
                 gamma_a_0=10.0, gamma_b_0=1.0,
                 init_state_concentration=1.0,
                 **kwargs):
        super(PoissonHDPHMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs),
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)

class PoissonDATruncHDPHMM(_PoissonHMMMixin, pyhsmm.models.DATruncHDPHMM):
    def __init__(self, N, K_max, alpha_obs=1.0, beta_obs=1.0, **kwargs):
        super(PoissonDATruncHDPHMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs), **kwargs)


class PoissonHSMM(_PoissonHSMMMixin, pyhsmm.models.HSMM):
    # TODO: Override PoissonHMMMixin to use HSMMStates
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0,
                 alpha_a_0=10.0, alpha_b_0=1.0,
                 init_state_concentration=1.0,
                 **kwargs):
        super(PoissonHSMM, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs),
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)


class PoissonHSMMPoissonDuration(_PoissonHSMMMixin, pyhsmm.models.HSMM):
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0,
                 alpha_dur=1.0, beta_dur=1.0, **kwargs):

        # Instantiate Poisson duration distributions
        duration_distns = [
            pyhsmm.distributions.PoissonDuration(
                alpha_0=alpha_dur, beta_0=beta_dur)
            for _ in range(K)]


        super(PoissonHSMMPoissonDuration, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs),
            dur_distns=duration_distns,
            **kwargs)


    def add_data(self,data,stateseq=None,trunc=30,
            right_censoring=True,left_censoring=False,**kwargs):
        return super(PoissonHSMMPoissonDuration, self).\
            add_data(data, stateseq=stateseq, trunc=trunc,
                     right_censoring=right_censoring,
                     left_censoring=left_censoring,
                     **kwargs)


class PoissonHSMMIntNegBinDuration(_PoissonIntNegBinHSMMMixin, pyhsmm.models.HSMMIntNegBin):
    def __init__(self, N, K, alpha_obs=1.0, beta_obs=1.0,
                 r_max=10, alpha_dur=10.0, beta_dur=1.0,
                 alpha_a_0=10.0, alpha_b_0=1.0,
                 gamma_a_0=10.0, gamma_b_0=1.0,
                 init_state_concentration=1.0,
                 **kwargs):

        # Instantiate Poisson duration distributions
        duration_distns = [
            pyhsmm.distributions.NegativeBinomialIntegerRDuration(
                r_discrete_distn=np.ones(r_max), alpha_0=alpha_dur, beta_0=beta_dur)
            for _ in range(K)]


        super(PoissonHSMMIntNegBinDuration, self).__init__(
            obs_distns=_make_obs_distns(K, N, alpha_obs, beta_obs),
            dur_distns=duration_distns,
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)

    def add_data(self,data,stateseq=None,trunc=30,
            right_censoring=True,left_censoring=False,**kwargs):
        return super(PoissonHSMMIntNegBinDuration, self).\
            add_data(data, stateseq=stateseq, trunc=trunc,
                     right_censoring=right_censoring,
                     left_censoring=left_censoring,
                     **kwargs)


class PoissonHDPHSMM(_PoissonHSMMMixin, pyhsmm.models.WeakLimitHDPHSMM):
    def __init__(self, N, K_max, alpha_obs=1.0, beta_obs=1.0,
                alpha_a_0=10.0, alpha_b_0=1.0,
                gamma_a_0=10.0, gamma_b_0=1.0,
                init_state_concentration=1.0,
                 **kwargs):
        super(PoissonHDPHSMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs),
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)


class PoissonIntNegBinHDPHSMM(_PoissonIntNegBinHSMMMixin, pyhsmm.models.WeakLimitHDPHSMM):
    def __init__(self, N, K_max, alpha_obs=1.0, beta_obs=1.0,
                 r_max=10, alpha_dur=10.0, beta_dur=1.0,
                 alpha_a_0=10.0, alpha_b_0=1.0,
                 gamma_a_0=10.0, gamma_b_0=1.0,
                 init_state_concentration=1.0,
                 **kwargs):

        # Instantiate Poisson duration distributions
        duration_distns = [
            pyhsmm.distributions.NegativeBinomialIntegerRDuration(
                r_discrete_distn=np.ones(r_max), alpha_0=alpha_dur, beta_0=beta_dur)
            for _ in range(K_max)]


        super(PoissonIntNegBinHDPHSMM, self).__init__(
            obs_distns=_make_obs_distns(K_max, N, alpha_obs, beta_obs),
            dur_distns=duration_distns,
            alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0,
            init_state_concentration=init_state_concentration,
            **kwargs)

    def add_data(self,data,stateseq=None,trunc=30,
        right_censoring=True,left_censoring=False,**kwargs):
        return super(PoissonIntNegBinHDPHSMM, self).\
            add_data(data, stateseq=stateseq, trunc=trunc,
                     right_censoring=right_censoring,
                     left_censoring=left_censoring,
                     **kwargs)

