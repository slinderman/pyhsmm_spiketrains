from pybasicbayes.abstractions import Collapsed
from pybasicbayes.distributions import ProductDistribution, Poisson

from numpy import arange, array, exp, log, isscalar
from scipy.special import gammaln, psi
from hips.inference.hmc import hmc

import numpy as np

# Make a product distribution that implements collapsed gibbs sampling
class PoissonVector(ProductDistribution, Collapsed):
    def __init__(self, N, alpha_0=1, beta_0=1):
        """
        Initialize with C cells with specified gamma hypers
        :param N:
        :return:
        """
        self.N = N
        if isscalar(alpha_0):
            alpha_0 = alpha_0*np.ones(N)
        elif isinstance(alpha_0, np.ndarray):
            assert len(alpha_0) == N
            alpha_0 = alpha_0

        if isscalar(beta_0):
            beta_0 = beta_0*np.ones(N)
        elif isinstance(beta_0, np.ndarray):
            assert len(beta_0) == N
            beta_0 = beta_0

        self.poissons = []
        for n in range(N):
            self.poissons.append(Poisson(alpha_0=alpha_0[n],
                                         beta_0=beta_0[n]))

        super(PoissonVector, self).__init__(self.poissons)

    def log_likelihood(self,x):
        return np.ravel(
            sum(distn.log_likelihood(x[...,sl])
            for distn,sl in zip(self._distns,self._slices)))

    @property
    def lmbdas(self):
        return np.array([p.lmbda for p in self.poissons])

    @property
    def hypers(self):
        alpha_0 = np.array([p.alpha_0 for p in self.poissons])
        beta_0 = np.array([p.beta_0 for p in self.poissons])
        return alpha_0, beta_0

    @hypers.setter
    def hypers(self, (alpha_0, beta_0)):
        assert len(alpha_0) == self.N
        assert len(beta_0) == self.N

        for n,p in enumerate(self.poissons):
            p.alpha_0 = alpha_0[n]
            p.beta_0 = beta_0[n]

    # Override the collapsed methods
    def log_marginal_likelihood(self,x):
        if isinstance(x, np.ndarray):
            return sum(distn.log_marginal_likelihood(x[...,sl])
                    for distn,sl in zip(self._distns,self._slices)).squeeze()
        elif isinstance(x, list):
            assert all(map(lambda xi: isinstance(xi, np.ndarray), x))
            lml = 0
            for distn,sl in zip(self._distns, self._slices):
                sliced_data = [xs[...,sl] for xs in x]
                lml += distn.log_marginal_likelihood(sliced_data)
            return lml

    def resample(self,data=[],n=None,tots=None):
        # Resample rates
        if None not in (n,tots):
            for p, tot in zip(self.poissons,tots):
                p.resample(stats=(n,tot))
        else:
            super(PoissonVector,self).resample(data=data)


