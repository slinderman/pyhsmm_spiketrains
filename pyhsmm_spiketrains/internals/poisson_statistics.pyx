# distutils: extra_compile_args = -O3 -w -ffast-math -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

from scipy.special import gammaln

from libc.stdint cimport int32_t
from cython cimport floating, integral
from cython.parallel import prange

def fast_likelihoods(
    floating[:,::1] gammalns, floating[:,::1] lmbdas,
    integral[:,::1] data, floating[:,::1] likelihoods):
    cdef int T = data.shape[0]        # length of data
    cdef int D = data.shape[1]        # dimension of data
    cdef int N = likelihoods.shape[1] # number of states

    cdef int t, d, n

    cdef floating[:,::1] loglmbdas = np.log(lmbdas)

    with nogil:
        for t in prange(T):
            for d in range(D):
                for n in range(N):
                    likelihoods[t,n] += \
                        -lmbdas[n,d] + data[t,d]*loglmbdas[n,d] - gammalns[t,d]

def fast_statistics(
        int32_t[::1] stateseq, integral[:,::1] data,
        integral[::1] ns, integral[:,::1] tots):
    cdef int T = data.shape[0] # length of data
    cdef int D = data.shape[1] # dimension of data
    cdef int N = ns.shape[0]   # number of states

    cdef int t, d

    for t in range(T):
        ns[stateseq[t]] += 1
        for d in range(D):
            tots[stateseq[t],d] += data[t,d]


# def fast_log_marginal_likelihood(floating[:,::1] gammalns,
#                                  floating alpha_0, floating alpha_post,
#                                  floating beta_0,  floating beta_post):
#
#     cdef floating lml = 0
#     lml += (gammaln(alpha_post) - alpha_post * np.log(beta_post))
#     lml -= (gammaln(alpha_0)    - alpha_0    * np.log(beta_0))
#     lml += - np.sum(gammalns)
#
# def fast_crp_score(
#         int t, int32_t[::1] z, integral[:,::1] data,
#
#     ):
#