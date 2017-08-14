# distutils: extra_compile_args = -O3 -w -ffast-math -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t
from cython cimport floating, integral
from cython.parallel import prange

def fast_likelihoods(
    floating[:,::1] gammalns,
    floating[:,::1] lmbdas,
    integral[:,::1] data,
    int32_t[:,::1] mask,
    floating[:,::1] likelihoods):
    cdef int T = data.shape[0]        # length of data
    cdef int D = data.shape[1]        # dimension of data
    cdef int N = likelihoods.shape[1] # number of states

    cdef int t, d, n

    cdef floating[:,::1] loglmbdas = np.log(lmbdas)

    with nogil:
        for t in prange(T):
            for d in range(D):
                if mask[t,d] > 0:
                    for n in range(N):
                        likelihoods[t,n] += \
                            -lmbdas[n,d] + data[t,d]*loglmbdas[n,d] - gammalns[t,d]

def fast_statistics(
        int32_t[::1] stateseq,
        integral[:,::1] data,
        int32_t[:,::1] mask,
        integral[::1] ns,
        integral[:,::1] tots):
    cdef int T = data.shape[0] # length of data
    cdef int D = data.shape[1] # dimension of data
    cdef int N = ns.shape[0]   # number of states

    cdef int t, d

    for t in range(T):
        ns[stateseq[t]] += 1
        for d in range(D):
            if mask[t,d] > 0:
                tots[stateseq[t],d] += data[t,d]
