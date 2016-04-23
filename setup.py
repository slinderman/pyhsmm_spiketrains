from distutils.core import setup
import numpy as np

from Cython.Build import cythonize

setup(
    name='pyhsmm_spiketrains',
    version='0.1',
    description='Bayesian inference for Poisson latent state space models',
    author='Scott Linderman',
    author_email='slinderman@seas.harvard.edu',
    url='http://www.github.com/slinderman/pyhsmm-spiketrains',
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
)
