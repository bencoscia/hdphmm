"""
hdphmm
Determine latent states in multi-dimensional time series using the hierarchical Dirichlet process hidden Markov model
"""

# Add imports here
from .hdphmm import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
