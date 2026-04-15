"""
NNFE — Neural Network Finite Elements.

Top-level package exposing the primary user-facing classes:

- :class:`~nnfe.nnfe_object.NNFE`: main solver object that couples the FE
  problem, ML model, sampler, plotter, and project management.
- :class:`~nnfe.networks.MLP`, :class:`~nnfe.networks.DNN`,
  :class:`~nnfe.networks.ResNet`, :class:`~nnfe.networks.DenseNet`:
  JAX/Equinox neural network architectures.
"""

import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from .nnfe_object import NNFE
from .networks import MLP, DNN, ResNet, DenseNet
