
import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from .nnfe_object import NNFE
from .networks import MLP, DNN, ResNet, DenseNet
