
import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from .nnfe_object import NNFE

__all__ = [
    "NNFE",
]
