
import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from ._control import NNFE

__all__ = [
    "NNFE",
]
