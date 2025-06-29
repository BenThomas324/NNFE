
from pyfiglet import Figlet
import importlib.metadata

f = Figlet(font='starwars')
print(f.renderText('NNFE'))

__version__ = importlib.metadata.version(__package__)


from .control.natural import NNFE

__all__ = [
    "NNFE",
]
