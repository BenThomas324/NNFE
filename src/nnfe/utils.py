
from pathlib import Path
import yaml
import jax.numpy as np

valid_samplers = ["uniform"]
valid_PDEs = [""]

def validate_sampler(sampler):
    if sampler not in valid_samplers:
        raise ValueError(f"Sampler '{sampler}' not recognized. \n"
                         f"Valid options are: {valid_samplers}")
    return sampler

# Tell PyYAML to treat Path objects exactly like normal strings when dumping
def path_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

# Register it for both standard and safe dumpers
yaml.add_multi_representer(Path, path_representer)
yaml.add_multi_representer(Path, path_representer, Dumper=yaml.SafeDumper)

# --- NumPy Handling ---
def numpy_int_representer(dumper, data):
    return dumper.represent_int(int(data))

def numpy_float_representer(dumper, data):
    return dumper.represent_float(float(data))

def numpy_array_representer(dumper, data):
    return dumper.represent_list(data.tolist())

# Register them with the SafeDumper
yaml.add_multi_representer(np.integer, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_multi_representer(np.floating, numpy_float_representer, Dumper=yaml.SafeDumper)
yaml.add_multi_representer(np.ndarray, numpy_array_representer, Dumper=yaml.SafeDumper)

def get_dict(d: dict, key: str) -> dict:
    """
    Safely retrieves a sub-dictionary. 
    If the key is missing OR the value is None, returns {}.
    """
    val = d.get(key)
    return val if val is not None else {}

def get_Path(d: dict, key: str) -> Path:
    """
    Safely retrieves a Path object from a dictionary. 
    If the key is missing OR the value is None, returns None.
    """
    val = d.get(key)
    return Path(val) if val is not None else None
