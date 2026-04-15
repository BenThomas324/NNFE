"""
Shared utilities for NNFE.

Contains:
- Input validation helpers (samplers, PDEs).
- PyYAML multi-representers so that ``Path`` objects and JAX/NumPy scalar and
  array types round-trip cleanly through ``yaml.safe_dump``.
- Convenience accessors for pulling typed values out of raw config dicts.
"""

from pathlib import Path
import yaml
import jax.numpy as np

valid_samplers = ["uniform"]
valid_PDEs = [""]


def validate_sampler(sampler: str) -> str:
    """Validate that *sampler* is a recognised sampler name.

    Args:
        sampler: Name of the sampler to validate.

    Returns:
        The same *sampler* string if valid.

    Raises:
        ValueError: If *sampler* is not in ``valid_samplers``.
    """
    if sampler not in valid_samplers:
        raise ValueError(f"Sampler '{sampler}' not recognized. \n"
                         f"Valid options are: {valid_samplers}")
    return sampler


# --- PyYAML representers -------------------------------------------------------

def path_representer(dumper: yaml.Dumper, data: Path) -> yaml.ScalarNode:
    """Represent a :class:`pathlib.Path` as a plain YAML string."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))


# Register for both standard and safe dumpers so Path objects survive round-trips.
yaml.add_multi_representer(Path, path_representer)
yaml.add_multi_representer(Path, path_representer, Dumper=yaml.SafeDumper)


# --- NumPy / JAX array representers -------------------------------------------

def numpy_int_representer(dumper: yaml.Dumper, data) -> yaml.ScalarNode:
    """Represent a JAX/NumPy integer scalar as a plain YAML int."""
    return dumper.represent_int(int(data))


def numpy_float_representer(dumper: yaml.Dumper, data) -> yaml.ScalarNode:
    """Represent a JAX/NumPy float scalar as a plain YAML float."""
    return dumper.represent_float(float(data))


def numpy_array_representer(dumper: yaml.Dumper, data) -> yaml.SequenceNode:
    """Represent a JAX/NumPy array as a YAML sequence."""
    return dumper.represent_list(data.tolist())


# Register with SafeDumper so config dicts containing arrays serialise correctly.
yaml.add_multi_representer(np.integer, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_multi_representer(np.floating, numpy_float_representer, Dumper=yaml.SafeDumper)
yaml.add_multi_representer(np.ndarray, numpy_array_representer, Dumper=yaml.SafeDumper)


# --- Config dict helpers -------------------------------------------------------

def get_dict(d: dict, key: str) -> dict:
    """Safely retrieve a sub-dictionary from *d*.

    Args:
        d: Source dictionary.
        key: Key to look up.

    Returns:
        The value at *d[key]* if it is a non-``None`` dict, otherwise ``{}``.
    """
    val = d.get(key)
    return val if val is not None else {}


def get_Path(d: dict, key: str) -> Path:
    """Safely retrieve a :class:`pathlib.Path` from *d*.

    Args:
        d: Source dictionary.
        key: Key to look up.

    Returns:
        ``Path(d[key])`` if the value exists and is not ``None``, else ``None``.
    """
    val = d.get(key)
    return Path(val) if val is not None else None
