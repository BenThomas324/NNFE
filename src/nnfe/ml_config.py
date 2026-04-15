"""
Frozen dataclasses that describe the machine-learning configuration for NNFE.

Hierarchy::

    MLConfig
    ├── networks: dict[str, NetworkConfig]
    └── optimizer: OptimizerConfig
"""

from dataclasses import dataclass
import yaml
from .utils import get_dict, get_Path


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for a single neural network.

    Attributes:
        name: Name of the network class in :mod:`nnfe.networks`
            (e.g. ``"DNN"``, ``"ResNet"``).
        kwargs: Keyword arguments forwarded to the network constructor.
            The special value ``"dofs"`` for ``out_size`` is resolved at
            runtime by :class:`~nnfe.ml.MLManager` to the problem's total DoF
            count.
        load_model: Optional path to a serialised Equinox model file
            (``.eqx``).  When set, weights are loaded instead of randomly
            initialised.
        static: If ``True``, the network's parameters are frozen during
            training (excluded from gradient updates).
    """

    name: str
    kwargs: dict
    load_model: str = None
    static: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "NetworkConfig":
        """Construct a :class:`NetworkConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``name``, ``kwargs``, and optionally
                ``load_model`` and ``static``.

        Returns:
            A new :class:`NetworkConfig` instance.
        """
        name = params["name"]
        kwargs = params["kwargs"]
        load_model = params.get("load_model", None)
        static = params.get("static", False)

        return cls(name=name,
                   kwargs=kwargs,
                   load_model=load_model,
                   static=static)


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Optax optimizer and its learning-rate schedule.

    Attributes:
        name: Name of the Optax optimizer (e.g. ``"adam"``).
        lr_scheduler: Whether to use a learning-rate schedule.  When
            ``True``, ``scheduler`` must be supplied.
        optimizer_kwargs: Extra keyword arguments forwarded to the Optax
            optimizer (excluding ``learning_rate``, which is injected from
            the scheduler).
        scheduler: Nested dict describing a
            :func:`optax.join_schedules` setup.  Expected structure::

                boundaries: [<step>, ...]
                schedules:
                  s0:
                    name: <optax schedule name>
                    kwargs: {...}
                  ...
    """

    name: str
    lr_scheduler: bool = True
    optimizer_kwargs: dict = None
    scheduler: dict = None

    @classmethod
    def from_dict(cls, params: dict) -> "OptimizerConfig":
        """Construct an :class:`OptimizerConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``name`` and optionally ``lr_scheduler``,
                ``optimizer_kwargs``, and ``scheduler``.

        Returns:
            A new :class:`OptimizerConfig` instance.
        """
        name = params["name"]
        lr_scheduler = params.get("lr_scheduler", True)
        optimizer_kwargs = params.get("optimizer_kwargs", None)
        scheduler = get_dict(params, "scheduler")
        return cls(name=name,
                   lr_scheduler=lr_scheduler,
                   optimizer_kwargs=optimizer_kwargs,
                   scheduler=scheduler)


@dataclass(frozen=True)
class MLConfig:
    """Top-level machine-learning configuration consumed by
    :class:`~nnfe.ml.MLManager`.

    Attributes:
        networks: Ordered mapping of network names to their
            :class:`NetworkConfig` objects.  When more than one network is
            present they are combined additively via
            :class:`~nnfe.networks.Sum_models`.
        optimizer: Optimizer and learning-rate schedule configuration.
        epochs: Total number of training iterations.
        batch_size: Fraction of the training set to sample per step
            (value in ``(0, 1]``).
        rng_key: Integer seed for JAX PRNG initialisation.  Defaults to
            ``0``; may also be set to a project-level key at runtime.
    """

    networks: dict[str, NetworkConfig]
    optimizer: OptimizerConfig
    epochs: int
    batch_size: int
    rng_key: int | str = 0

    @classmethod
    def from_dict(cls, params: dict) -> "MLConfig":
        """Construct an :class:`MLConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``networks``, ``optimizer``, ``epochs``,
                ``batch_size``, and optionally ``rng_key``.

        Returns:
            A new :class:`MLConfig` instance.
        """
        networks = {net_key: NetworkConfig.from_dict(net_params)
                    for net_key, net_params in params["networks"].items()}
        optimizer = OptimizerConfig.from_dict(params["optimizer"])
        rng_key = params.get("rng_key", 0)
        if rng_key is None:
            rng_key = 0

        return cls(networks=networks,
                   optimizer=optimizer,
                   epochs=params["epochs"],
                   batch_size=params["batch_size"],
                   rng_key=rng_key)

    @classmethod
    def from_yaml(cls, path) -> "MLConfig":
        """Load an :class:`MLConfig` from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A new :class:`MLConfig` instance.
        """
        with open(path) as f:
            params = yaml.safe_load(f)

        return cls.from_dict(params)