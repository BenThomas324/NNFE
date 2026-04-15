"""
Machine-learning manager for NNFE.

:class:`MLManager` is the single entry point for constructing and owning the
neural network, optimizer, and optimizer state.  It is designed to be used
independently of the rest of the NNFE stack (i.e. it does not import anything
FE-specific) so it can be reused in other JAX/Equinox projects.

Networks are sourced from :mod:`nnfe.networks` and optimizers from
:mod:`optax`.  Weight initialisation uses truncated-normal distributions with
a very small standard deviation (``1e-6``) so that the network starts close to
zero — a sensible initial condition for residual-minimisation problems.
"""

from dataclasses import asdict

import equinox as eqx
from .ml_config import MLConfig
import optax
import jax
import jax.numpy as np
import equinox as eqx
import yaml
from pathlib import Path
import jax.tree_util as jtu
import copy

import nnfe.networks as networks
from .plotter import *

class MLManager():
    """Manages the neural network, optimizer, and training state for NNFE.

    On construction this class:

    1. Builds or loads the network(s) from :mod:`nnfe.networks`.
    2. Initialises (or loads) the weights.
    3. Builds the Optax optimizer and its learning-rate schedule.
    4. Initialises the optimizer state.

    Attributes:
        config: The :class:`~nnfe.ml_config.MLConfig` used to build this
            manager.
        network: The Equinox model (single or combined via
            :class:`~nnfe.networks.Sum_models`).
        filter: A pytree mirror of *network* whose leaves are ``True`` for
            trainable parameters and ``False`` for frozen ones.
        optimizer: The Optax optimizer.
        lr_scheduler: The Optax learning-rate schedule callable.
        opt_state: Current Optax optimizer state.
        epochs: Total training epochs (cast to ``int``).
        batch_size: Batch fraction (cast to ``int`` after scaling by the
            sampler size in :class:`~nnfe.nnfe_object.NNFE`).
    """

    def __init__(self,
                 MLConfig,
                **kwargs):

        # Store config object
        self.config = MLConfig

        # Generate key
        if type(MLConfig.rng_key) == int:
            key = jax.random.PRNGKey(MLConfig.rng_key)
        else:
            key = jax.random.PRNGKey(0)

        # Parse DoF kwarg if needed
        temp = ["dofs" == net_kwargs.kwargs["out_size"] for net_key, net_kwargs in MLConfig.networks.items()]
        if any(temp):
            out_size = kwargs["out_size"]

        self.network, self.filter = self.create_network(MLConfig.networks, out_size, key)
        self.optimizer, self.lr_scheduler = self.create_optimizer(MLConfig.optimizer)
        self.opt_state = self.optimizer.init(eqx.filter(self.network, eqx.is_array))

        self.epochs = int(MLConfig.epochs)
        self.batch_size = int(MLConfig.batch_size)

        return
    
    @classmethod
    def from_config(cls, MLConfig, **kwargs):        
        return cls(MLConfig=MLConfig, **kwargs)
    
    @classmethod
    def from_yaml(cls, param_file, **kwargs):
        param_file = Path(param_file)
        return cls.from_config(MLConfig=MLConfig.from_yaml(param_file), **kwargs)

    def trunc_weight(self, weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """Return a new weight matrix sampled from a truncated normal.

        Values are drawn from ``TruncNormal(0, 1e-6)`` clipped to ``[-1, 1]``,
        giving a near-zero initialisation suitable for residual minimisation.

        Args:
            weight: Existing weight array (used only for shape).
            key: JAX PRNG key.

        Returns:
            New weight array with the same shape as *weight*.
        """
        out, in_ = weight.shape
        stddev = 1e-6
        return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)

    def trunc_bias(self, bias: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """Return a new bias vector sampled from a truncated normal.

        Args:
            bias: Existing bias array (used only for shape).
            key: JAX PRNG key.

        Returns:
            New bias array with the same shape as *bias*.
        """
        out = bias.shape
        stddev = 1e-6
        return stddev * jax.random.truncated_normal(key, shape=(out), lower=-1, upper=1)

    def init_linear_weight(self, model, key):
        """Re-initialise all ``eqx.nn.Linear`` weights and biases in *model*.

        Replaces every weight matrix and bias vector in *model* with values
        drawn from :meth:`trunc_weight` / :meth:`trunc_bias`.  If the model
        has no biases the bias replacement is silently skipped.

        Args:
            model: An Equinox module whose ``Linear`` layers should be
                re-initialised.
            key: JAX PRNG key used to generate replacement values.

        Returns:
            A new Equinox module with re-initialised weights (and biases if
            present).
        """
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        try:
            get_biases = lambda m: [x.bias
                                    for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                    if is_linear(x)]
            biases = get_biases(model)
            new_biases = [self.trunc_bias(bias, subkey)
                        for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
            model = eqx.tree_at(get_biases, model, new_biases)
        except:
            pass

        weights = get_weights(model)
        new_weights = [self.trunc_weight(weight, subkey)
                        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        model = eqx.tree_at(get_weights, model, new_weights)
        return model

    def load_network(self, network, path):
        """Load serialised Equinox leaves from *path* into *network*.

        Attempts a direct deserialisation first.  If the on-disk weights are
        stored in a different floating-point dtype (e.g. ``float32`` vs
        ``float64``), the model is cast to ``float32`` before retrying.

        Args:
            network: An Equinox module whose structure matches the serialised
                file.
            path: Path to the ``.eqx`` file produced by
                :func:`equinox.tree_serialise_leaves`.

        Returns:
            The same module with its leaves replaced by the loaded weights.
        """
        try:
            network = eqx.tree_deserialise_leaves(path, network)
        except RuntimeError:
            # Define a function to cast an array to float32
            def to_float32(x):
                if eqx.is_array(x):
                    return x.astype(np.float32)
                return x

            # Apply the function to the model
            network_float32 = jax.tree.map(to_float32, network)
            network = eqx.tree_deserialise_leaves(path, network_float32)

        return network

    def network_from_config(self,
                            NetworkConfig,
                            **kwargs):
        """Instantiate a single network from a :class:`~nnfe.ml_config.NetworkConfig`.

        Resolves the ``"dofs"`` placeholder for ``out_size``, injects the
        activation function from :mod:`jax.nn`, constructs the network, and
        either initialises or loads its weights.  Prints the total parameter
        count.

        Args:
            NetworkConfig: A :class:`~nnfe.ml_config.NetworkConfig` instance.
            **kwargs: Must include ``out_size`` (int) and ``key``
                (:class:`jax.random.PRNGKey`).

        Returns:
            An initialised Equinox module.
        """

        network_kwargs = copy.deepcopy(NetworkConfig.kwargs)
        if network_kwargs["out_size"] == "dofs":
            network_kwargs["out_size"] = kwargs["out_size"]
        network_kwargs["key"] = kwargs["key"]

        network_kwargs["activation"] = getattr(jax.nn, network_kwargs["activation"])
        model = getattr(networks, NetworkConfig.name)(**network_kwargs)

        if NetworkConfig.load_model is None:
            model = self.init_linear_weight(model, kwargs["key"])
        else:
            model = self.load_network(model, NetworkConfig.load_model)

        del network_kwargs["key"]
        network_kwargs["activation"] = network_kwargs["activation"].__name__
        model_num = jax.tree.leaves(eqx.filter(model, eqx.is_array))
        model_num = [len(np.ravel(a)) for a in model_num]
        print("Number of parameters: ", sum(model_num))
        return model

    def filtering(self, filter, model_ind):
        """Mark the sub-model at index *model_ind* as non-trainable in *filter*.

        Sets the corresponding subtree of the filter pytree to ``False`` so
        that Equinox excludes it from gradient computation.

        Args:
            filter: A pytree (mirroring ``network``) of boolean leaves.
            model_ind: Index into ``network.models`` to freeze.

        Returns:
            Updated filter pytree.
        """
        filter = eqx.tree_at(
            lambda tree: (tree.models[model_ind]),
            filter,
            replace=(False)
        )
        return filter

    def create_network(self, params, out_size, key=0):
        """Build the full network (or combined network) from config.

        Iterates over all network configs, instantiates each via
        :meth:`network_from_config`, and combines them with
        :class:`~nnfe.networks.Sum_models` if more than one is present.
        Also builds the trainability filter, freezing any networks marked
        ``static=True``.

        Args:
            params: Ordered dict of :class:`~nnfe.ml_config.NetworkConfig`
                objects (from ``MLConfig.networks``).
            out_size: Output dimension (number of DoFs).
            key: JAX PRNG key.

        Returns:
            Tuple of ``(network, filter)`` where *filter* is a pytree of
            booleans indicating trainable parameters.
        """
        models = []
        filter_inds = []
        for i, net_vals in enumerate(params.values()):
            key, subkey = jax.random.split(key)
            models.append(self.network_from_config(net_vals, out_size=out_size, key=subkey))
            if net_vals.static:
                filter_inds.append(i)
            else:
                filter_inds.append(False)

        if len(models) == 1:
            model = models[0]
        else:
            model = networks.Sum_models(models)

        filter = jtu.tree_map(lambda _: True, model)
        for f in filter_inds:
            if type(f) == int:
                filter = self.filtering(filter, f)

        return model, filter

    def create_optimizer(self, OptimizerConfig):
        """Build the Optax optimizer and learning-rate schedule.

        Constructs a piecewise :func:`optax.join_schedules` from the
        ``scheduler`` sub-config and injects it as the ``learning_rate``
        argument of the optimizer.

        Args:
            OptimizerConfig: An :class:`~nnfe.ml_config.OptimizerConfig`
                instance.

        Returns:
            Tuple of ``(optimizer, scheduler)`` where *scheduler* is the
            callable learning-rate schedule.
        """

        if OptimizerConfig.lr_scheduler:
            schedule_params = copy.deepcopy(OptimizerConfig.scheduler)
            boundaries = schedule_params["boundaries"]
            schedulers = []
            for s in schedule_params["schedules"]:
                p = schedule_params["schedules"][s]
                sched = getattr(optax, p["name"])
                schedulers.append(sched(**p["kwargs"]))

            scheduler = optax.join_schedules(schedulers, boundaries)

        optimizer_kwargs = copy.deepcopy(OptimizerConfig.optimizer_kwargs)
        optimizer_kwargs["learning_rate"] = scheduler
        optimizer = getattr(optax, OptimizerConfig.name)(**optimizer_kwargs)

        return optimizer, scheduler

    def dump_config(self, save_dir: Path, filename: str):
        """Dumps this specific manager's configuration to a YAML file."""
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename
        
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self.config), f, sort_keys=False)