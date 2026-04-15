"""
Neural network architectures for NNFE.

All models are Equinox modules and are therefore JAX-compatible (JIT,
``vmap``, ``grad``, etc.).  The public API exported by :mod:`nnfe` is:

- :data:`MLP` — alias for :class:`equinox.nn.MLP`.
- :class:`DNN` — feed-forward network with variable hidden-layer widths.
- :class:`ResNet` — residual network with skip connections.
- :class:`DenseNet` — densely connected network where each layer receives
  the outputs of all previous layers.
- :class:`Sum_models` — combines several models by summing their outputs,
  used when ``MLConfig`` specifies multiple networks.
"""

import equinox as eqx
import jax
import jax.random as jrandom
from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray
from typing import (
    Literal,
    Optional,
    Union,
)
import jax.numpy as np
import jax.tree_util as jtu


identity = lambda x: x

MLP = eqx.nn.MLP

class Sum_models(eqx.Module, strict=True):
    """Combined model for NNFE, which can be used to combine multiple models into one.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    models: tuple[eqx.Module, ...]
    out_size: int

    def __init__(self, models: tuple[eqx.Module, ...]):
        self.models = models
        self.out_size = self.models[-1].out_size
    
        for i in range(len(models) - 1):
            assert models[i].out_size == self.out_size

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        # Why can't I set as self.out_size?
        y = np.zeros((self.models[-1].out_size,))
        for model in self.models:
            y += model(x[:model.in_size])
        return y

class DNN(eqx.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    hidden_layers: tuple[Union[int, Literal["scalar"]], ...]

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        hidden_layers: tuple[Union[int, Literal["scalar"]], ...],
        activation: Callable = jax.nn.relu,
        final_activation: Callable = identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = jax.numpy.float64 if dtype is None else dtype
        keys = jrandom.split(key, len(hidden_layers) + 1)
        layers = []
        if len(hidden_layers) == 0:
            layers.append(
                eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                eqx.nn.Linear(in_size, hidden_layers[0], use_bias, dtype=dtype, key=keys[0])
            )
            for i in range(len(hidden_layers) - 1):
                layers.append(
                    eqx.nn.Linear(
                        hidden_layers[i], hidden_layers[i+1], use_bias, dtype=dtype, key=keys[i + 1]
                    )
                )
            layers.append(
                eqx.nn.Linear(hidden_layers[-1], out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layers = hidden_layers
        self.activation = eqx.filter_vmap(activation)
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("nnfe.DNN")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x

class ResNet(eqx.Module, strict=True):
    """Residual network (ResNet) with element-wise skip connections.

    Each hidden layer adds its output to the previous hidden-layer output
    (skip connection), which helps gradient flow and allows training of deeper
    networks.  All hidden layers share the same ``width_size``.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = jax.numpy.float64 if dtype is None else dtype
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                eqx.nn.Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    eqx.nn.Linear(
                        width_size, width_size, use_bias, dtype=dtype, key=keys[i + 1]
                    )
                )
            layers.append(
                eqx.nn.Linear(width_size, out_size, use_final_bias, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("nnfe.ResNet")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for i, layer in enumerate(self.layers[:-1]):
            z = layer(x)
            layer_activation = jtu.tree_map(
                lambda z: z[i] if eqx.is_array(z) else z, self.activation
            )
            if i == 0:
                x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, z)
            else:
                x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, z) + x
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x
    
class DenseNet(eqx.Module, strict=True):
    """Densely connected network (DenseNet).

    Inspired by DenseNet (Huang et al. 2017): every layer receives the
    feature maps of all preceding layers as additional input.  This maximises
    feature reuse and provides strong gradient flow throughout the network.

    !!! faq

        If you get a TypeError saying an object is not a valid JAX type, see the
            [FAQ](https://docs.kidger.site/equinox/faq/)."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = jax.numpy.float64 if dtype is None else dtype
        num_keys = int((depth + 1) * (depth + 2) / 2)
        keys = jrandom.split(key, num_keys)
        layers = []

        def get_layers(iter, depth, in_size, width_size, out_size, keys):
            inner_layer = []
            for i in range(depth - iter):
                if i != depth - iter - 1:
                    inner_layer.append(eqx.nn.Linear(in_size, width_size, use_bias, dtype=dtype, key=keys[i]))
                else:
                    inner_layer.append(eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[i]))
            return inner_layer

        if depth == 0:
            layers.append(
                eqx.nn.Linear(in_size, out_size, use_final_bias, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(get_layers(0, depth, in_size, width_size, out_size, keys[:depth-1]))
            v0 = 0
            for i in range(depth-1):
                v1 = int(depth - i * (depth - (i + 1)) / 2)
                v0 = v0 + v1
                layers.append(
                    get_layers(i+1, depth, width_size, width_size, out_size, keys[v0:v0+v1])
                )

        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(lambda: activation, axis_size=width_size)()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("nnfe.DenseNet")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        xs = [l(x) for l in self.layers[0]]
        for j, layer in enumerate(self.layers[1:-1]):
            xs = [l(xs[0]) + xs[i+1] for i, l in enumerate(layer)]
            xs[0] = self.activation(xs[0])
        x = self.layers[-1][0](xs[0]) + xs[1]
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x