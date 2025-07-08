
"""
This file creates/loads the neural network, optimizer, and initializes the weights.
Can be used generically, not just for NNFE, pulls models from equinox and models.py
"""

import equinox as eqx
import optax
import jax
import jax.numpy as np
import equinox as eqx

import nnfe.networks as networks
from nnfe.plotting import *

class ML():
    def __init__(self, ml_params):
        
        self.network_params = ml_params["Network"]
        self.optimizer_params = ml_params["Optimizer"]
        self.network = self.create_network(self.network_params, ml_params["Key"])
        self.optimizer = self.create_optimizer(self.optimizer_params)
        self.opt_state = self.optimizer.init(eqx.filter(self.network, eqx.is_array))
        return

    def trunc_weight(self, weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        out, in_ = weight.shape
        stddev = 1e-5
        return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)

    def trunc_bias(self, bias: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        out = bias.shape
        stddev = 1e-5
        return stddev * jax.random.truncated_normal(key, shape=(out), lower=-1, upper=1)

    def init_linear_weight(self, model, key):
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

    def create_network(self, params, key):
        if key == "random":
            # Make random key, use random directory key as prev
            key = onp.random.randint(0, int(1e6))
            params["Key"] = key # Save the randomized key
            key = jax.random.PRNGKey(key)
        elif type(key) == int:
            key = jax.random.PRNGKey(key)
        else:
            raise ValueError("Key must be 'random' or an integer")
        
        params["kwargs"]["key"] = key
        params["kwargs"]["activation"] = getattr(jax.nn, params["kwargs"]["activation"])
        model = getattr(networks, params["name"])(**params["kwargs"])

        if params["load_model"] is None:
            model = self.init_linear_weight(model, key)
        else:
            model = eqx.tree_deserialise_leaves(params["load_model"], model)

        del params["kwargs"]["key"]
        params["kwargs"]["activation"] = params["kwargs"]["activation"].__name__
        model_num = jax.tree.leaves(eqx.filter(model, eqx.is_array))
        model_num = [len(np.ravel(a)) for a in model_num]
        print("Number of parameters: ", sum(model_num))
        return model

    def create_optimizer(self, params):

        if params["scheduler"]["toggle"]:
            boundaries = params["scheduler"]["boundaries"]
            schedulers = []
            for s in params["scheduler"]["list"]:
                p = params["scheduler"]["list"][s]
                sched = getattr(optax, p["name"])
                schedulers.append(sched(**p["kwargs"]))

            scheduler = optax.join_schedules(schedulers, boundaries)

        else:
            scheduler = optax.constant_schedule(params["learning_rate"])

        params["options"]["learning_rate"] = scheduler
        optimizer = getattr(optax, params["name"])(**params["options"])

        # plot_learning_rate(scheduler, params["epochs"], results_dir)
        return optimizer
