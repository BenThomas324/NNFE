
"""
This file creates/loads the neural network, optimizer, and initializes the weights.
Can be used generically, not just for NNFE, pulls models from equinox and models.py
"""

import equinox as eqx
import optax
import jax
import jax.numpy as np
import equinox as eqx
import yaml
from pathlib import Path
import jax.tree_util as jtu

import nnfe.networks as networks
from nnfe.plotter import *

class ML():
    def __init__(self, param_file, output_size=None, key=0):

        param_file = Path(param_file)
        with open(param_file) as f:
            ml_params = yaml.safe_load(f)

        if type(ml_params["Networks"]["key"]) == int:
            # Make random key, use random directory key as prev
            key = jax.random.PRNGKey(ml_params["Networks"]["key"])
        elif ml_params["Networks"]["key"] == "random":
            key = jax.random.PRNGKey(key)
        else:
            raise ValueError("Key must be 'random' or an integer")
        del ml_params["Networks"]["key"]

        self.network_params = ml_params["Networks"]
        self.optimizer_params = ml_params["Optimizer"]

        self.network, self.filter = self.create_network(self.network_params, output_size, key)
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

    def network_from_params(self, params, output_size, key):

        if params["kwargs"]["out_size"] == "dofs":
            params["kwargs"]["out_size"] = output_size

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

    def filtering(self, filter, model_ind):
        filter = eqx.tree_at(
            lambda tree: tree.models[model_ind],
            filter,
            replace=False
        )
        return filter

    def create_network(self, params, output_size, key=0):
        models = []
        filter_inds = []
        for i, p in enumerate(params):
            key, subkey = jax.random.split(key)
            models.append(self.network_from_params(params[p], output_size, subkey))
            if params[p].get("static", False):
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

    def create_optimizer(self, params):

        if params["scheduler"]["toggle"]:
            boundaries = params["scheduler"]["boundaries"]
            schedulers = []
            for s in params["scheduler"]["schedules"]:
                p = params["scheduler"]["schedules"][s]
                sched = getattr(optax, p["name"])
                schedulers.append(sched(**p["kwargs"]))

            scheduler = optax.join_schedules(schedulers, boundaries)

        else:
            scheduler = optax.constant_schedule(params["learning_rate"])

        params["options"]["learning_rate"] = scheduler
        optimizer = getattr(optax, params["name"])(**params["options"])

        return optimizer
