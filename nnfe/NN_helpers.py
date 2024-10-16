
import equinox as eqx
import optax
import jax
import jax.numpy as np
import numpy as onp
import networks as networks
from plotting import *

def trunc_weight(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out, in_ = weight.shape
  stddev = 1e-6
  return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-1, upper=1)

def trunc_bias(bias: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out = bias.shape
  stddev = 1e-6
  return stddev * jax.random.truncated_normal(key, shape=(out), lower=-1, upper=1)

def init_linear_weight(model, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  get_biases = lambda m: [x.bias
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  biases = get_biases(model)
  new_weights = [trunc_weight(weight, subkey)
                 for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_biases = [trunc_bias(bias, subkey)
                 for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  new_model = eqx.tree_at(get_biases, new_model, new_biases)
  return new_model

def create_network(params, key):
    key = jax.random.PRNGKey(key)
    params["kwargs"]["key"] = key
    params["kwargs"]["activation"] = getattr(jax.nn, params["kwargs"]["activation"])
    model = getattr(networks, params["name"])(**params["kwargs"])

    # Add custom init
    if params["load_model"] is None:
        model = init_linear_weight(model, key)
    else:
        model = eqx.tree_deserialise_leaves(params["load_model"], model)

    return model


def create_optimizer(params, results_dir):

    if params["scheduler"]["toggle"]:
        boundaries = params["scheduler"]["boundaries"]
        schedulers = []
        for s in params["scheduler"]["list"]:
            p = params["scheduler"]["list"][s]
            sched = getattr(optax, p["name"])
            del p["name"]
            schedulers.append(sched(**p))

        scheduler = optax.join_schedules(schedulers, boundaries)

    else:
        scheduler = optax.constant_schedule(params["learning_rate"])

    params["options"]["learning_rate"] = scheduler
    optimizer = getattr(optax, params["name"])(**params["options"])

    plot_learning_rate(scheduler, params["epochs"], results_dir)
    return optimizer
