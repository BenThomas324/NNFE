"""
This file is intended to create the data used for training NNFE.
Since we don't require data, we may want to play with various samplings
of the grid to be trained. It may also be used for adaptive sampling
in the future.
"""

import numpy as onp
import jax.numpy as np
import jax

class Sampler():

    def __init__(self, params, rng_key=0):

        if not hasattr(self, params["training"]["name"]):
            raise ValueError("Sampler not found: ", params["training"]["name"])
        else:
            self.training_sampler = getattr(self, params["training"]["name"])

        if not hasattr(self, params["testing"]["name"]):
            raise ValueError("Sampler not found: ", params["testing"]["name"])
        else:
            self.testing_sampler = getattr(self, params["testing"]["name"])

        self.X = self.training_sampler(**params["training"]["kwargs"])
        self.Y = self.testing_sampler(**params["testing"]["kwargs"])

        batch_size = params.get("batch_size", 1)
        if batch_size <= 1.:
            self.batch_size = int(batch_size * self.X.shape[0])
        else: self.batch_size = batch_size

        self.rng_key = jax.random.key(rng_key)
        self.draw_batch = jax.jit(self.draw_batch)
        return

    def safe_eval(self, expr):
        if type(expr) is str:
            return eval(expr)
        elif type(expr) in [float, int]:
            return expr
        else:
            raise ValueError("Expression must be a valid string, float, or int")

    def uniform(self, mins=[0.], maxes=[1.], samples=[5], between_training=False):

        assert len(mins) == len(maxes) == len(samples), "Mins, maxes, and samples must be the same length"
        n = len(samples)
        mins = onp.array([self.safe_eval(m) for m in mins])
        maxes = onp.array([self.safe_eval(m) for m in maxes])
        samples = onp.array([self.safe_eval(s) for s in samples])

        if between_training:
            samples -= 1
            diffs = (maxes - mins)/(2*samples)
            mins += diffs
            maxes -= diffs

        vals = []
        for i in range(n):
            vals.append(onp.linspace(mins[i], maxes[i], samples[i]))

        grid = onp.meshgrid(*vals)

        return onp.vstack([g.flatten() for g in grid]).T

    def draw_batch(self, rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        batch = jax.random.choice(subkey, self.X, (self.batch_size,), replace=False)
        return rng_key, batch