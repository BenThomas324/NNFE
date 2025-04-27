"""
This file is intended to create the data used for training NNFE.
Since we don't require data, we may want to play with various samplings
of the grid to be trained. It may also be used for adaptive sampling
in the future.
"""

import numpy as onp

class Sampler():

    def __init__(self, params):

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

        return

    def uniform(self, mins=[0.], maxes=[1.], samples=[5], between_training=False):

        assert len(mins) == len(maxes) == len(samples), "Mins, maxes, and samples must be the same length"
        n = len(samples)
        mins = onp.array(mins)
        maxes = onp.array(maxes)
        samples = onp.array(samples)

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



