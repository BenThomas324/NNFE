"""
This file is intended to create the data used for training NNFE.
Since we don't require data, we may want to play with various samplings
of the grid to be trained. It may also be used for adaptive sampling
in the future.
"""

import numpy as onp

def generate_data_grid(ranges):

    vals = []
    for r in ranges:
        vals.append(onp.linspace(r[0], r[1], r[2]))

    grid = onp.meshgrid(*vals)

    return onp.vstack([g.flatten() for g in grid]).T

