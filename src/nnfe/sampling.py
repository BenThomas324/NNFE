"""
Training and testing point samplers for NNFE.

NNFE is a data-free method: no ground-truth solution data is required.
Instead, control-variable samples (i.e. the FE parameter space) are drawn
from a prescribed distribution and the residual is minimised over that
distribution.

:class:`Sampler` currently supports uniform grid sampling.  The design is
extensible — new samplers can be added as methods and registered in
:data:`~nnfe.utils.valid_samplers`.

Possible future work includes adaptive sampling strategies that concentrate
points in regions of high residual.
"""

import numpy as onp
import jax.numpy as np
import jax
from .utils import validate_sampler

class Sampler():
    """Manages the control-variable point sets used for training and testing.

    On construction the full training set (``X``) and test set (``Y``) are
    generated and stored as NumPy arrays.  During training, mini-batches are
    drawn from ``X`` without replacement via :meth:`draw_batch`.

    Attributes:
        config: :class:`~nnfe.nnfe_config.SamplerConfig` used to build this
            sampler.
        training_sampler: Bound method that generates training samples.
        testing_sampler: Bound method that generates test samples.
        X: Full training point set, shape ``(N_train, n_ctrl_vars)``.
        Y: Full test point set, shape ``(N_test, n_ctrl_vars)``.
    """

    def __init__(self, config, rng_key=0):
        """Initialise and pre-generate both point sets.

        Args:
            config: :class:`~nnfe.nnfe_config.SamplerConfig` instance.
            rng_key: Integer seed (currently unused; JAX-based batching is
                seeded at draw time).
        """
        self.config = config

        self.training_sampler = getattr(self, validate_sampler(config.training_sampler))
        self.testing_sampler = getattr(self, validate_sampler(config.testing_sampler))

        self.X = self.training_sampler(**config.training_kwargs)
        self.Y = self.testing_sampler(**config.testing_kwargs)
        return

    def safe_eval(self, expr):
        """Return *expr* as a numeric value, rejecting arbitrary strings.

        Args:
            expr: A ``float`` or ``int`` value.

        Returns:
            The input unchanged.

        Raises:
            ValueError: If *expr* is not a ``float`` or ``int``.
        """
        if type(expr) in [float, int]:
            return expr
        else:
            raise ValueError("Expression must be a valid string, float, or int")

    def uniform(self, mins=[0.], maxes=[1.], samples=[5], between_training=False):
        """Generate a uniform Cartesian grid of control-variable samples.

        For each dimension, *samples* equally-spaced points are generated
        between *mins* and *maxes* using :func:`numpy.linspace`, then the
        full tensor-product grid is assembled via :func:`numpy.meshgrid`.

        Args:
            mins: Lower bounds for each dimension.
            maxes: Upper bounds for each dimension.
            samples: Number of points per dimension.
            between_training: If ``True``, shift the grid so points fall
                *between* training points (half-step offset), reducing the
                sample count by one per dimension.

        Returns:
            NumPy array of shape ``(prod(samples), n_dims)`` containing all
            grid points.

        Raises:
            AssertionError: If *mins*, *maxes*, and *samples* have different
                lengths.
        """
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

    def draw_batch(self, rng_key, batch_size):
        """Randomly sample a mini-batch from the training set without replacement.

        Args:
            rng_key: JAX PRNG key.  A new key is split off internally so the
                caller's key is advanced.
            batch_size: Number of samples to draw.

        Returns:
            Tuple of ``(updated_rng_key, batch)`` where *batch* has shape
            ``(batch_size, n_ctrl_vars)``.
        """
        rng_key, subkey = jax.random.split(rng_key)
        batch = jax.random.choice(subkey, self.X, (batch_size,), replace=False)
        return rng_key, batch