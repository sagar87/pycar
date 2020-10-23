"""Main module."""
from numpyro.distributions import constraints
import numpyro.distributions as dist

import jax.numpy as jnp
from jax import lax
from jax.ops import index_add
import jax.random as random

from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)


class SparseCAR(dist.Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(
        self, loc, scale, D_sparse, W_sparse, eigenvals, alpha, tau, validate_args=None
    ):
        """Connects to the next available port.

        Args:
          loc: loc parameter of the normal RV
          scale: scale parameter of the normal RV
          D_sparse: N length vector containing the number of neighbours of each
            node in the adjacency matrix W
          W_sparse: (N x 2) matrix encoding the neighbour relationship as a
            graph edgeset
          eigenvals: Eigenvalues of D^(1/2)WD^(1/2)
          alpha: alpha parameter (0 < alpha < 1)
          tau: tau parameter

        Returns:
          A SparseCAR distribution.
        """
        self.loc, self.scale = promote_shapes(loc, scale)
        self.D_sparse = D_sparse
        self.W_sparse = W_sparse
        self.eigenvals = eigenvals
        self.alpha = alpha
        self.tau = tau
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(SparseCAR, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps * self.scale

    @validate_sample
    def log_prob(self, value):
        phi_D = value * self.D_sparse
        phi_W = jnp.zeros(value.shape[0])
        phi_W1 = index_add(
            phi_W,
            self.W_sparse[:, 0],
            phi_W[self.W_sparse[:, 0]] + value[self.W_sparse[:, 1]],
        )
        phi_W2 = index_add(
            phi_W1,
            self.W_sparse[:, 1],
            phi_W[self.W_sparse[:, 1]] + value[self.W_sparse[:, 0]],
        )
        ldet_terms = jnp.log1p(-self.alpha * self.eigenvals)

        return 0.5 * (
            value.shape[0] * jnp.log(self.tau)
            + jnp.sum(ldet_terms)
            - self.tau * (phi_D @ value - self.alpha * (phi_W2 @ value))
        )


class SparseICAR(dist.Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale, W_sparse, validate_args=None):
        """Connects to the next available port.

        Args:
          loc: loc parameter of the normal RV
          scale: scale parameter of the normal RV
          W_sparse: (N x 2) matrix encoding the neighbour relationship as a
            graph edgeset
        """
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.W_sparse = W_sparse
        super(SparseICAR, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps * self.scale

    @validate_sample
    def log_prob(self, value):
        phi = jnp.sum(
            jnp.power(value[self.W_sparse[:, 0]] - value[self.W_sparse[:, 1]], 2)
        )
        scale = 0.001 * value.shape[0]
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * scale)
        value_scaled = jnp.sum(value) / scale
        penalty = -0.5 * value_scaled ** 2 - normalize_term

        return -0.5 * phi + penalty
