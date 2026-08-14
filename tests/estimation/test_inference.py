"""Tests for Normal and its conjugate update."""

import math
from functools import reduce

import pytest
from pydantic import ValidationError

from increment.estimation.inference import DIFFUSE_SIGMA, Normal


class TestNormal:
    def test_frozen(self):
        n = Normal(mu=0.0, sigma=1.0)
        with pytest.raises((TypeError, ValueError)):
            n.mu = 1.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_sigma_positive(self):
        """Sigma must be > 0."""
        Normal(mu=0.0, sigma=1e-6)  # tiny positive is fine
        with pytest.raises(ValidationError):
            Normal(mu=0.0, sigma=0.0)
        with pytest.raises(ValidationError):
            Normal(mu=0.0, sigma=-1.0)

    def test_precision(self):
        n = Normal(mu=0.0, sigma=2.0)
        assert n.precision == pytest.approx(0.25, rel=1e-12)

    def test_diffuse(self):
        d = Normal.diffuse()
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(DIFFUSE_SIGMA)


class TestUpdate:
    def test_diffuse_prior_recovers_the_observation(self):
        """Against an uninformative prior the posterior is the likelihood."""
        obs = Normal(mu=0.5, sigma=0.1)
        post = Normal.diffuse().update(obs)
        assert post.mu == pytest.approx(obs.mu, rel=1e-6)
        assert post.sigma == pytest.approx(obs.sigma, rel=1e-6)

    def test_informative_prior_is_precision_weighted(self):
        prior = Normal(mu=0.0, sigma=0.5)
        obs = Normal(mu=1.0, sigma=0.5)
        post = prior.update(obs)
        expected_sigma = math.sqrt(1.0 / (prior.precision + obs.precision))
        expected_mu = expected_sigma**2 * (prior.mu * prior.precision + obs.mu * obs.precision)
        assert post.sigma == pytest.approx(expected_sigma, rel=1e-12)
        assert post.mu == pytest.approx(expected_mu, rel=1e-12)

    def test_precision_is_additive(self):
        """The defining property of the conjugate update."""
        prior, obs = Normal(mu=-0.3, sigma=0.8), Normal(mu=1.2, sigma=0.4)
        assert prior.update(obs).precision == pytest.approx(
            prior.precision + obs.precision, rel=1e-12
        )

    def test_posterior_mean_lies_between_inputs(self):
        prior, obs = Normal(mu=0.0, sigma=0.5), Normal(mu=2.0, sigma=0.5)
        assert prior.mu < prior.update(obs).mu < obs.mu

    def test_sharper_prior_pulls_posterior_toward_it(self):
        obs = Normal(mu=1.0, sigma=0.5)
        loose = Normal(mu=0.0, sigma=10.0).update(obs)
        tight = Normal(mu=0.0, sigma=0.05).update(obs)
        assert tight.mu < loose.mu

    def test_is_pure(self):
        """Neither operand is mutated."""
        prior, obs = Normal(mu=0.1, sigma=1.0), Normal(mu=0.9, sigma=0.3)
        prior.update(obs)
        assert (prior.mu, prior.sigma) == (0.1, 1.0)
        assert (obs.mu, obs.sigma) == (0.9, 0.3)

    def test_commutative(self):
        """Evidence order does not change the posterior."""
        a, b = Normal(mu=0.5, sigma=0.2), Normal(mu=0.9, sigma=0.3)
        assert a.update(b).mu == pytest.approx(b.update(a).mu, rel=1e-12)
        assert a.update(b).sigma == pytest.approx(b.update(a).sigma, rel=1e-12)

    def test_associative(self):
        a, b, c = (
            Normal(mu=0.5, sigma=0.2),
            Normal(mu=0.9, sigma=0.3),
            Normal(mu=-0.2, sigma=0.4),
        )
        left, right = a.update(b).update(c), a.update(b.update(c))
        assert left.mu == pytest.approx(right.mu, rel=1e-12)
        assert left.sigma == pytest.approx(right.sigma, rel=1e-12)

    def test_chaining_folds_in_every_observation(self):
        """Regression: chained updates must not discard earlier evidence.

        The previous stateful NormalNormal always read its immutable prior,
        so `.update(a).update(b)` silently equalled `.update(b)`.
        """
        prior = Normal(mu=0.1, sigma=1.0)
        a, b = Normal(mu=0.5, sigma=0.2), Normal(mu=0.9, sigma=0.3)

        chained = prior.update(a).update(b)
        dropped_first = prior.update(b)
        assert chained.mu != pytest.approx(dropped_first.mu, rel=1e-9)

        # Chaining equals pooling the observations by precision first.
        pooled_precision = a.precision + b.precision
        pooled = Normal(
            mu=(a.mu * a.precision + b.mu * b.precision) / pooled_precision,
            sigma=math.sqrt(1.0 / pooled_precision),
        )
        assert chained.mu == pytest.approx(prior.update(pooled).mu, rel=1e-12)
        assert chained.sigma == pytest.approx(prior.update(pooled).sigma, rel=1e-12)

    def test_reduce_over_many_observations(self):
        obs = [Normal(mu=float(i) / 10, sigma=0.2 + i / 100) for i in range(8)]
        folded = reduce(Normal.update, obs, Normal.diffuse())
        total_precision = sum(o.precision for o in obs) + Normal.diffuse().precision
        assert folded.precision == pytest.approx(total_precision, rel=1e-9)
