"""Independent likelihood references for the Beta/NIG/NIW kernels."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations
from math import factorial, nextafter
from typing import cast

import pytest

from increment.errors import InvalidRequestError
from increment.estimation._certified import Interval
from increment.estimation._sequential_likelihood import (
    Alternative,
    BernoulliState,
    BetaPrior,
    GaussianPrior,
    GaussianState,
    LikelihoodCertificate,
    bernoulli_evidence,
    gaussian_evidence,
)

F = Fraction
_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651"
)


@contextmanager
def _refuses(code: str, **context: object) -> Iterator[None]:
    with pytest.raises(InvalidRequestError) as raised:
        yield
    if code.startswith("estimation."):
        expected = code
    elif code.startswith("certified."):
        expected = "estimation." + code
    else:
        expected = "estimation.sequential_likelihood." + code
    assert raised.value.code == expected
    assert raised.value.context == context


def _decimal(value):
    value = F(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _rising(value, count):
    result = F(1)
    for i in range(count):
        result *= value + i
    return result


def _beta_q(state, prior):
    return (
        _rising(prior.a, state.successes)
        * _rising(prior.b, state.n - state.successes)
        / _rising(prior.a + prior.b, state.n)
    )


def _prior(dimension=1, *, nu=4):
    return GaussianPrior(
        F(1),
        F(nu),
        (F(0),) * dimension,
        tuple(tuple(F(2 if i == j else 0) for j in range(dimension)) for i in range(dimension)),
    )


def _scalar(n, mean, scatter):
    return GaussianState(n, (F(mean),), ((F(scatter),),))


def _vector(n, mean, scatter):
    return GaussianState(n, mean, scatter)


def _det(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _log_gamma_half_integer(value):
    value = F(value)
    if value.denominator == 1:
        return Decimal(factorial(int(value) - 1)).ln()
    assert value.denominator == 2
    k = int(value - F(1, 2))
    return _decimal(F(factorial(2 * k), 4**k * factorial(k))).ln() + _PI.ln() / 2


def _predictive_reference(state, prior):
    """Independent Decimal closed form; integer shapes allow factorial gamma."""
    if state.n == 0:
        return Decimal(0)
    n, d = state.n, len(state.mean)
    posterior = tuple(
        tuple(
            prior.scale[i][j]
            + state.scatter[i][j]
            + prior.kappa
            * n
            / (prior.kappa + n)
            * (state.mean[i] - prior.mean[i])
            * (state.mean[j] - prior.mean[j])
            for j in range(d)
        )
        for i in range(d)
    )
    if d == 2:
        # Evaluate both ordinary univariate gamma factors before removing
        # the Gaussian normalizer; this checks NIW's duplication cancellation.
        gamma = (
            _log_gamma_half_integer((prior.nu + n) / 2)
            + _log_gamma_half_integer((prior.nu + n - 1) / 2)
            - _log_gamma_half_integer(prior.nu / 2)
            - _log_gamma_half_integer((prior.nu - 1) / 2)
        )
        return (
            _decimal(prior.kappa / (prior.kappa + n)).ln()
            + _decimal(prior.nu / 2) * _decimal(_det(prior.scale)).ln()
            - _decimal((prior.nu + n) / 2) * _decimal(_det(posterior)).ln()
            + gamma
            + n * Decimal(2).ln()
        )
    a = prior.nu / 2
    assert a.denominator == 1
    final = a + F(n, 2)
    gamma = _log_gamma_half_integer(final) - _log_gamma_half_integer(a)
    return (
        _decimal(prior.kappa / (prior.kappa + n)).ln() / 2
        + gamma
        + _decimal(a) * _decimal(prior.scale[0][0] / 2).ln()
        - _decimal(final) * _decimal(posterior[0][0] / 2).ln()
    )


def _scalar_likelihood(state, mean):
    if not state.n:
        return Decimal(0)
    residual = _decimal(state.scatter[0][0]) + state.n * (_decimal(state.mean[0]) - mean) ** 2
    return -Decimal(state.n) / 2 * (1 + (residual / state.n).ln())


def _ray_likelihood(state, rho):
    """Independent covariance fit by projecting the mean and taking det(M)."""
    if not state.n:
        return Decimal(0)
    c, d, e = map(_decimal, (state.scatter[0][0], state.scatter[0][1], state.scatter[1][1]))
    a, b = map(_decimal, state.mean)
    scale = max(
        Decimal(0), ((e * a - d * b) * rho + c * b - d * a) / (e * rho * rho - 2 * d * rho + c)
    )
    dy, dd = a - scale * rho, b - scale
    fitted_det = (c + state.n * dy * dy) * (e + state.n * dd * dd) - (d + state.n * dy * dd) ** 2
    return -state.n + state.n * Decimal(state.n).ln() - Decimal(state.n) / 2 * fitted_det.ln()


def _assert_contains(interval, expected):
    assert interval is not None
    # Decimal oracles are calculated at 90 digits. This allowance is far
    # smaller than the requested production enclosure, and covers oracle rounding.
    slack = F(1, 10**75)
    assert interval.lo <= F(expected) + slack
    assert interval.hi >= F(expected) - slack


def _assert_finite(certificate, max_error=Fraction(1, 10**12)):
    assert certificate.status == "finite"
    assert certificate.log_predictive is not None
    assert certificate.log_null_sup is not None
    assert certificate.log_e is not None
    assert certificate.reason is None
    assert certificate.log_e == certificate.log_predictive - certificate.log_null_sup
    assert certificate.log_e.width <= max_error


@pytest.mark.parametrize(
    "n,successes,code,context",
    [
        (-1, 0, "certified.invalid_count", {"count": -1}),
        (0, 1, "successes_exceed_count", {"n": 0, "successes": 1}),
        (2, -1, "certified.invalid_count", {"count": -1}),
        (2, 3, "successes_exceed_count", {"n": 2, "successes": 3}),
        (2.0, 1, "certified.invalid_count", {"count": 2.0}),
        (True, 0, "certified.invalid_count", {"count": True}),
    ],
)
def test_bernoulli_counts_are_exact_and_valid(n, successes, code, context):
    with _refuses(code, **context):
        BernoulliState(n, successes)


@pytest.mark.parametrize(
    "a,b,code,context",
    [
        (0, 1, "nonpositive_beta_parameter", {"parameter": "a", "value": F(0)}),
        (1, -1, "nonpositive_beta_parameter", {"parameter": "b", "value": F(-1)}),
        (0.5, 1, "certified.inexact_value", {"value_type": "float"}),
        (F(1), float("inf"), "certified.inexact_value", {"value_type": "float"}),
    ],
)
def test_beta_prior_requires_positive_exact_parameters(a, b, code, context):
    with _refuses(code, **context):
        BetaPrior(a, b)


@pytest.mark.parametrize(
    "n,mean,scatter,code,context",
    [
        (0, [1], [[0]], "nonzero_empty_mean", {"mean": (F(1),)}),
        (0, [0], [[1]], "nonzero_small_sample_scatter", {"n": 0, "scatter": ((F(1),),)}),
        (1, [0], [[1]], "nonzero_small_sample_scatter", {"n": 1, "scatter": ((F(1),),)}),
        (
            2,
            [0],
            [[-1]],
            "non_psd_matrix",
            {"matrix": ((-1,),), "principal_minors": (-1, -1)},
        ),
        (
            2,
            [0, 0],
            [[1, 2], [2, 1]],
            "non_psd_matrix",
            {"matrix": ((1, 2), (2, 1)), "principal_minors": (1, 1, -3)},
        ),
        (
            2,
            [0, 0],
            [[1, 0], [1, 1]],
            "asymmetric_matrix",
            {"matrix": ((1, 0), (1, 1))},
        ),
        (
            2,
            [0, 0],
            [[1, 0], [0, 1]],
            "scatter_rank_exceeds_count",
            {"n": 2, "scatter": ((1, 0), (0, 1))},
        ),
        (
            2,
            [0, 0],
            [[1, 0]],
            "matrix_dimension_mismatch",
            {"dimension": 2, "row_lengths": (2,)},
        ),
        (2, [], [], "invalid_dimension", {"dimension": 0}),
        (
            2,
            [0, 0, 0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            "scatter_rank_exceeds_count",
            {"n": 2, "scatter": ((1, 0, 0), (0, 1, 0), (0, 0, 0))},
        ),
        (
            3,
            [0, 0, 0],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "scatter_rank_exceeds_count",
            {"n": 3, "scatter": ((1, 0, 0), (0, 1, 0), (0, 0, 1))},
        ),
        # Leading minors are all zero; only the (1, 2) principal minor is negative.
        (
            4,
            [0, 0, 0],
            [[0, 0, 0], [0, 1, 2], [0, 2, 1]],
            "non_psd_matrix",
            {
                "matrix": ((0, 0, 0), (0, 1, 2), (0, 2, 1)),
                "principal_minors": (0, 1, 1, 0, 0, -3, 0),
            },
        ),
        (2, [0.0], [[1]], "certified.inexact_value", {"value_type": "float"}),
        (
            2,
            [0, 0],
            [[0, 1], [1, 2]],
            "non_psd_matrix",
            {"matrix": ((0, 1), (1, 2)), "principal_minors": (0, 2, -1)},
        ),
    ],
)
def test_gaussian_state_rejects_invalid_exact_centered_summaries(n, mean, scatter, code, context):
    with _refuses(code, **context):
        GaussianState(n, mean, scatter)


@pytest.mark.parametrize(
    "kappa,nu,mean,scale,code,context",
    [
        (0, 2, [0], [[1]], "nonpositive_prior_kappa", {"kappa": F(0)}),
        (1, 0, [0], [[1]], "insufficient_prior_nu", {"nu": F(0), "dimension": 1}),
        (
            1,
            1,
            [0, 0],
            [[1, 0], [0, 1]],
            "insufficient_prior_nu",
            {"nu": F(1), "dimension": 2},
        ),
        (
            1,
            2,
            [0],
            [[0]],
            "nonpositive_definite_matrix",
            {"matrix": ((0,),), "principal_minors": (0, 0)},
        ),
        (
            1,
            2,
            [0, 0],
            [[1, 1], [1, 1]],
            "nonpositive_definite_matrix",
            {"matrix": ((1, 1), (1, 1)), "principal_minors": (1, 1, 0)},
        ),
        (
            1,
            2,
            [0, 0],
            [[1, 0], [1, 2]],
            "asymmetric_matrix",
            {"matrix": ((1, 0), (1, 2))},
        ),
    ],
)
def test_gaussian_prior_is_proper(kappa, nu, mean, scale, code, context):
    with _refuses(code, **context):
        GaussianPrior(kappa, nu, mean, scale)


def test_state_and_prior_copy_caller_owned_nested_inputs():
    mean, scatter, scale = [F(1), F(2)], [[F(2), F(1)], [F(1), F(3)]], [[F(3), F(1)], [F(1), F(2)]]
    state = GaussianState(4, mean, scatter)
    prior = GaussianPrior(1, 3, mean, scale)
    state_hash, prior_hash = hash(state), hash(prior)
    mean[0], scatter[0][0], scale[1][1] = F(99), F(99), F(99)
    assert state.mean == prior.mean == (1, 2)
    assert state.scatter == ((2, 1), (1, 3))
    assert prior.scale == ((3, 1), (1, 2))
    assert (hash(state), hash(prior)) == (state_hash, prior_hash)


def test_binary64_rows_have_exact_dyadic_order_and_partition_invariance():
    rows = [
        (2.0**50, 0.125),
        (nextafter(2.0**50, float("inf")), -0.5),
        (-(2.0**50), 0.25),
        (0.1, nextafter(0.1, 1.0)),
    ]
    whole = GaussianState.from_rows(rows, dimension=2)
    exact_mean = tuple(
        sum((F(*row[i].as_integer_ratio()) for row in rows), F(0)) / len(rows) for i in range(2)
    )
    assert whole.mean == exact_mean
    for ordering in permutations(rows):
        assert GaussianState.from_rows(ordering, dimension=2) == whole
        for split in range(len(rows) + 1):
            left = GaussianState.from_rows(ordering[:split], dimension=2)
            right = GaussianState.from_rows(ordering[split:], dimension=2)
            assert left.merge(right) == right.merge(left) == whole


def test_rows_are_snapshotted_and_scalar_centering_is_exact():
    rows = [[0.1], [0.2], [0.3]]
    state = GaussianState.from_rows(rows, dimension=1)
    exact = [F(*row[0].as_integer_ratio()) for row in rows]
    mean = sum(exact) / 3
    assert state.mean == (mean,)
    assert state.scatter == ((sum((x - mean) ** 2 for x in exact),),)
    rows[0][0] = 100
    assert state.mean == (mean,)


@pytest.mark.parametrize(
    "rows,dimension,code,context",
    [
        ([[1, 2]], 1, "row_dimension_mismatch", {"dimension": 1, "row_dimension": 2}),
        ([[1]], 2, "row_dimension_mismatch", {"dimension": 2, "row_dimension": 1}),
        ([], 0, "invalid_dimension", {"dimension": 0}),
        ([], 1.0, "invalid_dimension", {"dimension": 1.0}),
    ],
)
def test_rows_reject_wrong_dimensions(rows, dimension, code, context):
    with _refuses(code, **context):
        GaussianState.from_rows(rows, dimension=dimension)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rows_reject_nonfinite_observations(value):
    with _refuses("nonfinite_observation", value=value):
        GaussianState.from_rows([[value]], dimension=1)


def test_cannot_merge_dimensions():
    with _refuses("merge_dimension_mismatch", left_dimension=1, right_dimension=2):
        GaussianState.empty(1).merge(GaussianState.empty(2))


def test_certificate_validates_interval_semantics():
    q, d = Interval(F(-2), F(-1)), Interval(F(-4), F(-3))
    certificate = LikelihoodCertificate("finite", q, d, q - d, None)
    assert certificate.log_e == Interval(F(1), F(3))
    with _refuses("evidence_enclosure_mismatch", log_e=Interval.exact(2), required=Interval(1, 3)):
        LikelihoodCertificate("finite", q, d, Interval.exact(2), None)
    with _refuses("missing_finite_logs", log_predictive=None, log_null_sup=d, log_e=q - d):
        LikelihoodCertificate("finite", None, d, q - d, None)
    with _refuses(
        "inconsistent_nonfinite_logs",
        status="zero",
        log_null_sup=None,
        log_e=Interval.exact(-1000),
        reason="singular",
    ):
        LikelihoodCertificate("zero", q, None, Interval.exact(-1000), "singular")
    with _refuses(
        "inconsistent_nonfinite_logs", status="infinite", log_null_sup=None, log_e=None, reason=None
    ):
        LikelihoodCertificate("infinite", q, None, None, None)
    with pytest.raises(FrozenInstanceError):
        certificate.__setattr__("reason", "changed")


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_entirely_empty_prefix_is_exactly_one(alternative):
    beta = BetaPrior(1, 1)
    result = bernoulli_evidence(
        BernoulliState(0, 0), BernoulliState(0, 0), beta, beta, ratio=F(0), alternative=alternative
    )
    assert result == LikelihoodCertificate(
        "finite", Interval.exact(0), Interval.exact(0), Interval.exact(0), None
    )
    for dimension in (1, 2):
        prior, empty = _prior(dimension), GaussianState.empty(dimension)
        assert (
            gaussian_evidence(empty, empty, prior, prior, ratio=F(-2), alternative=alternative)
            == result
        )


@pytest.mark.parametrize(
    "control,treatment",
    [
        (BernoulliState(2, 1), BernoulliState(2, 2)),
        (BernoulliState(3, 0), BernoulliState(2, 0)),
        (BernoulliState(3, 3), BernoulliState(2, 2)),
        (BernoulliState(0, 0), BernoulliState(3, 1)),
    ],
)
def test_bernoulli_pooled_reference_including_endpoints_and_empty_arm(control, treatment):
    pc, pt = BetaPrior(2, 3), BetaPrior(1, 2)
    certificate = bernoulli_evidence(
        control, treatment, pc, pt, ratio=F(1), alternative="two-sided"
    )
    _assert_finite(certificate)
    n, successes = control.n + treatment.n, control.successes + treatment.successes
    probability = F(successes, n)
    likelihood = probability**successes * (1 - probability) ** (n - successes)
    with localcontext() as context:
        context.prec = 90
        _assert_contains(
            certificate.log_e,
            _decimal(_beta_q(control, pc) * _beta_q(treatment, pt) / likelihood).ln(),
        )
        _assert_contains(certificate.log_null_sup, _decimal(likelihood).ln())


def test_bernoulli_irrational_quadratic_root_against_decimal_closed_form():
    control = treatment = BernoulliState(2, 1)
    prior = BetaPrior(1, 1)
    certificate = bernoulli_evidence(
        control, treatment, prior, prior, ratio=F(2), alternative="two-sided"
    )
    _assert_finite(certificate)
    with localcontext() as context:
        context.prec = 90
        u = (Decimal(9) - Decimal(17).sqrt()) / 16
        likelihood = u * (1 - u) * (2 * u) * (1 - 2 * u)
        log_q = _decimal(F(1, 36)).ln()
        _assert_contains(certificate.log_null_sup, likelihood.ln())
        _assert_contains(certificate.log_e, log_q - likelihood.ln())


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("successes", [0, 1])
def test_zero_bernoulli_ratio_distinguishes_zero_null_likelihood(alternative, successes):
    c, t, prior = BernoulliState(3, 1), BernoulliState(2, successes), BetaPrior(1, 1)
    result = bernoulli_evidence(c, t, prior, prior, ratio=F(0), alternative=alternative)
    if successes and alternative != "less":
        assert result.status == "infinite"
        assert result.log_e is None and result.reason
    else:
        _assert_finite(result)
        with localcontext() as context:
            context.prec = 90
            likelihood = F(4, 27) * (F(1, 4) if successes else 1)
            _assert_contains(
                result.log_e, _decimal(_beta_q(c, prior) * _beta_q(t, prior) / likelihood).ln()
            )


@pytest.mark.parametrize("empty_control", [True, False])
@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_bernoulli_empty_arm_mle_feasibility_uses_an_interval(empty_control, alternative):
    empty, full, prior = BernoulliState(0, 0), BernoulliState(3, 3), BetaPrior(1, 1)
    c, t = (empty, full) if empty_control else (full, empty)
    result = bernoulli_evidence(c, t, prior, prior, ratio=F(2), alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        # An empty treatment may take any pT, but pT>=2*pC caps pC at 1/2.
        likelihood = F(1, 8) if not empty_control and alternative == "less" else F(1)
        _assert_contains(result.log_e, _decimal(F(1, 4) / likelihood).ln())


def test_bernoulli_directional_membership_and_null_nesting():
    c, t, prior = BernoulliState(8, 2), BernoulliState(8, 6), BetaPrior(1, 1)
    greater = [
        bernoulli_evidence(c, t, prior, prior, ratio=r, alternative="greater")
        for r in (F(1, 2), F(1), F(2), F(3), F(4))
    ]
    less = [
        bernoulli_evidence(c, t, prior, prior, ratio=r, alternative="less")
        for r in (F(1), F(2), F(3), F(4), F(5))
    ]
    for a, b in zip(greater[:3], greater[1:4], strict=True):
        assert a.log_e is not None and b.log_e is not None
        assert a.log_e.lo > b.log_e.hi
    for a, b in zip(less[2:-1], less[3:], strict=True):
        assert a.log_e is not None and b.log_e is not None
        assert a.log_e.hi < b.log_e.lo
    for result in (greater[3], greater[4], *less[:3]):
        with localcontext() as context:
            context.prec = 90
            likelihood = F(1, 4) ** 4 * F(3, 4) ** 12
            _assert_contains(result.log_null_sup, _decimal(likelihood).ln())


@pytest.mark.parametrize(
    "ratio,code,context",
    [
        (F(-1), "negative_bernoulli_ratio", {"ratio": F(-1)}),
        (1.0, "certified.inexact_value", {"value_type": "float"}),
    ],
)
def test_invalid_bernoulli_ratio_is_rejected(ratio, code, context):
    with _refuses(code, **context):
        bernoulli_evidence(
            BernoulliState(0, 0),
            BernoulliState(0, 0),
            BetaPrior(1, 1),
            BetaPrior(1, 1),
            ratio=ratio,
            alternative="two-sided",
        )


@pytest.mark.parametrize(
    "scatters,means,optimum",
    [((F(105, 16), F(63, 16)), (1, 5), F(19, 4)), ((4, 4), (1, 3), F(2)), ((4, 4), (-1, 1), F(0))],
)
def test_scalar_all_three_cubic_roots_triple_root_and_boundary(scatters, means, optimum):
    # Unequal scatters give roots 3/2, 11/4, 19/4, with the right maximum higher.
    # Equal scatters give a triple root, including u=0 in the final case.
    c, t, prior = _scalar(4, means[0], scatters[0]), _scalar(4, means[1], scatters[1]), _prior()
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _scalar_likelihood(c, _decimal(optimum)) + _scalar_likelihood(
            t, _decimal(optimum)
        )
        reference = _predictive_reference(c, prior) + _predictive_reference(t, prior) - denominator
        _assert_contains(result.log_null_sup, denominator)
        _assert_contains(result.log_e, reference)


@pytest.mark.parametrize("ratio,means", [(F(-2), (2, -4)), (F(0), (2, -3)), (F(1), (2, -3))])
def test_scalar_negative_treatment_unequal_variance_and_zero_or_negative_ratio(ratio, means):
    c, t, prior = _scalar(4, means[0], 2), _scalar(6, means[1], 17), _prior()
    alternative = "two-sided" if ratio <= 0 else "greater"
    result = gaussian_evidence(c, t, prior, prior, ratio=ratio, alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        treatment_mean = Decimal(0) if ratio == 0 else _decimal(t.mean[0])
        denominator = _scalar_likelihood(c, _decimal(c.mean[0])) + _scalar_likelihood(
            t, treatment_mean
        )
        _assert_contains(result.log_null_sup, denominator)
        _assert_contains(
            result.log_e,
            _predictive_reference(c, prior) + _predictive_reference(t, prior) - denominator,
        )


@pytest.mark.parametrize("alternative,treatment_mean", [("greater", -3), ("less", 3)])
def test_scalar_positive_control_boundary_keeps_negative_treatment_means(
    alternative, treatment_mean
):
    c, t, prior = _scalar(4, -2, 3), _scalar(4, treatment_mean, 7), _prior()
    result = gaussian_evidence(c, t, prior, prior, ratio=F(2), alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _scalar_likelihood(c, Decimal(0)) + _scalar_likelihood(
            t, Decimal(treatment_mean)
        )
        _assert_contains(result.log_null_sup, denominator)


def test_scalar_directional_composite_nulls_are_nested():
    c, t, prior = _scalar(4, 1, 3), _scalar(6, 4, 5), _prior()
    plus = [
        gaussian_evidence(c, t, prior, prior, ratio=r, alternative="greater")
        for r in (F(1), F(2), F(3))
    ]
    minus = [
        gaussian_evidence(c, t, prior, prior, ratio=r, alternative="less")
        for r in (F(5), F(6), F(7))
    ]
    for a, b in zip(plus[:-1], plus[1:], strict=True):
        assert a.log_e is not None and b.log_e is not None
        assert a.log_e.lo > b.log_e.hi
    for a, b in zip(minus[:-1], minus[1:], strict=True):
        assert a.log_e is not None and b.log_e is not None
        assert a.log_e.hi < b.log_e.lo


@pytest.mark.parametrize("empty_control,ratio", [(True, F(2)), (True, F(-2)), (False, F(0))])
def test_scalar_one_empty_arm_retains_other_arms_stationary_point(empty_control, ratio):
    empty, prior = GaussianState.empty(1), _prior()
    observed = _scalar(4, -3 if ratio < 0 else 3, 5)
    c, t = (empty, observed) if empty_control else (observed, empty)
    result = gaussian_evidence(c, t, prior, prior, ratio=ratio, alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _scalar_likelihood(observed, _decimal(observed.mean[0]))
        _assert_contains(result.log_null_sup, denominator)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_feasible_scalar_singular_fit_is_zero_evidence(alternative):
    result = gaussian_evidence(
        _scalar(1, 2, 0), _scalar(4, 4, 3), _prior(), _prior(), ratio=F(2), alternative=alternative
    )
    assert result.status == "zero"
    assert result.log_e is None and result.log_null_sup is None
    assert result.log_predictive is not None and result.reason


def test_singular_scalar_scatter_can_have_a_finite_constrained_likelihood():
    c, t, prior = _scalar(1, -2, 0), _scalar(1, -3, 0), _prior(nu=2)
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _scalar_likelihood(c, Decimal(0)) + _scalar_likelihood(t, Decimal(0))
        _assert_contains(
            result.log_e,
            _predictive_reference(c, prior) + _predictive_reference(t, prior) - denominator,
        )


@pytest.mark.parametrize(
    "ratio,alternative,expected",
    [
        (F(0), "two-sided", "finite"),
        (F(-1), "two-sided", "zero"),
        (F(1), "greater", "zero"),
        (F(1), "less", "finite"),
    ],
)
def test_treatment_singular_fit_depends_on_the_null(ratio, alternative, expected):
    result = gaussian_evidence(
        _scalar(4, 2, 3),
        _scalar(1, -2, 0),
        _prior(),
        _prior(),
        ratio=ratio,
        alternative=alternative,
    )
    assert result.status == expected
    if expected == "finite":
        _assert_finite(result)


@pytest.mark.parametrize("count", [3, 4, 5, 6])
def test_scalar_predictive_even_and_odd_gamma_recurrences(count):
    c, t, prior = _scalar(count, 2, 3), _scalar(4, 4, 7), _prior()
    result = gaussian_evidence(c, t, prior, prior, ratio=F(2), alternative="two-sided")
    with localcontext() as context:
        context.prec = 90
        _assert_contains(
            result.log_predictive, _predictive_reference(c, prior) + _predictive_reference(t, prior)
        )


@pytest.mark.parametrize("dimension", [1, 2])
def test_noninteger_proper_gaussian_shapes_are_supported(dimension):
    prior = _prior(dimension, nu=F(7, 3))
    state = _scalar(3, 1, 2) if dimension == 1 else _vector(3, (2, 1), ((2, 1), (1, 3)))
    result = gaussian_evidence(state, state, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)


@pytest.mark.parametrize("dimension", [1, 2])
def test_nonzero_prior_mean_and_correlated_prior_scale_enter_predictive(dimension):
    if dimension == 1:
        prior = GaussianPrior(F(3, 2), F(4), (F(7),), ((F(5),),))
        state = _scalar(5, 2, 3)
    else:
        prior = GaussianPrior(F(3, 2), F(4), (F(7), F(-2)), ((F(5), F(1)), (F(1), F(3))))
        state = _vector(5, (2, 1), ((5, 2), (2, 3)))
    result = gaussian_evidence(state, state, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        _assert_contains(result.log_predictive, 2 * _predictive_reference(state, prior))


@pytest.mark.parametrize("count", [3, 4, 5, 6])
def test_bivariate_correlated_niw_predictive_and_unrestricted_mle_reference(count):
    c = _vector(count, (2, 1), ((5, 2), (2, 3)))
    t = _vector(4, (6, 3), ((7, -1), (-1, 2)))
    prior = _prior(2)
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _ray_likelihood(c, Decimal(2)) + _ray_likelihood(t, Decimal(2))
        _assert_contains(
            result.log_predictive, _predictive_reference(c, prior) + _predictive_reference(t, prior)
        )
        _assert_contains(
            result.log_e,
            _predictive_reference(c, prior) + _predictive_reference(t, prior) - denominator,
        )


@pytest.mark.slow
def test_bivariate_sixth_degree_profile_has_three_positive_stationary_roots():
    # The determinant-product derivative is (x-1)Q(x) times a positive factor.
    # Descartes' rule and these brackets establish all three positive roots.
    c, t = _vector(4, (3, 1), ((1, 0), (0, 1))), _vector(4, (1, 3), ((8, 0), (0, 1)))
    prior = _prior(2)
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90

        def polynomial(x):
            return ((((126 * x + 41) * x - 652) * x - 134) * x - 1835) * x + 1176

        roots = []
        for lo, hi in [(Decimal("0.5"), Decimal("0.75")), (Decimal(2), Decimal(3))]:
            negative_at_lo = polynomial(lo) < 0
            assert negative_at_lo != (polynomial(hi) < 0)
            for _ in range(300):
                middle = (lo + hi) / 2
                if (polynomial(middle) < 0) == negative_at_lo:
                    lo = middle
                else:
                    hi = middle
            roots.append((lo + hi) / 2)
        denominator = _ray_likelihood(c, roots[1]) + _ray_likelihood(t, roots[1])
        other = _ray_likelihood(c, roots[0]) + _ray_likelihood(t, roots[0])
        assert denominator > other
        assert other > _ray_likelihood(c, Decimal(1)) + _ray_likelihood(t, Decimal(1))
        _assert_contains(result.log_null_sup, denominator)
        _assert_contains(
            result.log_e,
            _predictive_reference(c, prior) + _predictive_reference(t, prior) - denominator,
        )


def test_bivariate_sixth_degree_multiple_root_is_not_lost():
    # Here z*=1/2, so the three positive stationary roots coalesce at x=1.
    c, t = _vector(3, (3, 1), ((10, 0), (0, 10))), _vector(3, (1, 3), ((10, 0), (0, 10)))
    result = gaussian_evidence(c, t, _prior(2), _prior(2), ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        _assert_contains(
            result.log_null_sup, _ray_likelihood(c, Decimal(1)) + _ray_likelihood(t, Decimal(1))
        )


@pytest.mark.parametrize("means", [((-2, -1), (-3, -1)), ((0, 0), (0, 0))])
def test_bivariate_clipped_and_identically_constant_branches(means):
    c, t = (_vector(4, mean, ((2, 0), (0, 3))) for mean in means)
    prior = _prior(2)
    result = gaussian_evidence(c, t, prior, prior, ratio=F(2), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _ray_likelihood(c, Decimal(0)) + _ray_likelihood(t, Decimal(0))
        _assert_contains(result.log_null_sup, denominator)


def test_bivariate_single_interior_arm_and_affine_clipping_transition():
    c = _vector(4, (2, 1), ((2, 0), (0, 3)))
    t = _vector(4, (-3, 1), ((2, 0), (0, 3)))
    # The treatment ray clips at control slope 1/9; the later branch attains
    # its global supremum at the control mean's unconstrained slope, 2.
    prior = _prior(2)
    result = gaussian_evidence(c, t, prior, prior, ratio=F(2), alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        _assert_contains(
            result.log_null_sup, _ray_likelihood(c, Decimal(2)) + _ray_likelihood(t, Decimal(4))
        )


@pytest.mark.parametrize("mean,ratio,optimum", [((2, -1), F(1), "tail"), ((-2, 1), F(1), "zero")])
def test_bivariate_null_supremum_can_be_a_domain_boundary_limit(mean, ratio, optimum):
    state, empty, prior = _vector(4, mean, ((2, 0), (0, 3))), GaussianState.empty(2), _prior(2)
    result = gaussian_evidence(state, empty, prior, prior, ratio=ratio, alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        q = F(mean[1] ** 2, 3) if optimum == "tail" else F(mean[0] ** 2, 2)
        denominator = -4 + 4 * Decimal(4).ln() - 2 * _decimal(6 * (1 + 4 * q)).ln()
        _assert_contains(result.log_null_sup, denominator)


@pytest.mark.parametrize("ratio,treatment_mean", [(F(-2), (-4, 1)), (F(0), (0, 1))])
def test_bivariate_zero_and_negative_ratio_keep_treatment_numerators_unrestricted(
    ratio, treatment_mean
):
    c, t = _vector(4, (2, 1), ((3, 1), (1, 2))), _vector(4, treatment_mean, ((4, -1), (-1, 2)))
    prior = _prior(2)
    result = gaussian_evidence(c, t, prior, prior, ratio=ratio, alternative="two-sided")
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _ray_likelihood(c, Decimal(2)) + _ray_likelihood(t, _decimal(2 * ratio))
        _assert_contains(result.log_null_sup, denominator)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_bivariate_empty_control_can_match_the_observed_treatment_ratio(alternative):
    empty, t, prior = GaussianState.empty(2), _vector(4, (-4, 1), ((3, 1), (1, 2))), _prior(2)
    result = gaussian_evidence(empty, t, prior, prior, ratio=F(-2), alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        _assert_contains(result.log_null_sup, _ray_likelihood(t, Decimal(-4)))


@pytest.mark.parametrize("alternative,mean", [("greater", (-3, -1)), ("less", (3, -1))])
def test_bivariate_one_sided_horizontal_cone_face(alternative, mean):
    c, t, prior = (
        _vector(4, (2, 1), ((3, 1), (1, 2))),
        _vector(4, mean, ((2, 0), (0, 3))),
        _prior(2),
    )
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        # Projection keeps treatment Y and sets D=0, hence q=mD²/Sdd.
        treatment_log = -4 + 4 * Decimal(4).ln() - 2 * Decimal(14).ln()
        denominator = _ray_likelihood(c, Decimal(2)) + treatment_log
        _assert_contains(result.log_null_sup, denominator)


@pytest.mark.parametrize("alternative,mean", [("greater", (-3, 1)), ("less", (3, 1))])
def test_bivariate_one_sided_interior_cone_candidate(alternative, mean):
    c, t, prior = (
        _vector(4, (2, 1), ((3, 1), (1, 2))),
        _vector(4, mean, ((2, 1), (1, 3))),
        _prior(2),
    )
    result = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative=alternative)
    _assert_finite(result)
    with localcontext() as context:
        context.prec = 90
        denominator = _ray_likelihood(c, Decimal(2)) + _ray_likelihood(t, Decimal(mean[0]))
        _assert_contains(result.log_null_sup, denominator)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [F(-2), F(0), F(2)])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_bivariate_correlated_branch_cuts_dominate_feasible_grid_witnesses(ratio, alternative):
    c, t, prior = (
        _vector(5, (2, -1), ((5, 2), (2, 3))),
        _vector(4, (-1, 2), ((3, -1), (-1, 2))),
        _prior(2),
    )
    result = gaussian_evidence(c, t, prior, prior, ratio=ratio, alternative=alternative)
    _assert_finite(result)
    assert result.log_null_sup is not None
    with localcontext() as context:
        context.prec = 90
        for k in range(81):
            x = Decimal(k) / 8
            witness = _ray_likelihood(c, x) + _ray_likelihood(t, _decimal(ratio) * x)
            # A grid is only a LOWER witness for sup L, never a denominator
            # certificate or an assertion that the returned lower bound is small.
            assert F(witness) <= result.log_null_sup.hi + F(1, 10**75)
        for cut in (F(9, 8), F(5, 2)):
            witness = _ray_likelihood(c, _decimal(cut)) + _ray_likelihood(t, _decimal(ratio * cut))
            assert F(witness) <= result.log_null_sup.hi + F(1, 10**75)


def test_bivariate_singular_prefix_abstains_and_later_prefix_recovers():
    prior = _prior(2)
    rows = [(1, 1), (2, 2), (3, 1)]
    for count in (1, 2):
        singular = GaussianState.from_rows(rows[:count], dimension=2)
        result = gaussian_evidence(
            singular, singular, prior, prior, ratio=F(1), alternative="two-sided"
        )
        assert result.status == "zero" and result.log_e is None
        assert result.reason is not None
        assert result.log_predictive is not None
    full = GaussianState.from_rows(rows, dimension=2)
    _assert_finite(gaussian_evidence(full, full, prior, prior, ratio=F(1), alternative="two-sided"))


@pytest.mark.parametrize(
    "function,c,t,prior",
    [
        (bernoulli_evidence, BernoulliState(5, 2), BernoulliState(7, 5), BetaPrior(1, 1)),
        (gaussian_evidence, _scalar(4, 1, 3), _scalar(6, 3, 7), _prior()),
        (
            gaussian_evidence,
            _vector(4, (3, 1), ((1, 0), (0, 1))),
            _vector(4, (1, 3), ((1, 0), (0, 1))),
            _prior(2),
        ),
    ],
    ids=["bernoulli", "scalar", "vector"],
)
def test_refinement_is_nested_and_conservative(function, c, t, prior):
    loose = function(c, t, prior, prior, ratio=F(2), alternative="two-sided", max_error=F(1, 10**5))
    tight = function(
        c, t, prior, prior, ratio=F(2), alternative="two-sided", max_error=F(1, 10**25)
    )
    _assert_finite(loose, F(1, 10**5))
    _assert_finite(tight, F(1, 10**25))
    for wide, narrow in (
        (loose.log_predictive, tight.log_predictive),
        (loose.log_null_sup, tight.log_null_sup),
        (loose.log_e, tight.log_e),
    ):
        assert wide is not None and narrow is not None
        assert wide.lo <= narrow.lo <= narrow.hi <= wide.hi


@pytest.mark.slow
def test_large_bernoulli_counts_do_not_require_linear_work_or_overflow():
    count = 10**18
    c, t, prior = (
        BernoulliState(count, count // 4),
        BernoulliState(count, 3 * count // 4),
        BetaPrior(1, 1),
    )
    result = bernoulli_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    _assert_finite(result)
    assert result.log_e is not None
    assert result.log_e.lo > 10**15
    with localcontext() as context:
        context.prec = 90
        _assert_contains(result.log_null_sup, -2 * count * Decimal(2).ln())


@pytest.mark.slow
@pytest.mark.parametrize("exponent", [-600, 600])
@pytest.mark.parametrize("dimension", [1, 2])
def test_extreme_dyadic_units_preserve_evidence_when_priors_transform(exponent, dimension):
    factor = F(2) ** exponent
    prior = _prior(dimension)
    if dimension == 1:
        c, t = _scalar(4, 1, 3), _scalar(4, 3, 3)
    else:
        c, t = _vector(4, (3, 1), ((2, 1), (1, 3))), _vector(4, (1, 3), ((3, -1), (-1, 2)))

    def transform(state):
        return GaussianState(
            state.n,
            tuple(x * factor for x in state.mean),
            tuple(tuple(x * factor**2 for x in row) for row in state.scatter),
        )

    transformed_prior = GaussianPrior(
        prior.kappa,
        prior.nu,
        tuple(x * factor for x in prior.mean),
        tuple(tuple(x * factor**2 for x in row) for row in prior.scale),
    )
    base = gaussian_evidence(c, t, prior, prior, ratio=F(1), alternative="two-sided")
    scaled = gaussian_evidence(
        transform(c),
        transform(t),
        transformed_prior,
        transformed_prior,
        ratio=F(1),
        alternative="two-sided",
    )
    _assert_finite(base)
    _assert_finite(scaled)
    assert base.log_e is not None and scaled.log_e is not None
    assert max(base.log_e.lo, scaled.log_e.lo) <= min(base.log_e.hi, scaled.log_e.hi)


@pytest.mark.parametrize(
    "alternative,max_error,code,context",
    [
        ("invalid", F(1), "unknown_alternative", {"alternative": "invalid"}),
        ("greater", F(0), "nonpositive_max_error", {"max_error": F(0)}),
        ("less", F(-1), "nonpositive_max_error", {"max_error": F(-1)}),
    ],
)
def test_invalid_numerical_requests_are_rejected_even_at_empty_prefix(
    alternative, max_error, code, context
):
    with _refuses(code, **context):
        gaussian_evidence(
            GaussianState.empty(1),
            GaussianState.empty(1),
            _prior(),
            _prior(),
            ratio=F(1),
            alternative=alternative,
            max_error=max_error,
        )


def test_gaussian_dimension_mismatch_is_rejected_even_at_empty_prefix():
    with _refuses(
        "state_prior_dimension_mismatch",
        control_dimension=1,
        treatment_dimension=2,
        prior_control_dimension=1,
        prior_treatment_dimension=2,
    ):
        gaussian_evidence(
            GaussianState.empty(1),
            GaussianState.empty(2),
            _prior(),
            _prior(2),
            ratio=F(1),
            alternative="two-sided",
        )


def test_invalid_likelihood_request_exposes_condition_code_and_context():
    with pytest.raises(InvalidRequestError) as raised:
        gaussian_evidence(
            GaussianState.empty(1),
            GaussianState.empty(1),
            _prior(),
            _prior(),
            ratio=F(1),
            alternative=cast("Alternative", "invalid"),
        )
    assert raised.value.code == "estimation.sequential_likelihood.unknown_alternative"
    assert raised.value.context["alternative"] == "invalid"


def test_invalid_scatter_context_snapshots_the_caller_input():
    scatter = [[1, 0], [0, 1]]
    with pytest.raises(InvalidRequestError) as raised:
        GaussianState(2, [0, 0], scatter)
    recorded = cast(Sequence[Sequence[Fraction]], raised.value.context["scatter"])
    scatter[0][0] = 99
    assert raised.value.code == "estimation.sequential_likelihood.scatter_rank_exceeds_count"
    assert raised.value.context["n"] == 2
    # The refusal keeps the scatter it saw, not a view of the caller's list.
    assert [list(row) for row in recorded] == [[F(1), F(0)], [F(0), F(1)]]


@pytest.mark.parametrize(
    "status,predictive,reason,code,context",
    [
        ("invalid", None, None, "unknown_certificate_status", {"status": "invalid"}),
        (
            "zero",
            F(0),
            "singular",
            "invalid_log_interval",
            {"field": "log_predictive", "value_type": "Fraction"},
        ),
        ("zero", None, "", "invalid_certificate_reason", {"reason": ""}),
        (
            "finite",
            "bad",
            None,
            "invalid_log_interval",
            {"field": "log_predictive", "value_type": "str"},
        ),
    ],
)
def test_certificate_metadata_refusals_are_distinct(status, predictive, reason, code, context):
    with _refuses(code, **context):
        LikelihoodCertificate(status, predictive, None, None, reason)


def test_singular_candidate_failure_reports_numerical_context():
    from increment.estimation._sequential_likelihood import _scalar_log

    state = GaussianState(1, [2], [[0]])
    candidate = Interval(1, 3)
    with _refuses(
        "singular_candidate_residual",
        n=1,
        mean=F(2),
        scatter=F(0),
        candidate=candidate,
        residual=Interval(0, 1),
    ):
        _scalar_log(state, candidate, 10)
