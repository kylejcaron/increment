"""Behavioral inversion references, independent of the certified root solver."""

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, localcontext
from fractions import Fraction
from math import factorial
from typing import Any, cast

import pytest

from increment.errors import InvalidRequestError
from increment.estimation._certified import Interval
from increment.estimation._sequential_inversion import (
    ConfidenceBounds,
    EndpointCertificate,
    EndpointEvaluation,
    bernoulli_confidence_sequence,
    gaussian_confidence_sequence,
)
from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    GaussianPrior,
    GaussianState,
    LikelihoodCertificate,
)

F = Fraction


@contextmanager
def _refuses(code: str, **context: object) -> Iterator[None]:
    with pytest.raises(InvalidRequestError) as raised:
        yield
    if code.startswith("estimation."):
        expected = code
    elif code.startswith(("certified.", "sequential_likelihood.")):
        expected = "estimation." + code
    else:
        expected = "estimation.sequential_inversion." + code
    assert raised.value.code == expected
    for key, value in context.items():
        assert raised.value.context[key] == value


def _decimal(value):
    value = F(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _prior(dimension=1, mean=None):
    return GaussianPrior(
        1,
        4,
        mean or (0,) * dimension,
        tuple(tuple(2 if i == j else 0 for j in range(dimension)) for i in range(dimension)),
    )


def _scalar(n=80, mean=1, variance=Fraction(1, 10)):
    return GaussianState(n, (F(mean),), ((n * F(variance),),))


def _vector(n=80, mean=(1, 1)):
    return GaussianState(n, mean, ((F(n, 10), F(n, 50)), (F(n, 50), F(n, 5))))


def _predictive(state, prior):
    """Independent conjugate formula with even n and factorial gamma ratios."""
    if not state.n:
        return Decimal(0)
    n, dimension = state.n, len(state.mean)
    assert n % 2 == 0 and prior.nu == 4
    posterior = [
        [
            prior.scale[i][j]
            + state.scatter[i][j]
            + prior.kappa
            * n
            / (prior.kappa + n)
            * (state.mean[i] - prior.mean[i])
            * (state.mean[j] - prior.mean[j])
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    if dimension == 1:
        return (
            _decimal(prior.kappa / (prior.kappa + n)).ln() / 2
            + Decimal(factorial(1 + n // 2)).ln()
            + 2 * _decimal(prior.scale[0][0] / 2).ln()
            - (2 + n // 2) * _decimal(posterior[0][0] / 2).ln()
        )

    def determinant(matrix):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Gamma duplication: Γ₂((4+n)/2)/Γ₂(2) * 2**n = (3)_n.
    return (
        _decimal(prior.kappa / (prior.kappa + n)).ln()
        + 2 * _decimal(determinant(prior.scale)).ln()
        - (2 + n // 2) * _decimal(determinant(posterior)).ln()
        + _decimal(F(factorial(n + 2), 2)).ln()
    )


def _add(a, b):
    result = [a[i] if i < len(a) else F(0) for i in range(max(len(a), len(b)))]
    for i, value in enumerate(b):
        result[i] += value
    while result and result[-1] == 0:
        result.pop()
    return result


def _multiply(a, b):
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return _add(result, [])


def _derivative(a):
    return [i * a[i] for i in range(1, len(a))]


def _evaluate(a, x):
    result = Decimal(0)
    for value in reversed(a):
        result = result * x + _decimal(value)
    return result


def _real_roots(coefficients):
    """Independent high-precision derivative subdivision, for these regular fixtures.

    Derivative roots partition the polynomial into monotone intervals. This
    reference uses Decimal sign bisection, not production Sturm/interval code.
    """
    coefficients = _add(coefficients, [])
    if len(coefficients) < 2:
        return []
    if len(coefficients) == 2:
        return [-_decimal(coefficients[0] / coefficients[1])]
    bound = _decimal(2 + max(abs(v / coefficients[-1]) for v in coefficients[:-1]))
    critical = _real_roots(_derivative(coefficients))
    points = [-bound, *(x for x in critical if -bound < x < bound), bound]
    roots = [x for x in critical if abs(_evaluate(coefficients, x)) < Decimal("1e-55")]
    for lo, hi in zip(points, points[1:], strict=False):
        sign = _evaluate(coefficients, lo)
        if sign * _evaluate(coefficients, hi) >= 0:
            continue
        for _ in range(200):
            mid = (lo + hi) / 2
            value = _evaluate(coefficients, mid)
            if value == 0:
                lo = hi = mid
                break
            if sign * value < 0:
                hi = mid
            else:
                lo, sign = mid, value
        roots.append((lo + hi) / 2)
    return sorted(roots)


def _scalar_null(control, treatment, ratio, direction):
    polynomials = [
        [s.scatter[0][0] + s.n * s.mean[0] ** 2, -2 * s.n * s.mean[0] * k, s.n * k * k]
        for s, k in ((control, F(1)), (treatment, ratio))
    ]
    stationary = _add(
        [control.n * v for v in _multiply(_derivative(polynomials[0]), polynomials[1])],
        [treatment.n * v for v in _multiply(_derivative(polynomials[1]), polynomials[0])],
    )

    def log_likelihood(state, mean):
        residual = _decimal(state.scatter[0][0]) + state.n * (mean - _decimal(state.mean[0])) ** 2
        return -Decimal(state.n) / 2 * (1 + (residual / state.n).ln())

    candidates = [
        log_likelihood(control, x) + log_likelihood(treatment, _decimal(ratio) * x)
        for x in [Decimal(0), *_real_roots(stationary)]
        if x >= 0
    ]
    boundary = min(treatment.mean[0], 0) if direction == "greater" else max(treatment.mean[0], 0)
    candidates.append(
        log_likelihood(control, Decimal(0)) + log_likelihood(treatment, _decimal(boundary))
    )
    feasible = treatment.mean[0] <= ratio * control.mean[0]
    if direction == "less":
        feasible = treatment.mean[0] >= ratio * control.mean[0]
    if control.mean[0] >= 0 and feasible:
        candidates.append(sum(log_likelihood(s, _decimal(s.mean[0])) for s in (control, treatment)))
    return max(candidates)


def _bivariate_null(control, treatment, ratio, direction):
    """Independent full face enumeration, differentiating rational determinants."""
    sign = -1 if direction == "greater" else 1
    states = (control, treatment)
    rays = []
    cuts = {F(0)}
    for state, k in zip(states, (F(1), ratio), strict=True):
        a, b = state.mean
        c, d, e = state.scatter[0][0], state.scatter[0][1], state.scatter[1][1]
        determinant = c * e - d * d
        vertex = (e * a * a - 2 * d * a * b + c * b * b) / determinant
        denominator = [c, -2 * d * k, e * k * k]
        numerator = _add(denominator, [state.n * v for v in _multiply([a, -k * b], [a, -k * b])])
        affine = (c * b - d * a, k * (e * a - d * b))
        if affine[1] and -affine[0] / affine[1] > 0:
            cuts.add(-affine[0] / affine[1])
        rays.append((numerator, denominator, affine, vertex))
    a, b = treatment.mean
    if ratio * b and a / (ratio * b) > 0:
        cuts.add(a / (ratio * b))
    cuts = sorted(cuts)
    candidates = []
    for lo, hi in zip(cuts, [*cuts[1:], None], strict=True):
        sample = (lo + hi) / 2 if hi is not None else lo + 1
        rational = []
        for state, (num, den, affine, vertex) in zip(states, rays, strict=True):
            rational.append(
                (num, den)
                if affine[0] + affine[1] * sample > 0
                else ([1 + state.n * vertex], [F(1)])
            )
        c, d, e = treatment.scatter[0][0], treatment.scatter[0][1], treatment.scatter[1][1]
        horizontal = rays[1][3] - max(F(0), sign * (e * a - d * b)) ** 2 / (e * (c * e - d * d))
        treatment_modes = [rational[1], ([1 + treatment.n * horizontal], [F(1)])]
        if b >= 0 and sign * (a - ratio * sample * b) >= 0:
            treatment_modes.append(([F(1)], [F(1)]))
        for treatment_mode in treatment_modes:
            modes = (rational[0], treatment_mode)
            stationary = []
            for i, (num, den) in enumerate(modes):
                derivative = _add(
                    _multiply(_derivative(num), den), [-v for v in _multiply(num, _derivative(den))]
                )
                other_num, other_den = modes[1 - i]
                term = _multiply(derivative, _multiply(other_num, other_den))
                stationary = _add(stationary, [states[i].n * v for v in term])
            points = [_decimal(lo)]
            if hi is not None:
                points.append(_decimal(hi))
            points.extend(
                x
                for x in _real_roots(stationary)
                if x >= _decimal(lo) and (hi is None or x <= _decimal(hi))
            )
            if hi is None:
                points.append(None)
            for point in points:
                value = Decimal(0)
                for state, (num, den) in zip(states, modes, strict=True):
                    num, den = _add(num, []), _add(den, [])
                    multiplier = (
                        _decimal(num[-1] / den[-1])
                        if point is None
                        else _evaluate(num, point) / _evaluate(den, point)
                    )
                    s = state.scatter
                    determinant = s[0][0] * s[1][1] - s[0][1] ** 2
                    value += (
                        -state.n
                        + state.n * Decimal(state.n).ln()
                        - Decimal(state.n) / 2 * (_decimal(determinant) * multiplier).ln()
                    )
                candidates.append(value)
    return max(candidates)


def _assert_reference_brackets(bounds, control, treatment, prior_control, prior_treatment):
    assert bounds.status == "interval"
    with localcontext() as context:
        context.prec = 80
        log_q = _predictive(control, prior_control) + _predictive(treatment, prior_treatment)
        threshold = -_decimal(bounds.alpha).ln()
        oracle = _scalar_null if len(control.mean) == 1 else _bivariate_null
        for direction, endpoint in (
            ("greater", bounds.lower_certificate),
            ("less", bounds.upper_certificate),
        ):
            if bounds.alternative not in ("two-sided", direction):
                continue
            assert endpoint.status == "finite"
            lo, hi = endpoint.bracket
            assert lo is not None and hi is not None and hi - lo <= bounds.max_width
            assert endpoint.rejected is not None and endpoint.accepted is not None
            rejected_log_e = log_q - oracle(control, treatment, endpoint.rejected.ratio, direction)
            accepted_log_e = log_q - oracle(control, treatment, endpoint.accepted.ratio, direction)
            assert rejected_log_e >= threshold - Decimal("1e-35")
            assert accepted_log_e <= threshold + Decimal("1e-35")


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_beta_one_success_has_analytic_ratio_endpoints(alternative):
    state, prior, alpha = BernoulliState(1, 1), BetaPrior(1, 1), F(1, 5)
    bounds = bernoulli_confidence_sequence(
        state, state, prior, prior, alpha=alpha, alternative=alternative
    )
    # Q=1/4; constrained likelihood is r for r<=1, or 1/r for r>=1. Every
    # direction inverts against the same 1/alpha.
    if alternative != "less":
        assert bounds.lower is not None and 0 <= alpha / 4 - bounds.lower <= bounds.max_width
        inside = bounds.lower_certificate.bracket[1]
        assert inside is not None and inside >= alpha / 4
    else:
        assert bounds.lower == 0
    if alternative != "greater":
        assert bounds.upper is not None and 0 <= bounds.upper - 4 / alpha <= bounds.max_width
        inside = bounds.upper_certificate.bracket[0]
        assert inside is not None and inside <= 4 / alpha
    else:
        assert bounds.upper is None and bounds.upper_certificate.status == "unbounded"


@pytest.mark.slow
@pytest.mark.parametrize("width_bits", [20, 96])
def test_exact_threshold_midpoint_keeps_outer_bracket(width_bits):
    state, prior = BernoulliState(1, 1), BetaPrior(1, 1)
    bounds = bernoulli_confidence_sequence(
        state,
        state,
        prior,
        prior,
        alpha=F(1, 2),
        alternative="greater",
        max_width=F(1, 2**width_bits),
    )
    # The crossing r=1/8 is exactly a dyadic midpoint: equality is not guessed.
    # 96 bits exercises the final permitted subdivision rather than an early exit.
    endpoint = bounds.lower_certificate
    assert endpoint.status == "finite"
    lo, hi = endpoint.bracket
    assert lo is not None and hi is not None
    assert lo <= F(1, 8) <= hi
    assert hi - lo <= bounds.max_width


@pytest.mark.slow
def test_beta_interior_probability_optimizer_has_independent_reference():
    control, treatment, prior = BernoulliState(20, 6), BernoulliState(20, 15), BetaPrior(1, 1)
    bounds = bernoulli_confidence_sequence(
        control,
        treatment,
        prior,
        prior,
        alpha=F(1, 10),
        max_width=F(1, 10000),
    )
    assert bounds.lower is not None and bounds.upper is not None
    assert bounds.status == "interval" and bounds.lower < F(5, 2) < bounds.upper
    with localcontext() as context:
        context.prec = 80
        log_q = sum(
            _decimal(
                F(factorial(s.successes) * factorial(s.n - s.successes), factorial(s.n + 1))
            ).ln()
            for s in (control, treatment)
        )

        def reference(ratio, direction):
            r = _decimal(ratio)
            c, t = _decimal(F(6, 20)), _decimal(F(15, 20))
            feasible = t <= r * c if direction == "greater" else t >= r * c
            points = [(c, t)] if feasible else []
            successes = control.successes + treatment.successes
            linear = successes * (1 + r) + r * 5 + 14
            discriminant = linear * linear - 4 * r * 40 * successes
            for u in (
                (linear - discriminant.sqrt()) / (80 * r),
                (linear + discriminant.sqrt()) / (80 * r),
            ):
                if 0 < u < 1 and 0 < r * u < 1:
                    points.append((u, r * u))
            log_likelihood = max(
                6 * pc.ln() + 14 * (1 - pc).ln() + 15 * pt.ln() + 5 * (1 - pt).ln()
                for pc, pt in points
            )
            return log_q - log_likelihood

        for direction, endpoint in (
            ("greater", bounds.lower_certificate),
            ("less", bounds.upper_certificate),
        ):
            assert endpoint.status == "finite"
            assert endpoint.rejected is not None and endpoint.accepted is not None
            assert reference(endpoint.rejected.ratio, direction) >= Decimal(10).ln() - Decimal(
                "1e-35"
            )
            assert reference(endpoint.accepted.ratio, direction) <= Decimal(10).ln() + Decimal(
                "1e-35"
            )


@pytest.mark.parametrize(
    "control,treatment",
    [(BernoulliState(0, 0), BernoulliState(0, 0)), (BernoulliState(4, 0), BernoulliState(9, 0))],
)
def test_beta_empty_and_zero_event_prefixes_are_full_domain(control, treatment):
    bounds = bernoulli_confidence_sequence(
        control, treatment, BetaPrior(1, 1), BetaPrior(1, 1), alpha=F(1, 20)
    )
    assert bounds.status == "full-domain" and not bounds.empty
    assert bounds.lower == 0 and bounds.upper is None
    assert bounds.upper_certificate.status == "unbounded"


@pytest.mark.slow
@pytest.mark.parametrize(
    "control,treatment",
    [(BernoulliState(1, 1), BernoulliState(1, 0)), (BernoulliState(0, 0), BernoulliState(1, 0))],
)
def test_beta_zero_treatment_keeps_zero_and_finite_upper_when_identified(control, treatment):
    prior = BetaPrior(1, 1)
    bounds = bernoulli_confidence_sequence(control, treatment, prior, prior, alpha=F(1, 5))
    assert bounds.lower == 0
    if control.n:
        # Q=1/4, sup L=1/(4r) for r>=1/2, hence Eless=r, crossing 1/alpha at 5.
        assert bounds.upper is not None and 0 <= bounds.upper - 5 <= bounds.max_width
    else:
        assert bounds.status == "full-domain" and bounds.upper is None


@pytest.mark.slow
def test_zero_control_has_mathematically_unbounded_upper():
    bounds = bernoulli_confidence_sequence(
        BernoulliState(20, 0),
        BernoulliState(20, 20),
        BetaPrior(1, 1),
        BetaPrior(1, 1),
        alpha=F(1, 20),
    )
    assert bounds.lower is not None and bounds.lower > 1
    assert bounds.upper is None and bounds.upper_certificate.status == "unbounded"
    assert bounds.upper_certificate.reason is not None


@pytest.mark.slow
def test_all_success_prefix_contracts_around_one():
    prior = BetaPrior(1, 1)
    small = bernoulli_confidence_sequence(
        BernoulliState(4, 4), BernoulliState(4, 4), prior, prior, alpha=F(1, 10)
    )
    large = bernoulli_confidence_sequence(
        BernoulliState(40, 40), BernoulliState(40, 40), prior, prior, alpha=F(1, 10)
    )
    assert small.lower is not None and small.upper is not None
    assert large.lower is not None and large.upper is not None
    assert small.lower < large.lower < 1 < large.upper < small.upper
    with localcontext() as context:
        context.prec = 80
        # L(r)=r**40 below one; Q=1/41**2 for uniform priors.
        root = (_decimal(F(1, 10 * 41**2)).ln() / 40).exp()
        assert _decimal(large.lower) <= root <= _decimal(large.lower + large.max_width)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_gaussian_bounds_match_independent_high_precision_profiles(dimension):
    control = _scalar() if dimension == 1 else _vector()
    treatment = _scalar(mean=2) if dimension == 1 else _vector(mean=(2, 1))
    pc, pt = _prior(dimension, control.mean), _prior(dimension, treatment.mean)
    bounds = gaussian_confidence_sequence(
        control, treatment, pc, pt, alpha=F(1, 10), max_width=F(1, 10000)
    )
    assert bounds.lower is not None and bounds.upper is not None
    assert bounds.lower < 2 < bounds.upper
    _assert_reference_brackets(bounds, control, treatment, pc, pt)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_negative_gaussian_ratios_have_finite_negative_bounds(dimension):
    control = _scalar() if dimension == 1 else _vector()
    treatment = _scalar(mean=-2) if dimension == 1 else _vector(mean=(-2, 1))
    pc, pt = _prior(dimension, control.mean), _prior(dimension, treatment.mean)
    bounds = gaussian_confidence_sequence(
        control, treatment, pc, pt, alpha=F(1, 10), max_width=F(1, 1000)
    )
    assert bounds.lower is not None and bounds.upper is not None
    assert bounds.lower < -2 < bounds.upper < 0
    _assert_reference_brackets(bounds, control, treatment, pc, pt)


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_gaussian_directional_tail_matches_central_alpha(dimension):
    c = _scalar() if dimension == 1 else _vector()
    t = _scalar(mean=2) if dimension == 1 else _vector(mean=(2, 1))
    pc, pt = _prior(dimension, c.mean), _prior(dimension, t.mean)
    central = gaussian_confidence_sequence(c, t, pc, pt, alpha=F(1, 10), max_width=F(1, 1000))
    lower = gaussian_confidence_sequence(
        c, t, pc, pt, alpha=F(1, 10), alternative="greater", max_width=F(1, 1000)
    )
    upper = gaussian_confidence_sequence(
        c, t, pc, pt, alpha=F(1, 10), alternative="less", max_width=F(1, 1000)
    )
    assert central.lower is not None and central.upper is not None
    assert lower.lower is not None and upper.upper is not None
    assert abs(lower.lower - central.lower) <= central.max_width
    assert abs(upper.upper - central.upper) <= central.max_width
    assert lower.upper is None and upper.lower is None
    halved = gaussian_confidence_sequence(c, t, pc, pt, alpha=F(1, 20), max_width=F(1, 1000))
    assert halved.lower is not None and halved.lower < central.lower


@pytest.mark.parametrize("dimension", [1, 2])
def test_empty_gaussian_prefix_has_full_real_domain(dimension):
    empty, prior = GaussianState.empty(dimension), _prior(dimension)
    bounds = gaussian_confidence_sequence(empty, empty, prior, prior, alpha=F(1, 20))
    assert bounds.status == "full-domain"
    assert bounds.lower is None and bounds.upper is None and not bounds.empty


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_gaussian_zero_control_mean_has_proved_one_infinite_endpoint(dimension, sign):
    c = _scalar(mean=0) if dimension == 1 else _vector(mean=(0, 1))
    t = _scalar(mean=2 * sign) if dimension == 1 else _vector(mean=(2 * sign, 1))
    bounds = gaussian_confidence_sequence(
        c,
        t,
        _prior(dimension, c.mean),
        _prior(dimension, t.mean),
        alpha=F(1, 10),
        max_width=F(1, 1000),
    )
    assert bounds.status == "interval" and not bounds.empty
    if sign > 0:
        assert bounds.lower is not None and bounds.lower > 0 and bounds.upper is None
        infinite = bounds.upper_certificate
    else:
        assert bounds.lower is None and bounds.upper is not None and bounds.upper < 0
        infinite = bounds.lower_certificate
    # The control MLE is on the common zero-numerator face. In this direction
    # the treatment MLE is also feasible for every r, so Q/sup L <= 1.
    assert infinite.status == "unbounded"
    assert infinite.log_e_upper is not None and infinite.log_e_upper < bounds.log_threshold.lo


@pytest.mark.slow
@pytest.mark.parametrize("sign,alternative", [(1, "less"), (-1, "greater")])
def test_zero_treatment_denominator_certifies_signed_unbounded_tail(sign, alternative):
    control, treatment = _vector(), _vector(mean=(2 * sign, 0))
    bounds = gaussian_confidence_sequence(
        control,
        treatment,
        _prior(2, control.mean),
        _prior(2, treatment.mean),
        alpha=F(1, 10),
        alternative=alternative,
    )
    # A vanishing positive treatment denominator puts its unconstrained MLE
    # in every requested null closure while leaving the control mean unrestricted.
    endpoint = bounds.upper_certificate if sign > 0 else bounds.lower_certificate
    assert endpoint.status == "unbounded"
    assert endpoint.log_e_upper is not None and endpoint.log_e_upper < bounds.log_threshold.lo
    assert (bounds.upper if sign > 0 else bounds.lower) is None


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_one_empty_gaussian_arm_preserves_sign_information_without_point_interval(dimension):
    c = GaussianState.empty(dimension)
    t = _scalar(mean=2) if dimension == 1 else _vector(mean=(2, 1))
    bounds = gaussian_confidence_sequence(
        c,
        t,
        _prior(dimension),
        _prior(dimension, t.mean),
        alpha=F(1, 10),
        max_width=F(1, 1000),
    )
    assert bounds.status == "interval" and not bounds.empty
    assert bounds.lower == 0 and bounds.upper is None
    assert bounds.lower_certificate.status == "finite"
    assert bounds.upper_certificate.status == "unbounded"


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_singular_prefix_abstention_is_temporary(dimension):
    c = GaussianState.from_rows([(1,) * dimension], dimension=dimension)
    t = GaussianState.from_rows([(2,) * dimension], dimension=dimension)
    prior = _prior(dimension)
    early = gaussian_confidence_sequence(c, t, prior, prior, alpha=F(1, 10))
    assert early.status == "abstained" and early.lower is None and early.upper is None
    c = c.merge(_scalar() if dimension == 1 else _vector())
    t = t.merge(_scalar(mean=2) if dimension == 1 else _vector(mean=(2, 1)))
    later = gaussian_confidence_sequence(c, t, prior, prior, alpha=F(1, 10), max_width=F(1, 1000))
    assert later.status == "interval"
    assert later.lower is not None and later.lower > 0 and later.upper is not None


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
def test_extreme_alpha_keeps_logarithmic_full_domain_proof(dimension):
    c = _scalar() if dimension == 1 else _vector()
    t = _scalar(mean=2) if dimension == 1 else _vector(mean=(2, 1))
    prior = _prior(dimension)
    alpha = F(1, 10**1000)
    bounds = gaussian_confidence_sequence(c, t, prior, prior, alpha=alpha)
    assert bounds.alpha == alpha and bounds.status == "full-domain"
    for endpoint in (bounds.lower_certificate, bounds.upper_certificate):
        assert endpoint.status == "unbounded"
        assert endpoint.log_e_upper is not None and endpoint.log_e_upper < bounds.log_threshold.lo
    with localcontext() as context:
        context.prec = 90
        reference = Decimal(1000) * Decimal(10).ln()
        assert bounds.log_threshold.lo <= F(reference) <= bounds.log_threshold.hi


@pytest.mark.slow
def test_extreme_beta_alpha_retains_unresolved_infinity_and_safe_outer_lower():
    state, prior = BernoulliState(1, 1), BetaPrior(1, 1)
    alpha = F(1, 10**400)
    bounds = bernoulli_confidence_sequence(state, state, prior, prior, alpha=alpha)
    assert bounds.lower is not None and bounds.lower <= alpha / 8
    assert bounds.upper is None and bounds.upper_certificate.status == "unresolved"
    assert bounds.upper_certificate.reason is not None
    assert bounds.status == "unresolved" and not bounds.empty


@pytest.mark.slow
def test_unattainable_width_reports_outward_resolution_bracket():
    state, prior = BernoulliState(2, 2), BetaPrior(1, 1)
    bounds = bernoulli_confidence_sequence(
        state,
        state,
        prior,
        prior,
        alpha=F(1, 5),
        alternative="greater",
        max_width=F(1, 10**100),
    )
    assert bounds.status == "unresolved"
    endpoint = bounds.lower_certificate
    lo, hi = endpoint.bracket
    assert lo is not None and hi is not None and hi - lo > bounds.max_width
    # Q=1/9 and Egreater=1/(9*r**2), so crossing is sqrt(alpha/9).
    assert lo * lo <= F(1, 45) <= hi * hi
    assert endpoint.reason is not None


@pytest.mark.slow
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_observations_outside_positive_control_domain_can_prove_empty(dimension, alternative):
    c = _scalar(mean=-10) if dimension == 1 else _vector(mean=(-10, 1))
    t = _scalar(mean=1) if dimension == 1 else _vector()
    pc, pt = _prior(dimension, c.mean), _prior(dimension, t.mean)
    bounds = gaussian_confidence_sequence(c, t, pc, pt, alpha=F(1, 10), alternative=alternative)
    # The unrestricted proper mixture may predict negative control observations
    # far better than every mean in the declared positive control domain.
    assert bounds.empty and bounds.status == "empty"
    assert bounds.domain_log_e is not None and bounds.domain_log_e.lo >= bounds.log_threshold.hi


@pytest.mark.slow
@pytest.mark.parametrize(
    "control_mean,treatment_mean,control_projection,treatment_projection",
    [
        ((-1, 2), (1, 2), (0, F(5, 2)), (1, 2)),
        ((2, -1), (1, 2), (F(7, 3), 0), (1, 2)),
        ((-2, -1), (1, 2), (0, 0), (1, 2)),
        ((1, 2), (2, -1), (1, 2), (F(3, 2), 0)),
    ],
)
def test_bivariate_domain_certificate_matches_exact_projection_coordinates(
    control_mean,
    treatment_mean,
    control_projection,
    treatment_projection,
):
    control = GaussianState(80, control_mean, ((2, 1), (1, 3)))
    treatment = GaussianState(100, treatment_mean, ((5, -1), (-1, 2)))
    pc, pt = _prior(2, control.mean), _prior(2, treatment.mean)
    bounds = gaussian_confidence_sequence(control, treatment, pc, pt, alpha=F(1, 10))
    assert bounds.empty
    assert bounds.domain_log_e is not None
    with localcontext() as context:
        context.prec = 90

        def likelihood_at(state, projection):
            residual = [_decimal(m - x) for m, x in zip(state.mean, projection, strict=True)]
            matrix = [
                [
                    _decimal(state.scatter[i][j]) + state.n * residual[i] * residual[j]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
            return (
                -state.n + state.n * Decimal(state.n).ln() - Decimal(state.n) * determinant.ln() / 2
            )

        reference = (
            _predictive(control, pc)
            + _predictive(treatment, pt)
            - likelihood_at(control, control_projection)
            - likelihood_at(treatment, treatment_projection)
        )
        assert bounds.domain_log_e.lo <= F(reference) <= bounds.domain_log_e.hi


@pytest.mark.slow
def test_finite_gaussian_power_and_width_contraction():
    prior = _prior()
    small = gaussian_confidence_sequence(
        _scalar(n=40), _scalar(n=40, mean=2), prior, prior, alpha=F(1, 10), max_width=F(1, 1000)
    )
    large = gaussian_confidence_sequence(
        _scalar(n=200), _scalar(n=200, mean=2), prior, prior, alpha=F(1, 10), max_width=F(1, 1000)
    )
    assert small.lower is not None and small.upper is not None
    assert large.lower is not None and large.upper is not None
    assert 1 < large.lower < 2 < large.upper
    assert small.lower < large.lower and large.upper < small.upper


@pytest.mark.slow
def test_equivalent_gaussian_input_partitions_have_identical_certificates():
    rows = [(F(3, 4), F(7, 8)), (F(5, 4), F(7, 8)), (F(1), F(5, 4))] * 20
    whole = GaussianState.from_rows(rows, dimension=2)
    merged = GaussianState.from_rows(rows[::2], dimension=2).merge(
        GaussianState.from_rows(rows[1::2], dimension=2)
    )
    reordered = GaussianState.from_rows(list(reversed(rows)), dimension=2)
    treatment = _vector(n=60, mean=(2, 1))
    prior = _prior(2)
    results = [
        gaussian_confidence_sequence(
            c, treatment, prior, prior, alpha=F(1, 10), max_width=F(1, 1000)
        )
        for c in (whole, merged, reordered)
    ]
    assert results[0] == results[1] == results[2]


@pytest.mark.slow
def test_equivalent_bernoulli_partitions_have_identical_certificates():
    parts = (BernoulliState(7, 2), BernoulliState(13, 4))
    c = BernoulliState(sum(p.n for p in parts), sum(p.successes for p in parts))
    t, prior = BernoulliState(20, 15), BetaPrior(1, 1)
    assert bernoulli_confidence_sequence(
        c, t, prior, prior, alpha=F(1, 10), max_width=F(1, 1000)
    ) == bernoulli_confidence_sequence(
        BernoulliState(20, 6), t, prior, prior, alpha=F(1, 10), max_width=F(1, 1000)
    )


def test_endpoint_certificate_copies_a_caller_owned_bracket():
    """The constructor stores its own tuple, not the caller's mutable list."""
    supplied = [F(0), F(1)]
    endpoint = EndpointCertificate(
        "unresolved",
        cast(tuple[Fraction | None, Fraction | None], supplied),
        reason="explicit outer bracket",
    )
    supplied[0] = F(-99)
    assert endpoint.bracket == (F(0), F(1))


@pytest.mark.parametrize(
    "function,states,prior",
    [
        (bernoulli_confidence_sequence, (BernoulliState(0, 0),) * 2, BetaPrior(1, 1)),
        (gaussian_confidence_sequence, (GaussianState.empty(1),) * 2, _prior()),
    ],
)
@pytest.mark.parametrize(
    "kwargs,code,context",
    [
        ({"alpha": F(0)}, "invalid_alpha", {"alpha": F(0)}),
        ({"alpha": F(1)}, "invalid_alpha", {"alpha": F(1)}),
        ({"alpha": F(-1)}, "invalid_alpha", {"alpha": F(-1)}),
        ({"alpha": 0.05}, "certified.inexact_value", {"value_type": "float"}),
        ({"alpha": True}, "certified.inexact_value", {"value_type": "bool"}),
        (
            {"alpha": F(1, 10), "max_width": F(0)},
            "certified.nonpositive_root_width",
            {"max_width": F(0)},
        ),
        (
            {"alpha": F(1, 10), "max_width": -1},
            "certified.nonpositive_root_width",
            {"max_width": F(-1)},
        ),
        (
            {"alpha": F(1, 10), "max_width": 0.01},
            "certified.inexact_value",
            {"value_type": "float"},
        ),
        (
            {"alpha": F(1, 10), "max_width": True},
            "certified.inexact_value",
            {"value_type": "bool"},
        ),
        (
            {"alpha": F(1, 10), "alternative": "invalid"},
            "sequential_likelihood.unknown_alternative",
            {"alternative": "invalid"},
        ),
    ],
)
def test_invalid_arguments_are_rejected_even_on_empty_prefix(
    function, states, prior, kwargs, code, context
):
    with _refuses(code, **context):
        function(*states, prior, prior, **kwargs)


def test_gaussian_dimension_mismatch_is_rejected_on_empty_prefix():
    with _refuses(
        "sequential_likelihood.state_prior_dimension_mismatch",
        control_dimension=1,
        treatment_dimension=2,
        prior_control_dimension=1,
        prior_treatment_dimension=2,
    ):
        gaussian_confidence_sequence(
            GaussianState.empty(1), GaussianState.empty(2), _prior(), _prior(2), alpha=F(1, 10)
        )


@pytest.mark.slow
def test_near_one_alpha_is_not_recovered_from_rounded_confidence():
    state, prior = BernoulliState(1, 1), BetaPrior(1, 1)
    alpha = F(2**1200 - 1, 2**1200)
    bounds = bernoulli_confidence_sequence(
        state, state, prior, prior, alpha=alpha, alternative="greater"
    )
    assert bounds.alpha == alpha and bounds.status == "interval"
    lo, hi = bounds.lower_certificate.bracket
    assert bounds.lower_certificate.status == "finite"
    assert lo is not None and hi is not None and lo <= alpha / 4 <= hi
    assert hi - lo <= bounds.max_width
    assert 0 < bounds.log_threshold.lo <= bounds.log_threshold.hi < F(1, 2**1199)


def test_metadata_rejects_invalid_field_combinations():
    with _refuses("incomplete_finite_endpoint", bracket=(F(0), F(1)), rejected=None, accepted=None):
        EndpointCertificate("finite", (F(0), F(1)))
    with _refuses("reversed_bracket", lower=F(1), upper=F(0)):
        EndpointCertificate("unresolved", (F(1), F(0)), reason="invalid")
    endpoint = EndpointCertificate("unbounded", (None, None), reason="domain endpoint")
    negative = EndpointCertificate("unresolved", (F(-1), F(0)), reason="outside domain")
    with _refuses("negative_bernoulli_bound", lower=F(-1)):
        ConfidenceBounds(
            F(-1),
            None,
            negative,
            endpoint,
            "nonnegative",
            F(1, 10),
            "two-sided",
            F(1, 100),
            Interval(1, 3),
        )


def _evaluation(ratio, log_e):
    zero, evidence = Interval.exact(0), Interval.exact(log_e)
    return EndpointEvaluation(
        F(ratio), LikelihoodCertificate("finite", evidence, zero, evidence, None)
    )


@pytest.mark.parametrize(
    "kwargs,code,context",
    [
        (
            {"status": "invalid", "bracket": (None, None)},
            "unknown_endpoint_status",
            {"status": "invalid"},
        ),
        (
            {"status": "unresolved", "bracket": (F(0),), "reason": "outer"},
            "invalid_bracket_length",
            {"bracket": (F(0),)},
        ),
        (
            {"status": "unresolved", "bracket": (F(0), F(1)), "rejected": F(0)},
            "invalid_witness_type",
            {"witness": "rejected", "value_type": "Fraction"},
        ),
        (
            {"status": "unresolved", "bracket": (None, None), "reason": ""},
            "sequential_likelihood.invalid_certificate_reason",
            {"reason": ""},
        ),
        (
            {"status": "unresolved", "bracket": (None, None)},
            "missing_endpoint_reason",
            {"status": "unresolved"},
        ),
    ],
)
def test_endpoint_metadata_refusals_expose_condition_and_context(kwargs, code, context):
    with _refuses(code, **context):
        EndpointCertificate(**kwargs)


def test_endpoint_evaluation_requires_a_likelihood_certificate():
    with _refuses("invalid_evidence_type", value_type="NoneType"):
        EndpointEvaluation(F(0), cast(LikelihoodCertificate, None))


@pytest.mark.parametrize("witness", ["rejected", "accepted"])
def test_outside_witness_context_snapshots_mutable_bracket(witness):
    bracket = [F(0), F(1)]
    witnesses: dict[str, Any] = {witness: _evaluation(2, 0)}
    with pytest.raises(InvalidRequestError) as raised:
        EndpointCertificate(
            "unresolved",
            cast(tuple[Fraction | None, Fraction | None], bracket),
            reason="outer",
            **witnesses,
        )
    bracket[0] = F(-1)
    assert raised.value.code == "estimation.sequential_inversion.witness_outside_bracket"
    context = raised.value.context
    assert context["witness"] == witness
    assert context["ratio"] == F(2)
    assert cast(tuple[Fraction | None, Fraction | None], context["bracket"]) == (F(0), F(1))
    with pytest.raises(TypeError):
        cast(dict[str, object], context)["ratio"] = F(0)
    with pytest.raises(TypeError):
        cast(list[Fraction], context["bracket"])[0] = F(0)


@pytest.mark.parametrize(
    "change,code,context",
    [
        ({"domain": "invalid"}, "unknown_domain", {"domain": "invalid"}),
        ({"log_threshold": F(1)}, "invalid_threshold_type", {"value_type": "Fraction"}),
        (
            {"domain_log_e": F(1)},
            "invalid_domain_evidence_type",
            {"value_type": "Fraction"},
        ),
        (
            {"lower_certificate": None},
            "invalid_endpoint_type",
            {"side": "lower", "value_type": "NoneType"},
        ),
        (
            {"max_width": F(1, 2)},
            "endpoint_width_exceeded",
            {"side": "lower", "bracket": (F(0), F(1)), "max_width": F(1, 2)},
        ),
        (
            {"lower": F(1)},
            "bound_coordinate_mismatch",
            {"side": "lower", "bound": F(1), "expected": F(0)},
        ),
    ],
)
def test_confidence_metadata_refusals_expose_condition_and_context(change, code, context):
    args: dict[str, Any] = {
        "lower": F(0),
        "upper": None,
        "lower_certificate": EndpointCertificate(
            "finite", (F(0), F(1)), _evaluation(0, 2), _evaluation(1, 0)
        ),
        "upper_certificate": EndpointCertificate("unbounded", (None, None), reason="upper tail"),
        "domain": "real",
        "alpha": F(1, 10),
        "alternative": "greater",
        "max_width": F(1),
        "log_threshold": Interval.exact(1),
    }
    args.update(change)
    with _refuses(code, **context):
        ConfidenceBounds(**args)


@pytest.mark.parametrize(
    "rejected_ratio,rejected_log,accepted_ratio,accepted_log,code",
    [
        (0, 0, 1, 0, "uncertified_rejection"),
        (1, 2, 1, 0, "rejected_coordinate_mismatch"),
        (0, 2, 1, 2, "uncertified_nonrejection"),
        (0, 2, 0, 0, "accepted_coordinate_mismatch"),
    ],
)
def test_confidence_bounds_distinguish_unproved_witnesses_from_wrong_coordinates(
    rejected_ratio, rejected_log, accepted_ratio, accepted_log, code
):
    rejected = _evaluation(rejected_ratio, rejected_log)
    accepted = _evaluation(accepted_ratio, accepted_log)
    endpoint = EndpointCertificate("finite", (F(0), F(1)), rejected, accepted)
    threshold = Interval.exact(1)
    contexts = {
        "uncertified_rejection": {"evidence": rejected.evidence, "threshold": threshold},
        "rejected_coordinate_mismatch": {"ratio": F(1), "expected": F(0)},
        "uncertified_nonrejection": {"evidence": accepted.evidence, "threshold": threshold},
        "accepted_coordinate_mismatch": {"ratio": F(0), "expected": F(1)},
    }
    with _refuses(code, side="lower", **contexts[code]):
        ConfidenceBounds(
            F(0),
            None,
            endpoint,
            EndpointCertificate("unbounded", (None, None), reason="upper tail"),
            "real",
            F(1, 10),
            "greater",
            F(1),
            threshold,
        )


def test_uniform_bound_at_threshold_does_not_certify_nonrejection():
    threshold = Interval.exact(1)
    endpoint = EndpointCertificate("unbounded", (None, None), reason="uniform", log_e_upper=F(1))
    with _refuses("uncertified_uniform_bound", side="lower", log_e_upper=F(1), threshold=threshold):
        ConfidenceBounds(
            None, None, endpoint, endpoint, "real", F(1, 10), "two-sided", F(1), threshold
        )


def test_empty_endpoint_requires_rejection_evidence():
    threshold = Interval.exact(1)
    endpoint = EndpointCertificate("empty", (None, None), reason="unsupported claim")
    with _refuses(
        "uncertified_empty_endpoint", side="lower", domain_log_e=None, threshold=threshold
    ):
        ConfidenceBounds(
            None, None, endpoint, endpoint, "real", F(1, 10), "two-sided", F(1), threshold
        )


def test_reversed_outer_bounds_require_certified_emptiness():
    lower = EndpointCertificate("unresolved", (F(2), None), reason="outer")
    upper = EndpointCertificate("unresolved", (None, F(1)), reason="outer")
    with _refuses("uncertified_reversed_bounds", lower=F(2), upper=F(1)):
        ConfidenceBounds(
            F(2), F(1), lower, upper, "real", F(1, 10), "two-sided", F(1), Interval.exact(1)
        )


def test_sequential_inversion_refusals_name_the_offending_value() -> None:
    with _refuses("invalid_evidence_type", value_type="NoneType"):
        EndpointEvaluation(F(0), cast(LikelihoodCertificate, None))

    with _refuses("invalid_witness_type", witness="rejected", value_type="Fraction"):
        EndpointCertificate(
            "unresolved", (F(0), F(1)), reason="outer", rejected=cast(EndpointEvaluation, F(0))
        )

    args: dict[str, Any] = {
        "lower": F(0),
        "upper": None,
        "lower_certificate": EndpointCertificate(
            "finite", (F(0), F(1)), _evaluation(0, 2), _evaluation(1, 0)
        ),
        "upper_certificate": EndpointCertificate("unbounded", (None, None), reason="upper tail"),
        "domain": "real",
        "alpha": F(1, 10),
        "alternative": "greater",
        "max_width": F(1),
        "log_threshold": Interval.exact(1),
    }
    with _refuses("invalid_threshold_type", value_type="Fraction"):
        ConfidenceBounds(**{**args, "log_threshold": F(1)})  # ty: ignore[invalid-argument-type]
    with _refuses("invalid_endpoint_type", side="lower", value_type="NoneType"):
        ConfidenceBounds(**{**args, "lower_certificate": None})  # ty: ignore[invalid-argument-type]


def test_sequential_inversion_dimension_mismatch_names_every_dimension() -> None:
    prior = _prior()
    with pytest.raises(InvalidRequestError) as exc:
        gaussian_confidence_sequence(
            GaussianState.empty(1), GaussianState.empty(2), prior, _prior(2), alpha=F(1, 10)
        )
    assert exc.value.code == "estimation.sequential_likelihood.state_prior_dimension_mismatch"
    context = exc.value.context
    assert context["control_dimension"] == 1
    assert context["treatment_dimension"] == 2
    assert context["prior_control_dimension"] == 1
    assert context["prior_treatment_dimension"] == 2
