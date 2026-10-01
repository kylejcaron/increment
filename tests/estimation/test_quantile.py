"""Order-statistic quantile point + SE - pure array math, no frames."""

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, localcontext
from fractions import Fraction
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from increment.decision import ArmHypothesisKey
from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.estimation.quantile import (
    _bracket_ranks,
    _quantile_p_value,
    estimate_quantile_lift,
    estimate_quantile_lift_computation,
    log_quantile_se,
)
from increment.semantics.models import QuantileMetric
from increment.sources import SourceOperation


def test_point_is_the_raw_sample_quantile():
    """The point is returned untransformed: the caller forms the joint log
    ratio from two raw quantiles, never from two separately rounded logs."""
    rng = np.random.default_rng(7)
    y = rng.lognormal(mean=1.0, sigma=0.8, size=500)
    point, _ = log_quantile_se(y, q=0.5)
    assert point == float(np.quantile(y, 0.5))


@pytest.mark.parametrize("n,q", [(400, 0.9), (100, 0.9)])
def test_se_matches_standard_order_stat_bracket(n, q):
    """Independent reference: the textbook distribution-free bracket
    [Y_(a), Y_(b)] with 1-based ranks a = ppf(alpha/2), b = ppf(1-alpha/2)+1
    has exact coverage cdf(b-1) - cdf(a-1) >= 1-alpha. The Woodruff SE must
    invert exactly that bracket (0-based indices a-1 and b-1); using index a
    for the lower bound subtracts one binomial term too many (e.g. 0.9364
    instead of 0.9557 at n=100, q=0.9). Both cases here have a bracket wide
    enough (bracket size >= the calibrated HI cutoff at alpha=0.05, ~11.8
    distinct order statistics) that the resolution buffer is exactly
    zero -- narrow-bracket cases are covered by
    test_buffer_is_exactly_zero_once_the_bracket_spans_sixteen_distinct_values
    below, which pins the buffered formula instead."""
    rng = np.random.default_rng(7)
    alpha = 0.05
    y = np.sort(rng.lognormal(mean=1.0, sigma=0.8, size=n))
    a = int(binom.ppf(alpha / 2, n, q))  # 1-based lower rank
    b = int(binom.ppf(1 - alpha / 2, n, q)) + 1  # 1-based upper rank
    exact_coverage = binom.cdf(b - 1, n, q) - binom.cdf(a - 1, n, q)
    assert exact_coverage >= 1 - alpha  # the reference recipe itself is honest
    z = norm.ppf(1 - alpha / 2)
    want_se = (math.log(y[b - 1]) - math.log(y[a - 1])) / (2 * z)
    _, got_se = log_quantile_se(y, q=q, alpha=alpha)
    assert math.isclose(got_se, want_se, rel_tol=1e-12)


def test_quantile_bracket_resolves_sub_complement_precision_alpha():
    """Complementary lower-tail inversion keeps a resolvable tiny tail finite."""
    ranks = _bracket_ranks(1000, 0.5, 1e-20)
    assert ranks == (353, 648)
    y = np.linspace(1.0, 2.0, 1000)
    point, se = log_quantile_se(y, q=0.5, alpha=1e-20)
    assert math.isfinite(point) and math.isfinite(se) and se > 0.0


def test_quantile_bracket_refuses_genuinely_insufficient_n():
    """The analytic floor remains a sample-size refusal, not a tail failure."""
    ranks = _bracket_ranks(60, 0.5, 1e-20)
    assert ranks is None
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.linspace(1.0, 2.0, 60), q=0.5, alpha=1e-20)
    assert exc_info.value.code == "estimation.quantile.too_small_bound"
    assert exc_info.value.context["n_min"] == 68


@pytest.mark.parametrize("q", [0.1, 0.9])
def test_quantile_bracket_complementary_q_symmetry_and_tail_inequalities(q):
    """Asymmetric laws retain both direct tail inequalities and symmetry."""
    n, alpha = 1000, 0.05
    ranks = _bracket_ranks(n, q, alpha)
    mirror = _bracket_ranks(n, 1.0 - q, alpha)
    assert ranks is not None and mirror is not None
    assert mirror == (n - ranks[1] + 1, n - ranks[0] + 1)
    a, b = ranks
    tail = alpha / 2.0
    assert binom.cdf(a - 1, n, q) <= tail <= binom.cdf(a, n, q)
    assert binom.sf(b - 1, n, q) <= tail <= binom.sf(b - 2, n, q)


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
def test_quantile_bracket_includes_exact_tail_feasibility_boundary(q):
    from fractions import Fraction

    n = 20
    alpha = 2 * max(q, 1 - q) ** n
    ranks = _bracket_ranks(n, q, alpha)
    assert ranks is not None
    a, b = ranks
    if q <= 0.5:
        assert a == 1
    if q >= 0.5:
        assert b == n
    p = Fraction(q)
    tail = Fraction(alpha) / 2
    lower = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(a))
    upper = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(b, n + 1))
    assert lower <= tail and upper <= tail
    point, se = log_quantile_se(np.linspace(1.0, 2.0, n), q=q, alpha=alpha)
    assert point > 0 and math.isfinite(se) and se > 0


@pytest.mark.parametrize("q", [0.3, 0.7])
def test_quantile_endpoint_rounding_does_not_spend_extra_tail_probability(q):
    from fractions import Fraction

    from increment.estimation.quantile import _quantile_n_min

    n = 20
    alpha = 2 * max(q, 1 - q) ** n
    p = Fraction(q)
    assert max(p, 1 - p) ** n > Fraction(alpha) / 2
    assert _bracket_ranks(n, q, alpha) is None
    assert _quantile_n_min(q, alpha) == n + 1
    assert _bracket_ranks(n + 1, q, alpha) is not None
    with pytest.raises(InvalidRequestError) as rejected:
        log_quantile_se(np.linspace(1.0, 2.0, n), q=q, alpha=alpha)
    assert rejected.value.code == "estimation.quantile.too_small_bound"
    assert rejected.value.context["n_min"] == n + 1


@pytest.mark.parametrize(
    ("q", "alpha"),
    [
        (1e-20, 0.05),
        (np.nextafter(0.0, 1.0), 0.05),
        (np.nextafter(1.0, 0.0), 0.05),
        (0.3, 2 * 0.7**20),
    ],
)
def test_quantile_n_min_extreme_seed_meets_directed_endpoint_inequalities(q, alpha):
    """The adaptive Decimal seed remains exact at complement and subnormal edges."""
    from increment.estimation.quantile import _quantile_n_min

    n_min = _quantile_n_min(q, alpha)
    tail = Decimal.from_float(alpha / 2.0)
    lower = Context(prec=1600, rounding=ROUND_FLOOR)
    upper = Context(prec=1600, rounding=ROUND_CEILING)
    rate = Decimal.from_float(float(q))
    base = rate if q >= 0.5 else lower.subtract(Decimal(1), rate)
    mass = upper.power(base, n_min)
    previous = lower.power(base, n_min - 1)
    assert mass <= tail
    assert previous > tail


def test_quantile_n_min_preserves_known_extreme_floor():
    from increment.estimation.quantile import _quantile_n_min

    assert _quantile_n_min(1e-20, 0.05) == 368887945411393644965
    assert _quantile_n_min(np.nextafter(1.0, 0.0), 0.05) == 33226472269924403


def test_quantile_runtime_refuses_tiny_quantile_without_allocating_floor():
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.linspace(1.0, 2.0, 100), q=1e-20)
    assert exc_info.value.code == "estimation.quantile.too_small_bound"
    assert exc_info.value.context["n_min"] == 368887945411393644965


@pytest.mark.parametrize(
    "n,q,alpha",
    [(9, 0.3, 0.392006468), (20, 0.1, 0.6461463896210679)],
)
def test_quantile_interior_boundary_respects_exact_tail_budget(n, q, alpha):
    ranks = _bracket_ranks(n, q, alpha)
    assert ranks is not None
    a, b = ranks
    p, tail = Fraction(q), Fraction(alpha) / 2
    lower = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(a))
    upper = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(b, n + 1))
    assert lower <= tail and upper <= tail


def test_quantile_exact_interior_tail_equality_keeps_tight_bracket():
    assert _bracket_ranks(4, 0.5, 0.625) == (2, 3)


def _decimal_lower_tail(k: int, n: int, p: Decimal) -> Decimal:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)`` with ``k`` below the mode, summed
    downward from ``k`` at 60 digits until the (decreasing) terms fall 1e-40
    below the running sum."""
    with localcontext() as context:
        context.prec = 60
        f = 1 - p
        term = Decimal(math.comb(n, k)) * p**k * f ** (n - k)
        total = term
        for j in range(k, 0, -1):
            term = term * j * f / ((n - j + 1) * p)
            total += term
            if term < total * Decimal("1e-40"):
                break
        return total


@pytest.mark.parametrize("q", [0.3, 0.9])
@pytest.mark.parametrize("alpha", [0.05, 1e-8])
def test_large_asymmetric_bracket_ranks_meet_exact_tail_inequalities(q, alpha):
    """Beyond the exact recurrence's count, an asymmetric rate keeps both
    ranks covering (``P(X < a) <= alpha/2``, ``P(X >= b) <= alpha/2``) and
    tight (one rank inward exceeds the tail), against decimal arithmetic."""
    n = 2**16
    ranks = _bracket_ranks(n, q, alpha)
    assert ranks is not None
    a, b = ranks
    tail = Decimal(alpha) / 2
    with localcontext() as context:
        context.prec = 60
        rate, complement = Decimal(q), 1 - Decimal(q)
    assert _decimal_lower_tail(a - 1, n, rate) <= tail < _decimal_lower_tail(a, n, rate)
    assert (
        _decimal_lower_tail(n - b, n, complement)
        <= tail
        < _decimal_lower_tail(n - b + 1, n, complement)
    )


@pytest.mark.slow
@pytest.mark.parametrize("n", [2**20, 2**32, 2**40])
def test_large_central_bracket_ranks_match_factorial_oracle(n):
    """At a tail equal to an exact central CDF value, both ranks keep their
    coverage and tightness at counts where library CDFs lose digits."""
    from increment.estimation._certified import log_interval, log_rising

    middle = n // 2
    log_mass = (
        log_rising(Fraction(1), n)
        - 2 * log_rising(Fraction(1), middle)
        - n * log_interval(Fraction(2))
    )
    midpoint = (log_mass.lo + log_mass.hi) / 2
    for tail in (0.025, 5e-21):
        k = int(binom.ppf(tail, n, 0.5)) - 1
        with localcontext() as context:
            context.prec = 60
            mass = (Decimal(midpoint.numerator) / Decimal(midpoint.denominator)).exp()
            expected = Decimal("0.5") - mass / 2
            for j in range(1, middle - k):
                mass *= Decimal(middle - j + 1) / Decimal(middle + j)
                expected -= mass
            mass_at_k = mass * Decimal(k + 1) / Decimal(n - k)
            alpha = float(2 * expected)
        ranks = _bracket_ranks(n, 0.5, alpha)
        assert ranks is not None
        for rank in (ranks[0], n - ranks[1] + 1):
            assert rank in (k, k + 1)
            bound = expected if rank == k + 1 else expected - mass_at_k
            assert bound <= Decimal(alpha) / 2


def test_quantile_bracket_refuses_subrepresentable_tail():
    """A tail rounded below binary64 is the established numeric refusal."""
    with pytest.raises(InvalidRequestError) as exc_info:
        _bracket_ranks(1000, 0.5, np.nextafter(0.0, 1.0))
    assert exc_info.value.code == "estimation.meta.alpha_too_small"


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_quantile_invalid_alpha_keeps_diagnostics_refusal(alpha):
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.linspace(1.0, 2.0, 100), q=0.5, alpha=alpha)
    assert exc_info.value.code == "estimation.diagnostics.alpha"


def test_rejects_nonpositive_quantile():
    y = np.array([-1.0, 0.0, 2.0, 3.0, 4.0] * 20)
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(y, q=0.1)
    assert exc_info.value.code == "estimation.quantile.quantile_positive_log"


def test_rejects_tiny_n():
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.array([1.0, 2.0, 3.0]), q=0.99)
    assert exc_info.value.code == "estimation.quantile.too_small_bound"


def test_rejects_bad_q():
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.ones(100), q=1.0)
    assert exc_info.value.code == "estimation.quantile.log_quantile_se"


def test_refuses_p99_when_upper_bound_does_not_exist():
    """For q=0.99 at level 0.95 the distribution-free upper bound needs
    q**n <= alpha/2, i.e. n >= 368; in the band n in [25, 367] the clamped
    bracket covers as little as ~20% while claiming 95%. Must refuse."""
    rng = np.random.default_rng(2026)
    for n in (25, 100, 367):
        with pytest.raises(InvalidRequestError) as exc_info:
            log_quantile_se(rng.lognormal(1.0, 0.5, n), q=0.99)
        assert exc_info.value.code == "estimation.quantile.too_small_bound"
        assert exc_info.value.context["n_min"] == 368


def test_p99_accepted_at_smallest_valid_n():
    rng = np.random.default_rng(2026)
    y = rng.lognormal(1.0, 0.5, 368)  # smallest n with a 95% upper bound
    point, se = log_quantile_se(y, q=0.99)
    assert math.isfinite(point) and point > 0.0 and se > 0.0


def test_refuses_nan_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    y[3] = np.nan
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(y, q=0.5)
    assert exc_info.value.code == "estimation.quantile.values_contain_non"


def test_refuses_all_nan_values():
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.full(100, np.nan), q=0.5)
    assert exc_info.value.code == "estimation.quantile.values_contain_non"


def test_refuses_inf_values():
    rng = np.random.default_rng(2026)
    y = rng.lognormal(1.0, 0.5, 400)
    y[0] = np.inf
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(y, q=0.99)
    assert exc_info.value.code == "estimation.quantile.values_contain_non"


def test_refuses_negative_infinity_values():
    """A -inf value sorts FIRST, so checking only the sorted maximum's
    finiteness (the old guard) never sees it, while it still corrupts
    n/rank computations."""
    rng = np.random.default_rng(2026)
    y = rng.lognormal(1.0, 0.5, 400)
    y[0] = -np.inf
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(y, q=0.99)
    assert exc_info.value.code == "estimation.quantile.values_contain_non"


def test_rejects_multidimensional_values():
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(np.ones((10, 10)), q=0.5)
    assert exc_info.value.code == "estimation.quantile.values_one_dimensional"


def test_se_matches_classical_formula_for_continuous_data():
    """Continuous data must be numerically identical to the plain
    Woodruff formula -- the bracket resolves finely enough that the
    buffer term is exactly zero."""
    rng = np.random.default_rng(7)
    alpha = 0.05
    q = 0.9
    y = np.sort(rng.lognormal(mean=1.0, sigma=0.8, size=400))
    a = int(binom.ppf(alpha / 2, 400, q))
    b = int(binom.ppf(1 - alpha / 2, 400, q)) + 1
    z = norm.ppf(1 - alpha / 2)
    want_se = (math.log(y[b - 1]) - math.log(y[a - 1])) / (2 * z)
    _, got_se = log_quantile_se(y, q=q, alpha=alpha)
    assert math.isclose(got_se, want_se, rel_tol=1e-12)


@pytest.mark.parametrize("n", [6, 20])
def test_narrow_but_fully_distinct_bracket_pays_no_width_penalty(n):
    """A narrow bracket (few ranks) is not by itself a resolution
    problem -- only an ACTUAL TIE between order statistics is. Continuous
    data at small n has a bracket far under the calibrated cutoff by rank
    count, but every value inside it is still distinct, so the buffer
    must stay exactly zero and the formula must match classical."""
    rng = np.random.default_rng(9)
    alpha = 0.05
    q = 0.5
    y = np.sort(rng.lognormal(mean=1.0, sigma=0.8, size=n))
    a = int(binom.ppf(alpha / 2, n, q))
    b = int(binom.ppf(1 - alpha / 2, n, q)) + 1
    assert np.unique(y[a - 1 : b]).size == b - a + 1  # no ties in this bracket
    z = norm.ppf(1 - alpha / 2)
    want_se = (math.log(y[b - 1]) - math.log(y[a - 1])) / (2 * z)
    _, got_se = log_quantile_se(y, q=q, alpha=alpha)
    assert math.isclose(got_se, want_se, rel_tol=1e-12)


def test_rounded_currency_pays_no_width_penalty_when_the_bracket_resolves():
    """Cents-rounded data must match the plain classical (buffer-free)
    Woodruff formula exactly whenever the bracket spans enough distinct
    cent values -- ties alone must never trigger widening. True both at
    n=2000 (18 distinct cent values in this bracket) and at n=5000 with
    enough spread that its bracket resolves too (22 distinct values) --
    the buffer is a function of bracket resolution, not sample size."""
    rng = np.random.default_rng(11)
    for n, sigma in ((2000, 0.5), (5000, 0.8)):
        y = np.round(rng.lognormal(1.0, sigma, n), 2)
        y.sort()
        a = int(binom.ppf(0.025, n, 0.5))
        b = int(binom.ppf(0.975, n, 0.5)) + 1
        z = norm.ppf(0.975)
        want_se = (math.log(y[b - 1]) - math.log(y[a - 1])) / (2 * z)
        _, got_se = log_quantile_se(y, q=0.5)
        assert math.isclose(got_se, want_se, rel_tol=1e-9)


def test_rounded_currency_widening_is_bounded_when_the_bracket_does_not_resolve():
    """This n=5000 cents draw's median bracket holds just 11 distinct
    cent values (just under the calibrated cutoff), so the buffer
    legitimately activates. The widening must stay small and bounded:
    at least the classical (buffer-free) SE, and never more than 5%
    wider than the SE the same draw would report before rounding --
    the buffer corrects for recording resolution, it does not blow up
    because of it."""
    rng = np.random.default_rng(11)
    rng.lognormal(1.0, 0.5, 2000)  # advance past the resolved n=2000 draw above
    y_raw = np.sort(rng.lognormal(1.0, 0.5, 5000))
    y_rounded = np.sort(np.round(y_raw, 2))
    a = int(binom.ppf(0.025, 5000, 0.5))
    b = int(binom.ppf(0.975, 5000, 0.5)) + 1
    z = norm.ppf(0.975)
    classical_se = (math.log(y_rounded[b - 1]) - math.log(y_rounded[a - 1])) / (2 * z)
    _, se_rounded = log_quantile_se(y_rounded, q=0.5)
    _, se_unrounded = log_quantile_se(y_raw, q=0.5)
    assert se_rounded >= classical_se
    assert se_rounded <= 1.05 * se_unrounded


def _bracket(y: np.ndarray, q: float, alpha: float) -> tuple[float, float]:
    """The distribution-free order-statistic bracket ``[Y_(a), Y_(b)]``,
    computed independently of the module under test."""
    y = np.sort(y)
    a = int(binom.ppf(alpha / 2, y.size, q))
    b = int(binom.isf(alpha / 2, y.size, q)) + 1
    return float(y[a - 1]), float(y[b - 1])


def _tied_samples(seed: int) -> dict[str, tuple[np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    x = rng.lognormal(5.0, 0.6, 20000)
    ms_off_grid = np.round(x)
    keep = rng.random(x.size) < 0.01
    ms_off_grid[keep] = x[keep]
    counts = rng.poisson(3, 100).astype(float)
    price_endings = counts.copy()
    price_endings[(counts > 0) & (rng.random(counts.size) < 0.5)] -= 0.01
    return {
        "poisson": (counts, 0.9),
        "ms": (np.round(rng.lognormal(5.0, 0.6, 20000)), 0.5),
        "ms_off_grid": (ms_off_grid, 0.5),
        "price_endings": (price_endings, 0.9),
    }


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("kind", ["poisson", "ms", "ms_off_grid", "price_endings"])
def test_tied_interval_contains_the_order_statistic_bracket(kind, seed):
    """``[Y_(a), Y_(b)]`` covers the quantile with probability at least
    1 - alpha for any distribution, discrete or not. On a coarsely tied
    bracket the reported interval must contain it, whatever the recording
    grid: an off-grid value or a mix of .99/.00 endings must not shrink the
    interval back towards the classical one, which misses it whenever the
    sample quantile sits at one end of the bracket."""
    y, q = _tied_samples(seed)[kind]
    for alpha in (0.10, 0.05, 0.01):
        lower, upper = _bracket(y, q, alpha)
        point, se = log_quantile_se(y, q=q, alpha=alpha)
        half = norm.isf(alpha / 2) * se
        assert math.log(point) - half <= math.log(lower) + 1e-12
        assert math.log(point) + half >= math.log(upper) - 1e-12


def test_one_off_grid_value_does_not_undo_the_containment():
    """A lattice bracket [11, 12] with the sample median at 11 and one value
    recorded off the grid at 11.6: the classical interval around 11 stops
    short of 12, and so did a widening sized by the smallest gap between
    recorded values, which that single value shrinks to 0.4."""
    y = np.concatenate([np.full(50, 10.0), np.full(61, 11.0), [11.6], np.full(109, 12.0)])
    lower, upper = _bracket(y, 0.5, 0.05)
    assert (lower, upper) == (11.0, 12.0)
    point, se = log_quantile_se(y, q=0.5)
    assert point == 11.0
    assert math.log(point) + norm.isf(0.025) * se >= math.log(upper)


def test_fully_degenerate_bracket_refuses():
    y = np.full(100, 5.0)
    with pytest.raises(InvalidRequestError) as exc_info:
        log_quantile_se(y, q=0.5)
    assert exc_info.value.code == "estimation.quantile.degenerate_spread_order"


def test_a_collapsed_bracket_keeps_a_positive_width_no_wider_than_its_grid():
    """At q=0.125 every rank of this lattice's bracket holds 10: the
    classical width is zero, which the log-scale transport cannot carry.
    The reported width stays positive and within one recording cell."""
    y = np.sort(np.repeat(np.arange(10.0, 14.0), 80))
    assert _bracket(y, 0.125, 0.05) == (10.0, 10.0)
    point, se = log_quantile_se(y, q=0.125)
    assert point == 10.0
    assert 0.0 < norm.isf(0.025) * se <= math.log(11.0 / 10.0)


class _UnitSource:
    """Minimal MomentSource stub serving one metric's unit rows."""

    breakouts: tuple[str, ...] = ()
    operations: frozenset[SourceOperation] = frozenset()

    def __init__(self, df, metric):
        self._df, self.metrics, self.study_id = df, [metric], "s"
        self.capabilities = frozenset({"total"})

    def unit_frame(self, metric, *, covariates=()):
        return self._df


def test_quantile_lift_recovers_known_shift():
    rng = np.random.default_rng(11)
    n = 4000
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.0, 0.5, n) * 1.25  # +25% at every quantile
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    assert est.metric == "lat" and est.group_id == "treatment"
    lift = est.require_lift()
    assert 0.15 < lift.value < 0.35
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < 0.25 < lift.ub


def test_quantile_lift_no_widened_note_on_continuous_small_n():
    """A narrow bracket from small n is not a resolution problem for
    continuous data -- no arm has an actual tie, so no note is stamped
    even though the bracket rank count alone is well under the
    calibrated cutoff."""
    rng = np.random.default_rng(3)
    n = 20
    control = rng.lognormal(1.0, 0.8, n)
    treatment = rng.lognormal(1.0, 0.8, n) * 1.25
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    assert est.note is None


@pytest.mark.parametrize("offset", [1e6, 1e12, 1e15])
def test_quantile_lift_survives_a_large_offset(offset: float):
    """Two medians differing by exactly 1 at a large offset: the reported
    log ratio is ``log1p((Q_t - Q_c) / Q_c)`` (within one ulp of a 60-digit
    oracle) and the interval covers the true relative effect.
    ``log(Q_t) - log(Q_c)`` loses the whole effect at 1e15."""
    from decimal import Decimal, getcontext

    n = 101  # odd: the q=0.5 order statistic is a single observed value
    control = offset + 1000.0 * np.arange(n)  # median = offset + 50000, exactly
    treatment = control + 1.0  # every unit shifted by 1; median shifts by 1
    q_c, q_t = offset + 50000.0, offset + 50001.0
    assert float(np.quantile(control, 0.5)) == q_c
    assert float(np.quantile(treatment, 0.5)) == q_t
    getcontext().prec = 60
    want_log_rr = float((Decimal(q_t) / Decimal(q_c)).ln())
    true_lift = 1.0 / q_c

    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")

    lift = est.require_lift()
    assert lift.log_mean is not None
    assert abs(lift.log_mean - want_log_rr) <= math.ulp(want_log_rr)
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < true_lift < lift.ub


def test_quantile_cuped_refused():
    """Raises the canonical arm.metric.quantile_cuped compatibility spec
    via refuse_unsupported, the same code every other entry point raises
    for the quantile+CUPED hazard."""
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    df = pd.DataFrame({"unit_id": [1, 2], "group_id": ["control", "t"], "y": [1.0, 2.0]})
    with pytest.raises(UnsupportedRequestError) as exc_info:
        estimate_quantile_lift(
            _UnitSource(df, metric),
            metric,
            "control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )
    assert exc_info.value.code == "arm.metric.quantile_cuped"
    assert exc_info.value.context["metric"] == "lat"


def test_quantile_observational_method_name_refused():
    """A quantile readout is a randomized path too - an estimate_ate
    dispatch key as the label would mislabel it exactly the same way."""
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    df = pd.DataFrame({"unit_id": [1, 2], "group_id": ["control", "t"], "y": [1.0, 2.0]})
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_quantile_lift(
            _UnitSource(df, metric),
            metric,
            "control",
            methods=[Method(name="iptw")],
        )
    assert exc_info.value.code == "estimation.engine.method_name_observational"


def test_quantile_lift_handles_integer_group_column():
    """Regression: unit_frame's group_id may be native-dtype (e.g. int
    variant codes), same as the moments path's group_summary rows. The
    comparison against control_group must normalize to str on both sides,
    matching engine.py's _df_to_arms convention - otherwise an int group
    column with a str control_group falsely reports the control as
    missing, even though it's right there in the (stringified) group list.
    """
    rng = np.random.default_rng(5)
    n = 500
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.0, 0.5, n) * 1.25
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": [0] * n + [1] * n,  # integer variant codes
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "0")
    assert est.group_id == "1"
    assert 0.15 < est.require_lift().value < 0.35


def test_quantile_lift_error_names_metric_and_arm():
    """A log_quantile_se failure (e.g. too few units to bound the
    quantile) should identify which metric and arm triggered it, not just
    the bare order-statistic message."""
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.99)
    df = pd.DataFrame(
        {"unit_id": [1, 2, 3], "group_id": ["control", "control", "t"], "y": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    assert exc_info.value.code == "estimation.quantile.estimate_quantile.metric_arm"
    assert exc_info.value.context["metric"] == "lat"
    assert exc_info.value.context["arm"] == "control"


def test_quantile_lift_control_not_found_raises():
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    df = pd.DataFrame({"unit_id": [1, 2], "group_id": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    assert exc_info.value.code == "estimation.quantile.control_group_found"
    assert exc_info.value.context["control_group"] == "control"
    assert set(cast("tuple[str, ...]", exc_info.value.context["arms"])) == {"a", "b"}


def test_quantile_lift_multi_arm_estimates_each_non_control_arm():
    rng = np.random.default_rng(13)
    n = 800
    control = rng.lognormal(1.0, 0.5, n)
    t1 = rng.lognormal(1.0, 0.5, n) * 1.20
    t2 = rng.lognormal(1.0, 0.5, n) * 1.40
    df = pd.DataFrame(
        {
            "unit_id": range(3 * n),
            "group_id": ["control"] * n + ["t1"] * n + ["t2"] * n,
            "y": np.concatenate([control, t1, t2]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    ests = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    by_arm = {e.group_id: e for e in ests}
    assert set(by_arm) == {"t1", "t2"}
    assert 0.10 < by_arm["t1"].require_lift().value < 0.30
    assert 0.24 < by_arm["t2"].require_lift().value < 0.50


def test_quantile_empty_methods_emits_keyed_failures():
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    df = pd.DataFrame(
        {
            "unit_id": range(12),
            "group_id": ["control"] * 6 + ["treatment"] * 6,
            "y": [1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7],
        }
    )
    computation = estimate_quantile_lift_computation(
        _UnitSource(df, metric), metric, "control", methods=[]
    )
    key = ArmHypothesisKey("lat", "treatment", "itt")
    assert computation.results == ()
    assert computation.failures[key].code.endswith("no_decision_method")


def test_quantile_default_methods_stay_unadjusted():
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    n = 20
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": list(range(1, n + 1)) + list(range(2, n + 2)),
        }
    )
    rows = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", methods=None)
    assert len(rows) == 1
    assert rows[0].method == "unadjusted"


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.quantile.log_quantile_se",
            lambda: log_quantile_se(np.ones(100), q=1.0),
        ),  # estimation/quantile.py::log_quantile_se
        (
            "estimation.quantile.control_group_found",
            lambda: estimate_quantile_lift(
                _UnitSource(
                    pd.DataFrame({"unit_id": [1, 2], "group_id": ["a", "b"], "y": [1.0, 2.0]}),
                    QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5),
                ),
                QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5),
                "control",
            ),
        ),  # estimation/quantile.py::estimate_quantile_lift
        (
            "estimation.quantile.estimate_quantile.metric_arm",
            lambda: estimate_quantile_lift(
                _UnitSource(
                    pd.DataFrame(
                        {
                            "unit_id": [1, 2, 3],
                            "group_id": ["control", "control", "t"],
                            "y": [1.0, 2.0, 3.0],
                        }
                    ),
                    QuantileMetric(name="lat", entity="u", fact="f", quantile=0.99),
                ),
                QuantileMetric(name="lat", entity="u", fact="f", quantile=0.99),
                "control",
            ),
        ),  # estimation/quantile.py::estimate_quantile_lift::_for_arm
    ],
)
def test_quantile_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def test_estimate_quantile_lift_uses_supplied_unit_rows_without_reloading():
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    rng = np.random.default_rng(5)
    n = 500
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.0, 0.5, n) * 1.25
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["t"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    src = _UnitSource(df, metric)
    calls = {"n": 0}
    real = src.unit_frame

    def spy(metric, **kwargs):
        calls["n"] += 1
        return real(metric, **kwargs)

    src.unit_frame = spy  # ty: ignore[invalid-assignment]
    estimate_quantile_lift_computation(src, metric, "control", unit_rows=df)
    assert calls["n"] == 0


def test_interval_excludes_null_exactly_when_p_value_at_most_alpha():
    """Duality: for several alphas, the row's own interval at that
    alpha must exclude the null if and only if p_value() <= alpha --
    checked on continuous, tied/under-resolved, and extreme-alpha
    inputs."""
    rng = np.random.default_rng(12)
    control_continuous = rng.lognormal(5.0, 0.6, 3000)
    treatment_continuous = rng.lognormal(5.08, 0.6, 3000)
    control_tied = np.round(rng.lognormal(5.0, 0.6, 3000))
    treatment_tied = np.round(rng.lognormal(5.08, 0.6, 3000))
    for control, treatment in (
        (control_continuous, treatment_continuous),
        (control_tied, treatment_tied),
    ):
        p = _quantile_p_value(control, treatment, q=0.5, null=0.0)
        for alpha in (0.20, 0.10, 0.05, 0.01, 1e-4, 1e-8):
            point_c, se_c = log_quantile_se(control, q=0.5, alpha=alpha)
            point_t, se_t = log_quantile_se(treatment, q=0.5, alpha=alpha)
            log_rr = math.log(point_t) - math.log(point_c)
            se_joint = math.sqrt(se_c**2 + se_t**2)
            z = norm.isf(alpha / 2)
            excludes = abs(log_rr - 0.0) >= z * se_joint
            assert excludes == (p <= alpha)


def test_intervals_nest_as_alpha_widens():
    """The reported interval at a looser alpha must contain the interval
    at a tighter alpha -- the p-value's own decoupling from alpha must
    not disturb the reported interval's own alpha-monotonicity. Uses
    continuous data, where the resolution buffer is exactly zero and the
    formula reduces to the classical Woodruff bracket -- the regime where
    nesting is guaranteed, not merely empirically observed (the buffered/
    rounded regime's own alpha-monotonicity is grounded separately)."""
    rng = np.random.default_rng(3)
    y = rng.lognormal(5.0, 0.6, 3000)
    alphas = (0.20, 0.10, 0.05, 0.01, 1e-4)
    bounds = []
    for alpha in alphas:
        point, se = log_quantile_se(y, q=0.5, alpha=alpha)
        z = norm.isf(alpha / 2)
        bounds.append((math.log(point) - z * se, math.log(point) + z * se))
    for (lo_tight, hi_tight), (lo_wide, hi_wide) in zip(bounds, bounds[1:], strict=False):
        # alphas are listed largest -> smallest, so each next interval
        # (smaller alpha) must contain the previous (larger alpha) one.
        assert lo_wide <= lo_tight
        assert hi_wide >= hi_tight


def test_p_value_matches_classical_normal_reference_on_well_separated_arms():
    """Sanity floor: for arms separated enough to be clearly
    significant, the inversion's p-value must land in the same
    ballpark as the classical Normal reference computed at a fixed
    alpha=0.05 -- this is not a new statistical claim, just a
    consistency check on the inversion's numerics. Exact agreement is
    not expected: the Woodruff bracket's own SE varies somewhat with
    alpha even with the resolution buffer at zero (the bracket ranks
    themselves shift), which is exactly why the p-value cannot be read
    off a single fixed-alpha SE -- so the tolerance here is a magnitude
    check, not a tight numeric pin."""
    rng = np.random.default_rng(4)
    control = rng.lognormal(1.0, 0.5, 4000)
    treatment = rng.lognormal(1.02, 0.5, 4000)  # theta = 0.02, clearly but not extremely separated
    p = _quantile_p_value(control, treatment, q=0.5, null=0.0)
    point_c, se_c = log_quantile_se(control, q=0.5, alpha=0.05)
    point_t, se_t = log_quantile_se(treatment, q=0.5, alpha=0.05)
    log_rr = math.log(point_t) - math.log(point_c)
    se_joint = math.sqrt(se_c**2 + se_t**2)
    z_stat = log_rr / se_joint
    classical_p = 2.0 * norm.sf(abs(z_stat))
    assert classical_p / 3.0 <= p <= classical_p * 3.0


def test_row_carries_a_plain_float_not_a_reusable_reference():
    """A quantile LiftEstimate row must not carry either arm's raw
    per-unit outcome data -- the p-value is a plain number, not a
    reusable reference holding the source values."""
    rng = np.random.default_rng(6)
    n = 500
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.0, 0.5, n) * 1.25
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    assert est.quantile_p_value is None or isinstance(est.quantile_p_value, float)
    assert not hasattr(est, "quantile_p_value_reference")


def test_estimated_row_p_value_matches_standalone_inversion():
    """`estimate_quantile_lift` stamps the SAME value `_quantile_p_value`
    computes standalone from the same two arms -- no divergent
    recomputation inside the estimator."""
    rng = np.random.default_rng(8)
    n = 800
    control = np.round(rng.lognormal(2.0, 0.5, n), 2)
    treatment = np.round(rng.lognormal(2.05, 0.5, n), 2)
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control")
    want = _quantile_p_value(control, treatment, q=0.5, null=0.0)
    assert est.quantile_p_value == want
    assert est.p_value() == want


def test_p_value_alpha_independent_at_the_default_and_a_multiplicity_alpha():
    """A quantile row estimated at two different allocated alphas must
    report the SAME p-value even though the row's own interval differs."""
    rng = np.random.default_rng(15)
    n = 1500
    control = np.round(rng.lognormal(3.0, 0.5, n), 1)
    treatment = np.round(rng.lognormal(3.05, 0.5, n), 1)
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est_default,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=0.05)
    (est_tight,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=0.005)
    assert est_default.quantile_p_value == est_tight.quantile_p_value
    lift_default, lift_tight = est_default.require_lift(), est_tight.require_lift()
    assert lift_default.lb is not None and lift_default.ub is not None
    assert lift_tight.lb is not None and lift_tight.ub is not None
    # The reported interval still tracks alpha, by necessity: the tighter
    # allocation's interval must be no narrower than the looser one's.
    assert lift_tight.lb <= lift_default.lb
    assert lift_tight.ub >= lift_default.ub


def test_quantile_p_value_unset_with_an_informative_prior():
    """An informative prior shrinks the reported posterior interval;
    the inversion never reads a prior, so it must not be stamped there
    -- p_value() falls back to the generic posterior branch instead,
    which stays consistent with the actual (shrunk) reported interval."""
    rng = np.random.default_rng(21)
    n = 1200
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.05, 0.5, n)
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    prior = Normal(mu=0.0, sigma=0.02)  # tight enough to visibly shrink the MLE
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", prior=prior)
    assert est.quantile_p_value is None
    lift = est.require_lift()
    assert lift.lb is not None and lift.ub is not None and lift.alpha is not None
    # Duality still holds, but now against the ACTUAL (prior-shrunk)
    # reported interval, read through the generic posterior branch --
    # not against the flat inversion, which would disagree once the
    # prior has moved the interval off the unshrunk MLE.
    excludes = lift.lb > 0.0 or lift.ub < 0.0
    assert excludes == est.stat_sig()
    assert excludes == (est.p_value() <= lift.alpha)


def test_quantile_p_value_matches_flat_construction_without_a_prior():
    """Without a prior (the default), stamping stays on: same estimator,
    same arms, prior=None must reproduce the earlier no-prior behavior."""
    rng = np.random.default_rng(22)
    n = 800
    control = rng.lognormal(1.0, 0.5, n)
    treatment = rng.lognormal(1.05, 0.5, n)
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", prior=None)
    assert est.quantile_p_value is not None
    want = _quantile_p_value(control, treatment, q=0.5, null=0.0)
    assert est.quantile_p_value == want


def test_excludes_never_flips_back_false_as_alpha_grows():
    """Fine alpha-grid scan: once the reported interval excludes the null
    at some alpha, it must keep excluding it at every larger alpha up to
    the collapse boundary, on both continuous and tied/under-resolved
    data."""
    rng = np.random.default_rng(17)
    control_continuous = rng.lognormal(4.0, 0.5, 2000)
    treatment_continuous = rng.lognormal(4.06, 0.5, 2000)
    control_tied = np.round(rng.lognormal(4.0, 0.5, 2000))
    treatment_tied = np.round(rng.lognormal(4.06, 0.5, 2000))
    alphas = np.geomspace(1e-6, 0.999, 200)
    for control, treatment in (
        (control_continuous, treatment_continuous),
        (control_tied, treatment_tied),
    ):
        excludes_flags = []
        for alpha in alphas:
            try:
                point_c, se_c = log_quantile_se(control, q=0.5, alpha=alpha)
                point_t, se_t = log_quantile_se(treatment, q=0.5, alpha=alpha)
            except InvalidRequestError:
                excludes_flags.append(False)
                continue
            log_rr = math.log(point_t) - math.log(point_c)
            se_joint = math.sqrt(se_c**2 + se_t**2)
            z = norm.isf(alpha / 2.0)
            excludes_flags.append(bool(abs(log_rr) >= z * se_joint))
        # Non-decreasing as alpha grows: once True it never goes back to False.
        seen_true = False
        for excludes in excludes_flags:
            if excludes:
                seen_true = True
            assert not (seen_true and not excludes)


def _integer_arms_with_one_off_grid_value() -> tuple[np.ndarray, np.ndarray]:
    """Integer-rounded medians near 20 plus one control value off the grid:
    the data on which a step read from the bracket's smallest gap made the
    90% interval wider than the 95% one."""
    control = np.append(np.round(np.random.default_rng(0).lognormal(3.0, 0.25, 400)), 20.05)
    treatment = np.round(np.random.default_rng(100).lognormal(3.06, 0.25, 400))
    return control, treatment


@pytest.mark.parametrize("arm", ["control", "treatment", "price_endings"])
def test_tied_half_width_never_shrinks_as_alpha_shrinks(arm):
    """Each arm's reported log half-width (z * se) is non-increasing in
    alpha by construction, on tied data with an off-grid value and on a
    .99/.00 price grid alike -- the property the p-value's duality rests on."""
    control, treatment = _integer_arms_with_one_off_grid_value()
    y = {"control": control, "treatment": treatment, "price_endings": None}[arm]
    if y is None:
        y = np.random.default_rng(4).poisson(3, 100).astype(float)
        y[(y > 0) & (np.random.default_rng(5).random(y.size) < 0.5)] -= 0.01
    previous = math.inf
    for alpha in np.geomspace(1e-6, 0.999, 400):
        try:
            _, se = log_quantile_se(y, q=0.5, alpha=alpha)
        except InvalidRequestError:
            continue
        half = norm.isf(alpha / 2.0) * se
        assert half <= previous * (1.0 + 1e-12)  # z * (half / z) rounds
        previous = half


def test_p_value_is_dual_to_the_reported_interval_on_tied_data():
    """``p_value() <= alpha`` exactly when the row's own interval at that
    alpha excludes the null, including on the tied data where the 90%
    interval used to be wider than the 95% one (so a p-value of 0.040 sat
    beside a 90% interval that covered zero)."""
    control, treatment = _integer_arms_with_one_off_grid_value()
    df = pd.DataFrame(
        {
            "unit_id": range(control.size + treatment.size),
            "group_id": ["control"] * control.size + ["treatment"] * treatment.size,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est0,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=0.05)
    p = est0.p_value()
    for alpha in (0.9, 0.5, 0.2, 0.10, 0.05, 0.01, 1e-4, p * 1.001, p / 1.001):
        (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=alpha)
        assert est.p_value() == p
        assert est.stat_sig() == (p <= alpha)


def test_p_value_is_dual_to_the_reported_interval_on_resolved_tied_arms():
    """Integer-rounded arms whose 95% brackets span enough repeated values
    to be resolved keep the classical half-width, which is zero once the
    bracket has narrowed onto two tied adjacent ranks near alpha=1. That
    zero-width interval excludes every other value; read as a non-exclusion
    it left p_value() at 0.999999999999 beside a 95% interval excluding zero."""
    rng = np.random.default_rng(1)
    control = np.round(rng.lognormal(5.0, 1.0, 2000))
    treatment = np.round(rng.lognormal(5.1, 1.0, 2000))
    df = pd.DataFrame(
        {
            "unit_id": range(control.size + treatment.size),
            "group_id": ["control"] * control.size + ["treatment"] * treatment.size,
            "y": np.concatenate([control, treatment]),
        }
    )
    metric = QuantileMetric(name="lat", entity="u", fact="f", quantile=0.5)
    (est0,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=0.05)
    p = est0.p_value()
    assert p <= 0.05
    for alpha in (0.5, 0.2, 0.05, 0.01, 1e-3, 1e-4, p * 1.001, p / 1.001):
        (est,) = estimate_quantile_lift(_UnitSource(df, metric), metric, "control", alpha=alpha)
        assert est.p_value() == p
        assert est.stat_sig() == (p <= alpha)
