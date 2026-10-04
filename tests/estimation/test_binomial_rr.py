"""Exact independent-binomial risk-ratio inversion: numerical certification,
zero/positive-control geometry, and the exact-enumeration applicability
boundary.

Deterministic unit tests only; the 336-cell Monte Carlo calibration grid
lives in ``tests/estimation/test_rare_event_calibration.py`` (owned
separately -- see that module for coverage-frequency evidence).
"""

from __future__ import annotations

import math
import re
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb

import numpy as np
import pytest
from scipy.special import ndtr, ndtri
from scipy.stats import binom as _binom

from increment.estimation import binomial_rr as brr
from increment.estimation._binomial_support import chernoff_support, exponent_lower_bound
from increment.estimation._tails import SCIPY_BINOMIAL_ULP_ALLOWANCE
from increment.estimation.binomial_rr import _find_boundary
from increment.estimation.results import BINOMIAL_METHOD, BinomialConfidenceSet
from tests.estimation._binomial_endpoint_reference import assert_endpoints_contain_finer_reference

# --- Independent decimal oracle for scipy.stats.binom --------------------
# For integer n, binom.cdf/sf are finite polynomials in q; stdlib `decimal`
# evaluates them at any precision without sharing SciPy code or float routines.


def _dec_binom_cdf(x: int, n: int, q: float | Decimal, prec: int = 60) -> Decimal:
    with localcontext() as context:
        context.prec = prec
        if x < 0:
            return Decimal(0)
        if x >= n:
            return Decimal(1)
        qd = Decimal(q)
        if qd == 0:
            return Decimal(1)
        if qd == 1:
            return Decimal(0)
        if x > n // 2:
            return Decimal(1) - _dec_binom_cdf(n - x - 1, n, Decimal(1) - qd, prec)
        term = (Decimal(1) - qd) ** n
        total = term
        odds = qd / (Decimal(1) - qd)
        for j in range(x):
            term *= Decimal(n - j) / Decimal(j + 1) * odds
            total += term
        return +total


def _dec_tail(kind: str, n_c: int, n_t: int, k: int, q: float, p: float) -> Decimal:
    """Exact ``F_+(q, p) = P(K >= k)`` or ``F_-(q, p) = P(K <= k)`` for ``K = n_c X_t - n_t X_c``,
    summed over the control count in decimal arithmetic."""
    with localcontext() as context:
        context.prec = 60
        qd, total = Decimal(q), Decimal(0)
        for i in range(n_c + 1):
            pmf_i = Decimal(comb(n_c, i))
            pmf_i *= (qd**i if i else Decimal(1)) * ((1 - qd) ** (n_c - i) if i < n_c else 1)
            if kind == "plus":
                threshold = -((-(k + n_t * i)) // n_c) - 1
                treatment = Decimal(1) - _dec_binom_cdf(threshold, n_t, p)
            else:
                treatment = _dec_binom_cdf((k + n_t * i) // n_c, n_t, p)
            total += pmf_i * treatment
        return +total


def _refined_sup(evaluate, a: float, b: float) -> float:
    """Largest ``evaluate(q)`` over a grid of ``[a, b]`` refined around its best point until a
    round gains less than ``1e-12``: a LOWER bound on the supremum of a smooth tail."""
    qs = np.linspace(a, b, 129) if b > a else np.array([a])
    values = [evaluate(float(q)) for q in qs]
    best, at = max(values), float(qs[int(np.argmax(values))])
    step = (b - a) / 128.0
    while step > 1e-13:
        lo, hi = max(a, at - 2.0 * step), min(b, at + 2.0 * step)
        qs = np.linspace(lo, hi, 33)
        values = [evaluate(float(q)) for q in qs]
        gained = max(values) - best
        if max(values) > best:
            best, at = max(values), float(qs[int(np.argmax(values))])
        step = (hi - lo) / 32.0
        if gained < 1e-12:
            break
    return best


class TestCertifiedSupOutwardRounding:
    """`_certified_sup`/`_tail_plus`/`_tail_minus` must never report a
    value below the true mathematical supremum, even by a single ULP.
    """

    BETA = brr.nuisance_beta(0.05)

    def test_one_ulp_counterexample_stays_a_valid_upper_bound(self):
        # One-ulp counterexample: F_-(q,q) = 1 - q + q^2 is decreasing on this tiny
        # interval, so its exact supremum is the value AT the left
        # endpoint `a`, treating `a` as its exact binary64 value.
        a = 0.1
        b = math.nextafter(a, math.inf)
        window = brr._support_window(1, a, b)
        certified = brr._tail_minus(a, a, 1, 1, 0, window)
        exact_sup = 1.0 - a + a * a
        assert certified >= exact_sup

    def test_certified_sup_dominates_full_precision_decimal_evaluation(self):
        # A moderate, non-trivial (n_c, n_t, k) triple: the certified sup
        # (float64 + margin) must dominate a full decimal.getcontext-precision
        # evaluation at the reported argmax's neighborhood, not merely
        # come close to it.
        n_c, n_t, k = 40, 40, 3
        a, b = brr.clopper_pearson(12, n_c, 1e-6)
        window = brr._support_window(n_c, a, b)
        r = 1.3
        certified = brr._certified_sup(
            "plus", a, b, r, n_c, n_t, k, window, beta=self.BETA, rule=brr.NUISANCE_STOP
        ).upper
        # Independent decimal re-evaluation of F_+ at a fine grid over
        # [a, b]: the certified bound must be >= every grid point's exact
        # value (a weaker, cheap sanity check on top of the analytic
        # monotone/quadratic bound's own correctness).
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            q = a + frac * (b - a)
            exact = _dec_tail("plus", n_c, n_t, k, q, min(r * q, 1.0))
            assert certified >= float(exact)

    def test_singular_zero_endpoint_refines_an_interior_maximum(self):
        window = brr._support_window(1, 0.0, 0.75)
        rule = brr.NUISANCE_STOP

        certified = brr._certified_sup("plus", 0.0, 0.75, 1.0, 1, 1, 1, window, beta=0.0, rule=rule)

        # This tail is q(1 - q), whose maximum is 0.25 at q=0.5.  The
        # q=0 endpoint makes the curvature envelope singular, but refinement
        # must still tighten the initial monotone certificate of 0.75.
        assert certified.stopped
        assert 0.25 <= certified.upper <= 0.25 * (1.0 + rule.gap_fraction) + 1e-9

    def test_singular_saturated_endpoint_refines_an_interior_maximum(self):
        window = brr._support_window(1, 0.125, 0.5)
        rule = brr.NUISANCE_STOP

        certified = brr._certified_sup(
            "minus", 0.125, 0.5, 2.0, 1, 1, -1, window, beta=0.0, rule=rule
        )

        # q(1 - 2q) peaks at q=0.25; neither endpoint attains that value.
        assert certified.stopped
        assert 0.125 <= certified.upper <= 0.125 * (1.0 + rule.gap_fraction) + 1e-9


class TestTailLowerEnclosure:
    """A value `_tail_plus`/`_tail_minus` returns is inflated by the omitted control mass and
    the float margin; the enclosure removes both and never exceeds the exact tail."""

    @pytest.mark.parametrize(
        ("kind", "counts", "r"),
        [
            ("plus", (3, 12, 5, 12), 1.0),
            ("plus", (8, 40, 3, 25), 1.4),
            ("minus", (3, 12, 5, 12), 0.9),
            ("minus", (8, 40, 11, 25), 1.1),
            ("minus", (0, 6, 2, 6), 1.0),
        ],
    )
    def test_enclosure_never_exceeds_the_exact_tail(self, kind, counts, r):
        x_c, n_c, x_t, n_t = counts
        a, b = brr.clopper_pearson(x_c, n_c, brr.nuisance_beta(0.05))
        b = min(b, 1.0 / r) if kind == "minus" else b
        window = brr._support_window(n_c, a, b)
        k = n_c * x_t - n_t * x_c
        for frac in (0.0, 0.25, 0.6, 1.0):
            q = a + frac * (b - a)
            if kind == "plus":
                p = min(r * q, 1.0)
                value = brr._tail_plus(q, p, n_c, n_t, k, window)
            else:
                p = r * q
                value = brr._tail_minus(q, p, n_c, n_t, k, window)
            lower = brr.tail_lower_enclosure(value, window)
            assert 0.0 <= lower <= value
            assert Decimal(lower) <= _dec_tail(kind, n_c, n_t, k, q, p)

    @pytest.mark.parametrize("kind", ["plus", "minus"])
    def test_windowed_enclosure_is_below_the_full_enumeration(self, kind):
        n_c = n_t = 20_000
        x_c, x_t, r = 40, 55, 1.0
        a, b = brr.clopper_pearson(x_c, n_c, brr.nuisance_beta(0.05))
        window = brr._support_window(n_c, a, b)
        assert window[2] > 0.0 and window[1] - window[0] < n_c  # genuinely truncated
        k = n_c * x_t - n_t * x_c
        i = np.arange(n_c + 1)
        q = (a + b) / 2.0
        pmf = _binom.pmf(i, n_c, q)
        if kind == "plus":
            thresholds = np.ceil((k + n_t * i) / n_c).astype(np.int64) - 1
            full = float(np.dot(pmf, _binom.sf(thresholds, n_t, r * q)))
            value = brr._tail_plus(q, r * q, n_c, n_t, k, window)
        else:
            thresholds = np.floor((k + n_t * i) / n_c).astype(np.int64)
            full = float(np.dot(pmf, _binom.cdf(thresholds, n_t, r * q)))
            value = brr._tail_minus(q, r * q, n_c, n_t, k, window)
        assert value >= full
        assert brr.tail_lower_enclosure(value, window) <= full


def _sup_inputs(kind: str, counts: tuple[int, int, int, int], r: float):
    """``(a, b, r, n_c, n_t, k, window)`` of one nuisance supremum at the 5% level."""
    x_c, n_c, x_t, n_t = counts
    a, b = brr.clopper_pearson(x_c, n_c, brr.nuisance_beta(0.05))
    b = min(b, 1.0 / r) if kind == "minus" else b
    return a, b, r, n_c, n_t, n_c * x_t - n_t * x_c, brr._support_window(n_c, a, b)


class TestNuisanceStopContract:
    """The nuisance supremum search stops on a gap relative to the p-value it reports, and a
    search that runs out of iterations returns its valid bound with the gap it reached."""

    BETA = brr.nuisance_beta(0.05)
    CELLS = [
        pytest.param("plus", (3, 40, 12, 40), 1.3, id="plus-shifted-null"),
        pytest.param("plus", (4, 60, 14, 60), 1.0, id="plus-null"),
        pytest.param("minus", (14, 60, 4, 60), 1.0, id="minus-null"),
    ]

    @staticmethod
    def _decimal_sup(kind: str, inputs) -> float:
        a, b, r, n_c, n_t, k, _window = inputs

        def tail(q: float) -> float:
            p = min(r * q, 1.0) if kind == "plus" else r * q
            return float(_dec_tail(kind, n_c, n_t, k, q, p))

        return _refined_sup(tail, a, b)

    @pytest.mark.parametrize("tail", [0.0, 0.025, 0.5], ids=lambda t: f"tail-{t}")
    @pytest.mark.parametrize(("kind", "counts", "r"), CELLS)
    def test_certificate_brackets_an_independent_supremum(self, kind, counts, r, tail):
        inputs = _sup_inputs(kind, counts, r)
        rule = brr.NUISANCE_STOP
        cert = brr._certified_sup(
            kind, *inputs, beta=self.BETA, rule=rule, reading=brr._Reading(tail)
        )
        sup = self._decimal_sup(kind, inputs)
        assert cert.stopped
        assert cert.witness_lower <= cert.upper
        assert cert.witness_lower <= sup + 1e-9
        assert cert.upper >= sup - 1e-12
        gap = cert.upper - cert.witness_lower
        scale = max(self.BETA + cert.witness_lower, tail)
        assert gap <= rule.gap_fraction * scale + brr._certification_noise(inputs[-1])
        # The reported bound exceeds the supremum by no more than the declared gap.
        noise = brr._certification_noise(inputs[-1])
        assert cert.upper <= sup + rule.gap_fraction * scale + noise + 1e-9
        assert cert.iterations <= rule.max_iter

    @pytest.mark.parametrize(("kind", "counts", "r"), CELLS)
    def test_upper_never_rises_as_the_cap_grows(self, kind, counts, r):
        inputs = _sup_inputs(kind, counts, r)
        uppers = [
            brr._certified_sup(
                kind, *inputs, beta=self.BETA, rule=brr._StopRule(2.0**-40, cap)
            ).upper
            for cap in (10, 60, 240)
        ]
        # Bounds of sibling leaves are computed independently, so a longer search may rise by
        # rounding noise (far below the float margin) but never by more.
        slack = brr._eps_margin(1)
        assert uppers[0] >= uppers[1] - slack
        assert uppers[1] >= uppers[2] - slack

    @pytest.mark.parametrize(("kind", "counts", "r"), CELLS)
    def test_an_exhausted_cap_returns_a_valid_bound_and_says_so(self, kind, counts, r):
        inputs = _sup_inputs(kind, counts, r)
        cert = brr._certified_sup(kind, *inputs, beta=self.BETA, rule=brr._StopRule(2.0**-14, 3))
        assert not cert.stopped
        assert cert.iterations == 3
        assert cert.upper >= self._decimal_sup(kind, inputs) - 1e-12

    @pytest.mark.parametrize(("kind", "counts", "r"), CELLS)
    def test_a_settled_search_stops_once_the_comparison_with_the_level_is_certified(
        self, kind, counts, r
    ):
        inputs = _sup_inputs(kind, counts, r)
        rule = brr.NUISANCE_STOP
        sup = self._decimal_sup(kind, inputs)
        full = brr._certified_sup(kind, *inputs, beta=self.BETA, rule=rule)
        high, low = 3.0 * sup, sup / 3.0
        read_high = brr._Reading(high, settle=True)
        read_low = brr._Reading(low, settle=True)
        above = brr._certified_sup(kind, *inputs, beta=self.BETA, rule=rule, reading=read_high)
        below = brr._certified_sup(kind, *inputs, beta=self.BETA, rule=rule, reading=read_low)
        assert above.stopped and below.stopped
        assert self.BETA + above.upper < high  # the p-value is certified under the level
        assert self.BETA + below.witness_lower >= low  # and certified over it
        assert self.BETA + sup < high and self.BETA + sup >= low
        assert above.iterations <= full.iterations and below.iterations <= full.iterations
        assert min(above.iterations, below.iterations) < full.iterations
        assert above.upper >= sup - 1e-12 and below.upper >= sup - 1e-12

    @pytest.mark.parametrize(("kind", "counts", "r"), CELLS)
    def test_a_level_inside_the_bounds_at_the_cap_is_reported_as_not_below_it(
        self, kind, counts, r
    ):
        inputs = _sup_inputs(kind, counts, r)
        sup = self._decimal_sup(kind, inputs)
        level = self.BETA + sup
        capped = brr._certified_sup(
            kind,
            *inputs,
            beta=self.BETA,
            rule=brr._StopRule(2.0**-14, 3),
            reading=brr._Reading(level, settle=True),
        )
        assert not capped.stopped
        assert self.BETA + capped.upper >= level
        # Given the iterations to reach the declared gap, the same level is the gap's business.
        full = brr._certified_sup(
            kind,
            *inputs,
            beta=self.BETA,
            rule=brr.NUISANCE_STOP,
            reading=brr._Reading(level, settle=True),
        )
        assert full.stopped

    def test_an_empty_nuisance_domain_is_a_stopped_zero(self):
        window = brr._support_window(10, 0.5, 0.4)
        cert = brr._certified_sup(
            "minus", 0.5, 0.4, 1.0, 10, 10, 0, window, beta=self.BETA, rule=brr.NUISANCE_STOP
        )
        assert (cert.upper, cert.witness_lower, cert.stopped) == (0.0, 0.0, True)

    def test_a_p_value_far_below_the_tail_level_is_not_refined_past_the_precision_that_matters(
        self,
    ):
        """At 100 / 1,000 against 130 / 1,000 and risk ratio 0.65 the p-value is near 1e-7, far
        under the 0.025 tail. A gap relative to it runs out of iterations; floored at the tail
        level the search stops at the declared gap of that level, with a bound that is still a
        certified upper bound and within that gap of the supremum."""
        inputs = _sup_inputs("plus", (100, 1000, 130, 1000), 0.65)
        a, b, r, n_c, n_t, k, window = inputs
        rule, tail = brr.NUISANCE_STOP, 0.025
        relative = brr._certified_sup("plus", *inputs, beta=self.BETA, rule=rule)
        floored = brr._certified_sup(
            "plus", *inputs, beta=self.BETA, rule=rule, reading=brr._Reading(tail)
        )

        def enclosed(q: float) -> float:
            return brr.tail_lower_enclosure(brr._tail_plus(q, r * q, n_c, n_t, k, window), window)

        sup = _refined_sup(enclosed, a, b)
        assert not relative.stopped and relative.iterations == rule.max_iter
        assert floored.stopped and floored.iterations < relative.iterations // 8
        assert floored.upper >= sup - 1e-12
        assert floored.upper <= sup + rule.gap_fraction * tail + brr._certification_noise(window)

    def test_a_gap_target_below_the_certification_noise_is_not_chased(self):
        """At alpha = 1e-20 the nuisance budget is 3e-22 and a tail over a domain this small is
        below 1e-20, so a gap relative to the p-value is far under what a float tail certifies
        (its margin is about 1e-12): the search stops on the noise it cannot narrow, not at
        the cap."""
        beta = brr.nuisance_beta(1e-20)
        a, _ = brr.clopper_pearson(1, 1, beta)
        r = 1e20
        window = brr._support_window(1, a, 1.0 / r)
        cert = brr._certified_sup(
            "minus", a, 1.0 / r, r, 1, 1000, -999, window, beta=beta, rule=brr.NUISANCE_STOP
        )
        noise = brr._certification_noise(window)
        assert cert.stopped and cert.iterations == 0
        assert 0.0 <= cert.upper <= noise

    @pytest.mark.parametrize("tail", [0.0, 0.025], ids=lambda t: f"tail-{t}")
    def test_a_null_p_value_the_sixty_split_search_left_above_alpha_is_tightened_below_it(
        self, tail
    ):
        """5,778 / 57,780 against 5,985 / 57,780 at the null: the 60-iteration search shipped
        0.0642 (a supremum bound of 0.0321); refining to the declared gap gives 0.0487, with the
        gap read against the p-value or, as the interval and its verdict read it, the tail."""
        counts = (5778, 57780, 5985, 57780)
        a, b, r, n_c, n_t, k, window = _sup_inputs("plus", counts, 1.0)
        rule = brr.NUISANCE_STOP
        cert = brr._certified_sup(
            "plus",
            a,
            b,
            r,
            n_c,
            n_t,
            k,
            window,
            beta=self.BETA,
            rule=rule,
            reading=brr._Reading(tail),
        )

        def enclosed(q: float) -> float:
            return brr.tail_lower_enclosure(brr._tail_plus(q, q, n_c, n_t, k, window), window)

        sup = _refined_sup(enclosed, a, b)
        assert cert.stopped
        scale = max(self.BETA + sup, tail)
        assert sup - 1e-12 <= cert.upper <= sup + rule.gap_fraction * scale + 1e-8
        p = brr.p_two(1.0, *counts, self.BETA, tail=tail)
        assert p < 0.05
        assert p <= 0.0642288
        assert 2.0 * (self.BETA + sup) <= p + 1e-9

    @pytest.mark.parametrize("tail_read", [0.0, 0.025], ids=lambda t: f"read-against-{t}")
    @pytest.mark.parametrize(
        "counts",
        [
            (5778, 57780, 5985, 57780),
            (500, 5000, 560, 5000),
            (100, 1000, 130, 1000),
            (1000, 20000, 1095, 20000),
            (1000, 20000, 1093, 20000),
        ],
        ids=str,
    )
    def test_decisions_only_move_from_not_significant_to_significant(
        self, counts, tail_read, monkeypatch
    ):
        """Against the same search capped at 60 iterations the refined bound is never larger,
        so no cell leaves significance; near the tail allocation some enter it."""
        beta, tail = self.BETA, 0.025
        x_c, n_c, x_t, n_t = counts
        refined = {
            "plus": brr.p_plus(1.0, x_c, n_c, x_t, n_t, beta, tail=tail_read),
            "minus": brr.p_minus(1.0, x_c, n_c, x_t, n_t, beta, tail=tail_read),
        }
        monkeypatch.setattr(brr, "NUISANCE_STOP", brr._StopRule(brr.NUISANCE_STOP.gap_fraction, 60))
        capped = {
            "plus": brr.p_plus(1.0, x_c, n_c, x_t, n_t, beta, tail=tail_read),
            "minus": brr.p_minus(1.0, x_c, n_c, x_t, n_t, beta, tail=tail_read),
        }
        for kind in ("plus", "minus"):
            assert refined[kind] <= capped[kind] + brr._eps_margin(1)
            assert not (capped[kind] < tail <= refined[kind])
        if counts == (5778, 57780, 5985, 57780):
            assert refined["plus"] < tail <= capped["plus"]


class TestExtremeAlphaFiniteUpperBound:
    """A positive control count always has a finite mathematical upper
    endpoint (the CP lower bound eventually empties the nuisance domain);
    a numerical search CAP must never masquerade as genuine unboundedness.
    """

    def test_astra_extreme_alpha_repro_returns_finite_upper(self):
        ci = brr.confidence_interval(1, 1, 0, 1, alpha=1e-14, alternative="two-sided")
        assert ci.upper is not None
        assert math.isfinite(ci.upper)

    @pytest.mark.parametrize("alternative", ["two-sided", "less"])
    @pytest.mark.parametrize("alpha", [1e-14, 5e-307])
    def test_sparse_treatment_search_reaches_a_crossing_far_above_its_subunit_seed(
        self, alpha, alternative
    ):
        """With one control success in one control unit the Clopper-Pearson lower end is
        exactly ``beta / 2`` (Uniform(0, 1)), and at these alphas the certification margin
        exceeds the target until the restricted nuisance domain ``[a, 1/r]`` empties, so the
        evaluated crossing is exactly ``1 / a``. The point estimate (0.001) is more than 60
        doublings below it: the search must still bracket it, then stop within the declared
        resolution of it. At 5e-307 ``1 / a`` is finite but ``2 / a`` overflows, so the
        bracket's far end is the largest float."""
        counts = (1, 1, 1, 1000)
        a, _ = brr.clopper_pearson(1, 1, brr.nuisance_beta(alpha))
        crossing = 1.0 / a
        ci = brr.confidence_interval(*counts, alpha=alpha, alternative=alternative)
        tau = brr._endpoint_tolerance(brr._count_scale(*counts))
        assert ci.resolution_reached
        assert ci.upper is not None
        assert crossing * (1.0 - 1e-12) <= ci.upper <= crossing * math.exp(tau) * (1.0 + 1e-12)

    def test_a_crossing_beyond_the_float_range_is_a_coded_failure(self):
        """Below this alpha ``1 / a`` exceeds the largest float, so no upper endpoint is
        representable: a coded refusal, not an unbounded set or an arithmetic overflow."""
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.confidence_interval(1, 1, 1, 1000, alpha=1e-307, alternative="two-sided")
        assert exc_info.value.code == "estimation.binomial.tail_unrepresentable"

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_a_control_rate_the_binomial_pmf_cannot_evaluate_is_a_coded_refusal(self, alternative):
        """At this alpha the control's Clopper-Pearson lower end (``alpha / 64``) is a denormal
        for which SciPy's binomial PMF raises ``OverflowError``: the interval refuses with the
        module's coded failure and names the rate, rather than leaking the library's error."""
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.confidence_interval(1, 1, 1, 1000, alpha=4e-307, alternative=alternative)
        error = exc_info.value
        assert error.code == "estimation.binomial.tail_unrepresentable"
        assert error.context["n"] == 1
        rate = error.context["p"]
        assert isinstance(rate, float) and 0.0 < rate < sys.float_info.min

    def test_positive_control_upper_search_never_returns_none(self):
        # Within the validated Clopper-Pearson regime (beta = alpha/32
        # above `_CP_BETA_FLOOR`; x=2, n=3 is not a trivial exact shape)
        # a positive control always has a finite mathematical upper
        # endpoint, and the search must actually find it.
        for alpha in (1e-6, 1e-4, 0.01, 0.5, 0.999):
            ci = brr.confidence_interval(2, 3, 1, 5, alpha=alpha, alternative="two-sided")
            assert ci.upper is not None
            ci_less = brr.confidence_interval(2, 3, 1, 5, alpha=alpha, alternative="less")
            assert ci_less.upper is not None

    def test_alpha_below_the_validated_regime_raises_a_coded_failure_not_none(self):
        # Below `_CP_BETA_FLOOR` for a non-trivial shape, this module
        # refuses via a coded numerical failure -- never a silent
        # (incorrect) `upper=None` claim of unboundedness.
        with pytest.raises(brr.BinomialDataError):
            brr.confidence_interval(2, 3, 1, 5, alpha=1e-13, alternative="two-sided")

    def test_zero_control_is_the_only_analytically_unbounded_case(self):
        ci = brr.confidence_interval(0, 10, 5, 10, alpha=0.05, alternative="two-sided")
        assert ci.upper is None
        ci_less = brr.confidence_interval(0, 10, 5, 10, alpha=0.05, alternative="less")
        assert ci_less.upper is None


class TestToLiftBounds:
    def test_closed_floor_is_exact_not_nudged(self):
        # alternative="less" always has an EXACT R=0 lower endpoint; the
        # lift-scale floor of -1 must come out exactly, never off by a ULP.
        ci = brr.confidence_interval(3, 20, 5, 20, alpha=0.05, alternative="less")
        lower, _upper = brr.to_lift_bounds(ci)
        assert lower == -1.0

    def test_two_sided_zero_treatment_floor_is_exact_not_nudged(self):
        ci = brr.confidence_interval(3, 20, 0, 20, alpha=0.05, alternative="two-sided")
        assert ci.lower == 0.0
        lower, _upper = brr.to_lift_bounds(ci)
        assert lower == -1.0

    def test_search_derived_bounds_widen_not_narrow(self):
        ci = brr.confidence_interval(3, 20, 5, 20, alpha=0.05, alternative="two-sided")
        lower, upper = brr.to_lift_bounds(ci)
        assert lower <= ci.lower - 1.0
        assert upper is not None
        assert ci.upper is not None
        assert upper >= ci.upper - 1.0

    def test_none_upper_passes_through(self):
        ci = brr.confidence_interval(0, 10, 5, 10, alpha=0.05, alternative="two-sided")
        _lower, upper = brr.to_lift_bounds(ci)
        assert upper is None

    @pytest.mark.parametrize("closeness", [1e-9, 1e-5, 3e-4])
    def test_an_endpoint_at_the_float_range_converts_to_a_finite_lift_bound(self, closeness):
        """An upper crossing within the declared resolution of the largest float can be
        reported as that float (the search's outward end, clipped to its finite cap); the
        lift conversion must keep it finite, outward of the exact ``R - 1``, and constructible
        as the persisted set, which refuses an infinite endpoint."""
        root = sys.float_info.max * (1.0 - closeness)
        found = _find_boundary(
            lambda r: 1.0 if r <= root else 0.0,
            0.5,
            increasing=False,
            seed=1e-3,
            tau=2.0**-11,
            cap=sys.float_info.max,
        )
        assert found.endpoint is not None and found.endpoint >= root
        ci = brr.BinomialInterval(
            lower=0.5,
            upper=found.endpoint,
            geometry="central",
            p_value_null=0.4,
            endpoint_log_width=found.log_width,
            resolution_reached=found.reached,
            nuisance_gap_max=0.0,
            capped_probes=0,
        )
        lower, upper = brr.to_lift_bounds(ci)
        assert upper is not None and math.isfinite(upper)
        assert upper >= found.endpoint - 1.0
        BinomialConfidenceSet(
            lower=lower,
            upper=upper,
            alpha=0.05,
            decision_alpha=0.05,
            level=0.95,
            geometry="central",
            method=BINOMIAL_METHOD,
            x_c=1,
            n_c=1,
            x_t=1,
            n_t=1000,
            nuisance_beta=brr.nuisance_beta(0.05),
        )


class TestFastBinomMatchesScipy:
    """`_fast_binom_{cdf,sf,pmf}` bypass `scipy.stats.binom`'s generic
    per-call dispatch overhead for speed (see `binomial_rr._FULL_ENUMERATION_
    THRESHOLD`'s docstring); this pins them bit-for-bit identical to
    `scipy.stats.binom.{cdf,sf,pmf}`, including outside the `[0, n]`
    support where `_plus_threshold`/`_minus_threshold` can land.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_bit_exact_across_in_and_out_of_support_k(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 5000))
        p = float(rng.uniform(1e-6, 1 - 1e-6))
        k = rng.integers(-50, n + 50, size=int(rng.integers(1, 300))).astype(np.int64)
        assert np.array_equal(brr._fast_binom_cdf(k, n, p), _binom.cdf(k, n, p))
        assert np.array_equal(brr._fast_binom_sf(k, n, p), _binom.sf(k, n, p))
        assert np.array_equal(brr._fast_binom_pmf(k, n, p), _binom.pmf(k, n, p))

    def test_boundary_values_exact(self):
        n, p = 50, 0.3
        k = np.array([-1, 0, n - 1, n, n + 1], dtype=np.int64)
        assert np.array_equal(brr._fast_binom_cdf(k, n, p), _binom.cdf(k, n, p))
        assert np.array_equal(brr._fast_binom_sf(k, n, p), _binom.sf(k, n, p))
        assert np.array_equal(brr._fast_binom_pmf(k, n, p), _binom.pmf(k, n, p))


class TestExactThresholds:
    """The treatment count thresholds are exact integer quotients of ``k + n_t * i`` for every
    admitted arm size, including where that numerator exceeds float64's exact range.

    The reference is ``Fraction`` arithmetic. An adversarial numerator sits one count of
    ``n_t * (i - x_c)`` from a multiple of ``n_c``, where a float quotient (which rounds the
    numerator, then the division) lands on the wrong side of the integer.
    """

    ARM_SIZES = (10**8 + 7, 999_999_937, 2**29 - 1, 2**29 + 1, 10**9)

    @staticmethod
    def _adversarial_cells(n_c: int, n_t: int):
        """``(x_c, x_t, i_lo, i_hi)`` whose window holds a control count ``i`` with
        ``n_t * (i - x_c)`` congruent to ``0`` and to ``+-1`` modulo ``n_c``."""
        residues = {0}
        if math.gcd(n_t, n_c) == 1:
            inverse = pow(n_t, -1, n_c)
            residues |= {inverse, n_c - inverse}
        for shift in sorted(residues):
            for x_c in sorted({0, n_c - shift, n_c // 2 - shift // 2}):
                i = x_c + shift
                if not 0 <= x_c <= n_c or not 0 <= i <= n_c:
                    continue
                for x_t in (0, 1, n_t // 2, n_t - 1, n_t):
                    yield x_c, x_t, max(0, i - 3), min(n_c, i + 3)

    @pytest.mark.parametrize("n_t", ARM_SIZES)
    @pytest.mark.parametrize("n_c", ARM_SIZES)
    def test_thresholds_equal_the_exact_quotients_at_adversarial_counts(self, n_c, n_t):
        checked = 0
        for x_c, x_t, i_lo, i_hi in self._adversarial_cells(n_c, n_t):
            k = n_c * x_t - n_t * x_c
            plus = brr._plus_threshold(n_c, n_t, k, i_lo, i_hi)
            minus = brr._minus_threshold(n_c, n_t, k, i_lo, i_hi)
            for offset, i in enumerate(range(i_lo, i_hi + 1)):
                exact = Fraction(k + n_t * i, n_c)
                assert plus[offset] == math.ceil(exact) - 1, (x_c, x_t, i)
                assert minus[offset] == math.floor(exact), (x_c, x_t, i)
            checked += 1
        assert checked >= 5

    @pytest.mark.parametrize("n", ARM_SIZES)
    def test_thresholds_at_the_edges_of_the_support_are_exact(self, n):
        for x_c, x_t in ((0, 0), (0, n), (n, 0), (n, n), (1, n - 1), (n - 1, 1)):
            k = n * x_t - n * x_c
            for i_lo, i_hi in ((0, 4), (n - 4, n)):
                plus = brr._plus_threshold(n, n, k, i_lo, i_hi)
                minus = brr._minus_threshold(n, n, k, i_lo, i_hi)
                for offset, i in enumerate(range(i_lo, i_hi + 1)):
                    exact = Fraction(k + n * i, n)
                    assert plus[offset] == math.ceil(exact) - 1
                    assert minus[offset] == math.floor(exact)

    def test_random_numerators_up_to_the_ceiling_are_exact(self):
        rng = np.random.default_rng(20261004)
        for _ in range(200):
            n_c, n_t = (int(v) for v in rng.integers(1, brr.FINITE_SAMPLE_MAX_ARM_SIZE + 1, 2))
            x_c, x_t = int(rng.integers(0, n_c + 1)), int(rng.integers(0, n_t + 1))
            k = n_c * x_t - n_t * x_c
            i_lo = int(rng.integers(0, n_c + 1))
            i_hi = min(n_c, i_lo + 5)
            plus = brr._plus_threshold(n_c, n_t, k, i_lo, i_hi)
            minus = brr._minus_threshold(n_c, n_t, k, i_lo, i_hi)
            for offset, i in enumerate(range(i_lo, i_hi + 1)):
                exact = Fraction(k + n_t * i, n_c)
                assert plus[offset] == math.ceil(exact) - 1
                assert minus[offset] == math.floor(exact)

    @pytest.mark.parametrize(("n_c", "n_t"), [(10**9, 10**9), (999_999_937, 10**8 + 7)])
    def test_planning_offsets_give_the_runtime_thresholds(self, n_c, n_t):
        from increment.power import _binomial

        for x_c, x_t, i_lo, i_hi in self._adversarial_cells(n_c, n_t):
            k = n_c * x_t - n_t * x_c
            s = np.arange(i_lo, i_hi + 1, dtype=np.int64)
            plus = brr._plus_threshold(n_c, n_t, k, i_lo, i_hi)
            minus = brr._minus_threshold(n_c, n_t, k, i_lo, i_hi)
            assert np.array_equal(
                x_t + _binomial._threshold_offsets("plus", n_c, n_t, x_c, s), plus
            )
            assert np.array_equal(
                x_t + _binomial._threshold_offsets("minus", n_c, n_t, x_c, s), minus
            )


class TestApplicabilityBoundaryCalibration:
    """``FINITE_SAMPLE_MAX_ARM_SIZE`` is a compute-resource applicability boundary, not a
    scientific one: ordinary sizes up to it succeed (spot-checked below, and a rare count pair
    exactly at it), sizes far beyond it -- and exactly one arm-count above it -- are refused
    immediately (without attempting the expensive search), and the largest arm size in the full
    336-cell rare-event calibration manifest is admitted, which is checked structurally rather
    than by actually calling ``confidence_interval`` at that size (too slow to run as a routine
    regression).
    """

    def _record_searches(self, monkeypatch) -> list[str]:
        """Spy on the search entry points: any name listed was invoked."""
        calls: list[str] = []

        def spy(name: str):
            def record(*_args, **_kwargs):
                calls.append(name)

            return record

        monkeypatch.setattr(brr, "_find_boundary", spy("_find_boundary"))
        monkeypatch.setattr(brr, "clopper_pearson", spy("clopper_pearson"))
        return calls

    def test_refuses_immediately_far_above_the_ceiling(self, monkeypatch):
        """The ceiling is checked before any search work starts."""
        calls = self._record_searches(monkeypatch)
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.confidence_interval(
                1,
                brr.FINITE_SAMPLE_MAX_ARM_SIZE * 10,
                1,
                brr.FINITE_SAMPLE_MAX_ARM_SIZE * 10,
                alpha=0.05,
                alternative="two-sided",
            )
        assert calls == []
        assert exc_info.value.code == "estimation.binomial.arm_too_large_for_exact_enumeration"
        assert exc_info.value.context["max_arm_size"] == 1_000_000_000

    @pytest.mark.parametrize("above", ["control", "treatment"])
    def test_refuses_at_one_above_the_ceiling(self, monkeypatch, above):
        calls = self._record_searches(monkeypatch)
        n_c, n_t = (brr.FINITE_SAMPLE_MAX_ARM_SIZE + 1, 10)
        if above == "treatment":
            n_c, n_t = n_t, n_c
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.confidence_interval(1, n_c, 1, n_t, alpha=0.05, alternative="two-sided")
        assert calls == []
        assert exc_info.value.code == "estimation.binomial.arm_too_large_for_exact_enumeration"

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_a_rare_count_pair_exactly_at_the_ceiling_is_admitted(self, alternative):
        n = brr.FINITE_SAMPLE_MAX_ARM_SIZE
        ci = brr.confidence_interval(3, n, 5, n, alpha=0.05, alternative=alternative)
        assert math.isfinite(ci.lower) and 0.0 < ci.p_value_null <= 1.0
        if ci.upper is not None:
            assert ci.lower < 5 / 3 < ci.upper
        assert ci.lower <= 5 / 3

    def test_full_grid_maximum_is_admitted_by_the_ceiling(self):
        """The 336-cell rare-event calibration manifest computes ``n_c``
        from ``expected_events / p_c`` and ``n_t`` from the allocation
        ratio; its largest arm is ``p_c=1e-4, expected_events=100,
        ratio=(1, 4)``. Checked structurally -- against the same rounding
        formula the manifest uses, never by an actual call at this size.
        """
        p_c, expected_events, ratio = 1e-4, 100.0, (1, 4)
        c, t = ratio
        n_c = max(1, math.floor(expected_events / p_c + 0.5))
        n_t = max(1, math.floor(n_c * t / c + 0.5))
        assert (n_c, n_t) == (1_000_000, 4_000_000)
        assert n_c <= brr.FINITE_SAMPLE_MAX_ARM_SIZE
        assert n_t <= brr.FINITE_SAMPLE_MAX_ARM_SIZE

    def test_moderate_arm_sizes_succeed(self):
        for n in (100, 10_000, 200_000):
            ci = brr.confidence_interval(
                max(1, n // 20), n, max(1, n // 15), n, alpha=0.05, alternative="two-sided"
            )
            assert ci.upper is not None and math.isfinite(ci.upper)


def _analytic(root: float, *, increasing: bool, shape: str):
    """``(f, target)`` whose evaluated crossing sits at *root*, with no SciPy involved.

    Increasing: ``f(r) >= target`` iff ``r >= root``. Decreasing: iff ``r <= root``. ``step``
    jumps from 0 to 1 at the crossing, ``ratio`` is ``r / (r + root)`` and ``steep`` is
    ``0.5 * (r / root)**40`` held below a ceiling, so interpolation on it is badly misled
    on either side of the crossing. Decreasing forms are the mirrors in ``root / r``.
    """

    def unit(r: float) -> float:
        """The increasing form as a function of ``r / root`` (0 at 0, its ceiling at inf)."""
        if r <= 0.0:
            return 0.0
        if shape == "step":
            return 1.0 if r >= 1.0 else 0.0
        if shape == "ratio":
            return 1.0 if math.isinf(r) else r / (r + 1.0)
        return 0.5 * math.exp(0.5 if math.isinf(r) else min(40.0 * math.log(r), 0.5))

    if increasing:
        return (lambda r: unit(r / root)), 0.5
    return (lambda r: unit(root / r) if r > 0.0 else unit(math.inf)), 0.5


def _normal_tail(root: float, *, increasing: bool, sigma: float):
    """A p-value-like ``f`` with a Gaussian tail in ``log r``: ``ndtri(f)`` is linear in it."""
    z_target = float(ndtri(0.025))

    def f(r: float) -> float:
        if r <= 0.0:
            return 0.0 if increasing else 1.0
        u = math.log(r / root) / sigma
        return float(ndtr(z_target + u if increasing else z_target - u))

    return f, 0.025


def _counted(f):
    calls: list[float] = []

    def probe(r: float) -> float:
        calls.append(r)
        return f(r)

    return probe, calls


def _refine_spending(monkeypatch, calls: list[float]) -> list[tuple[float, int]]:
    """Record ``(bracket log width, evaluations spent)`` for each `_Crossing.refine` call."""
    refinements: list[tuple[float, int]] = []
    real = brr._Crossing.refine

    def wrapped(self, tau: float) -> None:
        before, width = len(calls), self.log_width()
        real(self, tau)
        refinements.append((width, len(calls) - before))

    monkeypatch.setattr(brr._Crossing, "refine", wrapped)
    return refinements


class TestEndpointResolutionContract:
    """``_count_scale`` is a search unit in log-risk-ratio units and ``_endpoint_tolerance``
    turns it into the bracket's log width at stop."""

    @pytest.mark.parametrize(
        "counts",
        [
            (0, 1, 0, 1),
            (0, 50, 25, 50),
            (50, 50, 50, 50),
            (1, 1_000_000, 0, 1_000_000),
            (4_000_000, 4_000_000, 3_999_999, 4_000_000),
        ],
    )
    def test_scale_is_finite_and_positive_at_every_boundary_count(self, counts):
        scale = brr._count_scale(*counts)
        assert math.isfinite(scale) and scale > 0.0

    def test_scale_approximates_the_log_risk_ratio_standard_error(self):
        x_c, n_c, x_t, n_t = 10_000, 100_000, 11_000, 100_000
        katz = math.sqrt(1 / x_c - 1 / n_c + 1 / x_t - 1 / n_t)
        assert brr._count_scale(x_c, n_c, x_t, n_t) == pytest.approx(katz, rel=1e-3)

    def test_scale_follows_the_failure_count_when_nearly_every_unit_converts(self):
        """A dense arm's risk ratio is pinned down by its few failures, not its many successes."""
        assert brr._count_scale(99_990, 100_000, 99_995, 100_000) < 0.2 * brr._count_scale(
            50_000, 100_000, 50_000, 100_000
        )

    def test_scale_shrinks_as_counts_grow(self):
        scales = [brr._count_scale(x, 100_000, 2 * x, 100_000) for x in (1, 10, 100, 1_000, 10_000)]
        assert scales == sorted(scales, reverse=True)

    def test_tolerance_is_one_128th_of_the_scale_between_its_clamps(self):
        assert brr._endpoint_tolerance(0.0141) == pytest.approx(0.0141 / 128.0)

    def test_tolerance_is_never_coarser_than_the_cap_or_finer_than_the_floor(self):
        assert brr._endpoint_tolerance(2.0) == 2.0**-11
        assert brr._endpoint_tolerance(1e-30) == 2.0**-40


class TestEndpointSearchOnAnalyticFunctions:
    """The inversion alone, on functions with a known crossing: outward endpoint, declared
    log width, null placement and disclosed resolution failure."""

    @pytest.mark.parametrize("tau", [0.1, 2.0**-11, 2.0**-30, 2.0**-40])
    @pytest.mark.parametrize("factor", [1e-9, 0.3, 0.999, 1.0000001, 1.9999, 3.0, 1e3])
    @pytest.mark.parametrize("seed", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("shape", ["step", "ratio", "steep"])
    @pytest.mark.parametrize("increasing", [True, False])
    def test_endpoint_is_outward_and_its_bracket_reaches_tau(
        self, increasing, shape, seed, factor, tau, monkeypatch
    ):
        root = seed * factor
        f, target = _analytic(root, increasing=increasing, shape=shape)
        probe, calls = _counted(f)
        refinements = _refine_spending(monkeypatch, calls)
        found = _find_boundary(probe, target, increasing=increasing, seed=seed, tau=tau)
        assert found.reached
        assert found.endpoint is not None
        assert 0.0 <= found.log_width <= tau
        slack = 1e-14 * root
        if increasing:
            assert found.endpoint <= root + slack
            assert found.endpoint * math.exp(found.log_width) >= root - slack
        else:
            assert found.endpoint >= root - slack
            assert found.endpoint * math.exp(-found.log_width) <= root + slack
        # The refinement spends at most one probe per halving of the bracket to tau, plus the
        # interpolation's slack: the rounding of the log coordinates cannot cost one more.
        ((width, spent),) = refinements
        assert spent <= math.ceil(math.log2(width / tau)) + brr._REFINE_SLACK

    @pytest.mark.parametrize("increasing", [True, False])
    @pytest.mark.parametrize("offset", [0.9, 1.0, 1.1])
    def test_interpolation_reaches_the_stop_in_far_fewer_probes_than_bisection(
        self, increasing, offset
    ):
        """On a Gaussian tail in log r -- the shape of the evaluated p-envelope -- the
        probit-linear probes cross a ln 2 bracket in a handful of evaluations."""
        tau = 2.0**-20
        f, target = _normal_tail(1.0, increasing=increasing, sigma=0.01)
        probe, calls = _counted(f)
        found = _find_boundary(probe, target, increasing=increasing, seed=offset, tau=tau)
        assert found.reached and found.log_width <= tau
        bisections = math.ceil(math.log2(math.log(2.0) / tau))
        assert len(calls) <= bisections // 2

    @pytest.mark.parametrize("direction", [math.inf, 0.0])
    @pytest.mark.parametrize("seed", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("increasing", [True, False])
    def test_a_crossing_one_float_from_the_seed_is_bracketed(self, increasing, seed, direction):
        root = math.nextafter(seed, direction)
        f, target = _analytic(root, increasing=increasing, shape="step")
        tau = 2.0**-30
        found = _find_boundary(f, target, increasing=increasing, seed=seed, tau=tau)
        assert found.reached and found.endpoint is not None
        assert found.log_width <= tau
        if increasing:
            assert found.endpoint <= root
            assert found.endpoint * math.exp(found.log_width) >= root * (1 - 1e-15)
        else:
            assert found.endpoint >= root
            assert found.endpoint * math.exp(-found.log_width) <= root * (1 + 1e-15)

    def test_a_set_that_starts_at_zero_is_exact(self):
        found = _find_boundary(lambda r: 1.0, 0.5, increasing=True, seed=3.0, tau=2.0**-11)
        assert (found.endpoint, found.log_width, found.reached) == (0.0, 0.0, True)

    def test_an_empty_decreasing_set_is_exact_zero(self):
        found = _find_boundary(lambda r: 0.0, 0.5, increasing=False, seed=3.0, tau=2.0**-11)
        assert (found.endpoint, found.log_width, found.reached) == (0.0, 0.0, True)

    def test_a_decreasing_set_that_never_empties_below_the_cap_is_unbounded(self):
        found = _find_boundary(
            lambda r: 1.0, 0.5, increasing=False, seed=3.0, tau=2.0**-11, cap=1e6
        )
        assert found.endpoint is None
        assert found.reached

    @pytest.mark.parametrize("shape", ["step", "ratio", "steep"])
    @pytest.mark.parametrize("increasing", [True, False])
    @pytest.mark.parametrize("offset", [-1e-6, -1e-9, 0.0, 1e-9, 1e-6])
    @pytest.mark.parametrize("tau", [0.1, 1e-3])
    def test_the_tested_null_sits_on_the_side_its_own_evaluation_gives(
        self, increasing, shape, offset, tau
    ):
        """The reported set excludes the null exactly when ``f(null) < target``, even when
        the null falls inside the final bracket (here, *tau* wide around the crossing)."""
        root = 2.0
        f, target = _analytic(root, increasing=increasing, shape=shape)
        resolve = root * (1.0 + offset)
        found = _find_boundary(f, target, increasing=increasing, seed=1.7, tau=tau, resolve=resolve)
        assert found.endpoint is not None
        included = resolve >= found.endpoint if increasing else resolve <= found.endpoint
        assert included == (f(resolve) >= target)

    @pytest.mark.parametrize("increasing", [True, False])
    def test_a_crossing_below_the_descent_floor_is_reported_as_unresolved(self, increasing):
        seed = 1.0
        root = seed * 2.0**-50
        f, target = _analytic(root, increasing=increasing, shape="step")
        found = _find_boundary(f, target, increasing=increasing, seed=seed, tau=2.0**-11)
        assert not found.reached
        assert found.log_width > 2.0**-11
        assert found.endpoint is not None
        # The unresolved endpoint is still the conservative one: outside the true set.
        assert found.endpoint <= root if increasing else found.endpoint >= root

    @pytest.mark.parametrize("increasing", [True, False])
    def test_a_crossing_just_above_the_descent_floor_is_resolved(self, increasing):
        seed = 1.0
        root = seed * 2.0**-39
        f, target = _analytic(root, increasing=increasing, shape="step")
        found = _find_boundary(f, target, increasing=increasing, seed=seed, tau=2.0**-11)
        assert found.reached
        assert found.log_width <= 2.0**-11


class TestPrecisionDisclosure:
    """An interval whose search did not reach its resolution or whose nuisance search ran out
    of iterations says so; a resolved one is silent."""

    @staticmethod
    def _interval(
        *, reached: bool = True, width: float = 1e-4, capped: int = 0, gap: float = 0.0
    ) -> brr.BinomialInterval:
        return brr.BinomialInterval(
            lower=0.9,
            upper=1.2,
            geometry="central",
            p_value_null=0.4,
            endpoint_log_width=width,
            resolution_reached=reached,
            nuisance_gap_max=gap,
            capped_probes=capped,
        )

    def test_a_resolved_interval_has_no_note(self):
        assert brr.precision_note(self._interval()) is None

    @pytest.mark.parametrize("width", [1e-2, math.inf])
    def test_an_unresolved_interval_carries_a_note(self, width):
        note = brr.precision_note(self._interval(reached=False, width=width))
        assert isinstance(note, str) and note.startswith(brr.PRECISION_NOTE_PREFIX)

    @pytest.mark.parametrize("gap", [3.5e-5, 1.234567e-6, 9.999e-9, 2.0000001e-3, 0.125])
    def test_a_capped_nuisance_search_discloses_its_count_and_a_gap_never_below_the_gap(self, gap):
        note = brr.precision_note(self._interval(capped=2, gap=gap))
        assert isinstance(note, str) and note.startswith(brr.NUISANCE_NOTE_PREFIX)
        assert "2 p-value probe(s)" in note
        printed = re.search(r"at most (\S+) in p-value", note)
        assert printed is not None
        stated = float(printed.group(1))
        assert gap <= stated <= gap * 1.01

    def test_both_disclosures_are_kept_and_lifted_out_of_a_longer_note(self):
        note = brr.precision_note(self._interval(reached=False, width=1e-2, capped=1, gap=1e-6))
        assert note is not None
        assert note.startswith(brr.PRECISION_NOTE_PREFIX) and brr.NUISANCE_NOTE_PREFIX in note
        assert brr.without_precision_note(f"kept | {note}") == "kept"
        assert brr.without_precision_note(f"kept; {note}") == "kept"

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            ("{}", None),
            ("kept | {}", "kept"),
            ("{}; kept", "kept"),
            ("{} | kept; more", "kept; more"),
            ("first; second | {} | third", "first; second | third"),
            ("first | second; {}", "first | second"),
            ("untouched; text | alone", "untouched; text | alone"),
        ],
    )
    @pytest.mark.parametrize("width", [1e-2, math.inf])
    def test_the_disclosure_lifts_out_of_a_longer_note_with_the_rest_intact(
        self, template, expected, width
    ):
        note = brr.precision_note(self._interval(reached=False, width=width))
        assert note is not None
        assert brr.without_precision_note(template.format(note)) == expected

    def test_an_absent_note_stays_absent(self):
        assert brr.without_precision_note(None) is None
        assert brr.without_precision_note("") is None

    @pytest.mark.parametrize(
        "counts",
        [(6, 60, 9, 60), (0, 30, 5, 30), (60, 600, 6, 600), (5, 100, 0, 100), (50, 50, 50, 50)],
    )
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_ordinary_cells_reach_the_declared_resolution(self, counts, alternative):
        ci = brr.confidence_interval(*counts, alpha=0.05, alternative=alternative)
        tau = brr._endpoint_tolerance(brr._count_scale(*counts))
        assert ci.resolution_reached
        assert 0.0 <= ci.endpoint_log_width <= tau
        assert (ci.capped_probes, ci.nuisance_gap_max) == (0, 0.0)
        assert brr.precision_note(ci) is None


class TestNuisanceCapOnTheInterval:
    """An interval whose nuisance searches hit their iteration cap near a decision is still a
    valid, conservative set, counts those probes and reports the gap they reached."""

    COUNTS = (100, 1000, 130, 1000)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_a_tight_cap_discloses_and_only_widens_the_interval(self, alternative, monkeypatch):
        reference = brr.confidence_interval(*self.COUNTS, alpha=0.05, alternative=alternative)
        assert reference.capped_probes == 0 and reference.nuisance_gap_max == 0.0

        monkeypatch.setattr(brr, "NUISANCE_STOP", brr._StopRule(2.0**-14, 3))
        capped = brr.confidence_interval(*self.COUNTS, alpha=0.05, alternative=alternative)

        assert capped.capped_probes > 0
        assert capped.nuisance_gap_max > 0.0
        note = brr.precision_note(capped)
        assert note is not None and brr.NUISANCE_NOTE_PREFIX in note
        assert capped.lower <= reference.lower
        if reference.upper is not None:
            assert capped.upper is not None and capped.upper >= reference.upper
        assert capped.p_value_null >= reference.p_value_null

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_the_disclosed_gap_bounds_the_reported_p_values_excess(self, alternative, monkeypatch):
        """The p-value an interval reports at the null is twice the smaller directional bound
        when two-sided, so the gap it discloses is stated in that p-value, not the directional
        one it was measured in."""
        reference = brr.confidence_interval(*self.COUNTS, alpha=0.05, alternative=alternative)
        monkeypatch.setattr(brr, "NUISANCE_STOP", brr._StopRule(2.0**-14, 3))
        capped = brr.confidence_interval(*self.COUNTS, alpha=0.05, alternative=alternative)

        assert capped.nuisance_gap_max > 0.0
        assert capped.p_value_null - reference.p_value_null <= capped.nuisance_gap_max

    def test_a_probe_far_under_the_tail_level_reaches_its_floored_gap_and_is_not_disclosed(self):
        """At 100 / 1,000 versus 130 / 1,000 the search at risk ratio 0.65 (a p-value near
        1e-7, far under the 0.025 tail) cannot reach a gap relative to it within the cap. Read
        against the tail level, as the interval and its verdict read it, the gap is reached."""
        x_c, n_c, x_t, n_t = self.COUNTS
        beta = brr.nuisance_beta(0.05)
        rule = brr.NUISANCE_STOP
        relative = brr._p_plus_certificate(0.65, x_c, n_c, x_t, n_t, beta, rule)
        floored = brr._p_plus_certificate(0.65, x_c, n_c, x_t, n_t, beta, rule, brr._Reading(0.025))
        assert not relative.stopped and relative.p < 0.025
        assert floored.stopped and floored.p < 0.025
        ci = brr.confidence_interval(*self.COUNTS, alpha=0.05, alternative="two-sided")
        assert ci.capped_probes == 0


class TestNullPValue:
    """``null_p_value`` is the p-value ``confidence_interval`` reports at the null and the
    recompute behind a row's verdict: the directional p-values the public tests return, and
    the same minimum for the two-sided one without refining the side that cannot be smaller."""

    BETA = brr.nuisance_beta(0.05)
    #: Counts with the treatment above, below and level with the control, null ratios either
    #: side of the observed one, and the sparse and zero-count edges.
    CELLS = [
        pytest.param((100, 1000, 140, 1000), 1.0, id="treatment-above"),
        pytest.param((140, 1000, 100, 1000), 1.0, id="treatment-below"),
        pytest.param((100, 1000, 100, 1000), 1.0, id="level"),
        pytest.param((100, 1000, 140, 1000), 1.6, id="null-above-observed"),
        pytest.param((100, 1000, 140, 1000), 0.9, id="null-below-observed"),
        pytest.param((0, 40, 5, 40), 1.0, id="zero-control"),
        pytest.param((5, 40, 0, 40), 1.0, id="zero-treatment"),
        pytest.param((6, 60, 9, 60), 1.3, id="sparse"),
    ]

    @pytest.mark.parametrize("tail", [0.025, 0.05])
    @pytest.mark.parametrize(("counts", "r"), CELLS)
    def test_the_p_value_is_the_smaller_directional_bound_whichever_side_is_refined(
        self, counts, r, tail
    ):
        beta = self.BETA
        x_c, n_c, x_t, n_t = counts
        plus = brr.p_plus(r, x_c, n_c, x_t, n_t, beta, tail=tail)
        minus = brr.p_minus(r, x_c, n_c, x_t, n_t, beta, tail=tail)

        def null(alternative: str) -> float:
            return brr.null_p_value(r, x_c, n_c, x_t, n_t, beta, alternative=alternative, tail=tail)

        assert null("two-sided") == min(1.0, 2.0 * min(plus, minus))
        assert null("greater") == plus
        assert null("less") == minus

    def test_the_side_that_cannot_be_smaller_is_not_refined_to_its_gap(self):
        """At 100 / 1,000 against 140 / 1,000 the p-value of ``R >= 1`` is near one: shown not to
        be the smaller of the two it stops almost at once, where refining it alone costs
        splits."""
        counts, r, tail = (100, 1000, 140, 1000), 1.0, 0.025
        rule = brr.NUISANCE_STOP
        plus = brr._p_plus_certificate(r, *counts, self.BETA, rule, brr._Reading(tail))
        alone = brr._p_minus_certificate(r, *counts, self.BETA, rule, brr._Reading(tail))
        beside = brr._p_minus_certificate(
            r, *counts, self.BETA, rule, brr._Reading(tail, ceiling=plus.p)
        )
        assert alone.stopped and beside.stopped
        assert beside.p >= plus.p and alone.p >= plus.p
        assert beside.gap >= alone.gap  # stopped before its gap target, not after it
        assert brr.null_p_value(r, *counts, self.BETA, alternative="two-sided", tail=tail) == (
            2.0 * plus.p
        )

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_the_certificates_follow_the_rule_they_are_given_not_the_module_default(
        self, alternative
    ):
        """A rule that ends every search after three splits reaches the null's certificates
        while the module default stays as shipped: the test being refined runs out of splits
        and reports the looser bound that leaves, where the shipped rule reaches its gap."""
        counts, r, tail = (100, 1000, 130, 1000), 1.0, 0.025

        def refined(rule: brr._StopRule) -> brr._PCertificate:
            certificates = brr._null_certificates(r, *counts, self.BETA, rule, alternative, tail)
            return certificates[0][1]

        shipped = refined(brr.NUISANCE_STOP)
        capped = refined(brr._StopRule(brr.NUISANCE_STOP.gap_fraction, 3))
        assert shipped.stopped and not capped.stopped
        assert capped.p > shipped.p and capped.gap > shipped.gap

    def test_an_unknown_alternative_is_refused(self):
        with pytest.raises(brr.InvalidRequestError):
            brr.null_p_value(1.0, 5, 50, 6, 50, self.BETA, alternative="both", tail=0.025)


class TestEndpointContainment:
    """A reported endpoint lies outside the finer crossing of the same p-envelope, no further
    from it than the declared resolution -- including risk ratios below 1, where a bracket
    measured in absolute risk-ratio units would be coarse relative to the endpoint."""

    @pytest.mark.parametrize(
        ("counts", "alternative"),
        [
            pytest.param((6, 60, 9, 60), "two-sided", id="sparse"),
            pytest.param((0, 30, 5, 30), "two-sided", id="zero-control"),
            pytest.param((0, 50, 25, 50), "greater", id="zero-control-pinned"),
            pytest.param((60, 600, 6, 600), "two-sided", id="reversed"),
            pytest.param((60, 600, 6, 600), "less", id="reversed-upper-bound"),
            pytest.param((5, 100, 0, 100), "two-sided", id="zero-treatment"),
            pytest.param((500, 1000, 1, 1000), "two-sided", id="ratio-near-zero"),
            pytest.param((1, 1000, 500, 1000), "two-sided", id="ratio-large"),
            pytest.param((500, 1000, 0, 1000), "two-sided", id="dense-control-zero-treatment"),
        ],
    )
    def test_endpoints_enclose_the_finer_reference(self, counts, alternative):
        alpha = 0.05
        ci = brr.confidence_interval(*counts, alpha=alpha, alternative=alternative)
        assert_endpoints_contain_finer_reference(
            ci.lower,
            ci.upper,
            counts=counts,
            tail_alpha=alpha,
            geometry=ci.geometry,
            beta=brr.nuisance_beta(alpha),
        )


@pytest.mark.slow
class TestExactBinomialLatency:
    """The exact route's per-contrast search work must stay bounded at production arm sizes
    while its endpoints stay outward of a finer reference.
    """

    @pytest.mark.parametrize(
        "counts",
        [
            pytest.param((10_000, 100_000, 11_000, 100_000), id="10pct"),
            pytest.param((90_000, 100_000, 91_000, 100_000), id="90pct"),
        ],
    )
    def test_100k_per_arm_two_sided_search_work_stays_bounded(self, monkeypatch, counts):
        """The outer search stays bounded at production arm sizes and conversion rates: the
        expensive tail evaluation is probed a few dozen times, never scanned across a grid.
        The bound is derived from the resolution contract: each directional search evaluates
        the origin, the unbounded check, the seed and at most two further bracketing points,
        spends at most its bisection count plus the interpolation slack narrowing the ln 2
        bracket to ``tau``, and probes the null once."""
        probes: list[float] = []

        def spy(fn):
            def probe(r, *args, **kwargs):
                probes.append(r)
                return fn(r, *args, **kwargs)

            return probe

        monkeypatch.setattr(brr, "_p_plus_certificate", spy(brr._p_plus_certificate))
        monkeypatch.setattr(brr, "_p_minus_certificate", spy(brr._p_minus_certificate))
        ci = brr.confidence_interval(*counts, alpha=0.05, alternative="two-sided")
        tau = brr._endpoint_tolerance(brr._count_scale(*counts))
        bisections = math.ceil(math.log2(math.log(2.0) / tau))
        per_search = 1 + 1 + 1 + 2 + bisections + brr._REFINE_SLACK + 1
        p_value_calls = 2
        assert len(probes) <= 2 * per_search + p_value_calls
        ratio = (counts[2] / counts[3]) / (counts[0] / counts[1])
        assert ci.upper is not None
        assert ci.lower < ratio < ci.upper

    @pytest.mark.parametrize(
        "counts", [(1_000, 10_000, 1_100, 10_000), (10, 20_000, 12, 20_000)], ids=str
    )
    def test_reported_interval_contains_the_finer_reference_within_tau(self, counts):
        """A looser bound is conservative, a tighter one an invalidated coverage guarantee:
        both endpoints sit outside the finer crossing and within the declared resolution of
        it."""
        alpha = 0.05
        ci = brr.confidence_interval(*counts, alpha=alpha, alternative="two-sided")
        assert_endpoints_contain_finer_reference(
            ci.lower,
            ci.upper,
            counts=counts,
            tail_alpha=alpha,
            geometry="central",
            beta=brr.nuisance_beta(alpha),
        )

    def test_p_value_null_is_the_public_p_value_at_the_null(self):
        """``p_value_null`` is the p-value for the tested null under the same alternative,
        computed directly rather than read off the endpoint search -- so it equals the
        public two-sided p-value at ``null_r``, read against the same tail level, exactly."""
        x_c, n_c, x_t, n_t = 1_000, 100_000, 1_100, 100_000
        ci = brr.confidence_interval(x_c, n_c, x_t, n_t, alpha=0.05, alternative="two-sided")
        expected = brr.p_two(1.0, x_c, n_c, x_t, n_t, brr.nuisance_beta(0.05), tail=0.025)
        assert ci.p_value_null == expected

    @pytest.mark.parametrize(
        "x_c,n_c,x_t,n_t",
        [
            (1_000, 20_000, 1_095, 20_000),  # lower endpoint straddled R = 1
            (1_100, 20_000, 1_000, 20_000),  # upper endpoint landed exactly on R = 1
            (1_000, 20_000, 1_093, 20_000),  # just short of significance: must cover
        ],
    )
    def test_interval_excludes_the_null_exactly_when_its_p_value_is_below_alpha(
        self, x_c, n_c, x_t, n_t
    ):
        """The endpoint search must never leave an endpoint on the far side of the tested
        null, or the displayed interval and the exact p-value (and `stat_sig`) disagree
        about significance."""
        ci = brr.confidence_interval(x_c, n_c, x_t, n_t, alpha=0.05, alternative="two-sided")
        excludes = ci.lower > 1.0 or (ci.upper is not None and ci.upper < 1.0)
        assert excludes == (ci.p_value_null < 0.05)

    @pytest.mark.parametrize(
        ("counts", "alternative"),
        [
            ((1_000, 20_000, 1_095, 20_000), "greater"),
            ((1_100, 20_000, 1_000, 20_000), "less"),
            ((1_000, 20_000, 1_093, 20_000), "greater"),
        ],
    )
    def test_one_sided_set_excludes_the_null_exactly_when_its_p_value_is_below_alpha(
        self, counts, alternative
    ):
        ci = brr.confidence_interval(*counts, alpha=0.05, alternative=alternative)
        if alternative == "greater":
            excludes = ci.lower > 1.0
        else:
            assert ci.upper is not None
            excludes = ci.upper < 1.0
        assert excludes == (ci.p_value_null < 0.05)


def _omitted_masses(n: int, i_lo: int, i_hi: int, q: float) -> tuple[Decimal, Decimal]:
    """Exact (decimal) control mass below ``i_lo`` and above ``i_hi`` at rate ``q``."""
    below = _dec_binom_cdf(i_lo - 1, n, q)
    above = Decimal(1) - _dec_binom_cdf(i_hi, n, q)
    return below, above


class TestSupportWindowTruncation:
    """The support window must stay conservative: the control mass it leaves
    out, at every nuisance rate in ``[a, b]``, is within the omitted mass it
    reports (and so within the requested budget), and it is a no-op below the
    enumeration threshold.
    """

    #: ``(n_c, a, b)`` nuisance intervals: rare, zero-control, central, high-rate,
    #: and intervals whose end is exactly 0 or 1.
    NUISANCE_INTERVALS = [
        pytest.param(1_000_000, 1e-4, 1e-4, id="rare-point"),
        pytest.param(4_000_000, 0.0, 5e-4, id="rare-zero-lower-end"),
        pytest.param(100_000, 0.45, 0.55, id="central"),
        pytest.param(100_000, 0.5, 0.5, id="central-point"),
        pytest.param(4_000_000, 0.9999, 0.9999, id="high-rate-point"),
        pytest.param(4_000_000, 0.9995, 1.0, id="upper-end-one"),
        pytest.param(100_000, 0.0, 1.0, id="whole-unit-interval"),
    ]

    def test_full_enumeration_below_threshold(self):
        assert brr._support_window(100, 0.1, 0.2) == (0, 100, 0.0)

    def test_windowed_tail_dominates_full_enumeration(self):
        n_c = n_t = 20_000
        k = 5
        a, b = brr.clopper_pearson(4000, n_c, 1e-6)
        window = brr._support_window(n_c, a, b)
        assert window[1] - window[0] < n_c  # genuinely truncated
        q, r = (a + b) / 2.0, 1.1
        p = min(r * q, 1.0)
        i_full = np.arange(n_c + 1)
        pmf_full = _binom.pmf(i_full, n_c, q)
        thresh_full = np.ceil((k + n_t * i_full) / n_c).astype(np.int64) - 1
        sf_full = _binom.sf(thresh_full, n_t, p)
        full = float(np.dot(pmf_full, sf_full))
        windowed = brr._tail_plus(q, p, n_c, n_t, k, window)
        assert windowed >= full

    @staticmethod
    def _assert_omitted_mass_is_reported(n_c: int, a: float, b: float, budget: float) -> None:
        i_lo, i_hi, omitted = brr._support_window(n_c, a, b, budget)
        assert 0 <= i_lo <= i_hi <= n_c
        assert omitted <= budget
        # Mass below the window falls with the rate and mass above rises, so the worst
        # rates are the interval's ends; interior rates confirm that.
        rates = [a + (b - a) * t for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        below_worst = max(_omitted_masses(n_c, i_lo, i_hi, q)[0] for q in rates)
        above_worst = max(_omitted_masses(n_c, i_lo, i_hi, q)[1] for q in rates)
        assert below_worst == _omitted_masses(n_c, i_lo, i_hi, a)[0]
        assert above_worst == _omitted_masses(n_c, i_lo, i_hi, b)[1]
        assert below_worst + above_worst <= Decimal(omitted)

    @pytest.mark.parametrize("budget", [1e-12, 1e-3])
    @pytest.mark.parametrize(("n_c", "a", "b"), NUISANCE_INTERVALS)
    def test_omitted_mass_bound_is_never_exceeded(self, n_c, a, b, budget):
        self._assert_omitted_mass_is_reported(n_c, a, b, budget)

    @pytest.mark.parametrize("budget", [1e-12, 1e-3])
    @pytest.mark.parametrize(
        ("x_c", "n_c"),
        [
            pytest.param(100, 1_000_000, id="rare-narrow"),
            pytest.param(3, 100_000, id="rare-wide"),
            pytest.param(1, 4_000_000, id="single-event-wide"),
            pytest.param(50_000, 100_000, id="central"),
            pytest.param(99_990, 100_000, id="high-rate"),
        ],
    )
    def test_omitted_mass_bound_holds_on_clopper_pearson_intervals(self, x_c, n_c, budget):
        a, b = brr.clopper_pearson(x_c, n_c, brr.nuisance_beta(0.05))
        self._assert_omitted_mass_is_reported(n_c, a, b, budget)

    def test_a_rare_rate_scans_only_the_counts_it_can_reach(self):
        """The window follows the nuisance rate, not the arm size: control counts at a
        1e-4 rate in a million units spread over tens of counts, so the scan is hundreds
        of terms rather than the thousands a rate-blind bound would keep."""
        i_lo, i_hi, _ = brr._support_window(1_000_000, 1e-4, 1e-4)
        assert i_lo <= 100 <= i_hi
        assert i_hi - i_lo + 1 <= 400

    def test_a_side_that_cannot_be_cut_keeps_its_whole_range(self):
        # A zero lower nuisance end keeps X = 0 reachable; an upper end of one keeps X = n.
        n_c = 100_000
        assert brr._support_window(n_c, 0.0, 0.3)[0] == 0
        assert brr._support_window(n_c, 0.3, 1.0)[1] == n_c
        assert brr._support_window(n_c, 0.0, 1.0) == (0, n_c, 0.0)

    def test_budget_parameter_is_honored_not_the_module_constant(self):
        """A caller passing a wider `budget` must get a NARROWER window
        (more omitted mass tolerated) than the module-default budget --
        if `budget` were ignored in favor of `_SUPPORT_TRUNCATION_BUDGET`,
        this would fail."""
        n_c = 20_000
        a, b = 0.1, 0.3
        default_window = brr._support_window(n_c, a, b)
        wide_budget_window = brr._support_window(n_c, a, b, budget=1e-3)
        assert wide_budget_window != default_window
        assert wide_budget_window[1] - wide_budget_window[0] < default_window[1] - default_window[0]
        assert wide_budget_window[2] > default_window[2]


def _dec_exponent(n: int, x: float, q: float, prec: int = 50) -> Decimal:
    """Exact ``n * D(x || q)`` for the Bernoulli relative entropy ``D``, ``0 < q < 1``.

    ``prec`` must exceed the decimal digits that ``1 - x`` and ``1 - q`` need to stay distinct
    from 1: about 330 for subnormal operands.
    """
    with localcontext() as context:
        context.prec = prec
        xd, qd = Decimal(x), Decimal(q)
        first = xd * (xd / qd).ln() if x > 0.0 else Decimal(0)
        second = (1 - xd) * ((1 - xd) / (1 - qd)).ln() if x < 1.0 else Decimal(0)
        return n * (first + second)


class TestChernoffSupportCut:
    """``chernoff_support`` cuts a control-count window from the Chernoff bound
    ``P(X <= m) <= exp(-n D(m/n || q))`` below the nuisance interval and its mirror above.
    Evaluating ``D`` in float64 is bounded by a derived allowance; here it is checked
    against exact decimal arithmetic.
    """

    @pytest.mark.parametrize("n", [257, 4_096, 100_000, 1_000_000, 4_000_000])
    def test_certified_exponent_is_a_tight_lower_bound_of_the_exact_one(self, n):
        factors = (0.0, 0.1, 0.5, 0.9, 1.0 - 1e-6, 1.0, 1.0 + 1e-6, 1.1, 2.0, 10.0, math.inf)
        for q in (1e-6, 1e-4, 1e-2, 0.3, 0.5, 0.9, 0.9999):
            for factor in factors:
                x = min(1.0, q * factor)
                lower = exponent_lower_bound(n, x, q)
                exact = _dec_exponent(n, x, q)
                assert Decimal(lower) <= exact, (n, x, q)
                assert exact - Decimal(lower) <= Decimal("1e-6"), (n, x, q)

    CELLS = [
        pytest.param(1_000_000, 1e-4, 1e-4, id="rare-point"),
        pytest.param(4_000_000, 1e-6, 5e-4, id="rare-wide"),
        pytest.param(100_000, 0.45, 0.55, id="central"),
        pytest.param(4_000_000, 0.9999, 0.9999, id="high-rate-point"),
        pytest.param(4_000_000, 0.9995, 0.99999, id="high-rate-wide"),
        pytest.param(257, 0.2, 0.4, id="just-above-the-enumeration-threshold"),
    ]

    @pytest.mark.parametrize("tail", [5e-13, 5e-4])
    @pytest.mark.parametrize(("n", "q_lo", "q_hi"), CELLS)
    def test_cut_is_certified_and_maximal_up_to_the_rounding_allowance(self, n, q_lo, q_hi, tail):
        i_lo, i_hi = chernoff_support(n, q_lo, q_hi, tail)
        target = -Decimal(tail).ln()
        slack = Decimal("1e-6")
        assert 0 <= i_lo <= i_hi <= n
        if i_lo > 0:  # counts below i_lo carry at most `tail`: the last one cut is certified
            assert _dec_exponent(n, (i_lo - 1) / n, q_lo) >= target
        if i_lo / n <= q_lo:  # the first one kept could not be cut
            assert _dec_exponent(n, i_lo / n, q_lo) < target + slack
        if i_hi < n:  # counts above i_hi carry at most `tail`
            assert _dec_exponent(n, (i_hi + 1) / n, q_hi) >= target
        if i_hi / n >= q_hi:
            assert _dec_exponent(n, i_hi / n, q_hi) < target + slack

    def test_a_rate_end_that_keeps_an_extreme_count_reachable_cannot_be_cut(self):
        n, tail = 100_000, 5e-13
        assert chernoff_support(n, 0.0, 0.3, tail)[0] == 0
        assert chernoff_support(n, 0.3, 1.0, tail)[1] == n
        assert chernoff_support(n, 0.0, 1.0, tail) == (0, n)

    #: Upper rates at and around the reciprocal's overflow point ``2**-1024``, down to the
    #: smallest subnormal, next to ordinary tiny rates.
    TINY_RATES = [
        pytest.param(5e-324, id="smallest-subnormal"),
        pytest.param(1e-320, id="deep-subnormal"),
        pytest.param(2.0**-1030, id="subnormal"),
        pytest.param(2.0**-1024, id="reciprocal-overflows"),
        pytest.param(math.nextafter(2.0**-1024, 1.0), id="reciprocal-just-finite"),
        pytest.param(2.0**-1022, id="smallest-normal"),
        pytest.param(1e-300, id="tiny-normal"),
    ]

    @pytest.mark.parametrize("q", TINY_RATES)
    @pytest.mark.parametrize("n", [257, 4_000_000])
    def test_a_tiny_upper_rate_leaves_only_the_zero_count(self, n, q):
        """``P(X >= 1) <= n q`` is far below the tail, so the cut is the single count 0
        whether or not ``1 / q`` is a finite float."""
        tail = 5e-13
        assert Decimal(n) * Decimal(q) <= Decimal(tail)
        assert chernoff_support(n, 0.0, q, tail) == (0, 0)

    @pytest.mark.parametrize(
        ("x", "q"),
        [
            pytest.param(1.0, 2.0**-1024, id="quotient-overflows"),
            pytest.param(1.0, 5e-324, id="quotient-overflows-smallest-subnormal"),
            pytest.param(0.5, 1e-320, id="quotient-overflows-half"),
            pytest.param(1e-6, 5e-324, id="quotient-finite-rate-subnormal"),
            pytest.param(1.0, 1e-300, id="quotient-huge-but-finite"),
            pytest.param(5e-324, 0.5, id="quotient-underflows"),
            pytest.param(1e-310, 0.999, id="quotient-subnormal"),
        ],
    )
    def test_certified_exponent_holds_where_the_quotient_leaves_the_float_range(self, x, q):
        n = 4_000_000
        lower = exponent_lower_bound(n, x, q)
        exact = _dec_exponent(n, x, q)
        assert math.isfinite(lower)
        assert Decimal(lower) <= exact
        assert exact - Decimal(lower) <= exact * Decimal("1e-13")

    @pytest.mark.parametrize(
        ("n", "x", "q"),
        [
            pytest.param(257, 2.898604e-318, 7.2465e-319, id="both-subnormal-ratio-4"),
            pytest.param(1_000_000, 1.1004230257e-314, 1.100423e-317, id="ratio-1000"),
            pytest.param(4_000_000, 1.5481265395e-314, 1.5481265e-317, id="wide-arm"),
            pytest.param(1_000_000, 3.95e-322, 1.86269744e-316, id="x-near-the-smallest-subnormal"),
            pytest.param(257, 2.916e-320, 2.8864495387e-314, id="q-above-x"),
            pytest.param(4_000_000, 4e-310, 4e-310, id="equal-rates"),
        ],
    )
    def test_certified_exponent_holds_where_the_result_is_subnormal(self, n, x, q):
        """A product in the subnormal range rounds by an absolute ``2**-1075``, not a relative
        ``2**-53``; the exponent must stay a lower bound there as well."""
        lower = exponent_lower_bound(n, x, q)
        assert math.isfinite(lower)
        assert Decimal(lower) <= _dec_exponent(n, x, q, prec=700)

    def test_a_tiny_rate_window_reports_the_mass_it_omits(self):
        """At a subnormal upper rate the window is the count 0 and ``P(X >= 1) <= n q``
        stays within the mass it reports as omitted."""
        n_c, q = 4_000_000, 2.0**-1024
        i_lo, i_hi, omitted = brr._support_window(n_c, 0.0, q)
        assert (i_lo, i_hi) == (0, 0)
        assert (
            Decimal(n_c) * Decimal(q) <= Decimal(omitted) <= Decimal(brr._SUPPORT_TRUNCATION_BUDGET)
        )

    def test_point_masses_at_the_unit_interval_ends_keep_one_count(self):
        n, tail = 100_000, 5e-13
        assert chernoff_support(n, 0.0, 0.0, tail) == (0, 0)
        assert chernoff_support(n, 1.0, 1.0, tail) == (n, n)

    @staticmethod
    def _ulps_from(value: float, steps: int) -> float:
        for _ in range(abs(steps)):
            value = math.nextafter(value, math.inf if steps > 0 else -math.inf)
        return value

    @pytest.mark.parametrize("tail", [5e-13, 5e-4])
    @pytest.mark.parametrize("n", [257, 4_096, 100_000, 1_000_000, 4_000_000])
    def test_an_end_count_is_cut_only_where_its_exact_probability_is_within_tail(self, n, tail):
        """The counts 0 and ``n`` are the cut's tightest case: ``P(X = 0) = (1-q)^n`` and
        ``P(X = n) = q^n`` are known exactly, so nothing but the float evaluation separates the
        cut from the exact edge ``q`` where they equal ``tail``. Around that edge, one ulp at a
        time, a cut must imply the exact probability is within ``tail``; the sweep must cross
        from uncut to cut so the check is not vacuous."""
        with localcontext() as context:
            context.prec = 60
            exponent = -Decimal(tail).ln() / n
            zero_edge = float(1 - (-exponent).exp())  # (1 - q)^n = tail
            full_edge = float((-exponent).exp())  # q^n = tail
            cut_zero, cut_full = [], []
            for steps in range(-64, 65):
                q = self._ulps_from(zero_edge, steps)
                i_lo = chernoff_support(n, q, q, tail)[0]
                cut_zero.append(i_lo >= 1)
                if i_lo >= 1:
                    assert (1 - Decimal(q)) ** n <= Decimal(tail), (n, steps)
                q = self._ulps_from(full_edge, steps)
                i_hi = chernoff_support(n, q, q, tail)[1]
                cut_full.append(i_hi < n)
                if i_hi < n:
                    assert Decimal(q) ** n <= Decimal(tail), (n, steps)
        assert not all(cut_zero) and any(cut_zero)
        assert not all(cut_full) and any(cut_full)


def _clear_every_cache() -> None:
    for value in vars(brr).values():
        if not isinstance(value, type) and hasattr(value, "cache_clear"):
            value.cache_clear()


class TestArrayCacheBound:
    """The tail-vector caches hold what a search revisits within a byte budget, whatever the
    arm size makes of each vector's length."""

    @staticmethod
    def _cache(max_bytes: int):
        calls: list[int] = []

        def make(size: int) -> np.ndarray:
            calls.append(size)
            return np.zeros(size)

        return brr._ArrayCache(make, max_bytes), calls

    def test_a_resident_vector_is_the_one_computed_and_is_not_rebuilt(self):
        cache, calls = self._cache(10_000)
        first = cache(100)
        assert cache(100) is first and calls == [100]

    def test_the_least_recently_used_vector_goes_first_when_the_bytes_run_out(self):
        cache, calls = self._cache(2_000)  # two 800-byte vectors fit, three do not
        for size in (100, 101, 100, 102):  # 101 is now the least recently used
            cache(size)
        calls.clear()
        cache(100)
        cache(102)
        assert calls == []
        cache(101)
        assert calls == [101]

    def test_a_vector_over_the_whole_budget_is_still_returned_and_held_alone(self):
        cache, calls = self._cache(100)
        cache(10)
        big = cache(1_000)
        assert cache(1_000) is big and calls == [10, 1_000]

    def test_clearing_empties_the_cache(self):
        cache, calls = self._cache(10_000)
        cache(100)
        cache.cache_clear()
        cache(100)
        assert calls == [100, 100]

    def test_a_search_under_a_smaller_budget_holds_no_more_than_it_and_reports_the_same(
        self, monkeypatch
    ):
        counts = (50, 1_000, 60, 1_000)
        _clear_every_cache()
        free = brr.confidence_interval(*counts, alpha=0.05, alternative="two-sided")
        held = {name: getattr(brr, name)._bytes for name in ("_control_pmf", "_treatment_tail")}
        assert all(size > 0 for size in held.values())

        _clear_every_cache()
        brr._confidence_interval_cached.cache_clear()
        for name, size in held.items():
            monkeypatch.setattr(getattr(brr, name), "_max_bytes", size // 8)
        bounded = brr.confidence_interval(*counts, alpha=0.05, alternative="two-sided")
        assert bounded == free
        for name, size in held.items():
            assert 0 < getattr(brr, name)._bytes <= size // 8


class TestCachedResultsEqualUncachedResults:
    """Reusing tail vectors inside a search must never change a result: every
    p-value and interval is ``==`` between a run whose ``_treatment_tail`` is
    the uncached function (nothing is ever reused) and a run that uses the
    cache, including reuse inside a single search."""

    CELLS = [
        (6, 60, 9, 60),  # full enumeration
        (40, 400, 55, 400),  # windowed control support
        (0, 30, 5, 30),  # zero control
        (5, 30, 0, 30),  # zero treatment
        (10, 10, 10, 10),  # all success
        (20, 2_000, 50, 5_000),  # unequal arms
    ]
    RATIOS = (0.5, 1.0, 1.3, 2.0)

    @staticmethod
    def _observe(x_c: int, n_c: int, x_t: int, n_t: int):
        beta = brr.nuisance_beta(0.05)
        observed = []
        for r in TestCachedResultsEqualUncachedResults.RATIOS:
            observed.append(brr.p_plus(r, x_c, n_c, x_t, n_t, beta))
            observed.append(brr.p_minus(r, x_c, n_c, x_t, n_t, beta))
        for alternative in ("two-sided", "greater", "less"):
            ci = brr.confidence_interval(x_c, n_c, x_t, n_t, alpha=0.05, alternative=alternative)
            observed.append((ci.lower, ci.upper, ci.geometry, ci.p_value_null))
        return observed

    @pytest.mark.parametrize("x_c,n_c,x_t,n_t", CELLS)
    def test_cached_tails_reproduce_uncached_results_exactly(self, x_c, n_c, x_t, n_t, monkeypatch):
        with monkeypatch.context() as patch:
            patch.setattr(brr, "_treatment_tail", brr._treatment_tail.__wrapped__)
            _clear_every_cache()
            uncached = self._observe(x_c, n_c, x_t, n_t)
        _clear_every_cache()
        cached = self._observe(x_c, n_c, x_t, n_t)
        assert cached == uncached


class TestScipyBinomErrorBudget:
    """The shared SciPy allowance is an empirical numerical assumption of the
    risk-ratio certificate. Check representative CDF ranks against
    independent decimal arithmetic.
    """

    def test_scipy_binom_cdf_within_stated_ulp_allowance(self):
        worst_ulps = 0.0
        cases = [
            (n, x, q)
            for n in (1, 2, 5, 10, 50, 100, 500, 1000)
            for x in (0, n // 3, n // 2, n)
            for q in (1e-6, 0.001, 0.3, 0.5, 0.9999, 1 - 1e-12)
        ]
        for n, x, q in cases:
            scipy_val = float(_binom.cdf(x, n, q))
            exact = _dec_binom_cdf(x, n, q)
            if exact < Decimal("1e-280"):
                continue  # below float64's representable range
            diff = abs(Decimal(scipy_val) - exact)
            ulp = math.ulp(scipy_val) if scipy_val != 0 else 5e-324
            worst_ulps = max(worst_ulps, float(diff) / ulp)
        assert worst_ulps < SCIPY_BINOMIAL_ULP_ALLOWANCE / 2.0, (
            f"observed {worst_ulps} ULPs exceeds half the stated allowance "
            f"{SCIPY_BINOMIAL_ULP_ALLOWANCE} -- re-derive the margin"
        )


class TestClopperPearsonOutwardRounding:
    def test_trivial_single_trial_shape_is_exact(self):
        a, b = brr.clopper_pearson(1, 1, 2e-6)
        assert a == 1e-6
        assert b == 1.0

    def test_below_the_validated_beta_floor_refuses(self):
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.clopper_pearson(3, 10, 1e-12)
        assert exc_info.value.code == "estimation.binomial.tail_unrepresentable"

    def test_subnormal_beta_whose_half_underflows_refuses_before_single_trial_branch(self):
        beta = math.nextafter(0.0, 1.0)
        assert beta > 0.0 and beta / 2.0 == 0.0
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.clopper_pearson(1, 1, beta)
        assert exc_info.value.code == "estimation.binomial.tail_unrepresentable"

    @pytest.mark.parametrize("beta", [1e-9, 1e-6, 1e-4])
    @pytest.mark.parametrize(
        ("n", "x"),
        [
            (n, x)
            for n in (1, 2, 10, 100, 10_000, 4_000_000)
            for x in sorted({0, 1, min(10, n // 2), min(100, n // 2), n - 1, n})
        ],
    )
    def test_nuisance_endpoints_enclose_decimal_binomial_tails(self, n, x, beta):
        lower, upper = brr.clopper_pearson(x, n, beta)
        with localcontext() as context:
            context.prec = 60
            half = Decimal(beta) / 2
            if x > 0:
                assert _dec_binom_cdf(x - 1, n, lower) >= Decimal(1) - half
            if x < n:
                assert _dec_binom_cdf(x, n, upper) <= half
