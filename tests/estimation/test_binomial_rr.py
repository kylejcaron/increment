"""Exact independent-binomial risk-ratio inversion: numerical certification,
zero/positive-control geometry, and the exact-enumeration applicability
boundary.

Deterministic unit tests only; the 336-cell Monte Carlo calibration grid
lives in ``tests/estimation/test_rare_event_calibration.py`` (owned
separately -- see that module for coverage-frequency evidence).
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from math import comb

import numpy as np
import pytest
from scipy.stats import binom as _binom

from increment.estimation import binomial_rr as brr
from increment.estimation._tails import SCIPY_BINOMIAL_ULP_ALLOWANCE
from increment.estimation.binomial_rr import _find_boundary

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


class TestCertifiedSupOutwardRounding:
    """`_certified_sup`/`_tail_plus`/`_tail_minus` must never report a
    value below the true mathematical supremum, even by a single ULP.
    """

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
        certified = brr._certified_sup("plus", a, b, r, n_c, n_t, k, window)
        # Independent decimal re-evaluation of F_+ at a fine grid over
        # [a, b]: the certified bound must be >= every grid point's exact
        # value (a weaker, cheap sanity check on top of the analytic
        # monotone/quadratic bound's own correctness).
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            q = a + frac * (b - a)
            p = min(r * q, 1.0)
            exact = Decimal(0)
            for i in range(n_c + 1):
                pmf_i = (
                    Decimal(comb(n_c, i)) * Decimal(q) ** i * (Decimal(1) - Decimal(q)) ** (n_c - i)
                )
                thresh = math.ceil((k + n_t * i) / n_c) - 1
                if thresh < 0:
                    sf = Decimal(1)
                elif thresh >= n_t:
                    sf = Decimal(0)
                else:
                    sf = Decimal(1) - _dec_binom_cdf(thresh, n_t, p)
                exact += pmf_i * sf
            assert certified >= float(exact)

    def test_singular_zero_endpoint_refines_an_interior_maximum(self):
        window = brr._support_window(1, 0.0, 0.75)

        certified = brr._certified_sup("plus", 0.0, 0.75, 1.0, 1, 1, 1, window)

        # This tail is q(1 - q), whose maximum is 0.25 at q=0.5.  The
        # q=0 endpoint makes the curvature envelope singular, but refinement
        # must still tighten the initial monotone certificate of 0.75.
        assert 0.25 <= certified <= 0.250001

    def test_singular_saturated_endpoint_refines_an_interior_maximum(self):
        window = brr._support_window(1, 0.125, 0.5)

        certified = brr._certified_sup("minus", 0.125, 0.5, 2.0, 1, 1, -1, window)

        # q(1 - 2q) peaks at q=0.25; neither endpoint attains that value.
        assert 0.125 <= certified <= 0.125001


class TestExtremeAlphaFiniteUpperBound:
    """A positive control count always has a finite mathematical upper
    endpoint (the CP lower bound eventually empties the nuisance domain);
    a numerical search CAP must never masquerade as genuine unboundedness.
    """

    def test_astra_extreme_alpha_repro_returns_finite_upper(self):
        ci = brr.confidence_interval(1, 1, 0, 1, alpha=1e-14, alternative="two-sided")
        assert ci.upper is not None
        assert math.isfinite(ci.upper)

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


class TestApplicabilityBoundaryCalibration:
    """``MAX_ARM_SIZE`` is a compute-resource applicability boundary, not a
    scientific one: ordinary sizes up to it succeed (spot-checked below),
    sizes far beyond it -- and exactly one arm-count above it -- are
    refused immediately (without attempting the expensive search), and
    the largest arm size in the full 336-cell rare-event calibration
    manifest is admitted, which is checked structurally rather than by
    actually calling ``confidence_interval`` at that size (too slow to run
    as a routine regression).
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
                brr.MAX_ARM_SIZE * 10,
                1,
                brr.MAX_ARM_SIZE * 10,
                alpha=0.05,
                alternative="two-sided",
            )
        assert calls == []
        assert exc_info.value.code == "estimation.binomial.arm_too_large_for_exact_enumeration"

    def test_refuses_at_one_above_the_ceiling(self, monkeypatch):
        calls = self._record_searches(monkeypatch)
        with pytest.raises(brr.BinomialDataError) as exc_info:
            brr.confidence_interval(
                1, brr.MAX_ARM_SIZE + 1, 1, 10, alpha=0.05, alternative="two-sided"
            )
        assert calls == []
        assert exc_info.value.code == "estimation.binomial.arm_too_large_for_exact_enumeration"

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
        assert n_c <= brr.MAX_ARM_SIZE
        assert n_t <= brr.MAX_ARM_SIZE

    def test_moderate_arm_sizes_succeed(self):
        for n in (100, 10_000, 200_000):
            ci = brr.confidence_interval(
                max(1, n // 20), n, max(1, n // 15), n, alpha=0.05, alternative="two-sided"
            )
            assert ci.upper is not None and math.isfinite(ci.upper)


@pytest.mark.slow
class TestExactBinomialLatency:
    """The exact route's per-contrast search work must stay bounded at
    production arm sizes, with the interval only ever widening (never
    tightening) relative to the untuned (max_bisect=60) reference, and
    the certified tail-probability search (_certified_sup) fully
    unaffected.
    """

    def test_100k_per_arm_two_sided_search_work_stays_bounded(self, monkeypatch):
        """The outer search stays bounded at production arm sizes: the
        expensive tail evaluation is probed a few dozen times, never
        scanned across a grid. The untuned ``max_bisect=60`` reference
        needs ~112 probes and roughly five times the wall clock."""
        probes: list[float] = []

        def spy(fn):
            def probe(r, *args, **kwargs):
                probes.append(r)
                return fn(r, *args, **kwargs)

            return probe

        monkeypatch.setattr(brr, "p_plus", spy(brr.p_plus))
        monkeypatch.setattr(brr, "p_minus", spy(brr.p_minus))
        ci = brr.confidence_interval(
            10_000, 100_000, 11_000, 100_000, alpha=0.05, alternative="two-sided"
        )
        assert len(probes) <= 64
        assert ci.lower < 1.1
        assert ci.upper is not None and ci.upper > 1.1

    def test_tuned_search_only_ever_widens_the_reported_interval(self):
        """Outward-monotone invariant: the tuned (max_bisect=10) search's
        reported bound must never be tighter than the untuned
        (max_bisect=60) reference's -- the tuned interval must CONTAIN
        the reference interval. A tighter tuned bound would be an
        invalidated coverage guarantee, not just a precision loss. Both
        directional searches are covered: the lower bound goes through
        `_find_boundary(increasing=True)` directly, the upper bound goes
        through `_bound_upper`'s own `_find_boundary(increasing=False)`.
        """
        beta = brr.nuisance_beta(0.05)
        for x_c, n_c, x_t, n_t in [(1_000, 10_000, 1_100, 10_000), (10, 20_000, 12, 20_000)]:
            seed = (n_c * x_t) / (n_t * x_c)

            def f_plus(r, x_c=x_c, n_c=n_c, x_t=x_t, n_t=n_t):
                return brr.p_plus(r, x_c, n_c, x_t, n_t, brr.nuisance_beta(0.05))

            def f_minus(r, x_c=x_c, n_c=n_c, x_t=x_t, n_t=n_t):
                return brr.p_minus(r, x_c, n_c, x_t, n_t, brr.nuisance_beta(0.05))

            reference_lower = _find_boundary(
                f_plus, 0.025, increasing=True, seed=seed, max_bisect=60
            )
            tuned_lower = _find_boundary(f_plus, 0.025, increasing=True, seed=seed)
            assert reference_lower is not None and tuned_lower is not None, (x_c, n_c, x_t, n_t)
            assert tuned_lower <= reference_lower, (x_c, n_c, x_t, n_t)
            assert tuned_lower == pytest.approx(reference_lower, rel=2e-3)

            a, _b = brr.clopper_pearson(x_c, n_c, beta)
            cap = 2.0 / a
            reference_upper = _find_boundary(
                f_minus, 0.025, increasing=False, seed=seed, cap=cap, max_bisect=60
            )
            tuned_upper = _find_boundary(f_minus, 0.025, increasing=False, seed=seed, cap=cap)
            assert reference_upper is not None and tuned_upper is not None, (x_c, n_c, x_t, n_t)
            assert tuned_upper >= reference_upper, (x_c, n_c, x_t, n_t)
            assert tuned_upper == pytest.approx(reference_upper, rel=2e-3)

    def test_p_value_null_is_bit_identical_regardless_of_max_bisect(self):
        """``p_value_null`` is the p-value for the tested null under the
        same alternative, computed directly rather than read off the
        endpoint search -- so it equals the public two-sided p-value at
        ``null_r`` exactly."""
        x_c, n_c, x_t, n_t = 1_000, 100_000, 1_100, 100_000
        ci = brr.confidence_interval(x_c, n_c, x_t, n_t, alpha=0.05, alternative="two-sided")
        expected = brr.p_two(1.0, x_c, n_c, x_t, n_t, brr.nuisance_beta(0.05))
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
        """The coarse endpoint search must never leave an endpoint on the far
        side of the tested null, or the displayed interval and the exact
        p-value (and `stat_sig`) disagree about significance."""
        ci = brr.confidence_interval(x_c, n_c, x_t, n_t, alpha=0.05, alternative="two-sided")
        excludes = ci.lower > 1.0 or (ci.upper is not None and ci.upper < 1.0)
        assert excludes == (ci.p_value_null < 0.05)


class TestSupportWindowTruncation:
    """The Hoeffding-window truncation must always be conservative (>=
    the full O(n) enumeration), and a no-op below the enumeration
    threshold.
    """

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

    def test_omitted_mass_bound_is_never_exceeded(self):
        # Hoeffding's bound must actually dominate the true omitted mass
        # (checked against a full decimal enumeration for a moderate n
        # where that is still tractable).
        n_c = 20_000
        a, b = 0.1, 0.3
        i_lo, i_hi, omitted = brr._support_window(n_c, a, b)
        if i_hi - i_lo >= n_c:
            pytest.skip("n_c below the full-enumeration threshold")
        q = 0.3  # worst case: at the window's own upper edge
        true_omitted = float(_binom.cdf(i_lo - 1, n_c, q) if i_lo > 0 else 0.0) + float(
            1.0 - _binom.cdf(i_hi, n_c, q) if i_hi < n_c else 0.0
        )
        assert omitted >= true_omitted

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
