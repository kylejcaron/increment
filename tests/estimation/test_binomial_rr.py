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
from increment.estimation._binomial_support import chernoff_support, exponent_lower_bound
from increment.estimation._tails import SCIPY_BINOMIAL_ULP_ALLOWANCE
from increment.estimation.binomial_rr import _find_boundary
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


def _analytic(root: float, *, increasing: bool, smooth: bool):
    """``(f, target)`` whose evaluated crossing sits at *root*, with no SciPy involved.

    Increasing: ``f(r) >= target`` iff ``r >= root``. Decreasing: iff ``r <= root``. The
    step form jumps from 0 to 1 at the crossing; the smooth form is ``r / (r + root)`` (or
    its mirror).
    """
    if smooth:
        if increasing:
            return (lambda r: r / (r + root)), 0.5
        return (lambda r: root / (r + root)), 0.5
    if increasing:
        return (lambda r: 1.0 if r >= root else 0.0), 0.5
    return (lambda r: 1.0 if r <= root else 0.0), 0.5


def _counted(f):
    calls: list[float] = []

    def probe(r: float) -> float:
        calls.append(r)
        return f(r)

    return probe, calls


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

    def test_tolerance_is_one_64th_of_the_scale_between_its_clamps(self):
        assert brr._endpoint_tolerance(0.0141) == pytest.approx(0.0141 / 64.0)

    def test_tolerance_is_never_coarser_than_the_cap_or_finer_than_the_floor(self):
        assert brr._endpoint_tolerance(2.0) == 2.0**-11
        assert brr._endpoint_tolerance(1e-30) == 2.0**-40


class TestEndpointSearchOnAnalyticFunctions:
    """The inversion alone, on functions with a known crossing: outward endpoint, declared
    log width, null placement and disclosed resolution failure."""

    @pytest.mark.parametrize("tau", [2.0**-11, 2.0**-30])
    @pytest.mark.parametrize("factor", [1e-9, 0.3, 0.999, 3.0, 1e3])
    @pytest.mark.parametrize("seed", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("smooth", [False, True])
    @pytest.mark.parametrize("increasing", [True, False])
    def test_endpoint_is_outward_and_its_bracket_reaches_tau(
        self, increasing, smooth, seed, factor, tau
    ):
        root = seed * factor
        f, target = _analytic(root, increasing=increasing, smooth=smooth)
        probe, calls = _counted(f)
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
        # f(0), f(cap), f(seed), at most 40 steps outward or inward, then halvings of ln 2.
        assert len(calls) <= 3 + 41 + math.ceil(math.log2(math.log(2.0) / tau))

    @pytest.mark.parametrize("direction", [math.inf, 0.0])
    @pytest.mark.parametrize("seed", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("increasing", [True, False])
    def test_a_crossing_one_float_from_the_seed_is_bracketed(self, increasing, seed, direction):
        root = math.nextafter(seed, direction)
        f, target = _analytic(root, increasing=increasing, smooth=False)
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

    @pytest.mark.parametrize("smooth", [False, True])
    @pytest.mark.parametrize("increasing", [True, False])
    @pytest.mark.parametrize("offset", [-1e-6, -1e-9, 0.0, 1e-9, 1e-6])
    def test_the_tested_null_sits_on_the_side_its_own_evaluation_gives(
        self, increasing, smooth, offset
    ):
        """The reported set excludes the null exactly when ``f(null) < target``, even when
        the null falls inside the final bracket (here, 1e-3 wide around the crossing)."""
        root = 2.0
        f, target = _analytic(root, increasing=increasing, smooth=smooth)
        resolve = root * (1.0 + offset)
        found = _find_boundary(
            f, target, increasing=increasing, seed=1.7, tau=1e-3, resolve=resolve
        )
        assert found.endpoint is not None
        included = resolve >= found.endpoint if increasing else resolve <= found.endpoint
        assert included == (f(resolve) >= target)

    @pytest.mark.parametrize("increasing", [True, False])
    def test_a_crossing_below_the_descent_floor_is_reported_as_unresolved(self, increasing):
        seed = 1.0
        root = seed * 2.0**-50
        f, target = _analytic(root, increasing=increasing, smooth=False)
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
        f, target = _analytic(root, increasing=increasing, smooth=False)
        found = _find_boundary(f, target, increasing=increasing, seed=seed, tau=2.0**-11)
        assert found.reached
        assert found.log_width <= 2.0**-11


class TestPrecisionDisclosure:
    """An interval whose search did not reach its resolution says so; a resolved one is silent."""

    @staticmethod
    def _interval(*, reached: bool, width: float) -> brr.BinomialInterval:
        return brr.BinomialInterval(
            lower=0.9,
            upper=1.2,
            geometry="central",
            p_value_null=0.4,
            endpoint_log_width=width,
            resolution_reached=reached,
        )

    def test_a_resolved_interval_has_no_note(self):
        assert brr.precision_note(self._interval(reached=True, width=1e-4)) is None

    @pytest.mark.parametrize("width", [1e-2, math.inf])
    def test_an_unresolved_interval_carries_a_note(self, width):
        note = brr.precision_note(self._interval(reached=False, width=width))
        assert isinstance(note, str) and note.startswith(brr.PRECISION_NOTE_PREFIX)

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
        assert brr.precision_note(ci) is None


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
    while its endpoints stay outward of a finer reference, and the certified tail-probability
    search (_certified_sup) is fully unaffected.
    """

    def test_100k_per_arm_two_sided_search_work_stays_bounded(self, monkeypatch):
        """The outer search stays bounded at production arm sizes: the expensive tail
        evaluation is probed a few dozen times, never scanned across a grid. The bound is
        derived from the resolution contract: each directional search evaluates the
        origin, the unbounded check, the seed and at most two further bracketing points,
        halves its ln 2 bracket to ``tau``, and probes the null once."""
        probes: list[float] = []

        def spy(fn):
            def probe(r, *args, **kwargs):
                probes.append(r)
                return fn(r, *args, **kwargs)

            return probe

        monkeypatch.setattr(brr, "p_plus", spy(brr.p_plus))
        monkeypatch.setattr(brr, "p_minus", spy(brr.p_minus))
        counts = (10_000, 100_000, 11_000, 100_000)
        ci = brr.confidence_interval(*counts, alpha=0.05, alternative="two-sided")
        tau = brr._endpoint_tolerance(brr._count_scale(*counts))
        bisections = math.ceil(math.log2(math.log(2.0) / tau))
        per_search = 1 + 1 + 1 + 2 + bisections + 1
        p_value_calls = 2
        assert len(probes) <= 2 * per_search + p_value_calls
        assert ci.lower < 1.1
        assert ci.upper is not None and ci.upper > 1.1

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
        public two-sided p-value at ``null_r`` exactly."""
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
        if hasattr(value, "cache_clear"):
            value.cache_clear()


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
