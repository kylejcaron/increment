"""The vectorised delta-method decision of count pairs is the runtime's decision.

Planning sums the delta-method rule over count lattices and the runtime applies it to one
contrast at a time, so the two derive the same rejection independently. At matched counts,
alternatives, tails and nulls they must agree pair for pair: the vectorised rule settles a pair
only where its derived radius certifies the side of the null, and the runtime row decides the
rest.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import binom
from scipy.stats import t as student_t

from increment.estimation.conversion_delta import (
    CRITICAL_AGREEMENT,
    CRITICAL_RELATIVE_ACCURACY,
    ROUNDING_UNITS,
    _crit,
    critical_values,
    delta_decision,
    delta_log_bounds,
    production_decision,
)
from increment.estimation.conversion_route import (
    dense_min_count,
    route_for_counts,
    routed_share,
    unrouted_share,
)
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import LiftEstimate
from tests.estimation._conversion_counts import CONVERSION_METRIC, count_summary

_ALTERNATIVES = ("two-sided", "greater", "less")


def _alpha(tail: float, alternative: str) -> float:
    return 2.0 * tail if alternative == "two-sided" else tail


def _runtime_row(
    counts: tuple[int, int, int, int], *, tail: float, alternative: str, null_lift: float
) -> LiftEstimate:
    """The row ``estimate_lift`` reports for ``counts = (x_c, n_c, x_t, n_t)``."""
    computation = estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(*counts),
        control_group="control",
        methods=[Method(name="unadjusted")],
        alpha=_alpha(tail, alternative),
        alternative=alternative,
        null_lift=null_lift,
    )
    assert not computation.failures, computation.failures
    (row,) = computation.results
    assert row.reference_kind == "t", "the counts were not routed to the delta method"
    return row


def _routed_pairs(
    n: int, p_c: float, p_t: float, *, tail: float, count: int, seed: int
) -> list[tuple[int, int]]:
    """Seeded count pairs drawn at the rates and kept inside the routed rectangle."""
    rng = np.random.default_rng(seed)
    floor = dense_min_count(tail)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < count:
        x_c, x_t = int(rng.binomial(n, p_c)), int(rng.binomial(n, p_t))
        if route_for_counts(x_c, n, x_t, n, tail_alpha=tail, mode="auto") == "asymptotic":
            assert min(x_c, n - x_c, x_t, n - x_t) >= floor
            pairs.append((x_c, x_t))
    return pairs


_CASES: list[tuple[int, float, float, float, str, float]] = [
    # (arm size, control rate, treatment rate, one-sided tail, alternative, null_lift)
    (1_236, 0.5, 0.53, 0.1, "two-sided", 0.0),
    (1_236, 0.5, 0.53, 0.1, "greater", 0.0),
    (20_000, 0.3, 0.33, 0.025, "two-sided", 0.03),
    (20_000, 0.3, 0.27, 0.025, "less", -0.05),
    (20_000, 0.3, 0.33, 0.025, "greater", 0.04),
    (300_000, 0.1, 0.105, 0.0005, "two-sided", -0.02),
]


class TestAgreementWithTheRuntimeRow:
    @pytest.mark.parametrize(
        ("n", "p_c", "p_t", "tail", "alternative", "null_lift"),
        _CASES,
        ids=[f"{c[0]}-{c[4]}-{c[5]}" for c in _CASES],
    )
    def test_the_vectorised_decision_and_the_runtime_row_agree_pair_for_pair(
        self, n, p_c, p_t, tail, alternative, null_lift
    ):
        pairs = _routed_pairs(n, p_c, p_t, tail=tail, count=25, seed=n + len(alternative))
        x_c = np.array([[c] for c, _ in pairs])
        x_t = np.array([[t] for _, t in pairs])
        decided = delta_decision(
            x_c, n, x_t.T, n, tail=tail, alternative=alternative, null_lift=null_lift
        )
        for index, (c, t) in enumerate(pairs):
            row = _runtime_row(
                (c, n, t, n), tail=tail, alternative=alternative, null_lift=null_lift
            )
            # The row's own verdict is what the runtime reports; the two directions it reads
            # are the ends of the same interval.
            runtime = row.stat_sig()
            deferred = production_decision(
                c, n, t, n, tail=tail, alternative=alternative, null_lift=null_lift
            )
            assert (deferred[0] or deferred[1]) == runtime, (c, t)
            if decided.settled[index, index]:
                vectorised = (decided.plus[index, index], decided.minus[index, index])
                assert vectorised == deferred, (c, t)

    @pytest.mark.parametrize(
        ("n", "p_c", "p_t", "tail"),
        [(1_236, 0.5, 0.53, 0.1), (20_000, 0.3, 0.33, 0.025), (300_000, 0.1, 0.105, 0.0005)],
    )
    def test_the_interval_ends_are_the_runtimes_to_rounding(self, n, p_c, p_t, tail):
        """The vectorised and the runtime's log-scale ends differ by rounding, orders of
        magnitude inside the agreement the decision defers within."""
        gaps = []
        for c, t in _routed_pairs(n, p_c, p_t, tail=tail, count=20, seed=n):
            row = _runtime_row((c, n, t, n), tail=tail, alternative="two-sided", null_lift=0.0)
            assert row.lift is not None and row.lift.lb is not None and row.lift.ub is not None
            lower, upper = delta_log_bounds(np.array([c]), n, np.array([t]), n, tail)
            gaps.append(abs(math.log1p(row.lift.lb) - lower[0]))
            gaps.append(abs(math.log1p(row.lift.ub) - upper[0]))
        assert max(gaps) < 1e-12

    @pytest.mark.parametrize(
        ("n", "p_c", "p_t", "tail"), [(1_236, 0.5, 0.53, 0.1), (20_000, 0.3, 0.33, 0.025)]
    )
    def test_the_vectorised_margin_stays_inside_its_derived_radius_of_the_runtimes(
        self, n, p_c, p_t, tail
    ):
        """The radius a pair is certified beyond covers the actual disagreement between the
        vectorised interval ends and the runtime row's, with orders of magnitude to spare."""
        u = 2.0**-53
        for c, t in _routed_pairs(n, p_c, p_t, tail=tail, count=30, seed=3 * n):
            row = _runtime_row((c, n, t, n), tail=tail, alternative="two-sided", null_lift=0.0)
            assert row.lift is not None and row.lift.lb is not None
            lower, _ = delta_log_bounds(np.array([c]), n, np.array([t]), n, tail)
            se = row.lift.value  # scale only
            scale = abs(math.log1p(row.lift.lb)) + abs(lower[0]) + abs(se)
            allowed = ROUNDING_UNITS * u * scale + CRITICAL_RELATIVE_ACCURACY * 10.0
            assert abs(math.log1p(row.lift.lb) - lower[0]) < allowed


class TestAnExactBorderlineNull:
    """A null that equals the runtime's own interval end to the last bit, shifted off zero, is a
    pair the vectorised rule cannot settle: the runtime row decides it, and decides it on the
    strict inequality the row's verdict uses."""

    COUNTS = (6_100, 20_000, 6_700, 20_000)
    TAIL = 0.025

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_pair_is_left_to_the_runtime_and_decided_as_it_decides(self, alternative):
        x_c, n_c, x_t, n_t = self.COUNTS
        first = _runtime_row(self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=0.0)
        assert first.lift is not None and first.lift.lb is not None and first.lift.ub is not None
        for end, reads in ((first.lift.lb, "plus"), (first.lift.ub, "minus")):
            if (reads == "plus" and alternative == "less") or (
                reads == "minus" and alternative == "greater"
            ):
                continue
            for null in (end, math.nextafter(end, -math.inf), math.nextafter(end, math.inf)):
                decided = delta_decision(
                    np.array([[x_c]]),
                    n_c,
                    np.array([[x_t]]),
                    n_t,
                    tail=self.TAIL,
                    alternative=alternative,
                    null_lift=null,
                )
                # Within the radius of the null: the vectorised rule must not decide it.
                assert not decided.settled[0, 0]
                assert not decided.plus[0, 0] and not decided.minus[0, 0]
                runtime = _runtime_row(
                    self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=null
                )
                plus, minus = production_decision(
                    x_c, n_c, x_t, n_t, tail=self.TAIL, alternative=alternative, null_lift=null
                )
                assert (plus or minus) == runtime.stat_sig()
                # The strict inequality: an interval end equal to the null does not reject.
                if null == end:
                    assert (plus if reads == "plus" else minus) is False


class TestCriticalValues:
    def test_the_interpolated_reference_agrees_with_the_inverse_survival(self):
        df = 2_000.0 + 40.0 * np.linspace(0.0, 1.0, 9_000)
        for tail in (0.1, 0.0005):
            exact = student_t.isf(tail, df)
            assert np.max(np.abs(critical_values(df, tail) / exact - 1.0)) < CRITICAL_AGREEMENT


class TestUnroutedShare:
    @pytest.mark.parametrize(
        ("n_c", "n_t", "p_c", "p_t", "tail"),
        [
            (1_000, 1_000, 0.45, 0.47, 0.1),
            (5_000, 5_000, 0.3, 0.32, 0.025),
            (900, 700, 0.5, 0.6, 0.1),
        ],
    )
    def test_it_is_the_complement_of_the_routed_share(self, n_c, n_t, p_c, p_t, tail):
        floor = dense_min_count(tail)
        routed = routed_share(n_c, n_t, p_c, p_t, tail_alpha=tail)
        assert unrouted_share(n_c, n_t, p_c, p_t, floor=floor) == pytest.approx(
            1.0 - routed, abs=1e-12
        )

    def test_it_keeps_its_relative_precision_where_it_is_small(self):
        """Ten standard deviations short of the floor, the unrouted mass is about 1e-22: a
        complement of the routed share would round it to zero."""
        n, p, floor = 1_500, 0.4, dense_min_count(0.1)
        arm = binom.cdf(floor - 1, n, p) + binom.sf(n - floor, n, p)
        expected = 2.0 * arm - arm * arm
        assert 0.0 < expected < 1e-15
        assert unrouted_share(n, n, p, p, floor=floor) == pytest.approx(expected, rel=1e-9)


def test_a_pair_the_count_rule_keeps_on_the_finite_sample_route_has_no_delta_row():
    kwargs: dict[str, Any] = {"tail": 0.025, "alternative": "two-sided", "null_lift": 0.0}
    with pytest.raises(ValueError, match="finite-sample route"):
        production_decision(0, 10_000, 12, 10_000, **kwargs)


class TestTheQuantileBracket:
    """The certificate brackets the reference quantile between its values at the smallest and
    largest degrees of freedom of a run, which relies on the runtime's own quantile decreasing in
    the degrees of freedom to within `CRITICAL_RELATIVE_ACCURACY`."""

    @pytest.mark.parametrize("tail", [0.1, 0.025, 0.0005])
    def test_the_runtimes_quantile_decreases_in_the_degrees_of_freedom(self, tail):
        df = np.geomspace(400.0, 2e7, 4_000)
        values = np.array([_crit(float(d), tail) for d in df])
        steps = values[1:] / values[:-1] - 1.0
        assert steps.max() <= CRITICAL_RELATIVE_ACCURACY
        assert np.all(np.abs(values / student_t.isf(tail, df) - 1.0) < CRITICAL_RELATIVE_ACCURACY)


class TestEveryCertifiedPairIsTheRuntimes:
    """Over a whole window of count pairs around the rejection boundary, every pair the
    vectorised rule certifies is decided as the runtime row decides it, and the pairs it leaves
    open are a thin band, not a region."""

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_a_window_of_pairs_agrees_with_the_runtime_row(self, alternative):
        n, tail, null = 1_236, 0.1, 0.0
        x_c = np.arange(560, 600)[:, None]
        x_t = np.arange(590, 680)[None, :]
        decided = delta_decision(x_c, n, x_t, n, tail=tail, alternative=alternative, null_lift=null)
        rng = np.random.default_rng(11)
        settled = np.argwhere(decided.settled)
        for a, b in settled[rng.choice(len(settled), 120, replace=False)]:
            plus, minus = production_decision(
                int(x_c[a, 0]),
                n,
                int(x_t[0, b]),
                n,
                tail=tail,
                alternative=alternative,
                null_lift=null,
            )
            assert (plus or minus) == bool(decided.plus[a, b] or decided.minus[a, b])
        # The cells next to a change of the vectorised decision are where the boundary is.
        rejects = decided.plus | decided.minus
        edge = np.argwhere(rejects[:, 1:] != rejects[:, :-1])
        for a, b in edge[:: max(1, len(edge) // 25)]:
            for c in (b, b + 1):
                plus, minus = production_decision(
                    int(x_c[a, 0]),
                    n,
                    int(x_t[0, c]),
                    n,
                    tail=tail,
                    alternative=alternative,
                    null_lift=null,
                )
                if decided.settled[a, c]:
                    assert (plus or minus) == bool(rejects[a, c])
        assert (~decided.settled).mean() < 0.05
