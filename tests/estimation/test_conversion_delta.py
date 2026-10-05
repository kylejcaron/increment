"""The planner's delta decision of a count pair is the runtime's own.

Planning decides every routed count pair with `production_decision`, which calls the runtime's
functions (``infer_lift`` on the arms' moments); the runtime applies the same rule to one
contrast at a time through ``estimate_lift``. At matched counts, alternatives, tails and nulls
the interval and the verdict are equal, bit for bit, including at a null on the interval's end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from increment.estimation.conversion_delta import delta_interval, production_decision
from increment.estimation.conversion_route import (
    dense_min_count,
    route_for_counts,
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


class TestTheRuntimesDecision:
    @pytest.mark.parametrize(
        ("n", "p_c", "p_t", "tail", "alternative", "null_lift"),
        _CASES,
        ids=[f"{c[0]}-{c[4]}-{c[5]}" for c in _CASES],
    )
    def test_production_decision_is_the_runtime_rows_verdict(
        self, n, p_c, p_t, tail, alternative, null_lift
    ):
        for c, t in _routed_pairs(n, p_c, p_t, tail=tail, count=25, seed=n + len(alternative)):
            row = _runtime_row(
                (c, n, t, n), tail=tail, alternative=alternative, null_lift=null_lift
            )
            plus, minus = production_decision(
                c, n, t, n, tail=tail, alternative=alternative, null_lift=null_lift
            )
            assert (plus or minus) == row.stat_sig(), (c, t)

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    @pytest.mark.parametrize(
        ("n", "p_c", "p_t", "tail"),
        [(1_236, 0.5, 0.53, 0.1), (20_000, 0.3, 0.33, 0.025), (300_000, 0.1, 0.105, 0.0005)],
    )
    def test_the_interval_equals_the_runtime_rows_exactly(self, alternative, n, p_c, p_t, tail):
        """`delta_interval` calls the runtime's own functions, so its interval is the one
        ``estimate_lift`` reports, bit for bit."""
        for c, t in _routed_pairs(n, p_c, p_t, tail=tail, count=25, seed=n + 5):
            row = _runtime_row((c, n, t, n), tail=tail, alternative=alternative, null_lift=0.0)
            assert row.lift is not None
            assert delta_interval(c, n, t, n, tail=tail, alternative=alternative) == (
                row.lift.lb,
                row.lift.ub,
            )


class TestAnExactBorderlineNull:
    """A null that equals the runtime's own interval end to the last bit, shifted off zero, is
    decided on the strict inequality the row's verdict uses."""

    COUNTS = (6_100, 20_000, 6_700, 20_000)
    TAIL = 0.025

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_pair_is_decided_as_the_runtime_decides_it(self, alternative):
        x_c, n_c, x_t, n_t = self.COUNTS
        first = _runtime_row(self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=0.0)
        assert first.lift is not None and first.lift.lb is not None and first.lift.ub is not None
        for end, reads in ((first.lift.lb, "plus"), (first.lift.ub, "minus")):
            if (reads == "plus" and alternative == "less") or (
                reads == "minus" and alternative == "greater"
            ):
                continue
            for null in (end, math.nextafter(end, -math.inf), math.nextafter(end, math.inf)):
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
