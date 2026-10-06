"""The planner's delta decision of a count pair is the runtime's own.

Planning decides every routed count pair with `production_decision`, which calls the runtime's
functions (``infer_lift`` on the arms' moments); the runtime applies the same rule to one
contrast at a time through ``estimate_lift``. At matched counts, alternatives, tails and nulls
the interval and the verdict are equal, bit for bit, including at a null on the interval's end,
whichever producer rounded the arms' stored moments within what `binary_counts` admits and
however close to -1 an interval end lies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Context, Decimal
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
import pytest

from increment.estimation.armstats import ArmStats, BinomialDataError, binary_counts
from increment.estimation.conversion_delta import delta_interval, production_decision
from increment.estimation.conversion_route import (
    dense_min_count,
    route_for_counts,
)
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import LiftEstimate
from increment.semantics.models import ConversionMetric, MeanMetric, RetentionMetric
from tests.estimation._conversion_counts import (
    CONVERSION_METRIC,
    arm_row,
    count_arms,
    count_summary,
    producer_arm,
)

_ALTERNATIVES = ("two-sided", "greater", "less")


def _alpha(tail: float, alternative: str) -> float:
    return 2.0 * tail if alternative == "two-sided" else tail


def _summary_row(
    summary: Sequence[Mapping[str, Any]],
    *,
    tail: float,
    alternative: str,
    null_lift: float,
    metric: ConversionMetric | RetentionMetric | MeanMetric = CONVERSION_METRIC,
) -> LiftEstimate:
    """The row ``estimate_lift`` reports for the stored moments in ``summary``."""
    computation = estimate_lift(
        metrics=[metric],
        summary=summary,
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


def _runtime_row(
    counts: tuple[int, int, int, int], *, tail: float, alternative: str, null_lift: float
) -> LiftEstimate:
    """The row ``estimate_lift`` reports for ``counts = (x_c, n_c, x_t, n_t)``."""
    return _summary_row(
        count_summary(*counts), tail=tail, alternative=alternative, null_lift=null_lift
    )


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


# What a producer may store for one arm, within what `binary_counts` admits: the first moment
# within an absolute 1e-6 of an integer sum, the second within the rounding of summing its
# units. A drift is pushed to the last step of a doubling ladder that `binary_counts` still
# accepts, so the bounds are the function's own, never restated here.
_DRIFT_COUNTS = (60_000, 200_000, 61_500, 200_000)
_DRIFT_TAIL = 0.025


def _admitted_edge(exact: ArmStats, drift: Callable[[ArmStats, float], ArmStats]) -> ArmStats:
    """``exact`` drifted as far as `binary_counts` still admits it: ``drift(exact, step)`` at the
    last step of a doubling ladder before the first refusal."""
    admitted, step = exact, 2.0**-60
    while True:
        candidate = drift(exact, step)
        try:
            binary_counts(candidate, "conversion")
        except BinomialDataError:
            return admitted
        admitted, step = candidate, 2.0 * step


def _second_moment(sign: float) -> Callable[[ArmStats, float], ArmStats]:
    return lambda arm, step: arm.model_copy(update={"cy2": arm.cy2 * (1.0 + sign * step)})


def _first_moment(sign: float) -> Callable[[ArmStats, float], ArmStats]:
    return lambda arm, step: arm.model_copy(update={"cy1": arm.cy1 + sign * step})


_PRODUCERS = (
    "canonical",
    "second_moment_control_high",
    "second_moment_control_low",
    "first_moment_control_high",
    "first_moment_control_low",
    "warehouse_producer",
)


@cache
def _arms(producer: str) -> tuple[ArmStats, ArmStats]:
    """``(control, treatment)`` of `_DRIFT_COUNTS` as ``producer`` stores them."""
    x_c, n_c, x_t, n_t = _DRIFT_COUNTS
    control, treatment = count_arms(*_DRIFT_COUNTS)
    if producer == "warehouse_producer":
        return (
            producer_arm(n_c, x_c, group_id="control"),
            producer_arm(n_t, x_t, group_id="treatment"),
        )
    drifts = {
        "second_moment_control_high": (_second_moment(1.0), _second_moment(-1.0)),
        "second_moment_control_low": (_second_moment(-1.0), _second_moment(1.0)),
        "first_moment_control_high": (_first_moment(1.0), _first_moment(-1.0)),
        "first_moment_control_low": (_first_moment(-1.0), _first_moment(1.0)),
    }
    if producer == "canonical":
        return control, treatment
    drift_c, drift_t = drifts[producer]
    return _admitted_edge(control, drift_c), _admitted_edge(treatment, drift_t)


def _stored_summary(producer: str) -> list[dict[str, Any]]:
    return [arm_row(arm) for arm in _arms(producer)]


@pytest.mark.parametrize("producer", _PRODUCERS)
class TestAcceptedProducerMoments:
    """Counts decide a conversion pair, not the rounding a producer stored in its moments.

    An arm `binary_counts` accepts is a declared 0/1 outcome, so its mean and variance are
    functions of its counts; the moments it stores carry the producer's summation error up to
    the admitted bounds. The runtime's row and the planner's pair, which sees only counts, are
    the same decision at an interval end whichever producer stored the arms."""

    def test_the_arms_are_admitted_on_their_counts(self, producer):
        control, treatment = _arms(producer)
        x_c, n_c, x_t, n_t = _DRIFT_COUNTS
        assert binary_counts(control, "conversion") == (x_c, n_c)
        assert binary_counts(treatment, "conversion") == (x_t, n_t)
        # A drifting producer stores moments other than the exact ones: the premise of the rest.
        stored = [(a.ref_y, a.cy1, a.cy2) for a in (control, treatment)]
        exact = [(a.ref_y, a.cy1, a.cy2) for a in count_arms(*_DRIFT_COUNTS)]
        assert (stored != exact) == (producer != "canonical")

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_interval_is_the_one_of_the_counts(self, producer, alternative):
        row = _summary_row(
            _stored_summary(producer), tail=_DRIFT_TAIL, alternative=alternative, null_lift=0.0
        )
        assert row.lift is not None
        assert (row.lift.lb, row.lift.ub) == delta_interval(
            *_DRIFT_COUNTS, tail=_DRIFT_TAIL, alternative=alternative
        )

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_verdict_at_an_interval_end_is_the_one_of_the_counts(self, producer, alternative):
        interval = delta_interval(*_DRIFT_COUNTS, tail=_DRIFT_TAIL, alternative=alternative)
        assert interval is not None
        lower, upper = interval
        for end, reads in ((lower, "plus"), (upper, "minus")):
            if (reads == "plus" and alternative == "less") or (
                reads == "minus" and alternative == "greater"
            ):
                continue
            for null in (end, math.nextafter(end, -math.inf), math.nextafter(end, math.inf)):
                row = _summary_row(
                    _stored_summary(producer),
                    tail=_DRIFT_TAIL,
                    alternative=alternative,
                    null_lift=null,
                )
                plus, minus = production_decision(
                    *_DRIFT_COUNTS,
                    tail=_DRIFT_TAIL,
                    alternative=alternative,
                    null_lift=null,
                )
                assert row.stat_sig() == (plus or minus), (reads, null)
                # The strict inequality: an interval end equal to the null does not reject.
                if null == end:
                    assert not row.stat_sig()

    @pytest.mark.parametrize(
        "metric",
        [
            CONVERSION_METRIC,
            RetentionMetric(name="conv", entity="user", fact="conv", threshold_days=(1, 8)),
        ],
        ids=["conversion", "retention"],
    )
    def test_a_retention_arm_is_decided_like_a_conversion_arm(self, producer, metric):
        row = _summary_row(
            _stored_summary(producer),
            tail=_DRIFT_TAIL,
            alternative="two-sided",
            null_lift=0.0,
            metric=metric,
        )
        assert row.lift is not None
        assert (row.lift.lb, row.lift.ub) == delta_interval(
            *_DRIFT_COUNTS, tail=_DRIFT_TAIL, alternative="two-sided"
        )


def test_a_mean_metric_reads_its_moments_as_stored():
    """Nothing declares a mean metric's arms 0/1, so the counts they happen to reconstruct do
    not replace the moments the producer stored."""
    from increment.estimation.inference import infer_lift
    from increment.estimation.variance import se_log_mean, stable_log_ratio

    control, treatment = _arms("warehouse_producer")
    c, t = control.to_summary(), treatment.to_summary()
    stored = infer_lift(
        metric="conv",
        group_id="treatment",
        method="unadjusted",
        log_rr=stable_log_ratio(c.mean, t.mean),
        se_t=se_log_mean(t.var, t.mean, t.n),
        se_c=se_log_mean(c.var, c.mean, c.n),
        alpha=_alpha(_DRIFT_TAIL, "two-sided"),
        arm_ns=(t.n, c.n),
        method_role="decision",
    )
    row = _summary_row(
        _stored_summary("warehouse_producer"),
        tail=_DRIFT_TAIL,
        alternative="two-sided",
        null_lift=0.0,
        metric=MeanMetric(name="conv", entity="user", fact="conv"),
    )
    assert stored.lift is not None and row.lift is not None
    assert (row.lift.lb, row.lift.ub) == (stored.lift.lb, stored.lift.ub)
    assert (row.lift.lb, row.lift.ub) != delta_interval(
        *_DRIFT_COUNTS, tail=_DRIFT_TAIL, alternative="two-sided"
    )


class TestIntervalEndsNearTotalLoss:
    """A treatment arm whose rate is a vanishing share of the control's puts both interval ends
    within 1e-9 of -1, where the last bit of an end is a relative error of 1e-7 or more in
    ``1 + lift``. The runtime's row and the planner's pair are one calculation there too, at
    nulls on and beside each end."""

    TAIL = 0.025
    N = 2**52
    COUNTS = (N // 2, N, 8 * dense_min_count(TAIL), N)

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_pair_is_decided_as_the_runtime_decides_it(self, alternative):
        interval = delta_interval(*self.COUNTS, tail=self.TAIL, alternative=alternative)
        assert interval is not None
        lower, upper = interval
        assert -1.0 < lower < upper < -1.0 + 1e-9
        for end, reads in ((lower, "plus"), (upper, "minus")):
            if (reads == "plus" and alternative == "less") or (
                reads == "minus" and alternative == "greater"
            ):
                continue
            nulls = [end]
            for direction in (-math.inf, math.inf):
                step = end
                for _ in range(3):
                    step = math.nextafter(step, direction)
                    nulls.append(step)
            for null in nulls:
                row = _runtime_row(
                    self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=null
                )
                assert row.lift is not None
                assert (row.lift.lb, row.lift.ub) == interval
                plus, minus = production_decision(
                    *self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=null
                )
                assert (plus or minus) == row.stat_sig(), null
                if null == end:
                    assert not row.stat_sig()


class TestLargeArmsNearSaturation:
    """An arm of 2**61 units with a few thousand failures keeps the variance of its failures.

    Its counts are admitted by `binary_counts` (a second moment is unconstrained once the
    rounding bound of the arm's units reaches one) and dense for the tail, so the row takes the
    delta method. A raw-sum centering of such an arm clamps the variance, ``failures``, to zero
    under its noise floor of ``8 * eps * n`` and the row would be refused as zero variance; the
    row is the interval of the counts instead, whatever second moment the producer stored, with
    the log standard error and log ratio the counts give in closed form."""

    N = 2**61
    # (tail, control successes, treatment successes). Float64 below 2**61 holds multiples of 256
    # only: the counts of the first three cases lie on that grid, the rest do not, so a count
    # is recovered from the stored reference and residual or it is read as a neighboring one.
    CASES = {
        "both_near_saturation": (0.025, N - 2560, N - 3584),
        "control_near_saturation_at_the_dense_floor": (0.1, N - 512, N // 2),
        "treatment_near_saturation": (0.025, N // 2, N - 4096),
        "control_513_failures": (0.1, N - 513, N // 2),
        "treatment_777_failures": (0.1, N // 2, N - 777),
        "both_off_the_grid": (0.1, N - 1027, N - 641),
        "both_off_the_grid_at_a_tighter_tail": (0.025, N - 2561, N - 3001),
        "control_at_the_dense_floor_off_the_grid": (0.025, N - 2140, N // 2),
    }
    STORED = ("counted", "clamped", "inflated")

    @classmethod
    def _noise_floor(cls) -> float:
        """Failures below this are a raw-sum centering's noise, ``4 * eps * 2 * n``."""
        return 8.0 * math.ulp(1.0) * cls.N

    @classmethod
    def _arm(cls, successes: int, group_id: str, stored: str) -> ArmStats:
        """The arm as a producer might store it: the correctly rounded rate as the reference and
        the exact residual of the integer sum from it, with the exact centered sum of squares,
        that sum lost to the noise floor of a raw-sum centering, or 1% above it."""
        failures = cls.N - successes
        exact = successes * failures / cls.N
        cy2 = exact * 1.01 if stored == "inflated" else exact
        if stored == "clamped" and failures < cls._noise_floor():
            cy2 = 0.0
        ref_y = successes / cls.N
        return ArmStats(
            study_id="e",
            metric="conv",
            group_id=group_id,
            n=cls.N,
            ref_y=ref_y,
            cy1=float(Fraction(successes) - cls.N * Fraction(ref_y)),
            cy2=cy2,
        )

    @classmethod
    def _stored(cls, case: str, stored: str) -> list[dict[str, Any]]:
        _, x_c, x_t = cls.CASES[case]
        control, treatment = cls._arm(x_c, "control", stored), cls._arm(x_t, "treatment", stored)
        return [arm_row(control), arm_row(treatment)]

    @pytest.mark.parametrize("stored", STORED)
    @pytest.mark.parametrize("case", CASES)
    def test_the_counts_are_admitted_and_dense(self, case, stored):
        tail, x_c, x_t = self.CASES[case]
        for successes, group_id in ((x_c, "control"), (x_t, "treatment")):
            arm = self._arm(successes, group_id, stored)
            assert binary_counts(arm, "conversion") == (successes, self.N)
        assert min(self.N - x_c, self.N - x_t) >= dense_min_count(tail)
        assert route_for_counts(x_c, self.N, x_t, self.N, tail_alpha=tail, mode="auto") == (
            "asymptotic"
        )

    @pytest.mark.parametrize("stored", STORED)
    def test_the_dense_floor_is_decided_on_the_exact_failure_count(self, stored):
        """Failures at the floor take the delta method and one fewer the finite-sample route,
        which refuses above its arm ceiling: a count read as the nearest multiple of 256 would
        put the floor's 2140 on the sparse side (2048) and decide it by the wrong route."""
        tail = 0.025
        floor = dense_min_count(tail)
        assert floor % 256 != 0
        for failures, route in ((floor, "asymptotic"), (floor - 1, "finite_sample")):
            x_c, x_t = self.N - failures, self.N // 2
            assert route_for_counts(x_c, self.N, x_t, self.N, tail_alpha=tail, mode="auto") == route
            computation = estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=[
                    arm_row(self._arm(x_c, "control", stored)),
                    arm_row(self._arm(x_t, "treatment", stored)),
                ],
                control_group="control",
                methods=[Method(name="unadjusted")],
                alpha=2.0 * tail,
            )
            if route == "asymptotic":
                (row,) = computation.results
                assert row.reference_kind == "t"
            else:
                assert not computation.results
                (failure,) = computation.failures.values()
                assert failure.code == "estimation.binomial.finite_sample_arm_ceiling_exceeded"

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    @pytest.mark.parametrize("stored", STORED)
    @pytest.mark.parametrize("case", CASES)
    def test_the_row_is_the_interval_of_the_counts(self, case, stored, alternative):
        tail, x_c, x_t = self.CASES[case]
        row = _summary_row(
            self._stored(case, stored), tail=tail, alternative=alternative, null_lift=0.0
        )
        interval = delta_interval(x_c, self.N, x_t, self.N, tail=tail, alternative=alternative)
        assert interval is not None
        assert row.lift is not None
        assert (row.lift.lb, row.lift.ub) == interval
        assert interval[0] < row.lift.value < interval[1]

    @pytest.mark.parametrize("case", CASES)
    def test_the_log_ratio_and_its_standard_error_are_the_closed_form_of_the_counts(self, case):
        """Independent of the estimator: ``log(p_t / p_c)`` to 60 digits, and the delta-method
        variance of each arm's log mean, ``failures / (successes * (n - 1))``. An arm mean is a
        float64, which near one resolves a rate to ``eps``, so the log ratio is checked to a few
        ``eps`` beside its relative tolerance; the standard error has no such floor."""
        tail, x_c, x_t = self.CASES[case]
        row = _summary_row(
            self._stored(case, "clamped"), tail=tail, alternative="two-sided", null_lift=0.0
        )
        assert row.lift is not None
        context = Context(prec=60)
        log_ratio = context.ln(context.divide(Decimal(x_t), Decimal(x_c)))
        assert row.lift.log_mean == pytest.approx(
            float(log_ratio), rel=1e-9, abs=4.0 * math.ulp(1.0)
        )
        variance = sum(Fraction(self.N - x, x * (self.N - 1)) for x in (x_c, x_t))
        assert row.lift.log_se == pytest.approx(math.sqrt(float(variance)), rel=1e-9, abs=0.0)

    @pytest.mark.parametrize("stored", STORED)
    @pytest.mark.parametrize("case", CASES)
    def test_the_verdict_at_an_interval_end_is_the_one_of_the_counts(self, case, stored):
        tail, x_c, x_t = self.CASES[case]
        interval = delta_interval(x_c, self.N, x_t, self.N, tail=tail, alternative="two-sided")
        assert interval is not None
        for end in interval:
            for null in (end, math.nextafter(end, -math.inf), math.nextafter(end, math.inf)):
                row = _summary_row(
                    self._stored(case, stored), tail=tail, alternative="two-sided", null_lift=null
                )
                plus, minus = production_decision(
                    x_c, self.N, x_t, self.N, tail=tail, alternative="two-sided", null_lift=null
                )
                assert row.stat_sig() == (plus or minus), null
                if null == end:
                    assert not row.stat_sig()
